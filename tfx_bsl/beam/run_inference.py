# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run batch inference on saved model."""

import abc
import base64
import os
import platform
import sys
import time
from typing import Any, Dict, Generator, Iterable, List, Mapping, NamedTuple, Sequence, Text, Tuple, Union

from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry
import googleapiclient
from googleapiclient import discovery
from googleapiclient import http
import numpy as np
import tensorflow as tf
from tfx_bsl.beam import shared
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.telemetry import util

# TODO(b/140306674): stop using the internal TF API.
from tensorflow.python.saved_model import loader_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2

# pylint: disable=g-import-not-at-top
try:
  import resource
except ImportError:
  resource = None

# TODO(b/131873699): Remove once 1.x support is dropped.
# pylint: disable=g-import-not-at-top
try:
  # We need to import this in order to register all quantiles ops, even though
  # it's not directly used.
  from tensorflow.contrib.boosted_trees.python.ops import quantile_ops as _  # pylint: disable=unused-import
except ImportError:
  pass
# pylint: enable=g-import-not-at-top

_DEFAULT_INPUT_KEY = 'examples'
_METRICS_DESCRIPTOR_INFERENCE = 'BulkInferrer'
_METRICS_DESCRIPTOR_IN_PROCESS = 'InProcess'
_METRICS_DESCRIPTOR_CLOUD_AI_PREDICTION = 'CloudAIPlatformPrediction'
_MILLISECOND_TO_MICROSECOND = 1000
_MICROSECOND_TO_NANOSECOND = 1000
_SECOND_TO_MICROSECOND = 1000000
_REMOTE_INFERENCE_NUM_RETRIES = 5

# We define the following aliases of Any because the actual types are not
# public.
_SignatureDef = Any
_MetaGraphDef = Any
_SavedModel = Any

_BulkInferResult = Union[prediction_log_pb2.PredictLog,
                         Tuple[tf.train.Example, regression_pb2.Regression],
                         Tuple[tf.train.Example,
                               inference_pb2.MultiInferenceResponse],
                         Tuple[tf.train.Example,
                               classification_pb2.Classifications]]


# TODO(b/151468119): Converts this into enum once we stop supporting Python 2.7
class OperationType(object):
  CLASSIFICATION = 'CLASSIFICATION'
  REGRESSION = 'REGRESSION'
  PREDICTION = 'PREDICTION'
  MULTIHEAD = 'MULTIHEAD'


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def RunInferenceImpl(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType
) -> beam.pvalue.PCollection:
  """Implementation of RunInference API.

  Args:
    examples: A PCollection containing examples.
    inference_spec_type: Model inference endpoint.

  Returns:
    A PCollection containing prediction logs.

  Raises:
    ValueError; when operation is not supported.
  """
  logging.info('RunInference on model: %s', inference_spec_type)

  batched_examples = examples | 'BatchExamples' >> beam.BatchElements()
  operation_type = _get_operation_type(inference_spec_type)
  # pylint: disable=no-value-for-parameter
  if operation_type == OperationType.CLASSIFICATION:
    return batched_examples | 'Classify' >> _Classify(inference_spec_type)
  elif operation_type == OperationType.REGRESSION:
    return batched_examples | 'Regress' >> _Regress(inference_spec_type)
  elif operation_type == OperationType.PREDICTION:
    return batched_examples | 'Predict' >> _Predict(inference_spec_type)
  elif operation_type == OperationType.MULTIHEAD:
    return (batched_examples
            | 'MultiInference' >> _MultiInference(inference_spec_type))
  else:
    raise ValueError('Unsupported operation_type %s' % operation_type)


_IOTensorSpec = NamedTuple('_IOTensorSpec',
                           [('input_tensor_alias', Text),
                            ('input_tensor_name', Text),
                            ('output_alias_tensor_names', Dict[Text, Text])])

_Signature = NamedTuple('_Signature', [('name', Text),
                                       ('signature_def', _SignatureDef)])


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _Classify(pcoll: beam.pvalue.PCollection,  # pylint: disable=invalid-name
              inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs classify PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'Classify' >> beam.ParDo(
                _BatchClassifyDoFn(inference_spec_type, shared.Shared()))
            | 'BuildPredictionLogForClassifications' >> beam.ParDo(
                _BuildPredictionLogForClassificationsDoFn()))
  else:
    raise NotImplementedError


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _Regress(pcoll: beam.pvalue.PCollection,  # pylint: disable=invalid-name
             inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs regress PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'Regress' >> beam.ParDo(
                _BatchRegressDoFn(inference_spec_type, shared.Shared()))
            | 'BuildPredictionLogForRegressions' >> beam.ParDo(
                _BuildPredictionLogForRegressionsDoFn()))
  else:
    raise NotImplementedError


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _Predict(pcoll: beam.pvalue.PCollection,  # pylint: disable=invalid-name
             inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs predict PTransform."""
  if _using_in_process_inference(inference_spec_type):
    predictions = (
        pcoll
        | 'Predict' >> beam.ParDo(
            _BatchPredictDoFn(inference_spec_type, shared.Shared())))
  else:
    predictions = (
        pcoll
        | 'RemotePredict' >> beam.ParDo(
            _RemotePredictDoFn(inference_spec_type, pcoll.pipeline.options)))
  return (predictions
          | 'BuildPredictionLogForPredictions' >> beam.ParDo(
              _BuildPredictionLogForPredictionsDoFn()))


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[tf.train.Example,
                                       tf.train.SequenceExample])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _MultiInference(pcoll: beam.pvalue.PCollection,  # pylint: disable=invalid-name
                    inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs multi inference PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (
        pcoll
        | 'MultiInference' >> beam.ParDo(
            _BatchMultiInferenceDoFn(inference_spec_type, shared.Shared()))
        | 'BuildMultiInferenceLog' >> beam.ParDo(_BuildMultiInferenceLogDoFn()))
  else:
    raise NotImplementedError


class _BaseDoFn(beam.DoFn, metaclass=abc.ABCMeta):
  """Base DoFn that performs bulk inference."""

  class _MetricsCollector(object):
    """A collector for beam metrics."""

    def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
      operation_type = _get_operation_type(inference_spec_type)
      proximity_descriptor = (
          _METRICS_DESCRIPTOR_IN_PROCESS
          if _using_in_process_inference(inference_spec_type) else
          _METRICS_DESCRIPTOR_CLOUD_AI_PREDICTION)
      namespace = util.MakeTfxNamespace(
          [_METRICS_DESCRIPTOR_INFERENCE, operation_type, proximity_descriptor])

      # Metrics
      self._inference_counter = beam.metrics.Metrics.counter(
          namespace, 'num_inferences')
      self._num_instances = beam.metrics.Metrics.counter(
          namespace, 'num_instances')
      self._inference_request_batch_size = beam.metrics.Metrics.distribution(
          namespace, 'inference_request_batch_size')
      self._inference_request_batch_byte_size = (
          beam.metrics.Metrics.distribution(
              namespace, 'inference_request_batch_byte_size'))
      # Batch inference latency in microseconds.
      self._inference_batch_latency_micro_secs = (
          beam.metrics.Metrics.distribution(
              namespace, 'inference_batch_latency_micro_secs'))
      self._model_byte_size = beam.metrics.Metrics.distribution(
          namespace, 'model_byte_size')
      # Model load latency in milliseconds.
      self._load_model_latency_milli_secs = beam.metrics.Metrics.distribution(
          namespace, 'load_model_latency_milli_secs')

      # Metrics cache
      self.load_model_latency_milli_secs_cache = None
      self.model_byte_size_cache = None

    def update_metrics_with_cache(self):
      if self.load_model_latency_milli_secs_cache is not None:
        self._load_model_latency_milli_secs.update(
            self.load_model_latency_milli_secs_cache)
        self.load_model_latency_milli_secs_cache = None
      if self.model_byte_size_cache is not None:
        self._model_byte_size.update(self.model_byte_size_cache)
        self.model_byte_size_cache = None

    def update(self, elements: List[Union[tf.train.Example,
                                          tf.train.SequenceExample]],
               latency_micro_secs: int) -> None:
      self._inference_batch_latency_micro_secs.update(latency_micro_secs)
      self._num_instances.inc(len(elements))
      self._inference_counter.inc(len(elements))
      self._inference_request_batch_size.update(len(elements))
      self._inference_request_batch_byte_size.update(
          sum(element.ByteSize() for element in elements))

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
    super().__init__()
    self._clock = None
    self._metrics_collector = self._MetricsCollector(inference_spec_type)

  def setup(self):
    self._clock = _ClockFactory.make_clock()

  def process(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]]
  ) -> Iterable[Any]:
    batch_start_time = self._clock.get_current_time_in_microseconds()
    outputs = self.run_inference(elements)
    result = self._post_process(elements, outputs)
    self._metrics_collector.update(
        elements,
        self._clock.get_current_time_in_microseconds() - batch_start_time)
    return result

  def finish_bundle(self):
    self._metrics_collector.update_metrics_with_cache()

  @abc.abstractmethod
  def run_inference(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]]
  ) -> Union[Mapping[Text, np.ndarray], Sequence[Mapping[Text, Any]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def _post_process(self, elements: List[Union[tf.train.Example,
                                               tf.train.SequenceExample]],
                    outputs: Any) -> Iterable[Any]:
    raise NotImplementedError


def _retry_on_unavailable_and_resource_error_filter(exception: Exception):
  """Retries for HttpError.

  Retries if error is unavailable (503) or resource exhausted (429).
  Resource exhausted may happen when qps or bandwidth exceeds quota.

  Args:
    exception: Exception from inference http request execution.
  Returns:
    A boolean of whether retry.
  """

  return (isinstance(exception, googleapiclient.errors.HttpError) and
          exception.resp.status in (503, 429))


@beam.typehints.with_input_types(List[Union[tf.train.Example,
                                            tf.train.SequenceExample]])
# Using output typehints triggers NotImplementedError('BEAM-2717)' on
# streaming mode on Dataflow runner.
# TODO(b/151468119): Consider to re-batch with online serving request size
# limit, and re-batch with RPC failures(InvalidArgument) regarding request size.
# @beam.typehints.with_output_types(prediction_log_pb2.PredictLog)
class _RemotePredictDoFn(_BaseDoFn):
  """A DoFn that performs predictions from a cloud-hosted TensorFlow model.

  Supports both batch and streaming processing modes.
  NOTE: Does not work on DirectRunner for streaming jobs [BEAM-7885].

  In order to request predictions, you must deploy your trained model to AI
  Platform Prediction in the TensorFlow SavedModel format. See
  [Exporting a SavedModel for prediction]
  (https://cloud.google.com/ai-platform/prediction/docs/exporting-savedmodel-for-prediction)
  for more details.

  To send binary data, you have to make sure that the name of an input ends in
  `_bytes`.

  NOTE: The returned `PredictLog` instances do not have `PredictRequest` part
  filled. The reason is that it is difficult to determine the input tensor name
  without having access to cloud-hosted model's signatures.
  """

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType,
               pipeline_options: PipelineOptions):
    super().__init__(inference_spec_type)
    self._ai_platform_prediction_model_spec = (
        inference_spec_type.ai_platform_prediction_model_spec)
    self._api_client = None

    project_id = (
        inference_spec_type.ai_platform_prediction_model_spec.project_id or
        pipeline_options.view_as(GoogleCloudOptions).project)
    if not project_id:
      raise ValueError('Either a non-empty project id or project flag in '
                       ' beam pipeline options needs be provided.')

    model_name = (
        inference_spec_type.ai_platform_prediction_model_spec.model_name)
    if not model_name:
      raise ValueError('A non-empty model name must be provided.')

    version_name = (
        inference_spec_type.ai_platform_prediction_model_spec.version_name)
    name_spec = 'projects/{}/models/{}'
    # If version is not specified, the default version for a model is used.
    if version_name:
      name_spec += '/versions/{}'
    self._full_model_name = name_spec.format(project_id, model_name,
                                             version_name)

  def setup(self):
    super().setup()
    # TODO(b/151468119): Add tfx_bsl_version and tfx_bsl_py_version to
    # user agent once custom header is supported in googleapiclient.
    self._api_client = discovery.build('ml', 'v1')

  # Retry _REMOTE_INFERENCE_NUM_RETRIES times with exponential backoff.
  @retry.with_exponential_backoff(
      initial_delay_secs=1.0,
      num_retries=_REMOTE_INFERENCE_NUM_RETRIES,
      retry_filter=_retry_on_unavailable_and_resource_error_filter)
  def _execute_request(
      self,
      request: http.HttpRequest) -> Mapping[Text, Sequence[Mapping[Text, Any]]]:
    result = request.execute()
    if 'error' in result:
      raise ValueError(result['error'])
    return result

  def _make_request(self, body: Mapping[Text, List[Any]]) -> http.HttpRequest:
    return self._api_client.projects().predict(
        name=self._full_model_name, body=body)

  def _prepare_instances_dict(
      self, elements: List[tf.train.Example]
  ) -> Generator[Mapping[Text, Any], None, None]:
    """Prepare instances by converting features to dictionary."""
    for example in elements:
      # TODO(b/151468119): support tf.train.SequenceExample
      if not isinstance(example, tf.train.Example):
        raise ValueError('Remote prediction only supports tf.train.Example')

      instance = {}
      for input_name, feature in example.features.feature.items():
        attr_name = feature.WhichOneof('kind')
        if attr_name is None:
          continue
        attr = getattr(feature, attr_name)
        values = self._parse_feature_content(
            attr.value, attr_name, self._sending_as_binary(input_name))
        # Flatten a sequence if its length is 1
        values = (values[0] if len(values) == 1 else values)
        instance[input_name] = values
      yield instance

  def _prepare_instances_serialized(
      self, elements: List[tf.train.Example]
  ) -> Generator[Mapping[Text, Text], None, None]:
    """Prepare instances by base64 encoding serialized examples."""
    for example in elements:
      yield {'b64': base64.b64encode(example.SerializeToString()).decode()}

  def _prepare_instances(
      self, elements: List[tf.train.Example]
  ) -> Generator[Mapping[Text, Any], None, None]:
    if self._ai_platform_prediction_model_spec.use_serialization_config:
      return self._prepare_instances_serialized(elements)
    else:
      return self._prepare_instances_dict(elements)

  @staticmethod
  def _sending_as_binary(input_name: Text) -> bool:
    """Whether data should be sent as binary."""
    return input_name.endswith('_bytes')

  @staticmethod
  def _parse_feature_content(values: Sequence[Any], attr_name: Text,
                             as_binary: bool) -> List[Any]:
    """Parse the content of tf.train.Feature object.

    If bytes_list, parse a list of bytes-like objects to a list of strings so
    that it would be JSON serializable.

    If float_list or int64_list, do nothing.

    If data should be sent as binary, mark it as binary by replacing it with
    a single attribute named 'b64'.
    """
    if as_binary:
      return [{'b64': base64.b64encode(x).decode()} for x in values]
    elif attr_name == 'bytes_list':
      return [x.decode() for x in values]
    else:
      # Converts proto RepeatedScalarContainer to list so it is
      # JSON-serializable
      return list(values)

  def run_inference(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]]
  ) -> Sequence[Mapping[Text, Any]]:
    body = {'instances': list(self._prepare_instances(elements))}
    request = self._make_request(body)
    response = self._execute_request(request)
    return response['predictions']

  def _post_process(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]],
      outputs: Sequence[Mapping[Text, Any]]
  ) -> Iterable[prediction_log_pb2.PredictLog]:
    result = []
    for output in outputs:
      predict_log = prediction_log_pb2.PredictLog()
      for output_alias, values in output.items():
        values = np.array(values)
        tensor_proto = tf.make_tensor_proto(
            values=values,
            dtype=tf.as_dtype(values.dtype).as_datatype_enum,
            shape=np.expand_dims(values, axis=0).shape)
        predict_log.response.outputs[output_alias].CopyFrom(tensor_proto)
      result.append(predict_log)
    return result


# TODO(b/131873699): Add typehints once
# [BEAM-8381](https://issues.apache.org/jira/browse/BEAM-8381)
# is fixed.
# TODO(b/143484017): Add batch_size back off in the case there are functional
# reasons large batch sizes cannot be handled.
class _BaseBatchSavedModelDoFn(_BaseDoFn):
  """A DoFn that runs in-process batch inference with a model.

    Models need to have the required serving signature as mentioned in
    [Tensorflow Serving](https://www.tensorflow.org/tfx/serving/signature_defs)

    This function will check model signatures first. Then it will load and run
    model inference in batch.
  """

  def __init__(
      self,
      inference_spec_type: model_spec_pb2.InferenceSpecType,
      shared_model_handle: shared.Shared,
  ):
    super().__init__(inference_spec_type)
    self._inference_spec_type = inference_spec_type
    self._shared_model_handle = shared_model_handle
    self._model_path = inference_spec_type.saved_model_spec.model_path
    self._tags = None
    self._signatures = _get_signatures(
        inference_spec_type.saved_model_spec.model_path,
        inference_spec_type.saved_model_spec.signature_name,
        _get_tags(inference_spec_type))
    self._session = None
    self._io_tensor_spec = None

  def setup(self):
    """Load the model.

    Note that worker may crash if exception is thrown in setup due
    to b/139207285.
    """

    super().setup()
    self._tags = _get_tags(self._inference_spec_type)
    self._io_tensor_spec = self._pre_process()

    if self._has_tpu_tag():
      # TODO(b/131873699): Support TPU inference.
      raise ValueError('TPU inference is not supported yet.')
    self._session = self._load_model()

  def _load_model(self):
    """Load a saved model into memory.

    Returns:
      Session instance.
    """

    def load():
      """Function for constructing shared LoadedModel."""
      # TODO(b/143484017): Do warmup and other heavy model construction here.
      result = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
      memory_before = _get_current_process_memory_in_bytes()
      start_time = self._clock.get_current_time_in_microseconds()
      tf.compat.v1.saved_model.loader.load(result, self._tags, self._model_path)
      end_time = self._clock.get_current_time_in_microseconds()
      memory_after = _get_current_process_memory_in_bytes()
      self._metrics_collector.load_model_latency_milli_secs_cache = (
          (end_time - start_time) / _MILLISECOND_TO_MICROSECOND)
      self._metrics_collector.model_byte_size_cache = (
          memory_after - memory_before)
      return result

    if not self._model_path:
      raise ValueError('Model path is not valid.')
    return self._shared_model_handle.acquire(load)

  def _pre_process(self) -> _IOTensorSpec:
    # Pre process functions will validate for each signature.
    io_tensor_specs = []
    for signature in self._signatures:
      if len(signature.signature_def.inputs) != 1:
        raise ValueError('Signature should have 1 and only 1 inputs')
      if (list(signature.signature_def.inputs.values())[0].dtype !=
          tf.string.as_datatype_enum):
        raise ValueError(
            'Input dtype is expected to be %s, got %s' %
            tf.string.as_datatype_enum,
            list(signature.signature_def.inputs.values())[0].dtype)
      io_tensor_specs.append(_signature_pre_process(signature.signature_def))
    input_tensor_name = ''
    input_tensor_alias = ''
    output_alias_tensor_names = {}
    for io_tensor_spec in io_tensor_specs:
      if not input_tensor_name:
        input_tensor_name = io_tensor_spec.input_tensor_name
        input_tensor_alias = io_tensor_spec.input_tensor_alias
      elif input_tensor_name != io_tensor_spec.input_tensor_name:
        raise ValueError('Input tensor must be the same for all Signatures.')
      for alias, tensor_name in io_tensor_spec.output_alias_tensor_names.items(
      ):
        output_alias_tensor_names[alias] = tensor_name
    if (not output_alias_tensor_names or not input_tensor_name or
        not input_tensor_alias):
      raise ValueError('No valid fetch tensors or feed tensors.')
    return _IOTensorSpec(input_tensor_alias, input_tensor_name,
                         output_alias_tensor_names)

  def _has_tpu_tag(self) -> bool:
    return (len(self._tags) == 2 and tf.saved_model.SERVING in self._tags and
            tf.saved_model.TPU in self._tags)

  def run_inference(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]]
  ) -> Mapping[Text, np.ndarray]:
    self._check_elements(elements)
    outputs = self._run_tf_operations(elements)
    return outputs

  def _run_tf_operations(
      self, elements: List[Union[tf.train.Example, tf.train.SequenceExample]]
  ) -> Mapping[Text, np.ndarray]:
    input_values = []
    for element in elements:
      input_values.append(element.SerializeToString())
    result = self._session.run(
        self._io_tensor_spec.output_alias_tensor_names,
        feed_dict={self._io_tensor_spec.input_tensor_name: input_values})
    if len(result) != len(self._io_tensor_spec.output_alias_tensor_names):
      raise RuntimeError('Output length does not match fetches')
    return result

  def _check_elements(
      self, elements: List[Union[tf.train.Example,
                                 tf.train.SequenceExample]]) -> None:
    """Unimplemented."""

    raise NotImplementedError


@beam.typehints.with_input_types(List[Union[tf.train.Example,
                                            tf.train.SequenceExample]])
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        classification_pb2.Classifications])
class _BatchClassifyDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on classification model."""

  def setup(self):
    signature_def = self._signatures[0].signature_def
    if signature_def.method_name != tf.saved_model.CLASSIFY_METHOD_NAME:
      raise ValueError(
          'BulkInferrerClassifyDoFn requires signature method '
          'name %s, got: %s' % tf.saved_model.CLASSIFY_METHOD_NAME,
          signature_def.method_name)
    super().setup()

  def _check_elements(
      self, elements: List[Union[tf.train.Example,
                                 tf.train.SequenceExample]]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Classify only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[tf.train.Example], outputs: Mapping[Text,
                                                                   np.ndarray]
  ) -> Iterable[Tuple[tf.train.Example, classification_pb2.Classifications]]:
    classifications = _post_process_classify(
        self._io_tensor_spec.output_alias_tensor_names, elements, outputs)
    return zip(elements, classifications)


@beam.typehints.with_input_types(List[Union[tf.train.Example,
                                            tf.train.SequenceExample]])
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        regression_pb2.Regression])
class _BatchRegressDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on regression model."""

  def setup(self):
    super().setup()

  def _check_elements(
      self, elements: List[Union[tf.train.Example,
                                 tf.train.SequenceExample]]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Regress only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[tf.train.Example], outputs: Mapping[Text,
                                                                   np.ndarray]
  ) -> Iterable[Tuple[tf.train.Example, regression_pb2.Regression]]:
    regressions = _post_process_regress(elements, outputs)
    return zip(elements, regressions)


@beam.typehints.with_input_types(List[Union[tf.train.Example,
                                            tf.train.SequenceExample]])
@beam.typehints.with_output_types(prediction_log_pb2.PredictLog)
class _BatchPredictDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on predict model."""

  def setup(self):
    signature_def = self._signatures[0].signature_def
    if signature_def.method_name != tf.saved_model.PREDICT_METHOD_NAME:
      raise ValueError(
          'BulkInferrerPredictDoFn requires signature method '
          'name %s, got: %s' % tf.saved_model.PREDICT_METHOD_NAME,
          signature_def.method_name)
    super().setup()

  def _check_elements(
      self, elements: List[Union[tf.train.Example,
                                 tf.train.SequenceExample]]) -> None:
    pass

  def _post_process(
      self, elements: Union[Sequence[tf.train.Example],
                            Sequence[tf.train.SequenceExample]],
      outputs: Mapping[Text, np.ndarray]
  ) -> Iterable[prediction_log_pb2.PredictLog]:
    input_tensor_alias = self._io_tensor_spec.input_tensor_alias
    signature_name = self._signatures[0].name
    batch_size = len(elements)
    for output_alias, output in outputs.items():
      if len(output.shape) < 1 or output.shape[0] != batch_size:
        raise ValueError(
            'Expected output tensor %s to have at least one '
            'dimension, with the first having a size equal to the input batch '
            'size %s. Instead found %s' %
            (output_alias, batch_size, output.shape))
    predict_log_tmpl = prediction_log_pb2.PredictLog()
    predict_log_tmpl.request.model_spec.signature_name = signature_name
    predict_log_tmpl.response.model_spec.signature_name = signature_name
    input_tensor_proto = predict_log_tmpl.request.inputs[input_tensor_alias]
    input_tensor_proto.dtype = tf.string.as_datatype_enum
    input_tensor_proto.tensor_shape.dim.add().size = 1

    result = []
    for i in range(batch_size):
      predict_log = prediction_log_pb2.PredictLog()
      predict_log.CopyFrom(predict_log_tmpl)
      predict_log.request.inputs[input_tensor_alias].string_val.append(
          elements[i].SerializeToString())
      for output_alias, output in outputs.items():
        # Mimic tensor::Split
        tensor_proto = tf.make_tensor_proto(
            values=output[i],
            dtype=tf.as_dtype(output[i].dtype).as_datatype_enum,
            shape=np.expand_dims(output[i], axis=0).shape)
        predict_log.response.outputs[output_alias].CopyFrom(tensor_proto)
      result.append(predict_log)
    return result


@beam.typehints.with_input_types(List[Union[tf.train.Example,
                                            tf.train.SequenceExample]])
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        inference_pb2.MultiInferenceResponse])
class _BatchMultiInferenceDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on multi-head model."""

  def _check_elements(
      self, elements: List[Union[tf.train.Example,
                                 tf.train.SequenceExample]]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Multi inference only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[tf.train.Example], outputs: Mapping[Text,
                                                                   np.ndarray]
  ) -> Iterable[Tuple[tf.train.Example, inference_pb2.MultiInferenceResponse]]:
    classifications = None
    regressions = None
    for signature in self._signatures:
      signature_def = signature.signature_def
      if signature_def.method_name == tf.saved_model.CLASSIFY_METHOD_NAME:
        classifications = _post_process_classify(
            self._io_tensor_spec.output_alias_tensor_names, elements, outputs)
      elif signature_def.method_name == tf.saved_model.REGRESS_METHOD_NAME:
        regressions = _post_process_regress(elements, outputs)
      else:
        raise ValueError('Signature method %s is not supported for '
                         'multi inference' % signature_def.method_name)
    result = []
    for i in range(len(elements)):
      response = inference_pb2.MultiInferenceResponse()
      for signature in self._signatures:
        signature_def = signature.signature_def
        inference_result = response.results.add()
        if (signature_def.method_name == tf.saved_model.CLASSIFY_METHOD_NAME and
            classifications):
          inference_result.classification_result.classifications.add().CopyFrom(
              classifications[i])
        elif (
            signature_def.method_name == tf.saved_model.REGRESS_METHOD_NAME and
            regressions):
          inference_result.regression_result.regressions.add().CopyFrom(
              regressions[i])
        else:
          raise ValueError('Signature method %s is not supported for '
                           'multi inference' % signature_def.method_name)
        inference_result.model_spec.signature_name = signature.name
      if len(response.results) != len(self._signatures):
        raise RuntimeError('Multi inference response result length does not '
                           'match the number of signatures')
      result.append((elements[i], response))
    return result


@beam.typehints.with_input_types(Tuple[tf.train.Example,
                                       classification_pb2.Classifications])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
class _BuildPredictionLogForClassificationsDoFn(beam.DoFn):
  """A DoFn that builds prediction log from classifications."""

  def process(
      self, element: Tuple[tf.train.Example, classification_pb2.Classifications]
  ) -> Iterable[prediction_log_pb2.PredictionLog]:
    (train_example, classifications) = element
    result = prediction_log_pb2.PredictionLog()
    result.classify_log.request.input.example_list.examples.add().CopyFrom(
        train_example)
    result.classify_log.response.result.classifications.add().CopyFrom(
        classifications)
    yield result


@beam.typehints.with_input_types(Tuple[tf.train.Example,
                                       regression_pb2.Regression])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
class _BuildPredictionLogForRegressionsDoFn(beam.DoFn):
  """A DoFn that builds prediction log from regressions."""

  def process(
      self, element: Tuple[tf.train.Example, regression_pb2.Regression]
  ) -> Iterable[prediction_log_pb2.PredictionLog]:
    (train_example, regression) = element
    result = prediction_log_pb2.PredictionLog()
    result.regress_log.request.input.example_list.examples.add().CopyFrom(
        train_example)
    result.regress_log.response.result.regressions.add().CopyFrom(regression)
    yield result


@beam.typehints.with_input_types(prediction_log_pb2.PredictLog)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
class _BuildPredictionLogForPredictionsDoFn(beam.DoFn):
  """A DoFn that builds prediction log from predictions."""

  def process(
      self, element: prediction_log_pb2.PredictLog
  ) -> Iterable[prediction_log_pb2.PredictionLog]:
    result = prediction_log_pb2.PredictionLog()
    result.predict_log.CopyFrom(element)
    yield result


@beam.typehints.with_input_types(Tuple[tf.train.Example,
                                       inference_pb2.MultiInferenceResponse])
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
class _BuildMultiInferenceLogDoFn(beam.DoFn):
  """A DoFn that builds prediction log from multi-head inference result."""

  def process(
      self, element: Tuple[tf.train.Example,
                           inference_pb2.MultiInferenceResponse]
  ) -> Iterable[prediction_log_pb2.PredictionLog]:
    (train_example, multi_inference_response) = element
    result = prediction_log_pb2.PredictionLog()
    (result.multi_inference_log.request.input.example_list.examples.add()
     .CopyFrom(train_example))
    result.multi_inference_log.response.CopyFrom(multi_inference_response)
    yield result


def _post_process_classify(
    output_alias_tensor_names: Mapping[Text, Text],
    elements: Sequence[tf.train.Example], outputs: Mapping[Text, np.ndarray]
) -> Sequence[classification_pb2.Classifications]:
  """Returns classifications from inference output."""

  # This is to avoid error "The truth value of an array with
  # more than one element is ambiguous."
  has_classes = False
  has_scores = False
  if tf.saved_model.CLASSIFY_OUTPUT_CLASSES in output_alias_tensor_names:
    classes = outputs[tf.saved_model.CLASSIFY_OUTPUT_CLASSES]
    has_classes = True
  if tf.saved_model.CLASSIFY_OUTPUT_SCORES in output_alias_tensor_names:
    scores = outputs[tf.saved_model.CLASSIFY_OUTPUT_SCORES]
    has_scores = True
  if has_classes:
    if classes.ndim != 2:
      raise ValueError('Expected Tensor shape: [batch_size num_classes] but '
                       'got %s' % classes.shape)
    if classes.dtype != tf.string.as_numpy_dtype:
      raise ValueError('Expected classes Tensor of %s. Got: %s' %
                       (tf.string.as_numpy_dtype, classes.dtype))
    if classes.shape[0] != len(elements):
      raise ValueError('Expected classes output batch size of %s, got %s' %
                       (len(elements), classes.shape[0]))
  if has_scores:
    if scores.ndim != 2:
      raise ValueError("""Expected Tensor shape: [batch_size num_classes] but
        got %s""" % scores.shape)
    if scores.dtype != tf.float32.as_numpy_dtype:
      raise ValueError('Expected classes Tensor of %s. Got: %s' %
                       (tf.float32.as_numpy_dtype, scores.dtype))
    if scores.shape[0] != len(elements):
      raise ValueError('Expected classes output batch size of %s, got %s' %
                       (len(elements), scores.shape[0]))
  num_classes = 0
  if has_classes and has_scores:
    if scores.shape[1] != classes.shape[1]:
      raise ValueError('Tensors class and score should match in shape[1]. '
                       'Got %s vs %s' % (classes.shape[1], scores.shape[1]))
    num_classes = classes.shape[1]
  elif has_classes:
    num_classes = classes.shape[1]
  elif has_scores:
    num_classes = scores.shape[1]

  result = []
  for i in range(len(elements)):
    a_classification = classification_pb2.Classifications()
    for c in range(num_classes):
      a_class = a_classification.classes.add()
      if has_classes:
        a_class.label = classes[i][c]
      if has_scores:
        a_class.score = scores[i][c]
    result.append(a_classification)
  if len(result) != len(elements):
    raise RuntimeError('Classifications length does not match elements')
  return result


def _post_process_regress(
    elements: Sequence[tf.train.Example],
    outputs: Mapping[Text, np.ndarray]) -> Sequence[regression_pb2.Regression]:
  """Returns regressions from inference output."""

  if tf.saved_model.REGRESS_OUTPUTS not in outputs:
    raise ValueError('No regression outputs found in outputs: %s' %
                     outputs.keys())
  output = outputs[tf.saved_model.REGRESS_OUTPUTS]
  batch_size = len(elements)
  if not (output.ndim == 1 or (output.ndim == 2 and output.shape[1] == 1)):
    raise ValueError("""Expected output Tensor shape to be either [batch_size]
                     or [batch_size, 1] but got %s""" % output.shape)
  if batch_size != output.shape[0]:
    raise ValueError(
        'Input batch size did not match output batch size: %s vs %s' %
        (batch_size, output.shape[0]))
  if output.dtype != tf.float32.as_numpy_dtype:
    raise ValueError('Expected output Tensor of %s. Got: %s' %
                     (tf.float32.as_numpy_dtype, output.dtype))
  if output.size != batch_size:
    raise ValueError('Expected output batch size to be %s. Got: %s' %
                     (batch_size, output.size))
  flatten_output = output.flatten()
  result = []
  for regression_result in flatten_output:
    regression = regression_pb2.Regression()
    regression.value = regression_result
    result.append(regression)

  # Add additional check to save downstream consumer checks.
  if len(result) != len(elements):
    raise RuntimeError('Regression length does not match elements')
  return result


def _signature_pre_process(signature: _SignatureDef) -> _IOTensorSpec:
  """Returns IOTensorSpec from signature."""

  if len(signature.inputs) != 1:
    raise ValueError('Signature should have 1 and only 1 inputs')
  input_tensor_alias = list(signature.inputs.keys())[0]
  if list(signature.inputs.values())[0].dtype != tf.string.as_datatype_enum:
    raise ValueError(
        'Input dtype is expected to be %s, got %s' % tf.string.as_datatype_enum,
        list(signature.inputs.values())[0].dtype)
  if signature.method_name == tf.saved_model.CLASSIFY_METHOD_NAME:
    input_tensor_name, output_alias_tensor_names = (
        _signature_pre_process_classify(signature))
  elif signature.method_name == tf.saved_model.PREDICT_METHOD_NAME:
    input_tensor_name, output_alias_tensor_names = (
        _signature_pre_process_predict(signature))
  elif signature.method_name == tf.saved_model.REGRESS_METHOD_NAME:
    input_tensor_name, output_alias_tensor_names = (
        _signature_pre_process_regress(signature))
  else:
    raise ValueError('Signature method %s is not supported' %
                     signature.method_name)
  return _IOTensorSpec(input_tensor_alias, input_tensor_name,
                       output_alias_tensor_names)


def _signature_pre_process_classify(
    signature: _SignatureDef) -> Tuple[Text, Dict[Text, Text]]:
  """Returns input tensor name and output alias tensor names from signature.

  Args:
    signature: SignatureDef

  Returns:
    A tuple of input tensor name and output alias tensor names.
  """

  if len(signature.outputs) != 1 and len(signature.outputs) != 2:
    raise ValueError('Classify signature should have 1 or 2 outputs')
  if tf.saved_model.CLASSIFY_INPUTS not in signature.inputs:
    raise ValueError('No classification inputs found in SignatureDef: %s' %
                     signature.inputs)
  input_tensor_name = signature.inputs[tf.saved_model.CLASSIFY_INPUTS].name
  output_alias_tensor_names = {}
  if (tf.saved_model.CLASSIFY_OUTPUT_CLASSES not in signature.outputs and
      tf.saved_model.CLASSIFY_OUTPUT_SCORES not in signature.outputs):
    raise ValueError(
        """Expected classification signature outputs to contain at
        least one of %s or %s. Signature was: %s""" %
        tf.saved_model.CLASSIFY_OUTPUT_CLASSES,
        tf.saved_model.CLASSIFY_OUTPUT_SCORES, signature)
  if tf.saved_model.CLASSIFY_OUTPUT_CLASSES in signature.outputs:
    output_alias_tensor_names[tf.saved_model.CLASSIFY_OUTPUT_CLASSES] = (
        signature.outputs[tf.saved_model.CLASSIFY_OUTPUT_CLASSES].name)
  if tf.saved_model.CLASSIFY_OUTPUT_SCORES in signature.outputs:
    output_alias_tensor_names[tf.saved_model.CLASSIFY_OUTPUT_SCORES] = (
        signature.outputs[tf.saved_model.CLASSIFY_OUTPUT_SCORES].name)
  return input_tensor_name, output_alias_tensor_names


def _signature_pre_process_predict(
    signature: _SignatureDef) -> Tuple[Text, Dict[Text, Text]]:
  """Returns input tensor name and output alias tensor names from signature.

  Args:
    signature: SignatureDef

  Returns:
    A tuple of input tensor name and output alias tensor names.
  """

  input_tensor_name = list(signature.inputs.values())[0].name
  output_alias_tensor_names = dict([
      (key, output.name) for key, output in signature.outputs.items()
  ])
  return input_tensor_name, output_alias_tensor_names


def _signature_pre_process_regress(
    signature: _SignatureDef) -> Tuple[Text, Dict[Text, Text]]:
  """Returns input tensor name and output alias tensor names from signature.

  Args:
    signature: SignatureDef

  Returns:
    A tuple of input tensor name and output alias tensor names.
  """

  if len(signature.outputs) != 1:
    raise ValueError('Regress signature should have 1 output')
  if tf.saved_model.REGRESS_INPUTS not in signature.inputs:
    raise ValueError('No regression inputs found in SignatureDef: %s' %
                     signature.inputs)
  input_tensor_name = signature.inputs[tf.saved_model.REGRESS_INPUTS].name
  if tf.saved_model.REGRESS_OUTPUTS not in signature.outputs:
    raise ValueError('No regression outputs found in SignatureDef: %s' %
                     signature.outputs)
  output_alias_tensor_names = {
      tf.saved_model.REGRESS_OUTPUTS:
          signature.outputs[tf.saved_model.REGRESS_OUTPUTS].name
  }
  return input_tensor_name, output_alias_tensor_names


def _using_in_process_inference(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> bool:
  return inference_spec_type.WhichOneof('type') == 'saved_model_spec'


def _get_signatures(model_path: Text, signatures: Sequence[Text],
                    tags: Sequence[Text]) -> Sequence[_Signature]:
  """Returns a sequence of {model_signature_name: signature}."""

  if signatures:
    signature_names = signatures
  else:
    signature_names = [tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  saved_model_pb = loader_impl.parse_saved_model(model_path)
  meta_graph_def = _get_meta_graph_def(saved_model_pb, tags)
  result = []
  for signature_name in signature_names:
    if signature_name in meta_graph_def.signature_def:
      result.append(
          _Signature(signature_name,
                     meta_graph_def.signature_def[signature_name]))
    else:
      raise RuntimeError('Signature %s could not be found in SavedModel' %
                         signature_name)
  return result


def _get_operation_type(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> Text:
  if _using_in_process_inference(inference_spec_type):
    signatures = _get_signatures(
        inference_spec_type.saved_model_spec.model_path,
        inference_spec_type.saved_model_spec.signature_name,
        _get_tags(inference_spec_type))
    if not signatures:
      raise ValueError('Model does not have valid signature to use')

    if len(signatures) == 1:
      method_name = signatures[0].signature_def.method_name
      if method_name == tf.saved_model.CLASSIFY_METHOD_NAME:
        return OperationType.CLASSIFICATION
      elif method_name == tf.saved_model.REGRESS_METHOD_NAME:
        return OperationType.REGRESSION
      elif method_name == tf.saved_model.PREDICT_METHOD_NAME:
        return OperationType.PREDICTION
      else:
        raise ValueError('Unsupported signature method_name %s' % method_name)
    else:
      for signature in signatures:
        method_name = signature.signature_def.method_name
        if (method_name != tf.saved_model.CLASSIFY_METHOD_NAME and
            method_name != tf.saved_model.REGRESS_METHOD_NAME):
          raise ValueError('Unsupported signature method_name for multi-head '
                           'model inference: %s' % method_name)
      return OperationType.MULTIHEAD
  else:
    # Remote inference supports predictions only.
    return OperationType.PREDICTION


def _get_meta_graph_def(saved_model_pb: _SavedModel,
                        tags: Sequence[Text]) -> _MetaGraphDef:
  """Returns MetaGraphDef from SavedModel."""

  for meta_graph_def in saved_model_pb.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set(tags):
      return meta_graph_def
  raise RuntimeError('MetaGraphDef associated with tags %s could not be '
                     'found in SavedModel' % tags)


def _get_current_process_memory_in_bytes():
  """Returns memory usage in bytes."""

  if resource is not None:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if _is_darwin():
      return usage
    return usage * 1024
  else:
    logging.warning('Resource module is not available for current platform, '
                    'memory usage cannot be fetched.')
  return 0


def _get_tags(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> Sequence[Text]:
  """Returns tags from ModelSpec."""

  if inference_spec_type.saved_model_spec.tag:
    return list(inference_spec_type.saved_model_spec.tag)
  else:
    return [tf.saved_model.SERVING]


def _is_darwin() -> bool:
  return sys.platform == 'darwin'


def _is_windows() -> bool:
  return platform.system() == 'Windows' or os.name == 'nt'


def _is_cygwin() -> bool:
  return platform.system().startswith('CYGWIN_NT')


class _Clock(object):

  def get_current_time_in_microseconds(self) -> int:
    return int(time.time() * _SECOND_TO_MICROSECOND)


class _FineGrainedClock(_Clock):

  def get_current_time_in_microseconds(self) -> int:
    return int(
        time.clock_gettime_ns(time.CLOCK_REALTIME) /  # pytype: disable=module-attr
        _MICROSECOND_TO_NANOSECOND)


class _ClockFactory(object):

  @staticmethod
  def make_clock() -> _Clock:
    if (hasattr(time, 'clock_gettime_ns') and not _is_windows()
        and not _is_cygwin()):
      return _FineGrainedClock()
    return _Clock()
