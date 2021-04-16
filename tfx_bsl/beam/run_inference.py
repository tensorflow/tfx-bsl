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
"""Run batch inference on a saved model."""

import abc
import base64
import os
import platform
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Sequence, Text, Tuple, TypeVar, Union

from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry
from apache_beam.utils import shared
import googleapiclient
from googleapiclient import discovery
from googleapiclient import http
import numpy as np
import tensorflow as tf
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.telemetry import util

# TODO(b/140306674): stop using the internal TF API.
from tensorflow.python.saved_model import loader_impl  # pylint: disable=g-direct-tensorflow-import

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2

try:
  # pylint: disable=g-import-not-at-top
  import resource
except ImportError:
  resource = None


# TODO(b/131873699): Remove once 1.x support is dropped.
try:
  # pylint: disable=g-import-not-at-top
  # We need to import this in order to register all quantiles ops, even though
  # it's not directly used.
  from tensorflow.contrib.boosted_trees.python.ops import quantile_ops as _  # pylint: disable=unused-import
except ImportError:
  pass


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


# TODO(b/151468119): Converts this into enum once we stop supporting Python 2.7
class _OperationType(object):
  CLASSIFICATION = 'CLASSIFICATION'
  REGRESSION = 'REGRESSION'
  MULTI_INFERENCE = 'MULTI_INFERENCE'
  PREDICTION = 'PREDICTION'


_K = TypeVar('_K')
_INPUT_TYPE = Union[tf.train.Example, tf.train.SequenceExample, bytes]
_OUTPUT_TYPE = prediction_log_pb2.PredictionLog


@beam.typehints.with_input_types(Union[_INPUT_TYPE, Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Union[_OUTPUT_TYPE, Tuple[_K, _OUTPUT_TYPE]])
class RunInferenceImpl(beam.PTransform):
  """Implementation of RunInference API."""

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
    self._inference_spec_type = inference_spec_type

  def infer_output_type(self, input_type):
    tuple_types = getattr(input_type, 'tuple_types', None)
    if tuple_types and len(tuple_types) == 2:
      return Tuple[tuple_types[0], _OUTPUT_TYPE]
    else:
      return _OUTPUT_TYPE

  def expand(self, examples: beam.PCollection) -> beam.PCollection:
    logging.info('RunInference on model: %s', self._inference_spec_type)

    if self.infer_output_type(examples.element_type) is _OUTPUT_TYPE:
      maybe_add_none_key = beam.Map(lambda x: (None, x))
      maybe_drop_none_key = beam.MapTuple(lambda _, v: v)
    else:
      identity = beam.Map(lambda x: x)
      maybe_add_none_key = identity
      maybe_drop_none_key = identity
    batched_keyed_examples = (
        examples
        | 'MaybeAddNoneKey' >> maybe_add_none_key
        | 'BatchExamples' >> beam.BatchElements())

    # pylint: disable=no-value-for-parameter
    operation_type = _get_operation_type(self._inference_spec_type)
    if operation_type == _OperationType.CLASSIFICATION:
      result = batched_keyed_examples | 'Classify' >> _Classify(
          self._inference_spec_type)
    elif operation_type == _OperationType.REGRESSION:
      result = batched_keyed_examples | 'Regress' >> _Regress(
          self._inference_spec_type)
    elif operation_type == _OperationType.MULTI_INFERENCE:
      result = (
          batched_keyed_examples
          | 'MultiInference' >> _MultiInference(self._inference_spec_type))
    elif operation_type == _OperationType.PREDICTION:
      result = batched_keyed_examples | 'Predict' >> _Predict(
          self._inference_spec_type)
    else:
      raise ValueError('Unsupported operation_type %s' % operation_type)

    result |= 'MaybeDropNoneKey' >> maybe_drop_none_key

    return result


_IOTensorSpec = NamedTuple('_IOTensorSpec',
                           [('input_tensor_alias', Text),
                            ('input_tensor_name', Text),
                            ('output_alias_tensor_names', Dict[Text, Text])])

_Signature = NamedTuple('_Signature', [('name', Text),
                                       ('signature_def', _SignatureDef)])


@beam.ptransform_fn
@beam.typehints.with_input_types(List[Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Tuple[_K, prediction_log_pb2.PredictionLog])
def _Classify(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs classify PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'Classify' >> beam.ParDo(
                _BatchClassifyDoFn(inference_spec_type, shared.Shared())))
  else:
    raise NotImplementedError


@beam.ptransform_fn
@beam.typehints.with_input_types(List[Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Tuple[_K, prediction_log_pb2.PredictionLog])
def _Regress(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs regress PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'Regress' >> beam.ParDo(
                _BatchRegressDoFn(inference_spec_type, shared.Shared())))
  else:
    raise NotImplementedError


@beam.ptransform_fn
@beam.typehints.with_input_types(List[Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Tuple[_K, prediction_log_pb2.PredictionLog])
def _MultiInference(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs multi inference PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'MultiInference' >> beam.ParDo(
                _BatchMultiInferenceDoFn(inference_spec_type, shared.Shared())))
  else:
    raise NotImplementedError


@beam.ptransform_fn
@beam.typehints.with_input_types(List[Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Tuple[_K, prediction_log_pb2.PredictionLog])
def _Predict(  # pylint: disable=invalid-name
    pcoll: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType):
  """Performs predict PTransform."""
  if _using_in_process_inference(inference_spec_type):
    return (pcoll
            | 'Predict' >> beam.ParDo(
                _BatchPredictDoFn(inference_spec_type, shared.Shared())))
  else:
    return (
        pcoll
        | 'RemotePredict'>> beam.ParDo(
            _BatchRemotePredictDoFn(
                inference_spec_type, pcoll.pipeline.options)))


@beam.typehints.with_input_types(List[Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Tuple[_K, prediction_log_pb2.PredictionLog])
class _BaseBatchDoFn(beam.DoFn, metaclass=abc.ABCMeta):
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

    def update(self, examples_count: int, examples_byte_size: int,
               latency_micro_secs: int):
      self._inference_batch_latency_micro_secs.update(latency_micro_secs)
      self._num_instances.inc(examples_count)
      self._inference_counter.inc(examples_count)
      self._inference_request_batch_size.update(examples_count)
      self._inference_request_batch_byte_size.update(examples_byte_size)

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
    super().__init__()
    self._metrics_collector = self._MetricsCollector(inference_spec_type)
    self._clock = None

  def setup(self):
    self._clock = _ClockFactory.make_clock()

  def process(
      self,
      keyed_examples: List[Tuple[_K, _INPUT_TYPE]]
      ) -> Iterable[Tuple[_K, prediction_log_pb2.PredictionLog]]:
    batch_start_time = self._clock.get_current_time_in_microseconds()
    keys, examples = zip(*keyed_examples)
    serialized_examples = [
        e if isinstance(e, bytes) else e.SerializeToString() for e in examples
    ]
    examples_count = len(serialized_examples)
    examples_byte_size = sum(len(se) for se in serialized_examples)
    self._check_examples(examples)
    outputs = self._run_inference(examples, serialized_examples)
    result = list(
        zip(keys, self._post_process(examples, serialized_examples, outputs)))
    self._metrics_collector.update(
        examples_count, examples_byte_size,
        self._clock.get_current_time_in_microseconds() - batch_start_time)
    return result

  def finish_bundle(self):
    self._metrics_collector.update_metrics_with_cache()

  @abc.abstractmethod
  def _check_examples(self, examples: List[_INPUT_TYPE]):
    raise NotImplementedError

  @abc.abstractmethod
  def _run_inference(
      self, examples: List[_INPUT_TYPE], serialized_examples: List[bytes]
      ) -> List[Mapping[Text, Union[np.ndarray, Any]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def _post_process(
      self,
      examples: List[_INPUT_TYPE],
      serialized_examples: List[bytes],
      outputs: List[Mapping[Text, Union[np.ndarray, Any]]]
      ) -> List[prediction_log_pb2.PredictionLog]:
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


# TODO(b/151468119): Consider to re-batch with online serving request size
# limit, and re-batch with RPC failures(InvalidArgument) regarding request size.
class _BatchRemotePredictDoFn(_BaseBatchDoFn):
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

  def _make_instances(
      self,
      examples: List[Union[tf.train.Example, tf.train.SequenceExample]],
      serialized_examples: List[bytes]
      )-> List[Mapping[Text, Any]]:
    if self._ai_platform_prediction_model_spec.use_serialization_config:
      return [{'b64': base64.b64encode(se).decode()}
              for se in serialized_examples]
    else:
      result = []
      for example in examples:
        instance = {}
        for name, feature in example.features.feature.items():
          attribute_kind = feature.WhichOneof('kind')
          if attribute_kind is None:
            continue
          values = self._make_values(name, feature, attribute_kind)
          instance[name] = values[0] if len(values) == 1 else values
        result.append(instance)
      return result

  @staticmethod
  def _make_values(name: Text, feature: Any, attribute_kind: Text) -> List[Any]:
    values = getattr(feature, attribute_kind).value
    if name.endswith('_bytes'):
      return [{'b64': base64.b64encode(x).decode()} for x in values]
    elif attribute_kind == 'bytes_list':
      return [x.decode() for x in values]
    else:
      # Converts proto RepeatedScalarContainer to list so it is
      # JSON-serializable.
      return list(values)

  def setup(self):
    super().setup()
    # TODO(b/151468119): Add tfx_bsl_version and tfx_bsl_py_version to
    # user agent once custom header is supported in googleapiclient.
    self._api_client = discovery.build('ml', 'v1')

  def _check_examples(self, examples: List[_INPUT_TYPE]):
    # TODO(b/131873699): Add support for tf.train.SequenceExample even when
    # use_serialization_config is not enabled (by appropriately modifying
    # _make_instances).
    allowed_types = (
        (tf.train.Example, tf.train.SequenceExample, bytes)
        if self._ai_platform_prediction_model_spec.use_serialization_config
        else tf.train.Example)
    if not all(isinstance(e, allowed_types) for e in examples):
      raise NotImplementedError(
          'RemotePredict supports raw and serialized tf.train.Example and '
          'raw and serialized tf.SequenceExample (the latter only when '
          'use_serialization_config is strue)')

  def _run_inference(
      self, examples: List[_INPUT_TYPE], serialized_examples: List[bytes]
      ) -> List[Mapping[Text, Any]]:
    body = {'instances': self._make_instances(examples, serialized_examples)}
    request = self._api_client.projects().predict(
        name=self._full_model_name, body=body)
    response = self._execute_request(request)
    return response['predictions']

  def _post_process(
      self,
      examples: List[_INPUT_TYPE],
      serialize_examples: List[bytes],
      outputs: List[Mapping[Text, Any]]
      ) -> List[prediction_log_pb2.PredictionLog]:
    del examples
    result = []
    for i, serialized_example in enumerate(serialize_examples):
      prediction_log = prediction_log_pb2.PredictionLog()
      predict_log = prediction_log.predict_log
      input_tensor_proto = predict_log.request.inputs[
          tf.saved_model.PREDICT_INPUTS]
      input_tensor_proto.dtype = tf.string.as_datatype_enum
      input_tensor_proto.tensor_shape.dim.add().size = 1
      input_tensor_proto.string_val.append(serialized_example)
      for output_alias, values in outputs[i].items():
        values = np.array(values)
        tensor_proto = tf.make_tensor_proto(
            values=values,
            dtype=tf.as_dtype(values.dtype).as_datatype_enum,
            shape=np.expand_dims(values, axis=0).shape)
        predict_log.response.outputs[output_alias].CopyFrom(tensor_proto)
      result.append(prediction_log)
    return result


# TODO(b/143484017): Add batch_size back off in the case there are functional
# reasons large batch sizes cannot be handled.
class _BaseBatchSavedModelDoFn(_BaseBatchDoFn):
  """A DoFn that runs in-process batch inference with a model.

    Models need to have the required serving signature as mentioned in
    [Tensorflow Serving](https://www.tensorflow.org/tfx/serving/signature_defs)

    This function will check model signatures first. Then it will load and run
    model inference in batch.
  """

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType,
               shared_model_handle: shared.Shared):
    super().__init__(inference_spec_type)
    self._inference_spec_type = inference_spec_type
    self._shared_model_handle = shared_model_handle
    self._model_path = inference_spec_type.saved_model_spec.model_path
    if not self._model_path:
      raise ValueError('Model path is not valid.')
    self._tags = _get_tags(inference_spec_type)
    self._signatures = _get_signatures(
        inference_spec_type.saved_model_spec.model_path,
        inference_spec_type.saved_model_spec.signature_name, self._tags)
    self._io_tensor_spec = self._make_io_tensor_spec()
    if self._has_tpu_tag():
      # TODO(b/161563144): Support TPU inference.
      raise NotImplementedError('TPU inference is not supported yet.')
    self._session = None

  def _has_tpu_tag(self) -> bool:
    return (len(self._tags) == 2 and tf.saved_model.SERVING in self._tags and
            tf.saved_model.TPU in self._tags)

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
    return self._shared_model_handle.acquire(load)

  def _make_io_tensor_spec(self) -> _IOTensorSpec:
    # Pre process functions will validate for each signature.
    io_tensor_specs = []
    for signature in self._signatures:
      if len(signature.signature_def.inputs) != 1:
        raise ValueError('Signature should have 1 and only 1 inputs')
      if (list(signature.signature_def.inputs.values())[0].dtype !=
          tf.string.as_datatype_enum):
        raise ValueError(
            'Input dtype is expected to be %s, got %s' %
            (tf.string.as_datatype_enum,
             list(signature.signature_def.inputs.values())[0].dtype))
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

  def setup(self):
    """Load the model.

    Note that worker may crash if exception is thrown in setup due
    to b/139207285.
    """
    super().setup()
    self._session = self._load_model()

  def _run_inference(
      self, examples: List[_INPUT_TYPE], serialized_examples: List[bytes]
      ) -> Mapping[Text, np.ndarray]:
    result = self._session.run(
        self._io_tensor_spec.output_alias_tensor_names,
        feed_dict={self._io_tensor_spec.input_tensor_name: serialized_examples})
    if len(result) != len(self._io_tensor_spec.output_alias_tensor_names):
      raise RuntimeError('Output length does not match fetches')
    return result


class _BatchClassifyDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on classification model."""

  def _check_examples(self, examples: List[_INPUT_TYPE]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Classify only supports raw or serialized tf.train.Example')

  def _post_process(
      self,
      examples: List[Union[tf.train.Example, bytes]],
      serialized_examples: List[bytes],
      outputs: Mapping[Text, np.ndarray]
  ) -> List[prediction_log_pb2.PredictionLog]:
    del serialized_examples
    # TODO(b/131873699): Can we fold prediction_log_pb2.PredictionLog building
    # into _post_process_classify?
    classifications = _post_process_classify(
        self._io_tensor_spec.output_alias_tensor_names, examples, outputs)
    result = []
    for example, classification in zip(examples, classifications):
      prediction_log = prediction_log_pb2.PredictionLog()
      input_example = (prediction_log.classify_log.request.input.example_list
                       .examples.add())
      (input_example.ParseFromString
       if isinstance(example, bytes)
       else input_example.CopyFrom)(example)
      (prediction_log.classify_log.response.result.classifications.add()
       .CopyFrom(classification))
      result.append(prediction_log)
    return result


class _BatchRegressDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on regression model."""

  def _check_examples(self, examples: List[_INPUT_TYPE]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Regress only supports raw or serialized tf.train.Example')

  def _post_process(
      self,
      examples: List[Union[tf.train.Example, bytes]],
      serialized_examples: List[bytes],
      outputs: Mapping[Text, np.ndarray]
      ) -> List[prediction_log_pb2.PredictionLog]:
    del serialized_examples
    # TODO(b/131873699): Can we fold prediction_log_pb2.PredictionLog building
    # into _post_process_regress?
    regressions = _post_process_regress(examples, outputs)
    result = []
    for example, regression in zip(examples, regressions):
      prediction_log = prediction_log_pb2.PredictionLog()
      input_example = (prediction_log.regress_log.request.input.example_list
                       .examples.add())
      (input_example.ParseFromString
       if isinstance(example, bytes)
       else input_example.CopyFrom)(example)
      prediction_log.regress_log.response.result.regressions.add().CopyFrom(
          regression)
      result.append(prediction_log)
    return result


class _BatchMultiInferenceDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on multi-head model."""

  def _check_examples(self, examples: List[_INPUT_TYPE]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Multi inference only supports raw or serialized tf.train.Example')

  def _post_process(
      self,
      examples: List[Union[tf.train.Example, bytes]],
      serialized_examples: List[bytes],
      outputs: Mapping[Text, np.ndarray]
      ) -> List[prediction_log_pb2.PredictionLog]:
    del serialized_examples
    classifications = None
    regressions = None
    for signature in self._signatures:
      signature_def = signature.signature_def
      if signature_def.method_name == tf.saved_model.CLASSIFY_METHOD_NAME:
        classifications = _post_process_classify(
            self._io_tensor_spec.output_alias_tensor_names, examples, outputs)
      elif signature_def.method_name == tf.saved_model.REGRESS_METHOD_NAME:
        regressions = _post_process_regress(examples, outputs)
      else:
        raise ValueError('Signature method %s is not supported for '
                         'multi inference' % signature_def.method_name)
    result = []
    for i, example in enumerate(examples):
      prediction_log = prediction_log_pb2.PredictionLog()
      input_example = (prediction_log.multi_inference_log.request.input
                       .example_list.examples.add())
      (input_example.ParseFromString
       if isinstance(example, bytes)
       else input_example.CopyFrom)(example)
      response = prediction_log.multi_inference_log.response
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
      result.append(prediction_log)
    return result


class _BatchPredictDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on predict model."""

  def _check_examples(self, examples: List[_INPUT_TYPE]):
    pass

  def _post_process(
      self,
      examples: List[_INPUT_TYPE],
      serialized_examples: List[bytes],
      outputs: Mapping[Text, np.ndarray]
      ) -> List[prediction_log_pb2.PredictionLog]:
    del examples
    input_tensor_alias = self._io_tensor_spec.input_tensor_alias
    signature_name = self._signatures[0].name
    batch_size = len(serialized_examples)
    for output_alias, output in outputs.items():
      if len(output.shape) < 1 or output.shape[0] != batch_size:
        raise ValueError(
            'Expected output tensor %s to have at least one '
            'dimension, with the first having a size equal to the input batch '
            'size %s. Instead found %s' %
            (output_alias, batch_size, output.shape))
    result = []
    for i, serialized_example in enumerate(serialized_examples):
      prediction_log = prediction_log_pb2.PredictionLog()
      predict_log = prediction_log.predict_log
      input_tensor_proto = predict_log.request.inputs[input_tensor_alias]
      input_tensor_proto.dtype = tf.string.as_datatype_enum
      input_tensor_proto.tensor_shape.dim.add().size = 1
      input_tensor_proto.string_val.append(serialized_example)
      predict_log.request.model_spec.signature_name = signature_name
      predict_log.response.model_spec.signature_name = signature_name
      for output_alias, output in outputs.items():
        # Mimic tensor::Split
        values = output[i]
        tensor_proto = tf.make_tensor_proto(
            values=values,
            dtype=tf.as_dtype(values.dtype).as_datatype_enum,
            shape=np.expand_dims(values, axis=0).shape)
        predict_log.response.outputs[output_alias].CopyFrom(tensor_proto)
      result.append(prediction_log)
    return result


def _post_process_classify(
    output_alias_tensor_names: Mapping[Text, Text],
    examples: List[tf.train.Example], outputs: Mapping[Text, np.ndarray]
) -> List[classification_pb2.Classifications]:
  """Returns classifications from inference output."""

  # This is to avoid error "The truth value of an array with
  # more than one example is ambiguous."
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
    if classes.shape[0] != len(examples):
      raise ValueError('Expected classes output batch size of %s, got %s' %
                       (len(examples), classes.shape[0]))
  if has_scores:
    if scores.ndim != 2:
      raise ValueError("""Expected Tensor shape: [batch_size num_classes] but
        got %s""" % scores.shape)
    if scores.dtype != tf.float32.as_numpy_dtype:
      raise ValueError('Expected classes Tensor of %s. Got: %s' %
                       (tf.float32.as_numpy_dtype, scores.dtype))
    if scores.shape[0] != len(examples):
      raise ValueError('Expected classes output batch size of %s, got %s' %
                       (len(examples), scores.shape[0]))
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
  for i in range(len(examples)):
    classifications = classification_pb2.Classifications()
    for c in range(num_classes):
      klass = classifications.classes.add()
      if has_classes:
        klass.label = classes[i][c]
      if has_scores:
        klass.score = scores[i][c]
    result.append(classifications)
  return result


def _post_process_regress(
    examples: List[tf.train.Example],
    outputs: Mapping[Text, np.ndarray]) -> List[regression_pb2.Regression]:
  """Returns regressions from inference output."""

  if tf.saved_model.REGRESS_OUTPUTS not in outputs:
    raise ValueError('No regression outputs found in outputs: %s' %
                     outputs.keys())
  output = outputs[tf.saved_model.REGRESS_OUTPUTS]
  batch_size = len(examples)
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
  for value in flatten_output:
    regression = regression_pb2.Regression()
    regression.value = value
    result.append(regression)
  # Add additional check to save downstream consumer checks.
  if len(result) != len(examples):
    raise RuntimeError('Regression length does not match examples')
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
  elif signature.method_name == tf.saved_model.REGRESS_METHOD_NAME:
    input_tensor_name, output_alias_tensor_names = (
        _signature_pre_process_regress(signature))
  elif signature.method_name == tf.saved_model.PREDICT_METHOD_NAME:
    input_tensor_name, output_alias_tensor_names = (
        _signature_pre_process_predict(signature))
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
        return _OperationType.CLASSIFICATION
      elif method_name == tf.saved_model.REGRESS_METHOD_NAME:
        return _OperationType.REGRESSION
      elif method_name == tf.saved_model.PREDICT_METHOD_NAME:
        return _OperationType.PREDICTION
      else:
        raise ValueError('Unsupported signature method_name %s' % method_name)
    else:
      for signature in signatures:
        method_name = signature.signature_def.method_name
        if (method_name != tf.saved_model.CLASSIFY_METHOD_NAME and
            method_name != tf.saved_model.REGRESS_METHOD_NAME):
          raise ValueError('Unsupported signature method_name for multi-head '
                           'model inference: %s' % method_name)
      return _OperationType.MULTI_INFERENCE
  else:
    # Remote inference supports predictions only.
    return _OperationType.PREDICTION


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
    if (hasattr(time, 'clock_gettime_ns') and not _is_windows() and
        not _is_cygwin()):
      return _FineGrainedClock()
    return _Clock()
