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
from concurrent import futures
import functools
import importlib
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Text, Tuple, TypeVar, Union

from absl import logging
import apache_beam as beam
from apache_beam.ml.inference import base
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.transforms import resources
from apache_beam.utils import retry
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
InputType = Union[tf.train.Example, tf.train.SequenceExample, bytes]
LoadOverrideFnType = Callable[[str, Sequence[str]], Any]
_OUTPUT_TYPE = prediction_log_pb2.PredictionLog


def _is_list_type(input_type: beam.typehints.typehints.TypeConstraint) -> bool:
  if hasattr(input_type, 'inner_type'):
    return input_type == beam.typehints.List[input_type.inner_type]
  return False


def _key_and_result_type(input_type: beam.typehints.typehints.TypeConstraint):
  """Get typehints for key and result type given an input typehint."""
  tuple_types = getattr(input_type, 'tuple_types', None)
  if tuple_types is not None and len(tuple_types) == 2:
    key_type = tuple_types[0]
    value_type = tuple_types[1]
  else:
    key_type = None
    value_type = input_type
  if _is_list_type(value_type):
    result_type = beam.typehints.List[_OUTPUT_TYPE]
  else:
    result_type = _OUTPUT_TYPE
  return key_type, result_type


def _using_in_process_inference(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> bool:
  return inference_spec_type.WhichOneof('type') == 'saved_model_spec'


def create_model_handler(
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    load_override_fn: Optional[LoadOverrideFnType],
    options_project_id: Optional[str]) -> base.ModelHandler:
  """Creates a ModelHandler based on the InferenceSpecType.

  Args:
    inference_spec_type: Model inference endpoint.
    load_override_fn: An option function to load the model, only used with
      saved models.
    options_project_id: The project id from pipeline options, only used if
      there was no project_id specified in the inference_spec_type proto.

  Returns:
    A ModelHandler appropriate for the inference_spec_type.
  """
  if _using_in_process_inference(inference_spec_type):
    return _get_saved_model_handler(inference_spec_type, load_override_fn)
  return _RemotePredictModelHandler(inference_spec_type, options_project_id)


# Output type is inferred from input.
@beam.typehints.with_input_types(Union[InputType, Tuple[_K, InputType],
                                       Tuple[_K, List[InputType]]])
class RunInferenceImpl(beam.PTransform):
  """Implementation of RunInference API."""

  def __init__(self,
               inference_spec_type: model_spec_pb2.InferenceSpecType,
               load_override_fn: Optional[LoadOverrideFnType] = None):
    """Initializes transform.

    Args:
      inference_spec_type: InferenceSpecType proto.
      load_override_fn: If provided, overrides the model loader fn of the
        underlying ModelHandler. This takes a model path and sequence of tags,
        and should return a model with interface compatible with tf.SavedModel.
    """
    self._inference_spec_type = inference_spec_type
    self._load_override_fn = load_override_fn

  # LINT.IfChange(close_to_resources)
  @staticmethod
  def _model_size_bytes(path: str) -> int:
    # We might be unable to compute the size of the model during pipeline
    # construction, but the model might still be accessible during pipeline
    # execution. In such cases we will provide a default value for the model
    # size. In general, it is a lot more costly to underestimate the size of
    # the model than to overestimate it.
    default_model_size = 1 << 30  # 1 GB.

    def file_size(directory, file):
      return max(tf.io.gfile.stat(os.path.join(directory, file)).length, 0)

    try:
      result = 0
      with futures.ThreadPoolExecutor() as executor:
        for directory, _, files in tf.io.gfile.walk(path):
          result += sum(
              executor.map(functools.partial(file_size, directory), files))
      if result == 0:
        result = default_model_size
      return result
    except OSError:
      return default_model_size

  @staticmethod
  def _make_close_to_resources(
      inference_spec_type: model_spec_pb2.InferenceSpecType) -> str:
    """Proximity resources not otherwise known (or visible) to Beam."""

    if _using_in_process_inference(inference_spec_type):
      # The model is expected to be loaded once per worker (as opposed to
      # once per thread), due to the use of beam.Shared in pertinent DoFns.
      #
      # The exact value of this constant is not important; it aims to signify
      # that there might be a non-trivial number of model loads.
      #
      # TODO(katsiapis): Auto(tune) this.
      estimated_num_workers = 100
      model_path = inference_spec_type.saved_model_spec.model_path
      model_size_bytes = RunInferenceImpl._model_size_bytes(model_path)
      return f'{model_path}[{model_size_bytes * estimated_num_workers}]'
    else:
      # The model is available remotely, so the size of the RPC traffic is
      # proportional to the size of the input.
      #
      # The exact value of this constant is not important; it aims to signify
      # that there might be a non-trivial amount of RPC traffic.
      #
      # TODO(katsiapis): Auto(tune) this.
      estimated_rpc_traffic_size_bytes = 1 << 40  # 1 TB.

      # TODO(katsiapis): Is it possible to query the AI platform to see what
      # zones the model is available in, so that we can instead provide a
      # descriptor along the lines of: f'zone1|zone2|...|zoneN[size]'?
      del estimated_rpc_traffic_size_bytes
      return ''
  # LINT.ThenChange(../../../../learning/serving/contrib/servables/tensorflow/flume/bulk-inference.cc:close_to_resources)

  def infer_output_type(self, input_type):
    key_type, result_type = _key_and_result_type(input_type)
    if key_type is not None:
      return beam.typehints.Tuple[key_type, result_type]
    return result_type

  def expand(self, examples: beam.PCollection) -> beam.PCollection:
    logging.info('RunInference on model: %s', self._inference_spec_type)
    output_type = self.infer_output_type(examples.element_type)
    # TODO(b/217271822): Do this unconditionally after BEAM-13690 is resolved.
    if resources.ResourceHint.is_registered('close_to_resources'):
      examples |= (
          'CloseToResources' >> beam.Map(lambda x: x).with_resource_hints(
              close_to_resources=self._make_close_to_resources(
                  self._inference_spec_type)))
    handler = create_model_handler(
        self._inference_spec_type, self._load_override_fn,
        examples.pipeline.options.view_as(GoogleCloudOptions).project)
    handler = _ModelHandlerWrapper(handler)
    return examples | 'BulkInference' >> base.RunInference(
        handler).with_output_types(output_type)


def _get_saved_model_handler(
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    load_override_fn: Optional[LoadOverrideFnType]) -> base.ModelHandler:
  """Get an in-process ModelHandler."""
  operation_type = _get_operation_type(inference_spec_type)
  if operation_type == _OperationType.CLASSIFICATION:
    return _ClassifyModelHandler(inference_spec_type, load_override_fn)
  elif operation_type == _OperationType.REGRESSION:
    return _RegressModelHandler(inference_spec_type, load_override_fn)
  elif operation_type == _OperationType.MULTI_INFERENCE:
    return _MultiInferenceModelHandler(inference_spec_type, load_override_fn)
  elif operation_type == _OperationType.PREDICTION:
    return _PredictModelHandler(inference_spec_type, load_override_fn)
  else:
    raise ValueError('Unsupported operation_type %s' % operation_type)


# Output type is inferred from input.
@beam.typehints.with_input_types(Union[InputType, Tuple[_K, InputType],
                                       Tuple[_K, List[InputType]]])
class RunInferencePerModelImpl(beam.PTransform):
  """Implementation of the vectorized variant of the RunInference API."""

  def __init__(self,
               inference_spec_types: Iterable[model_spec_pb2.InferenceSpecType],
               load_override_fn: Optional[LoadOverrideFnType] = None):
    """Initializes transform.

    Args:
      inference_spec_types: InferenceSpecType proto.
      load_override_fn: If provided, overrides the model loader fn of the
        underlying ModelHandler. This takes a model path and sequence of tags,
        and should return a model with interface compatible with tf.SavedModel.
    """
    self._inference_spec_types = tuple(inference_spec_types)
    self._load_override_fn = load_override_fn

  def infer_output_type(self, input_type):
    key_type, result_type = _key_and_result_type(input_type)
    result_type = beam.typehints.Tuple[(result_type,) *
                                       len(self._inference_spec_types)]
    if key_type is not None:
      return beam.typehints.Tuple[key_type, result_type]
    return result_type

  def expand(self, examples: beam.PCollection) -> beam.PCollection:
    output_type = self.infer_output_type(examples.element_type)

    # TODO(b/217442215): Obviate the need for this block (and instead rely
    # solely on the one within RunInferenceImpl::expand).
    # TODO(b/217271822): Do this unconditionally after BEAM-13690 is resolved.
    if resources.ResourceHint.is_registered('close_to_resources'):
      examples |= (
          'CloseToResources' >> beam.Map(lambda x: x).with_resource_hints(
              close_to_resources=','.join([
                  RunInferenceImpl._make_close_to_resources(s)  # pylint: disable=protected-access
                  for s in self._inference_spec_types
              ])))

    tuple_types = getattr(examples.element_type, 'tuple_types', None)
    if tuple_types is None or len(tuple_types) != 2:
      # The input is not a KV, so pair with a dummy key, run the inferences, and
      # drop the dummy key afterwards.
      return (examples
              | 'PairWithNone' >> beam.Map(lambda x: (None, x))
              | 'ApplyOnKeyedInput' >> RunInferencePerModelImpl(
                  self._inference_spec_types)
              | 'DropNone' >> beam.Values().with_output_types(output_type))

    def infer_iteration_output_type(input_type):
      """Infers ouput typehint for Iteration Ptransform based on input_type."""
      tuple_types = getattr(input_type, 'tuple_types', None)
      output_tuple_components = []
      if tuple_types is not None:
        output_tuple_components.extend(tuple_types)
        example_type = tuple_types[1]
      else:
        output_tuple_components.append(input_type)
        example_type = input_type

      if _is_list_type(example_type):
        inference_result_type = beam.typehints.List[_OUTPUT_TYPE]
      else:
        inference_result_type = _OUTPUT_TYPE
      output_tuple_components.append(inference_result_type)
      return beam.typehints.Tuple[output_tuple_components]

    @beam.ptransform_fn
    def Iteration(pcoll, inference_spec_type):  # pylint: disable=invalid-name
      return (pcoll
              | 'PairWithInput' >> beam.Map(lambda x: (x, x[1]))
              | 'RunInferenceImpl' >> RunInferenceImpl(inference_spec_type,
                                                       self._load_override_fn)
              | 'ExtendResults' >>
              beam.MapTuple(lambda k, v: k + (v,)).with_output_types(
                  infer_iteration_output_type(pcoll.element_type)))

    result = examples
    for i, inference_spec_type in enumerate(self._inference_spec_types):
      result |= f'Model[{i}]' >> Iteration(inference_spec_type)  # pylint: disable=no-value-for-parameter
    result |= 'ExtractResults' >> beam.Map(
        lambda tup: (tup[0], tuple(tup[2:]))).with_output_types(output_type)
    return result


_IOTensorSpec = NamedTuple('_IOTensorSpec',
                           [('input_tensor_alias', Text),
                            ('input_tensor_name', Text),
                            ('output_alias_tensor_names', Dict[Text, Text])])

_Signature = NamedTuple('_Signature', [('name', Text),
                                       ('signature_def', _SignatureDef)])


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


class _BaseModelHandler(base.ModelHandler, metaclass=abc.ABCMeta):
  """A basic TFX implementation of ModelHandler."""

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
    super().__init__()
    operation_type = _get_operation_type(inference_spec_type)
    proximity_descriptor = (
        _METRICS_DESCRIPTOR_IN_PROCESS
        if _using_in_process_inference(inference_spec_type) else
        _METRICS_DESCRIPTOR_CLOUD_AI_PREDICTION)
    self._metrics_namespace = util.MakeTfxNamespace(
        [_METRICS_DESCRIPTOR_INFERENCE, operation_type, proximity_descriptor])
    self._batch_elements_kwargs = {}
    for desc, val in inference_spec_type.batch_parameters.ListFields():
      self._batch_elements_kwargs[desc.name] = val

  def run_inference(
      self,
      examples: List[InputType],
      model: Any,
      inference_args=None) -> Iterable[prediction_log_pb2.PredictionLog]:
    serialized_examples = [
        e if isinstance(e, bytes) else e.SerializeToString() for e in examples
    ]
    self._check_examples(examples)
    outputs = self._run_inference(examples, serialized_examples, model)
    return self._post_process(examples, serialized_examples, outputs)

  def _check_examples(self, examples):
    pass

  def get_num_bytes(
      self, examples: Iterable[prediction_log_pb2.PredictionLog]) -> int:
    serialized_examples = [
        e if isinstance(e, bytes) else e.SerializeToString() for e in examples
    ]
    return sum(len(se) for se in serialized_examples)

  def get_metrics_namespace(self):
    return self._metrics_namespace

  def batch_elements_kwargs(self) -> Mapping[str, Any]:
    return self._batch_elements_kwargs

  @abc.abstractmethod
  def _post_process(
      self, examples: List[InputType], serialized_examples: List[bytes],
      outputs: List[Mapping[Text, Union[np.ndarray, Any]]]
  ) -> List[prediction_log_pb2.PredictionLog]:
    raise NotImplementedError

  @abc.abstractmethod
  def _run_inference(self, examples: List[InputType],
                     serialized_examples: List[bytes],
                     model) -> List[Mapping[Text, Any]]:
    raise NotImplementedError


# TODO(b/151468119): Consider to re-batch with online serving request size
# limit, and re-batch with RPC failures(InvalidArgument) regarding request size.
class _RemotePredictModelHandler(_BaseModelHandler):
  """Performs predictions from a cloud-hosted TensorFlow model.

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
               pipeline_options_project_id: Optional[str]):
    super().__init__(inference_spec_type)
    self._ai_platform_prediction_model_spec = (
        inference_spec_type.ai_platform_prediction_model_spec)
    self._api_client = None
    project_id = (
        inference_spec_type.ai_platform_prediction_model_spec.project_id or
        pipeline_options_project_id)
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

  def load_model(self):
    # TODO(b/151468119): Add tfx_bsl_version and tfx_bsl_py_version to
    # user agent once custom header is supported in googleapiclient.
    self._api_client = discovery.build('ml', 'v1')
    # load_model returns a locally hosted model. Since all these inferences
    # are run on vertexAI, no local model is present.
    return None

  def _check_examples(self, examples: List[InputType]):
    # TODO(b/131873699): Add support for tf.train.SequenceExample even when
    # use_serialization_config is not enabled (by appropriately modifying
    # _make_instances).
    allowed_types = (
        (tf.train.Example, tf.train.SequenceExample, bytes)
        if self._ai_platform_prediction_model_spec.use_serialization_config
        else tf.train.Example)
    if not all(isinstance(e, allowed_types) for e in examples):
      raise NotImplementedError(
          'RemotePredict supports raw and serialized tf.train.Example, raw and '
          'serialized tf.SequenceExample and raw bytes (the '
          'latter three only when use_serialization_config is true)')

  def _run_inference(self, examples: List[InputType],
                     serialized_examples: List[bytes],
                     model) -> List[Mapping[Text, Any]]:
    self._check_examples(examples)
    body = {'instances': self._make_instances(examples, serialized_examples)}
    request = self._api_client.projects().predict(
        name=self._full_model_name, body=body)
    response = self._execute_request(request)
    return response['predictions']

  def _post_process(
      self, examples: List[InputType], serialized_examples: List[bytes],
      outputs: List[Mapping[Text,
                            Any]]) -> List[prediction_log_pb2.PredictionLog]:
    del examples
    result = []
    for i, serialized_example in enumerate(serialized_examples):
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


class _BaseSavedModelHandler(_BaseModelHandler):
  """A spec that runs in-process batch inference with a model.

    Models need to have the required serving signature as mentioned in
    [Tensorflow Serving](https://www.tensorflow.org/tfx/serving/signature_defs)

    This function will check model signatures first. Then it will load and run
    model inference in batch.
  """

  def __init__(self, inference_spec_type: model_spec_pb2.InferenceSpecType,
               load_override_fn: Optional[LoadOverrideFnType]):
    super().__init__(inference_spec_type)
    self._inference_spec_type = inference_spec_type
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
    self._load_override_fn = load_override_fn

  def _has_tpu_tag(self) -> bool:
    return (len(self._tags) == 2 and tf.saved_model.SERVING in self._tags and
            tf.saved_model.TPU in self._tags)

  # TODO(b/159982957): Replace this with a mechinism that registers any custom
  # op.
  def _maybe_register_addon_ops(self):

    def _try_import(name):
      try:
        importlib.import_module(name)
      except (ImportError, tf.errors.NotFoundError):
        logging.info('%s is not available.', name)

    _try_import('tensorflow_text')
    _try_import('tensorflow_decision_forests')
    _try_import('struct2tensor')

  def load_model(self):
    if self._load_override_fn:
      return self._load_override_fn(self._model_path, self._tags)
    self._maybe_register_addon_ops()
    result = tf.compat.v1.Session(graph=tf.compat.v1.Graph())
    tf.compat.v1.saved_model.loader.load(result, self._tags, self._model_path)
    return result

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

  def _run_inference(self, examples: List[InputType],  # pytype: disable=signature-mismatch  # overriding-return-type-checks
                     serialized_examples: List[bytes],
                     model: Any) -> Mapping[Text, np.ndarray]:
    result = model.run(
        self._io_tensor_spec.output_alias_tensor_names,
        feed_dict={self._io_tensor_spec.input_tensor_name: serialized_examples})
    if len(result) != len(self._io_tensor_spec.output_alias_tensor_names):
      raise RuntimeError('Output length does not match fetches')
    return result


class _ClassifyModelHandler(_BaseSavedModelHandler):
  """Implements a spec for classification."""

  def _check_examples(self, examples: List[InputType]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Classify only supports raw or serialized tf.train.Example')

  def _post_process(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, examples: List[Union[tf.train.Example,
                                 bytes]], serialized_examples: List[bytes],
      outputs: Mapping[Text,
                       np.ndarray]) -> List[prediction_log_pb2.PredictionLog]:
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


class _RegressModelHandler(_BaseSavedModelHandler):
  """A DoFn that run inference on regression model."""

  def _check_examples(self, examples: List[InputType]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Regress only supports raw or serialized tf.train.Example')

  def _post_process(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, examples: List[Union[tf.train.Example,
                                 bytes]], serialized_examples: List[bytes],
      outputs: Mapping[Text,
                       np.ndarray]) -> List[prediction_log_pb2.PredictionLog]:
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


class _MultiInferenceModelHandler(_BaseSavedModelHandler):
  """A DoFn that runs inference on multi-head model."""

  def _check_examples(self, examples: List[InputType]):
    if not all(isinstance(e, (tf.train.Example, bytes)) for e in examples):
      raise ValueError(
          'Multi inference only supports raw or serialized tf.train.Example')

  def _post_process(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, examples: List[Union[tf.train.Example,
                                 bytes]], serialized_examples: List[bytes],
      outputs: Mapping[Text,
                       np.ndarray]) -> List[prediction_log_pb2.PredictionLog]:
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


class _PredictModelHandler(_BaseSavedModelHandler):
  """A DoFn that runs inference on predict model."""

  def _check_examples(self, examples: List[InputType]):
    pass

  def _post_process(  # pytype: disable=signature-mismatch  # overriding-parameter-type-checks
      self, examples: List[InputType], serialized_examples: List[bytes],
      outputs: Mapping[Text,
                       np.ndarray]) -> List[prediction_log_pb2.PredictionLog]:
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


def _get_tags(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> Sequence[Text]:
  """Returns tags from ModelSpec."""

  if inference_spec_type.saved_model_spec.tag:
    return list(inference_spec_type.saved_model_spec.tag)
  else:
    return [tf.saved_model.SERVING]


_T = TypeVar('_T')


def _flatten_examples(
    maybe_nested_examples: List[Union[_T, List[_T]]]
) -> Tuple[List[_T], Optional[List[int]], Optional[int]]:
  """Flattens nested examples, and returns corresponding nested list indices."""
  if (not maybe_nested_examples or
      not isinstance(maybe_nested_examples[0], list)):
    return maybe_nested_examples, None, None
  idx = []
  flattened = []
  for i in range(len(maybe_nested_examples)):
    for ex in maybe_nested_examples[i]:
      idx.append(i)
      flattened.append(ex)
  return flattened, idx, len(maybe_nested_examples)


def _nest_results(flat_results: Iterable[_T], idx: Optional[List[int]],
                  max_idx: Optional[int]) -> List[Union[_T, List[_T]]]:
  """Reverses operation of _flatten_examples if indices are provided."""
  if idx is None:
    return list(flat_results)
  nested_results = []
  for _ in range(max_idx):
    nested_results.append([])
  for result, i in zip(flat_results, idx):
    nested_results[i].append(result)
  return nested_results


# TODO(b/231328769): Overload batch args when available.
class _ModelHandlerWrapper(base.ModelHandler):
  """Wrapper that handles key forwarding and pre-batching of inputs.

  This wrapper accepts mapping ExampleType -> PredictType,
  and itself maps either

  * ExampleType -> PredictType

  * Tuple[K, ExampleType] -> Tuple[K, PredictType]

  * Tuple[K, List[ExampleType]] -> Tuple[K, List[PredictType]]

  The second mode can support forwarding metadata with a one-to-one relationship
  to examples, while the third supports forwarding metadata with a many-to-one
  relationship.

  Note that ExampleType can not be a Tuple or a List.
  """

  def __init__(self, model_handler: base.ModelHandler):
    super().__init__()
    self._model_handler = model_handler

  def load_model(self) -> Any:
    return self._model_handler.load_model()

  def run_inference(self,
                    batch: Sequence[Any],
                    model: Any,
                    inference_args=None) -> Sequence[Any]:
    if not batch:
      return []
    if isinstance(batch[0], tuple):
      keys, examples = zip(*batch)
    else:
      keys, examples = None, batch
    examples, nested_batch_idx, max_idx = _flatten_examples(examples)
    predictions = self._model_handler.run_inference(examples, model)
    predictions = _nest_results(predictions, nested_batch_idx, max_idx)
    if keys:
      return list(zip(keys, predictions))
    return predictions

  def get_num_bytes(self, batch: Any) -> int:
    if isinstance(batch[0], tuple):
      _, batch = zip(*batch)
    batch, _, _ = _flatten_examples(batch)
    return self._model_handler.get_num_bytes(batch)

  def get_metrics_namespace(self) -> str:
    return self._model_handler.get_metrics_namespace()
