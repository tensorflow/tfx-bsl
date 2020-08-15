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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
import base64
import collections
import os
import platform
import sys
import time
try:
  import resource
except ImportError:
  resource = None

from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.utils import retry
import googleapiclient
from googleapiclient import discovery
from googleapiclient import http
import numpy as np
import six
import tensorflow as tf
from tfx_bsl.beam import shared
from tfx_bsl.public.proto import model_spec_pb2
from tfx_bsl.telemetry import util
from typing import Any, Generator, Iterable, List, Mapping, Optional, \
  Sequence, Text, Tuple, Union

# TODO(b/140306674): stop using the internal TF API.
from tensorflow.python.saved_model import loader_impl
from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import inference_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_serving.apis import regression_pb2


# TODO(b/131873699): Remove once 1.x support is dropped.
# pylint: disable=g-import-not-at-top
try:
  # We need to import this in order to register all quantiles ops, even though
  # it's not directly used.
  from tensorflow.contrib.boosted_trees.python.ops import quantile_ops as _  # pylint: disable=unused-import
except ImportError:
  pass

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

# Public facing type aliases
ExampleType = Union[tf.train.Example, tf.train.SequenceExample]
QueryType = Tuple[Union[model_spec_pb2.InferenceSpecType, None], ExampleType]

_QueryBatchType = Tuple[
  Union[model_spec_pb2.InferenceSpecType, None],
  List[ExampleType]
]


# TODO(b/151468119): Converts this into enum once we stop supporting Python 2.7
class OperationType(object):
  CLASSIFICATION = 'CLASSIFICATION'
  REGRESSION = 'REGRESSION'
  PREDICTION = 'PREDICTION'
  MULTIHEAD = 'MULTIHEAD'


@beam.typehints.with_input_types(Union[ExampleType, QueryType])
# TODO(BEAM-10258): add type output annotations for polymorphic with_errors API
class RunInferenceImpl(beam.PTransform):
  """Implementation of RunInference API.

  Note: inference with a model PCollection and inference on queries require
    a Beam runner with stateful DoFn support.

  Args:
    examples: A PCollection containing examples. If inference_spec_type is
      None, this is interpreted as a PCollection of queries:
      (InferenceSpecType, Example)
    inference_spec_type: Model inference endpoint. Can be one of:
      - InferenceSpecType: specifies a fixed model to use for inference.
      - PCollection[InferenceSpecType]: specifies a secondary PCollection of
        models. Each example will use the most recent model for inference.
        (requires stateful DoFn support)
      - None: indicates that the primary PCollection contains
        (InferenceSpecType, Example) tuples. (requires stateful DoFn support)

  Returns:
    A PCollection containing prediction logs.
    Or, if with_errors() is enabled, a dict containing predictions and errors:
    {'predictions': ..., 'errors': ...}

  Raises:
    ValueError: When operation is not supported.
    NotImplementedError: If the selected API is not supported by the current
      runner.
  """
  def __init__(
    self, inference_spec_type: Union[model_spec_pb2.InferenceSpecType,
                                     beam.pvalue.PCollection] = None
  ):
    self._inference_spec_type = inference_spec_type
    self._with_errors = False

  def with_errors(self):
    """Enables runtime error handling.

    Once enabled, RunInference will catch runtime errors and return a dict
    containing a predictions stream and an errors stream:
    {
      'predictions': ...,
      'errors': ...
    }
    """
    self._with_errors = True
    return self

  def expand(self, examples):
    inference_results = None
    if type(self._inference_spec_type) is model_spec_pb2.InferenceSpecType:
      logging.info('RunInference on model: %s', self._inference_spec_type)
      inference_results = (
        examples
        | 'Format as queries' >> beam.Map(lambda x: (None, x))
        | '_RunInferenceCoreOnFixedModel' >> _RunInferenceCore(
          fixed_inference_spec_type=self._inference_spec_type,
          catch_errors=self._with_errors)
      )
    elif type(self._inference_spec_type) is beam.pvalue.PCollection:
      if not _runner_supports_stateful_dofn(examples.pipeline.runner):
        raise NotImplementedError(
          'Model streaming inference requires stateful DoFn support which is'
          'not provided by the current runner: %s'
          % repr(examples.pipeline.runner))

      logging.info('RunInference on dynamic models')
      inference_results = (
        examples
        | 'Join examples' >> _TemporalJoin(self._inference_spec_type)
        | '_RunInferenceCoreOnDynamicModel' >> _RunInferenceCore(
          catch_errors=self._with_errors))
    elif self._inference_spec_type is None:
      if not _runner_supports_stateful_dofn(examples.pipeline.runner):
        raise NotImplementedError(
          'Inference on queries requires stateful DoFn support which is not'
          'provided by the current runner: %s'
          % repr(examples.pipeline.runner))

      logging.info('RunInference on queries')
      inference_results = (
        examples | '_RunInferenceCoreOnQueries' >> _RunInferenceCore(
          catch_errors=self._with_errors))
    else:
      raise ValueError('Invalid type for inference_spec_type: %s'
                       % type(self._inference_spec_type))

    if self._with_errors:
      return inference_results
    else:
      return inference_results['predictions']


@beam.ptransform_fn
@beam.typehints.with_input_types(QueryType)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def _RunInferenceCore(
    queries: beam.pvalue.PCollection,
    fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None,
    catch_errors: bool = False
) -> beam.pvalue.PCollection:
  """Runs inference on queries and returns prediction logs.

  This internal run inference implementation operates on queries. Internally,
  these queries are grouped by model and inference runs in batches. If a
  fixed_inference_spec_type is provided, this spec is used for all inference
  requests which enables pre-configuring the model during pipeline
  construction. If the fixed_inference_spec_type is not provided, each input
  query must contain a valid InferenceSpecType and models will be loaded
  dynamically at runtime.

  Args:
    queries: A PCollection containing QueryType tuples.
    fixed_inference_spec_type: An optional model inference endpoint. If
      specified, this is "preloaded" during inference and models specified in
      query tuples are ignored. This requires the InferenceSpecType to be known
      at pipeline creation time. If this fixed_inference_spec_type is not
      provided, each input query must contain a valid InferenceSpecType and
      models will be loaded dynamically at runtime.
    catch_errors: if True, runtime errors will be caught and emitted in a
      separate 'errors' PCollection. Otherwise, runtime errors will be thrown
      in their original location.

  Returns:
    A dict containing a prediction and error PCollection:
    {
      'predictions': ...,
      'errors': ...
    }
    If catch_errors is False, the 'errors' PCollection will be empty.

  Raises:
    ValueError: when the fixed_inference_spec_type is invalid.
  """
  batched_queries = None
  if _runner_supports_stateful_dofn(queries.pipeline.runner):
    batched_queries = queries | 'BatchQueries' >> _BatchQueries()
  else:
    # If the current runner does not support stateful DoFn's, we fall back to
    # a simpler batching operation that assumes all queries share the same
    # inference spec.
    batched_queries = queries | 'BatchQueriesSimple' >> _BatchQueriesSimple()

  inference_results = None

  if fixed_inference_spec_type is None:
    # operation type is determined at runtime
    split = batched_queries | 'SplitByOperation' >> _SplitByOperation(
      catch_errors=catch_errors)

    operation_inference_results = [
      split[OperationType.CLASSIFICATION] | 'Classify' >> _Classify(
        catch_errors=catch_errors),
      split[OperationType.REGRESSION] | 'Regress' >> _Regress(
        catch_errors=catch_errors),
      split[OperationType.PREDICTION] | 'Predict' >> _Predict(
        catch_errors=catch_errors),
      split[OperationType.MULTIHEAD] | 'MultiInference' >> _MultiInference(
        catch_errors=catch_errors)
    ]

    predictions = (
      [x['predictions'] for x in operation_inference_results]
      | 'Flatten predictions' >> beam.Flatten())

    errors = (
      [split['errors']] + [x['errors'] for x in operation_inference_results]
      | 'Flatten errors' >> beam.Flatten())

    inference_results = {'predictions': predictions, 'errors': errors}
  else:
    # operation type is determined at pipeline construction time
    operation_type = _get_operation_type(fixed_inference_spec_type)

    if operation_type == OperationType.CLASSIFICATION:
      inference_results = batched_queries | 'Classify' >> _Classify(
        fixed_inference_spec_type=fixed_inference_spec_type,
        catch_errors=catch_errors)
    elif operation_type == OperationType.REGRESSION:
      inference_results = batched_queries | 'Regress' >> _Regress(
        fixed_inference_spec_type=fixed_inference_spec_type,
        catch_errors=catch_errors)
    elif operation_type == OperationType.PREDICTION:
      inference_results = batched_queries | 'Predict' >> _Predict(
        fixed_inference_spec_type=fixed_inference_spec_type,
        catch_errors=catch_errors)
    elif operation_type == OperationType.MULTIHEAD:
      inference_results = (
        batched_queries | 'MultiInference' >> _MultiInference(
          fixed_inference_spec_type=fixed_inference_spec_type,
          catch_errors=catch_errors))
    else:
      raise ValueError('Unsupported operation_type %s' % operation_type)

  return inference_results


@beam.ptransform_fn
@beam.typehints.with_input_types(QueryType)
@beam.typehints.with_output_types(_QueryBatchType)
def _BatchQueries(queries: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
  """Groups queries into batches."""

  def _add_key(query: QueryType) -> Tuple[bytes, QueryType]:
    """Adds serialized proto as key for QueryType tuples."""
    inference_spec, example = query
    key = (inference_spec.SerializeToString() if inference_spec else b'')
    return (key, (inference_spec, example))

  def _to_query_batch(
    query_list: Tuple[bytes, List[QueryType]]
  ) -> _QueryBatchType:
    """Converts a list of queries to a logical _QueryBatch."""
    inference_spec = query_list[1][0][0]
    examples = [x[1] for x in query_list[1]]
    return (inference_spec, examples)

  batches = (
    queries
    | 'Serialize inference_spec as key' >> beam.Map(_add_key)
    # TODO(hgarrereyn): GroupIntoBatches with automatic batch sizes
    | 'Batch' >> beam.GroupIntoBatches(1000)
    | 'Convert to QueryBatch' >> beam.Map(_to_query_batch)
  )
  return batches


@beam.ptransform_fn
@beam.typehints.with_input_types(QueryType)
@beam.typehints.with_output_types(_QueryBatchType)
def _BatchQueriesSimple(
  queries: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
  """Groups queries into batches.

  This version of _BatchQueries uses beam.BatchElements and works in runners
  that do not support stateful DoFn's. However, in this case we need to make
  the assumption that all queries share the same inference_spec.
  """

  def _to_query_batch(query_list: List[QueryType]) -> _QueryBatchType:
    """Converts a list of queries to a logical _QueryBatch."""
    inference_spec = query_list[0][0]
    examples = [x[1] for x in query_list]
    return (inference_spec, examples)

  batches = (
    queries
    | 'Batch' >> beam.BatchElements()
    | 'ToQueryBatch' >> beam.Map(_to_query_batch)
  )
  return batches


@beam.ptransform_fn
@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(_QueryBatchType)
def _SplitByOperation(batches, catch_errors=False):
  """A PTransform that splits a _QueryBatchType PCollection based on operation.

  Benchmarks demonstrated that this transform was a bottleneck (comprising
  nearly 25% of the total RunInference walltime) since looking up the operation
  type requires reading the saved model signature from disk. To improve
  performance, we use a caching layer inside each DoFn instance that saves a
  mapping of:

    {inference_spec.SerializeToString(): (operation_type, Optional[Exception])}

  In practice this cache reduces _SplitByOperation walltime by more than 90%.

  Args:
    catch_errors: if True, model signature errors are captured in a separate
      "errors" output containing Tuple[Exception, _QueryBatchType].

  Returns a DoOutputsTuple with keys:
    - "errors"
    - OperationType.CLASSIFICATION
    - OperationType.REGRESSION
    - OperationType.PREDICTION
    - OperationType.MULTIHEAD

  Raises:
    ValueError: If any inference_spec_type is None and catch_errors is False.
  """
  class _SplitDoFn(beam.DoFn):
    def __init__(self):
      # key -> (OperationType, Optional[Exception])
      self._cache = {}

    def process(self, batch):
      inference_spec, _ = batch

      if inference_spec is None:
        raise ValueError("InferenceSpecType cannot be None.")

      key = inference_spec.SerializeToString()
      cached = self._cache.get(key)

      if cached is None:
        cached = (None, None)
        if catch_errors:
          try:
            cached = (_get_operation_type(inference_spec), None)
          except Exception as e:
            cached = ('errors', e)
        else:
          cached = (_get_operation_type(inference_spec), None)

        self._cache[key] = cached

      operation_type, exception = cached
      if operation_type == 'errors':
        return [beam.pvalue.TaggedOutput(operation_type, (exception, batch))]
      else:
        return [beam.pvalue.TaggedOutput(operation_type, batch)]

  return (
    batches
    | 'SplitDoFn' >> beam.ParDo(_SplitDoFn()).with_outputs(
        'errors',
        OperationType.CLASSIFICATION,
        OperationType.REGRESSION,
        OperationType.PREDICTION,
        OperationType.MULTIHEAD
    ))


_IOTensorSpec = collections.namedtuple(
    '_IOTensorSpec',
    ['input_tensor_alias', 'input_tensor_name', 'output_alias_tensor_names'])

_Signature = collections.namedtuple('_Signature', ['name', 'signature_def'])


@six.add_metaclass(abc.ABCMeta)
class _BaseDoFn(beam.DoFn):
  """Base DoFn that performs bulk inference."""

  class _MetricsCollector(object):
    """A collector for beam metrics."""

    def __init__(self, operation_type: Text, proximity_descriptor: Text):
      """Initializes a metrics collector.

      Args:
        operation_type: A string describing the type of operation, e.g.
          "CLASSIFICATION".
        proximity_descriptor: A string describing the location of inference,
          e.g. "InProcess".
      """
      namespace = util.MakeTfxNamespace([
        _METRICS_DESCRIPTOR_INFERENCE, operation_type, proximity_descriptor])

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
      self._load_model_latency_milli_secs_cache = None
      self._model_byte_size_cache = None

    def commit_cached_metrics(self):
      """Updates any cached metrics.

      If there are no cached metrics, this has no effect. Cached metrics are
      automatically cleared after use.
      """
      if self._load_model_latency_milli_secs_cache is not None:
        self._load_model_latency_milli_secs.update(
            self._load_model_latency_milli_secs_cache)
        self._load_model_latency_milli_secs_cache = None
      if self._model_byte_size_cache is not None:
        self._model_byte_size.update(self._model_byte_size_cache)
        self._model_byte_size_cache = None

    def update_model_load(
      self, load_model_latency_milli_secs: int, model_byte_size: int):
      """Updates model loading metrics.

      Note: To commit model loading metrics, you must call
      commit_cached_metrics() after storing values with this method.

      Args:
        load_model_latency_milli_secs: Model loading latency in milliseconds.
        model_byte_size: Approximate model size in bytes.
      """
      self._load_model_latency_milli_secs_cache = load_model_latency_milli_secs
      self._model_byte_size_cache = model_byte_size

    def update_inference(
      self, elements: List[ExampleType], latency_micro_secs: int) -> None:
      """Updates inference metrics.

      Args:
        elements: A list of examples used for inference.
        latency_micro_secs: Total inference latency in microseconds.
      """
      self._inference_batch_latency_micro_secs.update(latency_micro_secs)
      self._num_instances.inc(len(elements))
      self._inference_counter.inc(len(elements))
      self._inference_request_batch_size.update(len(elements))
      self._inference_request_batch_byte_size.update(
          sum(element.ByteSize() for element in elements))

  def __init__(self, operation_type: Text, proximity_descriptor: Text):
    super(_BaseDoFn, self).__init__()
    self._clock = None
    self._metrics_collector = self._MetricsCollector(
      operation_type, proximity_descriptor)

  def setup(self):
    self._clock = _ClockFactory.make_clock()

  def process(self, batch: _QueryBatchType) -> Iterable[Any]:
    inference_spec, elements = batch
    batch_start_time = self._clock.get_current_time_in_microseconds()
    outputs = self.run_inference(inference_spec, elements)
    result = self._post_process(elements, outputs)
    self._metrics_collector.update_inference(
        elements,
        self._clock.get_current_time_in_microseconds() - batch_start_time)
    return result

  @abc.abstractmethod
  def run_inference(
      self,
      inference_spec: model_spec_pb2.InferenceSpecType,
      elements: List[ExampleType]
  ) -> Union[Mapping[Text, np.ndarray], Sequence[Mapping[Text, Any]]]:
    raise NotImplementedError

  @abc.abstractmethod
  def _post_process(self, elements: List[ExampleType],
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


@beam.typehints.with_input_types(_QueryBatchType)
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

  def __init__(
      self,
      pipeline_options: PipelineOptions,
      fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None
  ):
    super(_RemotePredictDoFn, self).__init__(
      OperationType.PREDICTION, _METRICS_DESCRIPTOR_CLOUD_AI_PREDICTION)
    self._pipeline_options = pipeline_options
    self._fixed_inference_spec_type = fixed_inference_spec_type

    self._ai_platform_prediction_model_spec = None
    self._api_client = None
    self._full_model_name = None

  def setup(self):
    super(_RemotePredictDoFn, self).setup()
    if self._fixed_inference_spec_type:
      self._setup_model(self._fixed_inference_spec_type)

  def _setup_model(
      self, inference_spec_type: model_spec_pb2.InferenceSpecType
  ):
    self._ai_platform_prediction_model_spec = (
        inference_spec_type.ai_platform_prediction_model_spec)

    project_id = (
        inference_spec_type.ai_platform_prediction_model_spec.project_id or
        self._pipeline_options.view_as(GoogleCloudOptions).project)
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
      self, elements: List[ExampleType]
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
      self, elements: List[ExampleType]
  ) -> Generator[Mapping[Text, Text], None, None]:
    """Prepare instances by base64 encoding serialized examples."""
    for example in elements:
      yield {'b64': base64.b64encode(example.SerializeToString()).decode()}

  def _prepare_instances(
      self, elements: List[ExampleType]
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
      self,
      inference_spec: model_spec_pb2.InferenceSpecType,
      elements: List[ExampleType]
  ) -> Sequence[Mapping[Text, Any]]:
    if not self._fixed_inference_spec_type:
      self._setup_model(inference_spec)
    body = {'instances': list(self._prepare_instances(elements))}
    request = self._make_request(body)
    response = self._execute_request(request)
    return response['predictions']

  def _post_process(
      self, elements: List[ExampleType], outputs: Sequence[Mapping[Text, Any]]
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
      shared_model_handle: shared.Shared,
      fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None,
      operation_type: Text = ''
  ):
    super(_BaseBatchSavedModelDoFn, self).__init__(
      operation_type, _METRICS_DESCRIPTOR_IN_PROCESS)
    self._shared_model_handle = shared_model_handle
    self._fixed_inference_spec_type = fixed_inference_spec_type

    self._model_path = None
    self._tags = None
    self._signatures = None
    self._session = None
    self._io_tensor_spec = None

  def setup(self):
    """Load the model.
    Note that worker may crash if exception is thrown in setup due
    to b/139207285.
    """
    super(_BaseBatchSavedModelDoFn, self).setup()
    if self._fixed_inference_spec_type:
      self._setup_model(self._fixed_inference_spec_type)

  def finish_bundle(self):
    # If we are using a fixed model, _setup_model will be called in DoFn.setup
    # and model loading metrics will be cached. To commit these metrics, we
    # need to call _metrics_collector.commit_cached_metrics() once during the
    # DoFn lifetime. DoFn.teardown() is not guaranteed to be called, so the
    # next best option is to call this in finish_bundle().
    if self._fixed_inference_spec_type:
      self._metrics_collector.commit_cached_metrics()

  def _setup_model(
      self, inference_spec_type: model_spec_pb2.InferenceSpecType
  ):
    self._model_path = inference_spec_type.saved_model_spec.model_path
    self._signatures = _get_signatures(
        inference_spec_type.saved_model_spec.model_path,
        inference_spec_type.saved_model_spec.signature_name,
        _get_tags(inference_spec_type))

    self._validate_model()

    self._tags = _get_tags(inference_spec_type)
    self._io_tensor_spec = self._pre_process()

    if self._has_tpu_tag():
      # TODO(b/131873699): Support TPU inference.
      raise ValueError('TPU inference is not supported yet.')
    self._session = self._load_model(inference_spec_type)

  def _validate_model(self):
    """Optional subclass model validation hook.

    Raises:
      ValueError: if model is invalid.
    """
    pass

  def _load_model(self, inference_spec_type: model_spec_pb2.InferenceSpecType):
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

      # Compute model loading metrics.
      load_model_latency_milli_secs = (
        (end_time - start_time) / _MILLISECOND_TO_MICROSECOND)
      model_byte_size = (memory_after - memory_before)
      self._metrics_collector.update_model_load(
        load_model_latency_milli_secs, model_byte_size)

      return result

    if not self._model_path:
      raise ValueError('Model path is not valid.')
    return self._shared_model_handle.acquire(
      load, tag=inference_spec_type.SerializeToString().decode('latin-1'))

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
      self,
      inference_spec_type: model_spec_pb2.InferenceSpecType,
      elements: List[ExampleType]
  ) -> Mapping[Text, np.ndarray]:
    if not self._fixed_inference_spec_type:
      self._setup_model(inference_spec_type)
      self._metrics_collector.commit_cached_metrics()
    self._check_elements(elements)
    outputs = self._run_tf_operations(elements)
    return outputs

  def _run_tf_operations(
      self, elements: List[ExampleType]
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

  def _check_elements(self, elements: List[ExampleType]) -> None:
    """Unimplemented."""

    raise NotImplementedError


@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        classification_pb2.Classifications])
class _BatchClassifyDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on classification model."""

  def __init__(
    self,
    shared_model_handle: shared.Shared,
    fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None
  ):
    super(_BatchClassifyDoFn, self).__init__(
      shared_model_handle, fixed_inference_spec_type,
      OperationType.CLASSIFICATION)

  def _validate_model(self):
    signature_def = self._signatures[0].signature_def
    if signature_def.method_name != tf.saved_model.CLASSIFY_METHOD_NAME:
      raise ValueError(
          'BulkInferrerClassifyDoFn requires signature method '
          'name %s, got: %s' % tf.saved_model.CLASSIFY_METHOD_NAME,
          signature_def.method_name)

  def _check_elements(
      self, elements: List[ExampleType]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Classify only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[ExampleType], outputs: Mapping[Text, np.ndarray]
  ) -> Iterable[Tuple[tf.train.Example, classification_pb2.Classifications]]:
    classifications = _post_process_classify(
        self._io_tensor_spec.output_alias_tensor_names, elements, outputs)
    return zip(elements, classifications)


@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        regression_pb2.Regression])
class _BatchRegressDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that run inference on regression model."""

  def __init__(
    self,
    shared_model_handle: shared.Shared,
    fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None
  ):
    super(_BatchRegressDoFn, self).__init__(
      shared_model_handle, fixed_inference_spec_type,
      OperationType.REGRESSION)

  def _validate_model(self):
    signature_def = self._signatures[0].signature_def
    if signature_def.method_name != tf.saved_model.REGRESS_METHOD_NAME:
      raise ValueError(
          'BulkInferrerRegressDoFn requires signature method '
          'name %s, got: %s' % tf.saved_model.REGRESS_METHOD_NAME,
          signature_def.method_name)

  def _check_elements(
      self, elements: List[ExampleType]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Regress only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[ExampleType], outputs: Mapping[Text, np.ndarray]
  ) -> Iterable[Tuple[tf.train.Example, regression_pb2.Regression]]:
    regressions = _post_process_regress(elements, outputs)
    return zip(elements, regressions)


@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(prediction_log_pb2.PredictLog)
class _BatchPredictDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on predict model."""

  def __init__(
    self,
    shared_model_handle: shared.Shared,
    fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None
  ):
    super(_BatchPredictDoFn, self).__init__(
      shared_model_handle, fixed_inference_spec_type,
      OperationType.PREDICTION)

  def _validate_model(self):
    signature_def = self._signatures[0].signature_def
    if signature_def.method_name != tf.saved_model.PREDICT_METHOD_NAME:
      raise ValueError(
          'BulkInferrerPredictDoFn requires signature method '
          'name %s, got: %s' % tf.saved_model.PREDICT_METHOD_NAME,
          signature_def.method_name)

  def _check_elements(
      self, elements: List[ExampleType]) -> None:
    pass

  def _post_process(
      self, elements: Sequence[ExampleType], outputs: Mapping[Text, np.ndarray]
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


@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(Tuple[tf.train.Example,
                                        inference_pb2.MultiInferenceResponse])
class _BatchMultiInferenceDoFn(_BaseBatchSavedModelDoFn):
  """A DoFn that runs inference on multi-head model."""

  def __init__(
    self,
    shared_model_handle: shared.Shared,
    fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None
  ):
    super(_BatchMultiInferenceDoFn, self).__init__(
      shared_model_handle, fixed_inference_spec_type,
      OperationType.MULTIHEAD)

  def _check_elements(
      self, elements: List[ExampleType]) -> None:
    if not all(isinstance(element, tf.train.Example) for element in elements):
      raise ValueError('Multi inference only supports tf.train.Example')

  def _post_process(
      self, elements: Sequence[ExampleType], outputs: Mapping[Text, np.ndarray]
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


def _BuildInferenceOperation(
  name: str,
  in_process_dofn: _BaseBatchSavedModelDoFn,
  remote_dofn: Optional[_BaseDoFn],
  build_prediction_log_dofn: beam.DoFn
):
  """Construct an operation specific inference sub-pipeline.

  Args:
    name: Name of the operation (e.g. "Classify").
    in_process_dofn: A _BaseBatchSavedModelDoFn class to use for in-process
      inference.
    remote_dofn: An optional DoFn that is used for remote inference. If not
      provided, attempts at remote inference will throw a NotImplementedError.
    build_prediction_log_dofn: A DoFn that can build prediction logs from the
      output of `in_process_dofn` and `remote_dofn`.

  Returns:
    A PTransform of the type (_QueryBatchType -> PredictionLog).

  Raises:
    NotImplementedError: if remote inference is attempted and not supported.
  """
  @beam.ptransform_fn
  @beam.typehints.with_input_types(_QueryBatchType)
  @beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
  def _Op(
      pcoll: beam.pvalue.PCollection,
      fixed_inference_spec_type: model_spec_pb2.InferenceSpecType = None,
      catch_errors: bool = False
  ): # pylint: disable=invalid-name
    raw_result = None
    errors = None

    if fixed_inference_spec_type is None:
      tagged = pcoll | ('TagInferenceType%s' % name) >> _TagUsingInProcessInference()

      in_process_result = (
        tagged['in_process']
        | ('InProcess%s' % name) >> _ParDoExceptionWrapper(
          in_process_dofn(shared.Shared()),
          catch_errors=catch_errors))

      if remote_dofn:
        remote_result = (
          tagged['remote']
          | ('Remote%s' % name) >> _ParDoExceptionWrapper(
            remote_dofn(pcoll.pipeline.options),
            catch_errors=catch_errors))

        raw_result = (
          [in_process_result[None], remote_result[None]]
          | 'MergePredictions' >> beam.Flatten())

        errors = (
          [in_process_result['errors'], remote_result['errors']]
          | 'MergeErrors' >> beam.Flatten())
      else:
        tagged['remote'] | 'NotImplemented' >> _NotImplementedTransform(
          'Remote inference is not supported for operation type: %s' % name)

        raw_result = in_process_result[None]
        errors = in_process_result['errors']
    else:
      inference_results = None
      if _using_in_process_inference(fixed_inference_spec_type):
        inference_results = (
          pcoll
          | ('InProcess%s' % name) >> _ParDoExceptionWrapper(
            in_process_dofn(
              shared.Shared(),
              fixed_inference_spec_type=fixed_inference_spec_type),
            catch_errors=catch_errors))
      else:
        if remote_dofn:
          inference_results = (
            pcoll
            | ('Remote%s' % name) >> _ParDoExceptionWrapper(
              remote_dofn(
                pcoll.pipeline.options,
                fixed_inference_spec_type=fixed_inference_spec_type),
              catch_errors=catch_errors))
        else:
          raise NotImplementedError('Remote inference is not supported for'
                                    'operation type: %s' % name)
      
      raw_result = inference_results[None]
      errors = inference_results['errors']

    predictions =  (
      raw_result
      | ('BuildPredictionLogFor%s' % name) >> beam.ParDo(
        build_prediction_log_dofn()))

    return {'predictions': predictions, 'errors': errors}

  return _Op


_Classify = _BuildInferenceOperation(
  'Classify', _BatchClassifyDoFn, None,
  _BuildPredictionLogForClassificationsDoFn)
_Regress = _BuildInferenceOperation(
  'Regress', _BatchRegressDoFn, None,
  _BuildPredictionLogForRegressionsDoFn)
_Predict = _BuildInferenceOperation(
  'Predict', _BatchPredictDoFn, _RemotePredictDoFn,
  _BuildPredictionLogForPredictionsDoFn)
_MultiInference = _BuildInferenceOperation(
  'MultiInference', _BatchMultiInferenceDoFn, None,
  _BuildMultiInferenceLogDoFn)


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
    signature: _SignatureDef) -> Tuple[Text, Mapping[Text, Text]]:
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
    signature: _SignatureDef) -> Tuple[Text, Mapping[Text, Text]]:
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
    signature: _SignatureDef) -> Tuple[Text, Mapping[Text, Text]]:
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


@beam.ptransform_fn
@beam.typehints.with_input_types(_QueryBatchType)
@beam.typehints.with_output_types(_QueryBatchType)
def _TagUsingInProcessInference(
  queries: beam.pvalue.PCollection) -> beam.pvalue.DoOutputsTuple:
  """Tags each query batch with 'in_process' or 'remote'."""
  return queries | 'TagBatches' >> beam.Map(
    lambda query: beam.pvalue.TaggedOutput(
      'in_process' if _using_in_process_inference(query[0]) else 'remote', query)
  ).with_outputs('in_process', 'remote')


@beam.ptransform_fn
def _NotImplementedTransform(
  pcoll: beam.pvalue.PCollection, message: Text = ''):
  """Raises NotImplementedError for each value in the input PCollection."""
  def _raise(x):
    raise NotImplementedError(message)
  pcoll | beam.Map(_raise)


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


def _runner_supports_stateful_dofn(
  runner: beam.pipeline.PipelineRunner) -> bool:
  """Returns True if if the provided runner supports stateful DoFn's."""
  # TODO: Implement.
  return True


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


class _TemporalJoinStream(object):
  PRIMARY = 1
  SECONDARY = 2


class _TemporalJoinDoFn(beam.DoFn):
  """A stateful DoFn that joins two PCollection streams.

  CACHE: holds the most recent item from the secondary stream
  EARLY_PRIMARY: holds any items from the primary stream received before the
    first item from the secondary stream
  """
  CACHE = beam.transforms.userstate.CombiningValueStateSpec(
      'cache', combine_fn=beam.combiners.ToListCombineFn())

  EARLY_PRIMARY = beam.transforms.userstate.CombiningValueStateSpec(
      'early_primary', combine_fn=beam.combiners.ToListCombineFn())

  def process(
    self,
    x,
    cache=beam.DoFn.StateParam(CACHE),
    early_primary=beam.DoFn.StateParam(EARLY_PRIMARY)
  ):
    key, tup = x
    value, stream_type = tup

    if stream_type == _TemporalJoinStream.PRIMARY:
      cached = cache.read()
      if len(cached) == 0:
        # accumulate in early_primary
        early_primary.add(value)
      else:
        return [(cached[0], value)]
    elif stream_type == _TemporalJoinStream.SECONDARY:
      cache.clear()
      cache.add(value)

      # dump any cached values from primary
      primary = early_primary.read()
      if len(primary) > 0:
        early_primary.clear()
        return [(value, x) for x in primary]
    else:
      return []


@beam.ptransform_fn
def _TemporalJoin(primary, secondary):
  """Performs a temporal join of two PCollections.

  Returns tuples of the type (b,a) where:
  - a is from the primary stream
  - b is from the secondary stream
  - b is the most recent item at the time a is processed (or a was processed
    before b and b is the first item in the stream)
  """
  tag_primary = primary | 'primary' >> beam.Map(
      lambda x: (x, _TemporalJoinStream.PRIMARY))
  tag_secondary = secondary | 'secondary' >> beam.Map(
      lambda x: (x, _TemporalJoinStream.SECONDARY))

  joined = [tag_primary, tag_secondary] \
            | beam.Flatten() \
            | 'Fake keys' >> beam.Map(lambda x: (0,x)) \
            | 'Join' >> beam.ParDo(_TemporalJoinDoFn())

  return joined


@beam.ptransform_fn
def _ParDoExceptionWrapper(
    pcoll: beam.pvalue.PCollection,
    dofn: beam.DoFn,
    catch_errors: bool = False
):
  """Runs a ParDo operation and optionally catches exceptions.

  Args:
    pcoll: input pcollection.
    dofn: a beam.DoFn that may raise exceptions.
    catch_errors:
      - if True, errors raised in DoFn.process will be caught and emitted
        in a separate 'errors' PCollection.
      - if False, errors from the internal DoFn will not be caught

  Returns: a DoOutputsTuple containing an additional 'errors' stream.
  """
  def _WrapDoFn(dofn, catch_errors=False):
    if catch_errors:
      _native_process = dofn.process
      def _process(x):
        out = None
        try:
          out = _native_process(x)
        except Exception as e:
          out = [beam.pvalue.TaggedOutput('errors', (e, x))]
        return out

      dofn.process = _process

    return dofn

  return (
    pcoll
    | beam.ParDo(_WrapDoFn(dofn, catch_errors)).with_outputs('errors'))
