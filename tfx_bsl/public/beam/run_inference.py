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
"""Public API of batch inference."""

from typing import Iterable, List, Optional, Tuple, TypeVar, Union

import apache_beam as beam
from apache_beam.ml.inference.base import ModelHandler
from tfx_bsl.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2

from tensorflow_serving.apis import prediction_log_pb2


_K = TypeVar('_K')
_KeyedBatchesInput = Tuple[_K, List[run_inference.InputType]]
_MaybeKeyedInput = Union[run_inference.InputType,
                         Tuple[_K, run_inference.InputType]]
_OUTPUT_TYPE = prediction_log_pb2.PredictionLog


# TODO(b/131873699): Add support for the following features:
#   - tf.train.SequenceExample as Input for RemotePredict.
#   - beam.Shared() initialization via Fingerprint for models CSE.
#   - Models as SideInput.
#   - Separate output PCollection for inference errors.
#   - TPU models.
#   - Investigate and add support for 'LocalServer' functionality, to enable
#     executing inference server with RPC interface to be launched alongside
#     each Beam worker (b/195136883).
@beam.ptransform_fn
@beam.typehints.with_input_types(_MaybeKeyedInput)
@beam.typehints.with_output_types(Union[_OUTPUT_TYPE, Tuple[_K, _OUTPUT_TYPE]])
def RunInference(
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    load_override_fn: Optional[run_inference.LoadOverrideFnType] = None
) -> beam.pvalue.PCollection:
  """Run inference with a model.

  There are two types of inference you can perform using this PTransform:
    1. In-process inference from a SavedModel instance. Used when
      `saved_model_spec` field is set in `inference_spec_type`.
    2. Remote inference by using a service endpoint. Used when
      `ai_platform_prediction_model_spec` field is set in
      `inference_spec_type`.

  Args:
    examples: A PCollection containing examples of the following possible kinds,
      each with their corresponding return type.
        - PCollection[Example]                   -> PCollection[PredictionLog]
            * Works with Classify, Regress, MultiInference, Predict and
              RemotePredict.

        - PCollection[SequenceExample]           -> PCollection[PredictionLog]
            * Works with Predict and (serialized) RemotePredict.

        - PCollection[bytes]                     -> PCollection[PredictionLog]
            * For serialized Example: Works with Classify, Regress,
              MultiInference, Predict and RemotePredict.
            * For everything else: Works with Predict and RemotePredict.

        - PCollection[Tuple[K, Example]]         -> PCollection[
                                                        Tuple[K, PredictionLog]]
            * Works with Classify, Regress, MultiInference, Predict and
              RemotePredict.

        - PCollection[Tuple[K, SequenceExample]] -> PCollection[
                                                        Tuple[K, PredictionLog]]
            * Works with Predict and (serialized) RemotePredict.

        - PCollection[Tuple[K, bytes]]           -> PCollection[
                                                        Tuple[K, PredictionLog]]
            * For serialized Example: Works with Classify, Regress,
              MultiInference, Predict and RemotePredict.
            * For everything else: Works with Predict and RemotePredict.

    inference_spec_type: Model inference endpoint.
    load_override_fn: Optional function taking a model path and sequence of
      tags, and returning a tf SavedModel. The loaded model must be equivalent
      in interface to the model that would otherwise be loaded. It is up to the
      caller to ensure compatibility. This argument is experimental and subject
      to change.

  Returns:
    A PCollection (possibly keyed) containing prediction logs.
  """
  return (examples
          | 'RunInferenceImpl' >> run_inference.RunInferenceImpl(
              inference_spec_type, load_override_fn))


@beam.ptransform_fn
@beam.typehints.with_input_types(_MaybeKeyedInput)
@beam.typehints.with_output_types(Union[Tuple[_OUTPUT_TYPE, ...],
                                        Tuple[_K, Tuple[_OUTPUT_TYPE, ...]]])
def RunInferencePerModel(
    examples: beam.pvalue.PCollection,
    inference_spec_types: Iterable[model_spec_pb2.InferenceSpecType],
    load_override_fn: Optional[run_inference.LoadOverrideFnType] = None
) -> beam.pvalue.PCollection:
  """Vectorized variant of RunInference (useful for ensembles).

  Args:
    examples: A PCollection containing examples of the following possible kinds,
      each with their corresponding return type.
        - PCollection[Example]                  -> PCollection[
                                                     Tuple[PredictionLog, ...]]
            * Works with Classify, Regress, MultiInference, Predict and
              RemotePredict.

        - PCollection[SequenceExample]          -> PCollection[
                                                     Tuple[PredictionLog, ...]]
            * Works with Predict and (serialized) RemotePredict.

        - PCollection[bytes]                    -> PCollection[
                                                     Tuple[PredictionLog, ...]]
            * For serialized Example: Works with Classify, Regress,
              MultiInference, Predict and RemotePredict.
            * For everything else: Works with Predict and RemotePredict.

        - PCollection[Tuple[K, Example]]        -> PCollection[
                                                     Tuple[K,
                                                           Tuple[PredictionLog,
                                                                 ...]]]
            * Works with Classify, Regress, MultiInference, Predict and
              RemotePredict.

        - PCollection[Tuple[K, SequenceExample]] -> PCollection[
                                                     Tuple[K,
                                                           Tuple[PredictionLog,
                                                                 ...]]]
            * Works with Predict and (serialized) RemotePredict.

        - PCollection[Tuple[K, bytes]]           -> PCollection[
                                                     Tuple[K,
                                                           Tuple[PredictionLog,
                                                                 ...]]]
            * For serialized Example: Works with Classify, Regress,
              MultiInference, Predict and RemotePredict.
            * For everything else: Works with Predict and RemotePredict.

    inference_spec_types: A flat iterable of Model inference endpoints.
      Inference will happen in a fused fashion (ie without data
      materialization), sequentially across Models within a Beam thread (but
      in parallel across threads and workers).
    load_override_fn: Optional function taking a model path and sequence of
      tags, and returning a tf SavedModel. The loaded model must be equivalent
      in interface to the model that would otherwise be loaded. It is up to the
      caller to ensure compatibility. This argument is experimental and subject
      to change.

  Returns:
    A PCollection (possibly keyed) containing a Tuple of prediction logs. The
    Tuple of prediction logs is 1-1 aligned with inference_spec_types.
  """
  return (examples
          |
          'RunInferencePerModelImpl' >> run_inference.RunInferencePerModelImpl(
              inference_spec_types, load_override_fn))


@beam.ptransform_fn
@beam.typehints.with_input_types(_KeyedBatchesInput)
@beam.typehints.with_output_types(Tuple[_K, List[_OUTPUT_TYPE]])
def RunInferenceOnKeyedBatches(
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    load_override_fn: Optional[run_inference.LoadOverrideFnType] = None
) -> beam.pvalue.PCollection:
  """Run inference over pre-batched keyed inputs.

  This API is experimental and may change in the future.

  Supports the same inference specs as RunInference. Inputs must consist of a
  keyed list of examples, and outputs consist of keyed list of prediction logs
  corresponding by index.

  Args:
    examples: A PCollection of keyed, batched inputs of type Example,
      SequenceExample, or bytes. Each type support inference specs corresponding
      to the unbatched cases described in RunInference. Supports
        - PCollection[Tuple[K, List[Example]]]
        - PCollection[Tuple[K, List[SequenceExample]]]
        - PCollection[Tuple[K, List[Bytes]]]
    inference_spec_type: Model inference endpoint.
    load_override_fn: Optional function taking a model path and sequence of
      tags, and returning a tf SavedModel. The loaded model must be equivalent
      in interface to the model that would otherwise be loaded. It is up to the
      caller to ensure compatibility. This argument is experimental and subject
      to change.

  Returns:
    A PCollection of Tuple[K, List[PredictionLog]].
  """
  return (examples
          | 'RunInferenceOnKeyedBatchesImpl' >> run_inference.RunInferenceImpl(
              inference_spec_type, load_override_fn))


@beam.ptransform_fn
@beam.typehints.with_input_types(_KeyedBatchesInput)
@beam.typehints.with_output_types(Tuple[_K, Tuple[List[_OUTPUT_TYPE]]])
def RunInferencePerModelOnKeyedBatches(
    examples: beam.pvalue.PCollection,
    inference_spec_types: Iterable[model_spec_pb2.InferenceSpecType],
    load_override_fn: Optional[run_inference.LoadOverrideFnType] = None
) -> beam.pvalue.PCollection:
  """Run inference over pre-batched keyed inputs on multiple models.

  This API is experimental and may change in the future.

  Supports the same inference specs as RunInferencePerModel. Inputs must consist
  of a keyed list of examples, and outputs consist of keyed list of prediction
  logs corresponding by index.

  Args:
    examples: A PCollection of keyed, batched inputs of type Example,
      SequenceExample, or bytes. Each type support inference specs corresponding
      to the unbatched cases described in RunInferencePerModel. Supports -
      PCollection[Tuple[K, List[Example]]] - PCollection[Tuple[K,
      List[SequenceExample]]] - PCollection[Tuple[K, List[Bytes]]]
    inference_spec_types: A flat iterable of Model inference endpoints.
      Inference will happen in a fused fashion (ie without data
      materialization), sequentially across Models within a Beam thread (but in
      parallel across threads and workers).
    load_override_fn: Optional function taking a model path and sequence of
      tags, and returning a tf SavedModel. The loaded model must be equivalent
      in interface to the model that would otherwise be loaded. It is up to the
      caller to ensure compatibility. This argument is experimental and subject
      to change.

  Returns:
    A PCollection containing Tuples of a key and lists of batched prediction
    logs from each model provided in inference_spec_types. The Tuple of batched
    prediction logs is 1-1 aligned with inference_spec_types. The individual
    prediction logs in the batch are 1-1 aligned with the rows of data in the
    batch key.
  """
  return (examples
          | 'RunInferencePerModelOnKeyedBatchesImpl' >>
          run_inference.RunInferencePerModelImpl(inference_spec_types,
                                                 load_override_fn))


def CreateModelHandler(
    inference_spec_type: model_spec_pb2.InferenceSpecType) -> ModelHandler:
  """Creates a Beam ModelHandler based on the inference spec type.

  There are two model handlers:
    1. In-process inference from a SavedModel instance. Used when
      `saved_model_spec` field is set in `inference_spec_type`.
    2. Remote inference by using a service endpoint. Used when
      `ai_platform_prediction_model_spec` field is set in
      `inference_spec_type`.

  Example Usage:

    ```
    from apache_beam.ml.inference import base

    tf_handler = CreateModelHandler(inference_spec_type)
    # unkeyed
    base.RunInference(tf_handler)

    # keyed
    base.RunInference(base.KeyedModelHandler(tf_handler))
    ```

  Args:
    inference_spec_type: Model inference endpoint.

  Returns:
    A Beam RunInference ModelHandler for TensorFlow
  """
  return run_inference.create_model_handler(inference_spec_type, None, None)
