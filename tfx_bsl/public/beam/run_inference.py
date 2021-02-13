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
# Lint as: python3
"""Publich API of batch inference."""

from typing import Tuple, TypeVar, Union

import apache_beam as beam
import tensorflow as tf
from tfx_bsl.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2

from tensorflow_serving.apis import prediction_log_pb2


_K = TypeVar('_K')
_INPUT_TYPE = Union[tf.train.Example, tf.train.SequenceExample, bytes]
_OUTPUT_TYPE = prediction_log_pb2.PredictionLog


@beam.ptransform_fn
@beam.typehints.with_input_types(Union[_INPUT_TYPE, Tuple[_K, _INPUT_TYPE]])
@beam.typehints.with_output_types(Union[_OUTPUT_TYPE, Tuple[_K, _OUTPUT_TYPE]])
def RunInference(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType
) -> beam.pvalue.PCollection:
  """Run inference with a model.

  There are two types of inference you can perform using this PTransform:
    1. In-process inference from a SavedModel instance. Used when
      `saved_model_spec` field is set in `inference_spec_type`.
    2. Remote inference by using a service endpoint. Used when
      `ai_platform_prediction_model_spec` field is set in
      `inference_spec_type`.

  TODO(b/131873699): Add support for the following features:
    1. tf.train.SequenceExample as Input for RemotePredict.
    2. beam.Shared() initialization via Fingerprint for models CSE.
    3. Models as SideInput.
    4. TPU models.

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

  Returns:
    A PCollection (possibly keyed) containing prediction logs.
  """
  return (
      examples |
      'RunInferenceImpl' >> run_inference.RunInferenceImpl(inference_spec_type))
