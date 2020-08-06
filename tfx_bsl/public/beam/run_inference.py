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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
import tensorflow as tf
from tfx_bsl.beam import run_inference
from tfx_bsl.beam.run_inference import ExampleType
from tfx_bsl.beam.run_inference import QueryType
from tfx_bsl.public.proto import model_spec_pb2
from typing import Union
from tensorflow_serving.apis import prediction_log_pb2


@beam.typehints.with_input_types(Union[ExampleType, QueryType])
# TODO(BEAM-10258): add type output annotations for polymorphic with_errors API
class RunInference(run_inference.RunInferenceImpl):
  """Run inference with a model.

   There are two types of inference you can perform using this PTransform:
   1. In-process inference from a SavedModel instance. Used when
     `saved_model_spec` field is set in `inference_spec_type`.
   2. Remote inference by using a service endpoint. Used when
     `ai_platform_prediction_model_spec` field is set in
     `inference_spec_type`.

  The inference model can be specified in three ways:
  1. For a fixed inference model, provide an InferenceSpecType object as a
     fixed parameter. This inference model will be used for all examples.
     This form of the API has some performance benefits when running inference
     locally as it is easier to cache the inference model.
  2. In a pipeline where the inference model may be updated at runtime, you can
     specify a PCollection of InferenceSpecType objects as a side-input. Each
     example will be joined with the most recent inference spec from this
     PCollection (based on processing time). Any examples that arrive before
     the first model will be buffered until a model is available.
  3. For finer control, you can run inference on a stream of query tuples:
     (InferenceSpecType, Example) where each tuple specifies an example and a
     model to use for inference. Internally, queries with the same inference
     spec will be grouped together and inference will operate on batches of
     examples. To use this api, don't provide an inference_spec_type parameter.

  By default, exceptions encountered at runtime will be raised. To disable this
  behavior, you can use `RunInference(...).with_errors()` to catch runtime
  errors and emit them in a separate stream. After enabling this, the return
  type of this PTransform becomes a dict containing a prediction PCollection
  and an error PCollection:
  {
    'predictions': ...,
    'errors': ...
  }

  TODO(b/131873699): Add support for the following features:
  1. Bytes as Input.
  2. PTable Input.

  Args:
    examples: A PCollection containing examples. If inference_spec_type is
      None, this is interpreted as a PCollection of queries:
      (InferenceSpecType, Example)
    inference_spec_type: Model inference endpoint. Can be one of:
      - InferenceSpecType: specifies a fixed model to use for inference.
      - PCollection[InferenceSpecType]: specifies a secondary PCollection of
        models. Each example will use the most recent model for inference.
      - None: indicates that the primary PCollection contains
        (InferenceSpecType, Example) tuples.

  Returns:
    A PCollection containing prediction logs.
    Or, if with_errors() is enabled, a dict containing predictions and errors:
    {'predictions': ..., 'errors': ...}
  """
  pass
