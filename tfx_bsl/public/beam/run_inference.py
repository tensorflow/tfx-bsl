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
"""Public API of batch inference."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
import tensorflow as tf
import pyarrow as pa
from typing import Text, Optional, TypeVar
from tfx_bsl.beam import run_inference
from tfx_bsl.public.proto import model_spec_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_metadata.proto.v0 import schema_pb2

MixedExample = TypeVar('MixedExample', tf.train.Example, tf.train.SequenceExample)

@beam.ptransform_fn
@beam.typehints.with_input_types(MixedExample)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def RunInference(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    schema: Optional[schema_pb2.Schema] = None
) -> beam.pvalue.PCollection:
  """Run inference with a model.

   There are two types of inference you can perform using this PTransform:
   1. In-process inference from a SavedModel instance. Used when
     `saved_model_spec` field is set in `inference_spec_type`.
   2. Remote inference by using a service endpoint. Used when
     `ai_platform_prediction_model_spec` field is set in
     `inference_spec_type`.

   TODO(b/131873699): Add support for the following features:
   1. Bytes as Input.
   2. PTable Input.
   3. Models as SideInput.

  Args:
    examples: A PCollection containing examples.
    inference_spec_type: Model inference endpoint.
    schema [optional]: required for predict models that requires
      multi-tensor inputs.

  Returns:
    A PCollection containing prediction logs.
  """

  return (examples
          | 'RunInferenceOnExamples' >> run_inference.RunInferenceOnExamples(
                  inference_spec_type, schema=schema))
