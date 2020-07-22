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
import pyarrow as pa
from typing import Union, Text, Optional
from tfx_bsl.tfxio import test_util
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tf_sequence_example_record
from tfx_bsl.beam import run_inference
from tfx_bsl.beam import run_inference_arrow
from tfx_bsl.public.proto import model_spec_pb2
from tensorflow_serving.apis import prediction_log_pb2
from tensorflow_metadata.proto.v0 import schema_pb2

from tfx_bsl.beam.bsl_constants import _RECORDBATCH_COLUMN
from tfx_bsl.beam.bsl_constants import DataType


@beam.ptransform_fn
@beam.typehints.with_input_types(tf.train.Example)
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

  Args:
    examples: A PCollection containing examples.
    inference_spec_type: Model inference endpoint.
    Schema [optional]: required for models that requires
      multi-tensor inputs.

  Returns:
    A PCollection containing prediction logs.
  """

  data_type = DataType.EXAMPLE
  converter = tf_example_record.TFExampleBeamRecord(
    physical_format="inmem",
    telemetry_descriptors=[],
    schema=schema,
    raw_record_column_name=_RECORDBATCH_COLUMN)

  if schema:
    tfxio = test_util.InMemoryTFExampleRecord(
      schema=schema, raw_record_column_name=_RECORDBATCH_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
      arrow_schema=tfxio.ArrowSchema(),
      tensor_representations=tfxio.TensorRepresentations())

  return (examples
          | 'ParseExamples' >> beam.Map(tf.train.Example.SerializeToString)
          | 'ConvertToRecordBatch' >> converter.BeamSource()
          | 'RunInferenceImpl' >> run_inference_arrow.RunInferenceImpl(
                  inference_spec_type, data_type,
                  tensor_adapter_config=tensor_adapter_config))


@beam.ptransform_fn
@beam.typehints.with_input_types(tf.train.SequenceExample)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def RunInferenceOnSequenceExamples(  # pylint: disable=invalid-name
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

  Args:
    examples: A PCollection containing sequence examples.
    inference_spec_type: Model inference endpoint.
    Schema [optional]: required for models that requires
      multi-tensor inputs.

  Returns:
    A PCollection containing prediction logs.
  """

  data_type = DataType.SEQUENCEEXAMPLE
  converter = tf_sequence_example_record.TFSequenceExampleBeamRecord(
    physical_format="inmem",
    telemetry_descriptors=[],
    schema=schema,
    raw_record_column_name=_RECORDBATCH_COLUMN)

  if schema:
    tfxio = test_util.InMemoryTFExampleRecord(
      schema=schema, raw_record_column_name=_RECORDBATCH_COLUMN)
    tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
      arrow_schema=tfxio.ArrowSchema(),
      tensor_representations=tfxio.TensorRepresentations())

  return (examples
          | 'ParseExamples' >> beam.Map(tf.train.Example.SerializeToString)
          | 'ConvertToRecordBatch' >> converter.BeamSource()
          | 'RunInferenceImpl' >> run_inference_arrow.RunInferenceImpl(
                  inference_spec_type, data_type,
                  tensor_adapter_config=tensor_adapter_config))


@beam.ptransform_fn
@beam.typehints.with_input_types(pa.RecordBatch)
@beam.typehints.with_output_types(prediction_log_pb2.PredictionLog)
def RunInferenceOnRecordBatch(  # pylint: disable=invalid-name
    examples: beam.pvalue.PCollection,
    inference_spec_type: model_spec_pb2.InferenceSpecType,
    tensor_adapter_config: Optional[tensor_adapter.TensorAdapterConfig] = None
) -> beam.pvalue.PCollection:
  """Run inference with a model.

   There are two types of inference you can perform using this PTransform:
   1. In-process inference from a SavedModel instance. Used when
     `saved_model_spec` field is set in `inference_spec_type`.
   2. Remote inference by using a service endpoint. Used when
     `ai_platform_prediction_model_spec` field is set in
     `inference_spec_type`.

  Args:
    examples: A PCollection containing RecordBatch.
    inference_spec_type: Model inference endpoint.

  Returns:
    A PCollection containing prediction logs.
  """

  return (
      examples | 'RunInferenceImpl' >> run_inference_arrow.RunInferenceImpl(
                        inference_spec_type, tensor_adapter_config))
