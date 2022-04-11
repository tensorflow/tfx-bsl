# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx_bsl.telemetry.collection."""

import apache_beam as beam
import pyarrow as pa
from tfx_bsl.beam import test_helpers
from tfx_bsl.telemetry import collection

from absl.testing import absltest
from tensorflow_metadata.proto.v0 import schema_pb2


class CollectionTest(absltest.TestCase):

  def testTrackRecordBatchBytes(self):
    inputs = pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int32())],
                                        ["f1"])
    expected_num_bytes = inputs.nbytes

    with beam.Pipeline(**test_helpers.make_test_beam_pipeline_kwargs()) as p:
      _ = (
          p | beam.Create([inputs])
          | collection.TrackRecordBatchBytes("TestNamespace",
                                             "num_bytes_count"))

    pipeline_result = p.run()
    result_metrics = pipeline_result.metrics()
    actual_counter = result_metrics.query(
        beam.metrics.metric.MetricsFilter().with_name(
            "num_bytes_count"))["counters"]
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, expected_num_bytes)

  def testTrackRecordTensorRepresentations(self):
    num_dense_tensors = 3
    num_varlen_sparse_tensors = 2
    num_sparse_tensors = 1
    num_ragged_tensors = 4
    tensor_representations = {}
    for i in range(num_dense_tensors):
      tensor_representations[f"dense{i}"] = (
          schema_pb2.TensorRepresentation(
              dense_tensor=schema_pb2.TensorRepresentation.DenseTensor()))
    for i in range(num_varlen_sparse_tensors):
      tensor_representations[f"varlen{i}"] = (
          schema_pb2.TensorRepresentation(
              varlen_sparse_tensor=schema_pb2.TensorRepresentation
              .VarLenSparseTensor()))
    for i in range(num_sparse_tensors):
      tensor_representations[f"sparse{i}"] = (
          schema_pb2.TensorRepresentation(
              sparse_tensor=schema_pb2.TensorRepresentation.SparseTensor()))
    for i in range(num_ragged_tensors):
      tensor_representations[f"ragged{i}"] = (
          schema_pb2.TensorRepresentation(
              ragged_tensor=schema_pb2.TensorRepresentation.RaggedTensor()))

    expected_counters = {
        "dense_tensor": num_dense_tensors,
        "varlen_sparse_tensor": num_varlen_sparse_tensors,
        "sparse_tensor": num_sparse_tensors,
        "ragged_tensor": num_ragged_tensors,
    }

    with beam.Pipeline(**test_helpers.make_test_beam_pipeline_kwargs()) as p:
      _ = (
          p | beam.Create([tensor_representations])
          | collection.TrackTensorRepresentations(
              counter_namespace="TestNamespace"))

    pipeline_result = p.run()
    result_metrics = pipeline_result.metrics()
    for kind, expected_count in expected_counters.items():
      actual_counter = result_metrics.query(
          beam.metrics.metric.MetricsFilter().with_name(kind))["counters"]
      self.assertLen(
          actual_counter,
          1,
          msg=f"Actual and expected lengths of {kind} counter are different.")
      self.assertEqual(
          actual_counter[0].committed,
          expected_count,
          msg=f"Actual and expected values for {kind} counter are different.")


if __name__ == "__main__":
  absltest.main()
