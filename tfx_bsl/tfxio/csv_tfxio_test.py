# Copyright 2020 Google LLC
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
"""Tests for tfx_bsl.tfxio.csv."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import csv_tfxio
from tfx_bsl.tfxio import telemetry_test_util
from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2

FLAGS = flags.FLAGS

_COLUMN_NAMES = ["int_feature", "float_feature", "string_feature"]

_SCHEMA = text_format.Parse(
    """
  feature {
    name: "int_feature"
    type: INT
    value_count {
      min: 0
      max: 2
    }
  }
  feature {
    name: "float_feature"
    type: FLOAT
    value_count {
      min: 0
      max: 2
    }
  }
  feature {
    name: "string_feature"
    type: BYTES
    value_count {
      min: 0
      max: 2
    }
  }
""", schema_pb2.Schema())

_TELEMETRY_DESCRIPTORS = ["Some", "Component"]

_ROWS = [b'1,2.0,"abc"\n', b'2,3.0,"xyz"\n']
_RAW_RECORDS = [b'1,2.0,"abc"', b'2,3.0,"xyz"']

_SCHEMA_TEST_CASES = [
    dict(
        testcase_name="unordered_schema",
        schema=text_format.Parse(
            """
      feature {
        name: "string_feature"
        type: BYTES
        value_count {
          min: 0
          max: 2
        }
      }
      feature {
        name: "int_feature"
        type: INT
        value_count {
          min: 0
          max: 2
        }
      }
      feature {
        name: "float_feature"
        type: FLOAT
        value_count {
          min: 0
          max: 2
        }
      }
    """, schema_pb2.Schema()))
]


def _GetExpectedArrowSchema(tfxio, raw_record_column_name=None):
  if tfxio._can_produce_large_types:
    int_type = pa.large_list(pa.int64())
    float_type = pa.large_list(pa.float32())
    bytes_type = pa.large_list(pa.large_binary())
  else:
    int_type = pa.list_(pa.int64())
    float_type = pa.list_(pa.float32())
    bytes_type = pa.list_(pa.binary())
  fields = [
      pa.field("int_feature", int_type),
      pa.field("float_feature", float_type),
      pa.field("string_feature", bytes_type)
  ]
  if raw_record_column_name is not None:
    fields.append(pa.field(raw_record_column_name, bytes_type))
  return pa.schema(fields)


def _GetExpectedColumnValues(tfxio):
  if tfxio._can_produce_large_types:
    int_type = pa.large_list(pa.int64())
    float_type = pa.large_list(pa.float32())
    bytes_type = pa.large_list(pa.large_binary())
  else:
    int_type = pa.list_(pa.int64())
    float_type = pa.list_(pa.float32())
    bytes_type = pa.list_(pa.binary())

  return {
      "int_feature": pa.array([[1], [2]], type=int_type),
      "float_feature": pa.array([[2.0], [3.0]], type=float_type),
      "string_feature": pa.array([[b"abc"], [b"xyz"]], type=bytes_type),
  }


def _WriteInputs(filename):
  with tf.io.gfile.GFile(filename, "w") as out_file:
    for row in _ROWS:
      out_file.write(row)


class CsvRecordTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super(CsvRecordTest, cls).setUpClass()
    cls._example_file = os.path.join(FLAGS.test_tmpdir, "csvtexttest",
                                     "input.csv")
    tf.io.gfile.makedirs(os.path.dirname(cls._example_file))
    _WriteInputs(cls._example_file)

  def _MakeTFXIO(self, column_names, schema=None, raw_record_column_name=None):
    return csv_tfxio.CsvTFXIO(
        self._example_file,
        column_names=column_names,
        schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

  def _ValidateRecordBatch(self,
                           tfxio,
                           record_batch,
                           raw_record_column_name=None):
    self.assertIsInstance(record_batch, pa.RecordBatch)
    self.assertEqual(record_batch.num_rows, 2)
    expected_column_values = _GetExpectedColumnValues(tfxio)
    expected_arrow_schema = _GetExpectedArrowSchema(tfxio,
                                                    raw_record_column_name)
    self.assertTrue(
        record_batch.schema.equals(expected_arrow_schema),
        "Expected: {} ; got {}".format(expected_arrow_schema,
                                       record_batch.schema))
    for i, field in enumerate(record_batch.schema):
      if field.name == raw_record_column_name:
        continue
      self.assertTrue(
          record_batch.column(i).equals(expected_column_values[field.name]),
          "Column {} did not match ({} vs {}).".format(
              field.name, record_batch.column(i),
              expected_column_values[field.name]))

    if raw_record_column_name is not None:
      self.assertEqual(record_batch.schema.names[-1], raw_record_column_name)
      self.assertEqual(record_batch.columns[-1].flatten().to_pylist(),
                       _RAW_RECORDS)

  def testImplicitTensorRepresentations(self):
    """Tests inferring of tensor representation."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=_SCHEMA)
    self.assertEqual(
        {
            "int_feature":
                text_format.Parse(
                    """varlen_sparse_tensor { column_name: "int_feature"}""",
                    schema_pb2.TensorRepresentation()),
            "float_feature":
                text_format.Parse(
                    """varlen_sparse_tensor { column_name: "float_feature"}""",
                    schema_pb2.TensorRepresentation()),
            "string_feature":
                text_format.Parse(
                    """varlen_sparse_tensor { column_name: "string_feature" }""",
                    schema_pb2.TensorRepresentation()),
        }, tfxio.TensorRepresentations())

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch)
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 3)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("string_feature", dict_of_tensors)

    p = beam.Pipeline()
    record_batch_pcoll = p | tfxio.BeamSource(batch_size=1000)
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, "csv", "text")

  def testExplicitTensorRepresentations(self):
    """Tests when the tensor representation is explicitely provided in the schema."""
    schema = schema_pb2.Schema()
    schema.CopyFrom(_SCHEMA)
    tensor_representations = {
        "my_feature":
            text_format.Parse(
                """
            dense_tensor {
             column_name: "string_feature"
             shape { dim { size: 1 } }
             default_value { bytes_value: "abc" }
           }""", schema_pb2.TensorRepresentation())
    }
    schema.tensor_representation_group[""].CopyFrom(
        schema_pb2.TensorRepresentationGroup(
            tensor_representation=tensor_representations))

    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=schema)
    self.assertEqual(tensor_representations, tfxio.TensorRepresentations())

  def testProjection(self):
    """Test projecting of a TFXIO."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=_SCHEMA)

    projected_tfxio = tfxio.Project(["int_feature"])

    # The projected_tfxio still has original schema
    self.assertTrue(projected_tfxio.ArrowSchema().equals(
        _GetExpectedArrowSchema(tfxio)))

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch)
      expected_schema = projected_tfxio.ArrowSchema()
      self.assertTrue(
          record_batch.schema.equals(expected_schema),
          "actual: {}; expected: {}".format(record_batch.schema,
                                            expected_schema))
      tensor_adapter = projected_tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 1)
      self.assertIn("int_feature", dict_of_tensors)

    with beam.Pipeline() as p:
      # Setting the batch_size to make sure only one batch is generated.
      record_batch_pcoll = p | projected_tfxio.BeamSource(batch_size=len(_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testAttachRawRecordColumn(self):
    raw_record_column_name = "raw_records"
    tfxio = self._MakeTFXIO(
        _COLUMN_NAMES,
        schema=_SCHEMA,
        raw_record_column_name=raw_record_column_name)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch, raw_record_column_name)

    with beam.Pipeline() as p:
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testOptionalSchema(self):
    """Tests when the schema is not provided."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES)
    with self.assertRaisesRegexp(ValueError, ".*TFMD schema not provided.*"):
      tfxio.ArrowSchema()

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch)

    with beam.Pipeline() as p:
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  @parameterized.named_parameters(_SCHEMA_TEST_CASES)
  def testSchema(self, schema):
    """Tests various valid schemas."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=schema)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch)

    with beam.Pipeline() as p:
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)


if __name__ == "__main__":
  absltest.main()
