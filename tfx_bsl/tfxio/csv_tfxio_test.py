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

_UNORDERED_SCHEMA = text_format.Parse(
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
""", schema_pb2.Schema())

_TELEMETRY_DESCRIPTORS = ["Some", "Component"]

_ROWS = [b'1,2.0,"abc"\n', b'2,3.0,"xyz"\n']
_RAW_RECORDS = [b'1,2.0,"abc"', b'2,3.0,"xyz"']
_EXPECTED_PHYSICAL_FORMAT = "text"

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

_EXPECTED_ARROW_SCHEMA = pa.schema([
    pa.field("int_feature", pa.large_list(pa.int64())),
    pa.field("float_feature", pa.large_list(pa.float32())),
    pa.field("string_feature", pa.large_list(pa.large_binary()))
])

_EXPECTED_COLUMN_VALUES = {
    "int_feature":
        pa.array([[1], [2]], type=pa.large_list(pa.int64())),
    "float_feature":
        pa.array([[2.0], [3.0]], type=pa.large_list(pa.float32())),
    "string_feature":
        pa.array([[b"abc"], [b"xyz"]], type=pa.large_list(pa.large_binary())),
}


def _WriteInputs(filename, include_header_line=False):
  with tf.io.gfile.GFile(filename, "w") as out_file:
    if include_header_line:
      out_file.write("HEADER_LINE\n")
    for row in _ROWS:
      out_file.write(row)


_CSV_TFXIO_IMPL_TEST_CASES = [
    dict(testcase_name="csv_tfxio", use_beam_record_csv_tfxio=False),
    dict(testcase_name="beam_record_csv_tfxio", use_beam_record_csv_tfxio=True),
]


class CsvRecordTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._example_file = os.path.join(FLAGS.test_tmpdir, "csvtexttest",
                                     "input.csv")
    tf.io.gfile.makedirs(os.path.dirname(cls._example_file))
    _WriteInputs(cls._example_file)
    cls._example_file_with_header_line = os.path.join(
        FLAGS.test_tmpdir, "csvtexttest", "input_with_header_line.csv")
    _WriteInputs(cls._example_file_with_header_line,
                 include_header_line=True)

  def _MakeTFXIO(self, column_names, schema=None, raw_record_column_name=None,
                 skip_header_lines=0, use_input_with_header_line=False,
                 make_beam_record_tfxio=False):
    assert not make_beam_record_tfxio or not use_input_with_header_line, (
        "Invalid _MakeTFXIO parameter combination")
    if make_beam_record_tfxio:
      return csv_tfxio.BeamRecordCsvTFXIO(
          physical_format=_EXPECTED_PHYSICAL_FORMAT,
          column_names=column_names,
          schema=schema,
          raw_record_column_name=raw_record_column_name,
          telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

    input_file = (
        self._example_file_with_header_line
        if use_input_with_header_line else self._example_file)
    return csv_tfxio.CsvTFXIO(
        file_pattern=input_file,
        column_names=column_names,
        schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
        skip_header_lines=skip_header_lines)

  def _MakePipelineInputs(self, pipeline, use_beam_record_csv_tfxio=False):
    if use_beam_record_csv_tfxio:
      return pipeline | "CreateCSVLines" >> beam.Create(
          _RAW_RECORDS, reshuffle=False)
    return pipeline

  def _ValidateRecordBatch(self,
                           record_batch,
                           raw_record_column_name=None):
    self.assertIsInstance(record_batch, pa.RecordBatch)
    self.assertEqual(record_batch.num_rows, 2)
    expected_schema = _EXPECTED_ARROW_SCHEMA
    if raw_record_column_name is not None:
      expected_schema = pa.schema(
          list(expected_schema) +
          [pa.field(raw_record_column_name, pa.large_list(pa.large_binary()))])
    self.assertTrue(
        record_batch.schema.equals(expected_schema),
        "Expected: {} ; got {}".format(expected_schema,
                                       record_batch.schema))
    for i, field in enumerate(record_batch.schema):
      if field.name == raw_record_column_name:
        continue
      self.assertTrue(
          record_batch.column(i).equals(_EXPECTED_COLUMN_VALUES[field.name]),
          "Column {} did not match ({} vs {}).".format(
              field.name, record_batch.column(i),
              _EXPECTED_COLUMN_VALUES[field.name]))

    if raw_record_column_name is not None:
      self.assertEqual(record_batch.schema.names[-1], raw_record_column_name)
      self.assertEqual(record_batch.columns[-1].flatten().to_pylist(),
                       _RAW_RECORDS)

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testImplicitTensorRepresentations(self, use_beam_record_csv_tfxio):
    """Tests inferring of tensor representation."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=_SCHEMA,
                            make_beam_record_tfxio=use_beam_record_csv_tfxio)
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
      self._ValidateRecordBatch(record_batch)
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 3)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("string_feature", dict_of_tensors)

    p = beam.Pipeline()
    record_batch_pcoll = (
        self._MakePipelineInputs(p, use_beam_record_csv_tfxio)
        | tfxio.BeamSource(batch_size=len(_ROWS)))
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, "csv",
                                        _EXPECTED_PHYSICAL_FORMAT)

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testExplicitTensorRepresentations(self, use_beam_record_csv_tfxio):
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

    tfxio = self._MakeTFXIO(
        _COLUMN_NAMES, schema=schema,
        make_beam_record_tfxio=use_beam_record_csv_tfxio)

    expected_tensor_representations = {
        "int_feature":
            text_format.Parse(
                """varlen_sparse_tensor { column_name: "int_feature"}""",
                schema_pb2.TensorRepresentation()),
        "float_feature":
            text_format.Parse(
                """varlen_sparse_tensor { column_name: "float_feature"}""",
                schema_pb2.TensorRepresentation()),
    }
    expected_tensor_representations.update(tensor_representations)

    self.assertEqual(expected_tensor_representations,
                     tfxio.TensorRepresentations())

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testProjection(self, use_beam_record_csv_tfxio):
    """Test projecting of a TFXIO."""
    tfxio = self._MakeTFXIO(
        _COLUMN_NAMES, schema=_SCHEMA,
        make_beam_record_tfxio=use_beam_record_csv_tfxio)

    projected_tfxio = tfxio.Project(["int_feature"])

    # The projected_tfxio still has original schema
    self.assertTrue(projected_tfxio.ArrowSchema().equals(
        _EXPECTED_ARROW_SCHEMA))

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch)
      expected_schema = projected_tfxio.ArrowSchema()
      self.assertTrue(
          record_batch.schema.equals(expected_schema),
          "actual: {}; expected: {}".format(record_batch.schema,
                                            expected_schema))
      tensor_adapter = projected_tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertLen(dict_of_tensors, 1)

    p = beam.Pipeline()
    record_batch_pcoll = (
        self._MakePipelineInputs(p, use_beam_record_csv_tfxio)
        | tfxio.BeamSource(batch_size=len(_ROWS)))
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, "csv",
                                        _EXPECTED_PHYSICAL_FORMAT)

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testAttachRawRecordColumn(self, use_beam_record_csv_tfxio):
    raw_record_column_name = "raw_records"
    tfxio = self._MakeTFXIO(
        _COLUMN_NAMES,
        schema=_SCHEMA,
        raw_record_column_name=raw_record_column_name,
        make_beam_record_tfxio=use_beam_record_csv_tfxio)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, raw_record_column_name)

    with beam.Pipeline() as p:
      record_batch_pcoll = (
          self._MakePipelineInputs(p, use_beam_record_csv_tfxio)
          | tfxio.BeamSource(batch_size=len(_ROWS)))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testOptionalSchema(self, use_beam_record_csv_tfxio):
    """Tests when the schema is not provided."""
    tfxio = self._MakeTFXIO(
        _COLUMN_NAMES, make_beam_record_tfxio=use_beam_record_csv_tfxio)
    with self.assertRaisesRegex(ValueError, ".*TFMD schema not provided.*"):
      tfxio.ArrowSchema()

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch)

    with beam.Pipeline() as p:
      record_batch_pcoll = (
          self._MakePipelineInputs(p, use_beam_record_csv_tfxio)
          | tfxio.BeamSource(batch_size=len(_ROWS)))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  @parameterized.named_parameters(*_CSV_TFXIO_IMPL_TEST_CASES)
  def testUnorderedSchema(self, use_beam_record_csv_tfxio):
    """Tests various valid schemas."""
    tfxio = self._MakeTFXIO(_COLUMN_NAMES, schema=_UNORDERED_SCHEMA,
                            make_beam_record_tfxio=use_beam_record_csv_tfxio)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch)

    with beam.Pipeline() as p:
      record_batch_pcoll = (
          self._MakePipelineInputs(p, use_beam_record_csv_tfxio)
          | tfxio.BeamSource(batch_size=len(_ROWS)))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testSkipHeaderLines(self):
    tfxio = self._MakeTFXIO(_COLUMN_NAMES,
                            use_input_with_header_line=True,
                            skip_header_lines=1)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch)

    with beam.Pipeline() as p:
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

if __name__ == "__main__":
  absltest.main()
