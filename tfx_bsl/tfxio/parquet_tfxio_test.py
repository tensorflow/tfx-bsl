import os

import apache_beam as beam
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from absl import flags
from absl.testing import absltest
from apache_beam.testing import util as beam_testing_util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import telemetry_test_util

from tfx_bsl.tfxio.parquet_tfxio import ParquetTFXIO

FLAGS = flags.FLAGS
_COLUMN_NAMES = ["int_feature", "float_feature", "string_feature"]
_TELEMETRY_DESCRIPTORS = ["Some", "Component"]
_ROWS = {
    "int_feature": [[1], [2]],
    "float_feature": [[2.0], [3.0]],
    "string_feature": [["abc"], ["xyz"]]
}
_NUM_ROWS = len(next(_ROWS))

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

_EXPECTED_ARROW_SCHEMA = pa.schema([
    pa.field("int_feature", pa.large_list(pa.int64())),
    pa.field("float_feature", pa.large_list(pa.float32())),
    pa.field("string_feature", pa.large_list(pa.large_binary()))
])

_EXPECTED_PROJECTED_ARROW_SCHEMA = pa.schema([
    pa.field("int_feature", pa.large_list(pa.int64())),
])

_EXPECTED_COLUMN_VALUES = {
    "int_feature":
        pa.array([[1], [2]], type=pa.large_list(pa.int64())),
    "float_feature":
        pa.array([[2.0], [3.0]], type=pa.large_list(pa.float32())),
    "string_feature":
        pa.array([[b"abc"], [b"xyz"]], type=pa.large_list(pa.large_binary())),
}


def _WriteInputs(filename):
  df = pd.DataFrame(_ROWS)
  table = pa.Table.from_pandas(df, schema=_EXPECTED_ARROW_SCHEMA)
  pq.write_table(table, filename)


class ParquetRecordTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._example_file = os.path.join(FLAGS.test_tmpdir, "parquettest",
                                     "examples.parquet")
    tf.io.gfile.makedirs(os.path.dirname(cls._example_file))
    _WriteInputs(cls._example_file)

  def testImplicitTensorRepresentations(self):
    """Tests inferring of tensor representation."""
    tfxio = ParquetTFXIO(file_pattern=self._example_file,
                         column_names=_COLUMN_NAMES,
                         schema=_UNORDERED_SCHEMA,
                         telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
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
      self._ValidateRecordBatch(record_batch, _EXPECTED_ARROW_SCHEMA)
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 3)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("string_feature", dict_of_tensors)

    pipeline = beam.Pipeline()
    record_batch_pcoll = (pipeline | tfxio.BeamSource(batch_size=_NUM_ROWS))
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = pipeline.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, 'parquet',
                                        'parquet')

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

    tfxio = ParquetTFXIO(file_pattern=self._example_file,
                         column_names=_COLUMN_NAMES,
                         schema=schema,
                         telemetry_descriptors=_TELEMETRY_DESCRIPTORS)
    self.assertEqual(tensor_representations, tfxio.TensorRepresentations())

  def testProjection(self):
    """Test projecting of a TFXIO."""
    tfxio = ParquetTFXIO(file_pattern=self._example_file,
                         column_names=_COLUMN_NAMES,
                         schema=_UNORDERED_SCHEMA,
                         telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

    projected_tfxio = tfxio.Project(["int_feature"])

    # The projected_tfxio has the projected schema
    self.assertTrue(
        projected_tfxio.ArrowSchema().equals(_EXPECTED_PROJECTED_ARROW_SCHEMA))

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, _EXPECTED_PROJECTED_ARROW_SCHEMA)
      expected_schema = projected_tfxio.ArrowSchema()
      self.assertListEqual(
          record_batch.schema.names,
          expected_schema.names,
          "actual: {}; expected: {}".format(record_batch.schema.names,
                                            expected_schema.names))
      self.assertListEqual(
          record_batch.schema.types,
          expected_schema.types,
          "actual: {}; expected: {}".format(record_batch.schema.types,
                                            expected_schema.types))
      tensor_adapter = projected_tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 1)
      self.assertIn("int_feature", dict_of_tensors)

    pipeline = beam.Pipeline()
    record_batch_pcoll = (pipeline | projected_tfxio.BeamSource(batch_size=_NUM_ROWS))
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = pipeline.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, 'parquet',
                                        'parquet')

  def testOptionalSchema(self):
    """Tests when the schema is not provided."""
    tfxio = ParquetTFXIO(file_pattern=self._example_file,
                         column_names=_COLUMN_NAMES,
                         telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

    with self.assertRaisesRegex(ValueError, ".*TFMD schema not provided.*"):
      tfxio.ArrowSchema()

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, _EXPECTED_ARROW_SCHEMA)

    pipeline = beam.Pipeline()
    record_batch_pcoll = (pipeline | tfxio.BeamSource(batch_size=_NUM_ROWS))
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = pipeline.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(self, pipeline_result,
                                        _TELEMETRY_DESCRIPTORS, 'parquet',
                                        'parquet')

  def testUnorderedSchema(self):
    """Tests various valid schemas."""
    tfxio = ParquetTFXIO(file_pattern=self._example_file,
                         column_names=_COLUMN_NAMES,
                         schema=_UNORDERED_SCHEMA)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, _EXPECTED_ARROW_SCHEMA)

    with beam.Pipeline() as p:
      record_batch_pcoll = (p | tfxio.BeamSource(batch_size=_NUM_ROWS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def _ValidateRecordBatch(self, record_batch, expected_arrow_schema):
    self.assertIsInstance(record_batch, pa.RecordBatch)
    self.assertEqual(record_batch.num_rows, 2)
    # when reading the parquet files and then transforming them to RecordBatches,
    # metadata is populated, specifically the pandas metadata.
    # We do not assert that metadata.
    self.assertListEqual(
        record_batch.schema.names,
        expected_arrow_schema.names,
        "Expected: {} ; got {}".format(expected_arrow_schema.names, record_batch.schema.names))
    self.assertListEqual(
        record_batch.schema.types,
        expected_arrow_schema.types,
        "Expected: {} ; got {}".format(expected_arrow_schema.types, record_batch.schema.types))
    for i, field in enumerate(record_batch.schema):
      self.assertTrue(
          record_batch.column(i).equals(_EXPECTED_COLUMN_VALUES[field.name]),
          "Column {} did not match ({} vs {}).".format(
              field.name, record_batch.column(i),
              _EXPECTED_COLUMN_VALUES[field.name]))


if __name__ == "__main__":
  absltest.main()
