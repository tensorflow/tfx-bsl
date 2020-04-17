# Copyright 2019 Google LLC
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
"""Tests for tfx_bsl.tfxio.tf_example_record."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import telemetry_test_util
from tfx_bsl.tfxio import tf_example_record
from google.protobuf import text_format
from absl.testing import absltest
from tensorflow_metadata.proto.v0 import schema_pb2


FLAGS = flags.FLAGS

_SCHEMA = text_format.Parse("""
  feature {
    name: "int_feature"
    type: INT
    value_count {
      min: 1
      max: 1
    }
  }
  feature {
    name: "float_feature"
    type: FLOAT
    value_count {
      min: 4
      max: 4
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

_IS_LEGACY_SCHEMA = (
    "generate_legacy_feature_spec" in
    schema_pb2.Schema.DESCRIPTOR.fields_by_name)

# Enforce a consistent behavior in inferring TensorRepresentations from the
# schema.
if _IS_LEGACY_SCHEMA:
  _SCHEMA.generate_legacy_feature_spec = False

_EXAMPLES = [
    """
  features {
    feature { key: "int_feature" value { int64_list { value: [1] } }
    }
    feature {
      key: "float_feature"
      value { float_list { value: [1.0, 2.0, 3.0, 4.0] } }
    }
    feature { key: "string_feature" value { } }
  }
""",
    """
  features {
    feature { key: "int_feature" value { int64_list { value: [2] } } }
    feature { key: "float_feature"
      value { float_list { value: [2.0, 3.0, 4.0, 5.0] } }
    }
    feature {
      key: "string_feature"
      value { bytes_list { value: ["foo", "bar"] } }
    }
  }
""",
    """
  features {
    feature { key: "int_feature" value { int64_list { value: [3] } } }
    feature {
      key: "float_feature"
      value { float_list { value: [4.0, 5.0, 6.0, 7.0] } }
    }
  }
""",
]


_SERIALIZED_EXAMPLES = [text_format.Parse(
    pbtxt, tf.train.Example()).SerializeToString() for pbtxt in _EXAMPLES]


def GetExpectedColumnValues(tfxio):
  if tfxio._can_produce_large_types:
    int_type = pa.large_list(pa.int64())
    float_type = pa.large_list(pa.float32())
    bytes_type = pa.large_list(pa.large_binary())
  else:
    int_type = pa.list_(pa.int64())
    float_type = pa.list_(pa.float32())
    bytes_type = pa.list_(pa.binary())

  return {
      "int_feature":
          pa.array([[1], [2], [3]], type=int_type),
      "float_feature":
          pa.array([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7]], type=float_type),
      "string_feature":
          pa.array([None, ["foo", "bar"], None], type=bytes_type),
  }


def _WriteInputs(filename):
  with tf.io.TFRecordWriter(filename, "GZIP") as w:
    for s in _SERIALIZED_EXAMPLES:
      w.write(s)


class TfExampleRecordTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TfExampleRecordTest, cls).setUpClass()
    cls._example_file = os.path.join(
        FLAGS.test_tmpdir, "tfexamplerecordtest", "input.recordio.gz")
    tf.io.gfile.makedirs(os.path.dirname(cls._example_file))
    _WriteInputs(cls._example_file)

  def _MakeTFXIO(self, schema, raw_record_column_name=None):
    return tf_example_record.TFExampleRecord(
        self._example_file, schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

  def _ValidateRecordBatch(
      self, tfxio, record_batch, raw_record_column_name=None):
    self.assertIsInstance(record_batch, pa.RecordBatch)
    self.assertEqual(record_batch.num_rows, 3)
    expected_column_values = GetExpectedColumnValues(tfxio)
    for i, field in enumerate(record_batch.schema):
      if field.name == raw_record_column_name:
        continue
      self.assertTrue(record_batch.column(i).equals(
          expected_column_values[field.name]),
                      "Column {} did not match ({} vs {})."
                      .format(field.name, record_batch.column(i),
                              expected_column_values[field.name]))

    if raw_record_column_name is not None:
      if tfxio._can_produce_large_types:
        raw_record_column_type = pa.large_list(pa.large_binary())
      else:
        raw_record_column_type = pa.list_(pa.binary())
      self.assertEqual(record_batch.schema.names[-1], raw_record_column_name)
      self.assertTrue(
          record_batch.columns[-1].type.equals(raw_record_column_type))
      self.assertEqual(record_batch.columns[-1].flatten().to_pylist(),
                       _SERIALIZED_EXAMPLES)

  def testImplicitTensorRepresentations(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    self.assertEqual(
        {
            "int_feature": text_format.Parse(
                """varlen_sparse_tensor { column_name: "int_feature" }""",
                schema_pb2.TensorRepresentation()),
            "float_feature": text_format.Parse(
                """varlen_sparse_tensor { column_name: "float_feature" }""",
                schema_pb2.TensorRepresentation()),
            "string_feature": text_format.Parse(
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
    telemetry_test_util.ValidateMetrics(
        self, pipeline_result, _TELEMETRY_DESCRIPTORS,
        "tf_example", "tfrecords_gzip")

  def testExplicitTensorRepresentations(self):
    schema = schema_pb2.Schema()
    schema.CopyFrom(_SCHEMA)
    tensor_representations = {
        "my_feature":
            text_format.Parse("""
            dense_tensor {
             column_name: "string_feature"
             shape { dim { size: 2 } }
             default_value { bytes_value: "zzz" }
           }""", schema_pb2.TensorRepresentation())
    }
    schema.tensor_representation_group[""].CopyFrom(
        schema_pb2.TensorRepresentationGroup(
            tensor_representation=tensor_representations))

    tfxio = self._MakeTFXIO(schema)
    self.assertEqual(tensor_representations,
                     tfxio.TensorRepresentations())

  def testProjection(self):
    schema = schema_pb2.Schema()
    schema.CopyFrom(_SCHEMA)
    tensor_representations = {
        "dense_string":
            text_format.Parse(
                """dense_tensor {
             column_name: "string_feature"
             shape { dim { size: 2 } }
             default_value { bytes_value: "zzz" }
           }""", schema_pb2.TensorRepresentation()),
        "varlen_string":
            text_format.Parse(
                """varlen_sparse_tensor {
             column_name: "string_feature"
           }""", schema_pb2.TensorRepresentation()),
        "varlen_float":
            text_format.Parse(
                """varlen_sparse_tensor {
             column_name: "float_feature"
           }""", schema_pb2.TensorRepresentation()),
    }
    schema.tensor_representation_group[""].CopyFrom(
        schema_pb2.TensorRepresentationGroup(
            tensor_representation=tensor_representations))

    tfxio = self._MakeTFXIO(schema)
    self.assertEqual(tensor_representations, tfxio.TensorRepresentations())

    projected_tfxio = tfxio.Project(
        ["dense_string", "varlen_string", "varlen_float"])
    self.assertEqual(tensor_representations,
                     projected_tfxio.TensorRepresentations())

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(tfxio, record_batch)
      expected_schema = projected_tfxio.ArrowSchema()
      self.assertTrue(
          record_batch.schema.equals(expected_schema),
          "actual: {}; expected: {}".format(
              record_batch.schema, expected_schema))
      tensor_adapter = projected_tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 3)
      self.assertIn("dense_string", dict_of_tensors)
      self.assertIn("varlen_string", dict_of_tensors)
      self.assertIn("varlen_float", dict_of_tensors)

    with beam.Pipeline() as p:
      # Setting the betch_size to make sure only one batch is generated.
      record_batch_pcoll = p | projected_tfxio.BeamSource(
          batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testAttachRawRecordColumn(self):
    raw_example_column_name = "raw_records"
    tfxio = self._MakeTFXIO(_SCHEMA, raw_example_column_name)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      self._ValidateRecordBatch(tfxio, record_batch, raw_example_column_name)

    with beam.Pipeline() as p:
      # Setting the batch_size to make sure only one batch is generated.
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)


if __name__ == "__main__":
  absltest.main()
