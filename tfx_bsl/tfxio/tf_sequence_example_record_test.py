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

import os

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.arrow import path
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import telemetry_test_util
from tfx_bsl.tfxio import test_case
from tfx_bsl.tfxio import tf_sequence_example_record

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2


FLAGS = flags.FLAGS

_SEQUENCE_COLUMN_NAME = "##SEQUENCE##"
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
    name: "$SEQ"
    type: STRUCT
    struct_domain {
      feature {
        name: "int_feature"
        type: INT
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
    }
  }
  tensor_representation_group {
    key: ""
    value {
      tensor_representation {
        key: "int_feature"
        value { varlen_sparse_tensor { column_name: "int_feature" } }
      }
      tensor_representation {
        key: "float_feature"
        value { varlen_sparse_tensor { column_name: "float_feature" } }
      }
      tensor_representation {
        key: "seq_string_feature"
        value { ragged_tensor {
                    feature_path { step: "$SEQ" step: "string_feature" } } }
      }
      tensor_representation {
        key: "seq_int_feature"
        value { ragged_tensor {
                    feature_path { step: "$SEQ" step: "int_feature" } } }
      }
    }
  }
""".replace("$SEQ", _SEQUENCE_COLUMN_NAME), schema_pb2.Schema())

_TELEMETRY_DESCRIPTORS = ["Some", "Component"]


_EXAMPLES = [
    """
  context {
    feature { key: "int_feature" value { int64_list { value: [1] } }
    }
    feature {
      key: "float_feature"
      value { float_list { value: [1.0, 2.0, 3.0, 4.0] } }
    }
  }
  feature_lists {
    feature_list {
      key: "int_feature"
      value {
        feature { int64_list { value: [1, 2] } }
        feature { int64_list { value: [3] } }
      }
    }
  }
""",
    """
  context {
    feature { key: "int_feature" value { int64_list { value: [2] } } }
    feature { key: "float_feature"
      value { float_list { value: [2.0, 3.0, 4.0, 5.0] } }
    }
  }
  feature_lists {
    feature_list {
      key: "string_feature"
      value {
        feature { bytes_list { value: ["foo", "bar"] } }
        feature { bytes_list { value: [] } }
      }
    }
  }
""",
    """
  context {
    feature { key: "int_feature" value { int64_list { value: [3] } } }
  }
  feature_lists {
    feature_list {
      key: "int_feature"
      value {
        feature { int64_list { value: [4] } }
      }
    }
    feature_list {
      key: "string_feature"
      value {
        feature { bytes_list { value: ["baz"] } }
      }
    }
  }
""",
]


_SERIALIZED_EXAMPLES = [
    text_format.Parse(pbtxt, tf.train.SequenceExample()).SerializeToString()
    for pbtxt in _EXAMPLES
]


_EXPECTED_COLUMN_VALUES = {
    path.ColumnPath(["int_feature"]):
        pa.array([[1], [2], [3]], type=pa.large_list(pa.int64())),
    path.ColumnPath(["float_feature"]):
        pa.array([[1, 2, 3, 4], [2, 3, 4, 5], None],
                 type=pa.large_list(pa.float32())),
    path.ColumnPath([_SEQUENCE_COLUMN_NAME, "int_feature"]):
        pa.array([[[1, 2], [3]], None, [[4]]],
                 pa.large_list(pa.large_list(pa.int64()))),
    path.ColumnPath([_SEQUENCE_COLUMN_NAME, "string_feature"]):
        pa.array([None, [[b"foo", b"bar"], []], [[b"baz"]]],
                 pa.large_list(pa.large_list(pa.large_binary())))
}


def _CreateExamplesAsTensors():
  if tf.executing_eagerly():
    sparse_tensor_factory = tf.SparseTensor
  else:
    sparse_tensor_factory = tf.compat.v1.SparseTensorValue
  ragged_tensor_factory = tf.ragged.constant

  return [{
      "int_feature":
          sparse_tensor_factory(
              values=[1], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[1.0, 2.0, 3.0, 4.0],
              indices=[[0, 0], [0, 1], [0, 2], [0, 3]],
              dense_shape=[1, 4]),
      "seq_string_feature":
          ragged_tensor_factory([[]], dtype=tf.string, ragged_rank=2),
      "seq_int_feature":
          ragged_tensor_factory([[[1, 2], [3]]], dtype=tf.int64, ragged_rank=2),
  }, {
      "int_feature":
          sparse_tensor_factory(
              values=[2], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[2.0, 3.0, 4.0, 5.0],
              indices=[[0, 0], [0, 1], [0, 2], [0, 3]],
              dense_shape=[1, 4]),
      "seq_string_feature":
          ragged_tensor_factory([[[b"foo", b"bar"], []]],
                                dtype=tf.string,
                                ragged_rank=2),
      "seq_int_feature":
          ragged_tensor_factory([[]], dtype=tf.int64, ragged_rank=2),
  }, {
      "int_feature":
          sparse_tensor_factory(
              values=[3], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[], indices=np.empty((0, 2)), dense_shape=[1, 0]),
      "seq_string_feature":
          ragged_tensor_factory([[[b"baz"]]], dtype=tf.string, ragged_rank=2),
      "seq_int_feature":
          ragged_tensor_factory([[[4]]], dtype=tf.int64, ragged_rank=2),
  }]


_EXPECTED_TENSORS = _CreateExamplesAsTensors()


def _WriteInputs(filename):
  with tf.io.TFRecordWriter(filename, "GZIP") as w:
    for s in _SERIALIZED_EXAMPLES:
      w.write(s)


class TfSequenceExampleRecordTest(test_case.TfxBslTestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls._example_file = os.path.join(
        FLAGS.test_tmpdir, "tfsequenceexamplerecordtest", "input.recordio.gz")
    tf.io.gfile.makedirs(os.path.dirname(cls._example_file))
    _WriteInputs(cls._example_file)

  def _MakeTFXIO(self, schema, raw_record_column_name=None):
    return tf_sequence_example_record.TFSequenceExampleRecord(
        self._example_file, schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=_TELEMETRY_DESCRIPTORS)

  def _ValidateRecordBatch(
      self, record_batch, raw_record_column_name=None):
    self.assertIsInstance(record_batch, pa.RecordBatch)
    self.assertEqual(record_batch.num_rows, 3)
    for i, field in enumerate(record_batch.schema):
      if field.name == raw_record_column_name:
        continue
      if field.name == _SEQUENCE_COLUMN_NAME:
        self.assertTrue(pa.types.is_struct(field.type))
        for seq_column, seq_field in zip(
            record_batch.column(i).flatten(), list(field.type)):
          expected_array = _EXPECTED_COLUMN_VALUES[path.ColumnPath(
              [_SEQUENCE_COLUMN_NAME, seq_field.name])]
          self.assertTrue(
              seq_column.equals(expected_array),
              "Sequence column {} did not match ({} vs {})".format(
                  seq_field.name, seq_column, expected_array))
        continue
      self.assertTrue(
          record_batch.column(i).equals(_EXPECTED_COLUMN_VALUES[path.ColumnPath(
              [field.name])]), "Column {} did not match ({} vs {}).".format(
                  field.name, record_batch.column(i),
                  _EXPECTED_COLUMN_VALUES[path.ColumnPath([field.name])]))

    if raw_record_column_name is not None:
      self.assertEqual(record_batch.schema.names[-1], raw_record_column_name)
      self.assertTrue(record_batch.columns[-1].type.equals(
          pa.large_list(pa.large_binary())))
      self.assertEqual(record_batch.columns[-1].flatten().to_pylist(),
                       _SERIALIZED_EXAMPLES)

  @test_case.named_parameters(*[
      dict(testcase_name="attach_raw_records", attach_raw_records=True),
      dict(testcase_name="noattach_raw_records", attach_raw_records=False),
  ])
  def testE2E(self, attach_raw_records):
    raw_column_name = "raw_records" if attach_raw_records else None
    tfxio = self._MakeTFXIO(_SCHEMA, raw_column_name)

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, raw_column_name)
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 4)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("seq_string_feature", dict_of_tensors)
      self.assertIn("seq_int_feature", dict_of_tensors)

    p = beam.Pipeline()
    record_batch_pcoll = p | tfxio.BeamSource(batch_size=1000)
    beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(
        self, pipeline_result, _TELEMETRY_DESCRIPTORS,
        "tf_sequence_example", "tfrecords_gzip")

  @test_case.named_parameters(*[
      dict(testcase_name="attach_raw_records", attach_raw_records=True),
      dict(testcase_name="noattach_raw_records", attach_raw_records=False),
  ])
  def testProjection(self, attach_raw_records):
    raw_column_name = "raw_records" if attach_raw_records else None
    tfxio = self._MakeTFXIO(_SCHEMA, raw_column_name).Project(
        ["int_feature", "seq_string_feature"])
    self.assertEqual(set(["int_feature", "seq_string_feature"]),
                     set(tfxio.TensorRepresentations()))

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch, raw_column_name)
      expected_schema = tfxio.ArrowSchema()
      self.assertTrue(
          record_batch.schema.equals(expected_schema),
          "actual: {}; expected: {}".format(
              record_batch.schema, expected_schema))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 2)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("seq_string_feature", dict_of_tensors)

    with beam.Pipeline() as p:
      # Setting the betch_size to make sure only one batch is generated.
      record_batch_pcoll = p | tfxio.BeamSource(
          batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testProjectionNoSequenceFeature(self):
    tfxio = self._MakeTFXIO(_SCHEMA).Project(["int_feature"])
    arrow_schema = tfxio.ArrowSchema()
    self.assertLen(arrow_schema, 1)
    self.assertIn("int_feature", arrow_schema.names)
    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self._ValidateRecordBatch(record_batch)
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 1)
      self.assertIn("int_feature", dict_of_tensors)

    with beam.Pipeline() as p:
      # Setting the betch_size to make sure only one batch is generated.
      record_batch_pcoll = p | tfxio.BeamSource(
          batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testProjectEmpty(self):
    tfxio = self._MakeTFXIO(_SCHEMA).Project([])
    self.assertEmpty(tfxio.ArrowSchema())
    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]
      self.assertEqual(record_batch.num_columns, 0)
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertEmpty(dict_of_tensors)
    with beam.Pipeline() as p:
      # Setting the betch_size to make sure only one batch is generated.
      record_batch_pcoll = p | tfxio.BeamSource(
          batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testTensorAdapter(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    tensor_adapter = tfxio.TensorAdapter()

    def _AssertFn(tensor_dicts_list):
      # Sort the produced list of batched tensor dicts to avoid errors due to
      # order of elements in the produced PCollection.
      tensor_dicts_list = sorted(
          tensor_dicts_list, key=lambda x: x["int_feature"].values[0])

      for i, tensors_dict in enumerate(tensor_dicts_list):
        self.assertLen(tensors_dict, 4)
        for name, tensor in tensors_dict.items():
          self.assertTensorsEqual(
              tensor,
              _EXPECTED_TENSORS[i][name],
              msg=f"For tensor {name} at index {i}")

    with beam.Pipeline() as p:
      # Expected tensors are batched with batch_size = 1.
      record_batches = p | tfxio.BeamSource(batch_size=1)
      tensors = record_batches | beam.Map(tensor_adapter.ToBatchTensors)
      beam_testing_util.assert_that(tensors, _AssertFn)

  def testTensorFlowDataset(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    # Expected tensors are batched with batch_size = 1.
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, parsed_examples_dict in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      self.assertLen(parsed_examples_dict, 4)
      for name, tensor in parsed_examples_dict.items():
        self.assertTensorsEqual(
            tensor,
            _EXPECTED_TENSORS[i][name],
            msg=f"For tensor {name} at index {i}")


class TFSequenceExampleBeamRecordTest(test_case.TfxBslTestCase):

  def testE2E(self):
    raw_record_column_name = "raw_record"
    tfxio = tf_sequence_example_record.TFSequenceExampleBeamRecord(
        physical_format="inmem",
        telemetry_descriptors=["some", "component"],
        schema=_SCHEMA,
        raw_record_column_name=raw_record_column_name,
    )

    def _AssertFn(record_batches):
      self.assertLen(record_batches, 1)
      record_batch = record_batches[0]
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      tensor_adapter = tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 4)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("seq_string_feature", dict_of_tensors)
      self.assertIn("seq_int_feature", dict_of_tensors)

    with beam.Pipeline() as p:
      record_batch_pcoll = (
          p
          | "CreateInMemRecords" >> beam.Create(_SERIALIZED_EXAMPLES)
          | "BeamSource" >>
          tfxio.BeamSource(batch_size=len(_SERIALIZED_EXAMPLES)))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)


if __name__ == "__main__":
  test_case.main()
