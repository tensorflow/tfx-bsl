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
import unittest

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import telemetry_test_util
from tfx_bsl.tfxio import tf_example_record

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
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
    feature {
      key: "varlen_feature"
      value { int64_list { value: [1, 2, 3] } }
    }
    feature {
      key: "row_lengths"
      value { int64_list { value: [2, 1] } }
    }
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
    feature {
      key: "varlen_feature"
      value { int64_list { value: [4] } }
    }
    feature {
      key: "row_lengths"
      value { int64_list { value: [1] } }
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
    feature {
      key: "varlen_feature"
      value { int64_list { value: [5, 6] } }
    }
    feature {
      key: "row_lengths"
      value { int64_list { value: [1, 1] } }
    }
  }
""",
]

_SERIALIZED_EXAMPLES = [
    text_format.Parse(pbtxt, tf.train.Example()).SerializeToString()
    for pbtxt in _EXAMPLES
]


def CreateExamplesAsTensors():
  if tf.executing_eagerly():
    sparse_tensor_factory = tf.SparseTensor
  else:
    sparse_tensor_factory = tf.compat.v1.SparseTensorValue

  return [{
      "int_feature":
          sparse_tensor_factory(
              values=[1], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[1.0, 2.0, 3.0, 4.0],
              indices=[[0, 0], [0, 1], [0, 2], [0, 3]],
              dense_shape=[1, 4]),
      "string_feature":
          sparse_tensor_factory(
              values=[], indices=np.empty((0, 2)), dense_shape=[1, 0])
  }, {
      "int_feature":
          sparse_tensor_factory(
              values=[2], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[2.0, 3.0, 4.0, 5.0],
              indices=[[0, 0], [0, 1], [0, 2], [0, 3]],
              dense_shape=[1, 4]),
      "string_feature":
          sparse_tensor_factory(
              values=[b"foo", b"bar"],
              indices=[[0, 0], [0, 1]],
              dense_shape=[1, 2])
  }, {
      "int_feature":
          sparse_tensor_factory(
              values=[3], indices=[[0, 0]], dense_shape=[1, 1]),
      "float_feature":
          sparse_tensor_factory(
              values=[4.0, 5.0, 6.0, 7.0],
              indices=[[0, 0], [0, 1], [0, 2], [0, 3]],
              dense_shape=[1, 4]),
      "string_feature":
          sparse_tensor_factory(
              values=[], indices=np.empty((0, 2)), dense_shape=[1, 0])
  }]


_EXAMPLES_AS_TENSORS = CreateExamplesAsTensors()


_EXPECTED_COLUMN_VALUES = {
    "int_feature":
        pa.array([[1], [2], [3]], type=pa.large_list(pa.int64())),
    "float_feature":
        pa.array([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7]],
                 type=pa.large_list(pa.float32())),
    "string_feature":
        pa.array([None, ["foo", "bar"], None],
                 type=pa.large_list(pa.large_binary())),
}


def _WriteInputs(filename):
  with tf.io.TFRecordWriter(filename, "GZIP") as w:
    for s in _SERIALIZED_EXAMPLES:
      w.write(s)


class TfExampleRecordTest(tf.test.TestCase, parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
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
    for i, field in enumerate(record_batch.schema):
      if field.name == raw_record_column_name:
        continue
      self.assertTrue(record_batch.column(i).equals(
          _EXPECTED_COLUMN_VALUES[field.name]),
                      "Column {} did not match ({} vs {})."
                      .format(field.name, record_batch.column(i),
                              _EXPECTED_COLUMN_VALUES[field.name]))

    if raw_record_column_name is not None:
      self.assertEqual(record_batch.schema.names[-1], raw_record_column_name)
      self.assertTrue(record_batch.columns[-1].type.equals(
          pa.large_list(pa.large_binary())))
      self.assertEqual(record_batch.columns[-1].flatten().to_pylist(),
                       _SERIALIZED_EXAMPLES)

  def _AssertSparseTensorEqual(self, lhs, rhs):
    self.assertAllEqual(lhs.values, rhs.values)
    self.assertAllEqual(lhs.indices, rhs.indices)
    self.assertAllEqual(lhs.dense_shape, rhs.dense_shape)

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
        "varlen_int":
            text_format.Parse(
                """varlen_sparse_tensor {
             column_name: "int_feature"
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
        ["dense_string", "varlen_int", "varlen_float"])
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
      self.assertIn("varlen_int", dict_of_tensors)
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

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testRecordBatches(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    options = dataset_options.RecordBatchesOptions(
        batch_size=len(_EXAMPLES), shuffle=False, num_epochs=1)
    for record_batch in tfxio.RecordBatches(options):
      self._ValidateRecordBatch(tfxio, record_batch)

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testRecordBatchesWithRawRecords(self):
    raw_example_column_name = "raw_records"
    tfxio = self._MakeTFXIO(_SCHEMA, raw_example_column_name)
    options = dataset_options.RecordBatchesOptions(
        batch_size=len(_EXAMPLES), shuffle=False, num_epochs=1)
    for record_batch in tfxio.RecordBatches(options):
      self._ValidateRecordBatch(tfxio, record_batch, raw_example_column_name)

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testRecordBatchesWithProject(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    feature_name = "string_feature"
    projected_tfxio = tfxio.Project([feature_name])
    options = dataset_options.RecordBatchesOptions(
        batch_size=len(_EXAMPLES), shuffle=False, num_epochs=1)
    for record_batch in projected_tfxio.RecordBatches(options):
      self._ValidateRecordBatch(projected_tfxio, record_batch)
      self.assertIn(feature_name, record_batch.schema.names)
      self.assertLen(record_batch.schema.names, 1)

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testTensorFlowDataset(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, parsed_examples_dict in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      self.assertLen(parsed_examples_dict, 3)
      for tensor_name, tensor in parsed_examples_dict.items():
        self._AssertSparseTensorEqual(
            tensor, _EXAMPLES_AS_TENSORS[i][tensor_name])

  def testTensorFlowDatasetGraphMode(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    with tf.compat.v1.Graph().as_default():
      ds = tfxio.TensorFlowDataset(options=options)
      iterator = tf.compat.v1.data.make_one_shot_iterator(ds)
      next_elem = iterator.get_next()
      records = []
      with tf.compat.v1.Session() as sess:
        while True:
          try:
            records.append(sess.run(next_elem))
          except tf.errors.OutOfRangeError:
            break
    for i, parsed_examples_dict in enumerate(records):
      self.assertLen(parsed_examples_dict, 3)
      for tensor_name, tensor in parsed_examples_dict.items():
        self._AssertSparseTensorEqual(
            tensor, _EXAMPLES_AS_TENSORS[i][tensor_name])

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testTensorFlowDatasetWithTensorRepresentation(self):
    schema = text_format.Parse("""
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
      tensor_representation_group {
    key: ""
    value {
      tensor_representation {
        key: "var_len_feature"
        value {
          varlen_sparse_tensor {
            column_name: "string_feature"
          }
        }
      }
    }
  }
    """, schema_pb2.Schema())
    tfxio = self._MakeTFXIO(schema)
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, parsed_examples_dict in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      self.assertLen(parsed_examples_dict, 3)
      for name in ["var_len_feature", "int_feature", "float_feature"]:
        self.assertIn(name, parsed_examples_dict)
      self._AssertSparseTensorEqual(parsed_examples_dict["var_len_feature"],
                                    _EXAMPLES_AS_TENSORS[i]["string_feature"])

  def testTensorFlowDatasetWithRaggedTensorRepresentation(self):
    schema = text_format.Parse(
        """
      feature {
        name: "varlen_feature"
        type: INT
      }
      feature {
        name: "row_lengths"
        type: INT
      }
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "ragged"
            value {
              ragged_tensor {
                feature_path { step: "varlen_feature" }
                partition { row_length: "row_lengths" }
              }
            }
          }
        }
      }
    """, schema_pb2.Schema())
    tfxio = self._MakeTFXIO(schema)
    projected_tfxio = tfxio.Project(["ragged"])

    expected_column_values = {
        "varlen_feature":
            pa.array([[1, 2, 3], [4], [5, 6]], type=pa.large_list(pa.int64())),
        "row_lengths":
            pa.array([[2, 1], [1], [1, 1]], type=pa.large_list(pa.int64())),
    }

    def _AssertFn(record_batch_list):
      self.assertLen(record_batch_list, 1)
      record_batch = record_batch_list[0]

      self.assertIsInstance(record_batch, pa.RecordBatch)
      self.assertEqual(record_batch.num_rows, 3)
      print(record_batch.schema)
      for i, field in enumerate(record_batch.schema):
        self.assertTrue(
            record_batch.column(i).equals(expected_column_values[field.name]),
            "Column {} did not match ({} vs {}).".format(
                field.name, record_batch.column(i),
                expected_column_values[field.name]))

      # self._ValidateRecordBatch(tfxio, record_batch)
      expected_schema = projected_tfxio.ArrowSchema()
      self.assertTrue(
          record_batch.schema.equals(expected_schema),
          "actual: {}; expected: {}".format(record_batch.schema,
                                            expected_schema))
      tensor_adapter = projected_tfxio.TensorAdapter()
      dict_of_tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(dict_of_tensors, 1)
      self.assertIn("ragged", dict_of_tensors)

      if tf.executing_eagerly():
        ragged_factory = tf.RaggedTensor.from_row_splits
      else:
        ragged_factory = tf.compat.v1.ragged.RaggedTensorValue
      expected_tensor = ragged_factory(
          values=ragged_factory(
              values=[1, 2, 3, 4, 5, 6], row_splits=[0, 2, 3, 4, 5, 6]),
          row_splits=[0, 2, 3, 5])
      self.assertAllEqual(dict_of_tensors["ragged"], expected_tensor)

    with beam.Pipeline() as p:
      # Setting the betch_size to make sure only one batch is generated.
      record_batch_pcoll = p | projected_tfxio.BeamSource(
          batch_size=len(_EXAMPLES))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

    if tf.executing_eagerly():
      ragged_factory = tf.RaggedTensor.from_row_splits
    else:
      ragged_factory = tf.compat.v1.ragged.RaggedTensorValue

    expected_tensors = [
        ragged_factory(
            values=ragged_factory(values=[1, 2, 3], row_splits=[0, 2, 3]),
            row_splits=[0, 2]),
        ragged_factory(
            values=ragged_factory(values=[4], row_splits=[0, 1]),
            row_splits=[0, 1]),
        ragged_factory(
            values=ragged_factory(values=[5, 6], row_splits=[0, 1, 2]),
            row_splits=[0, 2]),
    ]

    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, parsed_examples_dict in enumerate(
        projected_tfxio.TensorFlowDataset(options)):
      self.assertLen(parsed_examples_dict, 1)
      self.assertIn("ragged", parsed_examples_dict)
      self.assertAllEqual(parsed_examples_dict["ragged"], expected_tensors[i])

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testTensorFlowDatasetWithLabelKey(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1, label_key="string_feature")
    for i, (parsed_examples_dict, label_feature) in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      self._AssertSparseTensorEqual(
          label_feature, _EXAMPLES_AS_TENSORS[i]["string_feature"])
      self.assertLen(parsed_examples_dict, 2)
      for tensor_name, tensor in parsed_examples_dict.items():
        self._AssertSparseTensorEqual(
            tensor, _EXAMPLES_AS_TENSORS[i][tensor_name])

  @unittest.skipIf(not tf.executing_eagerly(), "Skip in non-eager mode.")
  def testProjectedTensorFlowDataset(self):
    tfxio = self._MakeTFXIO(_SCHEMA)
    feature_name = "string_feature"
    projected_tfxio = tfxio.Project([feature_name])
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, parsed_examples_dict in enumerate(
        projected_tfxio.TensorFlowDataset(options=options)):
      self.assertIn(feature_name, parsed_examples_dict)
      self.assertLen(parsed_examples_dict, 1)
      self._AssertSparseTensorEqual(parsed_examples_dict[feature_name],
                                    _EXAMPLES_AS_TENSORS[i][feature_name])

  @parameterized.named_parameters(*[
      dict(
          testcase_name="same_feature_name",
          schema_pbtxt="""
            feature {
              name: "string_feature"
              type: BYTES
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "string_feature"
                  value {
                    varlen_sparse_tensor {
                      column_name: "string_feature"
                    }
                  }
                }
              }
            }
          """,
          expected_parsing_config={
              "string_feature": tf.io.VarLenFeature(dtype=tf.string)
          },
          expected_rename_dict={"string_feature": "string_feature"}),
      dict(
          testcase_name="rename_one_feature",
          schema_pbtxt="""
            feature {
              name: "string_feature"
              type: BYTES
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "var_len_feature_1"
                  value {
                    varlen_sparse_tensor {
                      column_name: "string_feature"
                    }
                  }
                }
              }
            }
          """,
          expected_parsing_config={
              "string_feature": tf.io.VarLenFeature(dtype=tf.string)
          },
          expected_rename_dict={"string_feature": "var_len_feature_1"}),
      dict(
          testcase_name="sparse_feature",
          schema_pbtxt="""
            feature {
              name: "idx"
              type: INT
            }
            feature {
              name: "val"
              type: FLOAT
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "sparse_feature"
                  value {
                    sparse_tensor {
                      index_column_names: "idx"
                      value_column_name: "val"
                      dense_shape {
                        dim {
                          size: 1
                        }
                      }
                    }
                  }
                }
              }
            }
          """,
          expected_parsing_config={
              "_tfx_bsl_sparse_feature_sparse_feature":
                  tf.io.SparseFeature(
                      index_key=["idx"],
                      value_key="val",
                      size=[1],
                      dtype=tf.float32)
          },
          expected_rename_dict={
              "_tfx_bsl_sparse_feature_sparse_feature": "sparse_feature"
          }),
      dict(
          testcase_name="sparse_and_varlen_features_shared",
          schema_pbtxt="""
            feature {
              name: "idx"
              type: INT
            }
            feature {
              name: "val"
              type: FLOAT
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "sparse_feature"
                  value {
                    sparse_tensor {
                      index_column_names: "idx"
                      value_column_name: "val"
                      dense_shape {
                        dim {
                          size: 1
                        }
                      }
                    }
                  }
                }
                tensor_representation {
                  key: "varlen"
                  value {
                    varlen_sparse_tensor {
                      column_name: "val"
                    }
                  }
                }
              }
            }
          """,
          expected_parsing_config={
              "_tfx_bsl_sparse_feature_sparse_feature":
                  tf.io.SparseFeature(
                      index_key=["idx"],
                      value_key="val",
                      size=[1],
                      dtype=tf.float32),
              "val":
                  tf.io.VarLenFeature(dtype=tf.float32)
          },
          expected_rename_dict={
              "_tfx_bsl_sparse_feature_sparse_feature": "sparse_feature",
              "val": "varlen"
          }),
  ])
  def testValidGetTfExampleParserConfig(self, schema_pbtxt,
                                        expected_parsing_config,
                                        expected_rename_dict):
    schema = text_format.Parse(schema_pbtxt, schema_pb2.Schema())
    tfxio = self._MakeTFXIO(schema)

    parser_config, rename_dict = tfxio._GetTfExampleParserConfig()

    self.assertAllEqual(expected_parsing_config, parser_config)
    self.assertAllEqual(expected_rename_dict, rename_dict)

  def testValidGetTfExampleParserConfigWithRaggedFeature(self):
    schema_pbtxt = """
      feature {
        name: "row_lengths"
        type: INT
      }
      feature {
        name: "val"
        type: FLOAT
      }
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "ragged_feature"
            value {
              ragged_tensor {
                feature_path { step: "val" }
                partition { row_length: "row_lengths" }
                partition { uniform_row_length: 2 }
                row_partition_dtype: INT32
              }
            }
          }
        }
      }
    """
    schema = text_format.Parse(schema_pbtxt, schema_pb2.Schema())
    tfxio = self._MakeTFXIO(schema)

    parser_config, rename_dict = tfxio._GetTfExampleParserConfig()

    expected_parsing_config = {
        "_tfx_bsl_ragged_feature_ragged_feature":
            tf.io.RaggedFeature(
                value_key="val",
                partitions=[
                    tf.io.RaggedFeature.RowLengths("row_lengths"),
                    tf.io.RaggedFeature.UniformRowLength(2),
                ],
                dtype=tf.float32),
    }
    expected_rename_dict = {
        "_tfx_bsl_ragged_feature_ragged_feature": "ragged_feature",
    }

    self.assertAllEqual(expected_parsing_config, parser_config)
    self.assertAllEqual(expected_rename_dict, rename_dict)

  @parameterized.named_parameters(*[
      dict(
          testcase_name="invalid_duplicate_feature",
          schema_pbtxt="""
            feature {
              name: "string_feature"
              type: BYTES
              value_count {
                min: 0
                max: 2
              }
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "string_feature"
                  value {
                    varlen_sparse_tensor {
                      column_name: "string_feature"
                    }
                  }
                }
                tensor_representation {
                  key: "string_feature_2"
                  value {
                    varlen_sparse_tensor {
                      column_name: "string_feature"
                    }
                  }
                }
              }
            }
          """,
          error=ValueError,
          error_string="Unable to create a valid parsing config.*"),
      dict(
          testcase_name="sparse_and_fixed_feature",
          schema_pbtxt="""
            feature {
              name: "idx"
              type: INT
              value_count {
                min: 1
                max: 1
              }
            }
            feature {
              name: "val"
              type: FLOAT
              value_count {
                min: 1
                max: 1
              }
            }
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "sparse_feature"
                  value {
                    sparse_tensor {
                      index_column_names: "idx"
                      value_column_name: "val"
                      dense_shape {
                        dim {
                          size: 1
                        }
                      }
                    }
                  }
                }
                tensor_representation {
                  key: "fixed_feature"
                  value {
                    dense_tensor {
                      column_name: "val"
                    }
                  }
                }
              }
            }
          """,
          error=ValueError,
          error_string="Unable to create a valid parsing config.*"),
      dict(
          testcase_name="no_schema",
          schema_pbtxt="",
          error=ValueError,
          error_string="Unable to create a parsing config because no schema.*"),
  ])
  def testInvalidGetTfExampleParserConfig(self, schema_pbtxt, error,
                                          error_string):
    if not schema_pbtxt:
      schema = None
    else:
      schema = text_format.Parse(schema_pbtxt, schema_pb2.Schema())
    tfxio = self._MakeTFXIO(schema)

    with self.assertRaisesRegex(error, error_string):
      tfxio._GetTfExampleParserConfig()


class TFExampleBeamRecordTest(absltest.TestCase):

  def testE2E(self):
    raw_record_column_name = "raw_record"
    tfxio = tf_example_record.TFExampleBeamRecord(
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
      self.assertLen(dict_of_tensors, 3)
      self.assertIn("int_feature", dict_of_tensors)
      self.assertIn("float_feature", dict_of_tensors)
      self.assertIn("string_feature", dict_of_tensors)

    with beam.Pipeline() as p:
      record_batch_pcoll = (
          p
          | "CreateInMemRecords" >> beam.Create(_SERIALIZED_EXAMPLES)
          | "BeamSource" >>
          tfxio.BeamSource(batch_size=len(_SERIALIZED_EXAMPLES)))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)


if __name__ == "__main__":
  absltest.main()
