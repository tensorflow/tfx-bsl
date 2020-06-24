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
"""Tests for tfx_bsl.tfxio.tensor_to_arrow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_to_arrow
from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2


_TF_TYPE_TO_ARROW_TYPE = {
    tf.int8: pa.int8(),
    tf.int16: pa.int16(),
    tf.int32: pa.int32(),
    tf.int64: pa.int64(),
    tf.uint8: pa.uint8(),
    tf.uint16: pa.uint16(),
    tf.uint32: pa.uint32(),
    tf.uint64: pa.uint64(),
    tf.float32: pa.float32(),
    tf.float64: pa.float64(),
    tf.string: pa.binary(),
}


def _make_2d_varlen_sparse_tensor_test_cases():
  result = []
  for tf_type, arrow_type in _TF_TYPE_TO_ARROW_TYPE.items():
    if tf_type == tf.string:
      values = tf.constant([b"1", b"2", b"3"], dtype=tf.string)
      expected_array = pa.array([[b"1"], [], [b"2", b"3"], []],
                                type=pa.list_(arrow_type))
    else:
      values = tf.constant([1, 2, 3], dtype=tf_type)
      expected_array = pa.array([[1], [], [2, 3], []],
                                type=pa.list_(arrow_type))
    result.append(dict(
        testcase_name="2d_varlen_sparse_tensor_%s" % tf_type.name,
        type_specs={"sp": tf.SparseTensorSpec([None, None], tf_type)},
        expected_schema={
            "sp": pa.list_(arrow_type)
        },
        expected_tensor_representations={
            "sp": """varlen_sparse_tensor { column_name: "sp" }""",
        },
        tensor_input={
            "sp":
                tf.SparseTensor(
                    values=values,
                    indices=[[0, 0], [2, 0], [2, 1]],
                    dense_shape=[4, 2]),
        },
        expected_record_batch={
            "sp": expected_array
        }
        ))
  return result

_CONVERT_TEST_CASES = [
    dict(
        testcase_name="multiple_tensors",
        type_specs={
            "sp1": tf.SparseTensorSpec([None, None], tf.int32),
            "sp2": tf.SparseTensorSpec([None, None], tf.string),
        },
        expected_schema={
            "sp1": pa.list_(pa.int32()),
            "sp2": pa.list_(pa.binary()),
        },
        expected_tensor_representations={
            "sp1": """varlen_sparse_tensor { column_name: "sp1" }""",
            "sp2": """varlen_sparse_tensor { column_name: "sp2" }""",
        },
        tensor_input={
            "sp1":
                tf.SparseTensor(
                    values=tf.constant([1, 2], dtype=tf.int32),
                    indices=[[0, 0], [2, 0]],
                    dense_shape=[4, 1]),
            "sp2":
                tf.SparseTensor(
                    values=[b"aa", b"bb"],
                    indices=[[2, 0], [2, 1]],
                    dense_shape=[4, 2])
        },
        expected_record_batch={
            "sp1":
                pa.array([[1], [], [2], []], type=pa.list_(pa.int32())),
            "sp2":
                pa.array([[], [], [b"aa", b"bb"], []],
                         type=pa.list_(pa.binary()))
        })
] + _make_2d_varlen_sparse_tensor_test_cases()


class TensorToArrowTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_tensor_alike_equal(self, left, right):
    self.assertIsInstance(left, type(right))
    if isinstance(left, tf.SparseTensor):
      self.assertAllEqual(left.values, right.values)
      self.assertAllEqual(left.indices, right.indices)
      self.assertAllEqual(left.dense_shape, right.dense_shape)
    else:
      self.assertAllEqual(left, right)

  @parameterized.named_parameters(*_CONVERT_TEST_CASES)
  def test_convert(self, type_specs, expected_schema,
                   expected_tensor_representations, tensor_input,
                   expected_record_batch):
    converter = tensor_to_arrow.TensorsToRecordBatchConverter(type_specs)

    expected_schema = pa.schema(
        [pa.field(n, t) for n, t in sorted(expected_schema.items())])

    self.assertTrue(converter.arrow_schema().equals(expected_schema),
                    "actual: {}".format(converter.arrow_schema()))

    canonical_expected_tensor_representations = {}
    for n, r in expected_tensor_representations.items():
      if not isinstance(r, schema_pb2.TensorRepresentation):
        r = text_format.Parse(r, schema_pb2.TensorRepresentation())
      canonical_expected_tensor_representations[n] = r

    self.assertEqual(canonical_expected_tensor_representations,
                     converter.tensor_representations())

    rb = converter.convert(tensor_input)
    self.assertTrue(
        rb.equals(
            pa.record_batch(
                [arr for _, arr in sorted(expected_record_batch.items())],
                schema=expected_schema)))

    # Test that TensorAdapter(TensorsToRecordBatchConverter()) is identity.
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(
            arrow_schema=converter.arrow_schema(),
            tensor_representations=converter.tensor_representations()))
    adapter_output = adapter.ToBatchTensors(rb, produce_eager_tensors=True)
    self.assertEqual(adapter_output.keys(), tensor_input.keys())
    for k in adapter_output.keys():
      self._assert_tensor_alike_equal(adapter_output[k], tensor_input[k])

  def test_unable_to_handle(self):
    with self.assertRaisesRegex(ValueError, "No handler found"):
      tensor_to_arrow.TensorsToRecordBatchConverter(
          {"sp": tf.SparseTensorSpec([None, None, None], tf.int32)})

    with self.assertRaisesRegex(ValueError, "No handler found"):
      tensor_to_arrow.TensorsToRecordBatchConverter(
          {"sp": tf.SparseTensorSpec([None, None], tf.bool)})

  def test_incompatible_type_spec(self):
    converter = tensor_to_arrow.TensorsToRecordBatchConverter(
        {"sp": tf.SparseTensorSpec([None, None], tf.int32)})
    with self.assertRaisesRegex(TypeError, "Expected SparseTensorSpec"):
      converter.convert({
          "sp":
              tf.SparseTensor(
                  indices=[[0, 1]],
                  values=tf.constant([0], dtype=tf.int64),
                  dense_shape=[4, 1])
      })


if __name__ == "__main__":
  # Do not run these tests under TF1.x -- TensorToArrow does not support TF 1.x.
  if tf.__version__ >= "2":
    absltest.main()
