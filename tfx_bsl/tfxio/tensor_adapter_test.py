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
"""Tests for tfx_bsl.tfxio.tensor_adapter."""

import pickle

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import tensor_adapter

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2

_ALL_SUPPORTED_INT_VALUE_TYPES = [
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
]
_ALL_SUPPORTED_FLOATING_VALUE_TYPES = [pa.float32(), pa.float64()]
_ALL_SUPPORTED_STRING_VALUE_TYPES = [
    pa.binary(), pa.large_binary(),
    pa.string(), pa.large_string()
]
_ALL_SUPPORTED_VALUE_TYPES = (
    _ALL_SUPPORTED_INT_VALUE_TYPES + _ALL_SUPPORTED_FLOATING_VALUE_TYPES +
    _ALL_SUPPORTED_STRING_VALUE_TYPES)
_ARROW_TYPE_TO_TF_TYPE = {
    pa.int8(): tf.int8,
    pa.int16(): tf.int16,
    pa.int32(): tf.int32,
    pa.int64(): tf.int64,
    pa.uint8(): tf.uint8,
    pa.uint16(): tf.uint16,
    pa.uint32(): tf.uint32,
    pa.uint64(): tf.uint64,
    pa.float32(): tf.float32,
    pa.float64(): tf.float64,
    pa.binary(): tf.string,
    pa.large_binary(): tf.string,
    pa.string(): tf.string,
    pa.large_string(): tf.string,
}
_ARROW_TYPE_TO_NP_TYPE = {
    pa.int8(): np.dtype("int8"),
    pa.int16(): np.dtype("int16"),
    pa.int32(): np.dtype("int32"),
    pa.int64(): np.dtype("int64"),
    pa.uint8(): np.dtype("uint8"),
    pa.uint16(): np.dtype("uint16"),
    pa.uint32(): np.dtype("uint32"),
    pa.uint64(): np.dtype("uint64"),
    pa.float32(): np.dtype("float32"),
    pa.float64(): np.dtype("float64"),
    pa.binary(): np.dtype("object"),
    pa.large_binary(): np.dtype("object"),
    pa.string(): np.dtype("object"),
    pa.large_string(): np.dtype("object"),
}


def _Make1DSparseTensorTestCases():
  result = []
  tensor_representation_textpb = """
  sparse_tensor {
    index_column_names: ["key"]
    value_column_name: "value"
    dense_shape {
      dim {
        size: 100
      }
    }
  }
  """
  for t in _ALL_SUPPORTED_VALUE_TYPES:
    for list_type_factory in (("list", pa.list_), ("large_list",
                                                   pa.large_list)):
      expected_type_spec = tf.SparseTensorSpec([None, 100],
                                               _ARROW_TYPE_TO_TF_TYPE[t])
      if pa.types.is_integer(t):
        values = [[1, 2], None, [], [3]]
        expected_values = [1, 2, 3]
      elif pa.types.is_floating(t):
        values = [[1.0, 2.0], None, [], [3.0]]
        expected_values = [1.0, 2.0, 3.0]
      else:
        values = [[b"a", b"b"], None, [], [b"c"]]
        expected_values = [b"a", b"b", b"c"]
      indices = [[0, 99], None, [], [8]]

      if tf.executing_eagerly():
        expected_output = tf.sparse.SparseTensor(
            indices=[[0, 0], [0, 99], [3, 8]],
            values=tf.constant(
                expected_values, dtype=_ARROW_TYPE_TO_TF_TYPE[t]),
            dense_shape=(4, 100))
      else:
        expected_output = tf.compat.v1.SparseTensorValue(
            indices=[[0, 0], [0, 99], [3, 8]],
            values=np.array(expected_values, _ARROW_TYPE_TO_NP_TYPE[t]),
            dense_shape=(4, 100))

      result.append({
          "testcase_name":
              "1dsparse_tensor_{}_{}".format(t, list_type_factory[0]),
          "tensor_representation_textpb":
              tensor_representation_textpb,
          "record_batch":
              pa.RecordBatch.from_arrays([
                  pa.array(indices, type=list_type_factory[1](pa.int64())),
                  pa.array(values, type=list_type_factory[1](t))
              ], ["key", "value"]),
          "expected_output":
              expected_output,
          "expected_type_spec":
              expected_type_spec,
      })

  return result


def _MakeDenseTensorFromListArrayTestCases():
  result = []
  tensor_representation_textpb = """
  dense_tensor {
    column_name: "input"
    shape {
      dim {
        size: 4
      }
    }
  }
  """
  for t in _ALL_SUPPORTED_VALUE_TYPES:
    for list_type_factory in (("list", pa.list_), ("large_list",
                                                   pa.large_list)):
      expected_type_spec = tf.TensorSpec([None, 4], _ARROW_TYPE_TO_TF_TYPE[t])

      if pa.types.is_integer(t):
        values = [[1, 2, 3, 4], [5, 6, 7, 8]]
      elif pa.types.is_floating(t):
        values = [[1.0, 2.0, 4.0, 8.0], [-1.0, -2.0, -4.0, -8.0]]
      else:
        values = [[b"a", b"b", b"c", b"d"], [b"e", b"f", b"g", b"h"]]

      arrow_array = pa.array(values, type=list_type_factory[1](t))
      if tf.executing_eagerly():
        expected_output = tf.constant(values, dtype=_ARROW_TYPE_TO_TF_TYPE[t])
      else:
        expected_output = np.array(values, dtype=_ARROW_TYPE_TO_NP_TYPE[t])

      result.append({
          "testcase_name":
              "dense_from_{}_array_{}".format(list_type_factory[0], t),
          "tensor_representation_textpb":
              tensor_representation_textpb,
          "arrow_array":
              arrow_array,
          "expected_output":
              expected_output,
          "expected_type_spec":
              expected_type_spec,
      })

  return result


def _MakeIntDefaultFilledDenseTensorFromListArrayTestCases():
  tensor_representation_textpb = """
  dense_tensor {
    column_name: "input"
    shape {
      dim {
        size: 2
      }
      dim {
        size: 2
      }
    }
    default_value {
      int_value: 2
    }
  }
  """
  result = []
  for t in _ALL_SUPPORTED_INT_VALUE_TYPES:
    for list_type_factory in (("list", pa.list_), ("large_list",
                                                   pa.large_list)):
      arrow_array = pa.array([None, [1, 2, 3, 4], None],
                             type=list_type_factory[1](t))
      if tf.executing_eagerly():
        expected_output = tf.constant(
            [[2, 2, 2, 2], [1, 2, 3, 4], [2, 2, 2, 2]],
            dtype=_ARROW_TYPE_TO_TF_TYPE[t],
            shape=(3, 2, 2))
      else:
        expected_output = np.array([2, 2, 2, 2, 1, 2, 3, 4, 2, 2, 2, 2],
                                   dtype=_ARROW_TYPE_TO_NP_TYPE[t]).reshape(
                                       (3, 2, 2))
      result.append({
          "testcase_name":
              "default_filled_dense_from_{}_array_{}".format(
                  list_type_factory[0], t),
          "tensor_representation_textpb":
              tensor_representation_textpb,
          "arrow_array":
              arrow_array,
          "expected_output":
              expected_output,
          "expected_type_spec":
              tf.TensorSpec([None, 2, 2], _ARROW_TYPE_TO_TF_TYPE[t])
      })
  return result


def _MakeFloatingDefaultFilledDenseTensorFromListArrayTestCases():
  tensor_representation_textpb = """
  dense_tensor {
    column_name: "input"
    shape {
      dim {
        size: 2
      }
      dim {
        size: 1
      }
    }
    default_value {
      float_value: -1
    }
  }
  """
  result = []
  for t in _ALL_SUPPORTED_FLOATING_VALUE_TYPES:
    arrow_array = pa.array([None, [1, 2], None], type=pa.list_(t))
    if tf.executing_eagerly():
      expected_output = tf.constant([[-1, -1], [1, 2], [-1, -1]],
                                    dtype=_ARROW_TYPE_TO_TF_TYPE[t],
                                    shape=(3, 2, 1))
    else:
      expected_output = np.array([-1, -1, 1, 2, -1, -1],
                                 dtype=_ARROW_TYPE_TO_NP_TYPE[t]).reshape(
                                     (3, 2, 1))
    result.append({
        "testcase_name":
            "default_filled_dense_from_list_array_{}".format(t),
        "tensor_representation_textpb":
            tensor_representation_textpb,
        "arrow_array":
            arrow_array,
        "expected_output":
            expected_output,
        "expected_type_spec":
            tf.TensorSpec([None, 2, 1], dtype=_ARROW_TYPE_TO_TF_TYPE[t])
    })
  return result


def _MakeStringDefaultFilledDenseTensorFromListArrayTestCases():
  tensor_representation_textpb = """
  dense_tensor {
    column_name: "input"
    shape {
    }
    default_value {
      bytes_value: "nil"
    }
  }
  """
  result = []
  for t in _ALL_SUPPORTED_STRING_VALUE_TYPES:
    arrow_array = pa.array([None, ["hello"], None], type=pa.list_(t))
    if tf.executing_eagerly():
      expected_output = tf.constant(["nil", "hello", "nil"],
                                    dtype=_ARROW_TYPE_TO_TF_TYPE[t])
    else:
      expected_output = np.array([b"nil", b"hello", b"nil"],
                                 dtype=_ARROW_TYPE_TO_NP_TYPE[t])
    result.append({
        "testcase_name": "default_filled_dense_from_list_array_{}".format(t),
        "tensor_representation_textpb": tensor_representation_textpb,
        "arrow_array": arrow_array,
        "expected_output": expected_output,
        "expected_type_spec": tf.TensorSpec([None], _ARROW_TYPE_TO_TF_TYPE[t])
    })
  return result


def _MakeVarLenSparseTensorFromListArrayTestCases():
  tensor_representation_textpb = """
  varlen_sparse_tensor {
    column_name: "input"
  }
  """
  result = []
  for t in _ALL_SUPPORTED_VALUE_TYPES:
    if pa.types.is_integer(t):
      values = [[1, 2], None, [3], [], [5]]
      expected_values = [1, 2, 3, 5]
    elif pa.types.is_floating(t):
      values = [[1.0, 2.0], None, [3.0], [], [5.0]]
      expected_values = [1.0, 2.0, 3.0, 5.0]
    else:
      values = [["a", "b"], None, ["c"], [], ["d"]]
      expected_values = [b"a", b"b", b"c", b"d"]
    expected_sparse_indices = [[0, 0], [0, 1], [2, 0], [4, 0]]
    expected_dense_shape = [5, 2]
    expected_output = tf.compat.v1.SparseTensorValue(
        indices=np.array(expected_sparse_indices, dtype=np.int64),
        dense_shape=np.array(expected_dense_shape, dtype=np.int64),
        values=np.array(expected_values, dtype=_ARROW_TYPE_TO_NP_TYPE[t]))
    result.append({
        "testcase_name":
            "varlen_sparse_from_list_array_{}".format(t),
        "tensor_representation_textpb":
            tensor_representation_textpb,
        "arrow_array":
            pa.array(values, type=pa.list_(t)),
        "expected_output":
            expected_output,
        "expected_type_spec":
            tf.SparseTensorSpec(
                tf.TensorShape([None, None]), _ARROW_TYPE_TO_TF_TYPE[t])
    })

  return result


_ONE_TENSOR_TEST_CASES = (
    _MakeDenseTensorFromListArrayTestCases() +
    _MakeIntDefaultFilledDenseTensorFromListArrayTestCases() +
    _MakeFloatingDefaultFilledDenseTensorFromListArrayTestCases() +
    _MakeStringDefaultFilledDenseTensorFromListArrayTestCases() +
    _MakeVarLenSparseTensorFromListArrayTestCases())


def _MakeRaggedTensorDTypesTestCases():
  result = []
  tensor_representation_textpb = """
  ragged_tensor {
    feature_path {
      step: "ragged_feature"
    }
  }
  """
  for t in _ALL_SUPPORTED_VALUE_TYPES:
    for list_type_factory in (("list", pa.list_), ("large_list",
                                                   pa.large_list)):
      expected_type_spec = tf.RaggedTensorSpec([None, None],
                                               _ARROW_TYPE_TO_TF_TYPE[t],
                                               ragged_rank=1,
                                               row_splits_dtype=tf.int64)
      if pa.types.is_integer(t):
        values = [[1, 2], None, [], [3]]
        expected_values = [1, 2, 3]
      elif pa.types.is_floating(t):
        values = [[1.0, 2.0], None, [], [3.0]]
        expected_values = [1.0, 2.0, 3.0]
      else:
        values = [[b"a", b"b"], None, [], [b"c"]]
        expected_values = [b"a", b"b", b"c"]
      row_splits = np.asarray([0, 2, 2, 2, 3], dtype=np.int64)

      if tf.executing_eagerly():
        expected_output = tf.RaggedTensor.from_row_splits(
            values=tf.constant(
                expected_values, dtype=_ARROW_TYPE_TO_TF_TYPE[t]),
            row_splits=row_splits)
      else:
        expected_output = tf.compat.v1.ragged.RaggedTensorValue(
            values=np.array(expected_values, _ARROW_TYPE_TO_NP_TYPE[t]),
            row_splits=row_splits)

      result.append({
          "testcase_name":
              "1D_{}_{}".format(t, list_type_factory[0]),
          "tensor_representation_textpb":
              tensor_representation_textpb,
          "record_batch":
              pa.RecordBatch.from_arrays(
                  [pa.array(values, type=list_type_factory[1](t))],
                  ["ragged_feature"]),
          "expected_ragged_tensor":
              expected_output,
          "expected_type_spec":
              expected_type_spec,
      })

  return result


_RAGGED_TENSOR_TEST_CASES = (
    _MakeRaggedTensorDTypesTestCases() + [
        dict(
            testcase_name="Simple",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1], None, [2], [3, 4, 5], []], type=pa.list_(pa.int64()))
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([1, 2, 3, 4, 5]),
                row_splits=np.asarray([0, 1, 1, 2, 5, 5]))),
        dict(
            testcase_name="3D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[[1]], None, [[2]], [[3, 4], [5]], []],
                         type=pa.list_(pa.list_(pa.int64())))
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None]),
                tf.int64,
                ragged_rank=2,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.asarray([1, 2, 3, 4, 5]),
                    row_splits=np.asarray([0, 1, 2, 4, 5])),
                row_splits=np.asarray([0, 1, 1, 2, 4, 4]))),
        dict(
            testcase_name="StructOfList",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.StructArray.from_arrays([
                    pa.array([[1, 2, 3], [4]], pa.list_(pa.int64())),
                    pa.array([["a", "b", "c"], ["d"]], pa.list_(pa.binary()))
                ], ["inner_feature", "x2"])
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([1, 2, 3, 4]),
                row_splits=np.asarray([0, 3, 4]))),
        dict(
            testcase_name="ListOfStruct",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[{
                    "inner_feature": 1,
                    "x2": "a"
                }, {
                    "inner_feature": 2,
                    "x2": "b"
                }], [{
                    "inner_feature": 3,
                    "x2": "c"
                }]],
                         pa.list_(
                             pa.struct([("inner_feature", pa.int64()),
                                        ("x2", pa.binary())])))
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([1, 2, 3]), row_splits=np.asarray([0, 2, 3
                                                                    ]))),
        dict(
            testcase_name="NestedStructList",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature_1"
            step: "inner_feature_2"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.StructArray.from_arrays([
                    pa.array([[{
                        "inner_feature_2": "x",
                        "x2": "a"
                    }, {
                        "inner_feature_2": "y",
                        "x2": "b"
                    }], [{
                        "inner_feature_2": "z",
                        "x2": "c"
                    }]],
                             pa.list_(
                                 pa.struct([("inner_feature_2", pa.binary()),
                                            ("x3", pa.binary())]))),
                    pa.array([["a", "b", "c"], ["d"]], pa.list_(pa.binary()))
                ], ["inner_feature_1", "x2"])
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None]),
                tf.string,
                ragged_rank=1,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([b"x", b"y", b"z"]),
                row_splits=np.asarray([0, 2, 3]))),
        dict(
            testcase_name="ListStructStruct",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature_1"
            step: "inner_feature_2"
          }
          row_partition_dtype: INT32
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[{
                    "inner_feature_1": {
                        "inner_feature_2": 1,
                        "x2": "a"
                    }
                }, {
                    "inner_feature_1": {
                        "inner_feature_2": 2,
                        "x2": "b"
                    }
                }], [{
                    "inner_feature_1": {
                        "inner_feature_2": 3,
                        "x2": "c"
                    }
                }]],
                         pa.list_(
                             pa.struct([("inner_feature_1",
                                         pa.struct([
                                             ("inner_feature_2", pa.int64()),
                                             ("x2", pa.binary()),
                                         ]))])))
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int32),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([1, 2, 3]), row_splits=np.asarray([0, 2, 3
                                                                    ]))),
        dict(
            testcase_name="MixedLargeTypes",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[[1]], None, [[2]], [[3, 4], [5]], []],
                         type=pa.list_(pa.large_list(pa.int64())))
            ], ["ragged_feature"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None]),
                tf.int64,
                ragged_rank=2,
                row_splits_dtype=tf.int64),
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.asarray([1, 2, 3, 4, 5]),
                    row_splits=np.asarray([0, 1, 2, 4, 5])),
                row_splits=np.asarray([0, 1, 1, 2, 4, 4]))),
        dict(
            testcase_name="RaggedRank1Uniform1D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 2, 3, 4], None, [], [5, 6]],
                         type=pa.list_(pa.int64()))
            ], ["value"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, 2]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [1, 2],
            #    [3, 4],
            #  ],
            #  [],
            #  [],
            #  [
            #    [5, 6],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([[1, 2], [3, 4], [5, 6]]),
                row_splits=np.asarray([0, 2, 2, 2, 3])),
        ),
        dict(
            testcase_name="RaggedRank1Uniform2D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { uniform_row_length: 2 }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 1, 2, 2, 3, 3, 4, 4], None, [], [5, 5, 6, 6]],
                         type=pa.list_(pa.int64()))
            ], ["value"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, 2, 2]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [[1, 1], [2, 2]],
            #    [[3, 3], [4, 4]],
            #  ],
            #  [],
            #  [],
            #  [
            #    [[5, 5], [6, 6]],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([[[1, 1], [2, 2]], [[3, 3], [4, 4]],
                                   [[5, 5], [6, 6]]]),
                row_splits=np.asarray([0, 2, 2, 2, 3])),
        ),
        dict(
            testcase_name="RaggedRank2",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { row_length: "row_length" }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6]],
                         type=pa.list_(pa.int64())),
                pa.array([[4, 2], None, [], [1, 1]], type=pa.list_(pa.int64())),
            ], ["value", "row_length"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None]),
                tf.int64,
                ragged_rank=2,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [1, 2, 3, 4],
            #    [5, 6],
            #  ],
            #  [],
            #  [],
            #  [
            #    [5],
            #    [6],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.asarray([1, 2, 3, 4, 5, 6, 5, 6]),
                    row_splits=np.asarray([0, 4, 6, 7, 8])),
                row_splits=np.asarray([0, 2, 2, 2, 4])),
        ),
        dict(
            testcase_name="RaggedRank2Uniform1D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { row_length: "row_length" }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6]],
                         type=pa.list_(pa.int64())),
                pa.array([[2, 1], None, [], [1]], type=pa.list_(pa.int64())),
            ], ["value", "row_length"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None, 2]),
                tf.int64,
                ragged_rank=2,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [
            #      [1, 2],
            #      [3, 4],
            #    ],
            #    [
            #      [5, 6],
            #    ],
            #  ],
            #  [],
            #  [],
            #  [
            #    [
            #      [5, 6],
            #    ],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.asarray([[1, 2], [3, 4], [5, 6], [5, 6]]),
                    row_splits=np.asarray([0, 2, 3, 4])),
                row_splits=np.asarray([0, 2, 2, 2, 3])),
        ),
        dict(
            testcase_name="RaggedRank3UniformRaggedUniform",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { uniform_row_length: 2 }
          partition { row_length: "row_length" }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6, 7, 8]],
                         type=pa.list_(pa.int64())),
                pa.array([[2, 1], None, [], [1, 1]], type=pa.list_(pa.int64())),
            ], ["value", "row_length"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None, None, 2]),
                tf.int64,
                ragged_rank=3,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [
            #      [
            #        [1, 2],
            #        [3, 4],
            #      ],
            #      [
            #        [5, 6],
            #      ],
            #    ],
            #  ],
            #  [],
            #  [],
            #  [
            #    [
            #      [
            #        [5, 6],
            #      ],
            #      [
            #        [7, 8],
            #      ],
            #    ],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=tf.compat.v1.ragged.RaggedTensorValue(
                        values=np.asarray([[1, 2], [3, 4], [5, 6], [5, 6],
                                           [7, 8]]),
                        row_splits=np.asarray([0, 2, 3, 4, 5])),
                    row_splits=np.asarray([0, 2, 4])),
                row_splits=np.asarray([0, 1, 1, 1, 2])),
        ),
        dict(
            testcase_name="RaggedRank4RaggedUniformRagged",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path { step: "value" }
          partition { row_length: "row_length_2" }
          partition { uniform_row_length: 2 }
          partition { row_length: "row_length" }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6, 7, 8]],
                         type=pa.list_(pa.int64())),
                pa.array([[2, 1], None, [], [1, 1]], type=pa.list_(pa.int64())),
                pa.array([[1], None, [], [1]], type=pa.list_(pa.int64())),
            ], ["value", "row_length", "row_length_2"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None, None, None, 2]),
                tf.int64,
                ragged_rank=4,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [
            #      [
            #        [
            #          [1, 2],
            #          [3, 4],
            #        ],
            #        [
            #          [5, 6],
            #        ],
            #      ],
            #    ],
            #  ],
            #  [],
            #  [],
            #  [
            #    [
            #      [
            #        [
            #          [5, 6],
            #        ],
            #        [
            #          [7, 8],
            #        ],
            #      ],
            #    ],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=tf.compat.v1.ragged.RaggedTensorValue(
                        values=tf.compat.v1.ragged.RaggedTensorValue(
                            values=np.asarray([[1, 2], [3, 4], [5, 6], [5, 6],
                                               [7, 8]]),
                            row_splits=np.asarray([0, 2, 3, 4, 5])),
                        row_splits=np.asarray([0, 2, 4])),
                    row_splits=np.asarray([0, 1, 2])),
                row_splits=np.asarray([0, 1, 1, 1, 2])),
        ),
        dict(
            testcase_name="Struct_RaggedRank1Uniform1D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "parent"
            step: "value"
          }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.StructArray.from_arrays([
                    pa.array([[1, 2, 3, 4], None, [], [5, 6]],
                             type=pa.list_(pa.int64()))
                ], ["value"])
            ], ["parent"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, 2]),
                tf.int64,
                ragged_rank=1,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [1, 2],
            #    [3, 4],
            #  ],
            #  [],
            #  [],
            #  [
            #    [5, 6],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=np.asarray([[1, 2], [3, 4], [5, 6]]),
                row_splits=np.asarray([0, 2, 2, 2, 3])),
        ),
        dict(
            testcase_name="Struct_RaggedRank2",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "parent"
            step: "value"
          }
          partition { row_length: "row_length" }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.StructArray.from_arrays([
                    pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6]],
                             type=pa.list_(pa.int64())),
                    pa.array([[4, 2], None, [], [1, 1]], type=pa.list_(pa.int64())),
                ], ["value", "row_length"])
            ], ["parent"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None]),
                tf.int64,
                ragged_rank=2,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [1, 2, 3, 4],
            #    [5, 6],
            #  ],
            #  [],
            #  [],
            #  [
            #    [5],
            #    [6],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=np.asarray([1, 2, 3, 4, 5, 6, 5, 6]),
                    row_splits=np.asarray([0, 4, 6, 7, 8])),
                row_splits=np.asarray([0, 2, 2, 2, 4])),
        ),
        dict(
            testcase_name="Struct_RaggedRank4RaggedUniformRagged",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "parent"
            step: "value"
          }
          partition { row_length: "row_length_2" }
          partition { uniform_row_length: 2 }
          partition { row_length: "row_length" }
          partition { uniform_row_length: 2 }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.StructArray.from_arrays([
                    pa.array([[1, 2, 3, 4, 5, 6], None, [], [5, 6, 7, 8]],
                             type=pa.list_(pa.int64())),
                    pa.array([[2, 1], None, [], [1, 1]], type=pa.list_(pa.int64())),
                    pa.array([[1], None, [], [1]], type=pa.list_(pa.int64())),
                ], ["value", "row_length", "row_length_2"]),
            ], ["parent"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None, None, None, 2]),
                tf.int64,
                ragged_rank=4,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [
            #      [
            #        [
            #          [1, 2],
            #          [3, 4],
            #        ],
            #        [
            #          [5, 6],
            #        ],
            #      ],
            #    ],
            #  ],
            #  [],
            #  [],
            #  [
            #    [
            #      [
            #        [
            #          [5, 6],
            #        ],
            #        [
            #          [7, 8],
            #        ],
            #      ],
            #    ],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=tf.compat.v1.ragged.RaggedTensorValue(
                        values=tf.compat.v1.ragged.RaggedTensorValue(
                            values=np.asarray([[1, 2], [3, 4], [5, 6], [5, 6],
                                               [7, 8]]),
                            row_splits=np.asarray([0, 2, 3, 4, 5])),
                        row_splits=np.asarray([0, 2, 4])),
                    row_splits=np.asarray([0, 1, 2])),
                row_splits=np.asarray([0, 1, 1, 1, 2])),
        ),
        dict(
            testcase_name="ListStruct_RaggedRank1Uniform1D",
            tensor_representation_textpb="""
        ragged_tensor {
          feature_path {
            step: "parent"
            step: "struct"
            step: "value"
          }
          partition { row_length: "row_length" }
          row_partition_dtype: INT64
        }
        """,
            record_batch=pa.RecordBatch.from_arrays([
                pa.array([
                    [
                        {
                            "struct": {
                                "value": [1, 2, 3],
                                "row_length": [2, 1],
                            }
                        },
                        {
                            "struct": {
                                "value": [1],
                                "row_length": [1],
                            }
                        },
                    ],
                    None,
                    [],
                    [
                        {
                            "struct": {
                                "value": [2, 3, 4],
                                "row_length": [1, 2],
                            }
                        },
                    ],
                ],
                         pa.list_(
                             pa.struct([("struct",
                                         pa.struct([
                                             ("value", pa.list_(pa.int64())),
                                             ("row_length", pa.list_(pa.int64()))
                                         ]))])))
            ], ["parent"]),
            expected_type_spec=tf.RaggedTensorSpec(
                tf.TensorShape([None, None, None, None]),
                tf.int64,
                ragged_rank=3,
                row_splits_dtype=tf.int64),
            # expected: [
            #  [
            #    [
            #      [1, 2],
            #      [3],
            #    ],
            #    [
            #      [1],
            #    ],
            #  ],
            #  [],
            #  [],
            #  [
            #    [
            #      [2],
            #      [3, 4],
            #    ],
            #  ],
            # ]
            expected_ragged_tensor=tf.compat.v1.ragged.RaggedTensorValue(
                values=tf.compat.v1.ragged.RaggedTensorValue(
                    values=tf.compat.v1.ragged.RaggedTensorValue(
                        values=np.asarray([1, 2, 3, 1, 2, 3, 4]),
                        row_splits=np.asarray([0, 2, 3, 4, 5, 7])),
                    row_splits=np.asarray([0, 2, 3, 5])),
                row_splits=np.asarray([0, 2, 2, 2, 3]),
            ),
        ),
    ])

_INVALID_DEFAULT_VALUE_TEST_CASES = [
    dict(
        testcase_name="default_value_not_set",
        value_type=pa.int64(),
        default_value_pbtxt="",
        exception_regexp="Incompatible default value"),
    dict(
        testcase_name="mismatch_type",
        value_type=pa.binary(),
        default_value_pbtxt="float_value: 1.0",
        exception_regexp="Incompatible default value",
    ),
    dict(
        testcase_name="integer_out_of_range_int64_uint64max",
        value_type=pa.int64(),
        default_value_pbtxt="uint_value: 0xffffffffffffffff",
        exception_regexp="Integer default value out of range",
    ),
    dict(
        testcase_name="integer_out_of_range_int32_int64max",
        value_type=pa.int32(),
        default_value_pbtxt="int_value: 0x7fffffffffffffff",
        exception_regexp="Integer default value out of range",
    ),
]

_INVALID_SPARSE_TENSOR_TEST_CASES = [
    dict(
        testcase_name="dense_rank_not_equal_num_index_columns",
        tensor_representation_textpb="""
         sparse_tensor {
           index_column_names: ["key"]
           value_column_name: "value"
           dense_shape {
             dim {
               size: 10
             }
             dim {
               size: 5
             }
           }
         }
         """,
        arrow_schema={
            "key": pa.list_(pa.int64()),
            "value": pa.list_(pa.int64()),
        }),
    dict(
        testcase_name="invalid_index_column_type",
        tensor_representation_textpb="""
         sparse_tensor {
           index_column_names: ["key"]
           value_column_name: "value"
           dense_shape {
             dim {
               size: 10
             }
           }
         }
         """,
        arrow_schema={
            "key": pa.list_(pa.float32()),
            "value": pa.list_(pa.int64()),
        }),
]


class TensorAdapterTest(parameterized.TestCase, tf.test.TestCase):

  def assertSparseAllEqual(self, a, b):
    self.assertAllEqual(a.indices, b.indices)
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.dense_shape, b.dense_shape)

  def assertRaggedAllEqual(self, a, b):
    self.assertAllEqual(a.values, b.values)
    self.assertAllEqual(a.row_splits, b.row_splits)

  def assertNonEager(self, v):
    """Asserts `v` is not a Eager tensor."""
    self.assertIsInstance(v, (tf.compat.v1.ragged.RaggedTensorValue,
                              tf.compat.v1.SparseTensorValue, np.ndarray))

  def assertAdapterCanProduceNonEagerInEagerMode(self, adapter, record_batch):
    if tf.executing_eagerly():
      converted_non_eager = adapter.ToBatchTensors(
          record_batch, produce_eager_tensors=False)
      for v in converted_non_eager.values():
        self.assertNonEager(v)

  @test_util.deprecated_graph_mode_only
  def testRaiseOnRequestingEagerTensorsInGraphMode(self):
    tensor_representation = text_format.Parse(
        """
  sparse_tensor {
    index_column_names: ["key"]
    value_column_name: "value"
    dense_shape {
      dim {
        size: 100
      }
    }
  }
  """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([[1]]), pa.array([[2]])], ["key", "value"])
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    with self.assertRaisesRegex(RuntimeError, "eager mode was not enabled"):
      adapter.ToBatchTensors(record_batch, produce_eager_tensors=True)

  @parameterized.named_parameters(*_ONE_TENSOR_TEST_CASES)
  @test_util.run_in_graph_and_eager_modes
  def testOneTensorFromOneColumn(self, tensor_representation_textpb,
                                 arrow_array, expected_type_spec,
                                 expected_output):

    tensor_representation = text_format.Parse(tensor_representation_textpb,
                                              schema_pb2.TensorRepresentation())
    column_name = None
    if tensor_representation.HasField("dense_tensor"):
      column_name = tensor_representation.dense_tensor.column_name
    if tensor_representation.HasField("varlen_sparse_tensor"):
      column_name = tensor_representation.varlen_sparse_tensor.column_name

    record_batch = pa.RecordBatch.from_arrays([arrow_array], [column_name])
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    self.assertEqual(expected_type_spec, adapter.TypeSpecs()["output"])
    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 1)
    self.assertIn("output", converted)
    actual_output = converted["output"]
    if tf.executing_eagerly():
      self.assertTrue(
          expected_type_spec.is_compatible_with(actual_output),
          "{} is not compatible with spec {}".format(actual_output,
                                                     expected_type_spec))
    if isinstance(expected_output,
                  (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
      self.assertIsInstance(actual_output,
                            (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
      self.assertSparseAllEqual(expected_output, actual_output)
    else:
      self.assertAllEqual(expected_output, actual_output)

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @test_util.run_in_graph_and_eager_modes
  def test0DSparseTensor(self):
    values = [[1.0], None, [], [3.0]]
    expected_values = [1.0, 3.0]
    indices = [[99], None, [], [8]]

    if tf.executing_eagerly():
      expected_output = tf.sparse.SparseTensor(
          indices=[[0], [3]],
          values=tf.constant(expected_values, dtype=tf.float32),
          dense_shape=(4,))
    else:
      expected_output = tf.compat.v1.SparseTensorValue(
          indices=[[0], [3]],
          values=np.array(expected_values, np.float32),
          dense_shape=(4,))

    record_batch = pa.RecordBatch.from_arrays([
        pa.array(indices, type=("list", pa.list_)[1](pa.int64())),
        pa.array(values, type=("list", pa.list_)[1](pa.float32()))
    ], ["key", "value"])
    tensor_representation = text_format.Parse(
        """
              sparse_tensor {
                value_column_name: "value"
                dense_shape { }
              }
            """, schema_pb2.TensorRepresentation())
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))

    expected_type_spec = tf.SparseTensorSpec([None], tf.float32)

    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 1)
    self.assertIn("output", converted)
    actual_output = converted["output"]
    self.assertIsInstance(actual_output,
                          (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
    if tf.executing_eagerly():
      self.assertTrue(
          expected_type_spec.is_compatible_with(actual_output),
          "{} is not compatible with spec {}".format(actual_output,
                                                     expected_type_spec))

    self.assertSparseAllEqual(expected_output, actual_output)

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @parameterized.named_parameters(*_Make1DSparseTensorTestCases())
  @test_util.run_in_graph_and_eager_modes
  def test1DSparseTensor(self, tensor_representation_textpb, record_batch,
                         expected_type_spec, expected_output):
    tensor_representation = text_format.Parse(tensor_representation_textpb,
                                              schema_pb2.TensorRepresentation())
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 1)
    self.assertIn("output", converted)
    actual_output = converted["output"]
    self.assertIsInstance(actual_output,
                          (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
    if tf.executing_eagerly():
      self.assertTrue(
          expected_type_spec.is_compatible_with(actual_output),
          "{} is not compatible with spec {}".format(actual_output,
                                                     expected_type_spec))

    self.assertSparseAllEqual(expected_output, actual_output)

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @parameterized.named_parameters(
      dict(testcase_name="known_dense_shape", is_dense_shape_known=True),
      dict(testcase_name="unknown_dense_shape", is_dense_shape_known=False))
  @test_util.run_in_graph_and_eager_modes
  def test2DSparseTensor(self, is_dense_shape_known):
    dense_shape = [10, 20] if is_dense_shape_known else [-1, -1]
    tensor_representation = text_format.Parse(
        f"""
        sparse_tensor {{
          value_column_name: "values"
          index_column_names: ["d0", "d1"]
          dense_shape {{
            dim {{
              size: {dense_shape[0]}
            }}
            dim {{
              size: {dense_shape[1]}
            }}
          }}
        }}
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays(
        [
            pa.array([[1], None, [2], [3, 4, 5], []], type=pa.list_(
                pa.int64())),
            # Also test that the index column can be of an integral type other
            # than int64.
            pa.array([[9], None, [9], [7, 8, 9], []],
                     type=pa.list_(pa.uint32())),
            pa.array([[0], None, [0], [0, 1, 2], []], type=pa.list_(pa.int64()))
        ],
        ["values", "d0", "d1"])
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 1)
    self.assertIn("output", converted)
    actual_output = converted["output"]
    self.assertIsInstance(actual_output,
                          (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
    self.assertSparseAllEqual(
        tf.compat.v1.SparseTensorValue(
            dense_shape=[5]+dense_shape,
            indices=[[0, 9, 0], [2, 9, 0], [3, 7, 0], [3, 8, 1], [3, 9, 2]],
            values=tf.convert_to_tensor([1, 2, 3, 4, 5], dtype=tf.int64)),
        actual_output)

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @test_util.run_in_graph_and_eager_modes
  def testSparseTensorsReferSameColumns(self):
    tensor_representation1 = text_format.Parse(
        """
        sparse_tensor {
          value_column_name: "values"
          index_column_names: ["d0", "d1"]
          dense_shape {
            dim {
              size: 10
            }
            dim {
              size: 20
            }
          }
        }
        """, schema_pb2.TensorRepresentation())
    tensor_representation2 = text_format.Parse(
        """
        sparse_tensor {
          value_column_name: "d1"
          index_column_names: ["d0"]
          dense_shape {
            dim {
              size: 10
            }
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], None, [2], [3, 4, 5], []], type=pa.list_(pa.int64())),
        pa.array([[9], None, [9], [7, 8, 9], []], type=pa.list_(pa.int64())),
        pa.array([[0], None, [0], [0, 1, 2], []], type=pa.list_(pa.int64()))
    ], ["values", "d0", "d1"])
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema, {
            "output1": tensor_representation1,
            "output2": tensor_representation2
        }))
    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 2)
    self.assertIn("output1", converted)
    self.assertIn("output2", converted)
    actual_output1 = converted["output1"]
    actual_output2 = converted["output2"]
    self.assertIsInstance(actual_output1,
                          (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
    self.assertSparseAllEqual(
        tf.compat.v1.SparseTensorValue(
            dense_shape=[5, 10, 20],
            indices=[[0, 9, 0], [2, 9, 0], [3, 7, 0], [3, 8, 1], [3, 9, 2]],
            values=tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)),
        actual_output1)
    self.assertIsInstance(actual_output2,
                          (tf.SparseTensor, tf.compat.v1.SparseTensorValue))
    self.assertSparseAllEqual(
        tf.compat.v1.SparseTensorValue(
            dense_shape=[5, 10],
            indices=[[0, 9], [2, 9], [3, 7], [3, 8], [3, 9]],
            values=tf.constant([0, 0, 0, 1, 2], dtype=tf.int64)),
        actual_output2)

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @parameterized.named_parameters(_RAGGED_TENSOR_TEST_CASES)
  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensor(self, tensor_representation_textpb, record_batch,
                       expected_type_spec, expected_ragged_tensor):
    tensor_representation = text_format.Parse(tensor_representation_textpb,
                                              schema_pb2.TensorRepresentation())
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    converted = adapter.ToBatchTensors(record_batch)
    self.assertLen(converted, 1)
    self.assertIn("output", converted)
    actual_output = converted["output"]
    self.assertIsInstance(
        actual_output, (tf.RaggedTensor, tf.compat.v1.ragged.RaggedTensorValue))
    if tf.executing_eagerly():
      self.assertEqual(adapter.TypeSpecs()["output"], expected_type_spec)
      self.assertTrue(
          expected_type_spec.is_compatible_with(actual_output),
          "{} is not compatible with spec {}".format(actual_output,
                                                     expected_type_spec))

    print("actual:", actual_output)
    print("expected:", expected_ragged_tensor)
    self.assertRaggedAllEqual(actual_output, expected_ragged_tensor)
    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorFromStructArrayWithNoNestedness(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature"
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.StructArray.from_arrays(
            [pa.array([1, 2, 3]),
             pa.array(["a", "b", "c"])], ["inner_feature", "x2"])
    ], ["ragged_feature"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorStructTypeInvalidSteps(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "wrong_step"
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.StructArray.from_arrays([
            pa.array([[1, 2, 3]], pa.list_(pa.int64())),
            pa.array([["a", "b", "c"]], pa.list_(pa.binary()))
        ], ["inner_feature", "x2"])
    ], ["ragged_feature"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorStructTypeTooManySteps(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
            step: "inner_feature"
            step: "non_existant_feature"
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.StructArray.from_arrays([
            pa.array([[1, 2, 3]], pa.list_(pa.int64())),
            pa.array([["a", "b", "c"]], pa.list_(pa.binary()))
        ], ["inner_feature", "x2"])
    ], ["ragged_feature"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorStructTypeNonLeaf(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.StructArray.from_arrays([
            pa.array([[1, 2, 3]], pa.list_(pa.int64())),
            pa.array([["a", "b", "c"]], pa.list_(pa.binary()))
        ], ["inner_feature", "x2"])
    ], ["ragged_feature"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorWithoutSamePathForPartition(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "struct"
            step: "ragged_feature"
          }
          partition { row_length: "row_length" }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.StructArray.from_arrays(
            [pa.array([[1, 2, 3]], pa.list_(pa.int64()))], ["ragged_feature"]),
        pa.array([[1, 2]], pa.list_(pa.int64()))
    ], ["struct", "row_length"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorWithNestedRowLengths(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
          partition { row_length: "row_length" }
          row_partition_dtype: INT64
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[[1, 1]], None, [[2]], [[3, 3, 4], [5]], []],
                 type=pa.list_(pa.large_list(pa.int64()))),
        pa.array([[[2]], None, [[1]], [[2, 1], [1]], []],
                 type=pa.list_(pa.large_list(pa.int64()))),
    ], ["ragged_feature", "row_length"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorWithEmptyFeaturePath(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path { }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[[1, 1]], None, [[2]], [[3, 3, 4], [5]], []],
                 type=pa.list_(pa.large_list(pa.int64()))),
    ], ["ragged_feature"])
    with self.assertRaisesRegex(ValueError,
                                ".*Unable to handle tensor output.*"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                             {"output": tensor_representation}))

  @test_util.run_in_graph_and_eager_modes
  def testRaggedTensorSlicedRecordBatch(self):
    tensor_representation = text_format.Parse(
        """
        ragged_tensor {
          feature_path {
            step: "ragged_feature"
          }
        }
        """, schema_pb2.TensorRepresentation())
    record_batch = pa.RecordBatch.from_arrays(
        [pa.array([[1], None, [2], [3, 4, 5], []], type=pa.list_(pa.int64()))],
        ["ragged_feature"])
    record_batch = record_batch.slice(1, 3)
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           {"output": tensor_representation}))
    with self.assertRaisesRegex(
        ValueError,
        ".*Error raised when handling tensor 'output'"):
      adapter.ToBatchTensors(record_batch)

  @test_util.run_in_graph_and_eager_modes
  def testMultipleColumns(self):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [], [2, 3], None], type=pa.large_list(pa.int64())),
        pa.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
                 type=pa.list_(pa.float32())),
        pa.array([None, [b"a", b"b"], [b"c", b"d"], None],
                 type=pa.list_(pa.large_binary())),
        pa.array([[b"w"], [b"x"], [b"y"], [b"z"]], type=pa.list_(pa.string())),
    ], [
        "int64_ragged",
        "float_dense",
        "bytes_ragged",
        "bytes_dense",
    ])

    tensor_representations = {
        "int64_varlen_sparse":
            text_format.Parse(
                """
        varlen_sparse_tensor {
          column_name: "int64_ragged"
        }
        """, schema_pb2.TensorRepresentation()),
        "float_dense":
            text_format.Parse(
                """
        dense_tensor {
          column_name: "float_dense"
          shape {
            dim {
              size: 2
            }
            dim {
              size: 1
            }
          }
        }""", schema_pb2.TensorRepresentation()),
        "bytes_varlen_sparse":
            text_format.Parse(
                """
        varlen_sparse_tensor {
          column_name: "bytes_ragged"
        }
        """, schema_pb2.TensorRepresentation()),
        "bytes_dense":
            text_format.Parse(
                """
        dense_tensor {
          column_name: "bytes_dense"
          shape {
          }
        }
        """, schema_pb2.TensorRepresentation()),
        "bytes_default_filled_dense":
            text_format.Parse(
                """
        dense_tensor {
          column_name: "bytes_ragged"
          shape {
            dim {
              size: 2
            }
          }
          default_value {
            bytes_value: "kk"
          }
        }
        """, schema_pb2.TensorRepresentation()),
    }

    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(record_batch.schema,
                                           tensor_representations))
    type_specs = adapter.TypeSpecs()
    self.assertEqual(
        type_specs, {
            "int64_varlen_sparse":
                tf.SparseTensorSpec(shape=[None, None], dtype=tf.int64),
            "bytes_varlen_sparse":
                tf.SparseTensorSpec(shape=[None, None], dtype=tf.string),
            "float_dense":
                tf.TensorSpec(shape=[None, 2, 1], dtype=tf.float32),
            "bytes_dense":
                tf.TensorSpec(shape=[None], dtype=tf.string),
            "bytes_default_filled_dense":
                tf.TensorSpec(shape=[None, 2], dtype=tf.string),
        })

    tensors = adapter.ToBatchTensors(record_batch)
    self.assertLen(tensors, len(type_specs))
    self.assertSparseAllEqual(
        tf.SparseTensor(
            values=tf.constant([1, 2, 3], dtype=tf.int64),
            dense_shape=tf.constant([4, 2], dtype=tf.int64),
            indices=tf.constant([[0, 0], [2, 0], [2, 1]], dtype=tf.int64)),
        tensors["int64_varlen_sparse"])
    self.assertSparseAllEqual(
        tf.SparseTensor(
            values=tf.constant([b"a", b"b", b"c", b"d"]),
            dense_shape=tf.constant([4, 2], dtype=tf.int64),
            indices=tf.constant([[1, 0], [1, 1], [2, 0], [2, 1]],
                                dtype=tf.int64)),
        tensors["bytes_varlen_sparse"])
    self.assertAllEqual(
        tf.constant(
            [[[1.0], [2.0]], [[2.0], [3.0]], [[3.0], [4.0]], [[4.0], [5.0]]],
            dtype=tf.float32), tensors["float_dense"])
    self.assertAllEqual(
        tf.constant([b"w", b"x", b"y", b"z"]), tensors["bytes_dense"])
    self.assertAllEqual(
        tf.constant([[b"kk", b"kk"], [b"a", b"b"], [b"c", b"d"],
                     [b"kk", b"kk"]]), tensors["bytes_default_filled_dense"])

    if tf.executing_eagerly():
      for name, spec in type_specs.items():
        self.assertTrue(
            spec.is_compatible_with(tensors[name]),
            "{} is not compatible with spec {}".format(tensors[name], spec))

    self.assertAdapterCanProduceNonEagerInEagerMode(adapter, record_batch)

  def testRaiseOnUnsupportedTensorRepresentation(self):
    with self.assertRaisesRegex(ValueError, "Unable to handle tensor"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              pa.schema([pa.field("a", pa.list_(pa.int64()))]),
              {"tensor": schema_pb2.TensorRepresentation()}))

  def testRaiseOnNoMatchingHandler(self):
    with self.assertRaisesRegex(ValueError, "Unable to handle tensor"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              # nested lists are not supported now.
              pa.schema([
                  pa.field("unsupported_column", pa.list_(pa.list_(pa.int64())))
              ]),
              {
                  "tensor":
                      text_format.Parse(
                          """
                  dense_tensor {
                    column_name: "unsupported_column"
                    shape: {}
                  }
                  """, schema_pb2.TensorRepresentation())
              }))

  @parameterized.named_parameters(*_INVALID_DEFAULT_VALUE_TEST_CASES)
  def testRaiseOnInvalidDefaultValue(self, value_type, default_value_pbtxt,
                                     exception_regexp):
    tensor_representation = text_format.Parse(
        """
                  dense_tensor {
                    column_name: "column"
                    shape {}
                  }""", schema_pb2.TensorRepresentation())
    tensor_representation.dense_tensor.default_value.CopyFrom(
        text_format.Parse(default_value_pbtxt,
                          schema_pb2.TensorRepresentation.DefaultValue()))
    with self.assertRaisesRegex(ValueError, exception_regexp):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              pa.schema([pa.field("column", pa.list_(value_type))]),
              {"tensor": tensor_representation}))

  @parameterized.named_parameters(*_INVALID_SPARSE_TENSOR_TEST_CASES)
  def testRaiseOnInvalidSparseTensorRepresentation(self,
                                                   tensor_representation_textpb,
                                                   arrow_schema):
    tensor_representation = text_format.Parse(tensor_representation_textpb,
                                              schema_pb2.TensorRepresentation())
    with self.assertRaisesRegex(ValueError, "Unable to handle tensor"):
      tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              pa.schema([pa.field(k, v) for k, v in arrow_schema.items()]),
              {"tensor": tensor_representation}))

  def testRaiseOnDenseTensorSizeMismatch(self):
    tensor_representation = text_format.Parse(
        """
                  dense_tensor {
                    column_name: "column"
                    shape {}
                  }""", schema_pb2.TensorRepresentation())
    with self.assertRaisesRegex(ValueError,
                                ".*Error raised when handling tensor 'tensor'"):
      ta = tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              pa.schema([pa.field("column", pa.list_(pa.int64()))]),
              {"tensor": tensor_representation}))
      ta.ToBatchTensors(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], None, [2]], type=pa.list_(pa.int64()))],
              ["column"]))

  def testOriginalTypeSpecs(self):
    arrow_schema = pa.schema([pa.field("column1", pa.list_(pa.int32()))])
    tensor_representations = {
        "column1":
            text_format.Parse(
                """
                dense_tensor {
                  column_name: "column1"
                  shape {
                    dim {
                      size: 1
                    }
                  }
                }""", schema_pb2.TensorRepresentation())
    }
    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(arrow_schema,
                                           tensor_representations))
    self.assertLen(adapter.TypeSpecs(), 1)
    self.assertEqual(adapter.TypeSpecs(), adapter.OriginalTypeSpecs())

    adapter = tensor_adapter.TensorAdapter(
        tensor_adapter.TensorAdapterConfig(
            arrow_schema,
            tensor_representations,
            original_type_specs={
                "column1": tf.TensorSpec(dtype=tf.int32, shape=[None, 1]),
                "column2": tf.TensorSpec(dtype=tf.int32, shape=[None, 1])
            }))
    self.assertLen(adapter.TypeSpecs(), 1)
    self.assertLen(adapter.OriginalTypeSpecs(), 2)

    with self.assertRaisesRegex(ValueError,
                                "original_type_specs must be a superset"):
      adapter = tensor_adapter.TensorAdapter(
          tensor_adapter.TensorAdapterConfig(
              arrow_schema,
              tensor_representations,
              original_type_specs={
                  # mismatch spec of column1
                  "column1": tf.TensorSpec(dtype=tf.int64, shape=[None, 1]),
                  "column2": tf.TensorSpec(dtype=tf.int32, shape=[None, 1])
              }))

  def testPickleTensorAdapterConfig(self):
    config = tensor_adapter.TensorAdapterConfig(
        arrow_schema=pa.schema([pa.field("column1", pa.list_(pa.int32()))]),
        tensor_representations={
            "column1":
                text_format.Parse(
                    """
                dense_tensor {
                  column_name: "column1"
                  shape {
                    dim {
                      size: 1
                    }
                  }
                }""", schema_pb2.TensorRepresentation())
        },
        original_type_specs={
            "column1": tf.TensorSpec(dtype=tf.int32, shape=[None, 1]),
            "column2": tf.TensorSpec(dtype=tf.int32, shape=[None, 1])
        })
    unpickled_config = pickle.loads(pickle.dumps(config))
    self.assertEqual(config.arrow_schema, unpickled_config.arrow_schema)
    self.assertEqual(config.tensor_representations,
                     unpickled_config.tensor_representations)
    self.assertEqual(config.original_type_specs,
                     unpickled_config.original_type_specs)


if __name__ == "__main__":
  absltest.main()
