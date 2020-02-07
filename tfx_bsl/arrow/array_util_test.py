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
# limitations under the License
"""Tests for tfx_bsl.arrow.array_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pyarrow as pa
import six

from tfx_bsl.arrow import array_util

from absl.testing import absltest
from absl.testing import parameterized


_LIST_TYPE_PARAMETERS = [
    dict(testcase_name="list", list_type_factory=pa.list_),
    dict(testcase_name="large_list", list_type_factory=pa.large_list),
]


class ArrayUtilTest(parameterized.TestCase):

  def test_invalid_input_type(self):

    functions_expecting_list_array = [
        array_util.GetFlattenedArrayParentIndices,
    ]
    functions_expecting_array = [array_util.GetArrayNullBitmapAsByteArray]
    functions_expecting_binary_array = [array_util.GetBinaryArrayTotalByteSize]
    for f in itertools.chain(functions_expecting_list_array,
                             functions_expecting_array,
                             functions_expecting_binary_array):
      with self.assertRaisesRegex(
          TypeError, "incompatible function arguments"):
        f(1)

    for f in functions_expecting_list_array:
      with self.assertRaisesRegex(RuntimeError, "NotImplemented"):
        f(pa.array([1, 2, 3]))

    for f in functions_expecting_binary_array:
      with self.assertRaisesRegex(RuntimeError, "NotImplemented"):
        f(pa.array([[1, 2, 3]]))

  @parameterized.named_parameters(*_LIST_TYPE_PARAMETERS)
  def test_list_lengths(self, list_type_factory):
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([], type=list_type_factory(pa.int64())))
    self.assertTrue(list_lengths.equals(pa.array([], type=pa.int64())))
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([[1., 2.], [], [3.]], type=list_type_factory(pa.float32())))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int64())))
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([[1., 2.], None, [3.]], type=list_type_factory(pa.float64())))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int64())))

  @parameterized.named_parameters(*_LIST_TYPE_PARAMETERS)
  def test_element_lengths_list_array(self, list_type_factory):
    list_lengths = array_util.GetElementLengths(
        pa.array([], type=list_type_factory(pa.int64())))
    self.assertTrue(list_lengths.equals(pa.array([], type=pa.int64())))
    list_lengths = array_util.GetElementLengths(
        pa.array([[1., 2.], [], [3.]], list_type_factory(pa.float32())))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int64())))
    list_lengths = array_util.GetElementLengths(
        pa.array([[1., 2.], None, [3.]], list_type_factory(pa.float64())))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int64())))

  @parameterized.named_parameters(*[
      dict(testcase_name="binary", binary_like_type=pa.binary()),
      dict(testcase_name="string", binary_like_type=pa.string()),
      dict(testcase_name="large_binary", binary_like_type=pa.large_binary()),
      dict(testcase_name="large_string", binary_like_type=pa.large_string()),
  ])
  def test_element_lengths_binary_like(self, binary_like_type):

    list_lengths = array_util.GetElementLengths(
        pa.array([b"a", b"bb", None, b"", b"ccc"], type=binary_like_type))
    self.assertTrue(list_lengths.equals(pa.array([1, 2, 0, 0, 3],
                                                 type=pa.int64())))

  def test_element_lengths_unsupported_type(self):
    with self.assertRaisesRegex(RuntimeError, "NotImplemented"):
      array_util.GetElementLengths(pa.array([1, 2, 3], type=pa.int32()))

  def test_get_array_null_bitmap_as_byte_array(self):
    array = pa.array([], type=pa.int32())
    null_masks = array_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([], type=pa.uint8())))

    array = pa.array([1, 2, None, 3, None], type=pa.int32())
    null_masks = array_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(
        null_masks.equals(pa.array([0, 0, 1, 0, 1], type=pa.uint8())))

    array = pa.array([1, 2, 3])
    null_masks = array_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([0, 0, 0], type=pa.uint8())))

    array = pa.array([None, None, None], type=pa.int32())
    null_masks = array_util.GetArrayNullBitmapAsByteArray(array)
    self.assertTrue(null_masks.equals(pa.array([1, 1, 1], type=pa.uint8())))
    # Demonstrate that the returned array can be converted to a numpy boolean
    # array w/o copying
    np.testing.assert_equal(
        np.array([True, True, True]), null_masks.to_numpy().view(np.bool))

  @parameterized.named_parameters(*[
      dict(
          testcase_name="list",
          list_type_factory=pa.list_,
          parent_indices_type=pa.int32()),
      dict(
          testcase_name="large_list",
          list_type_factory=pa.large_list,
          parent_indices_type=pa.int64()),
  ])
  def test_get_flattened_array_parent_indices(self, list_type_factory,
                                              parent_indices_type):
    indices = array_util.GetFlattenedArrayParentIndices(
        pa.array([], type=list_type_factory(pa.int32())))
    self.assertTrue(indices.equals(pa.array([], type=parent_indices_type)))

    indices = array_util.GetFlattenedArrayParentIndices(
        pa.array([[1.], [2.], [], [3., 4.]],
                 type=list_type_factory(pa.float32())))
    self.assertTrue(
        indices.equals(pa.array([0, 1, 3, 3], type=parent_indices_type)))

    indices = array_util.GetFlattenedArrayParentIndices(
        pa.array([[1.], [2.], [], [3., 4.]],
                 type=list_type_factory(pa.float32())).slice(1))
    self.assertTrue(
        indices.equals(pa.array([0, 2, 2], type=parent_indices_type)))

  @parameterized.named_parameters(*[
      dict(testcase_name="binary", binary_like_type=pa.binary()),
      dict(testcase_name="string", binary_like_type=pa.string()),
      dict(testcase_name="large_binary", binary_like_type=pa.large_binary()),
      dict(testcase_name="large_string", binary_like_type=pa.large_string()),
  ])
  def test_get_binary_array_total_byte_size(self, binary_like_type):
    array = pa.array([b"abc", None, b"def", b"", b"ghi"], type=binary_like_type)
    self.assertEqual(9, array_util.GetBinaryArrayTotalByteSize(array))
    sliced_1_2 = array.slice(1, 2)
    self.assertEqual(3, array_util.GetBinaryArrayTotalByteSize(sliced_1_2))
    sliced_2 = array.slice(2)
    self.assertEqual(6, array_util.GetBinaryArrayTotalByteSize(sliced_2))

    empty_array = pa.array([], type=binary_like_type)
    self.assertEqual(0, array_util.GetBinaryArrayTotalByteSize(empty_array))

  def _value_counts_struct_array_to_dict(self, value_counts):
    result = {}
    for value_count in value_counts:
      value_count = value_count.as_py()
      result[value_count["values"]] = value_count["counts"]
    return result

  def test_value_counts_binary(self):
    binary_array = pa.array([b"abc", b"ghi", b"def", b"ghi", b"ghi", b"def"])
    expected_result = {b"abc": 1, b"ghi": 3, b"def": 2}
    self.assertDictEqual(self._value_counts_struct_array_to_dict(
        array_util.ValueCounts(binary_array)), expected_result)

  def test_value_counts_integer(self):
    int_array = pa.array([1, 4, 1, 3, 1, 4])
    expected_result = {1: 3, 4: 2, 3: 1}
    self.assertDictEqual(self._value_counts_struct_array_to_dict(
        array_util.ValueCounts(int_array)), expected_result)

  def test_value_counts_empty(self):
    empty_array = pa.array([])
    expected_result = {}
    self.assertDictEqual(self._value_counts_struct_array_to_dict(
        array_util.ValueCounts(empty_array)), expected_result)

_MAKE_LIST_ARRAY_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="parent_indices_not_arrow_int64",
        num_parents=1,
        parent_indices=pa.array([0], type=pa.int32()),
        values=pa.array([1]),
        expected_error=RuntimeError,
        expected_error_regexp="must be int64"
        ),
    dict(
        testcase_name="parent_indices_length_not_equal_to_values_length",
        num_parents=1,
        parent_indices=pa.array([0], type=pa.int64()),
        values=pa.array([1, 2]),
        expected_error=RuntimeError,
        expected_error_regexp="values array and parent indices array must be of the same length"
    ),
    dict(
        testcase_name="num_parents_too_small",
        num_parents=1,
        parent_indices=pa.array([1], type=pa.int64()),
        values=pa.array([1]),
        expected_error=RuntimeError,
        expected_error_regexp="Found a parent index 1 while num_parents was 1"
        )
]


_MAKE_LIST_ARRAY_TEST_CASES = [
    dict(
        testcase_name="parents_are_all_empty",
        num_parents=5,
        parent_indices=pa.array([], type=pa.int64()),
        values=pa.array([], type=pa.int64()),
        expected=pa.array([None, None, None, None, None],
                          type=pa.list_(pa.int64()))),
    dict(
        testcase_name="long_num_parent",
        num_parents=(long(1) if six.PY2 else 1),
        parent_indices=pa.array([0], type=pa.int64()),
        values=pa.array([1]),
        expected=pa.array([[1]])
    ),
    dict(
        testcase_name="leading nones",
        num_parents=3,
        parent_indices=pa.array([2], type=pa.int64()),
        values=pa.array([1]),
        expected=pa.array([None, None, [1]]),
    ),
    dict(
        testcase_name="same_parent_and_holes",
        num_parents=4,
        parent_indices=pa.array([0, 0, 0, 3, 3], type=pa.int64()),
        values=pa.array(["a", "b", "c", "d", "e"]),
        expected=pa.array([["a", "b", "c"], None, None, ["d", "e"]])
    )
]


class MakeListArrayFromParentIndicesAndValuesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_MAKE_LIST_ARRAY_INVALID_INPUT_TEST_CASES)
  def testInvalidInput(self, num_parents, parent_indices, values,
                       expected_error, expected_error_regexp):
    with self.assertRaisesRegex(expected_error, expected_error_regexp):
      array_util.MakeListArrayFromParentIndicesAndValues(
          num_parents, parent_indices, values)

  @parameterized.named_parameters(*_MAKE_LIST_ARRAY_TEST_CASES)
  def testMakeListArray(self, num_parents, parent_indices, values, expected):
    actual = array_util.MakeListArrayFromParentIndicesAndValues(
        num_parents, parent_indices, values)
    self.assertTrue(
        actual.equals(expected),
        "actual: {}, expected: {}".format(actual, expected))


_COO_FROM_LIST_ARRAY_TEST_CASES = [
    dict(
        testcase_name="flat_array",
        list_array=[1, 2, 3, 4],
        expected_coo=[0, 1, 2, 3],
        expected_dense_shape=[4],
        array_types=[pa.int32()],
    ),
    dict(
        testcase_name="empty_array",
        list_array=[],
        expected_coo=[],
        expected_dense_shape=[0],
        array_types=[pa.int64()],
    ),
    dict(
        testcase_name="empty_2d_array",
        list_array=[[]],
        expected_coo=[],
        expected_dense_shape=[1, 0],
        array_types=[pa.list_(pa.int64()),
                     pa.large_list(pa.string())]),
    dict(
        testcase_name="2d_ragged",
        list_array=[["a", "b"], ["c"], [], ["d", "e"]],
        expected_coo=[0, 0, 0, 1, 1, 0, 3, 0, 3, 1],
        expected_dense_shape=[4, 2],
        array_types=[pa.list_(pa.string()),
                     pa.large_list(pa.large_binary())]),
    dict(
        testcase_name="3d_ragged",
        list_array=[[["a", "b"], ["c"]], [[], ["d", "e"]]],
        expected_coo=[0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        expected_dense_shape=[2, 2, 2],
        array_types=[
            pa.list_(pa.list_(pa.binary())),
            pa.large_list(pa.large_list(pa.binary())),
            pa.large_list(pa.list_(pa.binary())),
            pa.list_(pa.large_list(pa.binary())),
        ],
    ),
]


class CooFromListArrayTest(parameterized.TestCase):

  @parameterized.named_parameters(*_COO_FROM_LIST_ARRAY_TEST_CASES)
  def testCooFromListArray(
      self, list_array, expected_coo, expected_dense_shape, array_types):

    for array_type in array_types:
      for input_array in [
          pa.array(list_array, type=array_type),
          # it should work for sliced arrays.
          pa.array(list_array + list_array,
                   type=array_type).slice(0, len(list_array)),
          pa.array(list_array + list_array,
                   type=array_type).slice(len(list_array)),
      ]:
        coo, dense_shape = array_util.CooFromListArray(input_array)
        self.assertTrue(coo.type.equals(pa.int64()))
        self.assertTrue(dense_shape.type.equals(pa.int64()))

        self.assertEqual(expected_coo, coo.to_pylist())
        self.assertEqual(expected_dense_shape, dense_shape.to_pylist())


_FILL_NULL_LISTS_TEST_CASES = [
    dict(
        testcase_name="empty_array",
        list_array=[],
        value_type=pa.int32(),
        fill_with=[0],
        expected=[],
    ),
    dict(
        testcase_name="only_one_null",
        list_array=[None],
        value_type=pa.int64(),
        fill_with=[0, 1],
        expected=[[0, 1]],
    ),
    dict(
        testcase_name="no_nulls",
        list_array=[[1], [2], [3]],
        value_type=pa.int64(),
        fill_with=[0],
        expected=[[1], [2], [3]],
    ),
    dict(
        testcase_name="all_nulls",
        list_array=[None, None, None],
        value_type=pa.int64(),
        fill_with=[0, 1],
        expected=[[0, 1], [0, 1], [0, 1]],
    ),
    dict(
        testcase_name="nulls_at_end",
        list_array=[[1], [2], None],
        value_type=pa.int64(),
        fill_with=[0, 1],
        expected=[[1], [2], [0, 1]],
    ),
    dict(
        testcase_name="nulls_at_beginning",
        list_array=[None, None, [1]],
        value_type=pa.int64(),
        fill_with=[],
        expected=[[], [], [1]],
    ),
    dict(
        testcase_name="nulls_scattered",
        list_array=[["a"], ["b"], ["c"], None, ["d"], None, ["e"]],
        value_type=pa.large_binary(),
        fill_with=["x", "x"],
        expected=[["a"], ["b"], ["c"], ["x", "x"], ["d"], ["x", "x"], ["e"]],
    )
]


def _cross_named_parameters(*named_parameters_dicts):
  result = []
  for product in itertools.product(*named_parameters_dicts):
    crossed = dict(product[0])
    testcase_name = crossed["testcase_name"]
    for d in product[1:]:
      testcase_name += "_" + d["testcase_name"]
      crossed.update(d)
    crossed["testcase_name"] = testcase_name
    result.append(crossed)
  return result


class FillNullListsTest(parameterized.TestCase):

  @parameterized.named_parameters(*_cross_named_parameters(
      _FILL_NULL_LISTS_TEST_CASES, _LIST_TYPE_PARAMETERS))
  def testFillNullLists(
      self, list_array, value_type, fill_with, expected, list_type_factory):
    actual = array_util.FillNullLists(
        pa.array(list_array, type=list_type_factory(value_type)),
        pa.array(fill_with, type=value_type))
    self.assertTrue(
        actual.equals(pa.array(expected, type=list_type_factory(value_type))),
        "{} vs {}".format(actual, expected))

  def testNonListArray(self):
    with self.assertRaisesRegex(RuntimeError, "NotImplemented"):
      array_util.FillNullLists(pa.array([1, 2, 3]), pa.array([4]))

  def testValueTypeDoesNotEqualFillType(self):
    with self.assertRaisesRegex(RuntimeError, "to be of the same type"):
      array_util.FillNullLists(pa.array([[1]]), pa.array(["a"]))


if __name__ == "__main__":
  absltest.main()
