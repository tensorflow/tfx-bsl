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
import six

from tfx_bsl.arrow import array_util
from tfx_bsl.pyarrow_tf import pyarrow as pa

from absl.testing import absltest
from absl.testing import parameterized


class ArrayUtilTest(absltest.TestCase):

  def test_invalid_input_type(self):

    functions_expecting_list_array = [
        array_util.ListLengthsFromListArray,
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
      with self.assertRaisesRegex(RuntimeError, "Expected ListArray but got"):
        f(pa.array([1, 2, 3]))

    for f in functions_expecting_binary_array:
      with self.assertRaisesRegex(RuntimeError, "Expected BinaryArray"):
        f(pa.array([[1, 2, 3]]))

  def test_list_lengths(self):
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([], type=pa.list_(pa.int64())))
    self.assertTrue(list_lengths.equals(pa.array([], type=pa.int32())))
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([[1., 2.], [], [3.]]))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int32())))
    list_lengths = array_util.ListLengthsFromListArray(
        pa.array([[1., 2.], None, [3.]]))
    self.assertTrue(list_lengths.equals(pa.array([2, 0, 1], type=pa.int32())))

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

  def test_get_flattened_array_parent_indices(self):
    indices = array_util.GetFlattenedArrayParentIndices(
        pa.array([], type=pa.list_(pa.int32())))
    self.assertTrue(indices.equals(pa.array([], type=pa.int32())))

    indices = array_util.GetFlattenedArrayParentIndices(
        pa.array([[1.], [2.], [], [3.]]))
    self.assertTrue(indices.equals(pa.array([0, 1, 3], type=pa.int32())))

  def test_get_binary_array_total_byte_size(self):
    binary_array = pa.array([b"abc", None, b"def", b"", b"ghi"])
    self.assertEqual(9, array_util.GetBinaryArrayTotalByteSize(binary_array))
    sliced_1_2 = binary_array.slice(1, 2)
    self.assertEqual(3, array_util.GetBinaryArrayTotalByteSize(sliced_1_2))
    sliced_2 = binary_array.slice(2)
    self.assertEqual(6, array_util.GetBinaryArrayTotalByteSize(sliced_2))

    unicode_array = pa.array([u"abc"])
    self.assertEqual(3, array_util.GetBinaryArrayTotalByteSize(unicode_array))

    empty_array = pa.array([], type=pa.binary())
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

if __name__ == "__main__":
  absltest.main()
