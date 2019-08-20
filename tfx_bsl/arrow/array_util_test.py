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

from tfx_bsl.arrow import array_util
from tfx_bsl.pyarrow_tf import pyarrow as pa

from absl.testing import absltest


class ArrowUtilTest(absltest.TestCase):

  def test_invalid_input_type(self):

    functions_expecting_list_array = [
        array_util.ListLengthsFromListArray,
    ]

    for f in functions_expecting_list_array:
      with self.assertRaisesRegex(
          TypeError, "incompatible function arguments"):
        f(1)

    for f in functions_expecting_list_array:
      with self.assertRaisesRegex(RuntimeError, "Expected ListArray but got"):
        f(pa.array([1, 2, 3]))

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


if __name__ == "__main__":
  absltest.main()
