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
"""Arrow Array utilities."""

import numpy as np
import pyarrow as pa
# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import ListLengthsFromListArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetElementLengths
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetFlattenedArrayParentIndices
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetArrayNullBitmapAsByteArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetBinaryArrayTotalByteSize
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import IndexIn
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import MakeListArrayFromParentIndicesAndValues as _MakeListArrayFromParentIndicesAndValues
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import CooFromListArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import FillNullLists
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetByteSize
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import CountInvalidUTF8
except ImportError:
  import sys
  sys.stderr.write("Error importing tfx_bsl_extension.arrow.array_util. "
                   "Some tfx_bsl functionalities are not available")
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top
# pylint: enable=unused-import


def ToSingletonListArray(array: pa.Array) -> pa.Array:
  """Converts an array of `type` to a `LargeListArray<type>`.

  Where result[i] is null if array[i] is null; [array[i]] otherwise.

  Args:
    array: an arrow Array.

  Returns:
    a LargeListArray.
  """
  array_size = len(array)
  # fast path: values are not copied.
  if array.null_count == 0:
    return pa.LargeListArray.from_arrays(
        pa.array(np.arange(0, array_size + 1, dtype=np.int32)), array)

  # null_mask[i] = 1 iff array[i] is null.
  null_mask = np.asarray(GetArrayNullBitmapAsByteArray(array))
  # presence_mask[i] = 0 iff array[i] is null
  presence_mask = np.subtract(1, null_mask, dtype=np.uint8)
  offsets_np = np.zeros((array_size + 1,), np.int32)
  np.cumsum(presence_mask, out=offsets_np[1:])

  # This is the null mask over offsets (but ListArray.from_arrays() uses it as
  # the null mask for the ListArray), so its length is array_size +1, but the
  # last element is always False.
  list_array_null_mask = np.zeros((array_size + 1,), bool)
  list_array_null_mask[:array_size] = null_mask.view(bool)
  values_non_null = array.take(pa.array(np.flatnonzero(presence_mask)))
  return pa.LargeListArray.from_arrays(
      pa.array(offsets_np, mask=list_array_null_mask), values_non_null)


def MakeListArrayFromParentIndicesAndValues(num_parents: int,
                                            parent_indices: pa.Array,
                                            values: pa.Array,
                                            empty_list_as_null: bool = True):
  """Makes an Arrow LargeListArray from parent indices and values.

  For example, if `num_parents = 6`, `parent_indices = [0, 1, 1, 3, 3]` and
  `values` is (an arrow Array of) `[0, 1, 2, 3, 4]`, then the result will
  be a `pa.LargeListArray` of integers:
  `[[0], [1, 2], <empty_list>, [3, 4], <empty_list>]`
  where `<empty_list>` is `null` if `empty_list_as_null` is True, or `[]` if
  False.

  Args:
    num_parents: integer, number of sub-list. Must be greater than or equal to
      `max(parent_indices) + 1`.
    parent_indices: an int64 pa.Array. Must be sorted in increasing order.
    values: a pa.Array. Its length must equal to the length of `parent_indices`.
    empty_list_as_null: if True, empty sub-lists will become null elements
      in the result ListArray. Otherwise they become empty sub-lists.

  Returns:
    A LargeListArray.
  """
  return _MakeListArrayFromParentIndicesAndValues(num_parents, parent_indices,
                                                  values, empty_list_as_null)
