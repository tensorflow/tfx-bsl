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

from typing import Tuple, Optional, Union

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


def ToSingletonListArray(array: pa.Array) -> pa.Array:  # pylint: disable=invalid-name
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


def MakeListArrayFromParentIndicesAndValues(num_parents: int,   # pylint: disable=invalid-name
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


def is_list_like(data_type: pa.DataType) -> bool:
  """Returns true if an Arrow type is list-like."""
  return pa.types.is_list(data_type) or pa.types.is_large_list(data_type)


def get_innermost_nested_type(arrow_type: pa.DataType) -> pa.DataType:
  """Returns the innermost type of a nested list type."""
  while is_list_like(arrow_type):
    arrow_type = arrow_type.value_type
  return arrow_type


def flatten_nested(
    array: pa.Array, return_parent_indices: bool = False
    ) -> Tuple[pa.Array, Optional[np.ndarray]]:
  """Flattens all the list arrays nesting an array.

  If `array` is not list-like, itself will be returned.

  Args:
    array: pa.Array to flatten.
    return_parent_indices: If True, also returns the parent indices array.

  Returns:
    A tuple. The first term is the flattened array. The second term is None
    if `return_parent_indices` is False; otherwise it's a parent indices array
    parallel to the flattened array: if parent_indices[i] = j, then
    flattened_array[i] belongs to the j-th element of the input array.
  """
  parent_indices = None

  while is_list_like(array.type):
    if return_parent_indices:
      cur_parent_indices = GetFlattenedArrayParentIndices(
          array).to_numpy()
      if parent_indices is None:
        parent_indices = cur_parent_indices
      else:
        parent_indices = parent_indices[cur_parent_indices]
    array = array.flatten()

  # the array is not nested at the first place.
  if return_parent_indices and parent_indices is None:
    parent_indices = np.arange(len(array))
  return array, parent_indices


def get_field(struct_array: pa.StructArray, field: Union[str, int]) -> pa.Array:
  """Returns struct_array.field(field) with null propagation.

  This function is equivalent to struct_array.field() but correctly handles
  null propagation (the parent struct's null values are propagated to children).

  Args:
    struct_array: A struct array which should be queried.
    field: The request field to retrieve.

  Returns:
    A pa.Array containing the requested field.

  Raises:
    KeyError: If field is not a child field in struct_array.
  """
  child_array = struct_array.field(field)

  # In case all values are present then there's no need for special handling.
  # We can return child_array as is to avoid a performance penalty caused by
  # constructing and flattening the returned array.
  if struct_array.null_count == 0:
    return child_array
  # is_valid returns a BooleanArray with two buffers the buffer at offset
  # 0 is always None and buffer 1 contains the data on which fields are
  # valid/not valid.
  # (https://arrow.apache.org/docs/format/Columnar.html#buffer-listing-for-each-layout)
  validity_bitmap_buffer = struct_array.is_valid().buffers()[1]

  # Construct a new struct array with a single field.  Calling flatten() on the
  # new array guarantees validity bitmaps are merged correctly.
  new_type = pa.struct([pa.field(field, child_array.type)])
  filtered_struct = pa.StructArray.from_buffers(
      new_type,
      len(struct_array), [validity_bitmap_buffer],
      null_count=struct_array.null_count,
      children=[child_array])
  return filtered_struct.flatten()[0]

