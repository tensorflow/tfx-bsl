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

import sys
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
from tfx_bsl.arrow import array_util
from tfx_bsl.arrow import path

# pytype: disable=import-error
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import RecordBatchTake
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import MergeRecordBatches as _MergeRecordBatches
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import TotalByteSize as _TotalByteSize
except ImportError as err:
  sys.stderr.write("Error importing tfx_bsl_extension.arrow.table_util. "
                   "Some tfx_bsl functionalities are not available: {}"
                   .format(err))
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error
# pylint: enable=unused-import


_EMPTY_RECORD_BATCH = pa.RecordBatch.from_arrays([], [])

_NUMPY_KIND_TO_ARROW_TYPE = {
    "i": pa.int64(),
    "u": pa.uint64(),
    "f": pa.float64(),
    "b": pa.int8(),
    "S": pa.large_binary(),
    "O": pa.large_binary(),
    "U": pa.large_binary(),
}


def TotalByteSize(table_or_batch: Union[pa.Table, pa.RecordBatch],
                  ignore_unsupported=False):
  """Returns the in-memory size of a record batch or a table."""
  if isinstance(table_or_batch, pa.Table):
    return sum([
        _TotalByteSize(b, ignore_unsupported)
        for b in table_or_batch.to_batches(max_chunksize=None)
    ])
  else:
    return _TotalByteSize(table_or_batch, ignore_unsupported)


def NumpyKindToArrowType(kind: str) -> Optional[pa.DataType]:
  return _NUMPY_KIND_TO_ARROW_TYPE.get(kind)


def MergeRecordBatches(record_batches: List[pa.RecordBatch]) -> pa.RecordBatch:
  """Merges a list of arrow RecordBatches into one. Similar to MergeTables."""
  if not record_batches:
    return _EMPTY_RECORD_BATCH
  first_schema = record_batches[0].schema
  assert any([r.num_rows > 0 for r in record_batches]), (
      "Unable to merge empty RecordBatches.")
  if (all([r.schema.equals(first_schema) for r in record_batches[1:]])
      # combine_chunks() cannot correctly handle the case where there are
      # 0 column. (ARROW-11232)
      and first_schema):
    one_chunk_table = pa.Table.from_batches(record_batches).combine_chunks()
    batches = one_chunk_table.to_batches(max_chunksize=None)
    assert len(batches) == 1
    return batches[0]
  else:
    # Our implementation of _MergeRecordBatches is different than
    # pa.Table.concat_tables(
    #     [pa.Table.from_batches([rb]) for rb in record_batches],
    #     promote=True).combine_chunks().to_batches()[0]
    # in its handling of struct-typed columns -- if two record batches have a
    # column of the same name but of different struct types, _MergeRecordBatches
    # will try merging (recursively) those struct types while concat_tables
    # will not. We should consider upstreaming our implementation because it's a
    # generalization
    return _MergeRecordBatches(record_batches)


def _CanonicalizeType(arrow_type: pa.DataType) -> pa.DataType:
  """Returns canonical version of the given type."""
  if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
    return pa.large_list(_CanonicalizeType(arrow_type.value_type))
  else:
    result = NumpyKindToArrowType(np.dtype(arrow_type.to_pandas_dtype()).kind)
    if result is None:
      raise NotImplementedError(f"Type {arrow_type} is not supported.")
    return result


def CanonicalizeRecordBatch(
    record_batch_with_primitive_arrays: pa.RecordBatch) -> pa.RecordBatch:
  """Converts primitive arrays in a pyarrow.RecordBatch to LargeListArrays.

  The produced LargeListArrays' elements are lists that contain single element
  of the array of the canonical pyarrow type.

  Args:
    record_batch_with_primitive_arrays: A pyarrow.RecordBatch where values are
      stored in primitive arrays or list arrays.

  Returns:
    pyArrow.RecordBatch with LargeListArray columns.
  """
  arrays = []
  for column_array in record_batch_with_primitive_arrays.columns:
    canonical_type = _CanonicalizeType(column_array.type)
    if canonical_type != column_array.type:
      column_array = column_array.cast(canonical_type)
    if pa.types.is_large_list(canonical_type):
      arrays.append(column_array)
    else:
      arrays.append(array_util.ToSingletonListArray(column_array))
  return pa.RecordBatch.from_arrays(
      arrays, record_batch_with_primitive_arrays.schema.names)


def enumerate_arrays(  # pylint: disable=invalid-name
    record_batch: pa.RecordBatch,
    enumerate_leaves_only: bool,
    wrap_flat_struct_in_list: bool = True
) -> Iterable[Tuple[path.ColumnPath, pa.Array]]:
  """Enumerates arrays in a RecordBatch.

  Define:
    primitive: primitive arrow arrays (e.g. Int64Array).
    nested_list := list<nested_list> | list<primitive> | null
    # note: a null array can be seen as a list<primitive>, which contains only
    #   nulls and the type of the primitive is unknown.
    # example:
    #   null,
    #   list<null>,  # like list<list<unknown_type>> with only null values.
    #   list<list<int64>>,
    struct := struct<{field: nested_list | struct}> | list<struct>
    # example:
    #   struct<{"foo": list<int64>},
    #   list<struct<{"foo": list<int64>}>>,
    #   struct<{"foo": struct<{"bar": list<list<int64>>}>}>

  This function assumes `record_batch` contains only nested_list and struct
  columns. It enumerates each column in `record_batch`, and if that column is
  a struct, it flattens the outer lists wrapping it (if any), and recursively
  enumerates the array of each field in the struct (also see
  `enumerate_leaves_only`).

  A ColumnPath is included in the result to address the enumerated array.
  Note that the ColumnPath merely addresses in the `record_batch` and struct
  arrays. It does not indicate whether / how a struct array is nested.

  Args:
    record_batch: The RecordBatch whose arrays to be visited.
    enumerate_leaves_only: If True, only enumerate leaf arrays. A leaf array
      is an array whose type does not have any struct nested in.
      Otherwise, also enumerate the struct arrays where the leaf arrays are
      contained.
    wrap_flat_struct_in_list: if True, and if a struct<[Ts]> array is
      encountered, it will be wrapped in a list array, so it becomes a
      list<struct<[Ts]>>, in which each sub-list contains one element.
      A caller can make use of this option to assume all the arrays enumerated
      here are list<inner_type>.
  Yields:
    A tuple. The first term is the path of the feature, and the second term is
    the feature array.
  """

  def _recursion_helper(   # pylint: disable=invalid-name
      feature_path: path.ColumnPath, array: pa.Array,
  ) -> Iterable[Tuple[path.ColumnPath, pa.Array]]:
    """Recursion helper."""
    array_type = array.type
    innermost_nested_type = array_util.get_innermost_nested_type(array_type)
    if pa.types.is_struct(innermost_nested_type):
      if not enumerate_leaves_only:
        # special handing for a flat struct array -- wrap it in a ListArray
        # whose elements are singleton lists. This way downstream can keep
        # assuming the enumerated arrays are list<*>.
        to_yield = array
        if pa.types.is_struct(array_type) and wrap_flat_struct_in_list:
          to_yield = array_util.ToSingletonListArray(array)
        yield (feature_path, to_yield)
      flat_struct_array, _ = array_util.flatten_nested(array)
      for field in flat_struct_array.type:
        field_name = field.name
        yield from _recursion_helper(
            feature_path.child(field_name),
            array_util.get_field(flat_struct_array, field_name))
    else:
      yield (feature_path, array)

  for column_name, column in zip(record_batch.schema.names,
                                 record_batch.columns):
    yield from _recursion_helper(
        path.ColumnPath([column_name]), column)


def get_array(   # pylint: disable=invalid-name
    record_batch: pa.RecordBatch,
    query_path: path.ColumnPath,
    return_example_indices: bool,
    wrap_flat_struct_in_list: bool = True,
) -> Tuple[pa.Array, Optional[np.ndarray]]:
  """Retrieve a nested array (and optionally example indices) from RecordBatch.

  This function has the same assumption over `record_batch` as
  `enumerate_arrays()` does.

  If the provided path refers to a leaf in the `record_batch`, then a
  "nested_list" will be returned. If the provided path does not refer to a leaf,
  a "struct" will be returned.

  See `enumerate_arrays()` for definition of "nested_list" and "struct".

  Args:
    record_batch: The RecordBatch whose arrays to be visited.
    query_path: The ColumnPath to lookup in the record_batch.
    return_example_indices: Whether to return an additional array containing the
      example indices of the elements in the array corresponding to the
      query_path.
    wrap_flat_struct_in_list: if True, and if the query_path leads to a
      struct<[Ts]> array, it will be wrapped in a list array, where each
      sub-list contains one element. Caller can make use of this option to
      assume this function always returns a list<inner_type>.

  Returns:
    A tuple. The first term is the feature array and the second term is the
    example_indices array for the feature array (i.e. array[i] came from the
    example at row example_indices[i] in the record_batch.).

  Raises:
    KeyError: When the query_path is empty, or cannot be found in the
    record_batch and its nested struct arrays.
  """

  def _recursion_helper(   # pylint: disable=invalid-name
      query_path: path.ColumnPath, array: pa.Array,
      example_indices: Optional[np.ndarray]
  ) -> Tuple[pa.Array, Optional[np.ndarray]]:
    """Recursion helper."""
    array_type = array.type
    if not query_path:
      if pa.types.is_struct(array_type) and wrap_flat_struct_in_list:
        array = array_util.ToSingletonListArray(array)
      return array, example_indices
    if not pa.types.is_struct(array_util.get_innermost_nested_type(array_type)):
      raise KeyError("Cannot process query_path ({}) inside an array of type "
                     "{}. Expecting a struct<...> or "
                     "(large_)list...<struct<...>>.".format(
                         query_path, array_type))
    flat_struct_array, parent_indices = array_util.flatten_nested(
        array, example_indices is not None)
    flat_indices = None
    if example_indices is not None:
      flat_indices = example_indices[parent_indices]

    step = query_path.steps()[0]

    try:
      child_array = array_util.get_field(flat_struct_array, step)
    except KeyError as exception:
      raise KeyError(f"query_path step ({step}) not in struct.") from exception

    relative_path = path.ColumnPath(query_path.steps()[1:])
    return _recursion_helper(relative_path, child_array, flat_indices)

  if not query_path:
    raise KeyError("query_path must be non-empty.")
  column_name = query_path.steps()[0]
  field_index = record_batch.schema.get_field_index(column_name)
  if field_index < 0:
    raise KeyError(f"query_path step 0 ({column_name}) not in record batch.")
  array = record_batch.column(field_index)
  array_path = path.ColumnPath(query_path.steps()[1:])

  example_indices = np.arange(
      record_batch.num_rows) if return_example_indices else None
  return _recursion_helper(array_path, array, example_indices)
