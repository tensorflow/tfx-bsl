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
from typing import List, Optional, Union

import numpy as np
import pyarrow as pa
from tfx_bsl.arrow import array_util

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
