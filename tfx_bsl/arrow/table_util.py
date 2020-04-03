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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import pyarrow as pa
from typing import List
# pytype: disable=import-error
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import MergeTables
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import RecordBatchTake
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import SliceTableByRowIndices
  from tfx_bsl.cc.tfx_bsl_extension.arrow.table_util import TotalByteSize
except ImportError as err:
  import sys
  sys.stderr.write("Error importing tfx_bsl_extension.arrow.table_util. "
                   "Some tfx_bsl functionalities are not available: {}"
                   .format(err))
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error
# pylint: enable=unused-import


_EMPTY_RECORD_BATCH = pa.RecordBatch.from_arrays([], [])


def MergeRecordBatches(record_batches: List[pa.RecordBatch]) -> pa.RecordBatch:
  """Merges a list of arrow RecordBatches into one. Similar to MergeTables."""
  if not record_batches:
    return _EMPTY_RECORD_BATCH
  first_schema = record_batches[0].schema
  assert any([r.num_rows > 0 for r in record_batches]), (
      "Unable to merge empty RecordBatches.")
  if all([r.schema.equals(first_schema) for r in record_batches[1:]]):
    one_chunk_table = pa.Table.from_batches(record_batches).combine_chunks()
  else:
    one_chunk_table = MergeTables(
        [pa.Table.from_batches([r]) for r in record_batches])

  batches = one_chunk_table.to_batches(max_chunksize=None)
  assert len(batches) == 1
  return batches[0]
