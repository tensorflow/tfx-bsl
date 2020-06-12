// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Note: ALL the functions declared here will be SWIG wrapped into
// pywrap_tensorflow_data_validation.
#ifndef THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_
#define THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_

#include <memory>
#include <vector>

#include "tfx_bsl/cc/util/status.h"

namespace arrow {
class Array;
class RecordBatch;
}  // namespace arrow

namespace tfx_bsl {
// Merges a list of record batches into one.
// The columns are concatenated (there will be only one chunk per column).
// Columns of the same name must be of the same type, or be a column of
// NullArrays.
// If a column in some tables are of type T, in some other tables are of
// NullArrays, the concatenated column is of type T, with nulls representing
// the rows from the table with NullArrays.
// If a column appears in some tables but not in some other tables, the
// concatenated column will contain nulls representing the rows from the table
// with that column missing.
Status MergeRecordBatches(
    const std::vector<std::shared_ptr<arrow::RecordBatch>>& record_batches,
    std::shared_ptr<arrow::RecordBatch>* result);

// Returns the total byte size of all the Arrays a table or a record batch
// consists of. Slicing offsets are taken consideration, but buffer sharing
// is not. So this number could be larger than the actual number of bytes the
// table or record batch occupies in RAM.
// If `ignore_unsupported` is true, these functions will not return an error
// when encountering unsupported columns and the result won't include
// the size of them, otherwise an error will be returned.
Status TotalByteSize(const arrow::RecordBatch& record_batch,
                     bool ignore_unsupported, size_t* result);

// Returns a RecordBatch that contains rows in `indices`.
Status RecordBatchTake(const arrow::RecordBatch& record_batch,
                       const arrow::Array& indices,
                       std::shared_ptr<arrow::RecordBatch>* result);
}  // namespace tfx_bsl

#endif  // THIRD_PARTY_PY_TENSORFLOW_DATA_VALIDATION_ARROW_CC_MERGE_H_
