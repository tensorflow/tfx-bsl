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

#include "tfx_bsl/cc/arrow/table_util.h"

#include <cstddef>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
using ::arrow::Array;
using ::arrow::RecordBatch;

Status TotalByteSize(const RecordBatch& record_batch,
                     const bool ignore_unsupported, size_t* result) {
  *result = 0;
  for (int i = 0; i < record_batch.num_columns(); ++i) {
    size_t array_size;
    auto status = GetByteSize(*record_batch.column(i), &array_size);
    if (ignore_unsupported && status.code() == error::UNIMPLEMENTED) {
      continue;
    }
    TFX_BSL_RETURN_IF_ERROR(status);
    *result += array_size;
  }
  return Status::OK();
}

// Returns a RecordBatch that contains rows in `indices`.
Status RecordBatchTake(const RecordBatch& record_batch, const Array& indices,
                       std::shared_ptr<RecordBatch>* result) {
  arrow::compute::FunctionContext ctx;
  arrow::compute::TakeOptions options;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      arrow::compute::Take(&ctx, record_batch, indices, options, result)));
  return Status::OK();
}

}  // namespace tfx_bsl
