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
#include "arrow/array/concatenate.h"
#include "arrow/compute/api.h"
#include "arrow/visitor_inline.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace {
using ::arrow::Array;
using ::arrow::ArrayVector;
using ::arrow::RecordBatch;
using ::arrow::Table;
using ::arrow::Type;

// Returns an empty table that has the same schema as `table`.
Status GetEmptyTableLike(const Table& table, std::shared_ptr<Table>* result) {
  std::vector<std::shared_ptr<Array>> empty_arrays;
  for (const auto& f : table.schema()->fields()) {
    empty_arrays.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
        arrow::MakeArrayOfNull(f->type(), /*length=*/0, &empty_arrays.back())));
  }
  *result = Table::Make(table.schema(), empty_arrays, 0);
  return Status::OK();
}

template <typename RowIndicesT>
Status SliceTableByRowIndicesInternal(
    const std::shared_ptr<Table>& table,
    const absl::Span<const RowIndicesT> row_indices_span,
    std::shared_ptr<Table>* result) {
  if (row_indices_span.empty()) {
    return GetEmptyTableLike(*table, result);
  }

  if (row_indices_span.back() >= table->num_rows()) {
    return errors::InvalidArgument("Row indices out of range.");
  }

  std::vector<std::shared_ptr<Table>> table_slices;
  // The following loop essentially turns consecutive indices into
  // ranges, and slice `table` by those ranges.
  for (int64_t begin = 0, end = 1; end <= row_indices_span.size(); ++end) {
    while (end < row_indices_span.size() &&
           row_indices_span[end] == row_indices_span[end - 1] + 1) {
      ++end;
    }
    // Verify that the row indices are sorted.
    if (end < row_indices_span.size() &&
        row_indices_span[begin] >= row_indices_span[end]) {
      return errors::InvalidArgument("Row indices are not sorted.");
    }
    table_slices.push_back(table->Slice(row_indices_span[begin], end - begin));
    begin = end;
  }

  auto status_or_concatenated = arrow::ConcatenateTables(table_slices);
  if (!status_or_concatenated.ok()) {
    return FromArrowStatus(status_or_concatenated.status());
  }

  return FromArrowStatus(
      status_or_concatenated.ValueOrDie()->CombineChunks(
          arrow::default_memory_pool(), result));
}

}  // namespace

Status SliceTableByRowIndices(const std::shared_ptr<Table>& table,
                              const std::shared_ptr<Array>& row_indices,
                              std::shared_ptr<Table>* result) {
  if (row_indices->type()->id() == arrow::Type::INT32) {
    const arrow::Int32Array* row_indices_array =
        static_cast<const arrow::Int32Array*>(row_indices.get());
    absl::Span<const int32_t> row_indices_span(
        row_indices_array->raw_values(), row_indices_array->length());
    return SliceTableByRowIndicesInternal(table, row_indices_span, result);
  } else if (row_indices->type()->id() == arrow::Type::INT64) {
    const arrow::Int64Array* row_indices_array =
        static_cast<const arrow::Int64Array*>(row_indices.get());
    absl::Span<const int64_t> row_indices_span(row_indices_array->raw_values(),
                                               row_indices_array->length());
    return SliceTableByRowIndicesInternal(table, row_indices_span, result);
  } else {
    return errors::InvalidArgument(
        "Expected row_indices to be an Int32Array or an Int64Array");
  }
}

Status TotalByteSize(const Table& table, const bool ignore_unsupported,
                     size_t* result) {
  *result = 0;
  for (int i = 0; i < table.num_columns(); ++i) {
    auto chunked_array = table.column(i);
    for (int j = 0; j < chunked_array->num_chunks(); ++j) {
      size_t array_size;
      auto status = GetByteSize(*chunked_array->chunk(j), &array_size);
      if (ignore_unsupported && status.code() == error::UNIMPLEMENTED) {
        continue;
      }
      TFX_BSL_RETURN_IF_ERROR(status);
      *result += array_size;
    }
  }
  return Status::OK();
}

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
  // TODO(zhuo): Take() can take RecordBatch starting from arrow 0.16.
  std::vector<std::shared_ptr<Array>> columns;
  for (int i = 0; i < record_batch.num_columns(); ++i) {
    columns.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(arrow::compute::Take(
        &ctx, *record_batch.column(i), indices, options, &columns.back())));
  }
  *result = RecordBatch::Make(record_batch.schema(), indices.length(),
                              std::move(columns));
  return Status::OK();
}

}  // namespace tfx_bsl
