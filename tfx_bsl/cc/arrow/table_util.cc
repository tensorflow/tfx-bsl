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
namespace {
using ::arrow::Array;
using ::arrow::ChunkedArray;
using ::arrow::Concatenate;
using ::arrow::Field;
using ::arrow::RecordBatch;
using ::arrow::Schema;
using ::arrow::Table;
using ::arrow::Type;


// Represents a field (i.e. column name and type) in the merged table.
class FieldRep {
 public:
  FieldRep(std::shared_ptr<Field> field, const int64_t leading_nulls)
      : field_(std::move(field)) {
    if (leading_nulls > 0) {
      arrays_or_nulls_.push_back(leading_nulls);
    }
  }

  // Appends `column` to the column of the field in merged table.
  Status AppendColumn(const std::shared_ptr<Array>& column) {
    const auto& this_type = field_->type();
    const auto& other_type = column->type();
    if (other_type->id() == Type::NA) {
      AppendNulls(column->length());
      return Status::OK();
    }
    if (this_type->id() == Type::NA) {
      field_ = arrow::field(field_->name(), other_type);
    } else if (!this_type->Equals(other_type)) {
      return errors::InvalidArgument(
          absl::StrCat("Trying to append a column of different type. ",
                       "Column name: ", field_->name(),
                       " , Current type: ", this_type->ToString(),
                       " , New type: ", other_type->ToString()));
    }
    arrays_or_nulls_.push_back(column);
    return Status::OK();
  }

  // Appends `num_nulls` nulls to the column of the field in the merged table.
  void AppendNulls(const int64_t num_nulls) {
    if (arrays_or_nulls_.empty() ||
        absl::holds_alternative<std::shared_ptr<Array>>(
            arrays_or_nulls_.back())) {
      arrays_or_nulls_.push_back(num_nulls);
    } else {
      absl::get<int64_t>(arrays_or_nulls_.back()) += num_nulls;
    }
  }

  const std::shared_ptr<Field>& field() const {
    return field_;
  }

  // Makes a merged array out of columns and nulls appended that can form
  // the column of the field in the merged table.
  Status ToMergedArray(std::shared_ptr<Array>* merged_array) const {
    if (field_->type()->id() == Type::NA) {
      arrow::NullBuilder null_builder;
      if (!arrays_or_nulls_.empty()) {
        TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(null_builder.AppendNulls(
            absl::get<int64_t>(arrays_or_nulls_.back()))));
      }
      TFX_BSL_RETURN_IF_ERROR(
          FromArrowStatus(null_builder.Finish(merged_array)));
    } else {
      std::vector<std::shared_ptr<Array>> arrays_to_merge;
      for (const auto& array_or_num_nulls : arrays_or_nulls_) {
        std::shared_ptr<Array> array;
        if (absl::holds_alternative<int64_t>(array_or_num_nulls)) {
          TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(arrow::MakeArrayOfNull(
              field_->type(), absl::get<int64_t>(array_or_num_nulls), &array)));
        } else {
          array = absl::get<std::shared_ptr<Array>>(array_or_num_nulls);
        }
        arrays_to_merge.push_back(std::move(array));
      }
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(Concatenate(
          arrays_to_merge, arrow::default_memory_pool(), merged_array)));
    }
    return Status::OK();
  }

  std::shared_ptr<Field> field_;
  std::vector<absl::variant<std::shared_ptr<Array>, int64_t>> arrays_or_nulls_;
};

}  // namespace

// TODO(zhuo): This can be replaced by
// pa.concat_tables(tables_from_rbs, promote=True)
//   .combine_chunks()
//   .to_batches()[0].
// Post arrow 0.17
Status MergeRecordBatches(
    const std::vector<std::shared_ptr<RecordBatch>>& record_batches,
    std::shared_ptr<RecordBatch>* result) {
  absl::flat_hash_map<std::string, int> field_index_by_field_name;
  std::vector<FieldRep> field_rep_by_field_index;
  int64_t total_num_rows = 0;
  for (const auto& rb : record_batches) {
    std::vector<bool> field_seen_by_field_index(field_rep_by_field_index.size(),
                                                false);
    for (int i = 0; i < rb->schema()->num_fields(); ++i) {
      const auto& f = rb->schema()->field(i);
      auto iter = field_index_by_field_name.find(f->name());
      if (iter == field_index_by_field_name.end()) {
        std::tie(iter, std::ignore) = field_index_by_field_name.insert(
            std::make_pair(f->name(), field_rep_by_field_index.size()));
        field_rep_by_field_index.emplace_back(f, total_num_rows);
        field_seen_by_field_index.push_back(true);
      }
      field_seen_by_field_index[iter->second] = true;
      FieldRep& field_rep = field_rep_by_field_index[iter->second];
      TFX_BSL_RETURN_IF_ERROR(field_rep.AppendColumn(rb->column(i)));
    }
    for (int i = 0; i < field_seen_by_field_index.size(); ++i) {
      if (!field_seen_by_field_index[i]) {
        field_rep_by_field_index[i].AppendNulls(rb->num_rows());
      }
    }
    total_num_rows += rb->num_rows();
  }

  std::vector<std::shared_ptr<Field>> fields;
  arrow::ArrayVector merged_arrays;
  for (const FieldRep& field_rep : field_rep_by_field_index) {
    fields.push_back(field_rep.field());
    std::shared_ptr<Array> merged_array;
    TFX_BSL_RETURN_IF_ERROR(field_rep.ToMergedArray(&merged_array));
    merged_arrays.push_back(std::move(merged_array));
  }
  *result = RecordBatch::Make(std::make_shared<Schema>(std::move(fields)),
                              total_num_rows, merged_arrays);
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
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      arrow::compute::Take(&ctx, record_batch, indices, options, result)));
  return Status::OK();
}
}  // namespace tfx_bsl
