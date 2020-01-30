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
#include "arrow/visitor_inline.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace {
using ::arrow::AllocateEmptyBitmap;
using ::arrow::Array;
using ::arrow::ArrayData;
using ::arrow::ArrayVector;
using ::arrow::Buffer;
using ::arrow::ChunkedArray;
using ::arrow::Concatenate;
using ::arrow::DataType;
using ::arrow::Field;
using ::arrow::Schema;
using ::arrow::Table;
using ::arrow::Type;

// arrow::Column has been removed since 0.15.
#if (ARROW_VERSION_MAJOR <= 0) && (ARROW_VERSION_MINOR <= 14)
// If a column contains multiple chunks, concatenates those chunks into one and
// makes a new column out of it.
Status CompactColumn(const arrow::Column& column,
                     std::shared_ptr<arrow::Column>* compacted) {
  std::shared_ptr<Array> merged_data_array;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      Concatenate(column.data()->chunks(), arrow::default_memory_pool(),
                  &merged_data_array)));
  *compacted =
      std::make_shared<arrow::Column>(column.field(), merged_data_array);
  return Status::OK();
}

Status CompactTable(const Table& table, std::shared_ptr<Table>* compacted) {
  std::vector<std::shared_ptr<arrow::Column>> compacted_columns;
  for (int i = 0; i < table.num_columns(); ++i) {
    std::shared_ptr<arrow::Column> compacted_column;
    TFX_BSL_RETURN_IF_ERROR(CompactColumn(*table.column(i), &compacted_column));
    compacted_columns.push_back(std::move(compacted_column));
  }
  *compacted = Table::Make(table.schema(), compacted_columns);
  return Status::OK();
}

std::shared_ptr<ChunkedArray> GetChunkedArrayFromColumn(
    const std::shared_ptr<arrow::Column>& column) {
  return column->data();
}

std::shared_ptr<Table> SliceTable(const Table& table, int64_t offset,
                                  int64_t length) {
  const int num_columns = table.num_columns();
  std::vector<std::shared_ptr<arrow::Column>> sliced_columns(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    sliced_columns[i] = table.column(i)->Slice(offset, length);
  }
  return Table::Make(table.schema(), sliced_columns, length);
}

#else

Status CompactTable(const Table& table, std::shared_ptr<Table>* compacted) {
  return FromArrowStatus(
      table.CombineChunks(arrow::default_memory_pool(), compacted));
}

std::shared_ptr<ChunkedArray> GetChunkedArrayFromColumn(
    const std::shared_ptr<ChunkedArray>& column) {
  return column;
}

std::shared_ptr<Table> SliceTable(const Table& table, int64_t offset,
                                  int64_t length) {
  return table.Slice(offset, length);
}

#endif

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

// Makes arrays of nulls of various types.
class ArrayOfNullsMaker {
 public:
  ArrayOfNullsMaker(const std::shared_ptr<DataType>& type,
                    const int64_t num_nulls) {
    out_ = std::make_shared<ArrayData>(type, num_nulls, num_nulls);
  }

  arrow::Status Make(std::shared_ptr<ArrayData>* out) {
    RETURN_NOT_OK(arrow::VisitTypeInline(*out_->type, this));
    *out = std::move(out_);
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::NullType&) { return arrow::Status::OK(); }

  arrow::Status Visit(const arrow::BooleanType&) {
    std::shared_ptr<Buffer> bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&bitmap));
    out_->buffers = {bitmap, bitmap};
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::FixedWidthType& fixed) {
    if (fixed.bit_width() % 8 != 0) {
      return arrow::Status::Invalid(
          "Invalid bit width. Expected single byte entries");
    }
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    std::shared_ptr<Buffer> data;
    RETURN_NOT_OK(
        AllocateBuffer(out_->null_count * fixed.bit_width() / 8, &data));
    out_->buffers = {null_bitmap, data};
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::BinaryType&) {
    return SetArrayDataForListAlike(/*type_has_data_buffer=*/true);
  }

  arrow::Status Visit(const arrow::ListType& l) {
    RETURN_NOT_OK(SetArrayDataForListAlike(/*type_has_data_buffer=*/false));
    out_->child_data.emplace_back();
    RETURN_NOT_OK(
        ArrayOfNullsMaker(l.value_type(), 0).Make(&out_->child_data.back()));
    return arrow::Status::OK();
  }

#if (ARROW_VERSION_MAJOR > 0) || (ARROW_VERSION_MINOR >= 14)
  arrow::Status Visit(const arrow::FixedSizeListType& l) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", l.ToString()));
  }
#endif

#if (ARROW_VERSION_MAJOR > 0) || (ARROW_VERSION_MINOR >= 15)
  arrow::Status Visit(const arrow::LargeStringType& l) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", l.ToString()));
  }
  arrow::Status Visit(const arrow::LargeBinaryType& l) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", l.ToString()));
  }
  arrow::Status Visit(const arrow::LargeListType& l) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", l.ToString()));
  }
#endif

  arrow::Status Visit(const arrow::StructType& s) {
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    out_->buffers = {null_bitmap};
    for (int i = 0; i < s.num_children(); ++i) {
      std::shared_ptr<ArrayData> child_data;
      RETURN_NOT_OK(ArrayOfNullsMaker(s.child(i)->type(), out_->null_count)
                        .Make(&child_data));
      out_->child_data.push_back(std::move(child_data));
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::DictionaryType& d) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", d.ToString()));
  }

  arrow::Status Visit(const arrow::UnionType& u) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", u.ToString()));
  }

  arrow::Status Visit(const arrow::ExtensionType& e) {
    return arrow::Status::NotImplemented(
        absl::StrCat("Make array of nulls: ", e.ToString()));
  }

 private:
  arrow::Status AllocateNullBitmap(std::shared_ptr<Buffer>* buf) const {
    RETURN_NOT_OK(AllocateEmptyBitmap(arrow::default_memory_pool(),
                                      out_->null_count, buf));
    return arrow::Status::OK();
  }

  arrow::Status AllocateNullOffsets(std::shared_ptr<Buffer>* buf) const {
    RETURN_NOT_OK(AllocateBuffer(arrow::default_memory_pool(),
                                 (out_->null_count + 1) * sizeof(int32_t),
                                 buf));
    // The offsets of an array of nulls are all 0.
    memset((*buf)->mutable_data(), 0, static_cast<size_t>((*buf)->size()));
    return arrow::Status::OK();
  }

  arrow::Status SetArrayDataForListAlike(const bool type_has_data_buffer) {
    std::shared_ptr<Buffer> null_bitmap;
    RETURN_NOT_OK(AllocateNullBitmap(&null_bitmap));
    std::shared_ptr<Buffer> offsets;
    RETURN_NOT_OK(AllocateNullOffsets(&offsets));
    out_->buffers = {null_bitmap, offsets};
    if (type_has_data_buffer) {
      out_->buffers.push_back(nullptr);
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<ArrayData> out_;
};

// Makes an Array of `type` and `length` which contains all nulls.
// Only supports ListArray for now.
// TODO(zhuo): this can be replaced by arrow::MakeArrayOfNull when 0.15 comes.
Status MakeArrayOfNulls(const std::shared_ptr<DataType>& type, int64_t length,
                        std::shared_ptr<Array>* result) {
  std::shared_ptr<ArrayData> array_data;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(ArrayOfNullsMaker(type, length).Make(&array_data)));
  *result = arrow::MakeArray(array_data);
  return Status::OK();
}

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
  Status AppendColumn(const ChunkedArray& column) {
    const auto& this_type = field_->type();
    const auto& other_type = column.type();
    if (other_type->id() == Type::NA) {
      AppendNulls(column.length());
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
    arrays_or_nulls_.insert(arrays_or_nulls_.end(),
                            column.chunks().begin(),
                            column.chunks().end());
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
          TFX_BSL_RETURN_IF_ERROR(MakeArrayOfNulls(
              field_->type(), absl::get<int64_t>(array_or_num_nulls), &array));
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

Status MergeTables(const std::vector<std::shared_ptr<Table>>& tables,
                   std::shared_ptr<Table>* result) {
  absl::flat_hash_map<std::string, int> field_index_by_field_name;
  std::vector<FieldRep> field_rep_by_field_index;
  int64_t total_num_rows = 0;
  for (const auto& t : tables) {
    std::vector<bool> field_seen_by_field_index(field_rep_by_field_index.size(),
                                                false);
    for (int i = 0; i < t->schema()->num_fields(); ++i) {
      const auto& f = t->schema()->field(i);
      auto iter = field_index_by_field_name.find(f->name());
      if (iter == field_index_by_field_name.end()) {
        std::tie(iter, std::ignore) = field_index_by_field_name.insert(
            std::make_pair(f->name(), field_rep_by_field_index.size()));
        field_rep_by_field_index.emplace_back(f, total_num_rows);
        field_seen_by_field_index.push_back(true);
      }
      field_seen_by_field_index[iter->second] = true;
      FieldRep& field_rep = field_rep_by_field_index[iter->second];
      TFX_BSL_RETURN_IF_ERROR(
          field_rep.AppendColumn(*GetChunkedArrayFromColumn(t->column(i))));
    }
    for (int i = 0; i < field_seen_by_field_index.size(); ++i) {
      if (!field_seen_by_field_index[i]) {
        field_rep_by_field_index[i].AppendNulls(t->num_rows());
      }
    }
    total_num_rows += t->num_rows();
  }

  std::vector<std::shared_ptr<Field>> fields;
  ArrayVector merged_arrays;
  for (const FieldRep& field_rep : field_rep_by_field_index) {
    fields.push_back(field_rep.field());
    std::shared_ptr<Array> merged_array;
    TFX_BSL_RETURN_IF_ERROR(field_rep.ToMergedArray(&merged_array));
    merged_arrays.push_back(std::move(merged_array));
  }
  *result =
      Table::Make(std::make_shared<Schema>(std::move(fields)),
                  merged_arrays, /*num_rows=*/total_num_rows);
  return Status::OK();
}

Status SliceTableByRowIndices(const std::shared_ptr<Table>& table,
                              const std::shared_ptr<Array>& row_indices,
                              std::shared_ptr<Table>* result) {
  if (row_indices->type()->id() != arrow::Type::INT32) {
    return errors::InvalidArgument("Expected row_indices to be a Int32Array");
  }
  const arrow::Int32Array* row_indices_int32_array =
      static_cast<const arrow::Int32Array*>(row_indices.get());
  absl::Span<const int32_t> row_indices_span(
      row_indices_int32_array->raw_values(), row_indices_int32_array->length());

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
    table_slices.push_back(
        SliceTable(*table, row_indices_span[begin], end - begin));
    begin = end;
  }

  // Make sure to never return a table with non-zero offset (that is a slice of
  // another table). This is needed because Array.flatten() is buggy and does
  // not handle offsets correctly.
  // TODO(zhuo): Remove once https://github.com/apache/arrow/pull/6006 is
  // available.
  if (table_slices.size() == 1) {
    table_slices.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(GetEmptyTableLike(*table, &table_slices.back()));
  }

  std::shared_ptr<Table> concatenated;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(arrow::ConcatenateTables(table_slices, &concatenated)));
  return CompactTable(*concatenated, result);
}

}  // namespace tfx_bsl
