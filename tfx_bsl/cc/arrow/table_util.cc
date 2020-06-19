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
using ::arrow::ArrayData;
using ::arrow::Concatenate;
using ::arrow::DataType;
using ::arrow::Field;
using ::arrow::RecordBatch;
using ::arrow::Schema;
using ::arrow::Type;

// TODO(zhuo): Clean this up once tfx-bsl builds exclusively against arrow
// 0.17+.
#if (ARROW_VERSION_MAJOR <= 0) && (ARROW_VERSION_MINOR < 17)
arrow::Status MakeArrayOfNullShim(const std::shared_ptr<DataType>& type,
                                  int64_t length,
                                  std::shared_ptr<Array>* result) {
  return arrow::MakeArrayOfNull(type, length, result);
}

#else

arrow::Status MakeArrayOfNullShim(const std::shared_ptr<DataType>& type,
                                  int64_t length,
                                  std::shared_ptr<Array>* result) {
  auto status_or = arrow::MakeArrayOfNull(type, length);
  ARROW_RETURN_NOT_OK(status_or);
  *result = std::move(status_or).ValueOrDie();
  return arrow::Status::OK();
}

#endif

std::shared_ptr<DataType> GetListValueType(const DataType& list_type) {
  switch (list_type.id()) {
    case Type::LIST:
      return static_cast<const arrow::ListType&>(list_type).value_type();
    case Type::LARGE_LIST:
      return static_cast<const arrow::LargeListType&>(list_type).value_type();
    default:
      return nullptr;
  }
}

// "Merges" array types with the following rules:
// Merge(X, Y) == Merge(Y, X)
// Merge(null, X) = X
// Merge(X, X) = X
// Merge(primitive_type_X, primite_type_Y) --> error
// Merge(list<X>, list<Y>) = list<Merge(X, Y)>
// Merge(struct<fields1>, struct<fields2>) -->
//   for f_name in intersect(fields1_names, fields2_names):
//     merged_field = Merge(fields1[f_name], fields2[f_name])
//   for f_name in fields1_names but not fields2_names
//     merged_field = fields1_names[f_name]
//   for f_name in fields2_names but not fields1_names
//     merged_field = fields2_names[f_name]
// Merge() can be called multiple times, while result() can only be called
// once, after which the object should not be used any more.
class ArrayTypeMerger {
 public:
  explicit ArrayTypeMerger(const std::shared_ptr<DataType>& t)
      : merged_type_(t) {}

  Status Merge(const std::shared_ptr<DataType>& type) {
    if (type->id() == Type::NA) {
      return Status::OK();
    }
    if (merged_type_->id() == Type::NA) {
      merged_type_ = type;
      return Status::OK();
    }
    if (merged_type_->Equals(type)) {
      return Status::OK();
    }
    // None of the two is null and they do not equal. Then at least they should
    // have the same type ID. (because list<> have the same ID as well as
    // struct<>).
    if (merged_type_->id() != type->id()) {
      return errors::FailedPrecondition(
          "Unable to merge incompatible type: ", merged_type_->ToString(),
          " vs ", type->ToString());
    }
    // At this point we are dealing with only list<> and struct<>.
    switch (type->id()) {
      case Type::LIST:
        return MergeListType(static_cast<const arrow::ListType&>(*type));
      case Type::LARGE_LIST:
        return MergeLargeListType(
            static_cast<const arrow::LargeListType&>(*type));
      case Type::STRUCT:
        return MergeStructType(static_cast<const arrow::StructType&>(*type));
      default:
        return errors::Unimplemented(
            "Merging nested type other than list/struct is not supported: ",
            type->ToString());
    }
  }

  std::shared_ptr<DataType> result() {
    return std::move(merged_type_);
  }

  Status MergeListType(const arrow::ListType& t) {
    ArrayTypeMerger m(GetListValueType(*merged_type_));
    TFX_BSL_RETURN_IF_ERROR(m.Merge(t.value_type()));
    merged_type_ = arrow::list(m.result());
    return Status::OK();
  }

  Status MergeLargeListType(const arrow::LargeListType& t) {
    ArrayTypeMerger m(GetListValueType(*merged_type_));
    TFX_BSL_RETURN_IF_ERROR(m.Merge(t.value_type()));
    merged_type_ = arrow::large_list(m.result());
    return Status::OK();
  }

  Status MergeStructType(const arrow::StructType& s) {
    const auto* merged_struct_type =
        static_cast<const arrow::StructType*>(merged_type_.get());

    std::vector<std::shared_ptr<Field>> children =
        merged_struct_type->children();
    absl::flat_hash_map<std::string, int> name_to_index;
    name_to_index.reserve(children.size());
    for (int i = 0; i < children.size(); ++i) {
      name_to_index[children[i]->name()] = i;
    }

    for (const auto& other_field : s.children()) {
      auto pos_and_inserted = name_to_index.insert(
          std::make_pair(other_field->name(), children.size()));
      if (pos_and_inserted.second) {
        children.push_back(other_field);
      } else {
        std::shared_ptr<Field>& field = children.at(
            pos_and_inserted.first->second);
        ArrayTypeMerger m(field->type());
        TFX_BSL_RETURN_IF_ERROR(m.Merge(other_field->type()));
        field = arrow::field(other_field->name(), m.result());
      }
    }
    merged_type_ = arrow::struct_(std::move(children));
    return Status::OK();
  }

 private:
  std::shared_ptr<DataType> merged_type_;
};

Status PromoteArrayDataToType(const std::shared_ptr<ArrayData>& array_data,
                              const std::shared_ptr<DataType>& target_type,
                              std::shared_ptr<ArrayData>* promoted) {
  const auto& current_type = array_data->type;
  if (current_type->Equals(target_type)) {
    *promoted = array_data;
    return Status::OK();
  }
  if (current_type->id() == Type::NA) {
    std::shared_ptr<Array> array_of_null;
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
        MakeArrayOfNullShim(target_type, array_data->length, &array_of_null)));
    *promoted = array_of_null->data();
    return Status::OK();
  }
  if (current_type->id() != target_type->id()) {
    return errors::FailedPrecondition(
        "Unable to promote array to incompatible type: ",
        current_type->ToString(), " vs ", target_type->ToString());
  }
  switch (target_type->id()) {
      case Type::LIST:
      case Type::LARGE_LIST: {
        std::shared_ptr<ArrayData> promoted_value_array_data;
        TFX_BSL_RETURN_IF_ERROR(PromoteArrayDataToType(
            array_data->child_data[0], GetListValueType(*target_type),
            &promoted_value_array_data));

        *promoted = array_data->Copy();
        (*promoted)->type = target_type;
        (*promoted)->child_data[0] = std::move(promoted_value_array_data);
        break;
      }
      case Type::STRUCT: {
        const auto& target_struct_type =
            static_cast<const arrow::StructType&>(*target_type);
        const auto& current_struct_type =
            static_cast<const arrow::StructType&>(*array_data->type);
        const int target_num_children = target_struct_type.num_children();
        std::vector<std::shared_ptr<ArrayData>> child_data(target_num_children);
        for (int i = 0; i < target_num_children; ++i) {
          const std::string& child_name = target_struct_type.child(i)->name();
          const int current_child_index =
              current_struct_type.GetFieldIndex(child_name);
          if (current_child_index < 0) {
            std::shared_ptr<Array> array_of_null;
            TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
                MakeArrayOfNullShim(target_struct_type.child(i)->type(),
                                    array_data->length, &array_of_null)));
            child_data[i] = array_of_null->data();
          } else {
            TFX_BSL_RETURN_IF_ERROR(PromoteArrayDataToType(
                array_data->child_data[current_child_index],
                target_struct_type.child(i)->type(), &child_data[i]));
          }
        }
        *promoted = array_data->Copy();
        (*promoted)->type = target_type;
        (*promoted)->child_data = std::move(child_data);
        break;
      }
      default:
        return errors::Unimplemented("Not implemented");
  }
  return Status::OK();
}

// Promotes `array` to `target_type`.
// The promotion rule is similar to the merge rule (documented above
// `ArrayTypeMerger`).
// When promoting a struct<> to another struct<>, fields that are in the target
// type but not in the current type will have an array of all nulls in the
// place.
Status PromoteArrayToType(const std::shared_ptr<Array>& array,
                          const std::shared_ptr<DataType>& target_type,
                          std::shared_ptr<Array>* promoted) {
  std::shared_ptr<ArrayData> promoted_data;
  TFX_BSL_RETURN_IF_ERROR(
      PromoteArrayDataToType(array->data(), target_type, &promoted_data));
  *promoted = arrow::MakeArray(promoted_data);
  return Status::OK();
}

// Represents a field (i.e. column name and type) in the merged RecordBatch.
class FieldRep {
 public:
  FieldRep(const std::string& field_name, const int64_t leading_nulls)
      : field_name_(field_name) {
    if (leading_nulls > 0) {
      arrays_or_nulls_.push_back(leading_nulls);
    }
  }

  const std::string& field_name() const { return field_name_; }

  // Appends `column` to the column of the field in merged table.
  Status AppendColumn(const std::shared_ptr<Array>& column) {
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

  // Makes a merged array out of columns and nulls appended that can form
  // the column of the field in the merged table.
  Status ToMergedArray(std::shared_ptr<Array>* merged_array) const {
    ArrayTypeMerger m(arrow::null());
    for (const auto& variant : arrays_or_nulls_) {
      if (absl::holds_alternative<std::shared_ptr<Array>>(variant)) {
        TFX_BSL_RETURN_IF_ERROR(
            m.Merge(absl::get<std::shared_ptr<Array>>(variant)->type()));
      }
    }
    std::shared_ptr<DataType> merged_type = m.result();

    std::vector<std::shared_ptr<Array>> arrays_to_concat;
    arrays_to_concat.reserve(arrays_or_nulls_.size());
    for (const auto& variant : arrays_or_nulls_) {
      arrays_to_concat.emplace_back();
      if (absl::holds_alternative<int64_t>(variant)) {
        TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
            MakeArrayOfNullShim(merged_type, absl::get<int64_t>(variant),
                                &arrays_to_concat.back())));
      } else {
        TFX_BSL_RETURN_IF_ERROR(
            PromoteArrayToType(absl::get<std::shared_ptr<Array>>(variant),
                               merged_type, &arrays_to_concat.back()));
      }
    }
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(Concatenate(
        arrays_to_concat, arrow::default_memory_pool(), merged_array)));
    return Status::OK();
  }

  std::string field_name_;
  std::vector<absl::variant<std::shared_ptr<Array>, int64_t>> arrays_or_nulls_;
};

}  // namespace

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
      const std::string& field_name = f->name();
      auto iter = field_index_by_field_name.find(field_name);
      if (iter == field_index_by_field_name.end()) {
        std::tie(iter, std::ignore) = field_index_by_field_name.insert(
            std::make_pair(field_name, field_rep_by_field_index.size()));
        field_rep_by_field_index.emplace_back(field_name, total_num_rows);
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
    merged_arrays.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(field_rep.ToMergedArray(&merged_arrays.back()));
    fields.push_back(
        arrow::field(field_rep.field_name(), merged_arrays.back()->type()));
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
