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
#include "tfx_bsl/cc/arrow/array_util.h"

#include "absl/strings/str_cat.h"
#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace {
using ::arrow::Array;
using ::arrow::ListArray;
using ::arrow::Int32Builder;

Status GetListArray(const Array& array, const ListArray** list_array) {
  if (array.type()->id() != arrow::Type::LIST) {
    return errors::InvalidArgument(absl::StrCat(
        "Expected ListArray but got type id: ", array.type()->ToString()));
  }
  *list_array = static_cast<const ListArray*>(&array);
  return Status::OK();
}

class ElementLengthsVisitor : public arrow::ArrayVisitor {
 public:
  ElementLengthsVisitor() {}
  ~ElementLengthsVisitor() override {}
  std::shared_ptr<Array> result() const { return result_; }
  arrow::Status Visit(const arrow::StringArray& string_array) override {
    return VisitInternal(string_array);
  }
  arrow::Status Visit(const arrow::BinaryArray& binary_array) override {
    return VisitInternal(binary_array);
  }
  arrow::Status Visit(const arrow::ListArray& list_array) override {
    return VisitInternal(list_array);
  }

 private:
  std::shared_ptr<Array> result_;

  template <class ListLikeArray>
  arrow::Status VisitInternal(const ListLikeArray& array) {
    Int32Builder lengths_builder;
    ARROW_RETURN_NOT_OK(lengths_builder.Reserve(array.length()));
    for (int i = 0; i < array.length(); ++i) {
      lengths_builder.UnsafeAppend(array.value_length(i));
    }
    return lengths_builder.Finish(&result_);
  }
};

}  // namespace

Status GetElementLengths(
    const Array& array,
    std::shared_ptr<Array>* list_lengths_array) {

  ElementLengthsVisitor v;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(array.Accept(&v)));
  *list_lengths_array = v.result();
  return Status::OK();
}

Status GetFlattenedArrayParentIndices(
    const Array& array,
    std::shared_ptr<Array>* parent_indices_array) {
  const ListArray* list_array;
  TFX_BSL_RETURN_IF_ERROR(GetListArray(array, &list_array));
  arrow::Int32Builder indices_builder;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      indices_builder.Reserve(list_array->value_offset(list_array->length()) -
                              list_array->value_offset(0))));
  for (int i = 0; i < list_array->length(); ++i) {
    const int range_begin = list_array->value_offset(i);
    const int range_end = list_array->value_offset(i + 1);
    for (int j = range_begin; j < range_end; ++j) {
      indices_builder.UnsafeAppend(i);
    }
  }
  return FromArrowStatus(indices_builder.Finish(parent_indices_array));
}

Status GetArrayNullBitmapAsByteArray(
    const Array& array,
    std::shared_ptr<Array>* byte_array) {
  arrow::UInt8Builder masks_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(masks_builder.Reserve(array.length())));
  // array.null_count() might be O(n). However array.data()->null_count
  // is just a number (although it can be kUnknownNullCount in which case
  // the else branch is followed).
  if (array.null_bitmap_data() == nullptr || array.data()->null_count == 0) {
    for (int i = 0; i < array.length(); ++i) {
      masks_builder.UnsafeAppend(0);
    }
  } else {
    for (int i = 0; i < array.length(); ++i) {
      masks_builder.UnsafeAppend(static_cast<uint8_t>(array.IsNull(i)));
    }
  }
  return FromArrowStatus(masks_builder.Finish(byte_array));
}

Status GetBinaryArrayTotalByteSize(const arrow::Array& array,
                                   size_t* total_byte_size) {
  // StringArray is a subclass of BinaryArray.
  if (!(array.type_id() == arrow::Type::BINARY ||
        array.type_id() == arrow::Type::STRING)) {
    return errors::InvalidArgument(
        absl::StrCat("Expected BinaryArray (or StringArray) but got: ",
                     array.type()->ToString()));
  }
  const arrow::BinaryArray* binary_array =
      static_cast<const arrow::BinaryArray*>(&array);
  *total_byte_size = binary_array->value_offset(binary_array->length()) -
         binary_array->value_offset(0);
  return Status::OK();
}

Status ValueCounts(const std::shared_ptr<arrow::Array>& array,
                   std::shared_ptr<arrow::Array>* values_and_counts_array) {
  arrow::compute::FunctionContext ctx;
  std::shared_ptr<arrow::Array> result;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
      arrow::compute::ValueCounts(&ctx, array, values_and_counts_array)));
  return Status::OK();
}

Status MakeListArrayFromParentIndicesAndValues(
    const size_t num_parents,
    const std::shared_ptr<arrow::Array>& parent_indices,
    const std::shared_ptr<Array>& values,
    std::shared_ptr<Array>* out) {
  if (parent_indices->type()->id() != arrow::Type::INT64) {
    return errors::InvalidArgument("Parent indices array must be int64.");
  }
  const size_t length = parent_indices->length();
  if (values->length() != length) {
    return errors::InvalidArgument(
        "values array and parent indices array must be of the same length: ",
        values->length(), " v.s. ", parent_indices->length());
  }
  const auto& parent_indices_int64 =
      *static_cast<const arrow::Int64Array*>(parent_indices.get());
  if (length != 0 && num_parents < parent_indices_int64.Value(length - 1) + 1) {
    return errors::InvalidArgument(absl::StrCat(
        "Found a parent index ", parent_indices_int64.Value(length - 1),
        " while num_parents was ", num_parents));
  }

  arrow::TypedBufferBuilder<bool> null_bitmap_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(null_bitmap_builder.Reserve(num_parents)));
  arrow::TypedBufferBuilder<int32_t> offsets_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(offsets_builder.Reserve(num_parents + 1)));

  offsets_builder.UnsafeAppend(0);
  for (int i = 0, current_pi = 0; i < num_parents; ++i) {
    if (current_pi >= parent_indices_int64.length() ||
        parent_indices_int64.Value(current_pi) != i) {
      null_bitmap_builder.UnsafeAppend(false);
    } else {
      while (current_pi < parent_indices_int64.length() &&
             parent_indices_int64.Value(current_pi) == i) {
        ++current_pi;
      }
      null_bitmap_builder.UnsafeAppend(true);
    }
    offsets_builder.UnsafeAppend(current_pi);
  }

  const int64_t null_count = null_bitmap_builder.false_count();
  std::shared_ptr<arrow::Buffer> null_bitmap_buffer;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(null_bitmap_builder.Finish(&null_bitmap_buffer)));
  std::shared_ptr<arrow::Buffer> offsets_buffer;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(offsets_builder.Finish(&offsets_buffer)));

  *out = std::make_shared<ListArray>(arrow::list(values->type()), num_parents,
                                     offsets_buffer, values, null_bitmap_buffer,
                                     null_count, /*offset=*/0);
  return Status::OK();
}

}  // namespace tfx_bsl
