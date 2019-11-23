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

#include "arrow/array/concatenate.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
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

// Makes a ListArray that contains the given one element (sub-list).
Status MakeSingletonListArray(const std::shared_ptr<Array>& element,
                              std::shared_ptr<Array>* list_array) {
  Int32Builder offsets_builder;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(offsets_builder.Reserve(2)));
  offsets_builder.UnsafeAppend(0);
  offsets_builder.UnsafeAppend(element->length());
  std::shared_ptr<Array> offsets_array;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(offsets_builder.Finish(&offsets_array)));

  return FromArrowStatus(ListArray::FromArrays(
      *offsets_array, *element, arrow::default_memory_pool(), list_array));
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

Status CooFromListArray(
    const std::shared_ptr<arrow::Array>& list_array,
    std::shared_ptr<arrow::Array>* coo_array,
    std::shared_ptr<arrow::Array>* dense_shape_array) {
  // A ListArray encodes its list structure using offsets, or "row splits"
  // where [row_splits[i], row_splits[i+1]) are the indices of values of
  // the i-th sub-list. For example:
  // [[a, b], [], [c]] is encoded as:
  // value_array: [a, b, c]
  // row_splits: [0, 2, 2, 3]
  // A k-nested ListArray is encoded recursively as a row_splits array
  // and a (k-1)-Nested ListArray (or a primitive array if k==0). A 1-nested
  // ListArray is a ListArray<primitive>.

  std::vector<absl::Span<const int32_t>> nested_row_splits;
  std::array<int32_t, 2> dummy_outermost_row_splits = {
      0, static_cast<int32_t>(list_array->length())};
  nested_row_splits.push_back(absl::MakeSpan(dummy_outermost_row_splits));

  // Strip `list_array` and populate `nested_row_splits` with row_splits of
  // each level.
  std::shared_ptr<arrow::Array> values = list_array;
  while (values->type()->id() == arrow::Type::LIST) {
    ListArray* list_array = static_cast<ListArray*>(values.get());
    absl::Span<const int32_t> row_splits = absl::MakeSpan(
        list_array->raw_value_offsets(), list_array->length() + 1);
    nested_row_splits.emplace_back(row_splits);
    // Note that the values array is not sliced even if `list_array` is, so
    // we slice it here.
    values = list_array->values()->Slice(
        row_splits.front(), row_splits.back() - row_splits.front());
  }

  // Allocate a buffer for the coordinates. A k-nested ListArray will be
  // converted to a sparse tensor of k+1 dimensions. The buffer for the
  // coordinates will contain all the coordinates concatenated, so it needs to
  // hold (k + 1) * num_values numbers.
  std::shared_ptr<arrow::Buffer> coo_buffer;
  const size_t coo_length = nested_row_splits.size();
  const size_t coo_buffer_size =
      coo_length * values->length() * sizeof(int64_t);
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(arrow::AllocateBuffer(
      arrow::default_memory_pool(), coo_buffer_size, &coo_buffer)));
  int64_t* coo_flat = reinterpret_cast<int64_t*>(coo_buffer->mutable_data());

  // Increments *idx until
  // offset_row_splits[*idx] <= v < offset_row_splits[*idx + 1].
  // returns row_split (with offset applied) at *idx. row_splits should not be
  // empty.
  const auto lookup_and_update = [](const int32_t v,
                                    const absl::Span<const int32_t> row_splits,
                                    size_t* idx) -> int32_t {
    // Note that row_splits does not always start with 0, as the ListArray
    // can be sliced (which causes the row_splits to also be sliced). In that
    // case, the slicing offset is row_splits[0].
    const int32_t row_split_offset = row_splits[0];
    while (*idx < row_splits.size() - 1) {
      const int32_t begin = row_splits[*idx] - row_split_offset;
      const int32_t end = row_splits[*idx + 1] - row_split_offset;
      if (v >= begin && v < end) break;
      ++(*idx);
    }
    return row_splits[*idx] - row_split_offset;
  };

  // COO for the `values`[i] is [x, ..., y, z] if `values`[i] is the z-th
  // element in its belonging sub-list, which is the y-th element in its
  // belonging sub-list,... which is the x-th element in its belonging sub-list,
  // which is the only element in the outermost, dummy "ListArray" denoted by
  // `dummy_outermost_row_splits`.
  //
  // Given i, the the index of an element in the value array of a ListArray,
  // its belonging sub-list is the j-th one, if
  // row_splits[j] <= i < row_splits[j + 1]. And i - row_splits[j] is that
  // element's position in the sub-list, thus the coordinate.

  // This vector stores the indices of the sub-lists at each level the last
  // leaf value belongs to.
  // Note that elements in this vector is non-decreasing as we go through the
  // values in order. That's why it persists outside the loop and gets updated
  // by lookup_and_update().
  std::vector<size_t> current_owning_sublist_indices(
      nested_row_splits.size(), 0);
  for (size_t i = 0; i < values->length(); ++i) {
    int64_t* current_coo = coo_flat + i * coo_length;
    int32_t current_idx = i;
    // The inner loop looks for the index in the belonging sub-list at each
    // level.
    for (int j = nested_row_splits.size() - 1; j >= 0; --j) {
      const int32_t row_split_begin = lookup_and_update(
          current_idx, nested_row_splits[j],
          &current_owning_sublist_indices[j]);
      current_coo[j] = current_idx - row_split_begin;
      current_idx = current_owning_sublist_indices[j];
    }
  }

  // The dense shape is the bounding box of the ListArray: the maximum lengths
  // of sub-lists in each level.
  arrow::Int64Builder dense_shape_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(dense_shape_builder.Reserve(coo_length)));
  for (const absl::Span<const int32_t> row_splits : nested_row_splits) {
    int32_t dimension_size = 0;
    for (int i = 0; i < row_splits.size() - 1; ++i) {
      dimension_size =
          std::max(dimension_size, row_splits[i + 1] - row_splits[i]);
    }
    dense_shape_builder.UnsafeAppend(dimension_size);
  }

  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(dense_shape_builder.Finish(dense_shape_array)));

  *coo_array = std::make_shared<arrow::Int64Array>(
      coo_length * values->length(), coo_buffer);
  return Status::OK();
}

Status FillNullLists(const std::shared_ptr<Array>& list_array,
                     const std::shared_ptr<Array>& fill_with,
                     std::shared_ptr<Array>* filled) {
  std::shared_ptr<arrow::DataType> type = list_array->type();
  if (type->id() != arrow::Type::LIST) {
    return errors::InvalidArgument(absl::StrCat(
        "Expected a ListArray but got: ", type->ToString()));
  }
  std::shared_ptr<arrow::DataType> value_type =
      static_cast<arrow::ListType*>(type.get())->value_type();
  if (!value_type->Equals(fill_with->type())) {
    return errors::InvalidArgument(absl::StrCat(
        "Expected a `fill_with` to be of the same type as "
        "`list_array`'s value_type, ",
        value_type->ToString(), " but got: ", fill_with->type()->ToString()));
  }

  std::shared_ptr<Array> singleton_list;
  TFX_BSL_RETURN_IF_ERROR(MakeSingletonListArray(fill_with, &singleton_list));
  std::vector<std::shared_ptr<Array>> array_fragments;
  int64_t begin = 0, end = 0;
  while (end < list_array->length()) {
    if (list_array->IsNull(end)) {
      if (begin != end) {
        array_fragments.push_back(list_array->Slice(begin, end - begin));
      }
      array_fragments.push_back(singleton_list);
      ++end;
      begin = end;
    } else {
      ++end;
    }
  }
  if (begin != end) {
    array_fragments.push_back(list_array->Slice(begin, end - begin));
  }

  if (array_fragments.empty()) {
    *filled = list_array;
    return Status::OK();
  }

  return FromArrowStatus(arrow::Concatenate(
      array_fragments, arrow::default_memory_pool(), filled));
}

}  // namespace tfx_bsl
