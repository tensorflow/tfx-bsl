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
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "arrow/api.h"
#include "arrow/compute/api.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

// TODO(b/171748040): clean this up.
#if ARROW_VERSION_MAJOR < 1
#include "arrow/compute/kernels/match.h"
#endif

namespace tfx_bsl {
namespace {
using ::arrow::Array;
using ::arrow::LargeListArray;
using ::arrow::ListArray;
using ::arrow::Int32Builder;
using ::arrow::Int64Builder;

// TODO(b/171748040): clean this up.
#if ARROW_VERSION_MAJOR < 1
using Datum = ::arrow::compute::Datum;

Status IndexInShim(
    const Datum& values, const Datum& value_set, Datum* result) {
  ::arrow::compute::FunctionContext ctx;
  return FromArrowStatus(
      arrow::compute::Match(&ctx, values, value_set, result));
}

Status ValueCountsShim(const Datum& values, std::shared_ptr<Array>* result) {
  ::arrow::compute::FunctionContext ctx;
  return FromArrowStatus(arrow::compute::ValueCounts(&ctx, values, result));
}
#else
using Datum = ::arrow::Datum;

Status IndexInShim(
    const Datum& values, const Datum& value_set, Datum* result) {
  TFX_BSL_ASSIGN_OR_RETURN_ARROW(*result,
                                 arrow::compute::IndexIn(values, value_set));
  return Status::OK();
}

Status ValueCountsShim(const Datum& values, std::shared_ptr<Array>* result) {
  TFX_BSL_ASSIGN_OR_RETURN_ARROW(*result, arrow::compute::ValueCounts(values));
  return Status::OK();
}
#endif

std::unique_ptr<Int32Builder> GetOffsetsBuilder(const arrow::ListArray&) {
  return absl::make_unique<Int32Builder>();
}
std::unique_ptr<Int64Builder> GetOffsetsBuilder(const arrow::LargeListArray&) {
  return absl::make_unique<Int64Builder>();
}

class ElementLengthsVisitor : public arrow::ArrayVisitor {
 public:
  ElementLengthsVisitor() = default;
  std::shared_ptr<Array> result() const { return result_; }
  arrow::Status Visit(const arrow::StringArray& string_array) override {
    return VisitInternal(string_array);
  }
  arrow::Status Visit(
      const arrow::LargeStringArray& large_string_array) override {
    return VisitInternal(large_string_array);
  }
  arrow::Status Visit(const arrow::BinaryArray& binary_array) override {
    return VisitInternal(binary_array);
  }
  arrow::Status Visit(
      const arrow::LargeBinaryArray& large_binary_array) override {
    return VisitInternal(large_binary_array);
  }
  arrow::Status Visit(const arrow::ListArray& list_array) override {
    return VisitInternal(list_array);
  }
  arrow::Status Visit(const arrow::LargeListArray& large_list_array) override {
    return VisitInternal(large_list_array);
  }

 private:
  std::shared_ptr<Array> result_;

  template <class ListLikeArray>
  arrow::Status VisitInternal(const ListLikeArray& array) {
    Int64Builder lengths_builder;
    ARROW_RETURN_NOT_OK(lengths_builder.Reserve(array.length()));
    for (int i = 0; i < array.length(); ++i) {
      lengths_builder.UnsafeAppend(array.value_length(i));
    }
    return lengths_builder.Finish(&result_);
  }
};

class GetFlattenedArrayParentIndicesVisitor : public arrow::ArrayVisitor {
 public:
  GetFlattenedArrayParentIndicesVisitor() = default;
  arrow::Status Visit(const arrow::ListArray& list_array) override {
    return VisitInternal(list_array);
  }
  arrow::Status Visit(const arrow::LargeListArray& large_list_array) override {
    return VisitInternal(large_list_array);
  }

  std::shared_ptr<Array> result() const { return result_; }

 private:
  template <class ListLikeArray>
  arrow::Status VisitInternal(const ListLikeArray& array) {
    auto lengths_builder = GetOffsetsBuilder(array);
    const size_t num_parent_indices =
        array.value_offset(array.length()) - array.value_offset(0);
    ARROW_RETURN_NOT_OK(lengths_builder->Reserve(num_parent_indices));
    for (size_t i = 0; i < array.length(); ++i) {
      const auto range_begin = array.value_offset(i);
      const auto range_end = array.value_offset(i + 1);
      if (range_begin > range_end) {
        return arrow::Status::Invalid(
            "Out-of-order ListArray offsets encountered at index ", i,
            ". This should never happen!");
      }
      for (size_t j = range_begin; j < range_end; ++j) {
        lengths_builder->UnsafeAppend(i);
      }
    }
    return lengths_builder->Finish(&result_);
  }
  std::shared_ptr<Array> result_;
};

class FillNullListsVisitor : public arrow::ArrayVisitor {
 public:
  FillNullListsVisitor(const Array& fill_with) : fill_with_(fill_with) {}

  arrow::Status Visit(const arrow::ListArray& list_array) override {
    return VisitInternal(list_array);
  }
  arrow::Status Visit(const arrow::LargeListArray& large_list_array) override {
    return VisitInternal(large_list_array);
  }

  std::shared_ptr<Array> result() const { return result_; }

 private:
  template <class ListLikeArray>
  arrow::Status MakeSingletonListArray(const ListLikeArray& list_array,
                                       const Array& element,
                                       std::shared_ptr<Array>* result) {
    auto offsets_builder = GetOffsetsBuilder(list_array);
    ARROW_RETURN_NOT_OK(offsets_builder->Reserve(2));
    offsets_builder->UnsafeAppend(0);
    offsets_builder->UnsafeAppend(element.length());
    std::shared_ptr<Array> offsets_array;
    ARROW_RETURN_NOT_OK(offsets_builder->Finish(&offsets_array));

    ARROW_ASSIGN_OR_RAISE(
        *result, ListLikeArray::FromArrays(*offsets_array, element,
                                           arrow::default_memory_pool()));
    return arrow::Status::OK();
  }

  template <class ListLikeArray>
  arrow::Status VisitInternal(const ListLikeArray& array) {
    std::shared_ptr<arrow::DataType> value_type = array.value_type();
    if (!value_type->Equals(fill_with_.type())) {
      return arrow::Status::Invalid(absl::StrCat(
          "Expected a `fill_with` to be of the same type as "
          "`list_array`'s value_type, ",
          value_type->ToString(), " but got: ", fill_with_.type()->ToString()));
    }

    std::shared_ptr<Array> singleton_list;
    std::vector<std::shared_ptr<Array>> array_fragments;
    int64_t begin = 0, end = 0;
    while (end < array.length()) {
      if (array.IsNull(end)) {
        if (!singleton_list) {
          ARROW_RETURN_NOT_OK(
              MakeSingletonListArray(array, fill_with_, &singleton_list));
        }
        if (begin != end) {
          array_fragments.push_back(array.Slice(begin, end - begin));
        }
        array_fragments.push_back(singleton_list);
        ++end;
        begin = end;
      } else {
        ++end;
      }
    }

    // If the array does not contain nulls, then signal that no filling is done
    // and no new array is produced.
    const bool filled = singleton_list != nullptr;
    if (!filled) {
      result_ = nullptr;
      return arrow::Status::OK();
    }

    if (begin != end) {
      array_fragments.push_back(array.Slice(begin, end - begin));
    }
    // TODO(zhuo): use the concatenate() that returns a Result<>.
    return arrow::Concatenate(array_fragments, arrow::default_memory_pool(),
                              &result_);
  }

  const Array& fill_with_;
  std::shared_ptr<Array> result_;
};

class GetBinaryArrayTotalByteSizeVisitor : public arrow::ArrayVisitor {
 public:
  GetBinaryArrayTotalByteSizeVisitor() = default;

  arrow::Status Visit(const arrow::BinaryArray& binary_array) {
    return VisitInternal(binary_array);
  }
  arrow::Status Visit(const arrow::LargeBinaryArray& large_binary_array) {
    return VisitInternal(large_binary_array);
  }
  arrow::Status Visit(const arrow::StringArray& string_array) {
    return VisitInternal(string_array);
  }
  arrow::Status Visit(const arrow::LargeStringArray& large_string_array) {
    return VisitInternal(large_string_array);
  }

  size_t get_result() const {
    return result_;
  }

 private:
  template<typename BinaryLikeArray>
  arrow::Status VisitInternal(const BinaryLikeArray& array) {
    result_ = array.value_offset(array.length()) - array.value_offset(0);
    return arrow::Status::OK();
  }

  size_t result_ = 0;
};

bool IsStringType(const arrow::DataType& t) {
  return t.id() == arrow::Type::STRING || t.id() == arrow::Type::LARGE_STRING;
}

bool IsBinaryType(const arrow::DataType& t) {
  return t.id() == arrow::Type::BINARY || t.id() == arrow::Type::LARGE_BINARY;
}

// A memo table that holds unique string_views and (optionally) a null.
// Each unique value (including the null) has an index in the table. The index
// is determined by table insertion order of the first occurence of a value.
class StringViewMemoTable {
 public:
  template<typename BinaryArrayT>
  StringViewMemoTable(const BinaryArrayT& values) : null_index_(-1) {
    lookup_table_.reserve(values.length());
    int size = 0;
    for (int i = 0; i < values.length(); ++i) {
      if (values.IsNull(i)) {
        if (null_index_ < 0) null_index_ = size++;
        continue;
      }
      auto value = values.GetView(i);
      auto iter_and_inserted =
        lookup_table_.insert(std::make_pair(
            absl::string_view(value.data(), value.size()), size));
      if (iter_and_inserted.second) ++size;
    }
  }

  int Lookup(absl::string_view value) const {
    auto iter = lookup_table_.find(value);
    if (iter == lookup_table_.end()) return -1;
    return iter->second;
  }

  int null_index() const {
    return null_index_;
  }

 private:
  // slot index of null value. -1 if null value does not exist in the table.
  int null_index_;
  // Maps a value to a slot index.
  absl::flat_hash_map<absl::string_view, int> lookup_table_;
};

template <typename BinaryArrayT>
Status LookupIndices(
    const BinaryArrayT& values,
    const StringViewMemoTable& memo_table,
    std::shared_ptr<Array>* matched_value_set_indices) {
  arrow::Int32Builder result_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.Reserve(values.length())));
  for (int i = 0; i < values.length(); ++i) {
    if (values.IsNull(i)) {
      if (memo_table.null_index() > 0) {
        result_builder.UnsafeAppend(memo_table.null_index());
      } else {
        result_builder.UnsafeAppendNull();
      }
      continue;
    }
    auto value = values.GetView(i);
    int index =
        memo_table.Lookup(absl::string_view(value.data(), value.size()));
    if (index < 0) {
      result_builder.UnsafeAppendNull();
    } else {
      result_builder.UnsafeAppend(index);
    }
  }
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(result_builder.Finish(matched_value_set_indices)));
  return Status::OK();
}

Status IndexInBinaryArray(const Array& values, const Array& values_set,
                          std::shared_ptr<Array>* matched_value_set_indices) {
  std::unique_ptr<StringViewMemoTable> memo_table;
  switch (values_set.type()->id()) {
    case arrow::Type::BINARY:
    case arrow::Type::STRING:
      memo_table = absl::make_unique<StringViewMemoTable>(
          static_cast<const arrow::BinaryArray&>(values_set));
      break;
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::LARGE_STRING:
      memo_table = absl::make_unique<StringViewMemoTable>(
          static_cast<const arrow::LargeBinaryArray&>(values_set));
      break;
    default:
      return errors::FailedPrecondition("values_set is not binary-like");
  }

  switch (values.type()->id()) {
    case arrow::Type::BINARY:
    case arrow::Type::STRING:
      TFX_BSL_RETURN_IF_ERROR(
          LookupIndices(static_cast<const arrow::BinaryArray&>(values),
                        *memo_table, matched_value_set_indices));
      break;
    case arrow::Type::LARGE_BINARY:
    case arrow::Type::LARGE_STRING:
      TFX_BSL_RETURN_IF_ERROR(
          LookupIndices(static_cast<const arrow::LargeBinaryArray&>(values),
                        *memo_table, matched_value_set_indices));
      break;
    default:
      return errors::FailedPrecondition("values is not binary-like");
  }
  return Status::OK();
}

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
    const Array& array, std::shared_ptr<Array>* parent_indices_array) {
  GetFlattenedArrayParentIndicesVisitor v;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(array.Accept(&v)));
  *parent_indices_array = v.result();
  return Status::OK();
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
  GetBinaryArrayTotalByteSizeVisitor v;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(array.Accept(&v)));
  *total_byte_size = v.get_result();
  return Status::OK();
}

Status ValueCounts(const std::shared_ptr<arrow::Array>& array,
                   std::shared_ptr<arrow::Array>* values_and_counts_array) {
  return ValueCountsShim(array, values_and_counts_array);
}

Status IndexIn(const std::shared_ptr<arrow::Array>& values,
             const std::shared_ptr<arrow::Array>& value_set,
             std::shared_ptr<arrow::Array>* matched_value_set_indices) {
  // arrow::compute::Match does not support LargeBinary (as of 0.17).
  // TODO(zhuo): clean up this once tfx_bsl is built with arrow 1.0+.
  auto values_type = values->type();
  auto value_set_type = value_set->type();
  if ((IsStringType(*values_type) && IsStringType(*value_set_type)) ||
      (IsBinaryType(*values_type) && IsBinaryType(*value_set_type))) {
    return IndexInBinaryArray(*values, *value_set, matched_value_set_indices);
  }
  Datum result;
  TFX_BSL_RETURN_IF_ERROR(IndexInShim(values, value_set, &result));
  if (result.is_array()) {
    *matched_value_set_indices = result.make_array();
    return Status::OK();
  } else {
    return errors::Internal(absl::StrCat("Match result Datum is not an array ",
                                         "but an instance of type",
                                         result.type()->name()));
  }
}

// TODO(zhuo): Make this return LargeListArray once consumers can handle it.
Status MakeListArrayFromParentIndicesAndValues(
    const size_t num_parents,
    const std::shared_ptr<arrow::Array>& parent_indices,
    const std::shared_ptr<Array>& values,
    const bool empty_list_as_null,
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
  if (empty_list_as_null) {
    TFX_BSL_RETURN_IF_ERROR(
        FromArrowStatus(null_bitmap_builder.Reserve(num_parents)));
  }
  arrow::TypedBufferBuilder<int64_t> offsets_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(offsets_builder.Reserve(num_parents + 1)));

  offsets_builder.UnsafeAppend(0);
  for (int64_t i = 0, current_pi = 0; i < num_parents; ++i) {
    if (current_pi >= parent_indices_int64.length() ||
        parent_indices_int64.Value(current_pi) != i) {
      if (empty_list_as_null) null_bitmap_builder.UnsafeAppend(false);
    } else {
      while (current_pi < parent_indices_int64.length() &&
             parent_indices_int64.Value(current_pi) == i) {
        ++current_pi;
      }
      if (empty_list_as_null) null_bitmap_builder.UnsafeAppend(true);
    }
    offsets_builder.UnsafeAppend(current_pi);
  }

  const int64_t null_count = null_bitmap_builder.false_count();
  std::shared_ptr<arrow::Buffer> null_bitmap_buffer;
  if (empty_list_as_null) {
    TFX_BSL_RETURN_IF_ERROR(
        FromArrowStatus(null_bitmap_builder.Finish(&null_bitmap_buffer)));
  }
  std::shared_ptr<arrow::Buffer> offsets_buffer;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(offsets_builder.Finish(&offsets_buffer)));

  *out = std::make_shared<LargeListArray>(
      arrow::large_list(values->type()), num_parents, offsets_buffer, values,
      null_bitmap_buffer, null_count, /*offset=*/0);
  return Status::OK();
}

namespace {
// A helper class used by CooFromListArray. It's needed because the offsets of
// the (nested) list arrays considered could be either int64 or int32 because
// we want to support a mixture of LargeListArray and ListArray (for example,
// a ListArray<LargeListArray<int32>>, is a valid input). This class offers a
// uniform view to either a span of int32 or int64 numbers.
class RowSplitsRep {
 public:
  RowSplitsRep(absl::Span<const int32_t> s) : rep_(s) {}
  RowSplitsRep(const ListArray& list_array)
      : rep_(absl::MakeSpan(list_array.raw_value_offsets(),
                            list_array.length() + 1)) {}
  RowSplitsRep(const LargeListArray& large_list_array)
      : rep_(absl::MakeSpan(large_list_array.raw_value_offsets(),
                            large_list_array.length() + 1)) {}

  // Copying is cheap.
  RowSplitsRep(const RowSplitsRep&) = default;
  RowSplitsRep& operator=(const RowSplitsRep&) = default;

  size_t front() const {
    if (is_int32()) {
      return get<int32_t>().front();
    }
    return get<int64_t>().front();
  }

  size_t back() const {
    if (is_int32()) {
      return get<int32_t>().back();
    }
    return get<int64_t>().back();
  }

  // Increments *idx until
  // offset_row_splits[*idx] <= v < offset_row_splits[*idx + 1].
  // returns row_split (with offset applied) at *idx. row_splits should not be
  // empty.
  int64_t LookupAndUpdate(const int64_t v, size_t* idx) const {
    if (is_int32()) {
      return LookupAndUpdateInternal<int32_t>(v, idx);
    }
    return LookupAndUpdateInternal<int64_t>(v, idx);
  }

  // Returns the maximum length of the sub-lists represented by the contained
  // offsets.
  size_t MaxLength() const {
    if (is_int32()) {
      return MaxLengthInternal<int32_t>();
    }
    return MaxLengthInternal<int64_t>();
  }

 private:
  bool is_int32() const {
    return absl::holds_alternative<absl::Span<const int32_t>>(rep_);
  }

  template <typename PayloadT>
  absl::Span<const PayloadT> get() const {
    static_assert(std::is_same<PayloadT, int64_t>::value ||
                      std::is_same<PayloadT, int32_t>::value,
                  "Must be either Span<const int64_t> or Span<const int32_t>");
    return absl::get<absl::Span<const PayloadT>>(rep_);
  }

  template<typename PayloadT>
  size_t MaxLengthInternal() const {
    auto row_splits = this->get<PayloadT>();
    size_t max_length = 0;
    for (int i = 0; i < row_splits.size() - 1; ++i) {
      size_t length = row_splits[i + 1] - row_splits[i];
      max_length = std::max(max_length, length);
    }
    return max_length;
  }

  template<typename PayloadT>
  int64_t LookupAndUpdateInternal(const int64_t v, size_t* idx) const {
    auto row_splits = this->get<PayloadT>();
    // Note that row_splits does not always start with 0, as the ListArray
    // can be sliced (which causes the row_splits to also be sliced). In that
    // case, the slicing offset is row_splits[0].
    const int64_t row_split_offset = row_splits[0];
    while (*idx < row_splits.size() - 1) {
      const int64_t begin = row_splits[*idx] - row_split_offset;
      const int64_t end = row_splits[*idx + 1] - row_split_offset;
      if (v >= begin && v < end) break;
      ++(*idx);
    }
    return row_splits[*idx] - row_split_offset;
  }

  absl::variant<absl::Span<const int32_t>, absl::Span<const int64_t>> rep_;
};

class GetByteSizeVisitor : public arrow::ArrayVisitor {
 public:
  GetByteSizeVisitor() : GetByteSizeVisitor(/*offset=*/0, /*length=*/-1) {}

  size_t result() const { return result_; }

#define VISIT_TYPE_WITH(TYPE, VISIT_FUNC)                       \
  arrow::Status Visit(const TYPE& array) override { \
    return VISIT_FUNC(array);                 \
  }

  VISIT_TYPE_WITH(arrow::Int8Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::Int16Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::Int32Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::Int64Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::UInt8Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::UInt16Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::UInt32Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::UInt64Array, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::FloatArray, NumericArrayImpl)
  VISIT_TYPE_WITH(arrow::DoubleArray, NumericArrayImpl)

  VISIT_TYPE_WITH(arrow::BinaryArray, BinaryLikeImpl)
  VISIT_TYPE_WITH(arrow::StringArray, BinaryLikeImpl)
  VISIT_TYPE_WITH(arrow::LargeBinaryArray, BinaryLikeImpl)
  VISIT_TYPE_WITH(arrow::LargeStringArray, BinaryLikeImpl)

  VISIT_TYPE_WITH(arrow::ListArray, ListLikeImpl)
  VISIT_TYPE_WITH(arrow::LargeListArray, ListLikeImpl)
#undef VISIT_TYPE_WITH

  arrow::Status Visit(const arrow::NullArray& array) {
    // a NullArray does not have any buffer allocated.
    result_ = 0;
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::StructArray& array) {
    result_ += GetNullBitmapByteSize(array);
    const int num_children = array.struct_type()->num_children();
    for (int i = 0; i < num_children; ++i) {
      // We don't need to pass on the offsets and lengths to the child
      // visitor because StructArray::field() handles it.
      GetByteSizeVisitor v;
      ARROW_RETURN_NOT_OK(array.field(i)->Accept(&v));
      result_ += v.result();
    }
    return arrow::Status::OK();
  }

  arrow::Status Visit(const arrow::BooleanArray& array) {
    result_ += GetNullBitmapByteSize(array);
    result_ += (GetArrayLength(array) + 7) / 8;
    return arrow::Status::OK();
  }

 private:
  // if `length` < 0, array.length() will be used.
  GetByteSizeVisitor(int64_t offset, int64_t length)
      : offset_(offset), length_(length), result_(0) {}

  int64_t GetArrayLength(const Array& array) const {
    if (length_ >= 0) return length_;
    return array.length();
  }

  size_t GetNullBitmapByteSize(const Array& array) const {
    if (!array.null_bitmap_data()) {
      return 0;
    }
    // Round to the next byte.
    return (GetArrayLength(array) + 7) / 8;
  }

  template<typename ListLikeT>
  arrow::Status ListLikeImpl(const ListLikeT& array) {
    const size_t length = GetArrayLength(array);
    // Size of the offsets.
    result_ +=
        (length + 1) * sizeof(typename ListLikeT::TypeClass::offset_type);
    result_ += GetNullBitmapByteSize(array);
    // Size of the child array. We delegate to another GetByteSizeVisitor.
    // Note that `array.values()` does not take the offsets into consideration.
    // We have to get the right offset and length from the offsets of `array`.
    const auto* value_offsets = array.raw_value_offsets();
    const int64_t child_offset = value_offsets[offset_];
    const int64_t child_length = value_offsets[offset_ + length] - child_offset;
    GetByteSizeVisitor child_visitor(child_offset, child_length);
    ARROW_RETURN_NOT_OK(array.values()->Accept(&child_visitor));
    result_ += child_visitor.result();
    return arrow::Status::OK();
  }

  template<typename BinaryLikeT>
  arrow::Status BinaryLikeImpl(const BinaryLikeT& array) {
    const size_t length = GetArrayLength(array);
    const auto* offsets = array.raw_value_offsets();
    // size of the offsets.
    result_ +=
        (length + 1) * sizeof(typename BinaryLikeT::TypeClass::offset_type);
    // size of the contents.
    result_ += offsets[offset_ + length] - offsets[offset_];

    result_ += GetNullBitmapByteSize(array);
    return arrow::Status::OK();
  }

  arrow::Status NumericArrayImpl(const arrow::PrimitiveArray& array) {
    auto type = array.type();
    auto primitive_type = static_cast<const arrow::PrimitiveCType*>(type.get());
    result_ += GetArrayLength(array) * (primitive_type->bit_width() / 8);
    result_ += GetNullBitmapByteSize(array);
    return arrow::Status::OK();
  }

  const int64_t offset_;
  const int64_t length_;
  size_t result_;
};
}  // namespace

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

  std::vector<RowSplitsRep> nested_row_splits;
  std::array<int32_t, 2> dummy_outermost_row_splits = {
      0, static_cast<int32_t>(list_array->length())};
  nested_row_splits.push_back(
      RowSplitsRep(absl::MakeSpan(dummy_outermost_row_splits)));

  // Strip `list_array` and populate `nested_row_splits` with row_splits of
  // each level.
  std::shared_ptr<arrow::Array> values = list_array;
  while (true) {
    bool is_list_array = true;
    switch (values->type()->id()) {
      case arrow::Type::LIST:  {
        ListArray* list_array = static_cast<ListArray*>(values.get());
        RowSplitsRep row_splits(*list_array);
        nested_row_splits.push_back(row_splits);
        // Note that the values array is not sliced even if `list_array` is, so
        // we slice it here.
        values = list_array->values()->Slice(
            row_splits.front(), row_splits.back() - row_splits.front());
        break;
      }
      case arrow::Type::LARGE_LIST: {
        LargeListArray* list_array = static_cast<LargeListArray*>(values.get());
        RowSplitsRep row_splits(*list_array);
        nested_row_splits.push_back(row_splits);
        // Note that the values array is not sliced even if `list_array` is, so
        // we slice it here.
        values = list_array->values()->Slice(
            row_splits.front(), row_splits.back() - row_splits.front());
        break;
      }
      default: {
        is_list_array = false;
        break;
      }
    }
    if (!is_list_array) break;
  }

  // Allocate a buffer for the coordinates. A k-nested ListArray will be
  // converted to a sparse tensor of k+1 dimensions. The buffer for the
  // coordinates will contain all the coordinates concatenated, so it needs to
  // hold (k + 1) * num_values numbers.
  const size_t coo_length = nested_row_splits.size();
  const size_t coo_buffer_size =
      coo_length * values->length() * sizeof(int64_t);
  std::shared_ptr<arrow::Buffer> coo_buffer;
  TFX_BSL_ASSIGN_OR_RETURN_ARROW(
      coo_buffer,
      arrow::AllocateBuffer(coo_buffer_size, arrow::default_memory_pool()));
  int64_t* coo_flat = reinterpret_cast<int64_t*>(coo_buffer->mutable_data());

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
    int64_t current_idx = i;
    // The inner loop looks for the index in the belonging sub-list at each
    // level.
    for (int j = nested_row_splits.size() - 1; j >= 0; --j) {
      const int64_t row_split_begin = nested_row_splits[j].LookupAndUpdate(
          current_idx, &current_owning_sublist_indices[j]);
      current_coo[j] = current_idx - row_split_begin;
      current_idx = current_owning_sublist_indices[j];
    }
  }

  // The dense shape is the bounding box of the ListArray: the maximum lengths
  // of sub-lists in each level.
  arrow::Int64Builder dense_shape_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(dense_shape_builder.Reserve(coo_length)));
  for (const auto& row_splits : nested_row_splits) {
    dense_shape_builder.UnsafeAppend(row_splits.MaxLength());
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
  FillNullListsVisitor v(*fill_with);
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(list_array->Accept(&v)));
  auto visit_result = v.result();
  if (visit_result) {
    *filled = std::move(visit_result);
  } else {
    // Visitor producing a nullptr means that no filling needs to be done. The
    // result equals to the input.
    *filled = list_array;
  }
  return Status::OK();
}

Status GetByteSize(const Array& array, size_t* result) {
  GetByteSizeVisitor v;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(array.Accept(&v)));
  *result = v.result();
  return Status::OK();
}
}  // namespace tfx_bsl
