// Copyright 2020 Google LLC
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

#include "tfx_bsl/cc/sketches/misragries_sketch.h"

#include <cstddef>
#include <memory>

#include "absl/memory/memory.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tfx_bsl/cc/util/utf8.h"

namespace tfx_bsl {
namespace sketches {

namespace {
using tfx_bsl::sketches::InputType;

// Accounts for rounding error due to float arithmetic.
// TODO(b/168224198): Investigate if weighted/unweighted cases should use
// different values of kEpsilon.
static constexpr double kEpsilon = 1e-8;

inline bool LessThanOrEqualToZero(double value) { return (value < kEpsilon); }

inline bool EqualToZero(double value) { return (abs(value) < kEpsilon); }

// Subtracts value from each weight in `item_counts`, removes elements with
// negative weight and moves items that have zero weight to `extra_items`.
//
// `DecrementCounters` expects (but doesn't validate) the following conditions:
//  - `item_counts` has more than `max_num_items` elements;
//  - `value` is (`max_num_items` + 1)-th largest weight in `item_counts`.
//
// The purpose of `DecrementCounters` is to trim `item_counts` to
// `max_num_items` items discarding items with smaller weights. However, when
// `item_counts` has multiple items having weight equal to
// (`max_num_items` + 1)-th largest, the number of items with positive weight
// after subtraction can be less than `max_num_items`. In this case, to prevent
// outputting fewer items than requested, we keep some extra items with zero
// weight in `extra_items`. Note that keeping zero-weight items is enough to
// fill `item_counts` to `max_num_items` elements since `value` is
// (`max_num_items` + 1)-th largest weight.
void DecrementCounters(double value, int max_num_items,
                       absl::flat_hash_map<std::string, double>* item_counts,
                       absl::flat_hash_set<std::string>* extra_items) {
  // `DecrementCounters` is expected to be called when `item_counts` has more
  // than `max_num_items` unique elements. It is, therefore, sufficient to keep
  // `extra_items` from the current call only to fill `item_counts` to
  // `max_num_items` elements. It also allows not to save weights of
  // `extra_items` since we know that they are 0 (relative to `total_error`).
  assert(item_counts->size() > max_num_items);
  extra_items->clear();
  for (auto iter = item_counts->begin(); iter != item_counts->end();) {
    iter->second -= value;
    if (LessThanOrEqualToZero(iter->second)) {
      if (EqualToZero(iter->second)) {
        extra_items->insert(iter->first);
      }
      item_counts->erase(iter++);
    } else {
      ++iter;
    }
  }

  // Remove excessive extra items.
  for (auto iter = extra_items->begin();
       iter != extra_items->end() &&
       item_counts->size() + extra_items->size() > max_num_items;) {
    extra_items->erase(iter++);
  }
}

// Updates the item counts with the values of the visited array. Constructor
// takes num_buckets, weights (can be nullptr), delta, and the item counts map
// as parameters.
class UpdateItemCountsVisitor : public arrow::ArrayVisitor  {
 public:
  UpdateItemCountsVisitor(
      const absl::optional<std::string>& invalid_utf8_placeholder,
      const absl::optional<int>& large_string_threshold,
      const absl::optional<std::string>& large_string_placeholder,
      int num_buckets, const arrow::FloatArray* weights, double& delta,
      InputType::Type& input_type,
      absl::flat_hash_map<std::string, double>& item_counts,
      absl::flat_hash_set<std::string>& extra_items)
      : invalid_utf8_placeholder_(invalid_utf8_placeholder),
        large_string_threshold_(large_string_threshold),
        large_string_placeholder_(large_string_placeholder),
        num_buckets_(num_buckets),
        weights_(weights),
        delta_(delta),
        input_type_(input_type),
        item_counts_(item_counts),
        extra_items_(extra_items) {}

  arrow::Status Visit(const arrow::BinaryArray& array) override {
    return VisitInternal(array, InputType::RAW_STRING);
  }
    arrow::Status Visit(const arrow::LargeBinaryArray& array) override {
      return VisitInternal(array, InputType::RAW_STRING);
    }
    arrow::Status Visit(const arrow::StringArray& array) override {
      return VisitInternal(array, InputType::RAW_STRING);
    }
    arrow::Status Visit(const arrow::LargeStringArray& array) override {
      return VisitInternal(array, InputType::RAW_STRING);
    }
    arrow::Status Visit(const arrow::Int8Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::Int16Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::Int32Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::Int64Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::UInt8Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::UInt16Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::UInt32Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::UInt64Array& array) override {
      return VisitInternal(array, InputType::INT);
    }
    arrow::Status Visit(const arrow::FloatArray& array) override {
      return VisitInternal(array, InputType::FLOAT);
    }
    arrow::Status Visit(const arrow::DoubleArray& array) override {
      return VisitInternal(array, InputType::FLOAT);
    }

 private:
  // Encodes a value as a string according to the stored type.
  template <typename T> std::string Encode(const T val) const {
    return input_type_ == InputType::FLOAT
                                     ? absl::StrFormat("%a", val)
                                     : absl::StrCat(val);
  }

  template <class T>
  arrow::Status VisitInternal(const T& array, const InputType::Type type) {
    if (input_type_ == InputType::UNSET) {
      input_type_ = type;
    }
    if (input_type_ != type) {
      return arrow::Status::TypeError(
          absl::StrFormat("sketch stored type error: stored %s given %s",
                          InputType::Type_Name(input_type_),
                          InputType::Type_Name(input_type_)));
    }

    if (!weights_) {
      AddItemsWithoutWeights(array);
    } else {
      AddItemsWithWeights(array);
    }
    return arrow::Status::OK();
  }

    // Updates item_counts_ with an Arrow BinaryArray of values.
    template <typename T>
    void AddItemsWithoutWeights(
        const arrow::BaseBinaryArray<T>& array) {
      for (int i = 0; i < array.length(); i++) {
        if (array.IsNull(i)) {
          continue;
        }
        const auto value = array.GetView(i);
        absl::string_view value_sv(value.data(), value.size());

        if (invalid_utf8_placeholder_) {
          if (!IsValidUtf8(value_sv)) {
            InsertItem(*invalid_utf8_placeholder_);
            continue;
          }
        }

        if (large_string_threshold_ &&
            value_sv.size() > *large_string_threshold_) {
          InsertItem(*large_string_placeholder_);
          continue;
        }

        InsertItem(value_sv);
      }
    }

    // Updates item_counts_ with an Arrow NumericArray of values.
    template <typename T>
    void AddItemsWithoutWeights(
        const arrow::NumericArray<T>& array) {
      for (int i = 0; i < array.length(); i++) {
        if (array.IsNull(i)) {
          continue;
        }
        const std::string item = Encode(array.Value(i));
        InsertItem(item);
      }
    }

    // Updates item_counts_ with an Arrow BinaryArray of values and their
    // weights.
    template <typename T>
    void AddItemsWithWeights(const arrow::BaseBinaryArray<T>& array) {
      for (int i = 0; i < array.length(); i++) {
        if (array.IsNull(i)) {
          continue;
        }
        const auto value = array.GetView(i);
        const auto item = absl::string_view(value.data(), value.size());
        const float weight = weights_->Value(i);
        InsertItem(item, weight);
      }
    }

    // Updates item_counts_ with an Arrow NumericArray of values and their
    // weights.
    template <typename T>
    void AddItemsWithWeights(const arrow::NumericArray<T>& array) {
      for (int i = 0; i < array.length(); i++) {
        if (array.IsNull(i)) {
          continue;
        }
        const std::string item = Encode(array.Value(i));
        const auto weight = weights_->Value(i);
        InsertItem(item, weight);
      }
    }

    // Performs one step of the algorithm:
    //  1) If the item is in item_counts_, increment its value by 1. Else, set
    //     its value to 1.
    //  2) If there are over num_buckets_ counters, decrease all counters by 1.,
    //     and remove those that become 0.
    void InsertItem(absl::string_view item) {
      // Begin with <= num_buckets_ items in the map.
      auto insertion_pair =
          item_counts_.insert(std::pair<std::string, double>(item, 1.));
      // Item is already in map.
      if (!insertion_pair.second) {
        insertion_pair.first->second += 1.0;
      // New item just added, so there might be an overflow.
      } else if (item_counts_.size() > num_buckets_) {
        DecrementCounters(/*value=*/1.0, num_buckets_, &item_counts_,
                          &extra_items_);
        delta_ += 1.0;
      }
    }

    // Performs one step of the algorithm for weighted values:
    //  1) If the item is in item_counts_, increment its value by its weight.
    //     Else, add it to item_counts_ with its weight.
    //  2) If there are greater than num_buckets_ counters, subtract the
    //     smallest weight in item_counts_ from every item in item_counts_ and
    //     remove all those that become zero.
    void InsertItem(absl::string_view item, float weight) {
      auto insertion_pair =
          item_counts_.insert(std::pair<std::string, double>(item, weight));
      // Item is already in map.
      if (!insertion_pair.second) {
        // TODO(b/168224198): Investigate and test the accumulated numerical
        // error due to floating point summation.
        insertion_pair.first->second += weight;
      // New item just added, so there might be an overflow.
      } else if (item_counts_.size() > num_buckets_) {
        // Find the smallest weight.
        double min_weight = std::min_element(
            item_counts_.begin(), item_counts_.end(),
            [](const std::pair<std::string, double>& lhs,
               const std::pair<std::string, double>& rhs){
              return lhs.second < rhs.second;
            })->second;
        DecrementCounters(min_weight, num_buckets_, &item_counts_,
                          &extra_items_);
        delta_ += min_weight;
      }
    }


    const absl::optional<std::string>& invalid_utf8_placeholder_;
    const absl::optional<int>& large_string_threshold_;
    const absl::optional<std::string>& large_string_placeholder_;
    const int num_buckets_;
    const arrow::FloatArray* weights_;
    double& delta_;
    InputType::Type& input_type_;
    absl::flat_hash_map<std::string, double>& item_counts_;
    absl::flat_hash_set<std::string>& extra_items_;
};

}  // namespace

MisraGriesSketch::MisraGriesSketch(
    int num_buckets, absl::optional<std::string> invalid_utf8_placeholder,
    absl::optional<int> large_string_threshold,
    absl::optional<std::string> large_string_placeholder)
    : num_buckets_(num_buckets),
      delta_(0.0),
      input_type_(InputType::UNSET),
      invalid_utf8_placeholder_(std::move(invalid_utf8_placeholder)),
      large_string_threshold_(std::move(large_string_threshold)),
      large_string_placeholder_(std::move(large_string_placeholder)) {
  item_counts_.reserve(num_buckets);
}

Status MisraGriesSketch::AddValues(const arrow::Array& items) {
  UpdateItemCountsVisitor v(invalid_utf8_placeholder_, large_string_threshold_,
                            large_string_placeholder_, num_buckets_, nullptr,
                            delta_, input_type_, item_counts_, extra_items_);
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(items.Accept(&v)));
  return Status::OK();
}

Status MisraGriesSketch::AddValues(
    const arrow::Array& items, const arrow::Array& weights) {
  if (items.length() != weights.length()) {
     return errors::InvalidArgument(
        "Length of item array must be equal to length of weight array: ",
         items.length(), " v.s. ", weights.length());
  }
  if (weights.type()->id() != arrow::Type::FLOAT) {
    return errors::InvalidArgument("Weight array must be float type.");
  }
  const auto& weight_array = static_cast<const arrow::FloatArray&>(weights);
  UpdateItemCountsVisitor v(invalid_utf8_placeholder_, large_string_threshold_,
                            large_string_placeholder_, num_buckets_,
                            &weight_array, delta_, input_type_, item_counts_,
                            extra_items_);
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(items.Accept(&v)));
  return Status::OK();
}

Status MisraGriesSketch::Merge(const MisraGriesSketch& other) {
  if (other.num_buckets_ != num_buckets_) {
    return errors::InvalidArgument(
        "Both sketches must have the same number of buckets: ",
         num_buckets_, " v.s. ", other.num_buckets_);
  }
  if (other.large_string_threshold_ != large_string_threshold_) {
    return errors::InvalidArgument(
        "Both sketches must have the same large_string_threshold.");
  }
  if (other.large_string_placeholder_ != large_string_placeholder_) {
    return errors::InvalidArgument(
        "Both sketches must have the same large_string_placeholder.");
  }
  if (other.invalid_utf8_placeholder_ != invalid_utf8_placeholder_) {
    return errors::InvalidArgument(
        "Both sketches must have the same invalid_utf8_placeholder.");
  }
  if (input_type_ == InputType::UNSET) {
    input_type_ = other.input_type_;
  }
  if (other.input_type_ != InputType::UNSET &&
      input_type_ != other.input_type_) {
    return errors::InvalidArgument(
        absl::StrFormat("Both sketches must have the same type (%s vs %s)",
                        InputType::Type_Name(input_type_),
                        InputType::Type_Name(other.input_type_)));
  }
  for (const auto& item : other.item_counts_) {
    auto insertion_pair = item_counts_.insert(item);
    if (!insertion_pair.second) {
      insertion_pair.first->second += item.second;
    }
  }
  for (const auto& item : other.extra_items_) {
    extra_items_.insert(item);
  }
  delta_ += other.delta_;
  Compress();
  return Status::OK();
}

void MisraGriesSketch::Compress() {
  if (item_counts_.size() <= num_buckets_) {
    return;
  }
  std::vector<double> weights;
  weights.reserve(item_counts_.size());
  for (const auto& it : item_counts_) {
    weights.push_back(it.second);
  }
  // Get the n-th largest weight, where n = num_buckets_
  std::nth_element(weights.begin(), weights.begin() + num_buckets_,
                   weights.end(), std::greater<double>());

  double nth_largest = weights[num_buckets_];
  // Remove all items with weights less than or equal to n-th largest weight.
  DecrementCounters(nth_largest, num_buckets_, &item_counts_, &extra_items_);
  delta_ += nth_largest;
}

Status MisraGriesSketch::Decode(std::string* item) const {
  switch (input_type_) {
    case InputType::RAW_STRING:
      return tfx_bsl::Status::OK();
    case InputType::INT:
      return tfx_bsl::Status::OK();
    case InputType::FLOAT:
      double out;
      if (!absl::SimpleAtod(*item, &out))
        return tfx_bsl::errors::InvalidArgument(
            absl::StrFormat("failed to decode %s as float", *item));
      *item = absl::StrCat(out);
      return tfx_bsl::Status::OK();
    default:
      return tfx_bsl::errors::InvalidArgument(absl::StrFormat(
          "unhandled input type %s", InputType::Type_Name(input_type_)));
  }
}

Status MisraGriesSketch::GetCounts(
    std::vector<std::pair<std::string, double>>& result) const {
  result.clear();
  result.reserve(item_counts_.size());
  // Add delta_ to each of the counts to get the upper bound estimate.
  for (const auto& pair : item_counts_) {
    result.emplace_back(pair.first, pair.second + delta_);
  }
  switch (input_type_) {
    // INT, RAW_STRING, and UNSET (empty sketch) do not need to be decoded.
    case InputType::INT:
      break;
    case InputType::RAW_STRING:
      break;
    case InputType::UNSET:
      break;
    case InputType::FLOAT:
      for (auto& item_w : result) {
        TFX_BSL_RETURN_IF_ERROR(Decode(&item_w.first));
      }
      break;
    default:
      return tfx_bsl::errors::FailedPrecondition(absl::StrCat(
          "unhandled input type ", InputType::Type_Name(input_type_)));
  }
  std::sort(
      result.begin(), result.end(),
      [](const std::pair<std::string, double>& x,
         const std::pair<std::string, double>& y) {
        if (x.second != y.second) {
          return x.second > y.second;
        }
        return x.first < y.first;
      }
  );
  // Fill the `result` up to `num_buckets_` items using `extra_items_`.
  if (result.size() < num_buckets_) {
    std::vector<std::string> ordered_extra_items;
    for (const auto& item : extra_items_) {
      if (item_counts_.find(item) == item_counts_.end()) {
        ordered_extra_items.emplace_back(item);
      }
    }
    std::sort(ordered_extra_items.begin(), ordered_extra_items.end());
    for (const auto& item : ordered_extra_items) {
      result.emplace_back(item, delta_);
      if (result.size() == num_buckets_) {
        break;
      }
    }
  }
  return tfx_bsl::Status::OK();
}

Status MisraGriesSketch::Estimate(
    std::shared_ptr<arrow::Array>* values_and_counts_array) const {
  std::vector<std::pair<std::string, double>> sorted_pairs;
  TFX_BSL_RETURN_IF_ERROR(GetCounts(sorted_pairs));
  // Combine the item and count vectors into an Arrow StructArray.
  arrow::LargeBinaryBuilder binary_builder;
  arrow::DoubleBuilder double_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(binary_builder.Reserve(sorted_pairs.size())));
  // Calculate total size of strings to append.
  std::size_t total_length = std::accumulate(
      sorted_pairs.begin(), sorted_pairs.end(), 0ULL,
      [](uint64_t sum, const std::pair<std::string, double>& pair) {
        return sum + pair.first.size();
      }
  );
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(binary_builder.ReserveData(total_length)));
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(double_builder.Reserve(sorted_pairs.size())));
  std::shared_ptr<arrow::Array> item_array;
  std::shared_ptr<arrow::Array> count_array;
  for (const auto& pair : sorted_pairs) {
    binary_builder.UnsafeAppend(pair.first);
    double_builder.UnsafeAppend(pair.second);
  }
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(binary_builder.Finish(&item_array)));
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(double_builder.Finish(&count_array)));

  static constexpr char kValuesFieldName[] = "values";
  static constexpr char kCountsFieldName[] = "counts";

  auto struct_array_or_error =
      arrow::StructArray::Make(
          std::vector<std::shared_ptr<arrow::Array>>{item_array, count_array},
          std::vector<std::string>{kValuesFieldName, kCountsFieldName});
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(struct_array_or_error.status()));
  *values_and_counts_array = std::move(struct_array_or_error).ValueOrDie();
  return Status::OK();
}

std::string MisraGriesSketch::Serialize() const{
  MisraGries mg_proto;
  mg_proto.set_num_buckets(num_buckets_);
  mg_proto.set_delta(delta_);
  mg_proto.set_input_type(input_type_);
  for (const auto& it : item_counts_) {
    mg_proto.add_items(it.first);
    mg_proto.add_weights(it.second);
  }
  for (const auto& item : extra_items_) {
    mg_proto.add_extra_items(item);
  }
  if (invalid_utf8_placeholder_) {
    mg_proto.set_replace_invalid_utf8_values(true);
    mg_proto.set_invalid_utf8_placeholder(*invalid_utf8_placeholder_);
  }
  if (large_string_threshold_) {
    mg_proto.set_large_string_threshold(*large_string_threshold_);
    mg_proto.set_large_string_placeholder(*large_string_placeholder_);
  } else {
    mg_proto.set_large_string_threshold(-1);
  }

  return mg_proto.SerializeAsString();
}

Status MisraGriesSketch::Deserialize(
    absl::string_view encoded, std::unique_ptr<MisraGriesSketch>* result) {
  MisraGries mg_proto;
  if (!mg_proto.ParseFromArray(encoded.data(), encoded.size()))
    return tfx_bsl::errors::InvalidArgument(
        "Failed to parse MisraGries sketch");
  absl::optional<std::string> invalid_utf8_placeholder;
  if (mg_proto.replace_invalid_utf8_values()) {
    invalid_utf8_placeholder = mg_proto.invalid_utf8_placeholder();
  }
  absl::optional<std::string> large_string_placeholder;
  absl::optional<int> large_string_threshold;
  if (mg_proto.large_string_threshold() >= 0) {
    large_string_threshold = mg_proto.large_string_threshold();
    large_string_placeholder = mg_proto.large_string_placeholder();
  }
  *result = absl::make_unique<MisraGriesSketch>(
      mg_proto.num_buckets(), std::move(invalid_utf8_placeholder),
      std::move(large_string_threshold), std::move(large_string_placeholder));
  if (mg_proto.delta() > 0.) {
    (*result)->delta_ = mg_proto.delta();
  }
  (*result)->input_type_ = mg_proto.input_type();
  int num_items = mg_proto.items_size();
  for (int i = 0; i < num_items; i++) {
    (*result)->item_counts_.emplace(mg_proto.items(i), mg_proto.weights(i));
  }
  for (const auto& item : mg_proto.extra_items()) {
    (*result)->extra_items_.emplace(item);
  }
  return Status::OK();
}

double MisraGriesSketch::GetDelta() const {
  return delta_;
}

double MisraGriesSketch::GetDeltaUpperBound(double global_weight) const {
  double m_prime = 0.;
  for (const auto& pair : item_counts_) {
    m_prime += pair.second;
  }
  return (global_weight - m_prime) / (num_buckets_ + 1);
}

InputType::Type MisraGriesSketch::GetInputType() const {
  return input_type_;
}

}  // namespace sketches
}  // namespace tfx_bsl
