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

#include "absl/strings/string_view.h"
#include "arrow/api.h"
#include "absl/strings/str_format.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace sketches {

namespace {

// Accounts for rounding error due to float arithmetic.
bool LessThanOrEqualToZero(double value) {
  // TODO(monicadsong@): Investigate if weighted/unweighted cases should use
  // different values of kEpsilon.
  static constexpr double kEpsilon = 1e-8;
  return (value < kEpsilon);
}

// Updates the item counts with the values of the visited array. Constructor
// takes num_buckets, weights (can be nullptr), delta, and the item counts map
// as parameters.
class UpdateItemCountsVisitor : public arrow::ArrayVisitor  {
 public:
    UpdateItemCountsVisitor(
        int num_buckets,
        const arrow::FloatArray* weights,
        double& delta,
        absl::flat_hash_map<std::string, double>& item_counts)
        : num_buckets_(num_buckets), weights_(weights), delta_(delta),
          item_counts_(item_counts) {}

    arrow::Status Visit(const arrow::BinaryArray& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::LargeBinaryArray& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::StringArray& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::LargeStringArray& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::Int8Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::Int16Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::Int32Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::Int64Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::UInt8Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::UInt16Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::UInt32Array& array) override {
      return VisitInternal(array);
    }
    arrow::Status Visit(const arrow::UInt64Array& array) override {
      return VisitInternal(array);
    }

 private:
    template <class T>
    arrow::Status VisitInternal(const T& array) {
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
        const auto item = absl::string_view(value.data(), value.size());
        InsertItem(item);
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
        std::string item = absl::StrCat(array.Value(i));
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
        const auto item = absl::StrCat(array.Value(i));
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
        DecrementCounters(1.0);
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
        // TODO(monicadsong@): Investigate and test the accumulated numerical
        // error due to floating point summation. See Kahan summation algorithm
        // for a possible solution.
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
        DecrementCounters(min_weight);
      }
    }

    // Subtract value from each item in item_counts_ and remove those that
    // become zero.
    void DecrementCounters(double value) {
      for (auto iter = item_counts_.begin(); iter!= item_counts_.end();) {
        iter->second -= value;
        if (LessThanOrEqualToZero(iter->second)) {
          item_counts_.erase(iter++);
        } else {
          ++iter;
        }
      }
      delta_ += value;
    }
    const int num_buckets_;
    const arrow::FloatArray* weights_;
    double& delta_;
    absl::flat_hash_map<std::string, double>& item_counts_;
};

}  // namespace

MisraGriesSketch::MisraGriesSketch(int num_buckets)
    : num_buckets_(num_buckets), delta_(0.0) {
  item_counts_.reserve(num_buckets);
}

Status MisraGriesSketch::AddValues(const arrow::Array& items) {
  UpdateItemCountsVisitor v(num_buckets_, nullptr, delta_, item_counts_);
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
  UpdateItemCountsVisitor v(num_buckets_, &weight_array, delta_, item_counts_);
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(items.Accept(&v)));
  return Status::OK();
}

Status MisraGriesSketch::Merge(MisraGriesSketch& other) {
  if (other.num_buckets_ != num_buckets_) {
    return errors::InvalidArgument(
        "Both sketches must have the same number of buckets: ",
         num_buckets_, " v.s. ", other.num_buckets_);
  }
  for (const auto& item : other.item_counts_) {
    auto insertion_pair = item_counts_.insert(item);
    if (!insertion_pair.second) {
      insertion_pair.first->second += item.second;
    }
  }
  // TODO(monicadsong@): Check if adding the delta_ is correct. Currently this
  // steps inflates the estimates (but still stays within acceptable error).
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
  for (auto iter = item_counts_.begin(); iter!= item_counts_.end();) {
    iter->second -= nth_largest;
    if (LessThanOrEqualToZero(iter->second)) {
      item_counts_.erase(iter++);
    } else {
      ++iter;
    }
  }
  delta_ += nth_largest;
}

std::vector<std::pair<std::string, double>> MisraGriesSketch::GetCounts() const{
  // Add delta_ to each of the counts to get the upper bound estimate.
  std::vector<std::pair<std::string, double>> result;
  for (const auto& pair : item_counts_) {
    result.emplace_back(pair.first, pair.second + delta_);
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
  return result;
}

Status MisraGriesSketch::Estimate(
    std::shared_ptr<arrow::Array>* values_and_counts_array) const {
  std::vector<std::pair<std::string, double>> sorted_pairs = GetCounts();
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
  for (const auto& it : item_counts_) {
    mg_proto.add_items(it.first);
    mg_proto.add_weights(it.second);
  }
  return mg_proto.SerializeAsString();
}

MisraGriesSketch MisraGriesSketch::Deserialize(absl::string_view encoded) {
  MisraGries mg_proto;
  mg_proto.ParseFromArray(encoded.data(), encoded.size());
  MisraGriesSketch mg_sketch{mg_proto.num_buckets()};
  if (mg_proto.has_delta()) {
    mg_sketch.delta_ = mg_proto.delta();
  }
  int num_items = mg_proto.items_size();
  for (int i = 0; i < num_items; i++) {
    mg_sketch.item_counts_.emplace(mg_proto.items(i), mg_proto.weights(i));
  }
  return mg_sketch;
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

}  // namespace sketches
}  // namespace tfx_bsl
