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

#ifndef TFX_BSL_CC_SKETCHES_MISRAGRIES_SKETCH_H_
#define TFX_BSL_CC_SKETCHES_MISRAGRIES_SKETCH_H_

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"

namespace tfx_bsl {
namespace sketches {

// The Misra-Gries (MG) sketch estimates the top k item counts (possibly
// weighted) in a data stream, where k is the number of buckets stored in the
// sketch.
//
// This implementation uses `delta_` to track the the number of the times that
// the values in `item_counts_` are decremented in the algorithm. It stores the
// lower bound estimate in `item_counts_` and returns the upper bound estimate,
// which is equal to the lower bound estimate + delta_, when queried. This
// algorithm guarantees for any item x with true count X:
//
//               item_counts_[x] <= X <= item_counts_[x] + delta_
//                         delta_ <= (m - m') / (k + 1)
//
// where m is the total weight of the items in the dataset and m' is the sum of
// the lower bound estimates in `item_counts_` [1].
//
// The sketch is mergeable, meaning it guarantees this error bound when merged
// with other MG sketches [2]. For datasets that are Zipfian distributed with
// parameter a, the algorithm provides an expected value of
// delta_ = m / (k ^ a) [3].
//
// This implementation is also designed to support both unweighted and weighted
// data streams [3]. In comparison to the Count-Min Sketch algorithm, the MG
// algorithm requires less space to achieve the same accuracy, but is slower
// [4]. The sketch can be stored on disk for usage later.
//
//    [1] http://www.cohenwang.com/edith/bigdataclass2013/lectures/lecture1.pdf
//    [2] https://www.cs.utah.edu/~jeffp/papers/merge-summ.pdf
//    [3] http://dimacs.rutgers.edu/~graham/pubs/papers/countersj.pdf
//    [4] http://www.cs.toronto.edu/~bor/2420f17/L11.pdf

class MisraGriesSketch {
 public:
  enum class OrderOnTie { kLexicographical, kReverseLexicographical };

  // invalid_utf8_placeholder: if provided, when AddValues() gets a
  //   BinaryArray it will treat elements that are not valid utf-8 sequences in
  //   that array as if they had the value of the placeholder.
  // large_string_threshold: if provided, when AddValues() gets a BinaryArray
  //   it will treat all the elements longer than the threshold as if they
  //   had the value of `large_string_placeholder`. This is useful to limit
  //   the memory usage of the sketch.
  // Note that `large_string_threshold` and `large_string_placeholder` must
  // be both specified or nullopt.
  MisraGriesSketch(int num_buckets,
                   absl::optional<std::string> invalid_utf8_placeholder,
                   absl::optional<int> large_string_threshold,
                   absl::optional<std::string> large_string_placeholder,
                   OrderOnTie order = OrderOnTie::kLexicographical);
  // This class is copyable and movable.
  ~MisraGriesSketch() = default;
  // Adds an array of items, and sets the stored type if it was previously
  // unset. Raises an error if the new types are incompatible.
  absl::Status AddValues(const arrow::Array& items);
  // Adds an array of items with their associated weights. Raises an error if
  // the weights are not a FloatArray, or the type of the newly added items are
  // incompatible with the stored type.
  absl::Status AddValues(const arrow::Array& items,
                         const arrow::Array& weights);
  // Merges another MisraGriesSketch into this sketch. Raises an error if the
  // sketches do not have the same number of buckets.
  absl::Status Merge(const MisraGriesSketch& other);
  // Returns the list of top-k items sorted in descending order of their counts.
  // Ties are resolved (reverse) lexicographically by item value
  // (depending on the value of reverse_lexicograph_order). Returns a non-OK
  // status if decoding fails.
  absl::Status GetCounts(
      std::vector<std::pair<std::string, double>>& result) const;
  // Creates a struct array <values, counts> of the top-k items.
  absl::Status Estimate(
      std::shared_ptr<arrow::Array>* values_and_counts_array) const;
  // Serializes the sketch into a string.
  std::string Serialize() const;
  // Deserializes the string to a MisraGries object.
  static absl::Status Deserialize(absl::string_view encoded,
                                  std::unique_ptr<MisraGriesSketch>* result);
  // Gets delta_.
  double GetDelta() const;
  // Gets theoretical upper bound on delta (for testing purposes).
  double GetDeltaUpperBound(double global_weight) const;
  // Gets the input type of this sketch.
  tfx_bsl::sketches::InputType::Type GetInputType() const;

 private:
  // Reduces size of the item_counts_ to num_buckets_ items.
  void Compress();
  // Encodes a value as a string for storage internally.
  template <typename T> std::string Encode(const T val) const {
    return input_type_ == InputType::FLOAT
                                     ? absl::StrFormat("%a", val)
                                     : absl::StrCat(val);
  }
  // In place decodes `item` into the form produced by Estimate. Returns true
  // on success, or false if item can not be decoded given the stored type.
  absl::Status Decode(std::string* item) const;
  // The number of items stored in item_counts_.
  int num_buckets_;
  // Tracks the maximum error due to subtractions.
  double delta_;
  // Tracks the type of values in this sketch. Values are stored internally as
  // strings, but may be decoded differently depending on their original type.
  tfx_bsl::sketches::InputType::Type input_type_;
  // Dictionary containing lower bound estimates of the item counts.
  absl::flat_hash_map<std::string, double> item_counts_;
  // Set containing extra elements that were discarded from item_counts_ during
  // the latest Compress or DecrementCounters. These are used to fill the result
  // to num_buckets_ size in Estimate.
  absl::flat_hash_set<std::string> extra_items_;
  OrderOnTie order_on_tie_;

  absl::optional<std::string> invalid_utf8_placeholder_;
  absl::optional<int> large_string_threshold_;
  absl::optional<std::string> large_string_placeholder_;
};

}  // namespace sketches
}  // namespace tfx_bsl

#endif  // TFX_BSL_CC_SKETCHES_MISRAGRIES_SKETCH_H_
