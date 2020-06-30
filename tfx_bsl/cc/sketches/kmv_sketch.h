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

#ifndef TFX_BSL_CC_SKETCHES_KMV_SKETCH_H_
#define TFX_BSL_CC_SKETCHES_KMV_SKETCH_H_

#include <cstdint>
#include <set>
#include <string>

#include "absl/strings/string_view.h"
#include "tfx_bsl/cc/util/status.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"

namespace tfx_bsl {
namespace sketches {

// The K-minimum Values (KMV) algorithm constructs a summary or "sketch" of a
// a dataset and uses it to estimate the number of unique items in the dataset.
// The algorithm is exact if the number of unique elements is smaller than the
// number of buckets used in the algorithm. Otherwise the expected error is
// 1 / sqrt(N) where N is the number of buckets used. There are no guarantees on
// the maximum possible error for this algorithm. See
//   Bar-Yossef Z., Jayram T.S., Kumar R., Sivakumar D., Trevisan L. (2002)
//   Counting Distinct Elements in a Data Stream. RANDOM 2002.
//   http://cs.haifa.ac.il/~ilan/randomized_algorithms/bar-yosef_jayram.pdf
//
// The KmvSketch class implements the KMV algorithm. The size of the sketch
// in the class does not depend on the size of the dataset and thus can be
// computed efficiently for large datasets. The sketch can be stored on disk
// for usage later.
class KmvSketch {
 public:
  // KmvSketch is copyable and movable.
  KmvSketch(int num_buckets);
  ~KmvSketch() = default;
  // Updates the sketch with an Arrow array of values. Supports numeric arrays
  // of all integral types, binary arrays, and string arrays.
  Status AddValues(const arrow::Array& array);
  // Merges the sketch from another object into this sketch. Returns error if
  // the other sketch has a different number of buckets than this sketch.
  Status Merge(KmvSketch& other);
  // Estimates the number of distinct elements.
  uint64_t Estimate() const;
  // Serializes the sketch into a string.
  std::string Serialize() const;
  // Converts the encoded sketch into a KmvSketch object.
  static KmvSketch Deserialize(absl::string_view encoded);

 private:
  // The max number of hashes stored in the sketch.
  int num_buckets_;
  // Ordered set to store the sketch. Ordering is needed because sketch
  // operations require querying the max element and keeping the k smallest
  // elements.
  std::set<uint64_t> hashes_;
  // Upper bound on the k smallest hashes. Any items with hashes larger than
  // max_limit_ will not be considered.
  uint64_t max_limit_;
};

}  // namespace sketches
}  // namespace tfx_bsl

#endif  // TFX_BSL_CC_SKETCHES_KMV_SKETCH_H_
