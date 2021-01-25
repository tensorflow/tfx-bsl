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
#ifndef THIRD_PARTY_PY_TFX_BSL_CC_SKETCHES_QUANTILES_SKETCH_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_SKETCHES_QUANTILES_SKETCH_H_

#include <memory>

#include "arrow/api.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"
#include "tfx_bsl/cc/util/status.h"

namespace tfx_bsl {
namespace sketches {

class QuantilesSketchImpl;

// A sketch to estimate quantiles of streams of numbers.
class QuantilesSketch {
 public:
  // eps: Controls the approximation error. Must be >0.
  // max_num_elements: An estimate of maximum number of input values. If not
  //   known at the time of construction, a large-enough number (e.g. 2^32) may
  //   be specified at the cost of extra memory usage. Must be >= 1.
  // num_streams: Number of quantile streams being processed at the same time.
  //   Must be >=1.
  static Status Make(double eps, int64_t max_num_elements, int64_t num_streams,
                     std::unique_ptr<QuantilesSketch>* result);
  ~QuantilesSketch();

  // Disallow copy; allow move.
  QuantilesSketch(const QuantilesSketch&) = delete;
  QuantilesSketch& operator=(const QuantilesSketch&) = delete;
  QuantilesSketch(QuantilesSketch&&);
  QuantilesSketch& operator=(QuantilesSketch&&);

  // Add values (with or without weights) to the sketch. If weights are not
  // provided, they are by default 1.0. If we consider that values are given by
  // rows and streams are given by columns, then values array must have
  // C-contiguous order (stream index varies the fastest). Weights are
  // considered to be the same for all streams.
  // Any numerical arrow array type is accepted. But they will be converted
  // to float64 if they are not of the type. Float truncation may happen (
  // for large int64 values).
  // Values with negative or zero weights will be ignored.
  // Nulls in the array will be skipped.
  Status AddValues(std::shared_ptr<arrow::Array> values);
  Status AddWeightedValues(std::shared_ptr<arrow::Array> values,
                           std::shared_ptr<arrow::Array> weights);

  // Compacts state of the sketch if it wasn't done before and no compact
  // sketches were merged to this, otherwise it is a no-op. Compact() before
  // Serialize() will reduce the size of the sketch. For instance, if
  // eps=0.0001, max_num_elements=2^32, num_streams=1 and the sketch is full,
  // then its size will be reduced by ~16x. Warning: Compact() affects the error
  // bound. It is recommended to multiply the desired worst-case error by 2/3 if
  // this method is going to be used.
  Status Compact();

  // Serializes the sketch into a string.
  Status Serialize(std::string& serialized) const;

  // Deserializes the sketch from a string.
  static Status Deserialize(absl::string_view serialized,
                            std::unique_ptr<QuantilesSketch>* result);

  // Merges the sketch with `other`.
  Status Merge(const QuantilesSketch& other);

  // Get quantiles of the numbers added so far. `quantiles` will be a
  // FixedSizeListArray<float64> where lists represent output for each stream.
  // `num_quantiles` must be >= 2.
  Status GetQuantiles(int64_t num_quantiles,
                      std::shared_ptr<arrow::Array>* quantiles);

 private:
  QuantilesSketch(std::unique_ptr<QuantilesSketchImpl> impl);
  std::unique_ptr<QuantilesSketchImpl> impl_;
};
}  // namespace sketches
}  // namespace tfx_bsl

#endif  // THIRD_PARTY_PY_TFX_BSL_CC_SKETCHES_QUANTILES_SKETCH_H_
