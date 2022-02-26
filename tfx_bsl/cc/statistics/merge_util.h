// Copyright 2021 Google LLC
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

#ifndef THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_MERGE_UTIL_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_MERGE_UTIL_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace statistics {
// TODO(b/202910677): Clearly document what's allowed for sharding, once it's
// finalized.

// TODO(b/215448985): Unexpose these once verification is done.
using FeatureId = std::pair<std::vector<std::string>, bool>;

FeatureId PathToFeatureId(const tensorflow::metadata::v0::Path& path);

class MutableDatasetView {
 public:
  virtual ~MutableDatasetView() = default;

  // Merge with dataset, updating in place. Dataset is assumed to contain the
  // same slice key, which is not checked here.
  virtual absl::Status MergeWith(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& dataset) = 0;
};

class MutableDatasetListView {
 public:
  virtual ~MutableDatasetListView() = default;

  // Merge with dataset, updating in place.
  virtual absl::Status MergeWith(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& dataset) = 0;
};

// Options for DatasetListAccumulator.
class AccumulatorOptions {
 public:
  AccumulatorOptions(int target_version, bool include_empty_placeholder)
      : target_version(target_version),
        include_empty_placeholder(include_empty_placeholder) {}

  static inline AccumulatorOptions Defaults() {
    return AccumulatorOptions(0, true);
  }
  // The DatasetFeatureStatisticsList version to return. Currently only version
  // zero is supported.
  int target_version;
  // If true, include an empty DatasetFeatureStatistics in otherwise empty
  // outputs.
  bool include_empty_placeholder;
};

// DatasetListAccumulator accumulates sharded DatasetFeatureStatistics into
// a single DatasetFeatureStatisticsList, and handles conversion to a target
// version. This class owns its underlying DatasetFeatureStatisticsList value.
class DatasetListAccumulator {
 public:
  explicit DatasetListAccumulator(const AccumulatorOptions options);
  // Retrieve the merger result. This releases ownership of the underlying
  // DatasetFeatureStatisticsList, and can be called at most once.
  absl::StatusOr<
      std::unique_ptr<tensorflow::metadata::v0::DatasetFeatureStatisticsList>>
  Get();

  // Merge with a DatasetFeatureStatistics shard, updating in place.
  absl::Status MergeShard(
      const tensorflow::metadata::v0::DatasetFeatureStatistics& shard);

 private:
  std::unique_ptr<tensorflow::metadata::v0::DatasetFeatureStatisticsList>
      value_;
  std::unique_ptr<MutableDatasetListView> view_;
  const AccumulatorOptions options_;
};

}  // namespace statistics
}  // namespace tfx_bsl

#endif  // THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_MERGE_UTIL_H_
