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

#include "tfx_bsl/cc/statistics/merge_util.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow_metadata/proto/v0/path.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace statistics {

FeatureId PathToFeatureId(const tensorflow::metadata::v0::Path& path) {
  return {std::vector<std::string>(path.step().begin(), path.step().end()),
          true};
}

namespace {
using tensorflow::metadata::v0::CrossFeatureStatistics;
using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::DatasetFeatureStatisticsList;
using tensorflow::metadata::v0::FeatureNameStatistics;
using SliceKey = std::string;

FeatureId GetFeatureId(const std::string& name) { return {{name}, false}; }

FeatureId GetFeatureId(const FeatureNameStatistics& feature_stats) {
  if (feature_stats.has_path()) return PathToFeatureId(feature_stats.path());
  return GetFeatureId(feature_stats.name());
}

SliceKey GetSliceKey(const DatasetFeatureStatistics& dataset_statistics) {
  return dataset_statistics.name();
}

// Returns false if v0 and v1 are nonzero (non-default in a proto) values, and
// differ from one another.
template <typename T>
bool DistinctNonzeroValues(const T& v0, const T& v1, const T& zero) {
  return v0 != zero && v1 != zero && v0 != v1;
}

absl::Status MergeFeatureStatistics(const FeatureNameStatistics& merge_from,
                                    FeatureNameStatistics* merge_to) {
  // Copy the path/name if not yet set; We don't need to check for matches here,
  // this will only be called on matching or newly initialized items.
  if (merge_from.has_path() && !merge_to->has_path())
    *merge_to->mutable_path() = merge_from.path();
  if (!merge_from.name().empty() && merge_to->name().empty())
    merge_to->set_name(merge_from.name());

  // Set the type.
  if (DistinctNonzeroValues<FeatureNameStatistics::Type>(
          merge_from.type(), merge_to->type(), FeatureNameStatistics::INT)) {
    return absl::InvalidArgumentError(
        "FeatureNameStatistics shards with different types");
  }
  if (merge_from.type() != 0) {
    merge_to->set_type(merge_from.type());
  }
  // Set the stats oneof.
  if (DistinctNonzeroValues<FeatureNameStatistics::StatsCase>(
          merge_from.stats_case(), merge_to->stats_case(),
          FeatureNameStatistics::STATS_NOT_SET)) {
    return absl::InvalidArgumentError(
        "FeatureNameStatistics shards with different stats");
  }
  // TODO(202910677): Consider making this stricter, to match the requirement
  // that we not merge two non-empty messages (other than common stats).
  switch (merge_from.stats_case()) {
    case FeatureNameStatistics::STATS_NOT_SET:
      break;
    case FeatureNameStatistics::kNumStats:
      merge_to->mutable_num_stats()->MergeFrom(merge_from.num_stats());
      break;
    case FeatureNameStatistics::kStringStats:
      merge_to->mutable_string_stats()->MergeFrom(merge_from.string_stats());
      break;
    case FeatureNameStatistics::kBytesStats:
      merge_to->mutable_bytes_stats()->MergeFrom(merge_from.bytes_stats());
      break;
    case FeatureNameStatistics::kStructStats:
      merge_to->mutable_struct_stats()->MergeFrom(merge_from.struct_stats());
      break;
  }
  // Concatenate common stats.
  for (const auto& custom_stat : merge_from.custom_stats())
    *merge_to->add_custom_stats() = custom_stat;

  return absl::OkStatus();
}

absl::Status MergeCrossFeatureStatistics(
    const CrossFeatureStatistics& merge_from,
    CrossFeatureStatistics* merge_to) {
  if (!merge_to->has_path_x())
    *merge_to->mutable_path_x() = merge_from.path_x();
  if (!merge_to->has_path_y())
    *merge_to->mutable_path_y() = merge_from.path_y();
  // Set the count.
  if (DistinctNonzeroValues<uint64_t>(merge_from.count(), merge_to->count(),
                                      0)) {
    return absl::InvalidArgumentError(
        "CrossFeatureStatistics shards with different count");
  }
  if (merge_from.count() != 0) {
    merge_to->set_count(merge_from.count());
  }
  // Set the cross_stats oneof.
  if (DistinctNonzeroValues<CrossFeatureStatistics::CrossStatsCase>(
          merge_from.cross_stats_case(), merge_to->cross_stats_case(),
          CrossFeatureStatistics::CROSS_STATS_NOT_SET)) {
    return absl::InvalidArgumentError(
        "CrossFeatureStatistics shards with different cross_stats");
  }
  // TODO(b/202910677): Consider strictness here too.
  switch (merge_from.cross_stats_case()) {
    case CrossFeatureStatistics::CROSS_STATS_NOT_SET:
      break;
    case CrossFeatureStatistics::kNumCrossStats:
      merge_to->mutable_num_cross_stats()->MergeFrom(
          merge_from.num_cross_stats());
      break;
    case CrossFeatureStatistics::kCategoricalCrossStats:
      merge_to->mutable_categorical_cross_stats()->MergeFrom(
          merge_from.categorical_cross_stats());
      break;
  }

  return absl::OkStatus();
}

// Mutable view into a DatasetFeatureStatistics. This class does not own its
// pointer, so the underlying value must outlive it.
class MutableDatasetViewImpl : public MutableDatasetView {
 public:
  explicit MutableDatasetViewImpl(
      tensorflow::metadata::v0::DatasetFeatureStatistics* value)
      : val_(value) {}

  // Merge with dataset, updating in place. Dataset is assumed to contain the
  // same slice key, which is not checked here.
  absl::Status MergeWith(const DatasetFeatureStatistics& dataset) {
    TFX_BSL_RETURN_IF_ERROR(SetNumExamples(dataset.num_examples()));
    TFX_BSL_RETURN_IF_ERROR(
        SetWeightedNumExamples(dataset.weighted_num_examples()));
    for (const auto& feature : dataset.features()) {
      FeatureId id = GetFeatureId(feature);
      TFX_BSL_RETURN_IF_ERROR(
          MergeFeatureStatistics(feature, GetOrCreateFeature(id)));
    }
    for (const auto& cross_feature : dataset.cross_features()) {
      std::pair<FeatureId, FeatureId> id = {
          PathToFeatureId(cross_feature.path_x()),
          PathToFeatureId(cross_feature.path_y())};
      TFX_BSL_RETURN_IF_ERROR(MergeCrossFeatureStatistics(
          cross_feature, GetOrCreateCrossFeature(id)));
    }
    return absl::OkStatus();
  }

 private:
  // Retrieve a pointer to a feature in val_, or create a new one.
  FeatureNameStatistics* GetOrCreateFeature(const FeatureId& id) {
    auto it = features_.find(id);
    if (it == features_.end()) {
      // Create a new feature and add it.
      FeatureNameStatistics* new_value = val_->add_features();
      features_.insert({id, new_value});
      return new_value;
    }
    return it->second;
  }

  // Retrieve a pointer to cross-feature in val_, or create a new one.
  CrossFeatureStatistics* GetOrCreateCrossFeature(
      const std::pair<FeatureId, FeatureId>& id) {
    auto it = cross_features_.find(id);
    if (it == cross_features_.end()) {
      // Create a new cross feature and add it.
      CrossFeatureStatistics* new_value = val_->add_cross_features();
      cross_features_.insert({id, new_value});
      return new_value;
    }
    return it->second;
  }

  // Sets the num_examples on val_, returning a non-OK status if the value is
  // nonzero and distinct from an existing nonzero count.
  absl::Status SetNumExamples(const uint64_t num_examples) {
    if (DistinctNonzeroValues<int>(num_examples, val_->num_examples(), 0))
      return absl::InvalidArgumentError(
          "DatasetFeatureStatistics have different num_examples");
    if (num_examples != 0) {
      val_->set_num_examples(num_examples);
    }
    return absl::OkStatus();
  }

  // Sets the weighted_num_examples on val_, returning a non-OK status if the
  // value is nonzero and distinct from an existing nonzero count.
  absl::Status SetWeightedNumExamples(const double weighted_num_examples) {
    if (DistinctNonzeroValues<double>(weighted_num_examples,
                                      val_->weighted_num_examples(), 0.0))
      return absl::InvalidArgumentError(
          "DatasetFeatureStatistics shards have different "
          "weighted_num_examples");
    if (weighted_num_examples != 0) {
      val_->set_weighted_num_examples(weighted_num_examples);
    }
    return absl::OkStatus();
  }

  tensorflow::metadata::v0::DatasetFeatureStatistics* const val_;

  absl::flat_hash_map<FeatureId,
                      tensorflow::metadata::v0::FeatureNameStatistics*>
      features_;

  absl::flat_hash_map<std::pair<FeatureId, FeatureId>,
                      tensorflow::metadata::v0::CrossFeatureStatistics*>
      cross_features_;
};

// Mutable view into a DatasetFeatureStatistics. This class does not own its
// pointer, so the underlying value must outlive it.
class MutableDatasetListViewImpl : public MutableDatasetListView {
 public:
  explicit MutableDatasetListViewImpl(
      tensorflow::metadata::v0::DatasetFeatureStatisticsList* value)
      : val_(value) {}

  // Merge with dataset, updating in place.
  absl::Status MergeWith(const DatasetFeatureStatistics& dataset) {
    const SliceKey slice_key = GetSliceKey(dataset);
    return GetOrCreateSlice(slice_key)->MergeWith(dataset);
  }

 private:
  MutableDatasetView* GetOrCreateSlice(
    const SliceKey& slice_key) {
  // Retrieve a pointer to an existing MutableDatasetView, or create a new
  // dataset in val_ and return a MutableDatasetView wrapping it.
    auto it = slices_.find(slice_key);
    if (it == slices_.end()) {
      DatasetFeatureStatistics* new_slice = val_->add_datasets();
      new_slice->set_name(slice_key);
      auto insert_val = slices_.insert(
          {slice_key, std::make_unique<MutableDatasetViewImpl>(new_slice)});
      return insert_val.first->second.get();
    }
    return it->second.get();
  }

  tensorflow::metadata::v0::DatasetFeatureStatisticsList* val_;
  absl::flat_hash_map<SliceKey, std::unique_ptr<MutableDatasetViewImpl>>
      slices_;
};
}  // namespace

DatasetListAccumulator::DatasetListAccumulator(const AccumulatorOptions options)
    : options_(options) {
  auto value = std::make_unique<DatasetFeatureStatisticsList>();
  std::unique_ptr<MutableDatasetListView> view(
      new MutableDatasetListViewImpl(value.get()));
  value_ = std::move(value);
  view_ = std::move(view);
}

absl::StatusOr<std::unique_ptr<DatasetFeatureStatisticsList>>
DatasetListAccumulator::Get() {
  if (value_ == nullptr)
    return absl::InvalidArgumentError("Get called more than once");
  if (options_.target_version != 0)
    return absl::UnimplementedError("Version > 0 unsupported.");
  if (options_.include_empty_placeholder && value_->datasets().empty())
    value_->add_datasets();
  return std::move(value_);
}

absl::Status DatasetListAccumulator::MergeShard(
    const DatasetFeatureStatistics& shard) {
  return view_->MergeWith(shard);
}

}  // namespace statistics
}  // namespace tfx_bsl
