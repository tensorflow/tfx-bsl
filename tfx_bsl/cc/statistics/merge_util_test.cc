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

#include <memory>

#include "google/protobuf/text_format.h"
#include "testing/base/public/gmock.h"
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "absl/status/status.h"
#include "tensorflow_metadata/proto/v0/path.pb.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"
#include "vector"

namespace {
using proto2::TextFormat;
using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::DatasetFeatureStatisticsList;
using tfx_bsl::statistics::AccumulatorOptions;
using tfx_bsl::statistics::DatasetListAccumulator;

// Helper for tests.
absl::StatusOr<std::unique_ptr<DatasetFeatureStatisticsList>> Merge(
    const std::vector<DatasetFeatureStatistics>& shards,
    const AccumulatorOptions& opts = AccumulatorOptions::Defaults()) {
  DatasetListAccumulator acc = DatasetListAccumulator(opts);
  for (const auto& shard : shards) {
    TFX_BSL_RETURN_IF_ERROR(acc.MergeShard(shard));
  }
  return acc.Get();
}

TEST(MergeUtilTest, MergesTwoSlices) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "num_examples: 451 "
                                  "weighted_num_examples: 3.5 ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice2' "
                                  "num_examples: 3 "
                                  "weighted_num_examples: 0.5 ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "num_examples: 451 "
                                  "weighted_num_examples: 3.5 "
                                  "} "
                                  "datasets: { "
                                  "name: 'slice2' "
                                  "num_examples: 3 "
                                  "weighted_num_examples: 0.5 "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, MergesSingleShardedSlice) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "num_examples: 451 ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "weighted_num_examples: 0.5 ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "num_examples: 451 "
                                  "weighted_num_examples: 0.5 "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, InconsistentNumExamplesIsAnError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "num_examples: 1 ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "num_examples: 2 ",
                                  &slice2));
  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, InconsistentWeightedNumExamplesIsAnError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "weighted_num_examples: 1 ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "weighted_num_examples: 2 ",
                                  &slice2));
  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, MergesDistinctFeaturesByName) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  name: 'feature_name1' "
                                  "  type: STRING "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  name: 'feature_name2' "
                                  "  type: FLOAT "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "features: { "
                                  "  name: 'feature_name1' "
                                  "  type: STRING "
                                  "} "
                                  "features: { "
                                  "  name: 'feature_name2' "
                                  "  type: FLOAT "
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, MergesDistinctFeaturesByPath) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  type: FLOAT "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "} "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  type: FLOAT "
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, MergesSameFeaturesByPath) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  string_stats: { "
                                  "     unique: 54 "
                                  "  } "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "  string_stats: { "
                                  "     unique: 54 "
                                  "  } "
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, InconsistentFeatureTypesIsAnError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: FLOAT "
                                  "} ",
                                  &slice2));
  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, InconsistentStatsOneofIsAnError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  string_stats: { "
                                  "  } "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  num_stats: { "
                                  "  } "
                                  "} ",
                                  &slice2));
  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, MergesDistinctCrossFeatures) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  count: 1234 "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature3' "
                                  "  } "
                                  "  count: 5678 "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  count: 1234 "
                                  "} "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature3' "
                                  "  } "
                                  "  count: 5678 "
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, MergesWithDerivedSource) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "  validation_derived_source: {"
                                  "     deriver_name: 'foo'"
                                  "     source_path: {step: 'source_path'}"
                                  "  }"
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  string_stats: { "
                                  "     unique: 54 "
                                  "  } "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "  string_stats: { "
                                  "     unique: 54 "
                                  "  } "
                                  "  validation_derived_source: {"
                                  "     deriver_name: 'foo'"
                                  "     source_path: {step: 'source_path'}"
                                  "  }"
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, ConflictingDerivedSourceIsError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  type: STRING "
                                  "  validation_derived_source: {"
                                  "     deriver_name: 'foo'"
                                  "  }"
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "features: { "
                                  "  path: { "
                                  "   step: 'path' "
                                  "   step: 'to' "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  validation_derived_source: {"
                                  "     deriver_name: 'bar'"
                                  "  }"
                                  "} ",
                                  &slice2));
  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, MergesSameCrossFeatures) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  count: 1234 "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  num_cross_stats: { "
                                  "   correlation: 0.6 "
                                  "  } "
                                  "} ",
                                  &slice2));
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(
      TextFormat::ParseFromString("datasets: { "
                                  "name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  count: 1234 "
                                  "  num_cross_stats: { "
                                  "   correlation: 0.6 "
                                  "  } "
                                  "} "
                                  "} ",
                                  &expected));
  auto output_or = Merge({slice1, slice2});
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList output = **output_or;
  EXPECT_THAT(output, testing::proto::IgnoringRepeatedFieldOrdering(
                          testing::EqualsProto(expected)));
}

TEST(MergeUtilTest, InconsistentCrossStatsIsAnError) {
  DatasetFeatureStatistics slice1;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  categorical_cross_stats: { "
                                  "  } "
                                  "} ",
                                  &slice1));
  DatasetFeatureStatistics slice2;
  ASSERT_TRUE(
      TextFormat::ParseFromString("name: 'slice1' "
                                  "cross_features: { "
                                  "  path_x: { "
                                  "   step: 'feature1' "
                                  "  } "
                                  "  path_y: { "
                                  "   step: 'feature2' "
                                  "  } "
                                  "  num_cross_stats: { "
                                  "  } "
                                  "} ",
                                  &slice2));

  auto output_or = Merge({slice1, slice2});
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, VersionNonzeroError) {
  auto output_or = Merge({}, AccumulatorOptions(1, true));
  ASSERT_FALSE(output_or.ok());
}

TEST(MergeUtilTest, EmptyPlaceholderTrue) {
  auto output_or = Merge({}, AccumulatorOptions(0, true));
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList expected;
  ASSERT_TRUE(TextFormat::ParseFromString("datasets: {} ", &expected));
  EXPECT_THAT(**output_or, testing::EqualsProto(expected));
}

TEST(MergeUtilTest, EmptyPlaceholderFalse) {
  auto output_or = Merge({}, AccumulatorOptions(0, false));
  ASSERT_OK(output_or.status());
  DatasetFeatureStatisticsList expected;
  EXPECT_THAT(**output_or, testing::EqualsProto(expected));
}

}  // namespace
