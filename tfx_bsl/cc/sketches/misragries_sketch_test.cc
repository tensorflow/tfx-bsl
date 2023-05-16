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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace sketches {

using arrow::Array;
// Tests to assess the accuracy of the MisraGries sketch.
// TODO: Decide if c++ tests for sketches should be included in finalized code.

const int kNumBuckets = 128;

MisraGriesSketch MakeSketch(int num_buckets,
                            bool reverse_lexicograph_order = false) {
  const MisraGriesSketch::OrderOnTie order_on_tie =
      reverse_lexicograph_order
          ? MisraGriesSketch::OrderOnTie::kReverseLexicographical
          : MisraGriesSketch::OrderOnTie::kLexicographical;
  return MisraGriesSketch(num_buckets, absl::nullopt, absl::nullopt,
                          absl::nullopt, order_on_tie);
}

void CreateArrowBinaryArrayFromVector(
    const std::vector<std::string> values,
    std::shared_ptr<Array>* array) {
  arrow::BinaryBuilder builder;
  ASSERT_TRUE(builder.Reserve(values.size()).ok());
  ASSERT_TRUE(builder.AppendValues(values).ok());
  ASSERT_TRUE(builder.Finish(array).ok());
}

void CreateArrowIntArrayFromVector(
    const std::vector<int> values,
    std::shared_ptr<Array>* array) {
  arrow::Int32Builder builder;
  ASSERT_TRUE(builder.Reserve(values.size()).ok());
  ASSERT_TRUE(builder.AppendValues(values).ok());
  ASSERT_TRUE(builder.Finish(array).ok());
}

void CreateArrowFloatArrayFromVector(
    const std::vector<float> values,
    std::shared_ptr<Array>* array) {
  arrow::FloatBuilder builder;
  ASSERT_TRUE(builder.Reserve(values.size()).ok());
  ASSERT_TRUE(builder.AppendValues(values).ok());
  ASSERT_TRUE(builder.Finish(array).ok());
}

void CreateArrowDoubleArrayFromVector(const std::vector<double> values,
                                      std::shared_ptr<Array>* array) {
  arrow::DoubleBuilder builder;
  ASSERT_TRUE(builder.Reserve(values.size()).ok());
  ASSERT_TRUE(builder.AppendValues(values).ok());
  ASSERT_TRUE(builder.Finish(array).ok());
}

// TODO(b/199515135): Delete this when migrating to py, since output can be
// a multimap (e.g. for floats).
absl::Status CreateCountsMap(
    const MisraGriesSketch& mg,
    absl::flat_hash_map<std::string, double>& counts_map) {
  std::vector<std::pair<std::string, double>> counts_vector;

  TFX_BSL_RETURN_IF_ERROR(mg.GetCounts(counts_vector));
  for (const auto& pair : counts_vector) {
    counts_map.insert({pair.first, pair.second});
  }
  return absl::OkStatus();
}

TEST(MisraGriesSketchTest, AddSimpleBinary) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);
  std::vector<std::string> values{ "foo", "foo", "bar" };
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);
  ASSERT_TRUE(mg.AddValues(*array).ok());
  std::vector<std::pair<std::string, double>> counts_vector;
  ASSERT_TRUE(mg.GetCounts(counts_vector).ok());
  EXPECT_THAT(counts_vector, testing::ElementsAre(testing::Pair("foo", 2),
                                                  testing::Pair("bar", 1)));
  EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(3));
}

TEST(MisraGriesSketchTest, AddBinaryReverse) {
  MisraGriesSketch mg =
      MakeSketch(kNumBuckets, /*reverse_lexicograph_order=*/true);
  std::vector<std::string> values{"foo", "bar"};
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);
  ASSERT_TRUE(mg.AddValues(*array).ok());

  std::vector<std::pair<std::string, double>> counts_vector;
  ASSERT_TRUE(mg.GetCounts(counts_vector).ok());
  EXPECT_THAT(counts_vector, testing::ElementsAre(testing::Pair("foo", 1),
                                                  testing::Pair("bar", 1)));
}

TEST(MisraGriesSketchTest, AddSimpleInt64) {
  MisraGriesSketch mg = MakeSketch(3);
  std::vector<int> values{ 1, 1, 1, 1, 2, 3 };
  std::shared_ptr<arrow::Array> array;
  CreateArrowIntArrayFromVector(values, &array);

  ASSERT_TRUE(mg.AddValues(*array).ok());

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

  EXPECT_EQ(counts["1"], 4);
  EXPECT_EQ(counts["3"], 1);
  EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(6));
}

TEST(MisraGriesSketchTest, AddSimpleDouble) {
  MisraGriesSketch mg = MakeSketch(3);
  std::vector<double> values{1.0, 1.0, 1.0, 1.0, 2.5, 1.00000001};
  std::shared_ptr<arrow::Array> array;
  CreateArrowDoubleArrayFromVector(values, &array);

  ASSERT_TRUE(mg.AddValues(*array).ok());

  std::vector<std::pair<std::string, double>> counts;
  ASSERT_TRUE(mg.GetCounts(counts).ok());
  // Note: 1.0 and 1.00000001 have distinct encoded representations but the
  // same decoded representation (1).
  EXPECT_THAT(counts,
              testing::ElementsAre(testing::Pair("1", 4), testing::Pair("1", 1),
                                   testing::Pair("2.5", 1)));
  EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(6));
}

TEST(MisraGriesSketchTest, AddDoubleReverse) {
  MisraGriesSketch mg = MakeSketch(3, /*reverse_lexicograph_order=*/true);
  std::vector<double> values{1.0, 1.0, 2.0, 2.0};
  std::shared_ptr<arrow::Array> array;
  CreateArrowDoubleArrayFromVector(values, &array);

  ASSERT_TRUE(mg.AddValues(*array).ok());

  std::vector<std::pair<std::string, double>> counts;
  ASSERT_TRUE(mg.GetCounts(counts).ok());
  // Note: 1.0 and 1.00000001 have distinct encoded representations but the
  // same decoded representation (1).
  EXPECT_THAT(counts, testing::ElementsAre(testing::Pair("2", 2),
                                           testing::Pair("1", 2)));
}

TEST(MisraGriesSketchTest, GetCountsOnEmptySketch) {
  MisraGriesSketch mg = MakeSketch(3);
  std::vector<std::pair<std::string, double>> counts;
  ASSERT_TRUE(mg.GetCounts(counts).ok());
  EXPECT_TRUE(counts.empty());
}

TEST(MisraGriesSketchTest, AddingDistinctTypesIsError) {
  MisraGriesSketch mg = MakeSketch(3);
  std::shared_ptr<arrow::Array> array1;
  CreateArrowFloatArrayFromVector({1.0, 2.0}, &array1);
  std::shared_ptr<arrow::Array> array2;
  CreateArrowBinaryArrayFromVector({"a", "b"}, &array2);

  ASSERT_TRUE(mg.AddValues(*array1).ok());
  ASSERT_FALSE(mg.AddValues(*array2).ok());
}

TEST(MisraGriesSketchTest, MixingFloatAndDoubleIsOK) {
  float val = 0.7;
  std::shared_ptr<arrow::Array> array1;
  CreateArrowFloatArrayFromVector({val}, &array1);
  std::shared_ptr<arrow::Array> array2;
  std::vector<double> array2_vals;
  array2_vals.push_back(val);
  CreateArrowDoubleArrayFromVector(array2_vals, &array2);

  MisraGriesSketch mg = MakeSketch(kNumBuckets);

  ASSERT_TRUE(mg.AddValues(*array1).ok());
  ASSERT_TRUE(mg.AddValues(*array2).ok());
  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg, counts).ok());
  EXPECT_THAT(counts, testing::UnorderedElementsAre(testing::Pair("0.7", 2)));
}

TEST(MisraGriesSketchTest, AddSimpleBinaryWithWeights) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);

  std::vector<std::string> values{ "foo", "foo", "bar" };
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);

  std::vector<float> weights{ 2.0, 3.0, 4.0 };
  std::shared_ptr<arrow::Array> weight_array;
  CreateArrowFloatArrayFromVector(weights, &weight_array);

  ASSERT_TRUE(mg.AddValues(*array, *weight_array).ok());;

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

  EXPECT_EQ(counts["foo"], 5);
  EXPECT_EQ(counts["bar"], 4);
  EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(9.0));
}

TEST(MisraGriesSketchTest, AddSimpleBinaryWithIntWeights) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);

  std::vector<std::string> values{ "foo", "foo", "bar" };
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);

  std::vector<int> weights{ 2, 3, 4 };
  std::shared_ptr<arrow::Array> weight_array;
  CreateArrowIntArrayFromVector(weights, &weight_array);

  absl::Status status = mg.AddValues(*array, *weight_array);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ("Weight array must be float type.", status.error_message());
}

TEST(MisraGriesSketchTest, MergeSimple) {
  std::vector<std::string> values{ "foo", "foo", "bar" };
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array).ok());
  MisraGriesSketch mg2 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg2.AddValues(*array).ok());
  absl::Status status = mg1.Merge(mg2);
  ASSERT_TRUE(status.ok());

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg1, counts).ok());

  EXPECT_EQ(counts["foo"], 4);
  EXPECT_EQ(counts["bar"], 2);
  EXPECT_LE(mg1.GetDelta(), mg1.GetDeltaUpperBound(6));
}

TEST(MisraGriesSketchTest, MergeDistinctTypesIsError) {
  std::shared_ptr<arrow::Array> array1;
  CreateArrowBinaryArrayFromVector({"foo", "foo", "bar"}, &array1);
  std::shared_ptr<arrow::Array> array2;
  CreateArrowFloatArrayFromVector({1.0, 3.0}, &array2);
  std::shared_ptr<arrow::Array> array3;
  CreateArrowIntArrayFromVector({5, 6}, &array3);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array1).ok());
  MisraGriesSketch mg2 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg2.AddValues(*array2).ok());
  MisraGriesSketch mg3 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg3.AddValues(*array3).ok());

  ASSERT_FALSE(mg1.Merge(mg2).ok());
  ASSERT_FALSE(mg1.Merge(mg3).ok());
  ASSERT_FALSE(mg2.Merge(mg3).ok());
}

TEST(MisraGriesSketchTest, MergeEmptyWithNonEmptySketch) {
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector({"a", "b"}, &array);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array).ok());
  MisraGriesSketch mg2 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.Merge(mg2).ok());

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg1, counts).ok());
  EXPECT_THAT(counts, testing::UnorderedElementsAre(testing::Pair("a", 1),
                                                    testing::Pair("b", 1)));
}

TEST(MisraGriesSketchTest, MergeNonEmptyWithEmptySketch) {
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector({"a", "b"}, &array);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array).ok());
  MisraGriesSketch mg2 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg2.Merge(mg1).ok());

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg2, counts).ok());
  EXPECT_THAT(counts, testing::UnorderedElementsAre(testing::Pair("a", 1),
                                                    testing::Pair("b", 1)));
}

TEST(MisraGriesSketchTest, AddWithFewBuckets) {
  // Construct a vector containing k elements equal to 0 and k elements
  // equal to integers 1 to k.
  constexpr double k = 1.0e6;
  std::vector<int> values;
  for (int i = 1; i < k + 1; ++i) {
    values.push_back(0);
    values.push_back(i);
  }
  // Shuffle vector.
  std::random_shuffle(values.begin(), values.end());
  std::shared_ptr<arrow::Array> array;
  CreateArrowIntArrayFromVector(values, &array);

  MisraGriesSketch mg = MakeSketch(2);
  ASSERT_TRUE(mg.AddValues(*array).ok());

  absl::flat_hash_map<std::string, double> counts;
  ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

  double estimated_count = counts["0"];
  EXPECT_LE(k, estimated_count);
  EXPECT_LE(estimated_count - mg.GetDelta(), k);
  EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(2 * k));
}

TEST(MisraGriesSketchTest, MergeWithFewBuckets) {
  MisraGriesSketch old_mg = MakeSketch(10);
  // For i = 1, ... 10, construct a vector containing
  //   k elements equal to i
  //   k elements equal to integers i * 10000 to i * 10000 + k - 1
  //   k/10 elements equal to i * 1000 + j for j = 1 to 10
  std::map<std::string, double> true_counts;
  constexpr int k = 1000;
  for (int i = 1; i <= 10; ++i) {
    std::vector<int> values;
    for (int j = 0; j < k; ++j) {
      values.push_back(i);
      values.push_back(i * 10000 + j);
      true_counts[absl::StrCat(i * 10000 + j)] = 1;
    }
    true_counts[absl::StrCat(i)] = k;

    for (int j = 1; j < 11; ++j) {
      for (int h = 0; h < k/10; ++h) {
         values.push_back(i * 1000 + j);
      }
      true_counts[absl::StrCat(i * 1000 + j)] = k/10;
    }
    // Add to new sketch object and merge with original.
    MisraGriesSketch new_mg = MakeSketch(10);
    std::random_shuffle(values.begin(), values.end());
    std::shared_ptr<arrow::Array> array;
    CreateArrowIntArrayFromVector(values, &array);
    ASSERT_TRUE(new_mg.AddValues(*array).ok());
    ASSERT_TRUE(old_mg.Merge(new_mg).ok());

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(old_mg, counts).ok());

    int global_weight = values.size() * i;
    EXPECT_LE(old_mg.GetDelta(), old_mg.GetDeltaUpperBound(global_weight));
    for (const auto& item : counts) {
      double estimated_count = item.second;
      double true_count = true_counts[item.first];
      EXPECT_LE(true_count, estimated_count);
      EXPECT_LE(estimated_count - old_mg.GetDelta(), true_count);
    }
  }
}

TEST(MisraGriesSketchTest, Estimate) {
  std::vector<std::string> values{ "a", "b", "c", "a", "b" };
  std::shared_ptr<arrow::Array> array;
  CreateArrowBinaryArrayFromVector(values, &array);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array).ok());

  std::shared_ptr<arrow::Array> values_and_counts_array;
  ASSERT_TRUE(mg1.Estimate(&values_and_counts_array).ok());
  EXPECT_LE(mg1.GetDelta(), mg1.GetDeltaUpperBound(5));

  auto result_struct =
      std::dynamic_pointer_cast<arrow::StructArray>(values_and_counts_array);
  auto result_values = static_cast<arrow::LargeBinaryArray*>
      (result_struct->GetFieldByName("values").get());
  auto result_counts = static_cast<arrow::DoubleArray*>
      (result_struct->GetFieldByName("counts").get());

  std::vector<std::pair<absl::string_view, float>> true_counts;
  true_counts.push_back({"a", 2});
  true_counts.push_back({"b", 2});
  true_counts.push_back({"c", 1});
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result_values->GetView(i), true_counts[i].first);
    EXPECT_EQ(result_counts->Value(i), true_counts[i].second);
  }
}

TEST(MisraGriesSketchTest, EstimateFloat) {
  std::shared_ptr<arrow::Array> array;
  CreateArrowFloatArrayFromVector({1.0, 0.7, 0.0}, &array);

  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  ASSERT_TRUE(mg1.AddValues(*array).ok());

  std::shared_ptr<arrow::Array> values_and_counts_array;
  ASSERT_TRUE(mg1.Estimate(&values_and_counts_array).ok());
  EXPECT_LE(mg1.GetDelta(), mg1.GetDeltaUpperBound(5));

  auto result_struct =
      std::dynamic_pointer_cast<arrow::StructArray>(values_and_counts_array);
  auto result_values = static_cast<arrow::LargeBinaryArray*>(
      result_struct->GetFieldByName("values").get());
  auto result_counts = static_cast<arrow::DoubleArray*>(
      result_struct->GetFieldByName("counts").get());

  std::vector<std::pair<absl::string_view, float>> true_counts;

  true_counts.push_back({"0", 1});
  true_counts.push_back({"0.7", 1});
  true_counts.push_back({"1", 1});
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(result_values->GetView(i), true_counts[i].first);
    EXPECT_EQ(result_counts->Value(i), true_counts[i].second);
  }
}

TEST(MisraGriesSketchTest, AddUnweightedLinearDistribution) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);
  // For i =  1, 2,... k, create vector of strings such that such that
  // frequency of str(i) is equal to i. Vector contains k * (k + 1) / 2 =
  // elements total.
  std::vector<std::string> items;
  int k = 200;
  int num_elements = k * (k + 1) / 2;
  items.reserve(num_elements);
  for (int i = 1; i < k + 1; i++) {
    for (int j = 0; j < i; j++) {
      items.push_back(absl::StrCat(i));
    }
  }

  for (int trial = 1; trial < 10; trial++) {
    // Shuffle vector and convert to Arrow BinaryArray.
    std::random_shuffle(items.begin(), items.end());
    std::shared_ptr<arrow::Array> array;
    CreateArrowBinaryArrayFromVector(items, &array);

    ASSERT_TRUE(mg.AddValues(*array).ok());
    int global_weight = num_elements * trial;
    EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(global_weight));

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

    for (auto const& item : counts) {
      int true_count = std::stoi(std::string(item.first)) * trial;
      int estimated_count = item.second;
      EXPECT_LE(true_count, estimated_count);
      EXPECT_LE(estimated_count - mg.GetDelta(), true_count);
    }
  }
}

// For i =  1, 2,... num_elements, create vector such such that the i-th element
// is equal to 1/(i ** 2), which follows the Zipf distribution with exponent
// parameter equal to 2. Many real-world datasets follow Zipf's law.
std::vector<float> GetZipfDistribution(int num_elements) {
  std::vector<float> zipf_frequencies;
  for (int i = 1; i < num_elements + 1; i++) {
    float zipf_frequency = 1. / pow(i, 2);
    zipf_frequencies.push_back(zipf_frequency);
  }
  return zipf_frequencies;
}

// Scale Zipf frequencies to integer values equal to frequency * N.
std::map<std::string, int> GetZipfTrueCounts(
    std::vector<float> frequencies, int N) {
  std::map<std::string, int> true_counts;
  int i = 1;
  for (float z : frequencies) {
    int true_count = z * N;
    true_counts[absl::StrCat(i)] = true_count;
    i++;
  }
  return true_counts;
}

// Create vector with a Zipfian distribution, i.e. item "i" has relative
// frequency 1/(i**2).
std::vector<std::string> CreateZipfVector(
    std::map<std::string, int>zipf_true_counts) {
  std::vector<std::string> items;
  for (auto const& it : zipf_true_counts) {
    for (int i = 0; i < it.second; i++) {
      items.push_back(it.first);
    }
  }
  return items;
}

TEST(MisraGriesSketchTest, AddUnweightedZipfDistribution) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);
  int k = 200;
  int N = 1000000;
  std::vector<float>zipf_frequencies = GetZipfDistribution(k);
  std::map<std::string, int> true_counts =
      GetZipfTrueCounts(zipf_frequencies, N);
  std::vector<std::string> items = CreateZipfVector(true_counts);
  for (int trial = 1; trial < 4; trial++) {
    std::random_shuffle(items.begin(), items.end());
    std::shared_ptr<arrow::Array> array;
    CreateArrowBinaryArrayFromVector(items, &array);
    ASSERT_TRUE(mg.AddValues(*array).ok());
    double global_weight = items.size() * trial;
    EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(global_weight));

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

    for (auto const& item : counts) {
      int true_count = true_counts[std::string(item.first)] * trial;
      int estimated_count = item.second;
      EXPECT_LE(true_count, estimated_count);
      EXPECT_LE(estimated_count - mg.GetDelta(), true_count);
    }
  }
}

TEST(MisraGriesSketchTest, AddWeightedZipfDistribution) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);

  int num_elements = 1000;
  std::vector<float> zipf_frequencies = GetZipfDistribution(num_elements);
  float total_weight = std::accumulate(
      zipf_frequencies.begin(), zipf_frequencies.end(), 0.);
  std::map<std::string, float> true_counts;
  std::vector<std::string> items(num_elements);
  for (int i = 1; i <= num_elements; i++) {
    items.push_back(absl::StrCat(i));
    true_counts[absl::StrCat(i)] = zipf_frequencies[i- 1];
  }

  for (int trial = 1; trial < 5; trial++) {
    std::random_shuffle(items.begin(), items.end());
    std::vector<float> weights;
    for (std::string item : items) {
      weights.push_back(true_counts[item]);
    }

    std::shared_ptr<arrow::Array> array;
    CreateArrowBinaryArrayFromVector(items, &array);
    std::shared_ptr<arrow::Array> weight_array;
    CreateArrowFloatArrayFromVector(weights, &weight_array);

    ASSERT_TRUE(mg.AddValues(*array, *weight_array).ok());
    double global_weight = total_weight * trial;
    EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(global_weight));

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

    for (auto const& item : counts) {
      double true_count = true_counts[std::string(item.first)] * trial;
      double estimated_count = item.second;
      EXPECT_LE(true_count, estimated_count + 1e-8);
      EXPECT_LE(estimated_count - mg.GetDelta(), true_count);
    }
  }
}

TEST(MisraGriesSketchTest, MergeManyValuesUnweightedZipfDistribution) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);
  int num_elements = 500;
  int N = 1000000;
  std::vector<float>zipf_frequencies = GetZipfDistribution(num_elements);
  std::map<std::string, int> true_counts =
      GetZipfTrueCounts(zipf_frequencies, N);
  std::vector<std::string> items = CreateZipfVector(true_counts);

  for (int trial = 1; trial < 5; trial++) {
    std::random_shuffle(items.begin(), items.end());
    std::shared_ptr<arrow::Array> array;
    CreateArrowBinaryArrayFromVector(items, &array);

    MisraGriesSketch new_mg = MakeSketch(kNumBuckets);
    ASSERT_TRUE(new_mg.AddValues(*array).ok());
    ASSERT_TRUE(mg.Merge(new_mg).ok());
    double global_weight = items.size() * trial;
    EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(global_weight));

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

    for (auto const& item : counts) {
      int true_count = true_counts[std::string(item.first)] * trial;
      int estimated_count = item.second;
      EXPECT_LE(true_count, estimated_count);
      EXPECT_LE(estimated_count - mg.GetDelta(), true_count);
    }
  }
}


TEST(MisraGriesSketchTest, ZipfSerializationPreservesAccuracy) {
  MisraGriesSketch mg = MakeSketch(kNumBuckets);
  int num_elements = 1000;
  std::vector<float> zipf_frequencies = GetZipfDistribution(num_elements);
  float total_weight = std::accumulate(
      zipf_frequencies.begin(), zipf_frequencies.end(), 0.);
  std::map<std::string, int> true_counts;
  std::vector<std::string> items(num_elements);
  for (int i = 1; i <= num_elements; i++) {
    items.push_back(absl::StrCat(i));
    true_counts[absl::StrCat(i)] = zipf_frequencies[i- 1];
  }

  for (int trial = 1; trial < 5; trial++) {
    std::random_shuffle(items.begin(), items.end());
    std::vector<float> weights;
    for (std::string item : items) {
      weights.push_back(true_counts[item]);
    }

    std::shared_ptr<arrow::Array> array;
    CreateArrowBinaryArrayFromVector(items, &array);
    std::shared_ptr<arrow::Array> weight_array;
    CreateArrowFloatArrayFromVector(weights, &weight_array);

    ASSERT_TRUE(mg.AddValues(*array, *weight_array).ok());
    // Perform serialization routine.
    std::string serialized_sketch = mg.Serialize();
    std::unique_ptr<MisraGriesSketch> mg_recovered;
    ASSERT_TRUE(
        MisraGriesSketch::Deserialize(serialized_sketch, &mg_recovered).ok());

    absl::flat_hash_map<std::string, double> counts;
    ASSERT_TRUE(CreateCountsMap(mg, counts).ok());

    absl::flat_hash_map<std::string, double> counts_recovered;
    ASSERT_TRUE(CreateCountsMap(*mg_recovered, counts_recovered).ok());

    double global_weight = total_weight * trial;
    EXPECT_LE(mg.GetDelta(), mg.GetDeltaUpperBound(global_weight));

    for (auto const& item : counts) {
      double true_count = true_counts[std::string(item.first)] * trial;
      double estimated_count = item.second;
      double estimated_count_recovered = counts_recovered[item.first];
      EXPECT_DOUBLE_EQ(estimated_count, estimated_count_recovered);
      EXPECT_LE(true_count, estimated_count);
      EXPECT_LE(estimated_count - mg.GetDelta(), true_count);
    }
  }
}

TEST(MisraGriesSketchTest, SerializationPreservesInputType) {
  std::shared_ptr<arrow::Array> array;
  CreateArrowFloatArrayFromVector({1.0}, &array);
  MisraGriesSketch mg1 = MakeSketch(kNumBuckets);
  EXPECT_EQ(mg1.GetInputType(), tfx_bsl::sketches::InputType::UNSET);
  ASSERT_TRUE(mg1.AddValues(*array).ok());

  EXPECT_EQ(mg1.GetInputType(), tfx_bsl::sketches::InputType::FLOAT);
  const std::string s = mg1.Serialize();

  std::unique_ptr<MisraGriesSketch> mg_recovered;
  ASSERT_TRUE(MisraGriesSketch::Deserialize(s, &mg_recovered).ok());

  EXPECT_EQ(mg_recovered->GetInputType(), tfx_bsl::sketches::InputType::FLOAT);
}

}  // namespace sketches
}  // namespace tfx_bsl
