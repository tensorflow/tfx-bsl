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

#include "tfx_bsl/cc/sketches/kmv_sketch.h"
#include <gtest/gtest.h>
#include "tfx_bsl/cc/util/status.h"
#include "absl/strings/str_format.h"

namespace tfx_bsl {
namespace sketches {

// Tests to assess the accuracy of the KMV sketch.
// TODO(monicadsong@): Convert these tests to python.

TEST(KmvSketchTest, AddSimpleLargeBinary) {
  KmvSketch kmv(128);

  arrow::LargeBinaryBuilder builder;
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("bar").ok());
  std::shared_ptr<arrow::LargeBinaryArray> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}

TEST(KmvSketchTest, AddSimpleBinary) {
  KmvSketch kmv(128);

  arrow::BinaryBuilder builder;
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("bar").ok());
  std::shared_ptr<arrow::BinaryArray> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}

TEST(KmvSketchTest, AddSimpleString) {
  KmvSketch kmv(128);

  arrow::StringBuilder builder;
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("foo").ok());
  ASSERT_TRUE(builder.Append("bar").ok());
  std::shared_ptr<arrow::StringArray> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}

TEST(KmvSketchTest, AddSimpleInt64) {
  KmvSketch kmv(128);

  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.Append(1).ok());
  ASSERT_TRUE(builder.Append(1).ok());
  ASSERT_TRUE(builder.Append(2).ok());
  std::shared_ptr<arrow::Int64Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}


TEST(KmvSketchTest, MergeSimple) {
  KmvSketch kmv1(128);
  arrow::LargeBinaryBuilder builder;
  ASSERT_TRUE(builder.Append("foo").ok());
  std::shared_ptr<arrow::LargeBinaryArray> array1;
  ASSERT_TRUE(builder.Finish(&array1).ok());
  ASSERT_TRUE(kmv1.AddValues(*array1).ok());;

  KmvSketch kmv2(128);
  arrow::LargeBinaryBuilder builder2;
  ASSERT_TRUE(builder2.Append("bar").ok());
  std::shared_ptr<arrow::LargeBinaryArray> array2;
  ASSERT_TRUE(builder2.Finish(&array2).ok());
  ASSERT_TRUE(kmv2.AddValues(*array2).ok());;

  Status status = kmv1.Merge(kmv2);
  ASSERT_TRUE(status.ok());
  EXPECT_EQ(2, kmv1.Estimate());
}

TEST(KmvSketchTest, AddManyValues) {
  KmvSketch kmv(128);
  for (int batch = 0; batch < 10; batch++) {
    arrow::Int64Builder builder;
    for (int i = 0; i < 1000; i++) {
      ASSERT_TRUE(builder.Append(batch*1000+i).ok());
    }
    std::shared_ptr<arrow::Int64Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    ASSERT_TRUE(kmv.AddValues(*array).ok());;
    uint64_t distinct_estimate = kmv.Estimate();
    int64_t distinct_true = 1000 * (batch + 1);
    int64_t expected_error = 1 / sqrt(128) * distinct_true;

    EXPECT_LE(abs(distinct_true - (int64_t)distinct_estimate),
              expected_error);
  }
}

TEST(KmvSketchTest, MergeManyValues) {
  KmvSketch kmv(128);

  int num_trials = 10;
  for (int trial = 0; trial < num_trials; trial++) {
    KmvSketch new_kmv(128);
    arrow::LargeBinaryBuilder builder;
    for (int i = 0; i < 1000; i++) {
      ASSERT_TRUE(builder.Append(absl::StrFormat("%d", trial*1000+i)).ok());
    }
    std::shared_ptr<arrow::LargeBinaryArray> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    ASSERT_TRUE(new_kmv.AddValues(*array).ok());;
    Status status = kmv.Merge(new_kmv);
    ASSERT_TRUE(status.ok());
  }
  uint64_t distinct_estimate = kmv.Estimate();
  int64_t distinct_true = 1000 * num_trials;
  int64_t expected_error = 1 / sqrt(128) * distinct_true;

  EXPECT_LE(abs(distinct_true - (int64_t)distinct_estimate),
            expected_error);
}

TEST(KmvSketchTest, SerializationPreservesAccuracy) {
  KmvSketch kmv(128);
  for (int batch = 0; batch < 10; batch++) {
    arrow::Int64Builder builder;
    for (int i = 0; i < 1000; i++) {
      ASSERT_TRUE(builder.Append(batch*1000+i).ok());
    }
    std::shared_ptr<arrow::Int64Array> array;
    ASSERT_TRUE(builder.Finish(&array).ok());
    ASSERT_TRUE(kmv.AddValues(*array).ok());;
    std::string serialized_sketch = kmv.Serialize();
    KmvSketch kmv_recovered = KmvSketch::Deserialize(serialized_sketch);

    uint64_t distinct_estimate = kmv.Estimate();
    uint64_t distinct_estimate_recovered = kmv_recovered.Estimate();
    EXPECT_EQ(distinct_estimate, distinct_estimate_recovered);

    int64_t distinct_true = 1000 * (batch + 1);
    int64_t expected_error = 1 / sqrt(128) * distinct_true;
    EXPECT_LE(abs(distinct_true - (int64_t)distinct_estimate),
              expected_error);
  }
}

TEST(KmvSketchTest, MergeDifferentNumBuckets) {
  arrow::Int64Builder builder;
  ASSERT_TRUE(builder.Append(1).ok());
  ASSERT_TRUE(builder.Append(2).ok());
  std::shared_ptr<arrow::Int64Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());

  KmvSketch kmv1(128);
  ASSERT_TRUE(kmv1.AddValues(*array).ok());;
  KmvSketch kmv2(64);
  ASSERT_TRUE(kmv2.AddValues(*array).ok());;

  Status status = kmv1.Merge(kmv2);
  ASSERT_FALSE(status.ok());
  ASSERT_EQ(("Both sketches must have the same number of buckets: 128 v.s. 64"),
            status.error_message());
}

}  // namespace sketches
}  // namespace tfx_bsl
