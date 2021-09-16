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
#include "tfx_bsl/cc/sketches/sketches.pb.h"
#include "tfx_bsl/cc/util/status.h"
#include "absl/strings/str_format.h"

namespace tfx_bsl {
namespace sketches {

// Tests to assess the accuracy of the KMV sketch.
// TODO(b/199515135): Convert these tests to python.

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

TEST(KmvSketchTest, AddSimpleDouble) {
  KmvSketch kmv(128);
  arrow::DoubleBuilder builder;
  ASSERT_TRUE(builder.Append(0.70000001).ok());
  ASSERT_TRUE(builder.Append(0.70000001).ok());
  ASSERT_TRUE(builder.Append(0.7).ok());
  std::shared_ptr<arrow::DoubleArray> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}

TEST(KmvSketchTest, SimilarFloatsAndDoublesAreDistinct) {
  KmvSketch kmv(128);
  arrow::DoubleBuilder builder;
  ASSERT_TRUE(builder.Append(0.70000001).ok());
  ASSERT_TRUE(builder.Append(0.70000001f).ok());
  std::shared_ptr<arrow::DoubleArray> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());;
  EXPECT_EQ(2, kmv.Estimate());
}

TEST(KmvSketchTest, FailsToAddDifferentTypes) {
  KmvSketch kmv(128);
  arrow::DoubleBuilder builder1;
  ASSERT_TRUE(builder1.Append(1).ok());
  std::shared_ptr<arrow::DoubleArray> array1;
  ASSERT_TRUE(builder1.Finish(&array1).ok());
  ASSERT_TRUE(kmv.AddValues(*array1).ok());;

  arrow::StringBuilder builder2;
  // 0x1p+0 is the internal encoding of 1.
  ASSERT_TRUE(builder2.Append("0x1p+0").ok());
  std::shared_ptr<arrow::StringArray> array2;
  ASSERT_TRUE(builder2.Finish(&array2).ok());
  ASSERT_FALSE(kmv.AddValues(*array2).ok());;
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

TEST(KmvSketchTest, MergeDistinctTypesIsError) {
  KmvSketch kmv1(128);
  arrow::LargeBinaryBuilder builder;
  ASSERT_TRUE(builder.Append("foo").ok());
  std::shared_ptr<arrow::LargeBinaryArray> array1;
  ASSERT_TRUE(builder.Finish(&array1).ok());
  ASSERT_TRUE(kmv1.AddValues(*array1).ok());;

  KmvSketch kmv2(128);
  arrow::DoubleBuilder builder2;
  ASSERT_TRUE(builder2.Append(44.44).ok());
  std::shared_ptr<arrow::DoubleArray> array2;
  ASSERT_TRUE(builder2.Finish(&array2).ok());
  ASSERT_TRUE(kmv2.AddValues(*array2).ok());;

  KmvSketch kmv3(128);
  arrow::Int64Builder builder3;
  ASSERT_TRUE(builder3.Append(44).ok());
  std::shared_ptr<arrow::Int64Array> array3;
  ASSERT_TRUE(builder3.Finish(&array3).ok());
  ASSERT_TRUE(kmv3.AddValues(*array3).ok());;


  ASSERT_FALSE(kmv1.Merge(kmv2).ok());
  ASSERT_FALSE(kmv1.Merge(kmv3).ok());
  ASSERT_FALSE(kmv2.Merge(kmv3).ok());
}

// This test case is expected to fail if the hash function changes at all, even
// if the change does not affect its distributional properties. The hard-
// coded value can be updated as long as a change is otherwise verified as
// working.
TEST(KmvSketchTest, CheckExactCount) {
  KmvSketch kmv(128);
  arrow::Int64Builder builder;
  for (int i = 0; i < 10000; i++) {
    ASSERT_TRUE(builder.Append(i).ok());
  }
  std::shared_ptr<arrow::Int64Array> array;
  ASSERT_TRUE(builder.Finish(&array).ok());
  ASSERT_TRUE(kmv.AddValues(*array).ok());
  EXPECT_EQ(kmv.Estimate(), 9670);
}

// TODO(zwestrick): derive more principled error bounds for this and the
// the following test.
TEST(KmvSketchTest, AddManyValues) {
  for (int valcount : {100, 1000, 10000}) {
    std::vector<double> rel_errs(100);
    for (int trial = 0; trial < 100; trial++) {
      KmvSketch kmv(128);
      arrow::Int64Builder builder;
      for (int i = 0; i < valcount; i++) {
        ASSERT_TRUE(builder.Append(trial * 1000000 + i).ok());
      }
      std::shared_ptr<arrow::Int64Array> array;
      ASSERT_TRUE(builder.Finish(&array).ok());
      ASSERT_TRUE(kmv.AddValues(*array).ok());

      uint64_t distinct_estimate = kmv.Estimate();
      int64_t distinct_true = valcount;
      double rel_error = abs(distinct_true - (int64_t)distinct_estimate) /
                         (double)distinct_true;
      rel_errs[trial] = rel_error;
    }
    std::sort(rel_errs.begin(), rel_errs.end());
    // We expect a 50th, 90th, and 95th percentile errors of 1.0 2.0 and 4.0 /
    // sqrt(K) respectively.
    EXPECT_LT(rel_errs[50], 1.0 / sqrt(128));
    EXPECT_LT(rel_errs[90], 2.0 / sqrt(128));
    EXPECT_LT(rel_errs[95], 4.0 / sqrt(128));
  }
}

TEST(KmvSketchTest, SerializationPreservesAccuracy) {
  for (int valcount : {100, 1000, 10000}) {
    std::vector<double> rel_errs(100);
    for (int trial = 0; trial < 100; trial++) {
      KmvSketch kmv(128);
      arrow::Int64Builder builder;
      for (int i = 0; i < valcount; i++) {
        ASSERT_TRUE(builder.Append(trial * 1000000 + i).ok());
      }
      std::shared_ptr<arrow::Int64Array> array;
      ASSERT_TRUE(builder.Finish(&array).ok());
      ASSERT_TRUE(kmv.AddValues(*array).ok());
      const std::string serialized_sketch = kmv.Serialize();
      const KmvSketch kmv_recovered = KmvSketch::Deserialize(serialized_sketch);

      uint64_t distinct_estimate = kmv_recovered.Estimate();
      int64_t distinct_true = valcount;
      double rel_error = abs(distinct_true - (int64_t)distinct_estimate) /
                         (double)distinct_true;
      rel_errs[trial] = rel_error;
    }
    std::sort(rel_errs.begin(), rel_errs.end());
    // We expect a 50th, 90th, and 95th percentile errors of 1.0 2.0 and 4.0 /
    // sqrt(K) respectively.
    EXPECT_LT(rel_errs[50], 1.0 / sqrt(128));
    EXPECT_LT(rel_errs[90], 2.0 / sqrt(128));
    EXPECT_LT(rel_errs[95], 4.0 / sqrt(128));
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
  // TODO(zwestrick): consider relaxing error tolerance of this test.
  EXPECT_LE(abs(distinct_true - (int64_t)distinct_estimate),
            expected_error);
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

TEST(KmvSketchTest, SerializationPreservesInputType) {
  std::shared_ptr<arrow::Array> array;
    arrow::DoubleBuilder builder;
  ASSERT_TRUE(builder.Append(1).ok());
  std::shared_ptr<arrow::DoubleArray> array1;
  ASSERT_TRUE(builder.Finish(&array).ok());
  KmvSketch kmv(128);
  ASSERT_TRUE(kmv.AddValues(*array).ok());

  const std::string serialized_sketch = kmv.Serialize();
  const KmvSketch kmv_recovered = KmvSketch::Deserialize(serialized_sketch);
  EXPECT_EQ(kmv_recovered.GetInputType(), InputType::FLOAT);
}

}  // namespace sketches
}  // namespace tfx_bsl
