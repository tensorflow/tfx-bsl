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

#include <cmath>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "third_party/farmhash/farmhash_fingerprint.h"


namespace tfx_bsl {
namespace sketches {
namespace {

// Creates a vector of hash values that is the same length as the input Arrow
// array.
class GetHashesVisitor : public arrow::ArrayVisitor {
 public:
  GetHashesVisitor() = default;
  const std::vector<uint64_t>& result() const { return result_; }

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
  std::vector<uint64_t> result_;
  template<typename T>
  arrow::Status VisitInternal(const arrow::NumericArray<T>& numeric_array) {
    result_.reserve(numeric_array.length());
    for (int i = 0; i < numeric_array.length(); i++) {
      auto value = numeric_array.Value(i);
      result_.push_back(farmhash::Fingerprint64(absl::StrCat(value)));
    }
    return arrow::Status::OK();
  }

  template<typename T>
  arrow::Status VisitInternal(const arrow::BaseBinaryArray<T>& binary_array) {
    result_.reserve(binary_array.length());
    for (int i = 0; i < binary_array.length(); i++) {
      auto string_view = binary_array.GetView(i);
      result_.push_back(
          farmhash::Fingerprint64(string_view.data(), string_view.size()));
    }
    return arrow::Status::OK();
  }
};

}  // namespace

KmvSketch::KmvSketch(const int num_buckets)
    : num_buckets_(num_buckets),
      max_limit_(std::numeric_limits<uint64_t>::max()) {}

Status KmvSketch::AddValues(const arrow::Array& arrow_array) {
  GetHashesVisitor v;
  TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(arrow_array.Accept(&v)));
  const std::vector<uint64_t>& hash_values = v.result();
  for (uint64_t hash_value : hash_values) {
    if (hash_value >= max_limit_) {
      continue;
    }
    hashes_.insert(hash_value);
    // If sketch reaches capacity after insertion, set max_limit_ to largest
    // hash.
    if (hashes_.size() == num_buckets_) {
      max_limit_ = *hashes_.rbegin();
    // If sketch exceeds capacity after insertion, evict largest hash and reset
    // max_limit_.
    } else if (hashes_.size() > num_buckets_) {
      hashes_.erase(max_limit_);
      max_limit_ = *hashes_.rbegin();
    }
  }
  return Status::OK();
}

Status KmvSketch::Merge(KmvSketch& other) {
  if (other.num_buckets_ != num_buckets_) {
    return errors::InvalidArgument(
        "Both sketches must have the same number of buckets: ",
         num_buckets_, " v.s. ", other.num_buckets_);
  }
  hashes_.insert(other.hashes_.begin(), other.hashes_.end());
  if (hashes_.size() < num_buckets_) {
    return Status::OK();
  }
  // Keep the smallest n elements of hashes_, where n = num_buckets_.
  auto first_hash_to_remove = hashes_.begin();
  std::advance(first_hash_to_remove, num_buckets_);
  hashes_.erase(first_hash_to_remove, hashes_.end());
  // Set max_limit_ to the largest hash.
  max_limit_ = *hashes_.rbegin();
  return Status::OK();
}

uint64_t KmvSketch::Estimate() const {
  // This represents the total range spanned by a random uint64.
  static const double kHashSpaceSize = pow(2, 64);
  size_t num_hashes = hashes_.size();
  if (num_hashes < num_buckets_) {
    return num_hashes;
  } else {
    uint64_t total =
        std::round((num_buckets_ - 1) * kHashSpaceSize / max_limit_);
    return total;
  }
}

std::string KmvSketch::Serialize() const {
  Kmv kmv_proto;
  kmv_proto.set_num_buckets(num_buckets_);
  for (uint64_t hash : hashes_) {
    kmv_proto.add_hashes(hash);
  }
  kmv_proto.set_max_limit(max_limit_);
  return kmv_proto.SerializeAsString();
}

KmvSketch KmvSketch::Deserialize(absl::string_view encoded) {
  Kmv kmv_proto;
  kmv_proto.ParseFromArray(encoded.data(), encoded.size());
  const proto2::RepeatedField<uint64>& proto_hashes = kmv_proto.hashes();

  KmvSketch kmv_new{kmv_proto.num_buckets()};

  kmv_new.hashes_.insert(proto_hashes.begin(), proto_hashes.end());
  kmv_new.max_limit_ = kmv_proto.max_limit();
  return kmv_new;
}

}  // namespace sketches
}  // namespace tfx_bsl
