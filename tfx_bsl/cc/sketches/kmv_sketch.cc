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
#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/sketches/sketches.pb.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"
#include <farmhash.h>

namespace tfx_bsl {
namespace sketches {
namespace {
using tfx_bsl::sketches::InputType;

// Creates a vector of the hash values of the non-null values in the array.
class GetHashesVisitor : public arrow::ArrayVisitor {
 public:
  GetHashesVisitor(InputType::Type& input_type) : input_type_(input_type) {}

  const std::vector<uint64_t>& result() const { return result_; }

  arrow::Status Visit(const arrow::BinaryArray& array) override {
    return VisitInternal(array, InputType::RAW_STRING);
  }
  arrow::Status Visit(const arrow::LargeBinaryArray& array) override {
    return VisitInternal(array, InputType::RAW_STRING);
  }
  arrow::Status Visit(const arrow::StringArray& array) override {
    return VisitInternal(array, InputType::RAW_STRING);
  }
  arrow::Status Visit(const arrow::LargeStringArray& array) override {
    return VisitInternal(array, InputType::RAW_STRING);
  }
  arrow::Status Visit(const arrow::Int8Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::Int16Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::Int32Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::Int64Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::UInt8Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::UInt16Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::UInt32Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::UInt64Array& array) override {
    return VisitInternal(array, InputType::INT);
  }
  arrow::Status Visit(const arrow::FloatArray& array) override {
    return VisitInternal(array, InputType::FLOAT);
  }
  arrow::Status Visit(const arrow::DoubleArray& array) override {
    return VisitInternal(array, InputType::FLOAT);
  }

 private:
  std::vector<uint64_t> result_;
  tfx_bsl::sketches::InputType::Type& input_type_;

  arrow::Status SetInputType(const InputType::Type type) {
    if (input_type_ == InputType::UNSET) {
      input_type_ = type;
    }
    if (input_type_ != type) {
      return arrow::Status::TypeError(
          absl::StrFormat("sketch stored type error: stored %s given %s",
                          InputType::Type_Name(input_type_),
                          InputType::Type_Name(input_type_)));
    }
    return arrow::Status::OK();
  }

  template <typename T>
  arrow::Status VisitInternal(const T& array, const InputType::Type type) {
    ARROW_RETURN_NOT_OK(SetInputType(type));
    AddHashes(array);
    return arrow::Status::OK();
  }

  template <typename T>
  void AddHashes(const arrow::NumericArray<T>& numeric_array) {
    result_.reserve(numeric_array.length() - numeric_array.null_count());
    for (int i = 0; i < numeric_array.length(); i++) {
      if (!numeric_array.IsNull(i)) {
        auto value = numeric_array.Value(i);
        if (input_type_ == InputType::FLOAT) {
          result_.push_back(
              farmhash::Fingerprint64(absl::StrFormat("%a", value)));
        } else {
          result_.push_back(farmhash::Fingerprint64(absl::StrCat(value)));
        }
      }
    }
  }

  template <typename T>
  void AddHashes(const arrow::BaseBinaryArray<T>& binary_array) {
    result_.reserve(binary_array.length() - binary_array.null_count());
    for (int i = 0; i < binary_array.length(); i++) {
      if (!binary_array.IsNull(i)) {
        auto string_view = binary_array.GetView(i);
        result_.push_back(
            farmhash::Fingerprint64(string_view.data(), string_view.size()));
      }
    }
  }
};

}  // namespace

KmvSketch::KmvSketch(const int num_buckets)
    : num_buckets_(num_buckets),
      max_limit_(std::numeric_limits<uint64_t>::max()),
      input_type_(InputType::UNSET) {}

Status KmvSketch::AddValues(const arrow::Array& arrow_array) {
  GetHashesVisitor v(input_type_);
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
  if (input_type_ == InputType::UNSET) {
    input_type_ = other.input_type_;
  }
  if (other.input_type_ != InputType::UNSET &&
      input_type_ != other.input_type_) {
    return errors::InvalidArgument(
        absl::StrFormat("Both sketches must have the same type (%s vs %s)",
                        InputType::Type_Name(input_type_),
                        InputType::Type_Name(other.input_type_)));
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
  kmv_proto.set_input_type(input_type_);
  return kmv_proto.SerializeAsString();
}

Status KmvSketch::Deserialize(absl::string_view encoded,
                              std::unique_ptr<KmvSketch>* result) {
  Kmv kmv_proto;
  if (!kmv_proto.ParseFromArray(encoded.data(), encoded.size()))
    return tfx_bsl::errors::InvalidArgument("Failed to parse Kmv sketch");
  const auto proto_hashes = kmv_proto.hashes();
  *result = std::make_unique<KmvSketch>(kmv_proto.num_buckets());
  (*result)->hashes_.insert(proto_hashes.begin(), proto_hashes.end());
  (*result)->max_limit_ = kmv_proto.max_limit();
  (*result)->input_type_ = kmv_proto.input_type();
  return Status::OK();
}

InputType::Type KmvSketch::GetInputType() const {
  return input_type_;
}
}  // namespace sketches
}  // namespace tfx_bsl
