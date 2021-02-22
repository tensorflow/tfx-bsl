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
#include "tfx_bsl/cc/coders/example_coder.h"

#include <google/protobuf/arena.h>
#include "absl/container/flat_hash_set.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace tfx_bsl {

namespace {

class FeatureEncoderInterface {
 public:
  virtual ~FeatureEncoderInterface() = default;
  virtual Status EncodeFeature(const int64_t index,
                               tensorflow::Feature* feature) = 0;

 protected:
  virtual void EncodeFeatureValues(int64_t start, int64_t end,
                                   tensorflow::Feature* feature) = 0;
};

template <typename ListT>
class FeatureEncoder : public FeatureEncoderInterface {
 public:
  FeatureEncoder(const std::shared_ptr<ListT>& list_array)
      : list_array_(list_array) {}
  virtual ~FeatureEncoder() {}
  Status EncodeFeature(const int64_t index, tensorflow::Feature* feature) {
    if (index >= list_array_->length()) {
      return errors::InvalidArgument(
          absl::StrCat("out-of-bound example index: ", index, " vs ",
                       list_array_->length()));
    }
    if (list_array_->IsValid(index)) {
      const int64_t start_offset = list_array_->value_offset(index);
      const int64_t end_offset = list_array_->value_offset(index + 1);
      EncodeFeatureValues(start_offset, end_offset, feature);
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<ListT> list_array_;
};

template <typename ListT>
class FloatEncoder : public FeatureEncoder<ListT> {
 public:
  FloatEncoder(const std::shared_ptr<ListT>& list_array,
               const std::shared_ptr<arrow::FloatArray>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) override {
    auto float_list = feature->mutable_float_list();
    for (int64_t offset = start; offset < end; ++offset) {
      float_list->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<arrow::FloatArray> values_array_;
};

template <typename ListT>
class IntEncoder : public FeatureEncoder<ListT> {
 public:
  IntEncoder(const std::shared_ptr<ListT>& list_array,
             const std::shared_ptr<arrow::Int64Array>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) override {
    auto int64_list = feature->mutable_int64_list();
    for (int64_t offset = start; offset < end; ++offset) {
      int64_list->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<arrow::Int64Array> values_array_;
};

template <typename ListT, typename BinaryArrayT>
class BytesEncoder : public FeatureEncoder<ListT> {
  using offset_type = typename BinaryArrayT::offset_type;

 public:
  BytesEncoder(const std::shared_ptr<ListT>& list_array,
               const std::shared_ptr<BinaryArrayT>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) override {
    auto bytes_list = feature->mutable_bytes_list();
    for (int64_t offset = start; offset < end; ++offset) {
      offset_type length;
      const char* value = reinterpret_cast<const char*>(
          values_array_->GetValue(offset, &length));
      bytes_list->add_value(value, length);
    }
  }

 private:
  std::shared_ptr<BinaryArrayT> values_array_;
};

template <typename ListT>
Status MakeFeatureEncoderHelper(const std::shared_ptr<ListT>& list_array,
                                std::unique_ptr<FeatureEncoderInterface>* out) {
  const std::shared_ptr<arrow::Array>& values_array = list_array->values();
  switch (values_array->type()->id()) {
    case arrow::Type::FLOAT:
      *out = absl::make_unique<FloatEncoder<ListT>>(
          list_array,
          std::static_pointer_cast<arrow::FloatArray>(values_array));
      return Status::OK();
    case arrow::Type::INT64:
      *out = absl::make_unique<IntEncoder<ListT>>(
          list_array,
          std::static_pointer_cast<arrow::Int64Array>(values_array));
      return Status::OK();
    case arrow::Type::BINARY:
      *out = absl::make_unique<BytesEncoder<ListT, arrow::BinaryArray>>(
          list_array,
          std::static_pointer_cast<arrow::BinaryArray>(values_array));
      return Status::OK();
    case arrow::Type::LARGE_BINARY:
      *out = absl::make_unique<BytesEncoder<ListT, arrow::LargeBinaryArray>>(
          list_array,
          std::static_pointer_cast<arrow::LargeBinaryArray>(values_array));
      return Status::OK();
    default:
      return errors::InvalidArgument("Bad field type");
  }
}

Status MakeFeatureEncoder(const std::shared_ptr<arrow::Array>& array,
                          std::unique_ptr<FeatureEncoderInterface>* out) {
  switch (array->type()->id()) {
    case arrow::Type::LIST: {
      std::shared_ptr<arrow::ListArray> list_array =
          std::static_pointer_cast<arrow::ListArray>(array);
      return MakeFeatureEncoderHelper<arrow::ListArray>(list_array, out);
    }
    case arrow::Type::LARGE_LIST: {
      std::shared_ptr<arrow::LargeListArray> large_list_array =
          std::static_pointer_cast<arrow::LargeListArray>(array);
      return MakeFeatureEncoderHelper<arrow::LargeListArray>(large_list_array,
                                                             out);
    }
    default:
      return errors::InvalidArgument("Expected ListArray or LargeListArray");
  }
}

}  // namespace

Status RecordBatchToExamples(const arrow::RecordBatch& record_batch,
                             std::vector<std::string>* serialized_examples) {
  std::vector<std::pair<std::string, std::unique_ptr<FeatureEncoderInterface>>>
      feature_encoders;
  feature_encoders.reserve(record_batch.num_columns());
  const std::vector<std::string> field_names =
      record_batch.schema()->field_names();

  // Checking that field names are distinct.
  std::unordered_set<std::string> distinct_field_names;
  for (const auto& name : field_names) {
    if (!distinct_field_names.insert(name).second) {
      return errors::InvalidArgument(
          "RecordBatch contains duplicate column names");
    }
  }

  for (const auto& name : field_names) {
    const std::shared_ptr<arrow::Array> array =
        record_batch.GetColumnByName(name);
    feature_encoders.emplace_back(name, nullptr);
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureEncoder(array, &feature_encoders.back().second));
  }

  serialized_examples->resize(record_batch.num_rows());
  for (int64_t example_index = 0; example_index < record_batch.num_rows();
       ++example_index) {
    google::protobuf::Arena arena;
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    auto* feature_map = example->mutable_features()->mutable_feature();
    for (const auto& p : feature_encoders) {
      tensorflow::Feature* feature = &(*feature_map)[p.first];
      TFX_BSL_RETURN_IF_ERROR(p.second->EncodeFeature(example_index, feature));
    }
    if (!example->SerializeToString(&(*serialized_examples)[example_index])) {
      return errors::DataLoss("Unable to serialize example");
    }
  }

  return Status::OK();
}

}  // namespace tfx_bsl
