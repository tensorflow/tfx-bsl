// Copyright 2019 Google LLC
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

#include <algorithm>
#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/util/status.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tfx_bsl {

// Decoder

namespace {

absl::string_view KindToStr(tensorflow::Feature::KindCase kind) {
  switch (kind) {
    case tensorflow::Feature::kBytesList:
      return "bytes_list";
    case tensorflow::Feature::kFloatList:
      return "float_list";
    case tensorflow::Feature::kInt64List:
      return "int64_list";
    case tensorflow::Feature::KIND_NOT_SET:
      return "kind-not-set";
    default:
      return "unknown-kind";
  }
}

// Implementation notes:
// a ~5x improvement in parsing performance (which is ~70% of
// parsing+arrow_building) is possible if we directly use
// proto2::io::CodedInputSteam and bypass the creation of the Example objects.
Status ParseExample(const absl::string_view serialized_example,
                    tensorflow::Example* example) {
  if (!example->ParseFromArray(serialized_example.data(),
                               serialized_example.size())) {
    return errors::DataLoss("Unable to parse example.");
  }
  return Status::OK();
}

}  // namespace

class FeatureDecoder {
 public:
  FeatureDecoder(const std::shared_ptr<::arrow::ArrayBuilder>& values_builder)
      : list_builder_(::arrow::default_memory_pool(), values_builder),
        feature_was_added_(false) {}
  virtual ~FeatureDecoder() {}

  // Called if the feature is present in the Example.
  Status DecodeFeature(
      const tensorflow::Feature& feature) {
    if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(list_builder_.AppendNull()));
    } else {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(list_builder_.Append()));
      TFX_BSL_RETURN_IF_ERROR(DecodeFeatureValues(feature));
    }
    if (feature_was_added_) {
      return errors::Internal(
          "Internal error: FinishFeature() must be called before "
          "DecodeFeature() can be called again.");
    }
    feature_was_added_ = true;
    return Status::OK();
  }

  Status AppendNull() {
    return FromArrowStatus(list_builder_.AppendNull());
  }

  // Called after a (possible) call to DecodeFeature.  If DecodeFeature was
  // called this will do nothing.  Otherwise it will add to the null count.
  Status FinishFeature() {
    if (!feature_was_added_) {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(list_builder_.AppendNull()));
    }
    feature_was_added_ = false;
    return Status::OK();
  }

  Status Finish(std::shared_ptr<::arrow::Array> *out) {
    return FromArrowStatus(list_builder_.Finish(out));
  }

  virtual std::shared_ptr<::arrow::DataType> DataType() = 0;

 protected:
  virtual Status DecodeFeatureValues(
      const tensorflow::Feature& feature) = 0;

 private:
  ::arrow::ListBuilder list_builder_;
  bool feature_was_added_;
};


class FloatDecoder : public FeatureDecoder {
 public:
  FloatDecoder(const std::shared_ptr<::arrow::FloatBuilder>& values_builder) :
    FeatureDecoder(values_builder), values_builder_(values_builder) {
  }

  static FloatDecoder* Make() {
    return new FloatDecoder(std::make_shared<::arrow::FloatBuilder>(
        arrow::float32(), arrow::default_memory_pool()));
  }

  std::shared_ptr<::arrow::DataType> DataType() override {
    return ::arrow::float32();
  }

 protected:
  Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kFloatList) {
      return errors::InvalidArgument(
          absl::StrCat("Feature had wrong type, expected float_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (float value : feature.float_list().value()) {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(values_builder_->Append(value)));
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<::arrow::FloatBuilder> values_builder_;
};


class IntDecoder : public FeatureDecoder {
 public:
  IntDecoder(const std::shared_ptr<::arrow::Int64Builder>& values_builder) :
    FeatureDecoder(values_builder), values_builder_(values_builder) {
  }

  static IntDecoder* Make() {
    return new IntDecoder(std::make_shared<::arrow::Int64Builder>(
        arrow::int64(), arrow::default_memory_pool()));
  }

  std::shared_ptr<::arrow::DataType> DataType() override {
    return ::arrow::int64();
  }

 protected:
  Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kInt64List) {
      return errors::InvalidArgument(
          absl::StrCat("Feature had wrong type, expected in64_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (auto value : feature.int64_list().value()) {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(values_builder_->Append(value)));
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<::arrow::Int64Builder> values_builder_;
};


class BytesDecoder : public FeatureDecoder {
 public:
  BytesDecoder(const std::shared_ptr<::arrow::BinaryBuilder>& values_builder) :
    FeatureDecoder(values_builder), values_builder_(values_builder) {
  }

  static BytesDecoder* Make() {
    return new BytesDecoder(std::make_shared<::arrow::BinaryBuilder>());
  }

  std::shared_ptr<::arrow::DataType> DataType() override {
    return ::arrow::binary();
  }

 protected:
  Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kBytesList) {
      return errors::InvalidArgument(
          absl::StrCat("Feature had wrong type, expected bytes_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (const std::string& value : feature.bytes_list().value()) {
      TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(values_builder_->Append(value)));
    }
    return Status::OK();
  }

 private:
  std::shared_ptr<::arrow::BinaryBuilder> values_builder_;
};


Status MakeFeatureDecoder(
      const tensorflow::metadata::v0::Feature& feature,
      std::unique_ptr<FeatureDecoder>* out) {
  switch (feature.type()) {
    case tensorflow::metadata::v0::FLOAT:
      out->reset(FloatDecoder::Make());
      break;
    case tensorflow::metadata::v0::INT:
      out->reset(IntDecoder::Make());
      break;
    case tensorflow::metadata::v0::BYTES:
      out->reset(BytesDecoder::Make());
      break;
    default:
      return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

// static
Status ExamplesToRecordBatchDecoder::Make(
    absl::optional<absl::string_view> serialized_schema,
    std::unique_ptr<ExamplesToRecordBatchDecoder>* result) {
  if (!serialized_schema) {
    *result =
        absl::WrapUnique(new ExamplesToRecordBatchDecoder(nullptr, nullptr));
    return Status::OK();
  }
  auto feature_decoders = absl::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>();
  auto schema = absl::make_unique<tensorflow::metadata::v0::Schema>();
  if (!schema->ParseFromArray(serialized_schema->data(),
                              serialized_schema->size())) {
    return errors::InvalidArgument("Unable to parse schema.");
  }
  for (const tensorflow::metadata::v0::Feature& feature : schema->feature()) {
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureDecoder(feature, &(*feature_decoders)[feature.name()]));
  }
  *result = absl::WrapUnique(new ExamplesToRecordBatchDecoder(
      std::move(schema), std::move(feature_decoders)));
  return Status::OK();
}

ExamplesToRecordBatchDecoder::ExamplesToRecordBatchDecoder(
    std::unique_ptr<const ::tensorflow::metadata::v0::Schema> schema,
    std::unique_ptr<
        absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
        feature_decoders)
    : schema_(std::move(schema)),
      feature_decoders_(std::move(feature_decoders)) {}

ExamplesToRecordBatchDecoder::~ExamplesToRecordBatchDecoder() {}

Status ExamplesToRecordBatchDecoder::DecodeBatch(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<::arrow::RecordBatch>* record_batch) const {
  return feature_decoders_
             ? DecodeFeatureDecodersAvailable(serialized_examples, record_batch)
             : DecodeFeatureDecodersUnavailable(serialized_examples,
                                                record_batch);
}

Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersAvailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<::arrow::RecordBatch>* record_batch) const {
  tensorflow::Example example;
  for (int i = 0; i < serialized_examples.size(); ++i) {
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_examples[i], &example));
    for (const auto& p : example.features().feature()) {
      const std::string& feature_name = p.first;
      const tensorflow::Feature& feature = p.second;
      if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        // treat features with no kind oneof populated as missing (null).
        continue;
      }
      const auto it = feature_decoders_->find(feature_name);
      if (it != feature_decoders_->end()) {
        Status status = it->second->DecodeFeature(feature);
        if (!status.ok()) {
          return Status(status.code(),
                        absl::StrCat(status.error_message(), " for feature \"",
                        feature_name, "\""));
        }
      }
    }
    for (const auto& p : *feature_decoders_) {
      TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeature());
    }
  }

  std::vector<std::shared_ptr<::arrow::Array>> arrays;
  std::vector<std::shared_ptr<::arrow::Field>> fields;
  for (const tensorflow::metadata::v0::Feature& feature : schema_->feature()) {
    FeatureDecoder& decoder = *(*feature_decoders_)[feature.name()];
    arrays.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays.back()));
    fields.push_back(::arrow::field(feature.name(), arrays.back()->type()));
  }
  *record_batch = ::arrow::RecordBatch::Make(
      arrow::schema(fields), serialized_examples.size(), arrays);
  return Status::OK();
}

// static
Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersUnavailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<::arrow::RecordBatch>* record_batch) {
  absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>
      feature_decoders;
  // all features which have been observed.  `feature_decoders` will only
  // contain features for which a values list was observed, otherwise the
  // feature type cannot be inferred and so the feature decoder cannot be
  // created.
  absl::flat_hash_set<std::string> all_features;

  tensorflow::Example example;
  for (int i = 0; i < serialized_examples.size(); ++i) {
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_examples[i], &example));
    for (const auto& p : example.features().feature()) {
      const std::string& feature_name = p.first;
      const tensorflow::Feature& feature = p.second;
      const auto& it = feature_decoders.find(feature_name);
      FeatureDecoder* feature_decoder = nullptr;
      if (it != feature_decoders.end()) {
        feature_decoder = it->second.get();
      } else {
        all_features.insert(feature_name);
        switch (feature.kind_case()) {
          case tensorflow::Feature::kInt64List:
            feature_decoder = IntDecoder::Make();
            break;
          case tensorflow::Feature::kFloatList:
            feature_decoder = FloatDecoder::Make();
            break;
          case tensorflow::Feature::kBytesList:
            feature_decoder = BytesDecoder::Make();
            break;
          case tensorflow::Feature::KIND_NOT_SET:
            // Leave feature_decoder as nullptr.
            break;
        }
        if (feature_decoder) {
          // Append i nulls.  Note that this will result in 0 nulls being
          // appended when i = 0, and generally will result in the number of
          // nulls being appended equal to the number of examples processsed
          // excluding the current example.
          for (int j = 0; j < i; ++j) {
            TFX_BSL_RETURN_IF_ERROR(feature_decoder->AppendNull());
          }
          feature_decoders[feature_name] = absl::WrapUnique(feature_decoder);
        }
      }
      if (feature_decoder) {
        Status status = feature_decoder->DecodeFeature(feature);
        if (!status.ok()) {
          return Status(status.code(),
                        absl::StrCat(status.error_message(), " for feature \"",
                        feature_name, "\""));
        }
      }
    }

    for (const auto& p : feature_decoders) {
      TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeature());
    }
  }

  std::vector<std::shared_ptr<::arrow::Array>> arrays;
  std::vector<std::shared_ptr<::arrow::Field>> fields;
  std::vector<std::string> sorted_features(all_features.begin(),
                                           all_features.end());
  std::sort(sorted_features.begin(), sorted_features.end());
  for (const std::string& feature_name : sorted_features) {
    const auto& it = feature_decoders.find(feature_name);
    if (it != feature_decoders.end()) {
      FeatureDecoder& decoder = *it->second;
      arrays.emplace_back();
      TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays.back()));
      fields.push_back(::arrow::field(feature_name, arrays.back()->type()));
    } else {
      arrays.emplace_back(new ::arrow::NullArray(serialized_examples.size()));
      fields.push_back(::arrow::field(feature_name, ::arrow::null()));
    }
  }

  *record_batch = ::arrow::RecordBatch::Make(
      arrow::schema(fields), serialized_examples.size(), arrays);
  return Status::OK();
}

// Encoder

namespace {
class FeatureEncoder {
 public:
  FeatureEncoder(const std::shared_ptr<::arrow::ListArray>& list_array):
    list_array_(list_array) {
  }
  virtual ~FeatureEncoder() {}
  Status EncodeFeature(const int32_t index, tensorflow::Feature* feature) {
    if (index >= list_array_->length()) {
      return errors::InvalidArgument(
          absl::StrCat("out-of-bound example index: ", index, " vs ",
                       list_array_->length()));
    }
    const int32_t start_offset = list_array_->raw_value_offsets()[index];
    const int32_t end_offset = list_array_->raw_value_offsets()[index + 1];
    if (list_array_->IsValid(index)) {
      EncodeFeatureValues(start_offset, end_offset, feature);
    }
    return Status::OK();
  }

 protected:
  virtual void EncodeFeatureValues(
      int32_t start, int32_t end, tensorflow::Feature* feature) = 0;

 private:
  std::shared_ptr<::arrow::ListArray> list_array_;
};


class FloatEncoder : public FeatureEncoder {
 public:
  FloatEncoder(const std::shared_ptr<::arrow::ListArray>& list_array,
               const std::shared_ptr<::arrow::FloatArray>& values_array):
      FeatureEncoder(list_array), values_array_(values_array) {
  }

 protected:
  void EncodeFeatureValues(
      int32_t start, int32_t end, tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_float_list()->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<::arrow::FloatArray> values_array_;
};


class IntEncoder : public FeatureEncoder {
 public:
  IntEncoder(const std::shared_ptr<::arrow::ListArray>& list_array,
             const std::shared_ptr<::arrow::Int64Array>& values_array):
      FeatureEncoder(list_array), values_array_(values_array) {
  }

 protected:
  void EncodeFeatureValues(
      int32_t start, int32_t end, tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_int64_list()->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<::arrow::Int64Array> values_array_;
};


class BytesEncoder : public FeatureEncoder {
 public:
  BytesEncoder(const std::shared_ptr<::arrow::ListArray>& list_array,
               const std::shared_ptr<::arrow::BinaryArray>& values_array):
      FeatureEncoder(list_array), values_array_(values_array) {
  }

 protected:
  void EncodeFeatureValues(
      int32_t start, int32_t end, tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_bytes_list()->add_value(
          values_array_->GetString(offset));
    }
  }

 private:
  std::shared_ptr<::arrow::BinaryArray> values_array_;
};


Status MakeFeatureEncoder(
      const std::shared_ptr<::arrow::Array>& array,
      std::unique_ptr<FeatureEncoder>* out) {
  if (array->type()->id() != ::arrow::Type::LIST) {
    return errors::InvalidArgument("Expected ListArray");
  }
  std::shared_ptr<::arrow::ListArray> list_array =
      std::static_pointer_cast<::arrow::ListArray>(array);
  const std::shared_ptr<::arrow::Array>& values_array = list_array->values();
  if (values_array->type()->id() == ::arrow::Type::FLOAT) {
    out->reset(
        new FloatEncoder(
            list_array,
            std::static_pointer_cast<::arrow::FloatArray>(values_array)));
  } else if (values_array->type()->id() == ::arrow::Type::INT64) {
    out->reset(
        new IntEncoder(
            list_array,
            std::static_pointer_cast<::arrow::Int64Array>(values_array)));
  } else if (values_array->type()->id() == ::arrow::Type::BINARY) {
    out->reset(
        new BytesEncoder(
            list_array,
            std::static_pointer_cast<::arrow::BinaryArray>(values_array)));
  } else {
    return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

}  // namespace

Status RecordBatchToExamples(
    const ::arrow::RecordBatch& record_batch,
    std::vector<std::string>* serialized_examples) {

  std::vector<std::pair<std::string, std::unique_ptr<FeatureEncoder>>>
  feature_encoders;

  for (
    int column_index = 0; column_index < record_batch.num_columns();
    ++column_index) {
    const std::shared_ptr<::arrow::Array> array = record_batch.column(
        column_index);
    feature_encoders.emplace_back(
        std::make_pair(record_batch.schema()->field(column_index)->name(),
                       nullptr));
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureEncoder(array, &feature_encoders.back().second));
  }
  std::vector<tensorflow::Example> examples;
  examples.reserve(record_batch.num_rows());
  for (
    int example_index = 0; example_index < record_batch.num_rows();
    ++example_index) {
    examples.emplace_back();
    auto* feature_map = examples.back().mutable_features()->mutable_feature();
    for (const auto& p : feature_encoders) {
      tensorflow::Feature* feature = &(*feature_map)[p.first];
      TFX_BSL_RETURN_IF_ERROR(p.second->EncodeFeature(example_index, feature));
    }
  }
  serialized_examples->clear();
  serialized_examples->reserve(examples.size());
  for (const auto& e : examples) {
    serialized_examples->push_back(e.SerializeAsString());
  }

  return Status::OK();
}

}  // namespace tfx_bsl
