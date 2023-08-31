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
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>

#include <google/protobuf/arena.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/coders/example_coder.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tfx_bsl {

namespace {

using tensorflow::metadata::v0::FeatureType;

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
// A ~2x improvement in the end-to-end (serialzied protos to RecordBatch)
// performance improvement  is possible if we directly use
// proto2::io::CodedInputSteam and bypass the creation of the Example objects.
absl::Status ParseExample(const absl::string_view serialized_example,
                          tensorflow::Example* example) {
  if (!example->ParseFromArray(serialized_example.data(),
                               serialized_example.size())) {
    return absl::DataLossError("Unable to parse example.");
  }
  return absl::OkStatus();
}

absl::Status ParseSequenceExample(
    const absl::string_view serialized_sequence_example,
    tensorflow::SequenceExample* sequence_example) {
  if (!sequence_example->ParseFromArray(serialized_sequence_example.data(),
                                        serialized_sequence_example.size())) {
    return absl::DataLossError("Unable to parse sequence example.");
  }
  return absl::OkStatus();
}

absl::Status TfmdFeatureToArrowField(
    const bool is_sequence_feature,
    const tensorflow::metadata::v0::Feature& feature,
    std::shared_ptr<arrow::Field>* out) {
  const FeatureType feature_type = feature.type();
  switch (feature_type) {
    case tensorflow::metadata::v0::FLOAT: {
      auto type = arrow::large_list(arrow::float32());
      if (is_sequence_feature) {
        type = arrow::large_list(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    case tensorflow::metadata::v0::INT: {
      auto type = arrow::large_list(arrow::int64());
      if (is_sequence_feature) {
        type = arrow::large_list(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    case tensorflow::metadata::v0::BYTES: {
      auto type = arrow::large_list(arrow::large_binary());
      if (is_sequence_feature) {
        type = arrow::large_list(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Bad field type for feature: ", feature.name(),
                       " with type: ", feature_type));
  }
  return absl::OkStatus();
}

}  // namespace

class FeatureDecoder {
 public:
  FeatureDecoder(const std::shared_ptr<arrow::ArrayBuilder>& values_builder)
      : list_builder_(std::make_unique<arrow::LargeListBuilder>(
            arrow::default_memory_pool(), values_builder)),
        feature_was_added_(false) {}
  virtual ~FeatureDecoder() {}

  // Called if the feature is present in the Example.
  absl::Status DecodeFeature(const tensorflow::Feature& feature) {
    if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(list_builder_->AppendNull());
    } else {
      TFX_BSL_RETURN_IF_ERROR_ARROW(list_builder_->Append());
      TFX_BSL_RETURN_IF_ERROR(DecodeFeatureValues(feature));
    }
    if (feature_was_added_) {
      return absl::InternalError(
          "Internal error: FinishFeature() must be called before "
          "DecodeFeature() can be called again.");
    }
    feature_was_added_ = true;
    return absl::OkStatus();
  }

  absl::Status AppendNull() {
    return FromArrowStatus(list_builder_->AppendNull());
  }

  // Called after a (possible) call to DecodeFeature. If DecodeFeature was
  // called, this will do nothing. Otherwise, it will add to the null count.
  absl::Status FinishFeature() {
    if (!feature_was_added_) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(list_builder_->AppendNull());
    }
    feature_was_added_ = false;
    return absl::OkStatus();
  }

  absl::Status Finish(std::shared_ptr<arrow::Array>* out) {
    return FromArrowStatus(list_builder_->Finish(out));
  }

 protected:
  virtual absl::Status DecodeFeatureValues(
      const tensorflow::Feature& feature) = 0;

 private:
  std::unique_ptr<arrow::LargeListBuilder> list_builder_;
  bool feature_was_added_;
};

class FloatDecoder : public FeatureDecoder {
 public:
  static FloatDecoder* Make() {
    return new FloatDecoder(std::make_shared<arrow::FloatBuilder>(
        arrow::float32(), arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kFloatList) {
      return absl::InvalidArgumentError(
          absl::StrCat("Feature had wrong type, expected float_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (float value : feature.float_list().value()) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
    }
    return absl::OkStatus();
  }

 private:
  FloatDecoder(const std::shared_ptr<arrow::FloatBuilder>& values_builder)
      : FeatureDecoder(values_builder), values_builder_(values_builder) {}

  std::shared_ptr<arrow::FloatBuilder> values_builder_;
};

class IntDecoder : public FeatureDecoder {
 public:
  static IntDecoder* Make() {
    return new IntDecoder(std::make_shared<arrow::Int64Builder>(
        arrow::int64(), arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kInt64List) {
      return absl::InvalidArgumentError(
          absl::StrCat("Feature had wrong type, expected int64_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (auto value : feature.int64_list().value()) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
    }
    return absl::OkStatus();
  }

 private:
  IntDecoder(const std::shared_ptr<arrow::Int64Builder>& values_builder)
      : FeatureDecoder(values_builder), values_builder_(values_builder) {}

  std::shared_ptr<arrow::Int64Builder> values_builder_;
};

class BytesDecoder : public FeatureDecoder {
 public:
  static BytesDecoder* Make() {
    return new BytesDecoder(std::make_shared<arrow::LargeBinaryBuilder>(
        arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureValues(
      const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kBytesList) {
      return absl::InvalidArgumentError(
          absl::StrCat("Feature had wrong type, expected bytes_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (const std::string& value : feature.bytes_list().value()) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
    }
    return absl::OkStatus();
  }

 private:
  BytesDecoder(std::shared_ptr<arrow::LargeBinaryBuilder> values_builder)
      : FeatureDecoder(values_builder),
        values_builder_(std::move(values_builder)) {}

  std::shared_ptr<arrow::LargeBinaryBuilder> values_builder_;
};

class FeatureListDecoder {
 public:
  FeatureListDecoder(const std::shared_ptr<arrow::ArrayBuilder>& values_builder)
      : inner_list_builder_(std::make_shared<arrow::LargeListBuilder>(
            arrow::default_memory_pool(), values_builder)),
        outer_list_builder_(std::make_unique<arrow::LargeListBuilder>(
            arrow::default_memory_pool(), inner_list_builder_)),
        feature_list_was_added_(false) {}
  virtual ~FeatureListDecoder() {}

  // Called if the feature list is present in the SequenceExample.
  absl::Status DecodeFeatureList(const tensorflow::FeatureList& feature_list) {
    if (feature_list.feature().empty()) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(outer_list_builder_->Append());
    } else {
      TFX_BSL_RETURN_IF_ERROR_ARROW(outer_list_builder_->Append());
      TFX_BSL_RETURN_IF_ERROR(DecodeFeatureListValues(feature_list));
    }
    if (feature_list_was_added_) {
      return absl::InternalError(
          "Internal error: FinishFeatureList() must be called before "
          "DecodeFeatureList() can be called again.");
    }
    feature_list_was_added_ = true;
    return absl::OkStatus();
  }

  absl::Status AppendNull() {
    return FromArrowStatus(outer_list_builder_->AppendNull());
  }

  absl::Status AppendInnerNulls(const int num_nulls) {
    TFX_BSL_RETURN_IF_ERROR_ARROW(outer_list_builder_->Append());
    TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->AppendNulls(num_nulls));
    return absl::OkStatus();
  }

  // Called after a (possible) call to DecodeFeatureList. If DecodeFeatureList
  // was called this will do nothing.  Otherwise it will add to the null count.
  absl::Status FinishFeatureList() {
    if (!feature_list_was_added_) {
      TFX_BSL_RETURN_IF_ERROR_ARROW(outer_list_builder_->AppendNull());
    }
    feature_list_was_added_ = false;
    return absl::OkStatus();
  }

  absl::Status Finish(std::shared_ptr<arrow::Array>* out) {
    return FromArrowStatus(outer_list_builder_->Finish(out));
  }

 protected:
  virtual absl::Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) = 0;
  std::shared_ptr<arrow::LargeListBuilder> inner_list_builder_;
  std::unique_ptr<arrow::LargeListBuilder> outer_list_builder_;
  bool feature_list_was_added_;
};

class FloatListDecoder : public FeatureListDecoder {
 public:
  static FloatListDecoder* Make() {
    return new FloatListDecoder(std::make_shared<arrow::FloatBuilder>(
        arrow::float32(), arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kFloatList) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->Append());
        for (float value : feature.float_list().value()) {
          TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->AppendNull());
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Feature had wrong type, expected float_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return absl::OkStatus();
  }

 private:
  FloatListDecoder(const std::shared_ptr<arrow::FloatBuilder>& values_builder)
      : FeatureListDecoder(values_builder), values_builder_(values_builder) {}

  std::shared_ptr<arrow::FloatBuilder> values_builder_;
};

class IntListDecoder : public FeatureListDecoder {
 public:
  static IntListDecoder* Make() {
    return new IntListDecoder(std::make_shared<arrow::Int64Builder>(
        arrow::int64(), arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kInt64List) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->Append());
        for (int value : feature.int64_list().value()) {
          TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->AppendNull());
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Feature had wrong type, expected int64_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return absl::OkStatus();
  }

 private:
  IntListDecoder(const std::shared_ptr<arrow::Int64Builder>& values_builder)
      : FeatureListDecoder(values_builder), values_builder_(values_builder) {}

  std::shared_ptr<arrow::Int64Builder> values_builder_;
};

class BytesListDecoder : public FeatureListDecoder {
 public:
  static BytesListDecoder* Make() {
    return new BytesListDecoder(std::make_shared<arrow::LargeBinaryBuilder>(
        arrow::default_memory_pool()));
  }

 protected:
  absl::Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kBytesList) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->Append());
        for (const std::string& value : feature.bytes_list().value()) {
          TFX_BSL_RETURN_IF_ERROR_ARROW(values_builder_->Append(value));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(inner_list_builder_->AppendNull());
      } else {
        return absl::InvalidArgumentError(
            absl::StrCat("Feature had wrong type, expected bytes_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return absl::OkStatus();
  }

 private:
  BytesListDecoder(std::shared_ptr<arrow::LargeBinaryBuilder> values_builder)
      : FeatureListDecoder(values_builder),
        values_builder_(std::move(values_builder)) {}

  std::shared_ptr<arrow::LargeBinaryBuilder> values_builder_;
};

// Decodes a sequence feature where its type is not known.
//
// If the type of the sequence feature is later determined,
// ConvertToTypedListDecoder can convert the current decoder into one of the
// appropriate type. If the type cannot be determined (and thus conversion does
// not occur), this decoder will create an array of type list_type(null),
// where null indicates that the inner list is of an unknown type.
class UnknownTypeFeatureListDecoder {
 public:
  static UnknownTypeFeatureListDecoder* Make() {
    return new UnknownTypeFeatureListDecoder();
  }
  absl::Status DecodeFeatureList(const tensorflow::FeatureList& feature_list) {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() != tensorflow::Feature::KIND_NOT_SET) {
        return absl::InternalError(
            "Attempted to decode a feature list that has a known type with the "
            "UnknownTypeFeatureListDecoder.");
      }
    }
    null_counts_.push_back(feature_list.feature().size());
    feature_list_was_added_ = true;
    return absl::OkStatus();
  }

  void AppendNull() { return null_counts_.push_back(-1); }

  absl::Status ConvertToTypedListDecoder(
      const tensorflow::Feature::KindCase& type,
      FeatureListDecoder** typed_list_decoder) {
    switch (type) {
      case tensorflow::Feature::kInt64List:
        *typed_list_decoder = IntListDecoder::Make();
        break;
      case tensorflow::Feature::kFloatList:
        *typed_list_decoder = FloatListDecoder::Make();
        break;
      case tensorflow::Feature::kBytesList:
        *typed_list_decoder = BytesListDecoder::Make();
        break;
      case tensorflow::Feature::KIND_NOT_SET:
        return absl::InternalError(
            "Attempted to convert an UnknownTypeFeatureListDecoder into a "
            "typed list decoder, but did not specify a valid type.");
        break;
    }
    for (int i = 0; i < null_counts_.size(); ++i) {
      if (null_counts_[i] == -1) {
        TFX_BSL_RETURN_IF_ERROR((*typed_list_decoder)->AppendNull());
      } else {
        TFX_BSL_RETURN_IF_ERROR(
            (*typed_list_decoder)->AppendInnerNulls(null_counts_[i]));
      }
    }
    return absl::OkStatus();
  }

  // Called after a (possible) call to DecodeFeatureList. If DecodeFeatureList
  // was called this will do nothing. Otherwise it will result in a Null being
  // appended to the outer-most list in the resulting arrow array.
  absl::Status FinishFeatureList() {
    if (!feature_list_was_added_) {
      null_counts_.push_back(-1);
    }
    feature_list_was_added_ = false;
    return absl::OkStatus();
  }

  // Becaues the feature type is not known, write the contents out as a
  // list_type(null). Here, null indicates that the inner list is of an unknown
  // type.
  absl::Status Finish(std::shared_ptr<arrow::Array>* out) {
    auto values_builder =
        std::make_shared<arrow::NullBuilder>(arrow::default_memory_pool());
    auto list_builder = std::make_unique<arrow::LargeListBuilder>(
        arrow::default_memory_pool(), values_builder);
    for (int i = 0; i < null_counts_.size(); ++i) {
      if (null_counts_[i] == -1) {
        TFX_BSL_RETURN_IF_ERROR_ARROW(list_builder->AppendNull());
      } else {
        TFX_BSL_RETURN_IF_ERROR_ARROW(list_builder->Append());
        TFX_BSL_RETURN_IF_ERROR_ARROW(
            values_builder->AppendNulls(null_counts_[i]));
      }
    }
    return FromArrowStatus(list_builder->Finish(out));
  }

 private:
  UnknownTypeFeatureListDecoder() {}
  std::vector<int64_t> null_counts_;
  bool feature_list_was_added_;
};  // namespace tfx_bsl

namespace {
// Decodes all top-level features in a specified features map.
// This will be called on a feature map for an individual Example or
// SequenceExample (context features only) while processing a batch of same.
// It can be used where no schema is available (which requires determining the
// coder type from the example(s) seen).
absl::Status DecodeTopLevelFeatures(
    const google::protobuf::Map<std::string, tensorflow::Feature>& features,
    absl::flat_hash_set<std::string>& all_features_seen,
    const int num_examples_already_processed,
    absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>&
        feature_decoders) {
  for (const auto& p : features) {
    const std::string& feature_name = p.first;
    const tensorflow::Feature& feature = p.second;
    const auto it = feature_decoders.find(feature_name);
    FeatureDecoder* feature_decoder = nullptr;
    if (it != feature_decoders.end()) {
      feature_decoder = it->second.get();
    } else {
      all_features_seen.insert(feature_name);
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
        // Handle the situation in which we see a feature for the first time
        // after already having processed some examples. In that case, append
        // a null for each example that has already been processed (in which
        // this feature was not seen), excluding the current example.
        for (int j = 0; j < num_examples_already_processed; ++j) {
          TFX_BSL_RETURN_IF_ERROR(feature_decoder->AppendNull());
        }
        feature_decoders[feature_name] = absl::WrapUnique(feature_decoder);
      }
    }
    if (feature_decoder) {
      absl::Status status = feature_decoder->DecodeFeature(feature);
      if (!status.ok()) {
        return absl::Status(status.code(),
                            absl::StrCat(status.message(), " for feature \"",
                                         feature_name, "\""));
      }
    }
  }

  for (const auto& p : feature_decoders) {
    TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeature());
  }

  return absl::OkStatus();
}

// Finishes features specified by all_features_names by converting them to
// Arrow Arrays and Fields using the corresponding decoder (if available). This
// function can be used to finish features in Examples or context features in
// SequenceExamples. If a feature name is provided for which there is no
// corresponding decoder available, it will create a NullArray for that feature.
absl::Status FinishTopLevelFeatures(
    const absl::flat_hash_set<std::string>& all_feature_names,
    const absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>&
        feature_decoders,
    const int num_examples, std::vector<std::shared_ptr<arrow::Array>>* arrays,
    std::vector<std::shared_ptr<arrow::Field>>* fields) {
  std::vector<std::string> sorted_features(all_feature_names.begin(),
                                           all_feature_names.end());
  std::sort(sorted_features.begin(), sorted_features.end());
  for (const std::string& feature_name : sorted_features) {
    const auto it = feature_decoders.find(feature_name);
    if (it != feature_decoders.end()) {
      FeatureDecoder& decoder = *it->second;
      arrays->emplace_back();
      TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays->back()));
      fields->push_back(arrow::field(feature_name, arrays->back()->type()));
    } else {
      arrays->push_back(std::make_shared<arrow::NullArray>(num_examples));
      fields->push_back(arrow::field(feature_name, arrow::null()));
    }
  }

  return absl::OkStatus();
}

}  // namespace

static absl::Status MakeFeatureDecoder(
    const tensorflow::metadata::v0::Feature& feature,
    std::unique_ptr<FeatureDecoder>* out) {
  const FeatureType feature_type = feature.type();
  switch (feature_type) {
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
      return absl::InvalidArgumentError(
          absl::StrCat("Bad field type for feature: ", feature.name(),
                       " with type: ", feature_type));
  }
  return absl::OkStatus();
}

// static
absl::Status ExamplesToRecordBatchDecoder::Make(
    absl::optional<absl::string_view> serialized_schema,
    std::unique_ptr<ExamplesToRecordBatchDecoder>* result) {
  if (!serialized_schema) {
    *result =
        absl::WrapUnique(new ExamplesToRecordBatchDecoder(nullptr, nullptr));
    return absl::OkStatus();
  }
  auto feature_decoders = std::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>();
  auto schema = std::make_unique<tensorflow::metadata::v0::Schema>();
  if (!schema->ParseFromArray(serialized_schema->data(),
                              serialized_schema->size())) {
    return absl::InvalidArgumentError("Unable to parse schema.");
  }
  std::vector<std::shared_ptr<arrow::Field>> arrow_schema_fields;
  for (const tensorflow::metadata::v0::Feature& feature : schema->feature()) {
    if (feature_decoders->find(feature.name()) != feature_decoders->end()) {
      // TODO(b/160886325): duplicated features in the (same environment) in the
      // schema should be a bug, but before TFDV checks for it, we tolerate it.
      // TODO(b/160885730): the coder is current not environment aware, which
      // means if there are two features of the same name but belonging to
      // different environments, the first feature will be taken.
      continue;
    }
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureDecoder(feature, &(*feature_decoders)[feature.name()]));
    arrow_schema_fields.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(TfmdFeatureToArrowField(
        /*is_sequence_feature=*/false, feature, &arrow_schema_fields.back()));
  }
  *result = absl::WrapUnique(new ExamplesToRecordBatchDecoder(
      arrow::schema(std::move(arrow_schema_fields)),
      std::move(feature_decoders)));
  return absl::OkStatus();
}

ExamplesToRecordBatchDecoder::ExamplesToRecordBatchDecoder(
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::unique_ptr<
        const absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
        feature_decoders)
    : arrow_schema_(std::move(arrow_schema)),
      feature_decoders_(std::move(feature_decoders)) {}

ExamplesToRecordBatchDecoder::~ExamplesToRecordBatchDecoder() {}

absl::Status ExamplesToRecordBatchDecoder::DecodeBatch(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  return feature_decoders_
             ? DecodeFeatureDecodersAvailable(serialized_examples, record_batch)
             : DecodeFeatureDecodersUnavailable(serialized_examples,
                                                record_batch);
}

std::shared_ptr<arrow::Schema> ExamplesToRecordBatchDecoder::ArrowSchema()
    const {
  return arrow_schema_;
}

absl::Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersAvailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_examples.size(); ++i) {
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_examples[i], example));
    for (const auto& p : example->features().feature()) {
      const std::string& feature_name = p.first;
      const tensorflow::Feature& feature = p.second;
      const auto it = feature_decoders_->find(feature_name);
      if (it != feature_decoders_->end()) {
        absl::Status status = it->second->DecodeFeature(feature);
        if (!status.ok()) {
          return absl::Status(status.code(),
                              absl::StrCat(status.message(), " for feature \"",
                                           feature_name, "\""));
        }
      }
    }
    for (const auto& p : *feature_decoders_) {
      TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeature());
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.reserve(arrow_schema_->fields().size());
  for (const std::shared_ptr<arrow::Field>& field : arrow_schema_->fields()) {
    FeatureDecoder& decoder = *feature_decoders_->at(field->name());
    arrays.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays.back()));
  }
  *record_batch = arrow::RecordBatch::Make(arrow_schema_,
                                           serialized_examples.size(), arrays);
  return absl::OkStatus();
}

absl::Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersUnavailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  SchemalessIncrementalExamplesDecoder incremental_decoder;

  google::protobuf::Arena arena;
  for (auto serialized_example : serialized_examples) {
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_example, example));
    TFX_BSL_RETURN_IF_ERROR(incremental_decoder.Add(*example));
  }
  return incremental_decoder.Finish(record_batch);
}

SchemalessIncrementalExamplesDecoder::SchemalessIncrementalExamplesDecoder() {}
SchemalessIncrementalExamplesDecoder::~SchemalessIncrementalExamplesDecoder() {}

absl::Status SchemalessIncrementalExamplesDecoder::Add(
    const tensorflow::Example& example) {
  return DecodeTopLevelFeatures(example.features().feature(), all_features_,
                                num_examples_processed_++, feature_decoders_);
}

void SchemalessIncrementalExamplesDecoder::Reset() {
  feature_decoders_.clear();
  all_features_.clear();
  num_examples_processed_ = 0;
}

absl::Status SchemalessIncrementalExamplesDecoder::Finish(
    std::shared_ptr<arrow::RecordBatch>* result) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  TFX_BSL_RETURN_IF_ERROR(
      FinishTopLevelFeatures(all_features_, feature_decoders_,
                             num_examples_processed_, &arrays, &fields));

  *result = arrow::RecordBatch::Make(arrow::schema(fields),
                                     num_examples_processed_, arrays);

  Reset();
  return absl::OkStatus();
}

static absl::Status MakeFeatureListDecoder(
    const tensorflow::metadata::v0::Feature& feature,
    std::unique_ptr<FeatureListDecoder>* out) {
  const FeatureType feature_type = feature.type();
  switch (feature_type) {
    case tensorflow::metadata::v0::FLOAT:
      out->reset(FloatListDecoder::Make());
      break;
    case tensorflow::metadata::v0::INT:
      out->reset(IntListDecoder::Make());
      break;
    case tensorflow::metadata::v0::BYTES:
      out->reset(BytesListDecoder::Make());
      break;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Bad field type for feature: ", feature.name(),
                       " with type: ", feature_type));
  }
  return absl::OkStatus();
}

absl::Status SequenceExamplesToRecordBatchDecoder::Make(
    const absl::optional<absl::string_view>& serialized_schema,
    const std::string& sequence_feature_column_name,
    std::unique_ptr<SequenceExamplesToRecordBatchDecoder>* result) {
  if (!serialized_schema) {
    *result = absl::WrapUnique(new SequenceExamplesToRecordBatchDecoder(
        sequence_feature_column_name, nullptr, nullptr, nullptr, nullptr));
    return absl::OkStatus();
  }
  auto context_feature_decoders = std::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>();
  auto sequence_feature_decoders = std::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureListDecoder>>>();
  auto schema = std::make_unique<tensorflow::metadata::v0::Schema>();
  if (!schema->ParseFromArray(serialized_schema->data(),
                              serialized_schema->size())) {
    return absl::InvalidArgumentError("Unable to parse schema.");
  }
  std::vector<std::shared_ptr<arrow::Field>> arrow_schema_fields;
  auto sequence_feature_schema_fields =
      std::make_unique<std::vector<std::shared_ptr<arrow::Field>>>();
  for (const tensorflow::metadata::v0::Feature& feature : schema->feature()) {
    if (feature.name() == sequence_feature_column_name) {
      // This feature is a top-level feature containing sequence features, as
      // identified by the sequence_feature_column_name.
      if (feature.type() != tensorflow::metadata::v0::STRUCT) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Found a feature in the schema with the "
            "sequence_feature_column_name (i.e., ",
            sequence_feature_column_name,
            ") that is not a struct. The sequence_feature_column_name should "
            "be used only for the top-level struct feature with a struct "
            "domain that contains each sequence feature as a child."));
      }
      for (const auto& child_feature : feature.struct_domain().feature()) {
        if (sequence_feature_decoders->find(child_feature.name()) !=
            sequence_feature_decoders->end()) {
          // TODO(b/160886325): duplicated features in the (same environment) in
          // the schema should be a bug, but before TFDV checks for it, we
          // tolerate it.
          // TODO(b/160885730): the coder is current not environment aware,
          // which means if there are two features of the same name but
          // belonging to different environments, the first feature will be
          // taken.
          continue;
        }
        TFX_BSL_RETURN_IF_ERROR(MakeFeatureListDecoder(
            child_feature,
            &(*sequence_feature_decoders)[child_feature.name()]));
        sequence_feature_schema_fields->emplace_back();
        TFX_BSL_RETURN_IF_ERROR(TfmdFeatureToArrowField(
            /*is_sequence_feature=*/true, child_feature,
            &sequence_feature_schema_fields->back()));
      }
      continue;
    }
    if (context_feature_decoders->find(feature.name()) !=
        context_feature_decoders->end()) {
      // TODO(b/160886325): duplicated features in the (same environment) in the
      // schema should be a bug, but before TFDV checks for it, we tolerate
      // it.
      // TODO(b/160885730): the coder is current not environment aware, which
      // means if there are two features of the same name but belonging to
      // different environments, the first feature will be taken.
      continue;
    }
    // If the feature is not the top-level sequence feature, it is a context
    // feature.
    TFX_BSL_RETURN_IF_ERROR(MakeFeatureDecoder(
        feature, &(*context_feature_decoders)[feature.name()]));
    arrow_schema_fields.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(TfmdFeatureToArrowField(
        /*is_sequence_feature=*/false, feature, &arrow_schema_fields.back()));
  }
  std::shared_ptr<arrow::StructType> sequence_features_struct_type = nullptr;
  if (!(*sequence_feature_schema_fields).empty()) {
    // Add a single top-level struct field to the arrow schema fields, which
    // contains all of the sequence feature fields.
    sequence_features_struct_type =
        std::make_shared<arrow::StructType>(*sequence_feature_schema_fields);
    arrow_schema_fields.push_back(arrow::field(sequence_feature_column_name,
                                               sequence_features_struct_type));
  }

  *result = absl::WrapUnique(new SequenceExamplesToRecordBatchDecoder(
      sequence_feature_column_name,
      arrow::schema(std::move(arrow_schema_fields)),
      std::move(sequence_features_struct_type),
      std::move(context_feature_decoders),
      std::move(sequence_feature_decoders)));
  return absl::OkStatus();
}

SequenceExamplesToRecordBatchDecoder::SequenceExamplesToRecordBatchDecoder(
    const std::string& sequence_feature_column_name,
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::shared_ptr<arrow::StructType> sequence_features_struct_type,
    std::unique_ptr<
        const absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
        context_feature_decoders,
    std::unique_ptr<const absl::flat_hash_map<
        std::string, std::unique_ptr<FeatureListDecoder>>>
        sequence_feature_decoders)
    : sequence_feature_column_name_(sequence_feature_column_name),
      arrow_schema_(std::move(arrow_schema)),
      sequence_features_struct_type_(std::move(sequence_features_struct_type)),
      context_feature_decoders_(std::move(context_feature_decoders)),
      sequence_feature_decoders_(std::move(sequence_feature_decoders)) {}

SequenceExamplesToRecordBatchDecoder::~SequenceExamplesToRecordBatchDecoder() {}

absl::Status SequenceExamplesToRecordBatchDecoder::DecodeBatch(
    const std::vector<absl::string_view>& serialized_sequence_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  return arrow_schema_ ? DecodeFeatureListDecodersAvailable(
                             serialized_sequence_examples, record_batch)
                       : DecodeFeatureListDecodersUnavailable(
                             serialized_sequence_examples, record_batch);
}

std::shared_ptr<arrow::Schema>
SequenceExamplesToRecordBatchDecoder::ArrowSchema() const {
  return arrow_schema_;
}

absl::Status
SequenceExamplesToRecordBatchDecoder::DecodeFeatureListDecodersAvailable(
    const std::vector<absl::string_view>& serialized_sequence_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_sequence_examples.size(); ++i) {
    auto* sequence_example =
        google::protobuf::Arena::CreateMessage<tensorflow::SequenceExample>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseSequenceExample(
        serialized_sequence_examples[i], sequence_example));
    for (const auto& p : sequence_example->context().feature()) {
      const std::string& context_feature_name = p.first;
      const tensorflow::Feature& context_feature = p.second;
      const auto it = context_feature_decoders_->find(context_feature_name);
      if (it != context_feature_decoders_->end()) {
        absl::Status status = it->second->DecodeFeature(context_feature);
        if (!status.ok()) {
          return absl::Status(status.code(),
                              absl::StrCat(status.message(), " for feature \"",
                                           context_feature_name, "\""));
        }
      }
    }
    for (const auto& p : *context_feature_decoders_) {
      TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeature());
    }

    for (const auto& p : sequence_example->feature_lists().feature_list()) {
      const std::string& feature_list_name = p.first;
      const tensorflow::FeatureList& feature_list = p.second;
      const auto it = sequence_feature_decoders_->find(feature_list_name);
      if (it != sequence_feature_decoders_->end()) {
        absl::Status status = it->second->DecodeFeatureList(feature_list);
        if (!status.ok()) {
          return absl::Status(
              status.code(),
              absl::StrCat(status.message(), " for sequence feature \"",
                           feature_list_name, "\""));
        }
      }
    }
    for (const auto& p : *sequence_feature_decoders_) {
      TFX_BSL_RETURN_IF_ERROR(p.second->FinishFeatureList());
    }
  }

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Array>> sequence_feature_arrays;

  for (const std::shared_ptr<arrow::Field>& field : arrow_schema_->fields()) {
    if (field->name() == sequence_feature_column_name_) {
      for (const std::shared_ptr<arrow::Field>& child_field :
           field->type()->fields()) {
        FeatureListDecoder& decoder =
            *sequence_feature_decoders_->at(child_field->name());
        sequence_feature_arrays.emplace_back();
        TFX_BSL_RETURN_IF_ERROR(
            decoder.Finish(&sequence_feature_arrays.back()));
      }
    } else {
      FeatureDecoder& decoder = *context_feature_decoders_->at(field->name());
      arrays.emplace_back();
      TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays.back()));
    }
  }

  if (sequence_features_struct_type_) {
    std::shared_ptr<arrow::StructArray> sequence_feature_array =
        std::make_shared<arrow::StructArray>(
            sequence_features_struct_type_, serialized_sequence_examples.size(),
            sequence_feature_arrays);
    arrays.push_back(sequence_feature_array);
  }

  *record_batch = arrow::RecordBatch::Make(
      arrow_schema_, serialized_sequence_examples.size(), arrays);
  return absl::OkStatus();
}

absl::Status
SequenceExamplesToRecordBatchDecoder::DecodeFeatureListDecodersUnavailable(
    const std::vector<absl::string_view>& serialized_sequence_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  SchemalessIncrementalSequenceExamplesDecoder incremental_decoder(
      sequence_feature_column_name_);
  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_sequence_examples.size(); ++i) {
    auto* sequence_example =
        google::protobuf::Arena::CreateMessage<tensorflow::SequenceExample>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseSequenceExample(
        serialized_sequence_examples[i], sequence_example));
    TFX_BSL_RETURN_IF_ERROR(incremental_decoder.Add(*sequence_example));
  }
  TFX_BSL_RETURN_IF_ERROR(incremental_decoder.Finish(record_batch));

  return absl::OkStatus();
}

SchemalessIncrementalSequenceExamplesDecoder::
    SchemalessIncrementalSequenceExamplesDecoder(
        const std::string& sequence_feature_column_name)
    : sequence_feature_column_name_(sequence_feature_column_name) {}
SchemalessIncrementalSequenceExamplesDecoder::
    ~SchemalessIncrementalSequenceExamplesDecoder() {}

absl::Status SchemalessIncrementalSequenceExamplesDecoder::Add(
    const tensorflow::SequenceExample& sequence_example) {
  TFX_BSL_RETURN_IF_ERROR(DecodeTopLevelFeatures(
      sequence_example.context().feature(), all_context_features_,
      num_examples_processed_, context_feature_decoders_));
  if (sequence_example.has_feature_lists()) {
    feature_lists_observed_ = true;
  }
  for (const auto& p : sequence_example.feature_lists().feature_list()) {
    const std::string& sequence_feature_name = p.first;
    const tensorflow::FeatureList& sequence_feature_list = p.second;
    FeatureListDecoder* sequence_feature_decoder = nullptr;
    UnknownTypeFeatureListDecoder* unknown_type_sequence_feature_decoder =
        nullptr;
    // Determine if there is an existing decoder for this sequence feature.
    const auto it = sequence_feature_decoders_.find(sequence_feature_name);
    if (it != sequence_feature_decoders_.end()) {
      if (absl::holds_alternative<std::unique_ptr<FeatureListDecoder>>(
              it->second)) {
        sequence_feature_decoder =
            absl::get<std::unique_ptr<FeatureListDecoder>>(it->second).get();
      } else {
        unknown_type_sequence_feature_decoder =
            absl::get<std::unique_ptr<UnknownTypeFeatureListDecoder>>(
                it->second)
                .get();
      }
    }
    // If there was an unknown type decoder for this sequence feature,
    // determine if its type can be determined from the current sequence
    // example. If so, convert it to a feature list decoder of the
    // appropriate type.
    if (unknown_type_sequence_feature_decoder) {
      for (const auto& feature : sequence_feature_list.feature()) {
        if (feature.kind_case() != tensorflow::Feature::KIND_NOT_SET) {
          TFX_BSL_RETURN_IF_ERROR(
              unknown_type_sequence_feature_decoder->ConvertToTypedListDecoder(
                  feature.kind_case(), &sequence_feature_decoder));
          sequence_feature_decoders_[sequence_feature_name] =
              absl::WrapUnique(sequence_feature_decoder);
          unknown_type_sequence_feature_decoder = nullptr;
          break;
        }
      }
    }
    if (sequence_feature_decoder == nullptr &&
        unknown_type_sequence_feature_decoder == nullptr) {
      // If there is no existing decoder for this sequence feature, create
      // one.
      if (sequence_feature_list.feature_size() == 0) {
        unknown_type_sequence_feature_decoder =
            UnknownTypeFeatureListDecoder::Make();
      } else {
        // Determine if the type can be identified from any of the features
        // in the feature list. Use the first type found. If there is a type
        // inconsistency, it will be found and addressed in the decoder.
        tensorflow::Feature::KindCase feature_kind_case =
            tensorflow::Feature::KIND_NOT_SET;
        for (const auto& feature : sequence_feature_list.feature()) {
          if (feature.kind_case() != tensorflow::Feature::KIND_NOT_SET) {
            feature_kind_case = feature.kind_case();
            break;
          }
        }
        switch (feature_kind_case) {
          case tensorflow::Feature::kInt64List:
            sequence_feature_decoder = IntListDecoder::Make();
            break;
          case tensorflow::Feature::kFloatList:
            sequence_feature_decoder = FloatListDecoder::Make();
            break;
          case tensorflow::Feature::kBytesList:
            sequence_feature_decoder = BytesListDecoder::Make();
            break;
          case tensorflow::Feature::KIND_NOT_SET:
            unknown_type_sequence_feature_decoder =
                UnknownTypeFeatureListDecoder::Make();
            break;
        }
      }  // end clause processing a feature list with > 0 features.
      if (unknown_type_sequence_feature_decoder) {
        // Handle the situation in which we see a sequence feature of
        // unknown type for the first time after already having processed
        // some sequence examples. In that case, append a null for each
        // sequence example that has already been processed (in which this
        // sequence feature was not seen), excluding the current sequence
        // example.
        for (int i = 0; i < num_examples_processed_; ++i) {
          unknown_type_sequence_feature_decoder->AppendNull();
        }
        sequence_feature_decoders_[sequence_feature_name] =
            absl::WrapUnique(unknown_type_sequence_feature_decoder);
      } else if (sequence_feature_decoder) {
        // Similarly handle the situation in which we see a sequence feature
        // of a known type for the first time after already having processed
        // some sequence examples.
        for (int i = 0; i < num_examples_processed_; ++i) {
          TFX_BSL_RETURN_IF_ERROR(sequence_feature_decoder->AppendNull());
        }
        sequence_feature_decoders_[sequence_feature_name] =
            absl::WrapUnique(sequence_feature_decoder);
      }
    }  // End adding new decoder.
    // Decode the current feature list using the appropriate feature
    // decoder.
    absl::Status status;
    if (sequence_feature_decoder) {
      status =
          sequence_feature_decoder->DecodeFeatureList(sequence_feature_list);
    } else if (unknown_type_sequence_feature_decoder) {
      status = unknown_type_sequence_feature_decoder->DecodeFeatureList(
          sequence_feature_list);
    }
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat(status.message(), " for sequence feature \"",
                       sequence_feature_name, "\""));
    }
  }  // End processing the current feature list.

  // Calling FinishFeatureList ensures that a Null is appended for a given
  // feature if it was not decoded (e.g., because it was not seen) in the
  // current SequenceExample.
  for (const auto& p : sequence_feature_decoders_) {
    if (absl::holds_alternative<std::unique_ptr<FeatureListDecoder>>(
            p.second)) {
      TFX_BSL_RETURN_IF_ERROR(
          absl::get<std::unique_ptr<FeatureListDecoder>>(p.second)
              .get()
              ->FinishFeatureList());
    } else {
      TFX_BSL_RETURN_IF_ERROR(
          absl::get<std::unique_ptr<UnknownTypeFeatureListDecoder>>(p.second)
              .get()
              ->FinishFeatureList());
    }
  }
  ++num_examples_processed_;

  return absl::OkStatus();
}

absl::Status SchemalessIncrementalSequenceExamplesDecoder::Finish(
    std::shared_ptr<arrow::RecordBatch>* result) {
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  TFX_BSL_RETURN_IF_ERROR(
      FinishTopLevelFeatures(all_context_features_, context_feature_decoders_,
                             num_examples_processed_, &arrays, &fields));

  std::vector<std::shared_ptr<arrow::Array>> sequence_feature_arrays;
  std::vector<std::shared_ptr<arrow::Field>> sequence_feature_fields;
  for (const auto& sequence_feature_decoder : sequence_feature_decoders_) {
    sequence_feature_arrays.emplace_back();
    if (absl::holds_alternative<std::unique_ptr<FeatureListDecoder>>(
            sequence_feature_decoder.second)) {
      TFX_BSL_RETURN_IF_ERROR(absl::get<std::unique_ptr<FeatureListDecoder>>(
                                  sequence_feature_decoder.second)
                                  .get()
                                  ->Finish(&sequence_feature_arrays.back()));
    } else {
      TFX_BSL_RETURN_IF_ERROR(
          absl::get<std::unique_ptr<UnknownTypeFeatureListDecoder>>(
              sequence_feature_decoder.second)
              .get()
              ->Finish(&sequence_feature_arrays.back()));
    }
    sequence_feature_fields.push_back(
        arrow::field(sequence_feature_decoder.first,
                     sequence_feature_arrays.back()->type()));
  }  // end getting arrays for every sequence feature.

  if (!sequence_feature_arrays.empty()) {
    const arrow::Result<std::shared_ptr<arrow::StructArray>>&
        result_or_sequence_feature_array = arrow::StructArray::Make(
            sequence_feature_arrays, sequence_feature_fields);
    absl::Status status =
        FromArrowStatus((result_or_sequence_feature_array.status()));
    if (status != absl::OkStatus()) {
      return absl::InternalError(absl::StrCat(
          "Attempt to make struct array from sequence features failed with "
          "status: ",
          status.message()));
    }
    arrays.push_back(result_or_sequence_feature_array.ValueOrDie());
    fields.push_back(
        arrow::field(sequence_feature_column_name_, arrays.back()->type()));
  } else if (feature_lists_observed_) {
    // If feature lists but no sequence features have been observed, still
    // add a sequence feature column containing a StructArray, but do not
    // include any child arrays in it.
    arrays.push_back(std::make_shared<arrow::StructArray>(
        std::make_shared<arrow::StructType>(sequence_feature_fields),
        num_examples_processed_, sequence_feature_arrays));
    fields.push_back(
        arrow::field(sequence_feature_column_name_, arrays.back()->type()));
  }

  *result = arrow::RecordBatch::Make(arrow::schema(fields),
                                     num_examples_processed_, arrays);

  Reset();
  return absl::OkStatus();
}

void SchemalessIncrementalSequenceExamplesDecoder::Reset() {
  context_feature_decoders_.clear();
  sequence_feature_decoders_.clear();
  all_context_features_.clear();
  feature_lists_observed_ = false;
  num_examples_processed_ = 0;
}

}  // namespace tfx_bsl
