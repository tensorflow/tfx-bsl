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
#include <functional>
#include <memory>

#include <google/protobuf/arena.h>
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/types/variant.h"
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
// A large (perhaps ~5x?) improvement in parsing performance (which is ~70% of
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

Status ParseSequenceExample(const absl::string_view serialized_sequence_example,
                            tensorflow::SequenceExample* sequence_example) {
  if (!sequence_example->ParseFromArray(serialized_sequence_example.data(),
                                        serialized_sequence_example.size())) {
    return errors::DataLoss("Unable to parse sequence example.");
  }
  return Status::OK();
}

// LargeListBuilder and ListBuilder don't share the same base class. We create
// wrappers of them so the wrappers can share.
// TODO(b/154119411): we should start always producing LargeList. Clean this
// up.
class ListBuilderInterface {
 public:
  virtual ~ListBuilderInterface() = default;
  virtual Status Append() = 0;
  virtual Status AppendNull() = 0;
  virtual Status AppendNulls(int64_t num_nulls) = 0;
  virtual Status Finish(std::shared_ptr<arrow::Array>* out) = 0;
  virtual std::shared_ptr<arrow::ArrayBuilder> wrapped() = 0;
};

template <typename ListBuilderT>
class ListBuilderWrapper : public ListBuilderInterface {
 public:
  ListBuilderWrapper(const std::shared_ptr<arrow::ArrayBuilder>& values_builder,
                     arrow::MemoryPool* memory_pool)
      : list_builder_(
            std::make_shared<ListBuilderT>(memory_pool, values_builder)) {}

  Status Append() override { return FromArrowStatus(list_builder_->Append()); }
  Status AppendNull() override {
    return FromArrowStatus(list_builder_->AppendNull());
  }
  Status AppendNulls(int64_t num_nulls) override {
    return FromArrowStatus(list_builder_->AppendNulls(num_nulls));
  }
  Status Finish(std::shared_ptr<arrow::Array>* out) override {
    return FromArrowStatus(list_builder_->Finish(out));
  }
  std::shared_ptr<arrow::ArrayBuilder> wrapped() override {
    return list_builder_;
  }

  // private:
  std::shared_ptr<ListBuilderT> list_builder_;
};

std::unique_ptr<ListBuilderInterface> MakeListBuilderWrapper(
    const bool large_list,
    const std::shared_ptr<arrow::ArrayBuilder>& values_builder,
    arrow::MemoryPool* memory_pool) {
  if (large_list) {
    return absl::make_unique<ListBuilderWrapper<arrow::LargeListBuilder>>(
        values_builder, memory_pool);
  }
  return absl::make_unique<ListBuilderWrapper<arrow::ListBuilder>>(
      values_builder, memory_pool);
}

// BinaryBuilder and LargeBinaryBuilder don't share the same base class. We
// create wrappers of them so the wrappers can share.
// TODO(b/154119411): we should start always producing LargeBinary. Clean this
// up.
class BinaryBuilderInterface {
 public:
  virtual ~BinaryBuilderInterface() = default;
  virtual Status Append(const std::string& str) = 0;
  virtual std::shared_ptr<arrow::ArrayBuilder> wrapped() const = 0;
};

template <typename BinaryBuilderT>
class BinaryBuilderWrapper : public BinaryBuilderInterface {
 public:
  BinaryBuilderWrapper(arrow::MemoryPool* memory_pool)
      : binary_builder_(std::make_shared<BinaryBuilderT>(memory_pool)) {}

  Status Append(const std::string& str) override {
    return FromArrowStatus(binary_builder_->Append(str));
  }

  std::shared_ptr<arrow::ArrayBuilder> wrapped() const override {
    return binary_builder_;
  }

 private:
  std::shared_ptr<BinaryBuilderT> binary_builder_;
};

std::unique_ptr<BinaryBuilderInterface> MakeBinaryBuilderWrapper(
    const bool large_binary, arrow::MemoryPool* memory_pool) {
  if (large_binary) {
    return absl::make_unique<BinaryBuilderWrapper<arrow::LargeBinaryBuilder>>(
        memory_pool);
  }
  return absl::make_unique<BinaryBuilderWrapper<arrow::BinaryBuilder>>(
      memory_pool);
}

Status TfmdFeatureToArrowField(const bool use_large_types,
                               const bool is_sequence_feature,
                               const tensorflow::metadata::v0::Feature& feature,
                               std::shared_ptr<arrow::Field>* out) {
  // Used for disambiguating overloaded functions.
  using ListFactoryType = std::shared_ptr<arrow::DataType> (*)(
      const std::shared_ptr<arrow::DataType>&);

  ListFactoryType list_factory = &arrow::list;
  if (use_large_types) {
    list_factory = &arrow::large_list;
  }

  switch (feature.type()) {
    case tensorflow::metadata::v0::FLOAT: {
      auto type = list_factory(arrow::float32());
      if (is_sequence_feature) {
        type = list_factory(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    case tensorflow::metadata::v0::INT: {
      auto type = list_factory(arrow::int64());
      if (is_sequence_feature) {
        type = list_factory(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    case tensorflow::metadata::v0::BYTES: {
      auto type = use_large_types ? list_factory(arrow::large_binary())
                                  : list_factory(arrow::binary());
      if (is_sequence_feature) {
        type = list_factory(type);
      }
      *out = arrow::field(feature.name(), type);
      break;
    }
    default:
      return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

}  // namespace

class FeatureDecoder {
 public:
  FeatureDecoder(const bool large_list,
                 const std::shared_ptr<arrow::ArrayBuilder>& values_builder)
      : list_builder_(MakeListBuilderWrapper(large_list, values_builder,
                                             arrow::default_memory_pool())),
        feature_was_added_(false) {}
  virtual ~FeatureDecoder() {}

  // Called if the feature is present in the Example.
  Status DecodeFeature(const tensorflow::Feature& feature) {
    if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
      TFX_BSL_RETURN_IF_ERROR(list_builder_->AppendNull());
    } else {
      TFX_BSL_RETURN_IF_ERROR(list_builder_->Append());
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

  Status AppendNull() { return list_builder_->AppendNull(); }

  // Called after a (possible) call to DecodeFeature. If DecodeFeature was
  // called, this will do nothing. Otherwise, it will add to the null count.
  Status FinishFeature() {
    if (!feature_was_added_) {
      TFX_BSL_RETURN_IF_ERROR(list_builder_->AppendNull());
    }
    feature_was_added_ = false;
    return Status::OK();
  }

  Status Finish(std::shared_ptr<arrow::Array>* out) {
    return list_builder_->Finish(out);
  }

 protected:
  virtual Status DecodeFeatureValues(const tensorflow::Feature& feature) = 0;

 private:
  std::unique_ptr<ListBuilderInterface> list_builder_;
  bool feature_was_added_;
};

class FloatDecoder : public FeatureDecoder {
 public:
  static FloatDecoder* Make(const bool large_list) {
    return new FloatDecoder(
        large_list, std::make_shared<arrow::FloatBuilder>(
                        arrow::float32(), arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureValues(const tensorflow::Feature& feature) override {
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
  FloatDecoder(const bool large_list,
               const std::shared_ptr<arrow::FloatBuilder>& values_builder)
      : FeatureDecoder(large_list, values_builder),
        values_builder_(values_builder) {}

  std::shared_ptr<arrow::FloatBuilder> values_builder_;
};

class IntDecoder : public FeatureDecoder {
 public:
  static IntDecoder* Make(const bool large_list) {
    return new IntDecoder(large_list,
                          std::make_shared<arrow::Int64Builder>(
                              arrow::int64(), arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureValues(const tensorflow::Feature& feature) override {
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
  IntDecoder(const bool large_list,
             const std::shared_ptr<arrow::Int64Builder>& values_builder)
      : FeatureDecoder(large_list, values_builder),
        values_builder_(values_builder) {}

  std::shared_ptr<arrow::Int64Builder> values_builder_;
};

class BytesDecoder : public FeatureDecoder {
 public:
  static BytesDecoder* Make(const bool large_list, const bool large_binary) {
    return new BytesDecoder(
        large_list, large_binary,
        MakeBinaryBuilderWrapper(large_binary, arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureValues(const tensorflow::Feature& feature) override {
    if (feature.kind_case() != tensorflow::Feature::kBytesList) {
      return errors::InvalidArgument(
          absl::StrCat("Feature had wrong type, expected bytes_list, found ",
                       KindToStr(feature.kind_case())));
    }
    for (const std::string& value : feature.bytes_list().value()) {
      TFX_BSL_RETURN_IF_ERROR(values_builder_->Append(value));
    }
    return Status::OK();
  }

 private:
  BytesDecoder(const bool large_list, const bool large_binary,
               std::unique_ptr<BinaryBuilderInterface> values_builder)
      : FeatureDecoder(large_list, values_builder->wrapped()),
        values_builder_(std::move(values_builder)) {}

  std::unique_ptr<BinaryBuilderInterface> values_builder_;
};

class FeatureListDecoder {
 public:
  FeatureListDecoder(const bool large_list,
                     const std::shared_ptr<arrow::ArrayBuilder>& values_builder)
      : inner_list_builder_(MakeListBuilderWrapper(
            large_list, values_builder, arrow::default_memory_pool())),
        outer_list_builder_(
            MakeListBuilderWrapper(large_list, inner_list_builder_->wrapped(),
                                   arrow::default_memory_pool())),
        feature_list_was_added_(false) {}
  virtual ~FeatureListDecoder() {}

  // Called if the feature list is present in the SequenceExample.
  Status DecodeFeatureList(const tensorflow::FeatureList& feature_list) {
    if (feature_list.feature().empty()) {
      TFX_BSL_RETURN_IF_ERROR(outer_list_builder_->Append());
    } else {
      TFX_BSL_RETURN_IF_ERROR(outer_list_builder_->Append());
      TFX_BSL_RETURN_IF_ERROR(DecodeFeatureListValues(feature_list));
    }
    if (feature_list_was_added_) {
      return errors::Internal(
          "Internal error: FinishFeatureList() must be called before "
          "DecodeFeatureList() can be called again.");
    }
    feature_list_was_added_ = true;
    return Status::OK();
  }

  Status AppendNull() { return outer_list_builder_->AppendNull(); }

  Status AppendInnerNulls(const int num_nulls) {
    TFX_BSL_RETURN_IF_ERROR(outer_list_builder_->Append());
    TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->AppendNulls(num_nulls));
    return Status::OK();
  }

  // Called after a (possible) call to DecodeFeatureList. If DecodeFeatureList
  // was called this will do nothing.  Otherwise it will add to the null count.
  Status FinishFeatureList() {
    if (!feature_list_was_added_) {
      TFX_BSL_RETURN_IF_ERROR(outer_list_builder_->AppendNull());
    }
    feature_list_was_added_ = false;
    return Status::OK();
  }

  Status Finish(std::shared_ptr<arrow::Array>* out) {
    return outer_list_builder_->Finish(out);
  }

 protected:
  virtual Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) = 0;
  std::unique_ptr<ListBuilderInterface> inner_list_builder_;
  std::unique_ptr<ListBuilderInterface> outer_list_builder_;
  bool feature_list_was_added_;
};

class FloatListDecoder : public FeatureListDecoder {
 public:
  static FloatListDecoder* Make(const bool large_list) {
    return new FloatListDecoder(
        large_list, std::make_shared<arrow::FloatBuilder>(
                        arrow::float32(), arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kFloatList) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->Append());
        for (float value : feature.float_list().value()) {
          TFX_BSL_RETURN_IF_ERROR(
              FromArrowStatus(values_builder_->Append(value)));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->AppendNull());
      } else {
        return errors::InvalidArgument(
            absl::StrCat("Feature had wrong type, expected float_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return Status::OK();
  }

 private:
  FloatListDecoder(const bool large_list,
                   const std::shared_ptr<arrow::FloatBuilder>& values_builder)
      : FeatureListDecoder(large_list, values_builder),
        values_builder_(values_builder) {}

  std::shared_ptr<arrow::FloatBuilder> values_builder_;
};

class IntListDecoder : public FeatureListDecoder {
 public:
  static IntListDecoder* Make(const bool large_list) {
    return new IntListDecoder(
        large_list, std::make_shared<arrow::Int64Builder>(
                        arrow::int64(), arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kInt64List) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->Append());
        for (int value : feature.int64_list().value()) {
          TFX_BSL_RETURN_IF_ERROR(
              FromArrowStatus(values_builder_->Append(value)));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->AppendNull());
      } else {
        return errors::InvalidArgument(
            absl::StrCat("Feature had wrong type, expected int64_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return Status::OK();
  }

 private:
  IntListDecoder(const bool large_list,
                 const std::shared_ptr<arrow::Int64Builder>& values_builder)
      : FeatureListDecoder(large_list, values_builder),
        values_builder_(values_builder) {}

  std::shared_ptr<arrow::Int64Builder> values_builder_;
};

class BytesListDecoder : public FeatureListDecoder {
 public:
  static BytesListDecoder* Make(const bool use_large_list,
                                const bool use_large_binary) {
    return new BytesListDecoder(
        use_large_list, use_large_binary,
        MakeBinaryBuilderWrapper(use_large_binary,
                                 arrow::default_memory_pool()));
  }

 protected:
  Status DecodeFeatureListValues(
      const tensorflow::FeatureList& feature_list) override {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() == tensorflow::Feature::kBytesList) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->Append());
        for (const std::string& value : feature.bytes_list().value()) {
          TFX_BSL_RETURN_IF_ERROR(values_builder_->Append(value));
        }
      } else if (feature.kind_case() == tensorflow::Feature::KIND_NOT_SET) {
        TFX_BSL_RETURN_IF_ERROR(inner_list_builder_->AppendNull());
      } else {
        return errors::InvalidArgument(
            absl::StrCat("Feature had wrong type, expected bytes_list, found ",
                         KindToStr(feature.kind_case())));
      }
    }
    return Status::OK();
  }

 private:
  BytesListDecoder(const bool large_list, const bool large_binary,
                   std::unique_ptr<BinaryBuilderInterface> values_builder)
      : FeatureListDecoder(large_list, values_builder->wrapped()),
        values_builder_(std::move(values_builder)) {}

  std::unique_ptr<BinaryBuilderInterface> values_builder_;
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
  static UnknownTypeFeatureListDecoder* Make(const bool use_large_types) {
    return new UnknownTypeFeatureListDecoder(use_large_types);
  }
  Status DecodeFeatureList(const tensorflow::FeatureList& feature_list) {
    for (const auto& feature : feature_list.feature()) {
      if (feature.kind_case() != tensorflow::Feature::KIND_NOT_SET) {
        return errors::Internal(
            "Attempted to decode a feature list that has a known type with the "
            "UnknownTypeFeatureListDecoder.");
      }
    }
    null_counts_.push_back(feature_list.feature().size());
    feature_list_was_added_ = true;
    return Status::OK();
  }

  void AppendNull() { return null_counts_.push_back(-1); }

  Status ConvertToTypedListDecoder(const tensorflow::Feature::KindCase& type,
                                   FeatureListDecoder** typed_list_decoder) {
    switch (type) {
      case tensorflow::Feature::kInt64List:
        *typed_list_decoder = IntListDecoder::Make(use_large_types_);
        break;
      case tensorflow::Feature::kFloatList:
        *typed_list_decoder = FloatListDecoder::Make(use_large_types_);
        break;
      case tensorflow::Feature::kBytesList:
        *typed_list_decoder =
            BytesListDecoder::Make(use_large_types_, use_large_types_);
        break;
      case tensorflow::Feature::KIND_NOT_SET:
        return errors::Internal(
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
    return Status::OK();
  }

  // Called after a (possible) call to DecodeFeatureList. If DecodeFeatureList
  // was called this will do nothing. Otherwise it will result in a Null being
  // appended to the outer-most list in the resulting arrow array.
  Status FinishFeatureList() {
    if (!feature_list_was_added_) {
      null_counts_.push_back(-1);
    }
    feature_list_was_added_ = false;
    return Status::OK();
  }

  // Becaues the feature type is not known, write the contents out as a
  // list_type(null). Here, null indicates that the inner list is of an unknown
  // type.
  Status Finish(std::shared_ptr<arrow::Array>* out) {
    std::shared_ptr<arrow::NullBuilder> values_builder =
        std::make_shared<arrow::NullBuilder>(arrow::default_memory_pool());
    std::unique_ptr<ListBuilderInterface> list_builder = MakeListBuilderWrapper(
        use_large_types_, values_builder, arrow::default_memory_pool());
    for (int i = 0; i < null_counts_.size(); ++i) {
      if (null_counts_[i] == -1) {
        TFX_BSL_RETURN_IF_ERROR(list_builder->AppendNull());
      } else {
        TFX_BSL_RETURN_IF_ERROR(list_builder->Append());
        TFX_BSL_RETURN_IF_ERROR(
            FromArrowStatus(values_builder->AppendNulls(null_counts_[i])));
      }
    }
    return list_builder->Finish(out);
  }

 private:
  UnknownTypeFeatureListDecoder(const bool use_large_types)
      : use_large_types_(use_large_types) {}
  const bool use_large_types_;
  std::vector<int64_t> null_counts_;
  bool feature_list_was_added_;
};  // namespace tfx_bsl

namespace {
// Decodes all top-level features in a specified features map.
// This will be called on a feature map for an individual Example or
// SequenceExample (context features only) while processing a batch of same.
// It can be used where no schema is available (which requires determining the
// coder type from the example(s) seen).
Status DecodeTopLevelFeatures(
    const google::protobuf::Map<std::string, tensorflow::Feature>& features,
    absl::flat_hash_set<std::string>& all_features_seen,
    const bool use_large_types, const int num_examples_already_processed,
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
          feature_decoder = IntDecoder::Make(use_large_types);
          break;
        case tensorflow::Feature::kFloatList:
          feature_decoder = FloatDecoder::Make(use_large_types);
          break;
        case tensorflow::Feature::kBytesList:
          feature_decoder =
              BytesDecoder::Make(use_large_types, use_large_types);
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

  return Status::OK();
}

// Finishes features specified by all_features_names by converting them to
// Arrow Arrays and Fields using the corresponding decoder (if available). This
// function can be used to finish features in Examples or context features in
// SequenceExamples. If a feature name is provided for which there is no
// corresponding decoder available, it will create a NullArray for that feature.
Status FinishTopLevelFeatures(
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

  return Status::OK();
}

}  // namespace

static Status MakeFeatureDecoder(
    const bool large_list, const tensorflow::metadata::v0::Feature& feature,
    std::unique_ptr<FeatureDecoder>* out) {
  switch (feature.type()) {
    case tensorflow::metadata::v0::FLOAT:
      out->reset(FloatDecoder::Make(large_list));
      break;
    case tensorflow::metadata::v0::INT:
      out->reset(IntDecoder::Make(large_list));
      break;
    case tensorflow::metadata::v0::BYTES:
      out->reset(BytesDecoder::Make(large_list, large_list));
      break;
    default:
      return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

// static
Status ExamplesToRecordBatchDecoder::Make(
    absl::optional<absl::string_view> serialized_schema,
    const bool use_large_types,
    std::unique_ptr<ExamplesToRecordBatchDecoder>* result) {
  if (!serialized_schema) {
    *result = absl::WrapUnique(
        new ExamplesToRecordBatchDecoder(use_large_types, nullptr, nullptr));
    return Status::OK();
  }
  auto feature_decoders = absl::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>();
  auto schema = absl::make_unique<tensorflow::metadata::v0::Schema>();
  if (!schema->ParseFromArray(serialized_schema->data(),
                              serialized_schema->size())) {
    return errors::InvalidArgument("Unable to parse schema.");
  }
  std::vector<std::shared_ptr<arrow::Field>> arrow_schema_fields;
  for (const tensorflow::metadata::v0::Feature& feature : schema->feature()) {
    TFX_BSL_RETURN_IF_ERROR(MakeFeatureDecoder(
        use_large_types, feature, &(*feature_decoders)[feature.name()]));
    arrow_schema_fields.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(
        TfmdFeatureToArrowField(use_large_types, /*is_sequence_feature=*/false,
                                feature, &arrow_schema_fields.back()));
  }
  *result = absl::WrapUnique(new ExamplesToRecordBatchDecoder(
      use_large_types, arrow::schema(std::move(arrow_schema_fields)),
      std::move(feature_decoders)));
  return Status::OK();
}

ExamplesToRecordBatchDecoder::ExamplesToRecordBatchDecoder(
    const bool use_large_types, std::shared_ptr<arrow::Schema> arrow_schema,
    std::unique_ptr<
        const absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
        feature_decoders)
    : arrow_schema_(std::move(arrow_schema)),
      feature_decoders_(std::move(feature_decoders)),
      use_large_types_(use_large_types) {}

ExamplesToRecordBatchDecoder::~ExamplesToRecordBatchDecoder() {}

Status ExamplesToRecordBatchDecoder::DecodeBatch(
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

Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersAvailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_examples.size(); ++i) {
    arena.Reset();
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_examples[i], example));
    for (const auto& p : example->features().feature()) {
      const std::string& feature_name = p.first;
      const tensorflow::Feature& feature = p.second;
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

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  arrays.reserve(arrow_schema_->fields().size());
  for (const std::shared_ptr<arrow::Field>& field : arrow_schema_->fields()) {
    FeatureDecoder& decoder = *feature_decoders_->at(field->name());
    arrays.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(decoder.Finish(&arrays.back()));
  }
  *record_batch = arrow::RecordBatch::Make(arrow_schema_,
                                           serialized_examples.size(), arrays);
  return Status::OK();
}

Status ExamplesToRecordBatchDecoder::DecodeFeatureDecodersUnavailable(
    const std::vector<absl::string_view>& serialized_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>
      feature_decoders;
  // all features which have been observed.  `feature_decoders` will only
  // contain features for which a values list was observed, otherwise the
  // feature type cannot be inferred and so the feature decoder cannot be
  // created.
  absl::flat_hash_set<std::string> all_features;

  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_examples.size(); ++i) {
    arena.Reset();
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseExample(serialized_examples[i], example));
    TFX_BSL_RETURN_IF_ERROR(
        DecodeTopLevelFeatures(example->features().feature(), all_features,
                               use_large_types_, i, feature_decoders));
  }
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  TFX_BSL_RETURN_IF_ERROR(FinishTopLevelFeatures(all_features, feature_decoders,
                                                 serialized_examples.size(),
                                                 &arrays, &fields));

  *record_batch = arrow::RecordBatch::Make(arrow::schema(fields),
                                           serialized_examples.size(), arrays);
  return Status::OK();
}

static Status MakeFeatureListDecoder(
    const bool use_large_types,
    const tensorflow::metadata::v0::Feature& feature,
    std::unique_ptr<FeatureListDecoder>* out) {
  switch (feature.type()) {
    case tensorflow::metadata::v0::FLOAT:
      out->reset(FloatListDecoder::Make(use_large_types));
      break;
    case tensorflow::metadata::v0::INT:
      out->reset(IntListDecoder::Make(use_large_types));
      break;
    case tensorflow::metadata::v0::BYTES:
      out->reset(BytesListDecoder::Make(use_large_types, use_large_types));
      break;
    default:
      return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

Status SequenceExamplesToRecordBatchDecoder::Make(
    const absl::optional<absl::string_view>& serialized_schema,
    const std::string& sequence_feature_column_name, const bool use_large_types,
    std::unique_ptr<SequenceExamplesToRecordBatchDecoder>* result) {
  if (!serialized_schema) {
    *result = absl::WrapUnique(new SequenceExamplesToRecordBatchDecoder(
        sequence_feature_column_name, use_large_types, nullptr, nullptr,
        nullptr, nullptr));
    return Status::OK();
  }
  auto context_feature_decoders = absl::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>();
  auto sequence_feature_decoders = absl::make_unique<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureListDecoder>>>();
  auto schema = absl::make_unique<tensorflow::metadata::v0::Schema>();
  if (!schema->ParseFromArray(serialized_schema->data(),
                              serialized_schema->size())) {
    return errors::InvalidArgument("Unable to parse schema.");
  }
  std::vector<std::shared_ptr<arrow::Field>> arrow_schema_fields;
  auto sequence_feature_schema_fields =
      absl::make_unique<std::vector<std::shared_ptr<arrow::Field>>>();
  for (const tensorflow::metadata::v0::Feature& feature : schema->feature()) {
    if (feature.name() == sequence_feature_column_name) {
      // This feature is a top-level feature containing sequence features, as
      // identified by the sequence_feature_column_name.
      if (feature.type() != tensorflow::metadata::v0::STRUCT) {
        return errors::InvalidArgument(
            "Found a feature in the schema with the "
            "sequence_feature_column_name (i.e., ",
            sequence_feature_column_name,
            ") that is not a struct. The sequence_feature_column_name should "
            "be used only for the top-level struct feature with a struct "
            "domain that contains each sequence feature as a child.");
      }
      for (const auto& child_feature : feature.struct_domain().feature()) {
        TFX_BSL_RETURN_IF_ERROR(MakeFeatureListDecoder(
            use_large_types, child_feature,
            &(*sequence_feature_decoders)[child_feature.name()]));
        sequence_feature_schema_fields->emplace_back();
        TFX_BSL_RETURN_IF_ERROR(TfmdFeatureToArrowField(
            use_large_types, /*is_sequence_feature=*/true, child_feature,
            &sequence_feature_schema_fields->back()));
      }
      continue;
    }
    // If the feature is not the top-level sequence feature, it is a context
    // feature.
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureDecoder(use_large_types, feature,
                           &(*context_feature_decoders)[feature.name()]));
    arrow_schema_fields.emplace_back();
    TFX_BSL_RETURN_IF_ERROR(
        TfmdFeatureToArrowField(use_large_types, /*is_sequence_feature=*/false,
                                feature, &arrow_schema_fields.back()));
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
      sequence_feature_column_name, use_large_types,
      arrow::schema(std::move(arrow_schema_fields)),
      std::move(sequence_features_struct_type),
      std::move(context_feature_decoders),
      std::move(sequence_feature_decoders)));
  return Status::OK();
}

SequenceExamplesToRecordBatchDecoder::SequenceExamplesToRecordBatchDecoder(
    const std::string& sequence_feature_column_name, const bool use_large_types,
    std::shared_ptr<arrow::Schema> arrow_schema,
    std::shared_ptr<arrow::StructType> sequence_features_struct_type,
    std::unique_ptr<
        const absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
        context_feature_decoders,
    std::unique_ptr<const absl::flat_hash_map<
        std::string, std::unique_ptr<FeatureListDecoder>>>
        sequence_feature_decoders)
    : sequence_feature_column_name_(sequence_feature_column_name),
      use_large_types_(use_large_types),
      arrow_schema_(std::move(arrow_schema)),
      sequence_features_struct_type_(std::move(sequence_features_struct_type)),
      context_feature_decoders_(std::move(context_feature_decoders)),
      sequence_feature_decoders_(std::move(sequence_feature_decoders)) {}

SequenceExamplesToRecordBatchDecoder::~SequenceExamplesToRecordBatchDecoder() {}

Status SequenceExamplesToRecordBatchDecoder::DecodeBatch(
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

Status SequenceExamplesToRecordBatchDecoder::DecodeFeatureListDecodersAvailable(
    const std::vector<absl::string_view>& serialized_sequence_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_sequence_examples.size(); ++i) {
    arena.Reset();
    auto* sequence_example =
        google::protobuf::Arena::CreateMessage<tensorflow::SequenceExample>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseSequenceExample(
        serialized_sequence_examples[i], sequence_example));
    for (const auto& p : sequence_example->context().feature()) {
      const std::string& context_feature_name = p.first;
      const tensorflow::Feature& context_feature = p.second;
      const auto it = context_feature_decoders_->find(context_feature_name);
      if (it != context_feature_decoders_->end()) {
        Status status = it->second->DecodeFeature(context_feature);
        if (!status.ok()) {
          return Status(status.code(),
                        absl::StrCat(status.error_message(), " for feature \"",
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
        Status status = it->second->DecodeFeatureList(feature_list);
        if (!status.ok()) {
          return Status(status.code(), absl::StrCat(status.error_message(),
                                                    " for sequence feature \"",
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
           field->type()->children()) {
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
  return Status::OK();
}

Status
SequenceExamplesToRecordBatchDecoder::DecodeFeatureListDecodersUnavailable(
    const std::vector<absl::string_view>& serialized_sequence_examples,
    std::shared_ptr<arrow::RecordBatch>* record_batch) const {
  absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>
      context_feature_decoders;
  std::map<std::string,
           absl::variant<std::unique_ptr<FeatureListDecoder>,
                         std::unique_ptr<UnknownTypeFeatureListDecoder>>>
      sequence_feature_decoders;
  // All context features that have been observed.
  // `context_feature_decoders` will contain only features for which a values
  // list was observed (otherwise the feature type cannot be inferred and so the
  // feature decoder cannot be created).
  absl::flat_hash_set<std::string> all_context_features;
  // If SequenceExamples that include feature_lists have been observed but none
  // of those examples include any sequence feature names, then this decoder
  // will add a sequence feature column to the resulting record batch that
  // contains a StructArray with no child arrays. This tracks whether
  // feature_lists have been observed.
  bool feature_lists_observed = false;

  google::protobuf::Arena arena;
  for (int i = 0; i < serialized_sequence_examples.size(); ++i) {
    arena.Reset();
    auto* sequence_example =
        google::protobuf::Arena::CreateMessage<tensorflow::SequenceExample>(&arena);
    TFX_BSL_RETURN_IF_ERROR(ParseSequenceExample(
        serialized_sequence_examples[i], sequence_example));
    TFX_BSL_RETURN_IF_ERROR(DecodeTopLevelFeatures(
        sequence_example->context().feature(), all_context_features,
        use_large_types_, i, context_feature_decoders));
    if (sequence_example->has_feature_lists()) {
      feature_lists_observed = true;
    }
    for (const auto& p : sequence_example->feature_lists().feature_list()) {
      const std::string& sequence_feature_name = p.first;
      const tensorflow::FeatureList& sequence_feature_list = p.second;
      FeatureListDecoder* sequence_feature_decoder = nullptr;
      UnknownTypeFeatureListDecoder* unknown_type_sequence_feature_decoder =
          nullptr;
      // Determine if there is an existing decoder for this sequence feature.
      const auto it = sequence_feature_decoders.find(sequence_feature_name);
      if (it != sequence_feature_decoders.end()) {
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
                unknown_type_sequence_feature_decoder
                    ->ConvertToTypedListDecoder(feature.kind_case(),
                                                &sequence_feature_decoder));
            sequence_feature_decoders[sequence_feature_name] =
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
              UnknownTypeFeatureListDecoder::Make(use_large_types_);
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
              sequence_feature_decoder = IntListDecoder::Make(use_large_types_);
              break;
            case tensorflow::Feature::kFloatList:
              sequence_feature_decoder =
                  FloatListDecoder::Make(use_large_types_);
              break;
            case tensorflow::Feature::kBytesList:
              sequence_feature_decoder =
                  BytesListDecoder::Make(use_large_types_, use_large_types_);
              break;
            case tensorflow::Feature::KIND_NOT_SET:
              unknown_type_sequence_feature_decoder =
                  UnknownTypeFeatureListDecoder::Make(use_large_types_);
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
          for (int j = 0; j < i; ++j) {
            unknown_type_sequence_feature_decoder->AppendNull();
          }
          sequence_feature_decoders[sequence_feature_name] =
              absl::WrapUnique(unknown_type_sequence_feature_decoder);
        } else if (sequence_feature_decoder) {
          // Similarly handle the situation in which we see a sequence feature
          // of a known type for the first time after already having processed
          // some sequence examples.
          for (int j = 0; j < i; ++j) {
            TFX_BSL_RETURN_IF_ERROR(sequence_feature_decoder->AppendNull());
          }
          sequence_feature_decoders[sequence_feature_name] =
              absl::WrapUnique(sequence_feature_decoder);
        }
      }  // End adding new decoder.
      // Decode the current feature list using the appropriate feature
      // decoder.
      Status status;
      if (sequence_feature_decoder) {
        status =
            sequence_feature_decoder->DecodeFeatureList(sequence_feature_list);
      } else if (unknown_type_sequence_feature_decoder) {
        status = unknown_type_sequence_feature_decoder->DecodeFeatureList(
            sequence_feature_list);
      }
      if (!status.ok()) {
        return Status(status.code(), absl::StrCat(status.error_message(),
                                                  " for sequence feature \"",
                                                  sequence_feature_name, "\""));
      }
    }  // End processing the current feature list.

    // Calling FinishFeatureList ensures that a Null is appended for a given
    // feature if it was not decoded (e.g., because it was not seen) in the
    // current SequenceExample.
    for (const auto& p : sequence_feature_decoders) {
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
  }  // End iterating through all sequence examples.

  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  TFX_BSL_RETURN_IF_ERROR(FinishTopLevelFeatures(
      all_context_features, context_feature_decoders,
      serialized_sequence_examples.size(), &arrays, &fields));

  std::vector<std::shared_ptr<arrow::Array>> sequence_feature_arrays;
  std::vector<std::shared_ptr<arrow::Field>> sequence_feature_fields;
  for (const auto& sequence_feature_decoder : sequence_feature_decoders) {
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
    Status status =
        FromArrowStatus((result_or_sequence_feature_array.status()));
    if (status != Status::OK()) {
      return errors::Internal(
          "Attempt to make struct array from sequence features failed with "
          "status: ",
          status);
    }
    arrays.push_back(result_or_sequence_feature_array.ValueOrDie());
    fields.push_back(
        arrow::field(sequence_feature_column_name_, arrays.back()->type()));
  } else if (feature_lists_observed) {
    // If feature lists but no sequence features have been observed, still
    // add a sequence feature column containing a StructArray, but do not
    // include any child arrays in it.
    arrays.push_back(std::make_shared<arrow::StructArray>(
        std::make_shared<arrow::StructType>(sequence_feature_fields),
        serialized_sequence_examples.size(), sequence_feature_arrays));
    fields.push_back(
        arrow::field(sequence_feature_column_name_, arrays.back()->type()));
  }

  *record_batch = arrow::RecordBatch::Make(
      arrow::schema(fields), serialized_sequence_examples.size(), arrays);
  return Status::OK();
}  // namespace tfx_bsl

// Encoder

namespace {

class FeatureEncoder {
 public:
  FeatureEncoder(const std::shared_ptr<arrow::ListArray>& list_array)
      : list_array_(list_array) {}
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
  virtual void EncodeFeatureValues(int32_t start, int32_t end,
                                   tensorflow::Feature* feature) = 0;

 private:
  std::shared_ptr<arrow::ListArray> list_array_;
};

class FloatEncoder : public FeatureEncoder {
 public:
  FloatEncoder(const std::shared_ptr<arrow::ListArray>& list_array,
               const std::shared_ptr<arrow::FloatArray>& values_array)
      : FeatureEncoder(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int32_t start, int32_t end,
                           tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_float_list()->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<arrow::FloatArray> values_array_;
};

class IntEncoder : public FeatureEncoder {
 public:
  IntEncoder(const std::shared_ptr<arrow::ListArray>& list_array,
             const std::shared_ptr<arrow::Int64Array>& values_array)
      : FeatureEncoder(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int32_t start, int32_t end,
                           tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_int64_list()->add_value(values_array_->Value(offset));
    }
  }

 private:
  std::shared_ptr<arrow::Int64Array> values_array_;
};

class BytesEncoder : public FeatureEncoder {
 public:
  BytesEncoder(const std::shared_ptr<arrow::ListArray>& list_array,
               const std::shared_ptr<arrow::BinaryArray>& values_array)
      : FeatureEncoder(list_array), values_array_(values_array) {}

 protected:
  void EncodeFeatureValues(int32_t start, int32_t end,
                           tensorflow::Feature* feature) override {
    for (int32_t offset = start; offset < end; ++offset) {
      feature->mutable_bytes_list()->add_value(
          values_array_->GetString(offset));
    }
  }

 private:
  std::shared_ptr<arrow::BinaryArray> values_array_;
};

Status MakeFeatureEncoder(const std::shared_ptr<arrow::Array>& array,
                          std::unique_ptr<FeatureEncoder>* out) {
  if (array->type()->id() != arrow::Type::LIST) {
    return errors::InvalidArgument("Expected ListArray");
  }
  std::shared_ptr<arrow::ListArray> list_array =
      std::static_pointer_cast<arrow::ListArray>(array);
  const std::shared_ptr<arrow::Array>& values_array = list_array->values();
  if (values_array->type()->id() == arrow::Type::FLOAT) {
    out->reset(new FloatEncoder(
        list_array, std::static_pointer_cast<arrow::FloatArray>(values_array)));
  } else if (values_array->type()->id() == arrow::Type::INT64) {
    out->reset(new IntEncoder(
        list_array, std::static_pointer_cast<arrow::Int64Array>(values_array)));
  } else if (values_array->type()->id() == arrow::Type::BINARY) {
    out->reset(new BytesEncoder(
        list_array,
        std::static_pointer_cast<arrow::BinaryArray>(values_array)));
  } else {
    return errors::InvalidArgument("Bad field type");
  }
  return Status::OK();
}

}  // namespace

Status RecordBatchToExamples(const arrow::RecordBatch& record_batch,
                             std::vector<std::string>* serialized_examples) {
  std::vector<std::pair<std::string, std::unique_ptr<FeatureEncoder>>>
      feature_encoders;
  feature_encoders.reserve(record_batch.num_columns());
  for (int column_index = 0; column_index < record_batch.num_columns();
       ++column_index) {
    const std::shared_ptr<arrow::Array> array =
        record_batch.column(column_index);
    feature_encoders.emplace_back(
        record_batch.schema()->field(column_index)->name(), nullptr);
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureEncoder(array, &feature_encoders.back().second));
  }

  google::protobuf::Arena arena;
  serialized_examples->resize(record_batch.num_rows());
  for (int example_index = 0; example_index < record_batch.num_rows();
       ++example_index) {
    arena.Reset();
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
