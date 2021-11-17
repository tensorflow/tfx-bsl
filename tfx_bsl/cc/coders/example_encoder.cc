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
#include "absl/status/status.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace tfx_bsl {

namespace {

class FeatureEncoderInterface {
 public:
  virtual ~FeatureEncoderInterface() = default;
  virtual absl::Status EncodeFeatures(
      const int64_t index, std::vector<tensorflow::Feature*>& features) = 0;
  virtual absl::Status ValidateNumProducedFeatures(
      const int expected, const std::string& field_name) = 0;

 protected:
  virtual void EncodeFeatureValues(int64_t start, int64_t end,
                                   tensorflow::Feature* feature) = 0;
};

template <typename ListT>
class FeatureEncoder : public FeatureEncoderInterface {
 public:
  virtual ~FeatureEncoder() {}

  // Encodes field's features at the given index.
  absl::Status EncodeFeatures(const int64_t index,
                              std::vector<tensorflow::Feature*>& features) {
    TFX_BSL_RETURN_IF_ERROR(ValidateIndex(index));
    if (IsValueValid(index)) {
      const int64_t start_offset = list_array_->value_offset(index);
      const int64_t end_offset = list_array_->value_offset(index + 1);
      EncodeFeatureValues(start_offset, end_offset, features.front());
    }
    return absl::OkStatus();
  }

  // Checks that the number of allocated features is as expected.
  absl::Status ValidateNumProducedFeatures(const int expected,
                                           const std::string& field_name) {
    if (expected != num_produced_features_) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected to produce ", expected, " features, got ",
                       num_produced_features_,
                       ". You may need to use "
                       "RecordBatchToExamplesEncoder with a schema, or there's "
                       "inconsistency between the schema and RecordBatch."));
    }
    return absl::OkStatus();
  }

 protected:
  FeatureEncoder(const std::shared_ptr<ListT>& list_array)
      : list_array_(list_array) {}

  // Checks the index for out-of-bound.
  absl::Status ValidateIndex(const int64_t index) {
    if (index >= list_array_->length()) {
      return absl::InvalidArgumentError(
          absl::StrCat("out-of-bound example index: ", index, " vs ",
                       list_array_->length()));
    }
    return absl::OkStatus();
  }

  // Checks whether value is present at a given index.
  bool IsValueValid(const int64_t index) { return list_array_->IsValid(index); }

  void set_num_produced_features(const int num_produced_features) {
    num_produced_features_ = num_produced_features;
  }

 private:
  // List that is being encoded.
  std::shared_ptr<ListT> list_array_;
  // Expected number of features that encoding will result into. Defaults to
  // single feature per input field.
  int num_produced_features_ = 1;
};

template <typename ListT>
class FloatEncoder : public FeatureEncoder<ListT> {
 public:
  static absl::Status Make(const std::shared_ptr<ListT>& list_array,
                           const std::shared_ptr<arrow::Array>& values_array,
                           std::unique_ptr<FeatureEncoderInterface>* result) {
    *result = absl::WrapUnique(new FloatEncoder<ListT>(
        list_array, std::static_pointer_cast<arrow::FloatArray>(values_array)));
    return absl::OkStatus();
  }

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) override {
    auto float_list = feature->mutable_float_list();
    for (int64_t offset = start; offset < end; ++offset) {
      float_list->add_value(values_array_->Value(offset));
    }
  }

 private:
  FloatEncoder(const std::shared_ptr<ListT>& list_array,
               const std::shared_ptr<arrow::FloatArray>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

  std::shared_ptr<arrow::FloatArray> values_array_;
};

template <typename ListT>
class IntEncoder : public FeatureEncoder<ListT> {
 public:
  static absl::Status Make(const std::shared_ptr<ListT>& list_array,
                           const std::shared_ptr<arrow::Array>& values_array,
                           std::unique_ptr<FeatureEncoderInterface>* result) {
    *result = absl::WrapUnique(new IntEncoder<ListT>(
        list_array, std::static_pointer_cast<arrow::Int64Array>(values_array)));
    return absl::OkStatus();
  }

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) override {
    auto int64_list = feature->mutable_int64_list();
    for (int64_t offset = start; offset < end; ++offset) {
      int64_list->add_value(values_array_->Value(offset));
    }
  }

 private:
  IntEncoder(const std::shared_ptr<ListT>& list_array,
             const std::shared_ptr<arrow::Int64Array>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

  std::shared_ptr<arrow::Int64Array> values_array_;
};

template <typename ListT, typename BinaryArrayT>
class BytesEncoder : public FeatureEncoder<ListT> {
  using offset_type = typename BinaryArrayT::offset_type;

 public:
  static absl::Status Make(const std::shared_ptr<ListT>& list_array,
                           const std::shared_ptr<arrow::Array>& values_array,
                           std::unique_ptr<FeatureEncoderInterface>* result) {
    *result = absl::WrapUnique(new BytesEncoder<ListT, BinaryArrayT>(
        list_array, std::static_pointer_cast<BinaryArrayT>(values_array)));
    return absl::OkStatus();
  }

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
  BytesEncoder(const std::shared_ptr<ListT>& list_array,
               const std::shared_ptr<BinaryArrayT>& values_array)
      : FeatureEncoder<ListT>(list_array), values_array_(values_array) {}

  std::shared_ptr<BinaryArrayT> values_array_;
};

template <typename ListT>
absl::Status MakeFeatureEncoderHelper(
    const std::shared_ptr<ListT>&, std::unique_ptr<FeatureEncoderInterface>*);

// Substitutes values in `split` by offsets of the given `list` that are located
// at positions with indices of current values in `split`.
absl::Status PropagateListSplit(
    const std::shared_ptr<arrow::LargeListArray>& list,
    std::shared_ptr<arrow::Int64Array>* split) {
  arrow::Int64Builder batch_split_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(batch_split_builder.Reserve((*split)->length())));
  for (int64_t i = 0; i < (*split)->length(); i++) {
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(
        batch_split_builder.Append(list->value_offset((*split)->Value(i)))));
  }
  return FromArrowStatus(batch_split_builder.Finish(split));
}

template <typename ListT>
class LargeListEncoder : public FeatureEncoder<ListT> {
 public:
  static absl::Status Make(const std::shared_ptr<ListT>& list_array,
                           const std::shared_ptr<arrow::Array>& values_array,
                           std::unique_ptr<FeatureEncoderInterface>* result) {
    auto values = values_array;
    auto batch_split =
        std::static_pointer_cast<arrow::Int64Array>(list_array->offsets());
    auto encoder = absl::WrapUnique(new LargeListEncoder<ListT>(list_array));
    TFX_BSL_RETURN_IF_ERROR(
        encoder->FlattenAndMakeRowLengthsEncoders(values, batch_split));
    // Values are now flat and batch split is translated to the last dimension.
    TFX_BSL_RETURN_IF_ERROR(encoder->MakeValuesEncoder(values, batch_split));

    // Update number of produced features: one per each partition + one for
    // values.
    encoder->set_num_produced_features(encoder->component_encoders_.size());
    *result = std::move(encoder);
    return absl::OkStatus();
  }

  absl::Status EncodeFeatures(const int64_t index,
                              std::vector<tensorflow::Feature*>& features) {
    TFX_BSL_RETURN_IF_ERROR(this->ValidateIndex(index));
    if (this->IsValueValid(index)) {
      for (int64_t i = 0; i < features.size(); i++) {
        std::vector<tensorflow::Feature*> features_subset{features[i]};
        TFX_BSL_RETURN_IF_ERROR(
            component_encoders_[i]->EncodeFeatures(index, features_subset));
      }
    }
    return absl::OkStatus();
  }

 protected:
  void EncodeFeatureValues(int64_t start, int64_t end,
                           tensorflow::Feature* feature) {}

 private:
  LargeListEncoder(const std::shared_ptr<ListT>& list_array)
      : FeatureEncoder<ListT>(list_array) {}

  // Flattens `value_array` and creates `IntEncoder` for each row length
  // partition of flattened dimension.
  absl::Status FlattenAndMakeRowLengthsEncoders(
      std::shared_ptr<arrow::Array>& values_array,
      std::shared_ptr<arrow::Int64Array>& batch_split) {
    while (values_array->type()->id() == arrow::Type::LARGE_LIST) {
      std::shared_ptr<arrow::Array> row_lengths;
      TFX_BSL_RETURN_IF_ERROR(GetElementLengths(*values_array, &row_lengths));
      const auto& row_lengths_list_array =
          arrow::LargeListArray::FromArrays(*batch_split, *row_lengths)
              .ValueOrDie();
      component_encoders_.emplace_back();
      TFX_BSL_RETURN_IF_ERROR(IntEncoder<arrow::LargeListArray>::Make(
          row_lengths_list_array,
          std::static_pointer_cast<arrow::Int64Array>(row_lengths),
          &component_encoders_.back()));
      auto values_list =
          std::static_pointer_cast<arrow::LargeListArray>(values_array);
      // Translate batch split to the next dimension and flatten values by 1
      // dimension.
      TFX_BSL_RETURN_IF_ERROR(PropagateListSplit(values_list, &batch_split));
      values_array = values_list->values();
    }
    return absl::OkStatus();
  }

  // Produces encoder for flat values.
  absl::Status MakeValuesEncoder(
      std::shared_ptr<arrow::Array>& flat_values,
      std::shared_ptr<arrow::Int64Array>& batch_split) {
    auto values_list_array =
        arrow::LargeListArray::FromArrays(*batch_split, *flat_values);
    TFX_BSL_RETURN_IF_ERROR(FromArrowStatus(values_list_array.status()));
    component_encoders_.emplace(component_encoders_.begin(), nullptr);
    TFX_BSL_RETURN_IF_ERROR(MakeFeatureEncoderHelper<arrow::LargeListArray>(
        values_list_array.ValueUnsafe(), &component_encoders_.front()));
    return absl::OkStatus();
  }

  std::vector<std::unique_ptr<FeatureEncoderInterface>> component_encoders_;
};

template <typename ListT>
absl::Status MakeFeatureEncoderHelper(
    const std::shared_ptr<ListT>& list_array,
    std::unique_ptr<FeatureEncoderInterface>* out) {
  const std::shared_ptr<arrow::Array>& values_array = list_array->values();
  switch (values_array->type()->id()) {
    case arrow::Type::FLOAT:
      return FloatEncoder<ListT>::Make(list_array, values_array, out);
    case arrow::Type::INT64:
      return IntEncoder<ListT>::Make(list_array, values_array, out);
    case arrow::Type::BINARY:
      return BytesEncoder<ListT, arrow::BinaryArray>::Make(list_array,
                                                           values_array, out);
    case arrow::Type::LARGE_BINARY:
      return BytesEncoder<ListT, arrow::LargeBinaryArray>::Make(
          list_array, values_array, out);
    case arrow::Type::LARGE_LIST:
      return LargeListEncoder<ListT>::Make(list_array, values_array, out);
    default:
      return absl::InvalidArgumentError("Bad field type");
  }
}

absl::Status MakeFeatureEncoder(const std::shared_ptr<arrow::Array>& array,
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
      return absl::InvalidArgumentError("Expected ListArray or LargeListArray");
  }
}

using FeatureNameToColumnsMap =
    std::unordered_map<std::string, std::vector<std::string>>;

// Checks for conflicts in names of produced features.
absl::Status ValidateProducedFeatureNames(
    const std::vector<std::string>& field_names,
    const FeatureNameToColumnsMap& nested_features) {
  // Checking that field names are distinct.
  std::unordered_set<std::string> distinct_field_names;
  for (const auto& name : field_names) {
    if (!distinct_field_names.insert(name).second) {
      return absl::InvalidArgumentError(
          "RecordBatch contains duplicate column names.");
    }
  }
  // Checking that nested feature component names are different from field
  // names.
  for (const auto& nested_feature : nested_features) {
    for (const auto& component_name : nested_feature.second) {
      if (!distinct_field_names.insert(component_name).second) {
        return absl::InvalidArgumentError(
            "RecordBatch contains nested component name conflicts.");
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status RecordBatchToExamples(
    const arrow::RecordBatch& record_batch,
    const FeatureNameToColumnsMap& nested_features,
    std::vector<std::string>* serialized_examples) {
  std::vector<std::pair<std::string, std::unique_ptr<FeatureEncoderInterface>>>
      feature_encoders;
  feature_encoders.reserve(record_batch.num_columns());
  const std::vector<std::string> field_names =
      record_batch.schema()->field_names();
  TFX_BSL_RETURN_IF_ERROR(
      ValidateProducedFeatureNames(field_names, nested_features));

  for (const auto& name : field_names) {
    const std::shared_ptr<arrow::Array> array =
        record_batch.GetColumnByName(name);
    feature_encoders.emplace_back(name, nullptr);
    TFX_BSL_RETURN_IF_ERROR(
        MakeFeatureEncoder(array, &feature_encoders.back().second));
    if (nested_features.find(name) != nested_features.end()) {
      // Check that actual nested depth of a field is consistent with
      // `nested_features`.
      TFX_BSL_RETURN_IF_ERROR(
          feature_encoders.back().second->ValidateNumProducedFeatures(
              nested_features.at(name).size(), name));
    } else {
      // Check that fields that are not nested are mapped into exactly one
      // feature.
      TFX_BSL_RETURN_IF_ERROR(
          feature_encoders.back().second->ValidateNumProducedFeatures(1, name));
    }
  }

  serialized_examples->resize(record_batch.num_rows());
  for (int64_t example_index = 0; example_index < record_batch.num_rows();
       ++example_index) {
    google::protobuf::Arena arena;
    auto* example = google::protobuf::Arena::CreateMessage<tensorflow::Example>(&arena);
    auto* feature_map = example->mutable_features()->mutable_feature();
    for (const auto& p : feature_encoders) {
      std::vector<tensorflow::Feature*> features;
      if (nested_features.find(p.first) != nested_features.end()) {
        for (const auto& component_name : nested_features.at(p.first)) {
          features.push_back(&(*feature_map)[component_name]);
        }
      } else {
        features.push_back(&(*feature_map)[p.first]);
      }
      TFX_BSL_RETURN_IF_ERROR(
          p.second->EncodeFeatures(example_index, features));
    }
    if (!example->SerializeToString(&(*serialized_examples)[example_index])) {
      return absl::DataLossError("Unable to serialize example");
    }
  }
  return absl::OkStatus();
}

}  // namespace tfx_bsl
