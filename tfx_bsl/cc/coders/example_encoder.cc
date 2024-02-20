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
#include <iterator>
#include <memory>
#include <vector>

#include <google/protobuf/arena.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/coders/example_coder.h"
#include "tfx_bsl/cc/util/status_util.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow_metadata/proto/v0/path.pb.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace tfx_bsl {

namespace {

class FeatureEncoderInterface {
 public:
  virtual ~FeatureEncoderInterface() = default;
  virtual absl::Status EncodeFeatures(
      int64_t index, std::vector<tensorflow::Feature*>& features) = 0;

 protected:
  virtual void EncodeFeatureValues(int64_t start, int64_t end,
                                   tensorflow::Feature* feature) = 0;
};

absl::Status DivideArray(const std::shared_ptr<arrow::Int64Array>& array,
                         int64_t divisor,
                         std::shared_ptr<arrow::Int64Array>* result) {
  if (divisor == 1) {
    *result = std::move(array);
    return absl::OkStatus();
  }
  auto builder = std::make_unique<arrow::Int64Builder>(
      arrow::int64(), arrow::default_memory_pool());
  TFX_BSL_RETURN_IF_ERROR_ARROW(builder->Reserve(array->length()));
  for (int64_t i = 0; i < array->length(); ++i) {
    TFX_BSL_RETURN_IF_ERROR_ARROW(builder->Append(array->Value(i) / divisor));
  }
  return FromArrowStatus(builder->Finish(result));
}

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

 private:
  // List that is being encoded.
  std::shared_ptr<ListT> list_array_;
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
    const std::shared_ptr<ListT>&, const std::vector<int64_t>&,
    std::unique_ptr<FeatureEncoderInterface>*);

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
                           const std::vector<int64_t>& row_length_adjustments,
                           std::unique_ptr<FeatureEncoderInterface>* result) {
    auto values = values_array;
    auto batch_split =
        std::static_pointer_cast<arrow::Int64Array>(list_array->offsets());
    auto encoder = absl::WrapUnique(new LargeListEncoder<ListT>(list_array));
    TFX_BSL_RETURN_IF_ERROR(encoder->FlattenAndMakeRowLengthsEncoders(
        values, batch_split, row_length_adjustments));
    // Values are now flat and batch split is translated to the last dimension.
    TFX_BSL_RETURN_IF_ERROR(encoder->MakeValuesEncoder(values, batch_split));
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
      std::shared_ptr<arrow::Int64Array>& batch_split,
      const std::vector<int64_t>& row_length_adjustments) {
    auto row_length_adjustment = row_length_adjustments.begin();
    while (values_array->type()->id() == arrow::Type::LARGE_LIST) {
      if (row_length_adjustment == row_length_adjustments.end()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Nested depth of large_list is larger than the number of "
            "provided partitions (or no partitions were provided). Expected ",
            row_length_adjustments.size(),
            " partitions. If the column represents RaggedTensor, then create "
            "the encoder with TFMD schema containing RaggedTensor "
            "TensorRepresentations with partitions."));
      }
      std::shared_ptr<arrow::Array> row_lengths;
      TFX_BSL_RETURN_IF_ERROR(GetElementLengths(*values_array, &row_lengths));
      auto int64_row_lengths =
          std::static_pointer_cast<arrow::Int64Array>(row_lengths);
      TFX_BSL_RETURN_IF_ERROR(DivideArray(
          int64_row_lengths, *row_length_adjustment++, &int64_row_lengths));
      const auto& row_lengths_list_array =
          arrow::LargeListArray::FromArrays(*batch_split, *int64_row_lengths)
              .ValueOrDie();
      component_encoders_.emplace_back();
      TFX_BSL_RETURN_IF_ERROR(IntEncoder<arrow::LargeListArray>::Make(
          row_lengths_list_array, int64_row_lengths,
          &component_encoders_.back()));
      auto values_list =
          std::static_pointer_cast<arrow::LargeListArray>(values_array);
      // Translate batch split to the next dimension and flatten values by 1
      // dimension.
      TFX_BSL_RETURN_IF_ERROR(PropagateListSplit(values_list, &batch_split));
      values_array = values_list->values();
    }
    if (row_length_adjustment != row_length_adjustments.end()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Nested depth of large_list is smaller than the number of partitions "
          "in the provided TensorRepresentation. Expected to have ",
          row_length_adjustments.size(), " partitions."));
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
        values_list_array.ValueUnsafe(), {}, &component_encoders_.front()));
    return absl::OkStatus();
  }

  std::vector<std::unique_ptr<FeatureEncoderInterface>> component_encoders_;
};

template <typename ListT>
absl::Status MakeFeatureEncoderHelper(
    const std::shared_ptr<ListT>& list_array,
    const std::vector<int64_t>& row_length_adjustments,
    std::unique_ptr<FeatureEncoderInterface>* out) {
  const std::shared_ptr<arrow::Array>& values_array = list_array->values();
  if (values_array->type()->id() != arrow::Type::LARGE_LIST &&
      !row_length_adjustments.empty()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected ", row_length_adjustments.size(),
        " partitions, but got flat values array. You may need to adjust "
        "RaggedTensor partitions in the provided schema."));
  }
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
      return LargeListEncoder<ListT>::Make(list_array, values_array,
                                           row_length_adjustments, out);
    default:
      return absl::InvalidArgumentError("Bad field type");
  }
}

absl::Status MakeFeatureEncoder(
    const std::shared_ptr<arrow::Array>& array,
    const std::vector<int64_t>& row_length_adjustments,
    std::unique_ptr<FeatureEncoderInterface>* out) {
  switch (array->type()->id()) {
    case arrow::Type::LIST: {
      std::shared_ptr<arrow::ListArray> list_array =
          std::static_pointer_cast<arrow::ListArray>(array);
      return MakeFeatureEncoderHelper<arrow::ListArray>(
          list_array, row_length_adjustments, out);
    }
    case arrow::Type::LARGE_LIST: {
      std::shared_ptr<arrow::LargeListArray> large_list_array =
          std::static_pointer_cast<arrow::LargeListArray>(array);
      return MakeFeatureEncoderHelper<arrow::LargeListArray>(
          large_list_array, row_length_adjustments, out);
    }
    default:
      return absl::InvalidArgumentError("Expected ListArray or LargeListArray");
  }
}

// Checks for conflicts in names of produced features.
absl::Status ValidateProducedFeatureNames(
    const std::vector<std::string>& field_names,
    const absl::flat_hash_map<std::string, RaggedTensorSpec>& ragged_specs) {
  // Check that RecordBatch field names are distinct.
  std::unordered_set<std::string> distinct_field_names;
  for (const auto& name : field_names) {
    if (!distinct_field_names.insert(name).second) {
      return absl::InvalidArgumentError(
          absl::StrCat("RecordBatch contains duplicate column names: ", name));
    }
  }
  // Check for conflicts between ragged source column names.
  std::unordered_set<std::string> distinct_ragged_source_column_names;
  for (const auto& spec : ragged_specs) {
    for (const auto& column_name : spec.second.source_feature_names) {
      if (!distinct_ragged_source_column_names.insert(column_name).second) {
        return absl::InvalidArgumentError(
            absl::StrCat("Source column \"", column_name,
                         "\" of ragged feature \"", spec.first,
                         "\" conflicts with another source column in the "
                         "same batch."));
      }
    }
  }
  return absl::OkStatus();
}

using NamedFeatureEncoder =
    std::pair<std::string, std::unique_ptr<FeatureEncoderInterface>>;

absl::Status MakeFeatureEncoders(
    const std::shared_ptr<arrow::RecordBatch>& record_batch,
    const absl::flat_hash_map<std::string, RaggedTensorSpec>& ragged_specs,
    std::vector<NamedFeatureEncoder>* result) {
  result->reserve(record_batch->num_columns());
  const std::vector<std::string>& field_names =
      record_batch->schema()->field_names();
  TFX_BSL_RETURN_IF_ERROR(
      ValidateProducedFeatureNames(field_names, ragged_specs));
  for (const auto& name : field_names) {
    const std::shared_ptr<arrow::Array>& array =
        record_batch->GetColumnByName(name);
    result->emplace_back(name, nullptr);
    const auto& ragged_partition = ragged_specs.find(name);
    const auto& row_length_adjustments =
        ragged_partition == ragged_specs.end()
            ? std::vector<int64_t>{}
            : ragged_partition->second.row_length_adjustments;
    absl::Status status = MakeFeatureEncoder(array, row_length_adjustments,
                                             &result->back().second);
    if (!status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Error encoding feature \"", name, "\": ", status.message()));
    }
  }
  return absl::OkStatus();
}

absl::Status GetRaggedSpecFromRepresentation(
    const tensorflow::metadata::v0::TensorRepresentation::RaggedTensor&
        representation,
    RaggedTensorSpec* result) {
  if (representation.feature_path().step_size() != 1) {
    return absl::InvalidArgumentError(
        "RaggedTensors with nested or empty value column paths are not "
        "supported.");
  }
  std::vector<std::string> feature_names{representation.feature_path().step(0)};
  std::vector<std::string> reversed_row_lengths_columns;
  reversed_row_lengths_columns.reserve(representation.partition_size());
  std::vector<int64_t> reversed_row_length_adjustments;
  reversed_row_length_adjustments.reserve(representation.partition_size());
  // Row lengths partitions will have to be adjusted by the product
  // of the intermediate uniform partitions since they are not
  // representing a nested level.
  int64_t current_adjustment = 1;
  for (auto partition = representation.partition().rbegin();
       partition != representation.partition().rend(); ++partition) {
    if (partition->has_row_length()) {
      reversed_row_lengths_columns.push_back(partition->row_length());
      reversed_row_length_adjustments.push_back(current_adjustment);
      current_adjustment = 1;
    } else {
      current_adjustment *= partition->uniform_row_length();
    }
  }
  feature_names.insert(
      feature_names.end(),
      std::make_move_iterator(reversed_row_lengths_columns.rbegin()),
      std::make_move_iterator(reversed_row_lengths_columns.rend()));
  *result = {
      std::move(feature_names),
      std::vector<int64_t>{
          std::make_move_iterator(reversed_row_length_adjustments.rbegin()),
          std::make_move_iterator(reversed_row_length_adjustments.rend())}};
  return absl::OkStatus();
}

}  // namespace

RecordBatchToExamplesEncoder::~RecordBatchToExamplesEncoder() {}

absl::Status RecordBatchToExamplesEncoder::Make(
    absl::optional<absl::string_view> serialized_schema,
    std::unique_ptr<RecordBatchToExamplesEncoder>* result) {
  absl::flat_hash_map<std::string, RaggedTensorSpec> ragged_specs;
  if (serialized_schema.has_value()) {
    auto schema = std::make_unique<tensorflow::metadata::v0::Schema>();
    if (!schema->ParseFromArray(serialized_schema->data(),
                                serialized_schema->size())) {
      return absl::InvalidArgumentError("Unable to parse TFMD schema.");
    }
    for (const auto& group : schema->tensor_representation_group()) {
      for (const auto& representation : group.second.tensor_representation()) {
        if (representation.second.has_ragged_tensor()) {
          TFX_BSL_RETURN_IF_ERROR(GetRaggedSpecFromRepresentation(
              representation.second.ragged_tensor(),
              &ragged_specs[representation.first]));
        }
      }
    }
  }
  *result = absl::WrapUnique(
      new RecordBatchToExamplesEncoder(std::move(ragged_specs)));
  return absl::OkStatus();
}

absl::Status RecordBatchToExamplesEncoder::Encode(
    const std::shared_ptr<arrow::RecordBatch>& record_batch,
    std::vector<std::string>* result) const {
  std::vector<NamedFeatureEncoder> encoders;
  TFX_BSL_RETURN_IF_ERROR(
      MakeFeatureEncoders(record_batch, ragged_specs_, &encoders));
  result->resize(record_batch->num_rows());
  for (int64_t example_index = 0; example_index < record_batch->num_rows();
       ++example_index) {
    google::protobuf::Arena arena;
    auto* example = google::protobuf::Arena::Create<tensorflow::Example>(&arena);
    auto* feature_map = example->mutable_features()->mutable_feature();
    for (const auto& encoder : encoders) {
      std::vector<tensorflow::Feature*> features;
      auto ragged_partition = ragged_specs_.find(encoder.first);
      if (ragged_partition != ragged_specs_.end()) {
        for (const auto& component_name :
             ragged_partition->second.source_feature_names) {
          features.push_back(&(*feature_map)[component_name]);
        }
      } else {
        features.push_back(&(*feature_map)[encoder.first]);
      }
      TFX_BSL_RETURN_IF_ERROR(
          encoder.second->EncodeFeatures(example_index, features));
    }
    if (!example->SerializeToString(&(*result)[example_index])) {
      return absl::DataLossError("Unable to serialize example");
    }
  }
  return absl::OkStatus();
}

}  // namespace tfx_bsl
