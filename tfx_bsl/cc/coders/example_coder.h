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
#ifndef TFX_BSL_CC_CODERS_EXAMPLE_CODER_H_
#define TFX_BSL_CC_CODERS_EXAMPLE_CODER_H_

#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tfx_bsl/cc/util/status.h"
#include "tensorflow_metadata/proto/v0/schema.pb.h"

namespace arrow {
class RecordBatch;
}  // namespace arrow

namespace tfx_bsl {
// ExamplesToRecordBatchDecoder converts a vector of Example protos to an Arrow
// RecordBatch.
//
// If a schema is provided then the record batch will contain only the fields
// from the schema, in the same order as the Schema.  The data type of the
// schema to determine the field types, with INT, BYTES and FLOAT fields in the
// schema corresponding to the Arrow data types list_type[int64],
// list_type[binary_type] and list_type[float32]. Where 'list' could be list or
// large_list and 'binary_type' could be binary or large_binary, depending
// on `use_large_types`.
//
// If a schema is not provided then the data type will be inferred, and chosen
// from list_type[int64], list_type[binary_type] and list_type[float32].  In the
// case where no data type can be inferred the arrow null type will be inferred.

class FeatureDecoder;
class ExamplesToRecordBatchDecoder {
 public:
  static Status Make(
      absl::optional<absl::string_view> serialized_schema,
      bool use_large_types,
      std::unique_ptr<ExamplesToRecordBatchDecoder>* result);
  ~ExamplesToRecordBatchDecoder();

  ExamplesToRecordBatchDecoder(const ExamplesToRecordBatchDecoder&) = delete;
  ExamplesToRecordBatchDecoder& operator=(const ExamplesToRecordBatchDecoder&) =
      delete;

  Status DecodeBatch(const std::vector<absl::string_view>& serialized_examples,
                     std::shared_ptr<::arrow::RecordBatch>* record_batch) const;

 private:
  ExamplesToRecordBatchDecoder(
      bool use_large_types,
      std::unique_ptr<const ::tensorflow::metadata::v0::Schema> schema,
      std::unique_ptr<
          absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
          feature_decoders);
  Status DecodeFeatureDecodersAvailable(
      const std::vector<absl::string_view>& serialized_examples,
      std::shared_ptr<::arrow::RecordBatch>* record_batch) const;
  Status DecodeFeatureDecodersUnavailable(
      const std::vector<absl::string_view>& serialized_examples,
      std::shared_ptr<::arrow::RecordBatch>* record_batch) const;

 private:
  const std::unique_ptr<const ::tensorflow::metadata::v0::Schema> schema_;
  const std::unique_ptr<
      absl::flat_hash_map<std::string, std::unique_ptr<FeatureDecoder>>>
      feature_decoders_;
  const bool use_large_types_;
};

// Converts a RecordBatch to a list of examples.
//
// The fields of the RecordBatch must have types list[int64], list[binary] or
// list[float32].
Status RecordBatchToExamples(
    const ::arrow::RecordBatch& record_batch,
    std::vector<std::string>* serialized_examples);

}  // namespace tfx_bsl

#endif  // TFX_BSL_CC_CODERS_EXAMPLE_CODER_H_
