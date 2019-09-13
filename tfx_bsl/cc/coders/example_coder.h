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

#include "arrow/record_batch.h"
#include "tensorflow/core/example/example.proto.h"
#include "tensorflow/core/lib/core/errors.h"
#include "third_party/tensorflow_metadata/proto/v0/schema.proto.h"

namespace tensorflow {
namespace tfx_bsl {

// Converts a vector of Example protos to an Arrow RecordBatch.
//
// If a schema is provided then the record batch will contain only the fields
// from the schema, in the same order as the Schema.  The data type of the
// schema to determine the field types, with INT, BYTES and FLOAT fields in the
// schema corresponding to the Arrow data types list[int64], list[binary] and
// list[float32].
//
// If a schema is not provided then the data type will be inferred, and chosen
// from  list[int64], list[binary] and list[float32].  In the case where no data
// type can be inferred the arrow null type will be inferred.
Status ExamplesToRecordBatch(
    const std::vector<::tensorflow::Example>& examples,
    const ::tensorflow::metadata::v0::Schema* schema,
    std::shared_ptr<::arrow::RecordBatch> *record_batch);

// Converts a RecordBatch to a list of examples.
//
// The fields of the RecordBatch must have types list[int64], list[binary] or
// list[float32].
Status RecordBatchToExamples(
    const ::arrow::RecordBatch& record_batch,
    std::vector<::tensorflow::Example>* examples);

}  // namespace tfx_bsl
}  // namespace tensorflow



#endif  // TFX_BSL_CC_CODERS_EXAMPLE_CODER_H_
