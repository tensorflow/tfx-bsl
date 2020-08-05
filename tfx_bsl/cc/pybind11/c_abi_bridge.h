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

#ifndef THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_C_ABI_BRIDGE_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_C_ABI_BRIDGE_H_
#include <memory>

#include "arrow/c/abi.h"

namespace arrow {
class Array;
class DataType;
class RecordBatch;
class Schema;
}  // namespace arrow

namespace tfx_bsl {
namespace internal {
class SchemaCAbiBridge {
 public:
  explicit SchemaCAbiBridge(const arrow::Schema& schema);
  explicit SchemaCAbiBridge(const arrow::DataType& data_type);
  SchemaCAbiBridge();

  std::shared_ptr<arrow::DataType> ToDataType();
  std::shared_ptr<arrow::Schema> ToSchema();

  uintptr_t c_schema_as_int();

  ~SchemaCAbiBridge();

  SchemaCAbiBridge(const SchemaCAbiBridge&) = delete;
  SchemaCAbiBridge& operator=(const SchemaCAbiBridge&) = delete;

 private:
  ArrowSchema c_schema_ = {};
};

class ArrayCAbiBridge {
 public:
  explicit ArrayCAbiBridge(const arrow::Array& arr);
  explicit ArrayCAbiBridge(const arrow::RecordBatch& record_batch);
  ArrayCAbiBridge();

  ~ArrayCAbiBridge();
  ArrayCAbiBridge(const ArrayCAbiBridge&) = delete;
  ArrayCAbiBridge& operator=(const ArrayCAbiBridge&) = delete;

  std::shared_ptr<arrow::Array> ToArray();
  std::shared_ptr<arrow::RecordBatch> ToRecordBatch();

  uintptr_t c_array_as_int();
  uintptr_t c_type_as_int();

 private:
  ArrowArray c_array_ = {};
  SchemaCAbiBridge c_type_ = {};
};

}  // namespace internal
}  // namespace tfx_bsl


#endif  // THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_C_ABI_BRIDGE_H_
