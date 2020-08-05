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

#include "tfx_bsl/cc/pybind11/c_abi_bridge.h"

#include "arrow/c/bridge.h"
#include "arrow/c/helpers.h"
#include "arrow/api.h"

namespace tfx_bsl {
namespace internal {
SchemaCAbiBridge::SchemaCAbiBridge(const arrow::Schema& schema) : c_schema_{} {
  auto r = arrow::ExportSchema(schema, &c_schema_);
  if (!r.ok()) {
    throw std::runtime_error(r.ToString());
  }
}

SchemaCAbiBridge::SchemaCAbiBridge(const arrow::DataType& data_type)
    : c_schema_{} {
  auto r = arrow::ExportType(data_type, &c_schema_);
  if (!r.ok()) {
    throw std::runtime_error(r.ToString());
  }
}

SchemaCAbiBridge::SchemaCAbiBridge() {}

std::shared_ptr<arrow::DataType> SchemaCAbiBridge::ToDataType() {
  auto r = arrow::ImportType(&c_schema_);
  if (!r.ok()) {
    throw std::runtime_error(r.status().ToString());
  }
  return r.ValueOrDie();
}

std::shared_ptr<arrow::Schema> SchemaCAbiBridge::ToSchema() {
  auto r = arrow::ImportSchema(&c_schema_);
  if (!r.ok()) {
    throw std::runtime_error(r.status().ToString());
  }
  return r.ValueOrDie();
}

uintptr_t SchemaCAbiBridge::c_schema_as_int() {
  return reinterpret_cast<uintptr_t>(&c_schema_);
}

SchemaCAbiBridge::~SchemaCAbiBridge() { ArrowSchemaRelease(&c_schema_); }


ArrayCAbiBridge::ArrayCAbiBridge(const arrow::Array& arr)
    : c_array_{}, c_type_(*arr.type()) {
  // type is exported in the initializer list.
  auto r = arrow::ExportArray(arr, &c_array_, /*out_schema=*/nullptr);
  if (!r.ok()) {
    throw std::runtime_error(r.ToString());
  }
}

ArrayCAbiBridge::ArrayCAbiBridge(const arrow::RecordBatch& record_batch)
    : c_array_{}, c_type_(*record_batch.schema()) {
  // type is exported in the initializer list.
  auto r = arrow::ExportRecordBatch(
      record_batch, &c_array_, /*out_schema=*/nullptr);
  if (!r.ok()) {
    throw std::runtime_error(r.ToString());
  }
}

ArrayCAbiBridge::ArrayCAbiBridge() {}

ArrayCAbiBridge::~ArrayCAbiBridge() { ArrowArrayRelease(&c_array_); }

std::shared_ptr<arrow::Array> ArrayCAbiBridge::ToArray() {
  auto r = arrow::ImportArray(&c_array_, c_type_.ToDataType());
  if (!r.ok()) {
    throw std::runtime_error(r.status().ToString());
  }
  return r.ValueOrDie();
}

std::shared_ptr<arrow::RecordBatch> ArrayCAbiBridge::ToRecordBatch() {
  auto r = arrow::ImportRecordBatch(&c_array_, c_type_.ToSchema());
  if (!r.ok()) {
    throw std::runtime_error(r.status().ToString());
  }
  return r.ValueOrDie();
}

uintptr_t ArrayCAbiBridge::c_array_as_int() {
  return reinterpret_cast<uintptr_t>(&c_array_);
}
uintptr_t ArrayCAbiBridge::c_type_as_int() {
  return c_type_.c_schema_as_int();
}

}  // namespace internal
}  // namespace tfx_bsl
