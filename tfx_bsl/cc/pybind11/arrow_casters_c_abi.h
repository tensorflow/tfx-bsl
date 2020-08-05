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

// pybind11 custom casters that bridge pyarrow and C++ Arrow using Arrow's
// C-ABI bridge.
#ifndef THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_C_ABI_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_C_ABI_H_

#include <memory>
#include <stdexcept>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "absl/strings/str_cat.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/pybind11/c_abi_bridge.h"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<std::shared_ptr<arrow::Array>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Array>,
                       _<std::shared_ptr<arrow::Array>>());

  bool load(handle src, bool unused_implicit_conversion) {
    tfx_bsl::internal::ArrayCAbiBridge w;
    try {
      src.attr("_export_to_c")((w.c_array_as_int()), (w.c_type_as_int()));
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call Array._export_to_c() -- check your "
                       "pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    value = w.ToArray();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::Array>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    tfx_bsl::internal::ArrayCAbiBridge w(*src);
    auto pyarrow_module = module::import("pyarrow");
    object py_arr;
    try {
      py_arr = pyarrow_module.attr("Array").attr("_import_from_c")(
          w.c_array_as_int(), w.c_type_as_int());
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call Array._import_from_c() -- check your "
                       "pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    return py_arr.release();
  }
};

template <>
struct type_caster<std::shared_ptr<arrow::Schema>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Schema>,
                       _<std::shared_ptr<arrow::Schema>>());

  bool load(handle src, bool unused_implicit_conversion) {
    tfx_bsl::internal::SchemaCAbiBridge w;
    try {
      src.attr("_export_to_c")(w.c_schema_as_int());
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call Schema._export_to_c() -- check your "
                       "pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    value = w.ToSchema();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::Schema>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    tfx_bsl::internal::SchemaCAbiBridge w(*src);
    auto pyarrow_module = module::import("pyarrow");
    object py_schema;
    try {
      py_schema = pyarrow_module.attr("Schema").attr("_import_from_c")(
          w.c_schema_as_int());
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call Schema._import_from_c() -- check your "
                       "pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    return py_schema.release();
  }
};

template <>
struct type_caster<std::shared_ptr<arrow::RecordBatch>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::RecordBatch>,
                       _<std::shared_ptr<arrow::RecordBatch>>());

  bool load(handle src, bool unused_implicit_conversion) {
    tfx_bsl::internal::ArrayCAbiBridge w;
    // ARROW-9098
    if (src.attr("num_columns").cast<int64_t>() == 0) {
      value = arrow::RecordBatch::Make(
          arrow::schema({}), src.attr("num_rows").cast<int64_t>(),
          std::vector<std::shared_ptr<arrow::Array>>{});
      return true;
    }
    try {
      src.attr("_export_to_c")((w.c_array_as_int()), (w.c_type_as_int()));
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call RecordBatch._export_to_c() -- check "
                       "your pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    value = w.ToRecordBatch();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::RecordBatch>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    tfx_bsl::internal::ArrayCAbiBridge w(*src);
    auto pyarrow_module = module::import("pyarrow");
    object py_record_batch;
    try {
      py_record_batch =
          pyarrow_module.attr("RecordBatch")
              .attr("_import_from_c")(w.c_array_as_int(), w.c_type_as_int());
    } catch (const error_already_set& e) {
      throw std::runtime_error(
          absl::StrCat("Unable to call RecordBatch._import_from_c() -- check "
                       "your pyarrow version (should be >= 0.17): ",
                       e.what()));
    }
    return py_record_batch.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_C_ABI_H_
