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

// pybind11 custom casters that bridge pyarrow and C++ Arrow using Arrow's
// Python-C API. Note that to use this bridge, pyarrow must be ABI compaible
// with tfx_bsl. See arrow_casters_c_abi.h for an alternative that does not
// assume ABI compatiblity.
#ifndef THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_PY_C_API_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_PY_C_API_H_

#include <memory>

#include "arrow/python/pyarrow.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "arrow/api.h"

namespace pybind11 {
namespace detail {
template <>
struct type_caster<std::shared_ptr<arrow::Array>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Array>,
                       _<std::shared_ptr<arrow::Array>>());

  bool load(handle src, bool unused_implicit_conversion) {
    auto arrow_result = arrow::py::unwrap_array(src.ptr());
    if (!arrow_result.ok()) return false;
    value = std::move(arrow_result).ValueOrDie();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::Array>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    return arrow::py::wrap_array(src);
  }
};

template <>
struct type_caster<std::shared_ptr<arrow::Schema>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Schema>,
                       _<std::shared_ptr<arrow::Schema>>());

  bool load(handle src, bool unused_implicit_conversion) {
    auto arrow_result = arrow::py::unwrap_schema(src.ptr());
    if (!arrow_result.ok()) return false;
    value = std::move(arrow_result).ValueOrDie();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::Schema>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    return arrow::py::wrap_schema(src);
  }
};

template <>
struct type_caster<std::shared_ptr<arrow::RecordBatch>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::RecordBatch>,
                       _<std::shared_ptr<arrow::RecordBatch>>());

  bool load(handle src, bool unused_implicit_conversion) {
    auto arrow_result = arrow::py::unwrap_batch(src.ptr());
    if (!arrow_result.ok()) return false;
    value = std::move(arrow_result).ValueOrDie();
    return true;
  }

  static handle cast(const std::shared_ptr<arrow::RecordBatch>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    return arrow::py::wrap_batch(src);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TFX_BSL_CC_PYBIND11_ARROW_CASTERS_PY_C_API_H_
