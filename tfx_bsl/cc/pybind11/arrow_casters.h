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

// pybind11 custom casters that bridges pyarrow and C++ Arrow.
#ifndef TFX_BSL_PYBIND11_ARROW_CASTERS_H_
#define TFX_BSL_PYBIND11_ARROW_CASTERS_H_

#include <memory>

#include "arrow/python/pyarrow.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "arrow/api.h"

namespace tfx_bsl {
namespace internal {
// TODO(zhuo): remove this shim once tfx-bsl builds exclusively against
// arrow 0.17+.
# if (ARROW_VERSION_MAJOR <= 0) && (ARROW_VERSION_MINOR < 17)
inline PyObject* WrapRecordBatch(
    const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  return arrow::py::wrap_record_batch(record_batch);
}

inline arrow::Status UnwrapRecordBatch(
    PyObject* py_record_batch, std::shared_ptr<arrow::RecordBatch>*
    unwrapped) {
  return arrow::py::unwrap_record_batch(py_record_batch, unwrapped);
}

# else

inline PyObject* WrapRecordBatch(
    const std::shared_ptr<arrow::RecordBatch>& record_batch) {
  return arrow::py::wrap_batch(record_batch);
}

inline arrow::Status UnwrapRecordBatch(
    PyObject* py_record_batch, std::shared_ptr<arrow::RecordBatch>* unwrapped) {
  auto arrow_result = arrow::py::unwrap_batch(py_record_batch);
  ARROW_RETURN_NOT_OK(arrow_result);
  *unwrapped = std::move(arrow_result).ValueOrDie();
  return arrow::Status::OK();
}

# endif
}  // namespace internal
}  // namespace tfx_bsl

namespace pybind11 {
namespace detail {
template <>
struct type_caster<std::shared_ptr<arrow::Array>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Array>,
                       _<std::shared_ptr<arrow::Array>>());

  bool load(handle src, bool unused_implicit_conversion) {
    return arrow::py::unwrap_array(src.ptr(), &value).ok();
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
    return arrow::py::unwrap_schema(src.ptr(), &value).ok();
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
    return tfx_bsl::internal::UnwrapRecordBatch(src.ptr(), &value).ok();
  }

  static handle cast(const std::shared_ptr<arrow::RecordBatch>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    return tfx_bsl::internal::WrapRecordBatch(src);
  }
};

template <>
struct type_caster<std::shared_ptr<arrow::Table>> {
 public:
  PYBIND11_TYPE_CASTER(std::shared_ptr<arrow::Table>,
                       _<std::shared_ptr<arrow::Table>>());

  bool load(handle src, bool unused_implicit_conversion) {
    return arrow::py::unwrap_table(src.ptr(), &value).ok();
  }

  static handle cast(const std::shared_ptr<arrow::Table>& src,
                     return_value_policy unused_return_value_policy,
                     handle unused_handle) {
    return arrow::py::wrap_table(src);
  }
};

}  // namespace detail
}  // namespace pybind11
#endif  // TFX_BSL_PYBIND11_ARROW_CASTERS_H_
