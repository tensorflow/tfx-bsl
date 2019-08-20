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
#include "tfx_bsl/cc/arrow/arrow_submodule.h"

#include "arrow/python/pyarrow.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tfx_bsl {
namespace {
void DefineArrayUtilSubmodule(pybind11::module arrow_module) {
  auto m = arrow_module.def_submodule("array_util");
  m.doc() = "Arrow Array utilities.";
  m.def(
      "ListLengthsFromListArray",
      [](const std::shared_ptr<arrow::Array>& array)
          -> std::shared_ptr<arrow::Array> {
        std::shared_ptr<arrow::Array> result;
        Status s = ListLengthsFromListArray(*array, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      "(Get lengths of lists in `list_array` in an int32 array. "
      "Note that null and empty list both are of length 0 and the returned "
      "array does not have any null element. \n"
      "For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2].");
}

}  // namespace

void DefineArrowSubmodule(pybind11::module main_module) {
  arrow::py::import_pyarrow();
  auto m = main_module.def_submodule("arrow");
  m.doc() = "Arrow utilities.";
  DefineArrayUtilSubmodule(m);
}

}  // namespace tfx_bsl
}  // namespace tensorflow
