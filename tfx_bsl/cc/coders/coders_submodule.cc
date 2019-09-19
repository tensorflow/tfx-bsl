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
#include "tfx_bsl/cc/coders/coders_submodule.h"

#include "arrow/python/pyarrow.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/coders/example_coder.h"
#include "tfx_bsl/cc/pybind11/absl_casters.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tfx_bsl/cc/util/status.h"
#include "include/pybind11/stl.h"

namespace tfx_bsl {
void DefineCodersSubmodule(pybind11::module main_module) {
  arrow::py::import_pyarrow();
  auto m = main_module.def_submodule("coders");
  m.doc() = "Coders.";
  m.def("ExamplesToRecordBatch",
        [](const std::vector<absl::string_view>& serialized_examples)
            -> std::shared_ptr<arrow::RecordBatch> {
          std::shared_ptr<arrow::RecordBatch> result;
          Status s = ExamplesToRecordBatch(serialized_examples,
                                           /*schema=*/nullptr, &result);
          if (!s.ok()) {
            throw std::runtime_error(s.ToString());
          }
          return result;
        });
}

}  // namespace tfx_bsl
