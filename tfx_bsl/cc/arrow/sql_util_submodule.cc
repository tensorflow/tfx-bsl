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
#include "tfx_bsl/cc/arrow/sql_util_submodule.h"

#include "tfx_bsl/cc/arrow/sql_util.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "pybind11/stl.h"

namespace tfx_bsl {
namespace py = ::pybind11;

void DefineSqlUtilSubmodule(py::module arrow_module) {
  auto m = arrow_module.def_submodule("sql_util");
  m.doc() = "Arrow Table SQL utilities.";

  py::class_<RecordBatchSQLSliceQuery>(m, "RecordBatchSQLSliceQuery")
      .def(py::init([](const std::string& sql,
                       std::shared_ptr<arrow::Schema> arrow_schema) {
             std::unique_ptr<RecordBatchSQLSliceQuery> result;
             Status s =
                 RecordBatchSQLSliceQuery::Make(sql, arrow_schema, &result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }),
           py::arg("sql"), py::arg("arrow_schema"))
      .def(
          "Execute",
          [](RecordBatchSQLSliceQuery* slice_query,
             const std::shared_ptr<arrow::RecordBatch>& record_batch) {
            std::vector<
                std::vector<std::vector<std::pair<std::string, std::string>>>>
                result;
            Status s = slice_query->Execute(*record_batch, &result);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
            return result;
          },
          py::call_guard<py::gil_scoped_release>());
}

}  // namespace tfx_bsl
