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

#include <memory>

#include "arrow/python/pyarrow.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/coders/example_coder.h"
#include "tfx_bsl/cc/coders/example_numpy_decoder.h"
#include "tfx_bsl/cc/pybind11/absl_casters.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tfx_bsl/cc/util/status.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"

namespace tfx_bsl {
namespace py = pybind11;

void DefineCodersSubmodule(py::module main_module) {
  arrow::py::import_pyarrow();
  auto m = main_module.def_submodule("coders");
  m.doc() = "Coders.";

  py::class_<ExamplesToRecordBatchDecoder>(m, "ExamplesToRecordBatchDecoder")
      .def(py::init([](absl::string_view serialized_schema) {
             std::unique_ptr<ExamplesToRecordBatchDecoder> result;
             Status s =
                 ExamplesToRecordBatchDecoder::Make(serialized_schema, &result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }),
           py::call_guard<py::gil_scoped_release>())
      .def(py::init([] {
             std::unique_ptr<ExamplesToRecordBatchDecoder> result;
             Status s =
                 ExamplesToRecordBatchDecoder::Make(absl::nullopt, &result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "DecodeBatch",
          [](ExamplesToRecordBatchDecoder* decoder,
             const std::vector<absl::string_view>& serialized_examples) {
            std::shared_ptr<arrow::RecordBatch> result;
            Status s = decoder->DecodeBatch(serialized_examples, &result);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
            return result;
          },
          py::call_guard<py::gil_scoped_release>());

  // We DO NOT RELEASE the GIL before calling ExampleToNumpyDict. It uses
  // Python C-APIs heavily and assumes GIL is held.
  m.def("ExampleToNumpyDict",
        [](absl::string_view serialized_example) -> py::object {
          PyObject* numpy_dict = nullptr;
          Status s = ExampleToNumpyDict(serialized_example, &numpy_dict);
          if (!s.ok()) {
            throw std::runtime_error(s.ToString());
          }
          // "steal" does notincrement the refcount of numpy_dict. (which is
          // already 1 after creation.
          return py::reinterpret_steal<py::object>(numpy_dict);
        });
}

}  // namespace tfx_bsl
