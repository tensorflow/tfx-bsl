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
#include "tfx_bsl/cc/sketches/sketches_submodule.h"

#include <memory>

#include "absl/strings/string_view.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tfx_bsl/cc/pybind11/absl_casters.h"
#include "tfx_bsl/cc/sketches/kmv_sketch.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"

namespace tfx_bsl {
namespace py = pybind11;

using ::tfx_bsl::sketches::KmvSketch;

void DefineKmvSketchSubmodule(py::module sketch_module) {
  auto m = sketch_module.def_submodule("kmv_sketch");
  m.doc() = "Pybind11 bindings for KmvSketch class.";
  py::class_<KmvSketch>(m, "KmvSketch")
      .def(py::init<const int &>())
      .def( "AddValues",
           [](KmvSketch &sketch, const std::shared_ptr<arrow::Array>& array) {
              Status s = sketch.AddValues(*array);
              if (!s.ok()) {
                throw std::runtime_error(s.ToString());
              }
            },
           py::doc("Add an Arrow array of values to this sketch."),
           py::call_guard<py::gil_scoped_release>())
      .def("Merge",
           [](KmvSketch &sketch, KmvSketch &other) {
             Status s = sketch.Merge(other);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
           },
           py::doc("Merge other KmvSketch object into this sketch."),
           py::call_guard<py::gil_scoped_release>())
      .def("Estimate", &KmvSketch::Estimate,
           py::doc("Estimate number of unique values."),
           py::call_guard<py::gil_scoped_release>())
      .def("Serialize", [](KmvSketch &sketch){
            std::string serialized;
            {
              // Release the GIL during the call to Serialize
              py::gil_scoped_release release_gil;
              serialized = sketch.Serialize();
            }
            return py::bytes(serialized);
           },
          py::doc("Serialize the sketch as a string."))
      .def_static("Deserialize",
           [](absl::string_view byte_string){
              return KmvSketch::Deserialize(byte_string);
            },
          py::doc("Deserialize the string to a KmvSketch object."))
      // Pickle support
      .def(py::pickle(
          [](KmvSketch &sketch) {  // __getstate__
            std::string serialized;
            {
              // Release the GIL during the call to Serialize
              py::gil_scoped_release release_gil;
              serialized = sketch.Serialize();
            }
            return py::bytes(serialized);
          },
          [](py::bytes byte_string) {  // __setstate__
            char* data;
            Py_ssize_t size;
            PyBytes_AsStringAndSize(byte_string.ptr(), &data, &size);
            return KmvSketch::Deserialize(absl::string_view(data, size));
          }
      ));
}

void DefineSketchesSubmodule(py::module main_module) {
  auto m = main_module.def_submodule("sketches");
  m.doc() = "Sketches.";
  DefineKmvSketchSubmodule(m);
}

}   // namespace tfx_bsl
