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
#include "tfx_bsl/cc/pybind11/absl_casters.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tfx_bsl/cc/sketches/kmv_sketch.h"
#include "tfx_bsl/cc/sketches/misragries_sketch.h"
#include "tfx_bsl/cc/sketches/quantiles_sketch.h"
#include "include/pybind11/pytypes.h"
#include "include/pybind11/stl.h"

namespace tfx_bsl {
namespace {
namespace py = pybind11;

using ::tfx_bsl::sketches::KmvSketch;
using ::tfx_bsl::sketches::MisraGriesSketch;
using ::tfx_bsl::sketches::QuantilesSketch;

void DefineKmvSketchClass(py::module sketch_module) {
  py::class_<KmvSketch>(sketch_module, "KmvSketch")
      .def(py::init<const int&>())
      .def( "AddValues",
           [](KmvSketch& sketch, const std::shared_ptr<arrow::Array>& array) {
              Status s = sketch.AddValues(*array);
              if (!s.ok()) {
                throw std::runtime_error(s.ToString());
              }
            },
           py::doc("Updates the sketch with an Arrow array of values."),
           py::call_guard<py::gil_scoped_release>())
      .def("Merge",
           [](KmvSketch& sketch, KmvSketch& other) {
             Status s = sketch.Merge(other);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
           },
           py::doc("Merges another KMV sketch into this sketch. Returns error "
                   "if the other sketch has a different number of buckets than "
                   "this sketch."),
           py::call_guard<py::gil_scoped_release>())
      .def("Estimate", &KmvSketch::Estimate,
           py::doc("Estimates the number of distinct elements."),
           py::call_guard<py::gil_scoped_release>())
      .def("Serialize", [](KmvSketch& sketch){
            std::string serialized;
            {
              // Release the GIL during the call to Serialize
              py::gil_scoped_release release_gil;
              serialized = sketch.Serialize();
            }
            return py::bytes(serialized);
           },
          py::doc("Serializes the sketch as a string."))
      .def_static("Deserialize",
           [](absl::string_view byte_string){
              return KmvSketch::Deserialize(byte_string);
            },
          py::doc("Deserializes the string to a KmvSketch object."),
          py::call_guard<py::gil_scoped_release>())
      // Pickle support
      .def(py::pickle(
          [](KmvSketch& sketch) {  // __getstate__
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

void DefineMisraGriesSketchClass(py::module sketch_module) {
  py::class_<MisraGriesSketch>(sketch_module, "MisraGriesSketch")
      .def(py::init<const int&>())
      .def( "AddValues",
           [](MisraGriesSketch& sketch,
              const std::shared_ptr<arrow::Array>& items) {
              Status s = sketch.AddValues(*items);
              if (!s.ok()) {
                throw std::runtime_error(s.ToString());
              }
            },
           py::doc("Adds an array of items."),
           py::call_guard<py::gil_scoped_release>())
      .def( "AddValues",
           [](MisraGriesSketch& sketch,
              const std::shared_ptr<arrow::Array>& items,
              const std::shared_ptr<arrow::Array>& weights) {
              Status s = sketch.AddValues(*items, *weights);
              if (!s.ok()) {
                throw std::runtime_error(s.ToString());
              }
            },
           py::doc("Adds an array of items with their associated weights. "
                   "Raises an error if the weights are not a FloatArray."),
           py::call_guard<py::gil_scoped_release>())
      .def("Merge",
           [](MisraGriesSketch& sketch, MisraGriesSketch& other) {
             Status s = sketch.Merge(other);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
           },
           py::doc("Merges another MisraGriesSketch into this sketch. Raises "
                   "an error if the sketches do not have the same number of "
                   "buckets."),
           py::call_guard<py::gil_scoped_release>())
      .def("Estimate",
           [](MisraGriesSketch& sketch){
             std::shared_ptr<arrow::Array> result;
             Status s = sketch.Estimate(&result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           },
           py::doc(
               "Creates a struct array <values, counts> of the top-k items."))
      .def("Serialize", [](MisraGriesSketch& sketch){
            std::string serialized;
            {
              // Release the GIL during the call to Serialize
              py::gil_scoped_release release_gil;
              serialized = sketch.Serialize();
            }
            return py::bytes(serialized);
           },
          py::doc("Serializes the sketch into a string."))
      .def_static("Deserialize",
           [](absl::string_view byte_string){
              return MisraGriesSketch::Deserialize(byte_string);
            },
          py::doc("Deserializes the string to a MisraGries object."),
          py::call_guard<py::gil_scoped_release>())
      // Pickle support
      .def(py::pickle(
          [](MisraGriesSketch& sketch) {  // __getstate__
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
            return MisraGriesSketch::Deserialize(absl::string_view(data, size));
          }
      ));
}

void DefineQuantilesSketchClass(py::module sketch_module) {
  py::class_<QuantilesSketch>(sketch_module, "QuantilesSketch")
      .def(py::init<double, int64_t>())
      .def(py::pickle(
          [](QuantilesSketch& sketch) {
            std::string serialized;
            {
              py::gil_scoped_release release_gil;
              serialized = sketch.Serialize();
            }
            return py::bytes(serialized);
          },
          [](py::bytes byte_string) {
            char* data;
            Py_ssize_t size;
            PyBytes_AsStringAndSize(byte_string.ptr(), &data, &size);
            return QuantilesSketch::Deserialize(absl::string_view(data, size));
          }))
      .def(
          "Merge",
          [](QuantilesSketch& sketch, QuantilesSketch& other) {
            sketch.Merge(other);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "AddValues",
          [](QuantilesSketch& sketch,
             const std::shared_ptr<arrow::Array>& values,
             const std::shared_ptr<arrow::Array>& weights) {
            Status s = sketch.AddWeightedValues(values, weights);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "AddValues",
          [](QuantilesSketch& sketch,
             const std::shared_ptr<arrow::Array>& values) {
            Status s = sketch.AddValues(values);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "GetQuantiles",
          [](QuantilesSketch& sketch, int64_t num_quantiles) {
            std::shared_ptr<arrow::Array> result;
            Status s = sketch.GetQuantiles(num_quantiles, &result);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
            return result;
          },
          py::call_guard<py::gil_scoped_release>());
}
}  // namespace

void DefineSketchesSubmodule(py::module main_module) {
  auto m = main_module.def_submodule("sketches");
  m.doc() = "Pybind11 bindings for sketch classes.";
  DefineKmvSketchClass(m);
  DefineMisraGriesSketchClass(m);
  DefineQuantilesSketchClass(m);
}

}   // namespace tfx_bsl
