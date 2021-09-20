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

#include "arrow/api.h"
#include "tfx_bsl/cc/coders/example_coder.h"
#include "tfx_bsl/cc/coders/example_numpy_decoder.h"
#include "tfx_bsl/cc/pybind11/absl_casters.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "tfx_bsl/cc/util/status.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

// Making the type cast opaque to prevent data copy on Python->C++ trip.
PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string, std::vector<std::string>>);

namespace tfx_bsl {
namespace py = pybind11;

void DefineCodersSubmodule(py::module main_module) {
  auto m = main_module.def_submodule("coders");
  m.doc() = "Coders.";

  py::class_<ExamplesToRecordBatchDecoder>(m, "ExamplesToRecordBatchDecoder")
      .def(py::init([](absl::string_view serialized_schema) {
             std::unique_ptr<ExamplesToRecordBatchDecoder> result;
             Status s = ExamplesToRecordBatchDecoder::Make(
                 serialized_schema,  &result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }),
           py::arg("serialized_schema"))
      .def(py::init([] {
             std::unique_ptr<ExamplesToRecordBatchDecoder> result;
             Status s = ExamplesToRecordBatchDecoder::Make(
                 absl::nullopt, &result);
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }))
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
          py::call_guard<py::gil_scoped_release>())
      .def("ArrowSchema",
           [](ExamplesToRecordBatchDecoder* decoder) {
             auto result = decoder->ArrowSchema();
             if (!result) {
               throw std::runtime_error(
                   "ExamplesToRecordBatchDecoder: Unable to get the arrow "
                   "schema if a TFMD schema was not provided at the "
                   "construction time.");
             }
             return result;
           });

  py::class_<SequenceExamplesToRecordBatchDecoder>(
      m, "SequenceExamplesToRecordBatchDecoder")
      .def(py::init([](const std::string& sequence_feature_column_name,
                       const absl::string_view serialized_schema) {
             std::unique_ptr<SequenceExamplesToRecordBatchDecoder> result;
             Status s;
             if (serialized_schema.empty()) {
               s = SequenceExamplesToRecordBatchDecoder::Make(
                   absl::nullopt, sequence_feature_column_name,
                   &result);
             } else {
               s = SequenceExamplesToRecordBatchDecoder::Make(
                   serialized_schema, sequence_feature_column_name,
                   &result);
             }
             if (!s.ok()) {
               throw std::runtime_error(s.ToString());
             }
             return result;
           }),
           py::arg("sequence_feature_column_name"),
           py::arg("serialized_schema") = "")
      .def(
          "DecodeBatch",
          [](SequenceExamplesToRecordBatchDecoder* decoder,
             const std::vector<absl::string_view>&
                 serialized_sequence_examples) {
            std::shared_ptr<arrow::RecordBatch> result;
            Status s =
                decoder->DecodeBatch(serialized_sequence_examples, &result);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
            return result;
          },
          py::call_guard<py::gil_scoped_release>())
       .def("ArrowSchema",
           [](SequenceExamplesToRecordBatchDecoder* decoder) {
             auto result = decoder->ArrowSchema();
             if (!result) {
               throw std::runtime_error(
                   "SequenceExamplesToRecordBatchDecoder: Unable to get the "
                   "arrow schema if a TFMD schema was not provided at the "
                   "construction time.");
             }
             return result;
           });
  // We DO NOT RELEASE the GIL before calling ExampleToNumpyDict. It uses
  // Python C-APIs heavily and assumes GIL is held.
  m.def("ExampleToNumpyDict",
        [](absl::string_view serialized_example) -> py::object {
          PyObject* numpy_dict = nullptr;
          Status s = ExampleToNumpyDict(serialized_example, &numpy_dict);
          if (!s.ok()) {
            throw std::runtime_error(s.ToString());
          }
          // "steal" does not increment the refcount of numpy_dict. (which is
          // already 1 after creation.
          return py::reinterpret_steal<py::object>(numpy_dict);
        });

  // Adding binding to opaque type to enable pass-by-reference below.
  py::bind_map<std::unordered_map<std::string, std::vector<std::string>>>(
      m, "FeatureNameToColumnsMap");
  m.def(
      "RecordBatchToExamples",
      [](std::shared_ptr<arrow::RecordBatch> record_batch,
         const std::unordered_map<std::string, std::vector<std::string>>&
             nested_features) -> py::list {
        std::vector<std::string> serialized_examples;
        {
          // Release the GIL during the call to RecordBatchToExamples.
          py::gil_scoped_release release_gil;
          Status s = RecordBatchToExamples(*record_batch, nested_features,
                                           &serialized_examples);
          if (!s.ok()) {
            throw std::runtime_error(s.ToString());
          }
        }
        py::list bytes_examples;
        for (auto& example : serialized_examples) {
          bytes_examples.append(py::bytes(example));
        }
        return bytes_examples;
      },
      py::arg("record_batch"),
      py::arg("nested_features") =
          std::unordered_map<std::string, std::vector<std::string>>());
}

}  // namespace tfx_bsl
