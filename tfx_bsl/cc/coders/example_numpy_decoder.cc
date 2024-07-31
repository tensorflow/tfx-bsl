/* Copyright 2019 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tfx_bsl/cc/coders/example_numpy_decoder.h"

#include <Python.h>

#include <string>

#include "numpy/arrayobject.h"
#include "numpy/npy_common.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"

namespace tfx_bsl {

using ::tensorflow::Example;
using ::tensorflow::Feature;

absl::Status ExampleToNumpyDict(absl::string_view serialized_proto,
                                PyObject** result) {
  // Import numpy. (This is actually a macro, and "ret" is the return value
  // if import fails.)
  import_array1(/*ret=*/absl::InternalError("Unable to import numpy."));
  Example example;
  if (!example.ParseFromArray(serialized_proto.data(),
                              serialized_proto.size())) {
    return absl::DataLossError("Failed to parse input proto.");
  }

  // Initialize Python result dict.
  *result = PyDict_New();

  // Iterate over the features and add it to the dict.
  for (const auto& p : example.features().feature()) {
    const std::string& feature_name = p.first;
    const Feature& feature = p.second;

    PyObject* feature_values_ndarray;

    switch (feature.kind_case()) {
      case Feature::kBytesList: {
        const auto& values = feature.bytes_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray = PyArray_SimpleNew(1, values_dims, NPY_OBJECT);
        PyObject** buffer = reinterpret_cast<PyObject**>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(feature_values_ndarray)));
        for (int i = 0; i < values.size(); ++i) {
          const std::string& v = values[i];
          buffer[i] = PyBytes_FromStringAndSize(v.data(), v.size());
        }
        break;
      }
      case Feature::kFloatList: {
        const auto& values = feature.float_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray = PyArray_SimpleNew(1, values_dims, NPY_FLOAT32);
        memcpy(reinterpret_cast<void*>(PyArray_DATA(
                   reinterpret_cast<PyArrayObject*>(feature_values_ndarray))),
               values.data(), values.size() * sizeof(float));
        break;
      }
      case Feature::kInt64List: {
        const auto& values = feature.int64_list().value();
        // Creating ndarray.
        npy_intp values_dims[] = {static_cast<npy_intp>(values.size())};
        feature_values_ndarray = PyArray_SimpleNew(1, values_dims, NPY_INT64);
        memcpy(reinterpret_cast<void*>(PyArray_DATA(
                   reinterpret_cast<PyArrayObject*>(feature_values_ndarray))),
               values.data(), values.size() * sizeof(int64_t));
        break;
      }
      case Feature::KIND_NOT_SET: {
        // If we have a feature with no value list, we consider it to be a
        // missing value.
        feature_values_ndarray = Py_None;
        Py_INCREF(Py_None);
        break;
      }
      default: {
        return absl::DataLossError("Invalid value list in input proto.");
      }
    }
    const int err = PyDict_SetItemString(*result, feature_name.data(),
                                         feature_values_ndarray);
    Py_XDECREF(feature_values_ndarray);
    if (err == -1) {
      return absl::InternalError("Failed to insert item into Dict.");
    }
  }
  return absl::OkStatus();
}

}  // namespace tfx_bsl
