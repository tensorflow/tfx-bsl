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
#include "tfx_bsl/cc/arrow/table_util.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "include/pybind11/stl.h"

namespace tfx_bsl {
namespace {
namespace py = ::pybind11;

std::function<
    std::shared_ptr<arrow::Array>(const std::shared_ptr<arrow::Array>&)>
WrapUnaryArrayFunction(Status (*func)(const arrow::Array&,
                                      std::shared_ptr<arrow::Array>*)) {
  return [func](const std::shared_ptr<arrow::Array>& array) {
    std::shared_ptr<arrow::Array> result;
    Status s = func(*array, &result);
    if (!s.ok()) {
      throw std::runtime_error(s.ToString());
    }
    return result;
  };
}

void DefineArrayUtilSubmodule(py::module arrow_module) {
  auto m = arrow_module.def_submodule("array_util");
  m.doc() = "Arrow Array utilities.";
  m.def(
      "ListLengthsFromListArray",
      WrapUnaryArrayFunction(&GetElementLengths),
      py::doc(
          "DEPRECATED. Use GetElementLengths instead.\n"
          "Get lengths of lists in `list_array` in an int32 array. "
          "Note that null and empty list both are of length 0 and the returned "
          "array does not have any null element.\n"
          "For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2]."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "GetElementLengths",
      WrapUnaryArrayFunction(&GetElementLengths),
      py::doc(
          "Get lengths of elements from a list-alike `array` (including binary "
          "and string arrays) in an int32 array. \n"
          "Note that null and empty elements both are of length 0 and the "
          "returned array does not have any null.\n"
          "For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2]."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "GetFlattenedArrayParentIndices",
      WrapUnaryArrayFunction(&GetFlattenedArrayParentIndices),
      py::doc(
          "Makes a int32 array of the same length as flattened `list_array`. "
          "returned_array[i] == j means i-th element in flattened `list_array`"
          "came from j-th list in `list_array`.\n"
          "For example [[1,2,3], [], None, [4,5]] => [0, 0, 0, 3, 3]."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "GetArrayNullBitmapAsByteArray",
      WrapUnaryArrayFunction(&GetArrayNullBitmapAsByteArray),
      py::doc(
          "Makes a uint8 array of the same length as `array`. "
          "returned_array[i] == True iff array[i] is null.\n"
          "Note that this returned array can be converted to a numpy bool array"
          "copy-free."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "GetBinaryArrayTotalByteSize",
      [](const std::shared_ptr<arrow::Array>& array) {
        size_t result;
        Status s = GetBinaryArrayTotalByteSize(*array, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Returns the total byte size of a BinaryArray (note that StringArray "
          "is a subclass of that so is also accepted here) i.e. the length of "
          "the concatenation of all the binary strings in the list), in a"
          "Python Long."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "ValueCounts",
      [](const std::shared_ptr<arrow::Array>& array) {
        std::shared_ptr<arrow::Array> result;
        Status s = ValueCounts(array, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc("Get counts of values in the array. Returns a struct array "
              "<values, counts>."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "MakeListArrayFromParentIndicesAndValues",
      [](size_t num_parents,
         const std::shared_ptr<arrow::Array>& parent_indices,
         const std::shared_ptr<arrow::Array>& values_array) {
        std::shared_ptr<arrow::Array> result;
        Status s = MakeListArrayFromParentIndicesAndValues(
            num_parents, parent_indices, values_array, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Makes an Arrow ListArray from parent indices and values."
          "For example, if num_parents = 6, parent_indices = [0, 1, 1, 3, 3] "
          "and values_array_py is (an arrow Array of) [0, 1, 2, 3, 4], then "
          "the result will be a ListArray of integers: [[0], [1, 2], None, [3, "
          "4], None]. `num_parents` must be a Python integer (int or long) and "
          "it must be greater than or equal to max(parent_indices) + 1. "
          "`parent_indices` must be a int64 1-D numpy array and the indices "
          "must be sorted in increasing order."
          "`values_array` must be an arrow Array and its length must equal to "
          "the length of `parent_indices`."),
      py::call_guard<py::gil_scoped_release>());
}

void DefineTableUtilSubmodule(pybind11::module arrow_module) {
  auto m = arrow_module.def_submodule("table_util");
  m.doc() = "Arrow Table utilities.";
  m.def(
      "MergeTables",
      [](const std::vector<std::shared_ptr<arrow::Table>>& tables) {
        std::shared_ptr<arrow::Table> result;
        Status s = MergeTables(tables, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Merges a list of arrow tables into one. \n"
          "The columns are concatenated (there will be only one chunk per "
          "column). Columns of the same name must be of the same type, or be a "
          "column of NullArrays. If a column in some tables are of type T, in "
          "some other tables are of NullArrays, the concatenated column is of "
          "type T, with nulls representing the rows from the table with "
          "NullArrays. If a column appears in some tables but not in some "
          "other tables, the concatenated column will contain nulls "
          "representing the rows from the table with that column missing."),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "SliceTableByRowIndices",
      [](const std::shared_ptr<arrow::Table>& table,
         const std::shared_ptr<arrow::Array>& row_indices) {
        std::shared_ptr<arrow::Table> result;
        Status s = SliceTableByRowIndices(table, row_indices, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Collects rows in `row_indices` from `table` and form a new Table."
          "The new Table is guaranteed to contain only one chunk."
          "`row_indices` must be an arrow::Int32Array. And the indices in it"
          "must be sorted in ascending order."),
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace

void DefineArrowSubmodule(pybind11::module main_module) {
  arrow::py::import_pyarrow();
  auto m = main_module.def_submodule("arrow");
  m.doc() = "Arrow utilities.";
  DefineArrayUtilSubmodule(m);
  DefineTableUtilSubmodule(m);
}

}  // namespace tfx_bsl
