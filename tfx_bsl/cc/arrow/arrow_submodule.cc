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

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "arrow/api.h"
#include "tfx_bsl/cc/arrow/array_util.h"
#include "tfx_bsl/cc/arrow/sql_util_submodule.h"
#include "tfx_bsl/cc/arrow/table_util.h"
#include "tfx_bsl/cc/pybind11/arrow_casters.h"
#include "pybind11/stl.h"

namespace tfx_bsl {
namespace {
namespace py = ::pybind11;

std::function<
    std::shared_ptr<arrow::Array>(const std::shared_ptr<arrow::Array>&)>
WrapUnaryArrayFunction(absl::Status (*func)(const arrow::Array&,
                                            std::shared_ptr<arrow::Array>*)) {
  return [func](const std::shared_ptr<arrow::Array>& array) {
    std::shared_ptr<arrow::Array> result;
    absl::Status s = func(*array, &result);
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
      py::doc("DEPRECATED. Use GetElementLengths instead."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "GetElementLengths",
      WrapUnaryArrayFunction(&GetElementLengths),
      py::doc(
          "Get lengths of elements from a list-alike `array` (including binary "
          "and string arrays) in an int64 array. \n"
          "Note that null and empty elements both are of length 0 and the "
          "returned array does not have any null.\n"
          "For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2]."),
      py::call_guard<py::gil_scoped_release>());

  m.def("GetFlattenedArrayParentIndices",
        WrapUnaryArrayFunction(&GetFlattenedArrayParentIndices),
        py::doc("Makes a int32 or int64 array of the same length as flattened "
                "`list_array`. returned_array[i] == j means i-th element in "
                "flattened `list_array` came from j-th list in `list_array`.\n"
                "Returns an Int32Array if the input is a ListArray, or "
                "Int64Array if the input is a LargeListArray. \n"
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
        absl::Status s = GetBinaryArrayTotalByteSize(*array, &result);
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
      "IndexIn",
      [](const std::shared_ptr<arrow::Array>& values,
         const std::shared_ptr<arrow::Array>& value_set) {
        std::shared_ptr<arrow::Array> result;
        absl::Status s = IndexIn(values, value_set, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "IndexIn examines each slot in a vaues against a value_set array.\n\n"
          "If the value is not found in value_set, null will be output."
          "If found, the index of occurrence within value_set "
          "(ignoring duplicates) will be output.\n\n"
          "For example given values = [99, 42, 3, null] and"
          "value_set = [3, 99], the output will be = [1, null, 0, null]\n\n"
          "Note: Null in the values is considered to match"
          "a null in the value_set array. For example given"
          "values = [99, 42, 3, null] and value_set = [3, 99, null],"
          "the output will be = [1, null, 0, 2]\n\n"
          "TODO(b/203116559): IndexIn links against Apache Arrow code "
          "which has different behavior for duplicate elements in value_set."
          "Version 2.0 would silently remove duplicates, version 4.0 "
          "disallowed them and from version 5.0 the index of the first element "
          "is used."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "MakeListArrayFromParentIndicesAndValues",
      [](size_t num_parents,
         const std::shared_ptr<arrow::Array>& parent_indices,
         const std::shared_ptr<arrow::Array>& values_array,
         const bool empty_list_as_null) {
        std::shared_ptr<arrow::Array> result;
        absl::Status s = MakeListArrayFromParentIndicesAndValues(
            num_parents, parent_indices, values_array, empty_list_as_null,
            &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Makes an Arrow LargeListArray from parent indices and values."
          "For example, if num_parents = 6, parent_indices = [0, 1, 1, 3, 3] "
          "and values_array_py is (an arrow Array of) [0, 1, 2, 3, 4], then "
          "the result will be a LargeListArray of integers: "
          "[[0], [1, 2], <empty_list>, [3, 4], <empty_list>] "
          "where <empty_list> is `null` if `empyt_list_as_null` is true, or "
          "`[]`if false. `num_parents` must be a Python integer (int or long) "
          "and it must be greater than or equal to max(parent_indices) + 1. "
          "`parent_indices` must be a int64 1-D numpy array and the indices "
          "must be sorted in increasing order."
          "`values_array` must be an arrow Array and its length must equal to "
          "the length of `parent_indices`."),
      py::call_guard<py::gil_scoped_release>());

  m.def(
      "CooFromListArray",
      [](const std::shared_ptr<arrow::Array>& list_array) {
        std::shared_ptr<arrow::Array> coo;
        std::shared_ptr<arrow::Array> dense_shape;
        absl::Status s = CooFromListArray(list_array, &coo, &dense_shape);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return std::make_pair(coo, dense_shape);
      },
      py::doc(
          "Converts a ListArray to a COO (coordinate list) represented sparse "
          "tensor.\n"
          "`list_array` should be a (Large)ListArray<InnerArray> where "
          "InnerArray is a (Large)ListArray<InnerArray> or any primitive array "
          "or binary array (i.e. nested lists are supported). \n"
          "Two arrays are produced: `coo_array` is an Int64Array that contains "
          "the coordinates of flattened values of `list_array`. If "
          "`list_array` is N-nested (ListArray<primitive> is 1-nested), each "
          "coordinate will contain N + 1 numbers. The coordinates are "
          "concatenated together. `dense_shape_array` is an Int64Array that "
          "contains the size of the bounding-box of `list_array`. Note that "
          "nulls and empty lists are not distinguished in the COO form."),
      py::call_guard<py::gil_scoped_release>());

  m.def("FillNullLists", [](const std::shared_ptr<arrow::Array>& list_array,
                            const std::shared_ptr<arrow::Array>& fill_with) {
    std::shared_ptr<arrow::Array> result;
    absl::Status s = FillNullLists(list_array, fill_with, &result);
    if (!s.ok()) {
      throw std::runtime_error(s.ToString());
    }
    return result;
  });

  m.def("GetByteSize", [](const std::shared_ptr<arrow::Array>& array) {
    size_t result;
    absl::Status s = GetByteSize(*array, &result);
    if (!s.ok()) {
      throw std::runtime_error(s.ToString());
    }
    return result;
  });

  m.def(
      "CountValidUTF8",
      [](const std::shared_ptr<arrow::Array>& array) {
        absl::StatusOr<size_t> count_or = CountValidUtf8(*array);
        if (!count_or.ok()) {
          throw std::runtime_error(count_or.status().ToString());
        }
        return *count_or;
      },
      py::doc("Returns the count of valid utf8 strings from a (large) string "
              "or binary array."),
      py::call_guard<py::gil_scoped_release>());
}

void DefineTableUtilSubmodule(pybind11::module arrow_module) {
  auto m = arrow_module.def_submodule("table_util");
  m.doc() = "Arrow Table utilities.";
  m.def(
      "MergeRecordBatches",
      [](const std::vector<std::shared_ptr<arrow::RecordBatch>>&
             record_batches) {
        std::shared_ptr<arrow::RecordBatch> result;
        absl::Status s = MergeRecordBatches(record_batches, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc(
          "Merges a list of record batches into one. "
          "The columns are concatenated. "
          "Columns of the same name must be of compatible types.\n"
          "Two types are compatible if:\n"
          "  - they are equal, or\n"
          "  - one of them is Null\n"
          "  - both are list<> or large_list<>, and their child types are "
          "compatible (note that large_list<> and list<> are not compatible), "
          "or\n"
          "  - both are struct<>, and their children of the same name are "
          "compatible. they don't need to have the same set of children.\n"
          "Rules for concatanating two compatible but not equal arrays:\n"
          "  1. if one of them is Null, then the result is of the other type, "
          "with nulls representing elements from the NullArray.\n"
          "  2. two compatible list<> arrays are concatenated recursively."
          "  3. two compatible struct<> arrays will result in a struct<> that "
          "contains children from both arrays. If on array is missing a child, "
          "it is considered as if it had that child as a NullArray. Child "
          "arrays are concatenated recusrively.\n"
          "Returns an error if there's any incompatibility."),
      py::call_guard<py::gil_scoped_release>());
  m.def(
      "TotalByteSize",
      [](const std::shared_ptr<arrow::RecordBatch>& record_batch,
         const bool ignore_unsupported) {
        size_t result;
        absl::Status s =
            TotalByteSize(*record_batch, ignore_unsupported, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::arg("record_batch"), py::arg("ignore_unsupported") = false,
      py::doc("Returns the total byte size of all the buffers a record batch "
              "consists of. This value might be larger than the actual memory "
              "occupied by those buffers because buffers might share the "
              "underlying memory"),
      py::call_guard<py::gil_scoped_release>());
  // TODO(zhuo): pa.RecordBatch.take is available starting from arrow 0.17.
  m.def(
      "RecordBatchTake",
      [](const std::shared_ptr<arrow::RecordBatch>& record_batch,
         const std::shared_ptr<arrow::Array>& indices) {
        std::shared_ptr<arrow::RecordBatch> result;
        absl::Status s = RecordBatchTake(record_batch, indices, &result);
        if (!s.ok()) {
          throw std::runtime_error(s.ToString());
        }
        return result;
      },
      py::doc("Returns a RecordBatch that contains rows in `indices`."),
      py::call_guard<py::gil_scoped_release>());
}

}  // namespace

void DefineArrowSubmodule(pybind11::module main_module) {
  auto m = main_module.def_submodule("arrow");
  m.doc() = "Arrow utilities.";
  DefineArrayUtilSubmodule(m);
  DefineTableUtilSubmodule(m);
  DefineSqlUtilSubmodule(m);
}
}  // namespace tfx_bsl
