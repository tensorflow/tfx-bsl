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
#include "tfx_bsl/cc/statistics/statistics_submodule.h"

#include <memory>
#include <stdexcept>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "tfx_bsl/cc/statistics/merge_util.h"
#include "pybind11/attr.h"
#include "pybind11/pytypes.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace {
namespace py = pybind11;

using tensorflow::metadata::v0::DatasetFeatureStatistics;
using tensorflow::metadata::v0::DatasetFeatureStatisticsList;
using tfx_bsl::statistics::AccumulatorOptions;
using tfx_bsl::statistics::DatasetListAccumulator;

void DefineDatasetListAccumulatorClass(py::module stats_module) {
  // TODO(b/202910677): I'm having trouble getting native proto casters working
  // in tfx_bsl. Until then, serialize at the boundary.
  py::class_<DatasetListAccumulator>(stats_module, "DatasetListAccumulator")
      .def(py::init([](int target_version, bool include_empty_placeholder) {
             AccumulatorOptions opts =
                 AccumulatorOptions(target_version, include_empty_placeholder);
             return DatasetListAccumulator(opts);
           }),
           py::arg("target_version") = 0,
           py::arg("include_empty_placeholder") = true)
      .def(
          "MergeDatasetFeatureStatistics",
          [](DatasetListAccumulator& acc, const std::string& shard_serialized) {
            DatasetFeatureStatistics shard;
            if (!shard.ParseFromString(shard_serialized)) {
              throw std::runtime_error(
                  "failed to deserialize DatasetFeatureStatistics");
            }
            absl::Status s = acc.MergeShard(shard);
            if (!s.ok()) {
              throw std::runtime_error(s.ToString());
            }
          },
          py::doc("Merges a collection of DatasetFeatureStatistics shards into "
                  "a single DatasetFeatureStatistics."),
          py::call_guard<py::gil_scoped_release>())
      .def("Get",
           [](DatasetListAccumulator& acc) {
             absl::StatusOr<std::unique_ptr<DatasetFeatureStatisticsList>>
                   result_or;
             {
               py::gil_scoped_release release_gil;
               result_or = acc.Get();
               if (!result_or.ok()) {
                 throw std::runtime_error(result_or.status().ToString());
               }
             }
             return py::bytes((*result_or)->SerializeAsString());
           }),
      py::doc("Retrieve a merged DatasetFeatureStatisticsList.");
}
}  // namespace

void DefineStatisticsSubmodule(py::module main_module) {
  auto m = main_module.def_submodule("statistics");
  m.doc() = "Pybind11 bindings for (TFDV) statistics utilities.";
  DefineDatasetListAccumulatorClass(m);
}

}  // namespace tfx_bsl
