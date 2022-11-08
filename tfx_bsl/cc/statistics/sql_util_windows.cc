// Copyright 2022 Google LLC
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
#include "absl/status/statusor.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace statistics {

// No-op version of EvaluatePredicate for Windows.
absl::StatusOr<bool> EvaluatePredicate(
    const tensorflow::metadata::v0::FeatureNameStatistics& feature_statistics,
    const std::string& query) {
  return absl::UnimplementedError(
      "EvaluatePredicate not supported on Windows.");
}

// No-op version of EvaluatePredicate for Windows.
absl::StatusOr<bool> EvaluatePredicate(
    const tensorflow::metadata::v0::FeatureNameStatistics&
        feature_statistics_base,
    const tensorflow::metadata::v0::FeatureNameStatistics&
        feature_statistics_test,
    const std::string& query) {
  return absl::UnimplementedError(
      "EvaluatePredicate not supported on Windows.");
}

}  // namespace statistics
}  // namespace tfx_bsl
