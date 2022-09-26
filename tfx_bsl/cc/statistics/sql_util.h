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
#ifndef THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_SQL_UTIL_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_SQL_UTIL_H_

#include <string>

#include "zetasql/public/evaluator.h"
#include "absl/status/statusor.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace statistics {

// APIs are experimental.

// Evaluates a GoogleSQL expression returning a bool with `feature_statistics`
// bound to `feature`.
zetasql_base::StatusOr<bool> EvaluatePredicate(
    const tensorflow::metadata::v0::FeatureNameStatistics& feature_statistics,
    const std::string& query);

// Evaluates a GoogleSQL expression returning a bool with
// `feature_statistics_base` bound to `feature_base` and
// `feature_statistics_test` bound to `feature_test`.
zetasql_base::StatusOr<bool> EvaluatePredicate(
    const tensorflow::metadata::v0::FeatureNameStatistics&
        feature_statistics_base,
    const tensorflow::metadata::v0::FeatureNameStatistics&
        feature_statistics_test,
    const std::string& query);

}  // namespace statistics
}  // namespace tfx_bsl

#endif  // THIRD_PARTY_PY_TFX_BSL_CC_STATISTICS_SQL_UTIL_H_
