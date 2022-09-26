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
#include "tfx_bsl/cc/statistics/sql_util.h"

#include <string>

#include "zetasql/public/analyzer_options.h"
#include "zetasql/public/evaluator.h"
#include "zetasql/public/value.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow_metadata/proto/v0/statistics.pb.h"

namespace tfx_bsl {
namespace statistics {

using tensorflow::metadata::v0::FeatureNameStatistics;

namespace {

zetasql_base::StatusOr<bool> GetResult(
    const zetasql_base::StatusOr<zetasql::Value>& result_or) {
  if (!result_or.ok()) return result_or.status();
  zetasql::Value value = result_or.value();
  if (value.is_null()) {
    // Maybe this should be false and not an error state?
    return absl::InternalError("Validation predicate returned null.");
  }
  if (!value.is_valid()) {
    return absl::InternalError("Validate predicate returned invalid value.");
  }
  if (!value.type()->IsBool()) {
    return absl::InternalError("Validation predicate returned non-bool.");
  }
  return value.bool_value();
}

}  // namespace

zetasql_base::StatusOr<bool> EvaluatePredicate(
    const FeatureNameStatistics& feature_statistics,
    const std::string& query) {
  zetasql::TypeFactory type_factory;
  const zetasql::ProtoType* proto_type;
  const absl::Status make_proto_type_status = type_factory.MakeProtoType(
      FeatureNameStatistics::descriptor(),
      &proto_type);
  if (!make_proto_type_status.ok()) return make_proto_type_status;

  zetasql::AnalyzerOptions options;
  const absl::Status analyzer_options_status =
      options.SetInScopeExpressionColumn("feature", proto_type);


  // TODO(zwestrick): Use TFX_BSL_RETURN_IF_ERROR once status utils stops
  // bringing in arrow libs.
  if (!analyzer_options_status.ok()) return analyzer_options_status;

  zetasql::PreparedExpression expr(query);
  const absl::Status expr_status = expr.Prepare(options);
  if (!expr_status.ok()) return expr_status;

  return GetResult(expr.Execute(
      {{"feature", zetasql::values::Proto(proto_type, feature_statistics)}}));
}

zetasql_base::StatusOr<bool> EvaluatePredicate(
    const FeatureNameStatistics&
        feature_statistics_base,
    const FeatureNameStatistics&
        feature_statistics_test,

    const std::string& query) {
  zetasql::TypeFactory type_factory;
  const zetasql::ProtoType* proto_type;
  const absl::Status make_proto_type_status = type_factory.MakeProtoType(
      FeatureNameStatistics::descriptor(),
      &proto_type);
  if (!make_proto_type_status.ok()) return make_proto_type_status;

  zetasql::AnalyzerOptions options;
  for (const auto& name : {"feature_base", "feature_test"}) {
    const absl::Status analyzer_options_status =
        options.AddExpressionColumn(name, proto_type);
    if (!analyzer_options_status.ok()) return analyzer_options_status;
  }

  zetasql::PreparedExpression expr(query);
  const absl::Status expr_status = expr.Prepare(options);
  if (!expr_status.ok()) return expr_status;

  return GetResult(expr.Execute(
      {{"feature_base",
        zetasql::values::Proto(proto_type, feature_statistics_base)},
       {"feature_test",
        zetasql::values::Proto(proto_type, feature_statistics_test)}}));
}

}  // namespace statistics
}  // namespace tfx_bsl
