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

#ifndef TFX_BSL_CC_UTIL_STATUS_UTIL_H_
#define TFX_BSL_CC_UTIL_STATUS_UTIL_H_

#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "arrow/api.h"

namespace tfx_bsl {
// Creates a tfx_bsl::Status from an arrow::Status.
inline absl::Status FromArrowStatus(::arrow::Status arrow_status) {
  if (ABSL_PREDICT_TRUE(arrow_status.ok())) return absl::OkStatus();

  if (arrow_status.IsNotImplemented()) {
    return absl::UnimplementedError(arrow_status.message());
  }

  return absl::InternalError(absl::StrCat("Arrow error ",
                                          arrow_status.CodeAsString(), " : ",
                                          arrow_status.message()));
}

// For propagating errors when calling a function.
#define TFX_BSL_RETURN_IF_ERROR(...)                       \
  do {                                                     \
    const absl::Status _status = (__VA_ARGS__);            \
    if (ABSL_PREDICT_FALSE(!_status.ok())) return _status; \
  } while (0)

#define _ASSIGN_OR_RETURN_ARROW_NAME(x, y) x##y

#define _TFX_BSL_ASSIGN_OR_RETURN_ARROW_IMPL(result_name, lhs, rexpr) \
  auto result_name = (rexpr);                                         \
  if (!result_name.status().ok()) {                                   \
    return FromArrowStatus(result_name.status());                     \
  }                                                                   \
  lhs = std::move(result_name).MoveValueUnsafe();

#define TFX_BSL_ASSIGN_OR_RETURN_ARROW(lhs, rexpr) \
  _TFX_BSL_ASSIGN_OR_RETURN_ARROW_IMPL(            \
      _ASSIGN_OR_RETURN_ARROW_NAME(__statusor, __COUNTER__), lhs, rexpr)

}  // namespace tfx_bsl

#endif   // TFX_BSL_CC_UTIL_STATUS_UTIL_H_
