// Copyright 2026 Google LLC
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

#include "tfx_bsl/cc/statistics/statistics_sql_submodule.h"
#include "pybind11/pybind11.h"

namespace tfx_bsl {
void DefineStatisticsSqlSubmodule(pybind11::module main_module) {
  // Dummy implementation to unblock builds on Linux.
  // This satisfies the linker for TFMA tests but provides no real SQL functionality.
}
}  // namespace tfx_bsl
