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
#ifndef THIRD_PARTY_PY_TFX_BSL_CC_ARROW_ARROW_SUBMODULE_H_
#define THIRD_PARTY_PY_TFX_BSL_CC_ARROW_ARROW_SUBMODULE_H_

#include "include/pybind11/pybind11.h"

namespace tensorflow {
namespace tfx_bsl {

void DefineArrowSubmodule(pybind11::module main_module);

}  // namespace tfx_bsl
}  // namespace tensorflow
#endif  // THIRD_PARTY_PY_TFX_BSL_CC_ARROW_ARROW_SUBMODULE_H_
