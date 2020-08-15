// Copyright 2020 Google LLC
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

// pybind11 custom casters that bridges pyarrow and C++ Arrow.
#ifndef TFX_BSL_PYBIND11_ARROW_CASTERS_H_
#define TFX_BSL_PYBIND11_ARROW_CASTERS_H_

#ifdef TFX_BSL_USE_ARROW_C_ABI
#include "tfx_bsl/cc/pybind11/arrow_casters_c_abi.h"
#else
#include "tfx_bsl/cc/pybind11/arrow_casters_py_c_api.h"
#endif  // TFX_BSL_USE_ARROW_C_ABI

#endif  // TFX_BSL_PYBIND11_ARROW_CASTERS_H_
