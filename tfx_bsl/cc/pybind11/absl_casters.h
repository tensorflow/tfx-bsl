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

// pybind11 custom casters that bridges python types and absl types.
#ifndef TFX_BSL_CC_PYBIND11_ABSL_STRING_VIEW_CASTER_H_
#define TFX_BSL_CC_PYBIND11_ABSL_STRING_VIEW_CASTER_H_

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "absl/strings/string_view.h"

namespace pybind11 {
namespace detail {
// absl::string_view caster.

#ifdef PYBIND11_HAS_STRING_VIEW

// If pybind11 registered a std::string_view caster, only add an
// absl::string_view caster if that's a different type.
template <typename T>
struct type_caster<
    T, std::enable_if_t<
           !std::is_same<absl::string_view, std::string_view>::value &&
           std::is_same<T, absl::string_view>::value>>
    : string_caster<absl::string_view, true> {};

#else

// If pybind11 didn't register a std::string_view caster, register an
// absl::string_view caster unconditionally.
template <>
struct type_caster<absl::string_view> : string_caster<absl::string_view, true> {
};

#endif

}  // namespace detail
}  // namespace pybind11
#endif  // TFX_BSL_CC_PYBIND11_ABSL_STRING_VIEW_CASTER_H_
