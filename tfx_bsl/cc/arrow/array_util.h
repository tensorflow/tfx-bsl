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
#ifndef TFX_BSL_CC_ARROW_ARRAY_UTIL_H_
#define TFX_BSL_CC_ARROW_ARRAY_UTIL_H_

#include <memory>

#include "arrow/api.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace tfx_bsl {

// Get lengths of lists in `list_array` in an int32 array.
// Note that null and empty list both are of length 0 and the returned array
// does not have any null element.
// For example [[1,2,3], [], None, [4,5]] => [3, 0, 0, 2]
Status ListLengthsFromListArray(
    const arrow::Array& array,
    std::shared_ptr<arrow::Array>* list_lengths_array);

}  // namespace tfx_bsl
}  // namespace tensorflow
#endif  // TFX_BSL_CC_ARROW_ARRAY_UTIL_H_
