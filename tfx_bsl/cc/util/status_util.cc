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
#include "tfx_bsl/cc/util/status_util.h"

#include "tfx_bsl/cc/util/status.h"

namespace tfx_bsl {
// For now, all arrow errors are converted to InternalError.
Status FromArrowStatus(::arrow::Status arrow_status) {
  if (arrow_status.ok()) return Status::OK();

  if (arrow_status.IsNotImplemented()) {
    return errors::Unimplemented(arrow_status.message());
  }

  return errors::Internal(absl::StrCat("Arrow error ",
                                       arrow_status.CodeAsString(), " : ",
                                       arrow_status.message()));
}

}  // namespace tfx_bsl
