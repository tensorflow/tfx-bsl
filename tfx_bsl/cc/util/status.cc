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

#include "tfx_bsl/cc/util/status.h"

#include "absl/strings/str_cat.h"

namespace tfx_bsl {
namespace {
std::string error_name(error::Code code) {
  switch (code) {
    case tfx_bsl::error::OK:
      return "OK";
      break;
    case tfx_bsl::error::CANCELLED:
      return "Cancelled";
      break;
    case tfx_bsl::error::UNKNOWN:
      return "Unknown";
      break;
    case tfx_bsl::error::INVALID_ARGUMENT:
      return "Invalid argument";
      break;
    case tfx_bsl::error::DEADLINE_EXCEEDED:
      return "Deadline exceeded";
      break;
    case tfx_bsl::error::NOT_FOUND:
      return "Not found";
      break;
    case tfx_bsl::error::ALREADY_EXISTS:
      return "Already exists";
      break;
    case tfx_bsl::error::PERMISSION_DENIED:
      return "Permission denied";
      break;
    case tfx_bsl::error::UNAUTHENTICATED:
      return "Unauthenticated";
      break;
    case tfx_bsl::error::RESOURCE_EXHAUSTED:
      return "Resource exhausted";
      break;
    case tfx_bsl::error::FAILED_PRECONDITION:
      return "Failed precondition";
      break;
    case tfx_bsl::error::ABORTED:
      return "Aborted";
      break;
    case tfx_bsl::error::OUT_OF_RANGE:
      return "Out of range";
      break;
    case tfx_bsl::error::UNIMPLEMENTED:
      return "Unimplemented";
      break;
    case tfx_bsl::error::INTERNAL:
      return "Internal";
      break;
    case tfx_bsl::error::UNAVAILABLE:
      return "Unavailable";
      break;
    case tfx_bsl::error::DATA_LOSS:
      return "Data loss";
      break;
    default:
      return absl::StrCat("Unknown code(", static_cast<int>(code), ")");
      break;
  }
}
}  // namespace

Status::Status(tfx_bsl::error::Code code, absl::string_view msg) {
  assert(code != tfx_bsl::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

const std::string& Status::empty_string() {
  static const auto& kEmptyString = *new std::string;
  return kEmptyString;
}


std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    return absl::StrCat(error_name(code()), ": ", state_->msg);
  }
}

void Status::IgnoreError() const {
  // no-op
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

}  // namespace tfx_bsl
