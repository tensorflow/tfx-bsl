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
#include "tfx_bsl/cc/arrow/array_util.h"

#include "absl/strings/str_cat.h"
#include "tfx_bsl/cc/util/status_util.h"

namespace tfx_bsl {
namespace {
using ::arrow::Array;
using ::arrow::ListArray;
using ::arrow::Int32Builder;

Status GetListArray(const Array& array, const ListArray** list_array) {
  if (array.type()->id() != arrow::Type::LIST) {
    return errors::InvalidArgument(absl::StrCat(
        "Expected ListArray but got type id: ", array.type()->ToString()));
  }
  *list_array = static_cast<const ListArray*>(&array);
  return Status::OK();
}

}  // namespace

Status ListLengthsFromListArray(
    const Array& array,
    std::shared_ptr<Array>* list_lengths_array) {
  const ListArray* list_array;
  TFX_BSL_RETURN_IF_ERROR(GetListArray(array, &list_array));

  Int32Builder lengths_builder;
  TFX_BSL_RETURN_IF_ERROR(
      FromArrowStatus(lengths_builder.Reserve(list_array->length())));
  for (int i = 0; i < list_array->length(); ++i) {
    lengths_builder.UnsafeAppend(list_array->value_length(i));
  }
  return FromArrowStatus(lengths_builder.Finish(list_lengths_array));
}

}  // namespace tfx_bsl
