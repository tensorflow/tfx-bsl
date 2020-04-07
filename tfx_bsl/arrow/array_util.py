# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Arrow Array utilities."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import numpy as np
import pyarrow as pa
# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
# pylint: disable=unused-import
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import ListLengthsFromListArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetElementLengths
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetFlattenedArrayParentIndices
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetArrayNullBitmapAsByteArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetBinaryArrayTotalByteSize
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import ValueCounts
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import MakeListArrayFromParentIndicesAndValues
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import CooFromListArray
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import FillNullLists
  from tfx_bsl.cc.tfx_bsl_extension.arrow.array_util import GetByteSize
except ImportError:
  import sys
  sys.stderr.write("Error importing tfx_bsl_extension.arrow.array_util. "
                   "Some tfx_bsl functionalities are not available")
# pytype: enable=import-error
# pylint: enable=g-import-not-at-top
# pylint: enable=unused-import


def ToSingletonListArray(array: pa.Array):
  """Converts an array of `type` to a `ListArray<type>`.

  Where result[i] is null if array[i] is null; [array[i]] otherwise.

  Args:
    array: an arrow Array.
  Returns:
    a ListArray.
  """
  array_size = len(array)
  # fast path: values are not copied.
  if array.null_count == 0:
    return pa.ListArray.from_arrays(
        pa.array(np.arange(0, array_size + 1, dtype=np.int32)), array)

  # null_mask[i] = 1 iff array[i] is null.
  null_mask = np.asarray(GetArrayNullBitmapAsByteArray(array))
  # presence_mask[i] = 0 iff array[i] is null
  presence_mask = np.subtract(1, null_mask, dtype=np.uint8)
  offsets_np = np.zeros((array_size + 1,), np.int32)
  np.cumsum(presence_mask, out=offsets_np[1:])

  # This is the null mask over offsets (but ListArray.from_arrays() uses it as
  # the null mask for the ListArray), so its length is array_size +1, but the
  # last element is always False.
  list_array_null_mask = np.zeros((array_size + 1,), np.bool)
  list_array_null_mask[:array_size] = null_mask.view(np.bool)
  values_non_null = array.take(pa.array(np.flatnonzero(presence_mask)))
  return pa.ListArray.from_arrays(
      pa.array(offsets_np, mask=list_array_null_mask), values_non_null)
