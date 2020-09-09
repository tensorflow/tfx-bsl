# Copyright 2020 Google LLC
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
"""Utilities for batching."""

from typing import Optional

# Beam might grow the batch size too large for Arrow BinaryArray / ListArray
# to hold the contents (e.g. if the sum of the length of a string feature in
# a batch exceeds 2GB). Before the decoder can produce LargeBinaryArray /
# LargeListArray, we have to cap the batch size.
_BATCH_SIZE_CAP = 1000


def GetBatchElementsKwargs(batch_size: Optional[int]):
  """Returns the kwargs to pass to beam.BatchElements()."""
  if batch_size is None:
    return {"max_batch_size": _BATCH_SIZE_CAP}
  return {
      "min_batch_size": batch_size,
      "max_batch_size": batch_size,
  }
