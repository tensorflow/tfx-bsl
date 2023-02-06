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

import inspect
from typing import Optional

import apache_beam as beam

# Beam might grow the batch size too large for Arrow BinaryArray / ListArray
# to hold the contents (e.g. if the sum of the length of a string feature in
# a batch exceeds 2GB). Before the decoder can produce LargeBinaryArray /
# LargeListArray, we have to cap the batch size.
_BATCH_SIZE_CAP = 1000


def GetBatchElementsKwargs(batch_size: Optional[int]):
  """Returns the kwargs to pass to beam.BatchElements()."""
  if batch_size is not None:
    return {
        "min_batch_size": batch_size,
        "max_batch_size": batch_size,
    }
  # Allow `BatchElements` to tune the values with the given parameters.
  result = {
      "min_batch_size": 1,
      "max_batch_size": _BATCH_SIZE_CAP,
      "target_batch_overhead": 0.05,
      "target_batch_duration_secs": 1,
      "variance": 0.25,
  }
  # We fix the parameters here to prevent Beam changes from immediately
  # affecting all dependencies.
  # TODO(b/266803710): Clean this up after deciding on optimal batch_size
  # selection logic.
  batch_elements_signature = inspect.signature(beam.BatchElements)
  if (
      "target_batch_duration_secs_including_fixed_cost"
      in batch_elements_signature.parameters
  ):
    result["target_batch_duration_secs_including_fixed_cost"] = 1
  return result
