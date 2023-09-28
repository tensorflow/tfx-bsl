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
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar
from absl import flags

import apache_beam as beam
from tfx_bsl.telemetry import util as telemetry_util

# Beam might grow the batch size too large for Arrow BinaryArray / ListArray
# to hold the contents (e.g. if the sum of the length of a string feature in
# a batch exceeds 2GB). Before the decoder can produce LargeBinaryArray /
# LargeListArray, we have to cap the batch size.
_BATCH_SIZE_CAP = 1000

# Experimental and will be removed in the future.
# Controls whether to delegate batch size tuning to `beam.BatchElements` or to
# batch records based on target size of the batch in bytes.
# TODO(b/266803710): Switch to byte size batching by default and clean this up.
_USE_BYTE_SIZE_BATCHING = flags.DEFINE_bool(
    "tfxio_use_byte_size_batching",
    False,
    (
        "By default input TFXIO sources will delegate tuning of the batch size "
        "of input data to Beam. If this flag is set to True, the sources will "
        "batch elements based on the target batch size in bytes."
    ),
)
# Batch size is determined by the target size in bytes, but not larger than the
# cap.
# Note that this upper bound in byte size applies to the sum of encoded records
# rather than the produced decoded batch itself. In most cases, however, the
# size of the latter is bounded above by the size of the former. Exception to
# this rule is a case when there are many empty features in the encoded
# examples, but even then the difference is not significant and it is likely
# that the actual size cap will be applied first.
_TARGET_BATCH_BYTES_SIZE = 20_971_520  # 20MiB
_BATCH_SIZE_CAP_WITH_BYTE_TARGET = 8192


def _UseByteSizeBatching() -> bool:
  """Cautious access to `tfxio_use_byte_size_batching` flag value."""
  return (
      _USE_BYTE_SIZE_BATCHING.value
      if flags.FLAGS.is_parsed()
      else _USE_BYTE_SIZE_BATCHING.default
  )


def GetBatchElementsKwargs(
    batch_size: Optional[int], element_size_fn: Callable[[Any], int] = len
) -> Dict[str, Any]:
  """Returns the kwargs to pass to beam.BatchElements()."""
  if batch_size is not None:
    return {
        "min_batch_size": batch_size,
        "max_batch_size": batch_size,
    }
  if _UseByteSizeBatching():
    min_element_size = int(
        math.ceil(_TARGET_BATCH_BYTES_SIZE / _BATCH_SIZE_CAP_WITH_BYTE_TARGET)
    )
    return {
        "min_batch_size": _TARGET_BATCH_BYTES_SIZE,
        "max_batch_size": _TARGET_BATCH_BYTES_SIZE,
        "element_size_fn": lambda e: max(element_size_fn(e), min_element_size),
    }
  # Allow `BatchElements` to tune the values with the given parameters.
  # We fix the tuning parameters here to prevent Beam changes from immediately
  # affecting all dependencies.
  result = {
      "min_batch_size": 1,
      "max_batch_size": _BATCH_SIZE_CAP,
      "target_batch_overhead": 0.05,
      "target_batch_duration_secs": 1,
      "variance": 0.25,
  }
  batch_elements_signature = inspect.signature(beam.BatchElements)
  if (
      "target_batch_duration_secs_including_fixed_cost"
      in batch_elements_signature.parameters
  ):
    result["target_batch_duration_secs_including_fixed_cost"] = 1
  return result


def _MakeAndIncrementBatchingMetrics(
    unused_element: Any,
    batch_size: Optional[int],
    telemetry_descriptors: Optional[Sequence[str]],
) -> None:
  """Increments metrics relevant to batching."""
  namespace = telemetry_util.MakeTfxNamespace(
      telemetry_descriptors or ["Unknown"]
  )
  beam.metrics.Metrics.counter(namespace, "tfxio_use_byte_size_batching").inc(
      int(_UseByteSizeBatching())
  )
  beam.metrics.Metrics.counter(namespace, "desired_batch_size").inc(
      batch_size or 0
  )


T = TypeVar("T")


@beam.ptransform_fn
@beam.typehints.with_input_types(T)
@beam.typehints.with_output_types(List[T])
def BatchRecords(
    records: beam.PCollection,
    batch_size: Optional[int],
    telemetry_descriptors: Optional[Sequence[str]],
    record_size_fn: Callable[[T], int] = len,
) -> beam.PCollection:
  """Batches collection of records tuning the batch size if not provided.

  Args:
    records: A PCollection of records to batch.
    batch_size: Desired batch size. If None, will be tuned for optimal
      performance.
    telemetry_descriptors: Descriptors to use for batching metrics.
    record_size_fn: Function used to determine size of each record in bytes.
      Only used if byte size-based batching is enabled. Defaults to `len`
      function suitable for bytes records.

  Returns:
    A PCollection of batched records.
  """
  _ = (
      records.pipeline
      | "CreateSole" >> beam.Create([None])
      | "IncrementMetrics"
      >> beam.Map(
          _MakeAndIncrementBatchingMetrics,
          batch_size=batch_size,
          telemetry_descriptors=telemetry_descriptors,
      )
  )
  return records | "BatchElements" >> beam.BatchElements(
      **GetBatchElementsKwargs(batch_size, record_size_fn)
  )
