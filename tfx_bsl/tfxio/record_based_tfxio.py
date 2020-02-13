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
"""Defines RecordBasedTFXIO interface.

Also common utilities used by its implementations.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc

import apache_beam as beam
import numpy as np
import pyarrow as pa
import six
from tfx_bsl.tfxio import tfxio
from typing import List, Optional, Text


@six.add_metaclass(abc.ABCMeta)
class RecordBasedTFXIO(tfxio.TFXIO):
  """Base class for all TFXIO implementations for record-based on-disk formats.

  `RecordBasedTFXIO` offers the following abstractions that are unique to
  record-based formats:

  `SupportAttachingRawRecords()`: indicates whether this implementation
    supports attaching the raw records as a `LargeList<LargeBinary>` column to
    the produced RecordBatches upon request. If a subclass implements this
    feature, then its `RawRecordToRecordBatch()` must consult
    `self.raw_record_column_name`, and make sure that the produced RecordBatches
    have the raw record column as the last column, with the given name (if not
    None); otherwise it's guaranteed that the raw record column is not
    requested (`self.raw_record_column_name` == None).

  `RawRecordBeamSource()`: returns a PTransform that produces PCollection[bytes]
    (of raw records).

  RawRecordToReocrdBatch(): returns a PTransform that takes `PCollection[bytes]`
    (expected to be what's produced by `RawRecordBeamSource()`) and produces
    `PCollection[RecordBatch]`. It's guaranteed that `BeamSource()` is a
    composition of `RawRecordBeamSource()` and `RawRecordToRecordBatch()`.
    This interface is useful if one wants to access both the raw records as
    well as the RecordBatches, because beam does not do Common Sub-expression
    Eliminination, it's more desirable to be able to cache the output of
    `RawRecordBeamSource()` and feed it to `RawRecordToRecordBatch()` than
    calling `BeamSource()` separately as redundant disk reads can be avoided.
  """

  def __init__(self, raw_record_column_name: Optional[Text] = None):
    super(RecordBasedTFXIO, self).__init__()
    if not self.SupportAttachingRawRecords():
      assert raw_record_column_name is None, (
          "{} did not support attaching raw records, but requested.".format(
              type(self)))
    self._raw_record_column_name = raw_record_column_name

  @property
  def raw_record_column_name(self) -> Optional[Text]:
    return self._raw_record_column_name

  def SupportAttachingRawRecords(self) -> bool:
    return False

  @abc.abstractmethod
  def RawRecordBeamSource(self) -> beam.PTransform:
    """Returns a PTransform that produces a PCollection[bytes]."""

  @abc.abstractmethod
  def RawRecordToRecordBatch(self,
                             batch_size: Optional[int] = None
                            ) -> beam.PTransform:
    """Returns a PTransform that converts raw records to Arrow RecordBatches.

    The PTransform takes PCollection[bytes] and outputs
    PCollection[RecordBatches].

    Args:
      batch_size: if not None, the `pa.RecordBatch` produced will be of the
        specified size. Otherwise it's automatically tuned by Beam.
    """
    pass

  @abc.abstractmethod
  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    """Returns the Arrow schema that does not contain the raw record column.

    Even if self.raw_record_column is not None.

    Returns:
      a pa.Schema.
    """
    pass

  def ArrowSchema(self) -> pa.Schema:
    schema = self._ArrowSchemaNoRawRecordColumn()
    if self._raw_record_column_name is not None:
      if schema.get_field_index(self._raw_record_column_name) != -1:
        raise ValueError(
            "Raw record column name {} collided with a column in the schema."
            .format(self._raw_record_column_name))
      schema = schema.append(
          pa.field(self._raw_record_column_name, pa.list_(pa.binary())))
    return schema

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:

    @beam.ptransform_fn
    @beam.typehints.with_input_types(beam.Pipeline)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pipeline: beam.pvalue.PCollection):
      return (
          pipeline
          | "ReadRawRecords" >> self.RawRecordBeamSource()
          | "RawRecordToRecordBatch" >> self.RawRecordToRecordBatch(
              batch_size))

    return _PTransformFn()  # pylint: disable=no-value-for-parameter


def CreateRawRecordColumn(raw_records: List[bytes]) -> pa.Array:
  """Returns an Array that satisfies the requirement of a raw record column."""
  return pa.ListArray.from_arrays(
      np.arange(0, len(raw_records) + 1, dtype=np.int64),
      pa.array(raw_records, type=pa.binary()))


def AppendRawRecordColumn(
    record_batch: pa.RecordBatch,
    column_name: Text,
    raw_records: List[bytes],
) -> pa.RecordBatch:
  """Appends `raw_records` as a new column in `record_batch`."""
  assert record_batch.num_rows == len(raw_records)
  schema = record_batch.schema
  assert schema.get_field_index(column_name) == -1
  raw_record_column = CreateRawRecordColumn(raw_records)
  return pa.RecordBatch.from_arrays(
      list(record_batch.columns) + [raw_record_column],
      list(schema.names) + [column_name])


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
