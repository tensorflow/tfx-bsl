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
from tfx_bsl.tfxio import telemetry
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

  def __init__(self, telemetry_descriptors: Optional[List[Text]],
               logical_format: Text,
               physical_format: Text,
               raw_record_column_name: Optional[Text] = None):
    super(RecordBasedTFXIO, self).__init__()
    if not self.SupportAttachingRawRecords():
      assert raw_record_column_name is None, (
          "{} did not support attaching raw records, but requested.".format(
              type(self)))
    self._telemetry_descriptors = telemetry_descriptors
    self._logical_format = logical_format
    self._physical_format = physical_format
    self._raw_record_column_name = raw_record_column_name

  @property
  def raw_record_column_name(self) -> Optional[Text]:
    return self._raw_record_column_name

  @property
  def telemetry_descriptors(self) -> Optional[List[Text]]:
    return self._telemetry_descriptors

  def SupportAttachingRawRecords(self) -> bool:
    return False

  def RawRecordBeamSource(self) -> beam.PTransform:
    """Returns a PTransform that produces a PCollection[bytes].

    Used together with RawRecordToRecordBatch(), it allows getting both the
    PCollection of the raw records and the PCollection of the RecordBatch from
    the same source. For example:

    record_batch = pipeline | tfxio.BeamSource()
    raw_record = pipeline | tfxio.RawRecordBeamSource()

    would result in the files being read twice, while the following would only
    read once:

    raw_record = pipeline | tfxio.RawRecordBeamSource()
    record_batch = raw_record | tfxio.RawRecordToRecordBatch()
    """

    @beam.typehints.with_input_types(beam.Pipeline)
    @beam.typehints.with_output_types(bytes)
    def _PTransformFn(pipeline: beam.Pipeline):
      return (pipeline | "ReadRawRecords" >> self._RawRecordBeamSourceInternal()
              | "CollectRawRecordTelemetry" >> telemetry.ProfileRawRecords(
                  self._telemetry_descriptors, self._logical_format,
                  self._physical_format))

    return beam.ptransform_fn(_PTransformFn)()

  def RawRecordToRecordBatch(self,
                             batch_size: Optional[int] = None
                            ) -> beam.PTransform:
    """Returns a PTransform that converts raw records to Arrow RecordBatches.

    The input PCollection must be from self.RawRecordBeamSource() (also see
    the documentation for that method).

    Args:
      batch_size: if not None, the `pa.RecordBatch` produced will be of the
        specified size. Otherwise it's automatically tuned by Beam.
    """

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll: beam.pvalue.PCollection):
      return (pcoll
              | "RawRecordToRecordBatch" >>
              self._RawRecordToRecordBatchInternal(batch_size)
              | "CollectRecordBatchTelemetry" >>
              telemetry.ProfileRecordBatches(self._telemetry_descriptors,
                                             self._logical_format,
                                             self._physical_format))

    return beam.ptransform_fn(_PTransformFn)()

  @abc.abstractmethod
  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    """Returns a PTransform that produces a PCollection[bytes]."""

  @abc.abstractmethod
  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:
    """Returns a PTransform that converts raw records to Arrow RecordBatches."""
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
      column_type = (pa.large_list(pa.large_binary()) if
                     self._can_produce_large_types else pa.list_(pa.binary()))
      if schema.get_field_index(self._raw_record_column_name) != -1:
        raise ValueError(
            "Raw record column name {} collided with a column in the schema."
            .format(self._raw_record_column_name))
      schema = schema.append(
          pa.field(self._raw_record_column_name, column_type))
    return schema

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:

    @beam.typehints.with_input_types(beam.Pipeline)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pipeline: beam.pvalue.PCollection):
      """Converts raw records to RecordBatches."""
      return (
          pipeline
          | "RawRecordBeamSource" >> self.RawRecordBeamSource()
          | "RawRecordToRecordBatch" >> self.RawRecordToRecordBatch(batch_size))

    return beam.ptransform_fn(_PTransformFn)()


def CreateRawRecordColumn(
    raw_records: List[bytes], produce_large_types: bool) -> pa.Array:
  """Returns an Array that satisfies the requirement of a raw record column."""
  list_array_factory = (
      pa.LargeListArray.from_arrays
      if produce_large_types else pa.ListArray.from_arrays)
  binary_type = pa.large_binary() if produce_large_types else pa.binary()
  return list_array_factory(
      np.arange(0, len(raw_records) + 1, dtype=np.int64),
      pa.array(raw_records, type=binary_type))


def AppendRawRecordColumn(
    record_batch: pa.RecordBatch,
    column_name: Text,
    raw_records: List[bytes],
    produce_large_types: bool
) -> pa.RecordBatch:
  """Appends `raw_records` as a new column in `record_batch`."""
  assert record_batch.num_rows == len(raw_records)
  schema = record_batch.schema
  assert schema.get_field_index(column_name) == -1
  raw_record_column = CreateRawRecordColumn(raw_records, produce_large_types)
  return pa.RecordBatch.from_arrays(
      list(record_batch.columns) + [raw_record_column],
      list(schema.names) + [column_name])
