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
from typing import Any, List, Optional, Text

import apache_beam as beam
import numpy as np
import pyarrow as pa
import six
import tensorflow as tf
from tfx_bsl.tfxio import telemetry
from tfx_bsl.tfxio import tfxio


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

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(bytes)
    def _PTransformFn(pcoll_or_pipeline: Any):
      return (pcoll_or_pipeline
              | "ReadRawRecords" >> self._RawRecordBeamSourceInternal()
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

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll_or_pipeline: Any):
      """Converts raw records to RecordBatches."""
      return (
          pcoll_or_pipeline
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


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def ReadTfRecord(pipeline: beam.Pipeline,
                 file_pattern: List[Text]) -> beam.pvalue.PCollection:
  """A Beam source that reads multiple TFRecord file patterns."""
  assert file_pattern, "Must provide at least one file pattern."
  # TODO(b/162261470): consider using beam.io.tfrecordio.ReadAllFromTFRecord
  # once the # concern over size estimation is addressed (also see
  # b/161935932#comment13).
  pcolls = []
  for i, f in enumerate(file_pattern):
    pcolls.append(pipeline
                  | "ReadFromTFRecord[%d]" % i >> beam.io.ReadFromTFRecord(
                      f, coder=beam.coders.BytesCoder()))

  return pcolls | "FlattenPCollsFromPatterns" >> beam.Flatten()


def DetectCompressionType(file_patterns: tf.Tensor) -> tf.Tensor:
  """A TF function that detects compression type given file patterns.

  It simply looks at the names (extensions) of files matching the patterns.

  Args:
    file_patterns: A 1-D string tensor that contains the file patterns.

  Returns:
    A scalar string tensor. The contents are either "GZIP", "" or
    "INVALID_MIXED_COMPRESSION_TYPES".
  """
  # Implementation notes:
  # Because the result of this function usually feeds to another
  # function that creates a TF dataset, and the whole dataset creating logic
  # is usually warpped in an input_fn in the trainer, this function must
  # be a pure composition of TF ops. To be compatible with
  # tf.compat.v1.make_oneshot_dataset, this function cannot be a @tf.function,
  # and it cannot contain any conditional / stateful op either.
  # Once we decide to stop supporting TF 1.x and tf.compat.v1, we can rewrite
  # this as a @tf.function, and use tf.cond / tf.case to make the logic more
  # readable.

  files = tf.io.matching_files(file_patterns)
  is_gz = tf.strings.regex_full_match(files, r".*\.gz$")
  all_files_are_not_gz = tf.math.reduce_all(~is_gz)
  all_files_are_gz = tf.math.reduce_all(is_gz)
  # Encode the 4 cases as integers 0b00 - 0b11 where
  # `all_files_are_not_gz` is bit 0
  # `all_files_are_gz` is bit 1
  # 00: invalid, some files are gz some files are not
  # 01: all are not gz
  # 10: all are gz
  # 11: the only possibility is `files` is empty, can be arbitrary.
  formats = tf.constant(["INVALID_MIXED_COMPRESSION_TYPES", "", "GZIP", ""])
  index = (tf.bitwise.left_shift(tf.cast(all_files_are_gz, tf.int32), 1) +
           tf.cast(all_files_are_not_gz, tf.int32))

  return formats[index]
