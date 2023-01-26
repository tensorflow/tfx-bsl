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
"""Contains PTransforms that collects telemetry from produced by TFXIO."""

import enum
from typing import Iterable, Callable, List, Optional, Text

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tfx_bsl.arrow import array_util
from tfx_bsl.arrow import table_util
from tfx_bsl.telemetry import util as telemetry_util


@beam.typehints.with_input_types(pa.RecordBatch)
@beam.typehints.with_output_types(pa.RecordBatch)
@beam.ptransform_fn
def ProfileRecordBatches(
    pcoll: beam.PCollection,
    telemetry_descriptors: Optional[List[Text]],
    logical_format: Text,
    physical_format: Text,
    distribution_update_probability: float = 0.1) -> beam.PCollection:
  """An identity transform to profile RecordBatches and updated Beam metrics.

  Args:
    pcoll: a PCollection[pa.RecordBatch]
    telemetry_descriptors: a set of descriptors that identify the component that
      invokes this PTransform. These will be used to construct the namespace
      to contain the beam metrics created within this PTransform. All such
      namespaces will be prefixed by "tfxio.". If None, a default "unknown"
      descriptor will be used.
    logical_format: the logical format of the data (before parsed into
      RecordBatches). Used to construct metric names.
    physical_format: the physical format in which the data is stored on disk.
      Used to construct metric names.
    distribution_update_probability: probability to update the expensive,
      per-row distributions.

  Returns:
    `pcoll` (identity function).
  """
  assert 0 < distribution_update_probability <= 1.0, (
      "Invalid probability: {}".format(distribution_update_probability))
  return pcoll | "ProfileRecordBatches" >> beam.ParDo(
      _ProfileRecordBatchDoFn(telemetry_descriptors, logical_format,
                              physical_format, distribution_update_probability))


@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(bytes)
@beam.ptransform_fn
def ProfileRawRecords(
    pcoll: beam.PCollection,
    telemetry_descriptors: Optional[List[Text]],
    logical_format: Text,
    physical_format: Text) -> beam.PCollection:
  """An identity transform to profile raw records for record based TFXIO."""
  return pcoll | "ProfileRawRecords" >> beam.ParDo(_ProfileRawRecordDoFn(
      telemetry_descriptors, logical_format, physical_format))


class _ValueType(enum.IntEnum):
  INT = 0
  FLOAT = 1
  STRING = 2
  NULL = 3  # pa.is_null()
  OTHER = 4


_IO_TELEMETRY_DESCRIPTOR = ["io"]
_UNKNOWN_TELEMETRY_DESCRIPTORS = ["UNKNOWN_COMPONENT"]


class _ProfileRecordBatchDoFn(beam.DoFn):
  """A DoFn that profiles RecordBatches and updates Beam counters.

  The following metrics are maintained:

  num_rows: Counter. Total number of rows.
  record_batch_byte_size: Distribution. In-memory size of the RecordBatches.
  num_columns: Distribution. Number of present columns per row.
      A column is present in a row if its value is not None.
  num_feature_values: Distribution. Number of (primitive) values per cell.
  num_feature_values[_ValueType]: Distribution. Similar to num_feature_values,
      but sliced by _ValueType.
  num_cells[_ValueType]: Counter. Total number of cells by _ValueType. Note that
      it's sliced by primitive_type if a column is of type
      list<primitive_type>. For other columns, the slice is OTHER.
  """

  def __init__(self, telemetry_descriptors: Optional[List[Text]],
               logical_format: Text,
               physical_format: Text, dist_update_prob: float):
    if telemetry_descriptors is None:
      telemetry_descriptors = _UNKNOWN_TELEMETRY_DESCRIPTORS
    metric_namespace = telemetry_util.MakeTfxNamespace(telemetry_descriptors +
                                                       _IO_TELEMETRY_DESCRIPTOR)
    namer = _GetMetricNamer(logical_format, physical_format)
    self._num_rows_dist = beam.metrics.Metrics.distribution(
        metric_namespace, namer("num_rows")
    )
    self._byte_size_dist = beam.metrics.Metrics.distribution(
        metric_namespace, namer("record_batch_byte_size"))
    self._num_columns_dist = beam.metrics.Metrics.distribution(
        metric_namespace, namer("num_columns"))
    self._num_feature_values_dist = beam.metrics.Metrics.distribution(
        metric_namespace, namer("num_feature_values"))
    self._num_feature_values_dist_by_type = {
        t: beam.metrics.Metrics.distribution(
            metric_namespace, namer("num_feature_values[{}]".format(t.name)))
        for t in _ValueType
    }
    self._num_cells_by_type = {
        t: beam.metrics.Metrics.counter(metric_namespace,
                                        namer("num_cells[{}]".format(t.name)))
        for t in _ValueType
    }
    self._dist_update_prob = dist_update_prob

  def _UpdateNumColumnsDist(self, record_batch: pa.RecordBatch) -> None:
    # Define number of columns of a row to be the number of cells in that row
    # whose values are not null.
    # It can be computed by summing up (element wise) the negation of null
    # flags (converted to integer) of all the arrays.
    null_bitmaps = [
        np.asarray(array_util.GetArrayNullBitmapAsByteArray(c)).view(bool)
        for c in record_batch]
    indicators = [(~bitmap).view(np.uint8) for bitmap in null_bitmaps]
    sum_indicators = np.zeros(record_batch.num_rows, dtype=np.int64)
    for indicator in indicators:
      np.add(sum_indicators, indicator, out=sum_indicators)
    for num_column in sum_indicators.tolist():
      self._num_columns_dist.update(num_column)

  def _UpdateNumValuesDist(self, record_batch: pa.RecordBatch) -> None:
    # Updates the distribution of number of values per cell.
    # Note that a cell could be of a deeper nested type (e.g.
    # Struct or nested ListArray), the number of values actually means
    # lengths of leaves.
    # For example, given the following row:
    # col1               |    col2
    # [[[1, 2], [3]]]    |    [{'a': [1, 2]}, {'b': [3]}]]
    # the number of values for col1 is 3
    # the number of values for col2 will be updated twice because there are
    # two leaves (col2.a, col2.b), with values 2, 1 respectively.

    # Algorithm: create a mapping `m` (int->int) for array `a` so that if
    # m[i] == j, then a[i] belongs to row j in the record batch.
    # Then, np.bincount(m, minlength=record_batch.num_rows)[i] is how many
    # values in `a` belong to row i. As we flatten the array, the mapping
    # needs to be maintained so that it maps a flattened value to a row.
    num_rows = record_batch.num_rows

    def _RecursionHelper(row_indices, array):
      """Flattens `array` while maintains the `row_indices`."""
      array_type = array.type
      if _IsListLike(array_type):
        parent_indices = np.asarray(
            array_util.GetFlattenedArrayParentIndices(array))
        _RecursionHelper(row_indices[parent_indices], array.flatten())
      elif pa.types.is_struct(array_type):
        for child in array.flatten():
          _RecursionHelper(row_indices, child)
      else:
        value_type = _GetValueType(array.type)
        dist_by_type = self._num_feature_values_dist_by_type[value_type]
        for num_values in np.bincount(row_indices, minlength=num_rows).tolist():
          dist_by_type.update(num_values)
          self._num_feature_values_dist.update(num_values)

    for column in record_batch:
      _RecursionHelper(np.arange(num_rows, dtype=np.int64), column)

  def _UpdateNumCellsCounters(self, record_batch: pa.RecordBatch) -> None:
    num_rows = record_batch.num_rows
    for column in record_batch:
      column_type = column.type
      if pa.types.is_null(column_type):
        self._num_cells_by_type[_ValueType.NULL].inc(num_rows)
        continue

      if _IsListLike(column_type):
        value_type = _GetValueType(column_type.value_type)
      else:
        value_type = _ValueType.OTHER
      self._num_cells_by_type[value_type].inc(num_rows - column.null_count)

  def process(self, record_batch: pa.RecordBatch) -> Iterable[pa.RecordBatch]:
    self._num_rows_dist.update(record_batch.num_rows)
    self._UpdateNumCellsCounters(record_batch)
    total_byte_size = table_util.TotalByteSize(
        record_batch, ignore_unsupported=True)
    self._byte_size_dist.update(total_byte_size)
    # These distributions are per-row therefore expensive to update because
    # dist.update() needs to be called num_rows * k times.
    if np.random.rand() < self._dist_update_prob:
      self._UpdateNumColumnsDist(record_batch)
      self._UpdateNumValuesDist(record_batch)
    yield record_batch


def _IsListLike(data_type: pa.DataType) -> bool:
  return pa.types.is_list(data_type) or pa.types.is_large_list(data_type)


def _GetValueType(data_type: pa.DataType) -> _ValueType:
  """Maps a `pa.DataType` to `ValueType`."""
  if pa.types.is_integer(data_type):
    return _ValueType.INT
  if pa.types.is_floating(data_type):
    return _ValueType.FLOAT
  if (pa.types.is_string(data_type) or
      pa.types.is_binary(data_type) or
      pa.types.is_large_string(data_type) or
      pa.types.is_large_binary(data_type)):
    return _ValueType.STRING
  if pa.types.is_null(data_type):
    return _ValueType.NULL
  return _ValueType.OTHER


class _ProfileRawRecordDoFn(beam.DoFn):
  """A DoFn that profiles raw records and updates Beam counters.

  The following metrics are maintained:

  num_raw_records: Counter. Total number of rows.
  raw_record_byte_size: Distribution. Byte size of the raw records.
  """

  def __init__(self, telemetry_descriptors: Optional[List[Text]],
               logical_format: Text, physical_format: Text):
    if telemetry_descriptors is None:
      telemetry_descriptors = _UNKNOWN_TELEMETRY_DESCRIPTORS
    metric_namespace = telemetry_util.MakeTfxNamespace(telemetry_descriptors +
                                                       _IO_TELEMETRY_DESCRIPTOR)
    namer = _GetMetricNamer(logical_format, physical_format)
    self._num_rows = beam.metrics.Metrics.counter(
        metric_namespace, namer("num_raw_records"))
    self._byte_size_dist = beam.metrics.Metrics.distribution(
        metric_namespace, namer("raw_record_byte_size"))

  def process(self, raw_record: bytes) -> Iterable[bytes]:
    self._num_rows.inc()
    self._byte_size_dist.update(len(raw_record))
    yield raw_record


def _GetMetricNamer(
    logical_format: Text, physical_format: Text) -> Callable[[Text], Text]:
  """Returns a function to construct beam metric names."""
  for f in (logical_format, physical_format):
    assert "[" not in f, "Invalid logical / physical format string: %s" % f
    assert "]" not in f, "Invalid logical / physical format string: %s" % f
    assert "-" not in f, "Invalid logical / physical format string: %s" % f

  def _Namer(base_name: Text) -> Text:
    assert "-" not in base_name, "Invalid metric name: %s" % base_name
    return "LogicalFormat[%s]-PhysicalFormat[%s]-%s" % (
        logical_format, physical_format, base_name)

  return _Namer
