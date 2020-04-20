# Copyright 2018 Google LLC
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
"""Decode CSV records into in-memory representation for tf data validation."""
# TODO(b/131315065): optimize the CSV decoder.

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import collections
import csv
import apache_beam as beam
import enum
import numpy as np
import pyarrow as pa
import six
import tensorflow as tf
from tfx_bsl.coders import batch_util
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Text, Union

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


CSVCell = bytes
# Text if Python3, bytes otherwise.
CSVLine = Union[Text, bytes]
ColumnName = Union[Text, bytes]


class ColumnType(enum.IntEnum):
  UNKNOWN = -1
  INT = statistics_pb2.FeatureNameStatistics.INT
  FLOAT = statistics_pb2.FeatureNameStatistics.FLOAT
  STRING = statistics_pb2.FeatureNameStatistics.STRING

  # We need the following to hold for type inference to work.
  assert UNKNOWN < INT
  assert INT < FLOAT
  assert FLOAT < STRING


ColumnInfo = collections.namedtuple(
    "ColumnInfo",
    [
        "name",  # type: ColumnName  # pytype: disable=ignored-type-comment
        "type",  # type: Optional[ColumnType]  # pytype: disable=ignored-type-comment
    ])

_SCHEMA_TYPE_TO_COLUMN_TYPE = {
    schema_pb2.INT: ColumnType.INT,
    schema_pb2.FLOAT: ColumnType.FLOAT,
    schema_pb2.BYTES: ColumnType.STRING
}


@beam.ptransform_fn
@beam.typehints.with_input_types(CSVLine)
@beam.typehints.with_output_types(pa.RecordBatch)
def CSVToRecordBatch(lines: beam.pvalue.PCollection,
                     column_names: List[Text],
                     desired_batch_size: Optional[int],
                     delimiter: Text = ",",
                     skip_blank_lines: bool = True,
                     schema: Optional[schema_pb2.Schema] = None,
                     multivalent_columns: Optional[List[Union[Text,
                                                              bytes]]] = None,
                     secondary_delimiter: Optional[Union[Text, bytes]] = None):
  """Decodes CSV records into Arrow RecordBatches.

  Args:
    lines: The pcollection of raw records (csv lines).
    column_names: List of feature names. Order must match the order in the CSV
      file.
    desired_batch_size: Batch size. The output Arrow RecordBatches will have as
      many rows as the `desired_batch_size`. If None, the batch size is auto
      tuned by beam.
    delimiter: A one-character string used to separate fields.
    skip_blank_lines: A boolean to indicate whether to skip over blank lines
      rather than interpreting them as missing values.
    schema: An optional schema of the input data. If this is provided, it must
      contain all columns.
    multivalent_columns: Columns that can contain multiple values. If
      secondary_delimiter is provided, this must also be provided.
    secondary_delimiter: Delimiter used for parsing multivalent columns. If
      multivalent_columns is provided, this must also be provided.

  Returns:
    RecordBatches of the CSV lines.

  Raises:
    ValueError: If the columns do not match the specified csv headers. Or if the
                schema has invalid feature types. Or if the schema does not
                contain all columns.
  """

  csv_lines = (lines | "ParseCSVLines" >> beam.ParDo(ParseCSVLine(delimiter)))

  if schema is not None:
    column_infos = _get_feature_types_from_schema(schema, column_names)
  else:
    # TODO(b/72746442): Consider using a DeepCopy optimization similar to TFT.
    # Do first pass to infer the feature types.
    column_infos = beam.pvalue.AsSingleton(
        csv_lines | "InferColumnTypes" >> beam.CombineGlobally(
            ColumnTypeInferrer(
                column_names=column_names,
                skip_blank_lines=skip_blank_lines,
                multivalent_columns=multivalent_columns,
                secondary_delimiter=secondary_delimiter)))

  # Do second pass to generate the RecordBatches.
  return (csv_lines
          | "BatchCSVLines" >> beam.BatchElements(
              **batch_util.GetBatchElementsKwargs(desired_batch_size))
          | "BatchedCSVRowsToArrow" >> beam.ParDo(
              BatchedCSVRowsToRecordBatch(
                  skip_blank_lines=skip_blank_lines,
                  multivalent_columns=multivalent_columns,
                  secondary_delimiter=secondary_delimiter), column_infos))


@beam.typehints.with_input_types(CSVLine)
@beam.typehints.with_output_types(beam.typehints.List[CSVCell])
class ParseCSVLine(beam.DoFn):
  """A beam.DoFn to parse CSVLines into List[CSVCell]."""

  def __init__(self, delimiter: Union[Text, bytes]):
    self._delimiter = delimiter
    self._reader = None

  def setup(self):
    self._reader = _CSVRecordReader(self._delimiter)

  def process(self, csv_line: CSVLine):
    yield self._reader.read_line(csv_line)


@beam.typehints.with_input_types(beam.typehints.List[CSVCell])
@beam.typehints.with_output_types(beam.typehints.List[ColumnInfo])
class ColumnTypeInferrer(beam.CombineFn):
  """A beam.CombineFn to infer CSV Column types.

  Its input can be produced by ParseCSVLine().
  """

  def __init__(
      self,
      column_names: List[ColumnName],
      skip_blank_lines: bool,
      multivalent_columns: Optional[Set[ColumnName]] = None,
      secondary_delimiter: Optional[Union[Text, bytes]] = None) -> None:
    """Initializes a feature type inferrer combiner."""
    self._column_names = column_names
    self._skip_blank_lines = skip_blank_lines
    self._multivalent_columns = (
        multivalent_columns if multivalent_columns is not None else set())
    if multivalent_columns:
      assert secondary_delimiter, ("secondary_delimiter must be specified if "
                                   "there are multivalent columns")
      self._multivalent_reader = _CSVRecordReader(secondary_delimiter)

  def create_accumulator(self) -> Dict[ColumnName, ColumnType]:
    """Creates an empty accumulator to keep track of the feature types."""
    return {}

  def add_input(self, accumulator: Dict[ColumnName, ColumnType],
                cells: List[CSVCell]) -> Dict[ColumnName, ColumnType]:
    """Updates the feature types in the accumulator using the input row.

    Args:
      accumulator: A dict containing the already inferred feature types.
      cells: A list containing feature values of a CSV record.

    Returns:
      A dict containing the updated feature types based on input row.

    Raises:
      ValueError: If the columns do not match the specified csv headers.
    """
    # If the row is empty and we don't want to skip blank lines,
    # add an empty string to each column.
    if not cells and not self._skip_blank_lines:
      cells = ["" for _ in range(len(self._column_names))]
    elif cells and len(cells) != len(self._column_names):
      raise ValueError("Columns do not match specified csv headers: %s -> %s" %
                       (self._column_names, cells))

    # Iterate over each feature value and update the type.
    for column_name, cell in zip(self._column_names, cells):

      # Get the already inferred type of the feature.
      previous_type = accumulator.get(column_name, None)
      if column_name in self._multivalent_columns:
        current_type = max([
            _infer_value_type(value)
            for value in self._multivalent_reader.read_line(cell)
        ])
      else:
        current_type = _infer_value_type(cell)

      # If the type inferred from the current value is higher in the type
      # hierarchy compared to the already inferred type, we update the type.
      # The type hierarchy is,
      #   INT (level 0) --> FLOAT (level 1) --> STRING (level 2)
      if previous_type is None or current_type > previous_type:
        accumulator[column_name] = current_type
    return accumulator

  def merge_accumulators(
      self, accumulators: List[Dict[ColumnName, ColumnType]]
  ) -> Dict[ColumnName, ColumnType]:
    """Merge the feature types inferred from the different partitions.

    Args:
      accumulators: A list of dicts containing the feature types inferred from
        the different partitions of the data.

    Returns:
      A dict containing the merged feature types.
    """
    result = {}
    for shard_types in accumulators:
      # Merge the types inferred in each partition using the type hierarchy.
      # Specifically, whenever we observe a type higher in the type hierarchy
      # we update the type.
      for feature_name, feature_type in six.iteritems(shard_types):
        if feature_name not in result or feature_type > result[feature_name]:
          result[feature_name] = feature_type
    return result

  def extract_output(
      self, accumulator: Dict[ColumnName, ColumnType]) -> List[ColumnInfo]:
    """Return a list of tuples containing the column info."""
    return [
        ColumnInfo(col_name, accumulator.get(col_name, ColumnType.UNKNOWN))
        for col_name in self._column_names
    ]


@beam.typehints.with_input_types(List[List[CSVCell]], List[ColumnInfo])
@beam.typehints.with_output_types(pa.RecordBatch)
class BatchedCSVRowsToRecordBatch(beam.DoFn):
  """DoFn to convert a batch of csv rows to a RecordBatch."""

  def __init__(self,
               skip_blank_lines: bool,
               multivalent_columns: Optional[Set[ColumnName]] = None,
               secondary_delimiter: Optional[Union[Text, bytes]] = None):
    self._skip_blank_lines = skip_blank_lines
    self._multivalent_columns = (
        multivalent_columns if multivalent_columns is not None else set())
    if multivalent_columns:
      assert secondary_delimiter, ("secondary_delimiter must be specified if "
                                   "there are multivalent columns")
      self._multivalent_reader = _CSVRecordReader(secondary_delimiter)

    self._column_handlers = None
    self._column_names = None
    self._column_arrow_types = None

  def _get_column_handler(
      self, column_info: ColumnInfo
  ) -> Callable[[CSVCell], Optional[Iterable[Union[int, float, bytes, Text]]]]:
    value_converter = _VALUE_CONVERTER_MAP[column_info.type]
    if column_info.name in self._multivalent_columns:
      # If the column is multivalent and unknown, we treat it as a univalent
      # column. This will result in a null array instead of a list<null>", as
      # TFDV does not support list<null>.
      if column_info.type is ColumnType.UNKNOWN:
        return lambda v: None
      return lambda v: [  # pylint: disable=g-long-lambda
          value_converter(sub_v)
          for sub_v in self._multivalent_reader.read_line(v)
      ]
    else:
      return lambda v: (value_converter(v),)

  def _process_column_infos(self, column_infos: List[ColumnInfo]):
    self._column_arrow_types = [_ARROW_TYPE_MAP[c.type] for c in column_infos]
    self._column_handlers = [self._get_column_handler(c) for c in column_infos]
    self._column_names = [c.name for c in column_infos]

  def process(self, batch: List[List[CSVCell]],
              column_infos: List[ColumnInfo]) -> Iterable[pa.RecordBatch]:
    if self._column_names is None:
      self._process_column_infos(column_infos)

    values_list_by_column = [[] for _ in self._column_handlers]
    for csv_row in batch:
      if not csv_row:
        if not self._skip_blank_lines:
          for l in values_list_by_column:
            l.append(None)
        continue
      if len(csv_row) != len(self._column_handlers):
        raise ValueError("Encountered a row of unexpected number of columns")
      for value, handler, values_list in (zip(csv_row, self._column_handlers,
                                              values_list_by_column)):
        values_list.append(handler(value) if value else None)

    arrow_arrays = [
        pa.array(l, type=t)
        for l, t in zip(values_list_by_column, self._column_arrow_types)
    ]
    yield pa.RecordBatch.from_arrays(arrow_arrays, self._column_names)


_VALUE_CONVERTER_MAP = {
    ColumnType.UNKNOWN: lambda x: None,
    ColumnType.INT: int,
    ColumnType.FLOAT: float,
    ColumnType.STRING: lambda x: x,
}

_ARROW_TYPE_MAP = {
    ColumnType.UNKNOWN: pa.null(),
    ColumnType.INT: pa.list_(pa.int64()),
    ColumnType.FLOAT: pa.list_(pa.float32()),
    ColumnType.STRING: pa.list_(pa.binary()),
}


class _CSVRecordReader(object):
  """A picklable wrapper for csv.reader that can decode one record at a time."""

  def __init__(self, delimiter: Union[Text, bytes]):
    self._delimiter = delimiter
    self._line_iterator = _MutableRepeat()
    self._reader = csv.reader(self._line_iterator, delimiter=delimiter)
    # Python 2 csv reader accepts bytes while Python 3 csv reader accepts
    # unicode.
    if six.PY2:
      self._to_reader_input = tf.compat.as_bytes
    else:
      self._to_reader_input = tf.compat.as_text

  def read_line(self, csv_line: CSVLine) -> List[CSVCell]:
    """Reads out bytes for PY2 and Unicode for PY3."""
    self._line_iterator.set_item(self._to_reader_input(csv_line))
    return [tf.compat.as_bytes(cell) for cell in next(self._reader)]

  def __getstate__(self):
    return (self._delimiter,)

  def __setstate__(self, state):
    self.__init__(*state)


class _MutableRepeat(object):
  """Similar to itertools.repeat, but the item can be set on the fly."""

  def __init__(self):
    self._item = None

  def set_item(self, item: Any):
    self._item = item

  def __iter__(self) -> Any:
    return self

  def __next__(self) -> Any:
    return self._item

  next = __next__


_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


def _infer_value_type(value: CSVCell) -> ColumnType:
  """Infer column type from the input value."""
  if not value:
    return ColumnType.UNKNOWN

  # Check if the value is of type INT.
  try:
    if _INT64_MIN <= int(value) <= _INT64_MAX:
      return ColumnType.INT
    # We infer STRING type when we have long integer values.
    return ColumnType.STRING
  except ValueError:
    # If the type is not INT, we next check for FLOAT type (according to our
    # type hierarchy). If we can convert the string to a float value, we
    # fix the type to be FLOAT. Else we resort to STRING type.
    try:
      float(value)
    except ValueError:
      return ColumnType.STRING
    return ColumnType.FLOAT


def _get_feature_types_from_schema(
    schema: schema_pb2.Schema,
    column_names: List[Union[bytes, Text]]) -> List[ColumnInfo]:
  """Get statistics feature types from the input schema."""
  feature_type_map = {}
  for feature in schema.feature:
    feature_type = _SCHEMA_TYPE_TO_COLUMN_TYPE.get(feature.type, None)
    if feature_type is None:
      raise ValueError("Schema contains invalid type: {}.".format(
          schema_pb2.FeatureType.Name(feature.type)))
    feature_type_map[feature.name] = feature_type

  column_infos = []
  for col_name in column_names:
    feature_type = feature_type_map.get(col_name, None)
    if feature_type is None:
      raise ValueError("Schema does not contain column: {}".format(col_name))
    column_infos.append(ColumnInfo(col_name, feature_type))
  return column_infos
