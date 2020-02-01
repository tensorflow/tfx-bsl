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
import six
import tensorflow as tf
from typing import Any, Dict, List, Text, Union

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


@beam.typehints.with_input_types(CSVLine)
@beam.typehints.with_output_types(beam.typehints.List[CSVCell])
class ParseCSVLine(beam.DoFn):
  """A beam.DoFn to parse CSVLines into List[CSVCell]."""

  __slots__ = ["_delimiter", "_reader"]

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

  def __init__(self, column_names: List[ColumnName],
               skip_blank_lines: bool) -> None:
    """Initializes a feature type inferrer combiner."""
    self._column_names = column_names
    self._skip_blank_lines = skip_blank_lines

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
      # Infer the type from the current feature value.
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


class _CSVRecordReader(object):
  """A picklable wrapper for csv.reader that can decode one record at a time."""

  __slots__ = ["_delimiter", "_line_iterator", "_reader", "_to_reader_input"]

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
