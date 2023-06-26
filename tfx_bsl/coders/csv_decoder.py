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

import csv
import enum
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Text, Tuple, Union

import apache_beam as beam
import numpy as np
import pyarrow as pa
from tfx_bsl.coders import batch_util

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

PARSE_CSV_LINE_YIELDS_RAW_RECORDS = True

CSVCell = bytes
CSVLine = Text
ColumnName = Text


class ColumnType(enum.IntEnum):
  """Enum for the type of a CSV column."""
  # column will not be in the result RecordBatch
  IGNORE = -2
  # column will be in the result RecordBatch but will be of Null type (which
  # means this column contains only empty value).
  UNKNOWN = -1
  INT = statistics_pb2.FeatureNameStatistics.INT
  FLOAT = statistics_pb2.FeatureNameStatistics.FLOAT
  STRING = statistics_pb2.FeatureNameStatistics.STRING

  # We need the following to hold for type inference to work.
  assert UNKNOWN < INT
  assert INT < FLOAT
  assert FLOAT < STRING


ColumnInfo = NamedTuple("ColumnInfo", [
    ("name", ColumnName),
    ("type", ColumnType),
])

_SCHEMA_TYPE_TO_COLUMN_TYPE = {
    schema_pb2.INT: ColumnType.INT,
    schema_pb2.FLOAT: ColumnType.FLOAT,
    schema_pb2.BYTES: ColumnType.STRING
}

_FEATURE_TYPE_TO_ARROW_TYPE = {
        ColumnType.UNKNOWN: pa.null(),
        ColumnType.INT: pa.large_list(pa.int64()),
        ColumnType.FLOAT: pa.large_list(pa.float32()),
        ColumnType.STRING: pa.large_list(pa.large_binary())
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
                     multivalent_columns: Optional[List[Text]] = None,
                     secondary_delimiter: Optional[Text] = None,
                     raw_record_column_name: Optional[Text] = None):
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
      contain a subset of columns in `column_names`. If a feature is in
      `column_names` but not in the schema, it won't be in the result
      RecordBatch.
    multivalent_columns: Columns that can contain multiple values. If
      secondary_delimiter is provided, this must also be provided.
    secondary_delimiter: Delimiter used for parsing multivalent columns. If
      multivalent_columns is provided, this must also be provided.
    raw_record_column_name: Optional name for a column containing the raw csv
      lines. If this is None, then this column will not be produced. This will
      always be the last column in the record batch.

  Returns:
    RecordBatches of the CSV lines.

  Raises:
    ValueError:
      * If the columns do not match the specified csv headers.
      * If the schema has invalid feature types.
      * If the schema does not contain all columns.
      * If raw_record_column_name exists in column_names
  """
  if (raw_record_column_name is not None and
      raw_record_column_name in column_names):
    raise ValueError(
        "raw_record_column_name: {} is already an existing column name. "
        "Please choose a different name.".format(raw_record_column_name))

  csv_lines_and_raw_records = (
      lines | "ParseCSVLines" >> beam.ParDo(ParseCSVLine(delimiter)))

  if schema is not None:
    column_infos = _GetColumnInfosFromSchema(schema, column_names)
  else:
    # TODO(b/72746442): Consider using a DeepCopy optimization similar to TFT.
    # Do first pass to infer the feature types.
    column_infos = beam.pvalue.AsSingleton(
        csv_lines_and_raw_records
        | "ExtractParsedCSVLines" >> beam.Keys()
        | "InferColumnTypes" >> beam.CombineGlobally(
            ColumnTypeInferrer(
                column_names=column_names,
                skip_blank_lines=skip_blank_lines,
                multivalent_columns=multivalent_columns,
                secondary_delimiter=secondary_delimiter)))

  # Do second pass to generate the RecordBatches.
  return (
      csv_lines_and_raw_records
      | "BatchCSVLines"
      >> batch_util.BatchRecords(
          desired_batch_size,
          telemetry_descriptors=["CSVToRecordBatch"],
          # The elements are tuples of parsed and unparsed CSVlines.
          record_size_fn=lambda kv: len(kv[1]) << 1,
      )
      | "BatchedCSVRowsToArrow"
      >> beam.ParDo(
          BatchedCSVRowsToRecordBatch(
              skip_blank_lines=skip_blank_lines,
              multivalent_columns=multivalent_columns,
              secondary_delimiter=secondary_delimiter,
              raw_record_column_name=raw_record_column_name,
          ),
          column_infos,
      )
  )


@beam.typehints.with_input_types(CSVLine)
@beam.typehints.with_output_types(Tuple[List[CSVCell], CSVLine])
class ParseCSVLine(beam.DoFn):
  """A beam.DoFn to parse CSVLines into Tuple(List[CSVCell], CSVLine).

  The CSVLine is the raw csv row. The raw csv row will always be output.
  """

  def __init__(self, delimiter: Text):
    self._delimiter = delimiter
    self._reader = None

  def setup(self):
    self._reader = _CSVRecordReader(self._delimiter)

  def process(self,
              csv_line: CSVLine) -> Iterable[Tuple[List[CSVCell], CSVLine]]:
    assert self._reader is not None, "Reader uninitialized. Call setup() first."
    line = self._reader.ReadLine(csv_line)
    yield (line, csv_line)


@beam.typehints.with_input_types(List[CSVCell])
@beam.typehints.with_output_types(List[ColumnInfo])
class ColumnTypeInferrer(beam.CombineFn):
  """A beam.CombineFn to infer CSV Column types.

  Its input can be produced by ParseCSVLine().
  """

  def __init__(
      self,
      column_names: List[ColumnName],
      skip_blank_lines: bool,
      multivalent_columns: Optional[Set[ColumnName]] = None,
      secondary_delimiter: Optional[Text] = None) -> None:
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
        # the reader only accepts str but v is bytes.
        values = self._multivalent_reader.ReadLine(cell.decode())  # pytype: disable=attribute-error  # trace-all-classes
        current_type = max([_InferValueType(value) for value in values
                           ]) if values else ColumnType.UNKNOWN
      else:
        current_type = _InferValueType(cell)

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
      for feature_name, feature_type in shard_types.items():
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


@beam.typehints.with_input_types(
    List[Tuple[List[CSVCell], CSVLine]],
    List[ColumnInfo])
@beam.typehints.with_output_types(pa.RecordBatch)
class BatchedCSVRowsToRecordBatch(beam.DoFn):
  """DoFn to convert a batch of csv rows to a RecordBatch."""

  def __init__(self,
               skip_blank_lines: bool,
               multivalent_columns: Optional[Set[ColumnName]] = None,
               secondary_delimiter: Optional[Text] = None,
               raw_record_column_name: Optional[Text] = None):
    self._skip_blank_lines = skip_blank_lines
    self._multivalent_columns = (
        multivalent_columns if multivalent_columns is not None else set())
    if multivalent_columns:
      assert secondary_delimiter, ("secondary_delimiter must be specified if "
                                   "there are multivalent columns")
      self._multivalent_reader = _CSVRecordReader(secondary_delimiter)
    self._raw_record_column_name = raw_record_column_name
    self._raw_record_column_type = _FEATURE_TYPE_TO_ARROW_TYPE.get(
        ColumnType.STRING)

    # Note that len(_column_handlers) == len(column_infos) but
    # len(_column_names) and len(_column_arrow_types) may not equal to that,
    # because columns of type IGNORE are not there.
    self._column_handlers = None
    self._column_names = []
    self._column_arrow_types = None

  def _get_column_handler(
      self, column_info: ColumnInfo
  ) -> Optional[Callable[[CSVCell], Optional[Iterable[Union[int, float,
                                                            bytes]]]]]:
    if column_info.type == ColumnType.IGNORE:
      return None
    value_converter = _VALUE_CONVERTER_MAP.get(column_info.type)
    assert value_converter is not None
    if column_info.name in self._multivalent_columns:
      # If the column is multivalent and unknown, we treat it as a univalent
      # column. This will result in a null array instead of a list<null>", as
      # TFDV does not support list<null>.
      if column_info.type is ColumnType.UNKNOWN:
        return lambda v: None
      return lambda v: [  # pylint: disable=g-long-lambda
          value_converter(sub_v)
          # the reader only accepts str but v is bytes.
          for sub_v in self._multivalent_reader.ReadLine(v.decode())
      ]
    else:
      return lambda v: (value_converter(v),)

  def _process_column_infos(self, column_infos: List[ColumnInfo]):
    self._column_handlers = [self._get_column_handler(c) for c in column_infos]
    self._column_arrow_types = [
        _FEATURE_TYPE_TO_ARROW_TYPE.get(c.type)
        for c in column_infos
        if c.type != ColumnType.IGNORE
    ]
    self._column_names = [
        c.name for c in column_infos if c.type != ColumnType.IGNORE]

  def process(self, batch_of_tuple: List[Tuple[List[CSVCell], CSVLine]],
              column_infos: List[ColumnInfo]) -> Iterable[pa.RecordBatch]:
    if self._column_handlers is None:
      self._process_column_infos(column_infos)

    raw_records = []
    values_list_by_column = [[] for _ in self._column_names]
    for (csv_row, raw_record) in batch_of_tuple:
      if not csv_row:
        if not self._skip_blank_lines:
          for l in values_list_by_column:
            l.append(None)
        continue
      if len(csv_row) != len(self._column_handlers):
        raise ValueError(
            "Encountered a row of unexpected number of columns: {} vs. {}"
            .format(len(csv_row), len(self._column_handlers)))
      column_idx = 0
      for csv_cell, handler in zip(csv_row, self._column_handlers):
        if handler is None:
          continue
        values_list_by_column[column_idx].append(
            handler(csv_cell) if csv_cell else None)
        column_idx += 1
      if self._raw_record_column_name is not None:
        raw_records.append([raw_record])

    arrow_arrays = [
        pa.array(l, type=t)
        for l, t in zip(values_list_by_column, self._column_arrow_types)
    ]

    if self._raw_record_column_name is not None:
      arrow_arrays.append(
          pa.array(raw_records, type=self._raw_record_column_type))
      self._column_names.append(self._raw_record_column_name)
    yield pa.RecordBatch.from_arrays(arrow_arrays, self._column_names)


_VALUE_CONVERTER_MAP = {
    ColumnType.UNKNOWN: lambda x: None,
    ColumnType.INT: int,
    ColumnType.FLOAT: float,
    ColumnType.STRING: lambda x: x,
}


def GetArrowSchema(column_names: List[Text],
                   schema: schema_pb2.Schema,
                   raw_record_column_name: Optional[Text] = None) -> pa.Schema:
  """Returns the arrow schema given columns and a TFMD schema.

  Args:
    column_names: List of feature names. This must match the features in schema.
    schema: The schema proto to base the arrow schema from.
    raw_record_column_name: An optional name for the column containing raw
     records. If this is not set, the arrow schema will not contain a raw
     records column.

  Returns:
    Arrow Schema based on the provided schema proto.

  Raises:
    ValueError:
     * If the schema contains a feature that does not exist in `column_names`.
     * If the feature type does not map to an arrow type.
     * If raw_record_column_name exists in column_names
  """
  schema_feature_names = [f.name for f in schema.feature]
  if not set(schema_feature_names).issubset(set(column_names)):
    raise ValueError(
        "Schema features are not a subset of column names: {} vs {}".format(
            schema_feature_names, column_names))

  fields = []
  column_name_to_schema_feature_map = {f.name: f for f in schema.feature}
  for col in column_names:
    feature = column_name_to_schema_feature_map.get(col)
    if feature is None:
      continue
    arrow_type = _FEATURE_TYPE_TO_ARROW_TYPE.get(
        _SCHEMA_TYPE_TO_COLUMN_TYPE.get(feature.type), None)
    if arrow_type is None:
      raise ValueError("Feature {} has unsupport type {}".format(
          feature.name, feature.type))
    fields.append(pa.field(feature.name, arrow_type))

  if raw_record_column_name is not None:
    if raw_record_column_name in column_names:
      raise ValueError(
          "raw_record_column_name: {} is already an existing column name. "
          "Please choose a different name.".format(raw_record_column_name))
    raw_record_type = _FEATURE_TYPE_TO_ARROW_TYPE.get(
        ColumnType.STRING)
    fields.append(pa.field(raw_record_column_name, raw_record_type))
  return pa.schema(fields)


class _CSVRecordReader(object):
  """A picklable wrapper for csv.reader that can parse one record at a time."""

  def __init__(self, delimiter: Text):
    self._delimiter = delimiter
    self._line_iterator = _MutableRepeat()
    self._reader = csv.reader(self._line_iterator, delimiter=delimiter)

  def ReadLine(self, csv_line: CSVLine) -> List[CSVCell]:
    """Reads out bytes for PY2 and Unicode for PY3."""
    self._line_iterator.SetItem(csv_line)
    return [cell.encode() for cell in next(self._reader)]

  def __getstate__(self):
    return (self._delimiter,)

  def __setstate__(self, state):
    self.__init__(*state)


class _MutableRepeat(object):
  """Similar to itertools.repeat, but the item can be set on the fly."""

  def __init__(self):
    self._item = None

  def SetItem(self, item: Any):
    self._item = item

  def __iter__(self) -> Any:
    return self

  def __next__(self) -> Any:
    return self._item

  next = __next__


_INT64_MIN = np.iinfo(np.int64).min
_INT64_MAX = np.iinfo(np.int64).max


def _InferValueType(value: CSVCell) -> ColumnType:
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


def _GetColumnInfosFromSchema(
    schema: schema_pb2.Schema,
    column_names: List[Text]) -> List[ColumnInfo]:
  """Get column name and type from the input schema."""
  feature_type_map = {}
  for feature in schema.feature:
    feature_type = _SCHEMA_TYPE_TO_COLUMN_TYPE.get(feature.type, None)
    if feature_type is None:
      raise ValueError("Schema contains invalid type: {}.".format(
          schema_pb2.FeatureType.Name(feature.type)))
    feature_type_map[feature.name] = feature_type

  column_infos = []
  for col_name in column_names:
    feature_type = feature_type_map.get(col_name, ColumnType.IGNORE)
    column_infos.append(ColumnInfo(col_name, feature_type))
  return column_infos
