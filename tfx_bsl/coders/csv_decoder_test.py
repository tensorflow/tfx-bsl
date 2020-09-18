# coding=utf-8
#
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

"""Tests for CSV decoder."""

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
import pyarrow as pa
from tfx_bsl.coders import csv_decoder
from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2

_TEST_CASES = [
    dict(
        testcase_name='simple',
        input_lines=['1,2.0,hello', '5,12.34,world'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_csv_cells=[
            [b'1', b'2.0', b'hello'],
            [b'5', b'12.34', b'world'],
        ],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
            pa.array([[b'hello'], [b'world']], pa.large_list(pa.large_binary()))
        ], ['int_feature', 'float_feature', 'str_feature'])),
    dict(
        testcase_name='missing_values',
        input_lines=[',,', '1,,hello', ',12.34,'],
        column_names=['f1', 'f2', 'f3'],
        expected_csv_cells=[
            [b'', b'', b''],
            [b'1', b'', b'hello'],
            [b'', b'12.34', b''],
        ],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([None, [1], None], pa.large_list(pa.int64())),
            pa.array([None, None, [12.34]], pa.large_list(pa.float32())),
            pa.array([None, [b'hello'], None], pa.large_list(
                pa.large_binary())),
        ], ['f1', 'f2', 'f3'])),
    dict(
        testcase_name='mixed_int_and_float',
        input_lines=['2,1.5', '1.5,2'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2', b'1.5'], [b'1.5', b'2']],
        expected_types=[
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.FLOAT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[2], [1.5]], pa.large_list(pa.float32())),
            pa.array([[1.5], [2]], pa.large_list(pa.float32()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='mixed_int_and_string',
        input_lines=['2,abc', 'abc,2'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2', b'abc'], [b'abc', b'2']],
        expected_types=[
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'2'], [b'abc']], pa.large_list(pa.large_binary())),
            pa.array([[b'abc'], [b'2']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='mixed_float_and_string',
        input_lines=['2.3,abc', 'abc,2.3'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2.3', b'abc'], [b'abc', b'2.3']],
        expected_types=[
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'2.3'], [b'abc']], pa.large_list(pa.large_binary())),
            pa.array([[b'abc'], [b'2.3']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='unicode',
        input_lines=[u'\U0001f951'],
        column_names=['f1'],
        expected_csv_cells=[[u'\U0001f951'.encode('utf-8')]],
        expected_types=[
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[u'\U0001f951'.encode('utf-8')]],
                     pa.large_list(pa.large_binary()))
        ], ['f1'])),
    dict(
        testcase_name='quotes',
        input_lines=['1,"ab,cd,ef"', '5,"wx,xy,yz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1', b'ab,cd,ef'], [b'5', b'wx,xy,yz']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']],
                     pa.large_list(pa.large_binary()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='space_delimiter',
        input_lines=['1 "ab,cd,ef"', '5 "wx,xy,yz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1', b'ab,cd,ef'], [b'5', b'wx,xy,yz']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[b'ab,cd,ef'], [b'wx,xy,yz']],
                     pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        delimiter=' '),
    dict(
        testcase_name='tab_delimiter',
        input_lines=['1\t"this is a \ttext"', '5\t'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1', b'this is a \ttext'], [b'5', b'']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[b'this is a \ttext'], None],
                     pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        delimiter='\t'),
    dict(
        testcase_name='negative_values',
        input_lines=['-1,-2.5'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'-1', b'-2.5']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[-1]], pa.large_list(pa.int64())),
            pa.array([[-2.5]], pa.large_list(pa.float32()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='int64_boundary',
        input_lines=[
            '%s,%s,%s,%s' % (
                str(np.iinfo(np.int64).min),
                str(np.iinfo(np.int64).max),
                str(np.iinfo(np.int64).min - 1),
                str(np.iinfo(np.int64).max + 1),
            )
        ],
        column_names=['int64min', 'int64max', 'int64min-1', 'int64max+1'],
        expected_csv_cells=[[
            b'-9223372036854775808', b'9223372036854775807',
            b'-9223372036854775809', b'9223372036854775808'
        ]],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[-9223372036854775808]], pa.large_list(pa.int64())),
            pa.array([[9223372036854775807]], pa.large_list(pa.int64())),
            pa.array([[b'-9223372036854775809']],
                     pa.large_list(pa.large_binary())),
            pa.array([[b'9223372036854775808']], pa.large_list(
                pa.large_binary()))
        ], ['int64min', 'int64max', 'int64min-1', 'int64max+1'])),
    dict(
        testcase_name='skip_blank_lines',
        input_lines=['', '1,2'],
        skip_blank_lines=True,
        column_names=['f1', 'f2'],
        expected_csv_cells=[[], [b'1', b'2']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.INT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1]], pa.large_list(pa.int64())),
            pa.array([[2]], pa.large_list(pa.int64()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='consider_blank_lines',
        input_lines=['', '1,2'],
        skip_blank_lines=False,
        column_names=['f1', 'f2'],
        expected_csv_cells=[[], [b'1', b'2']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.INT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([None, [1]], pa.large_list(pa.int64())),
            pa.array([None, [2]], pa.large_list(pa.int64()))
        ], ['f1', 'f2'])),
    dict(
        testcase_name='skip_blank_lines_single_column',
        input_lines=['', '1'],
        skip_blank_lines=True,
        column_names=['f1'],
        expected_csv_cells=[[], [b'1']],
        expected_types=[
            csv_decoder.ColumnType.INT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays(
            [pa.array([[1]], pa.large_list(pa.int64()))], ['f1'])),
    dict(
        testcase_name='consider_blank_lines_single_column',
        input_lines=['', '1'],
        skip_blank_lines=False,
        column_names=['f1'],
        expected_csv_cells=[[], [b'1']],
        expected_types=[
            csv_decoder.ColumnType.INT,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays(
            [pa.array([None, [1]], pa.large_list(pa.int64()))], ['f1'])),
    dict(
        testcase_name='empty_csv',
        input_lines=[],
        column_names=['f1'],
        expected_csv_cells=[],
        expected_types=[csv_decoder.ColumnType.UNKNOWN],
        expected_record_batch=[],
    ),
    dict(
        testcase_name='null_column',
        input_lines=['', ''],
        column_names=['f1'],
        expected_csv_cells=[[], []],
        expected_types=[csv_decoder.ColumnType.UNKNOWN],
        expected_record_batch=pa.RecordBatch.from_arrays(
            [pa.array([None, None], pa.null())], ['f1'])),
    dict(
        testcase_name='size_2_vector_int_multivalent',
        input_lines=['12|14'],
        column_names=['x'],
        expected_csv_cells=[[b'12|14']],
        expected_types=[csv_decoder.ColumnType.INT],
        expected_record_batch=pa.RecordBatch.from_arrays(
            [pa.array([[12, 14]], pa.large_list(pa.int64()))], ['x']),
        delimiter=' ',
        multivalent_columns=['x'],
        secondary_delimiter='|'),
    dict(
        testcase_name='space_and_comma_delimiter',
        input_lines=['1,2 "abcdef"', '5,1 "wxxyyz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1,2', b'abcdef'], [b'5,1', b'wxxyyz']],
        expected_types=[
            csv_decoder.ColumnType.INT, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1, 2], [5, 1]], pa.large_list(pa.int64())),
            pa.array([[b'abcdef'], [b'wxxyyz']], pa.large_list(
                pa.large_binary()))
        ], ['f1', 'f2']),
        delimiter=' ',
        multivalent_columns=['f1'],
        secondary_delimiter=','),
    dict(
        testcase_name='empty_multivalent_column',
        input_lines=[',test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'', b'test']],
        expected_types=[
            csv_decoder.ColumnType.UNKNOWN, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([None], pa.null()),
            pa.array([[b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='empty_values_multivalent_column',
        input_lines=['|,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'|', b'test']],
        expected_types=[
            csv_decoder.ColumnType.UNKNOWN, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([None], pa.null()),
            pa.array([[b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='empty_string_multivalent_column',
        input_lines=['|,test', 'a|b,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'|', b'test'], [b'a|b', b'test']],
        expected_types=[
            csv_decoder.ColumnType.STRING, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'', b''], [b'a', b'b']], pa.large_list(
                pa.large_binary())),
            pa.array([[b'test'], [b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='int_and_float_multivalent_column',
        input_lines=['1|2.3,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1|2.3', b'test']],
        expected_types=[
            csv_decoder.ColumnType.FLOAT, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1, 2.3]], pa.large_list(pa.float32())),
            pa.array([[b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='float_and_string_multivalent_column',
        input_lines=['2.3|abc,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2.3|abc', b'test']],
        expected_types=[
            csv_decoder.ColumnType.STRING, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'2.3', b'abc']], pa.large_list(pa.large_binary())),
            pa.array([[b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='int_and_string_multivalent_column',
        input_lines=['1|abc,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1|abc', b'test']],
        expected_types=[
            csv_decoder.ColumnType.STRING, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'1', b'abc']], pa.large_list(pa.large_binary())),
            pa.array([[b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='int_and_string_multivalent_column_multiple_lines',
        input_lines=['1|abc,test', '2|2,test'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1|abc', b'test'], [b'2|2', b'test']],
        expected_types=[
            csv_decoder.ColumnType.STRING, csv_decoder.ColumnType.STRING
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b'1', b'abc'], [b'2', b'2']],
                     pa.large_list(pa.large_binary())),
            pa.array([[b'test'], [b'test']], pa.large_list(pa.large_binary()))
        ], ['f1', 'f2']),
        multivalent_columns=['f1'],
        secondary_delimiter='|'),
    dict(
        testcase_name='with_schema',
        input_lines=['1,2.0,hello', '5,12.34,world'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_csv_cells=[
            [b'1', b'2.0', b'hello'],
            [b'5', b'12.34', b'world'],
        ],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.STRING,
        ],
        schema=text_format.Parse(
            """feature {
                name: "int_feature"
                type: INT
              }
              feature {
                name: "float_feature"
                type: FLOAT
              }
              feature {
                name: "str_feature"
                type: BYTES
              }
              """, schema_pb2.Schema()),
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
            pa.array([[b'hello'], [b'world']],
                     pa.large_list(pa.large_binary()))
        ], ['int_feature', 'float_feature', 'str_feature'])),
    dict(
        testcase_name='attach_raw_records',
        input_lines=['1,2.0,hello', '5,12.34,world'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_csv_cells=[
            [b'1', b'2.0', b'hello'],
            [b'5', b'12.34', b'world'],
        ],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
            pa.array([[b'hello'], [b'world']], pa.large_list(
                pa.large_binary())),
            pa.array([[b'1,2.0,hello'], [b'5,12.34,world']],
                     pa.large_list(pa.large_binary()))
        ], ['int_feature', 'float_feature', 'str_feature', 'raw_records']),
        raw_record_column_name='raw_records'),
    dict(
        testcase_name='with_schema_attach_raw_record',
        input_lines=['1,2.0,hello', '5,12.34,world'],
        column_names=['int_feature', 'float_feature', 'str_feature'],
        expected_csv_cells=[
            [b'1', b'2.0', b'hello'],
            [b'5', b'12.34', b'world'],
        ],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.STRING,
        ],
        schema=text_format.Parse(
            """feature {
                name: "int_feature"
                type: INT
              }
              feature {
                name: "float_feature"
                type: FLOAT
              }
              feature {
                name: "str_feature"
                type: BYTES
              }
              """, schema_pb2.Schema()),
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1], [5]], pa.large_list(pa.int64())),
            pa.array([[2.0], [12.34]], pa.large_list(pa.float32())),
            pa.array([[b'hello'], [b'world']],
                     pa.large_list(pa.large_binary())),
            pa.array([[b'1,2.0,hello'], [b'5,12.34,world']],
                     pa.large_list(pa.large_binary()))
        ], ['int_feature', 'float_feature', 'str_feature', 'raw_records'
           ]),
        raw_record_column_name='raw_records'),
    dict(
        testcase_name='multivalent_attach_raw_records',
        input_lines=['1,2 "abcdef"', '5,1 "wxxyyz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1,2', b'abcdef'], [b'5,1', b'wxxyyz']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ],
        expected_record_batch=pa.RecordBatch.from_arrays([
            pa.array([[1, 2], [5, 1]], pa.large_list(pa.int64())),
            pa.array([[b'abcdef'], [b'wxxyyz']], pa.large_list(
                pa.large_binary())),
            pa.array([[b'1,2 "abcdef"'], [b'5,1 "wxxyyz"']],
                     pa.large_list(pa.large_binary()))
        ], ['f1', 'f2', 'raw_records']),
        delimiter=' ',
        multivalent_columns=['f1'],
        secondary_delimiter=',',
        raw_record_column_name='raw_records'),
]


class CSVDecoderTest(parameterized.TestCase):
  """Tests for CSV decoder."""

  @parameterized.named_parameters(_TEST_CASES)
  def test_parse_csv_lines(self,
                           input_lines,
                           column_names,
                           expected_csv_cells,
                           expected_types,
                           expected_record_batch,
                           skip_blank_lines=False,
                           schema=None,
                           delimiter=',',
                           multivalent_columns=None,
                           secondary_delimiter=None,
                           raw_record_column_name=None):

    def _check_csv_cells(actual):
      for i in range(len(actual)):
        self.assertEqual(expected_csv_cells[i], actual[i][0])
        self.assertEqual(input_lines[i], actual[i][1])

    def _check_types(actual):
      self.assertLen(actual, 1)
      self.assertCountEqual([
          csv_decoder.ColumnInfo(n, t)
          for n, t in zip(column_names, expected_types)
      ], actual[0])

    def _check_record_batches(actual):
      """Compares a list of pa.RecordBatch."""
      if actual:
        self.assertTrue(actual[0].equals(expected_record_batch))
      else:
        self.assertEqual(expected_record_batch, actual)

    def _check_arrow_schema(actual):
      for record_batch in actual:
        expected_arrow_schema = csv_decoder.GetArrowSchema(
            column_names, schema, raw_record_column_name)
        self.assertEqual(record_batch.schema, expected_arrow_schema)

    with beam.Pipeline() as p:
      parsed_csv_cells_and_raw_records = (
          p | beam.Create(input_lines, reshuffle=False)
          | beam.ParDo(csv_decoder.ParseCSVLine(delimiter=delimiter)))
      inferred_types = (
          parsed_csv_cells_and_raw_records
          | beam.Keys()
          | beam.CombineGlobally(
              csv_decoder.ColumnTypeInferrer(
                  column_names,
                  skip_blank_lines=skip_blank_lines,
                  multivalent_columns=multivalent_columns,
                  secondary_delimiter=secondary_delimiter)))

      beam_test_util.assert_that(
          parsed_csv_cells_and_raw_records,
          _check_csv_cells,
          label='check_parsed_csv_cells')
      beam_test_util.assert_that(
          inferred_types, _check_types, label='check_types')

      record_batches = (
          parsed_csv_cells_and_raw_records
          | beam.BatchElements(min_batch_size=1000) | beam.ParDo(
              csv_decoder.BatchedCSVRowsToRecordBatch(
                  skip_blank_lines=skip_blank_lines,
                  multivalent_columns=multivalent_columns,
                  secondary_delimiter=secondary_delimiter,
                  raw_record_column_name=raw_record_column_name),
              beam.pvalue.AsSingleton(inferred_types)))
      beam_test_util.assert_that(
          record_batches, _check_record_batches, label='check_record_batches')
      if schema:
        beam_test_util.assert_that(
            record_batches, _check_arrow_schema, label='check_arrow_schema')

    # Testing CSVToRecordBatch
    with beam.Pipeline() as p:
      record_batches = (
          p | 'CreatingPColl' >> beam.Create(input_lines, reshuffle=False)
          | 'CSVToRecordBatch' >> csv_decoder.CSVToRecordBatch(
              column_names=column_names,
              delimiter=delimiter,
              skip_blank_lines=skip_blank_lines,
              desired_batch_size=1000,
              schema=schema,
              multivalent_columns=multivalent_columns,
              secondary_delimiter=secondary_delimiter,
              raw_record_column_name=raw_record_column_name))
      beam_test_util.assert_that(
          record_batches, _check_record_batches, label='check_record_batches')

  def test_csv_to_recordbatch_schema_features_subset_of_column_names(self):
    input_lines = ['1,2.0,hello', '5,12.34,world']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    schema = text_format.Parse("""feature { name: "int_feature" type: INT }""",
                               schema_pb2.Schema())
    self.assertEqual(
        csv_decoder.GetArrowSchema(column_names, schema),
        pa.schema([pa.field('int_feature', pa.large_list(pa.int64()))]))

    def _check_record_batches(record_batches):
      self.assertLen(record_batches, 1)
      self.assertTrue(record_batches[0].equals(
          pa.RecordBatch.from_arrays(
              [pa.array([[1], [5]], pa.large_list(pa.int64()))],
              ['int_feature'])))

    with beam.Pipeline() as p:
      record_batches = (
          p | 'CreatingPColl' >> beam.Create(input_lines, reshuffle=False)
          | 'CSVToRecordBatch' >> csv_decoder.CSVToRecordBatch(
              column_names=column_names,
              delimiter=',',
              desired_batch_size=1000,
              schema=schema))
      beam_test_util.assert_that(
          record_batches, _check_record_batches, label='check_record_batches')

  def test_invalid_row(self):
    input_lines = ['1,2.0,hello', '5,12.34']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, '.*Columns do not match specified csv headers.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | beam.ParDo(csv_decoder.ParseCSVLine(delimiter=','))
            | beam.Keys()
            | beam.CombineGlobally(
                csv_decoder.ColumnTypeInferrer(
                    column_names, skip_blank_lines=False)))
        beam_test_util.assert_that(result, lambda _: None)

  def test_invalid_schema_type(self):
    input_lines = ['1']
    column_names = ['f1']
    schema = text_format.Parse(
        """
              feature {
                name: "struct_feature"
                type: STRUCT
              }
              """, schema_pb2.Schema())
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, '.*Schema contains invalid type: STRUCT.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | 'CSVToRecordBatch' >> csv_decoder.CSVToRecordBatch(
                column_names=column_names,
                schema=schema,
                desired_batch_size=1000))
        beam_test_util.assert_that(result, lambda _: None)

  def test_invalid_raw_record_column_name(self):
    input_lines = ['1,2.0,hello', '5,12.34']
    schema = text_format.Parse(
        """
              feature {
                name: "int_feature"
                type: INT
              }
              feature {
                name: "float_feature"
                type: FLOAT
              }
              feature {
                name: "str_feature"
                type: BYTES
              }
              """, schema_pb2.Schema())
    column_names = ['int_feature', 'float_feature', 'str_feature']
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError, 'raw_record_column_name.* is already an existing column.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | 'CSVToRecordBatch' >> csv_decoder.CSVToRecordBatch(
                column_names=column_names,
                desired_batch_size=1000,
                raw_record_column_name='int_feature'))
        beam_test_util.assert_that(result, lambda _: None)
    with self.assertRaisesRegex(
        ValueError, 'raw_record_column_name.* is already an existing column.*'):
      csv_decoder.GetArrowSchema(
          column_names, schema, raw_record_column_name='int_feature')

  def test_get_arrow_schema_schema_feature_not_subset_of_column_names(self):
    schema = text_format.Parse(
        """
              feature {
                name: "f1"
                type: INT
              }
              feature {
                name: "f2"
                type: INT
              }
              """, schema_pb2.Schema())
    column_names = ['f1']
    with self.assertRaisesRegex(
        ValueError, 'Schema features are not a subset of column names'):
      csv_decoder.GetArrowSchema(column_names, schema)


if __name__ == '__main__':
  absltest.main()
