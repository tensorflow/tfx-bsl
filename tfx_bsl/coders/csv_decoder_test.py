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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import numpy as np
from tfx_bsl.coders import csv_decoder
from absl.testing import absltest
from absl.testing import parameterized


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
        ]),
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
        ]),
    dict(
        testcase_name='mixed_int_and_float',
        input_lines=['2,1.5', '1.5,2'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2', b'1.5'], [b'1.5', b'2']],
        expected_types=[
            csv_decoder.ColumnType.FLOAT,
            csv_decoder.ColumnType.FLOAT,
        ],
    ),
    dict(
        testcase_name='mixed_int_and_string',
        input_lines=['2,abc', 'abc,2'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2', b'abc'], [b'abc', b'2']],
        expected_types=[
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ],
    ),
    dict(
        testcase_name='mixed_float_and_string',
        input_lines=['2.3,abc', 'abc,2.3'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'2.3', b'abc'], [b'abc', b'2.3']],
        expected_types=[
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ],
    ),
    dict(
        testcase_name='unicode',
        input_lines=[u'\U0001f951'],
        column_names=['f1'],
        expected_csv_cells=[[u'\U0001f951'.encode('utf-8')]],
        expected_types=[
            csv_decoder.ColumnType.STRING,
        ]),
    dict(
        testcase_name='quotes',
        input_lines=['1,"ab,cd,ef"', '5,"wx,xy,yz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1', b'ab,cd,ef'], [b'5', b'wx,xy,yz']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ]),
    dict(
        testcase_name='space_delimiter',
        input_lines=['1 "ab,cd,ef"', '5 "wx,xy,yz"'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'1', b'ab,cd,ef'], [b'5', b'wx,xy,yz']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
        ],
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
        delimiter='\t'),
    dict(
        testcase_name='negative_values',
        input_lines=['-1,-2.5'],
        column_names=['f1', 'f2'],
        expected_csv_cells=[[b'-1', b'-2.5']],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.FLOAT,
        ]),
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
        column_names=['int64max', 'int64min', 'int64max+1', 'int64min-1'],
        expected_csv_cells=[[
            b'-9223372036854775808', b'9223372036854775807',
            b'-9223372036854775809', b'9223372036854775808'
        ]],
        expected_types=[
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.INT,
            csv_decoder.ColumnType.STRING,
            csv_decoder.ColumnType.STRING,
        ]),
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
    ),
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
    ),
    dict(
        testcase_name='skip_blank_lines_single_column',
        input_lines=['', '1'],
        skip_blank_lines=True,
        column_names=['f1'],
        expected_csv_cells=[[], [b'1']],
        expected_types=[
            csv_decoder.ColumnType.INT,
        ],
    ),
    dict(
        testcase_name='consider_blank_lines_single_column',
        input_lines=['', '1'],
        skip_blank_lines=False,
        column_names=['f1'],
        expected_csv_cells=[[], [b'1']],
        expected_types=[
            csv_decoder.ColumnType.INT,
        ],
    ),
    dict(
        testcase_name='empty_csv',
        input_lines=[],
        column_names=['f1'],
        expected_csv_cells=[],
        expected_types=[csv_decoder.ColumnType.UNKNOWN],
    ),
    dict(
        testcase_name='null_column',
        input_lines=['', ''],
        column_names=['f1'],
        expected_csv_cells=[[], []],
        expected_types=[csv_decoder.ColumnType.UNKNOWN],
    )
]


class CSVDecoderTest(parameterized.TestCase):
  """Tests for CSV decoder."""

  @parameterized.named_parameters(_TEST_CASES)
  def test_parse_csv_lines(self,
                           input_lines,
                           column_names,
                           expected_csv_cells,
                           expected_types,
                           skip_blank_lines=False,
                           delimiter=','):

    def _check_csv_cells(actual):
      self.assertEqual(expected_csv_cells, actual)

    def _check_types(actual):
      self.assertLen(actual, 1)
      self.assertCountEqual([
          csv_decoder.ColumnInfo(n, t)
          for n, t in zip(column_names, expected_types)
      ], actual[0])

    with beam.Pipeline() as p:
      parsed_csv_cells = (
          p | beam.Create(input_lines, reshuffle=False)
          | beam.ParDo(csv_decoder.ParseCSVLine(delimiter=delimiter)))
      inferred_types = parsed_csv_cells | beam.CombineGlobally(
          csv_decoder.ColumnTypeInferrer(
              column_names, skip_blank_lines=skip_blank_lines))

      beam_test_util.assert_that(
          parsed_csv_cells, _check_csv_cells, label='check_parsed_csv_cells')
      beam_test_util.assert_that(
          inferred_types, _check_types, label='check_types')

  def test_invalid_row(self):
    input_lines = ['1,2.0,hello', '5,12.34']
    column_names = ['int_feature', 'float_feature', 'str_feature']
    with self.assertRaisesRegexp(
        ValueError, '.*Columns do not match specified csv headers.*'):
      with beam.Pipeline() as p:
        result = (
            p | beam.Create(input_lines, reshuffle=False)
            | beam.ParDo(csv_decoder.ParseCSVLine(delimiter=','))
            | beam.CombineGlobally(
                csv_decoder.ColumnTypeInferrer(
                    column_names, skip_blank_lines=False)))
        beam_test_util.assert_that(result, lambda _: None)


if __name__ == '__main__':
  absltest.main()
