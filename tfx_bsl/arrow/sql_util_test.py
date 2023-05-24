# Copyright 2019 Google LLC
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
# limitations under the License
"""Tests for tfx_bsl.arrow.sql_util."""

import sys
import unittest

import pyarrow as pa
from tfx_bsl.arrow import sql_util

from absl.testing import absltest
from absl.testing import parameterized

_TEST_CASES_OF_PRIMITIVE_ARRAYS = [
    dict(
        testcase_name='with_no_filters',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2)
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[
            [[('f1', '1'), ('f2', '10')]],
            [[('f1', '2'), ('f2', '20')]],
            [[('f1', 'NULL'), ('f2', 'NULL')]],
            [[('f1', '3'), ('f2', '30')]],
        ]),
    dict(
        testcase_name='with_filter_of_null_value',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2)
              FROM
                UNNEST([example]) AS e
              WHERE f1 IS NOT NULL
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[
            [[('f1', '1'), ('f2', '10')]],
            [[('f1', '2'), ('f2', '20')]],
            [],
            [[('f1', '3'), ('f2', '30')]],
        ]),
    dict(
        testcase_name='with_filter_of_all_value',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2)
              FROM
                UNNEST([example]) AS e
              WHERE f1 = 0
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[
            [],
            [],
            [],
            [],
        ]),
]

_TEST_CASES_OF_LIST_ARRAYS = [
    dict(
        testcase_name='with_no_filters',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2)
              FROM
                example.f1,
                example.f2
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[[[('f1', '1'), ('f2', '10')],
                          [('f1', '1'), ('f2', '20')],
                          [('f1', '1'), ('f2', '30')],
                          [('f1', '2'), ('f2', '10')],
                          [('f1', '2'), ('f2', '20')],
                          [('f1', '2'), ('f2', '30')],
                          [('f1', '3'), ('f2', '10')],
                          [('f1', '3'), ('f2', '20')],
                          [('f1', '3'), ('f2', '30')]],
                         [[('f1', '4'), ('f2', '40')]], [], [], [], [],
                         [[('f1', 'NULL'), ('f2', 'NULL')]],
                         [[('f1', '7'), ('f2', 'NULL')]]]),
    dict(
        testcase_name='with_filter_of_f1',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2)
              FROM
                example.f1,
                example.f2
              WHERE
                f1 != 1
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[[[('f1', '2'), ('f2', '10')],
                          [('f1', '2'), ('f2', '20')],
                          [('f1', '2'), ('f2', '30')],
                          [('f1', '3'), ('f2', '10')],
                          [('f1', '3'), ('f2', '20')],
                          [('f1', '3'), ('f2', '30')]],
                         [[('f1', '4'), ('f2', '40')]], [], [], [], [], [],
                         [[('f1', '7'), ('f2', 'NULL')]]]),
    dict(
        testcase_name='with_only_one_filed_selected_with_alias',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1 as some_name)
              FROM
                example.f1
              WHERE
                f1 != 1
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[[[('some_name', '2')], [('some_name', '3')]],
                         [[('some_name', '4')]], [], [[('some_name', '5')]], [],
                         [[('some_name', '6')]], [], [[('some_name', '7')]]]),
]

_TEST_CASES_WITH_ONE_FIELD_FORMAT_NOT_SUPPORTED = [
    dict(
        testcase_name='with_supported_column_queried',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f2)
              FROM
                UNNEST([example]) AS e
              WHERE f2 != 1
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[[], [[('f2', '2')]], [], [[('f2', '3')]]],
        error=False),
    dict(
        testcase_name='with_unsupported_column_queried',
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1)
              FROM
                UNNEST([example]) AS e
            ) as slice_key
            FROM Examples as example;""",
        expected_output=None,
        error=True),
]

_TEST_CASES_WITH_ALL_FIELDS_FORMAT_NOT_SUPPORTED = [
    dict(
        testcase_name='with_no_field_queried',
        sql="""
          SELECT
          ARRAY(
            SELECT STRUCT(1 as field)
          ) as slice_key
          FROM Examples as example;""",
        expected_output=[[[('field', '1')]], [[('field', '1')]],
                         [[('field', '1')]], [[('field', '1')]]]),
]

_TEST_CASES_WITH_INVALID_STATEMENT = [
    dict(
        testcase_name='with_syntax_error',
        sql="""
          SELLLLLLECT
            *
          FROM Examples as example;""",
        error="""Unexpected identifier "SELLLLLLECT"""),
    dict(
        testcase_name='with_more_than_one_columns',
        sql="""
          SELECT
            f1, f2
          FROM Examples as example;""",
        error='Only one column should be returned.'),
    dict(
        testcase_name='with_no_struct_format_returned',
        sql="""
          SELECT
            f1
          FROM Examples as example;""",
        error='query result should in an Array of Struct type.'),
]

_TEST_CASES_OF_STRUCT_ARRAYS = [
    dict(
        testcase_name='on_struct_and_nested_struct',
        record_batch=pa.RecordBatch.from_arrays(
            [
                pa.array([1, 2, None], type=pa.int64()),
                pa.array(
                    [(3, 4.1), None, (5, 6.3)],
                    type=pa.struct(
                        [pa.field('a', pa.int32()), pa.field('b', pa.float32())]
                    ),
                ),
                pa.array(
                    [
                        (7.1, ('o_string', b'x_bytes')),
                        (8.2, ('p_string', b'y_bytes')),
                        (9.3, ('q_string', b'z_bytes')),
                    ],
                    type=pa.struct([
                        pa.field('c', pa.float64()),
                        pa.field(
                            'd',
                            pa.struct([
                                pa.field('e', pa.string()),
                                pa.field('f', pa.binary()),
                            ]),
                        ),
                    ]),
                ),
            ],
            ['f1', 'f2', 'f3'],
        ),
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(f1, f2.a as f2_a, f2.b as f2_b, f3.c as f3_c,
                f3.d.e as f3_d_e, f3.d.f as f3_d_f)
            ) as slice_key
          FROM Examples as example;""",
        expected_output=[
            [[
                ('f1', '1'),
                ('f2_a', '3'),
                ('f2_b', '4.1'),
                ('f3_c', '7.1'),
                ('f3_d_e', 'o_string'),
                ('f3_d_f', 'x_bytes'),
            ]],
            [[
                ('f1', '2'),
                ('f2_a', 'NULL'),
                ('f2_b', 'NULL'),
                ('f3_c', '8.2'),
                ('f3_d_e', 'p_string'),
                ('f3_d_f', 'y_bytes'),
            ]],
            [[
                ('f1', 'NULL'),
                ('f2_a', '5'),
                ('f2_b', '6.3'),
                ('f3_c', '9.3'),
                ('f3_d_e', 'q_string'),
                ('f3_d_f', 'z_bytes'),
            ]],
        ],
    ),
    dict(
        testcase_name='on_struct_of_list',
        record_batch=pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [([5], ([1.1],)), None, ([8], ([3.3],))],
                    type=pa.struct([
                        pa.field('int64_list', pa.list_(pa.int64())),
                        pa.field(
                            'f2',
                            pa.struct([
                                pa.field(
                                    'float64_list', pa.list_(pa.float64())
                                ),
                            ]),
                        ),
                    ]),
                ),
            ],
            ['f1'],
        ),
        sql="""
          SELECT
            ARRAY(
              SELECT
                STRUCT(int64_list as f1_int64_list,
                       float64_list as f1_f2_float64_list)
              FROM
                example.f1.int64_list,
                example.f1.f2.float64_list
            ) as slice_key
            FROM Examples as example;""",
        expected_output=[
            [[('f1_int64_list', '5'), ('f1_f2_float64_list', '1.1')]],
            [],
            [[('f1_int64_list', '8'), ('f1_f2_float64_list', '3.3')]],
        ],
    ),
]


# The RecordBatchSQLSliceQuery uses ZetaSQL which cannot be compiled on Windows.
# b/191377114
@unittest.skipIf(
    sys.platform.startswith('win'),
    'RecordBatchSQLSliceQuery is not supported on Windows.')
class RecordBatchSQLSliceQueryTest(parameterized.TestCase):

  @parameterized.named_parameters(*_TEST_CASES_OF_PRIMITIVE_ARRAYS)
  def test_query_primitive_arrays(self, sql, expected_output):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([1, 2, None, 3], type=pa.int64()),
        pa.array([10, 20, None, 30], type=pa.int32()),
    ], ['f1', 'f2'])

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices, expected_output)

  @parameterized.named_parameters(*_TEST_CASES_OF_LIST_ARRAYS)
  def test_query_list_arrays(self, sql, expected_output):
    # List of int32 & int64.
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1, 2, 3], [4], None, [5], [], [6], [None], [7]],
                 type=pa.list_(pa.int64())),
        pa.array([[10, 20, 30], [40], None, None, [], [], [None], [None]],
                 type=pa.list_(pa.int32())),
    ], ['f1', 'f2'])

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices, expected_output)

  @parameterized.named_parameters(*_TEST_CASES_OF_LIST_ARRAYS)
  def test_query_large_list_arrays(self, sql, expected_output):
    # Large list of int32 & int64.
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1, 2, 3], [4], None, [5], [], [6], [None], [7]],
                 type=pa.large_list(pa.int64())),
        pa.array([[10, 20, 30], [40], None, None, [], [], [None], [None]],
                 type=pa.large_list(pa.int32())),
    ], ['f1', 'f2'])

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)

    self.assertEqual(slices, expected_output)

  @parameterized.named_parameters(
      *_TEST_CASES_WITH_ONE_FIELD_FORMAT_NOT_SUPPORTED)
  def test_query_with_one_field_not_supported(self, sql, expected_output,
                                              error):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[[10, 100]], [[20, 200]], None, [[30, 300]]],
                 type=pa.list_(pa.list_(pa.int64()))),
        pa.array([1, 2, None, 3], type=pa.int32()),
    ], ['f1', 'f2'])

    if error:
      with self.assertRaisesRegex(RuntimeError,
                                  'Are you querying any unsupported column?'):
        query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    else:
      query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
      slices = query.Execute(record_batch)
      self.assertEqual(slices, expected_output)

  @parameterized.named_parameters(
      *_TEST_CASES_WITH_ALL_FIELDS_FORMAT_NOT_SUPPORTED)
  def test_query_with_all_fields_not_supported(self, sql, expected_output):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[[10, 100]], [[20, 200]], None, [[30, 300]]],
                 type=pa.list_(pa.list_(pa.int64()))),
    ], ['f1'])

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices, expected_output)

  @parameterized.named_parameters(*_TEST_CASES_WITH_INVALID_STATEMENT)
  def test_query_with_invalid_statement(self, sql, error):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1, 2, 3], [4], None, [5], [], [6], [None], [7]],
                 type=pa.list_(pa.int64())),
        pa.array([[10, 20, 30], [40], None, None, [], [], [None], [None]],
                 type=pa.list_(pa.int32())),
    ], ['f1', 'f2'])

    with self.assertRaisesRegex(RuntimeError, error):
      _ = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)

  def test_query_with_unexpected_record_batch_schema(self):
    record_batch_1 = pa.RecordBatch.from_arrays([
        pa.array([1, 2, 3], type=pa.int64()),
    ], ['f1'])
    record_batch_2 = pa.RecordBatch.from_arrays([
        pa.array([4, 5, 6], type=pa.int32()),
    ], ['f1'])
    sql = """SELECT ARRAY(SELECT STRUCT(f1)) as slice_key
          FROM Examples as example;"""

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch_1.schema)
    with self.assertRaisesRegex(RuntimeError, 'Unexpected RecordBatch schema.'):
      _ = query.Execute(record_batch_2)

  def test_query_with_empty_input(self):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([], type=pa.int64()),
    ], ['f1'])
    sql = """SELECT ARRAY(SELECT STRUCT(f1)) as slice_key
          FROM Examples as example;"""

    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices, [])

  def test_query_with_all_supported_types(self):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1], [2]], type=pa.list_(pa.int32())),
        pa.array([[10], [20]], type=pa.list_(pa.int64())),
        pa.array([[1.1], [2.2]], type=pa.list_(pa.float32())),
        pa.array([[10.1], [20.2]], type=pa.list_(pa.float64())),
        pa.array([['a'], ['b']], type=pa.list_(pa.string())),
        pa.array([['a+'], ['b+']], type=pa.list_(pa.large_string())),
        pa.array([[b'a_bytes'], [b'b_bytes']], type=pa.list_(pa.binary())),
        pa.array([[b'a_bytes+'], [b'b_bytes+']],
                 type=pa.list_(pa.large_binary())),
    ], [
        'int32_list',
        'int64_list',
        'float32_list',
        'float64_list',
        'string_list',
        'large_string_list',
        'binary_list',
        'large_binary_list',
    ])
    sql = """
      SELECT
        ARRAY(
          SELECT
            STRUCT(int32_list, int64_list,
              float32_list, float64_list,
              string_list, large_string_list,
              binary_list, large_binary_list)
          FROM
            example.int32_list,
            example.int64_list,
            example.float32_list,
            example.float64_list,
            example.string_list,
            example.large_string_list,
            example.binary_list,
            example.large_binary_list
        ) as slice_key
      FROM Examples as example;"""
    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices,
                     [[[('int32_list', '1'), ('int64_list', '10'),
                        ('float32_list', '1.1'), ('float64_list', '10.1'),
                        ('string_list', 'a'), ('large_string_list', 'a+'),
                        ('binary_list', 'a_bytes'),
                        ('large_binary_list', 'a_bytes+')]],
                      [[('int32_list', '2'), ('int64_list', '20'),
                        ('float32_list', '2.2'), ('float64_list', '20.2'),
                        ('string_list', 'b'), ('large_string_list', 'b+'),
                        ('binary_list', 'b_bytes'),
                        ('large_binary_list', 'b_bytes+')]]])

  @parameterized.named_parameters(*_TEST_CASES_OF_STRUCT_ARRAYS)
  def test_query_with_struct_arrays(self, record_batch, sql, expected_output):
    query = sql_util.RecordBatchSQLSliceQuery(sql, record_batch.schema)
    slices = query.Execute(record_batch)
    self.assertEqual(slices, expected_output)


if __name__ == '__main__':
  absltest.main()
