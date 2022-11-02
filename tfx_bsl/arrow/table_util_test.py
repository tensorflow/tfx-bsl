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
"""Tests for tfx_bsl.arrow.table_util."""

import pyarrow as pa
from tfx_bsl.arrow import table_util

from absl.testing import absltest
from absl.testing import parameterized


_MERGE_TEST_CASES = [
    dict(
        testcase_name="empty_input",
        inputs=[],
        expected_output=dict(),
    ),
    dict(
        testcase_name="basic_types",
        inputs=[
            {
                "bool": pa.array([False, None, True], type=pa.bool_()),
                "int64": pa.array([1, None, 3], type=pa.int64()),
                "uint64": pa.array([1, None, 3], type=pa.uint64()),
                "int32": pa.array([1, None, 3], type=pa.int32()),
                "uint32": pa.array([1, None, 3], type=pa.uint32()),
                "float": pa.array([1., None, 3.], type=pa.float32()),
                "double": pa.array([1., None, 3.], type=pa.float64()),
                "bytes": pa.array([b"abc", None, b"ghi"], type=pa.binary()),
                "large_bytes": pa.array([b"abc", None, b"ghi"],
                                        type=pa.large_binary()),
                "unicode": pa.array([u"abc", None, u"ghi"], type=pa.utf8()),
                "large_unicode": pa.array([u"abc", None, u"ghi"],
                                          type=pa.large_utf8()),
            },
            {
                "bool": pa.array([None, False], type=pa.bool_()),
                "int64": pa.array([None, 4], type=pa.int64()),
                "uint64": pa.array([None, 4], type=pa.uint64()),
                "int32": pa.array([None, 4], type=pa.int32()),
                "uint32": pa.array([None, 4], type=pa.uint32()),
                "float": pa.array([None, 4.], type=pa.float32()),
                "double": pa.array([None, 4.], type=pa.float64()),
                "bytes": pa.array([None, b"jkl"], type=pa.binary()),
                "large_bytes": pa.array([None, b"jkl"], type=pa.large_binary()),
                "unicode": pa.array([None, u"jkl"], type=pa.utf8()),
                "large_unicode": pa.array([None, u"jkl"], type=pa.large_utf8()),
            },
        ],
        expected_output={
            "bool":
                pa.array([False, None, True, None, False], type=pa.bool_()),
            "int64":
                pa.array([1, None, 3, None, 4], type=pa.int64()),
            "uint64":
                pa.array([1, None, 3, None, 4], type=pa.uint64()),
            "int32":
                pa.array([1, None, 3, None, 4], type=pa.int32()),
            "uint32":
                pa.array([1, None, 3, None, 4], type=pa.uint32()),
            "float":
                pa.array([1., None, 3., None, 4.], type=pa.float32()),
            "double":
                pa.array([1., None, 3., None, 4.], type=pa.float64()),
            "bytes":
                pa.array([b"abc", None, b"ghi", None, b"jkl"],
                         type=pa.binary()),
            "large_bytes":
                pa.array([b"abc", None, b"ghi", None, b"jkl"],
                         type=pa.large_binary()),
            "unicode":
                pa.array([u"abc", None, u"ghi", None, u"jkl"],
                         type=pa.utf8()),
            "large_unicode":
                pa.array([u"abc", None, u"ghi", None, u"jkl"],
                         type=pa.large_utf8()),
        }),
    dict(
        testcase_name="list",
        inputs=[
            {
                "list<int32>":
                    pa.array([[1, None, 3], None], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([None], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([], type=pa.list_(pa.int32())),
            },
            {
                "list<int32>": pa.array([[]], type=pa.list_(pa.int32())),
            },
        ],
        expected_output={
            "list<int32>":
                pa.array([[1, None, 3], None, None, []],
                         type=pa.list_(pa.int32()))
        }),
    dict(
        testcase_name="large_list",
        inputs=[
            {
                "large_list<int32>":
                    pa.array([[1, None, 3], None],
                             type=pa.large_list(pa.int32())),
            },
            {
                "large_list<int32>":
                    pa.array([None], type=pa.large_list(pa.int32())),
            },
            {
                "large_list<int32>":
                    pa.array([], type=pa.large_list(pa.int32())),
            },
            {
                "large_list<int32>":
                    pa.array([[]], type=pa.large_list(pa.int32())),
            },
        ],
        expected_output={
            "large_list<int32>":
                pa.array([[1, None, 3], None, None, []],
                         type=pa.large_list(pa.int32()))
        }),
    dict(
        testcase_name="struct",
        inputs=[{
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"abc", None, b"def"]),
                    pa.array([[None], [1, 2], []], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }, {
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"ghi"]),
                    pa.array([[3]], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }],
        expected_output={
            "struct<binary, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([b"abc", None, b"def", b"ghi"]),
                    pa.array([[None], [1, 2], [], [3]],
                             type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }),
    dict(
        testcase_name="missing_or_null_column_fixed_width",
        inputs=[
            {
                "int32": pa.array([None, None], type=pa.null())
            },
            {
                "int64": pa.array([None, None], type=pa.null())
            },
            {
                "int64": pa.array([123], type=pa.int64())
            },
            {
                "int32": pa.array([456], type=pa.int32())
            },
        ],
        expected_output={
            "int32":
                pa.array([None, None, None, None, None, 456], type=pa.int32()),
            "int64":
                pa.array([None, None, None, None, 123, None], type=pa.int64()),
        }),
    dict(
        testcase_name="missing_or_null_column_list_alike",
        inputs=[
            {
                "list<int32>": pa.array([None, None], type=pa.null())
            },
            {
                "utf8": pa.array([None, None], type=pa.null())
            },
            {
                "utf8": pa.array([u"abc"], type=pa.utf8())
            },
            {
                "list<int32>":
                    pa.array([None, [123, 456]], type=pa.list_(pa.int32()))
            },
        ],
        expected_output={
            "list<int32>":
                pa.array([None, None, None, None, None, None, [123, 456]],
                         type=pa.list_(pa.int32())),
            "utf8":
                pa.array([None, None, None, None, u"abc", None, None],
                         type=pa.utf8()),
        }),
    dict(
        testcase_name="missing_or_null_column_struct",
        inputs=[{
            "struct<int32, list<int32>>": pa.array([None, None], type=pa.null())
        }, {
            "list<utf8>": pa.array([None, None], type=pa.null())
        }, {
            "struct<int32, list<int32>>":
                pa.StructArray.from_arrays([
                    pa.array([1, 2, None], type=pa.int32()),
                    pa.array([[1], None, [3, 4]], type=pa.list_(pa.int32()))
                ], ["f1", "f2"])
        }, {
            "list<utf8>": pa.array([u"abc", None], type=pa.utf8())
        }],
        expected_output={
            "list<utf8>":
                pa.array(
                    [None, None, None, None, None, None, None, u"abc", None],
                    type=pa.utf8()),
            "struct<int32, list<int32>>":
                pa.array([
                    None, None, None, None, (1, [1]), (2, None),
                    (None, [3, 4]), None, None
                ],
                         type=pa.struct([
                             pa.field("f1", pa.int32()),
                             pa.field("f2", pa.list_(pa.int32()))
                         ])),
        }),
    dict(
        testcase_name="merge_list_of_null_and_list_of_list",
        inputs=[{
            "f": pa.array([[None, None], None], type=pa.list_(pa.null()))
        }, {
            "f": pa.array([[[123]], None], type=pa.list_(pa.list_(pa.int32())))
        }],
        expected_output={
            "f":
                pa.array([[None, None], None, [[123]], None],
                         type=pa.list_(pa.list_(pa.int32())))
        }),
    dict(
        testcase_name="merge_large_list_of_null_and_list_of_list",
        inputs=[{
            "f": pa.array([[None, None], None], type=pa.large_list(pa.null()))
        }, {
            "f": pa.array([[[123]], None],
                          type=pa.large_list(pa.large_list(pa.int32())))
        }],
        expected_output={
            "f":
                pa.array([[None, None], None, [[123]], None],
                         type=pa.large_list(pa.large_list(pa.int32())))
        }),
    dict(
        testcase_name="merge_sliced_list_of_null_and_list_of_list",
        inputs=[{
            "f": pa.array(
                [None, [None, None], None], type=pa.list_(pa.null())).slice(1)
        }, {
            "f": pa.array([[[123]], None], type=pa.list_(pa.list_(pa.int32())))
        }],
        expected_output={
            "f":
                pa.array([[None, None], None, [[123]], None],
                         type=pa.list_(pa.list_(pa.int32())))
        }),
    dict(
        testcase_name="merge_list_of_list_and_list_of_null",
        inputs=[{
            "f": pa.array([[[123]], None], type=pa.list_(pa.list_(pa.int32())))
        }, {
            "f": pa.array([[None, None], None], type=pa.list_(pa.null()))
        }],
        expected_output={
            "f":
                pa.array([[[123]], None, [None, None], None],
                         type=pa.list_(pa.list_(pa.int32())))
        }),
    dict(
        testcase_name="merge_list_of_null_and_null",
        inputs=[{
            "f": pa.array([None], type=pa.null())
        }, {
            "f": pa.array([[None, None], None], type=pa.list_(pa.null()))
        }],
        expected_output={
            "f": pa.array([None, [None, None], None], type=pa.list_(pa.null()))
        }),
    dict(
        testcase_name="merge_compatible_struct_missing_field",
        inputs=[{
            "f": pa.array([{"a": [1]}, {"a": [2, 3]}]),
        }, {
            "f": pa.array([{"b": [1.0]}]),
        }],
        expected_output={
            "f": pa.array([
                {"a": [1], "b": None},
                {"a": [2, 3], "b": None},
                {"a": None, "b": [1.0]}])
        }),
    dict(
        testcase_name="merge_compatible_struct_null_type",
        inputs=[{
            "f":
                pa.array([{"a": [[1]]}],
                         type=pa.struct([
                             pa.field("a",
                                      pa.large_list(pa.large_list(pa.int32())))
                         ])),
        }, {
            "f":
                pa.array([{"a": None}, {"a": None}],
                         type=pa.struct([pa.field("a", pa.null())])),
        }],
        expected_output={
            "f":
                pa.array([{"a": [[1]]},
                          {"a": None},
                          {"a": None}],
                         type=pa.struct([
                             pa.field("a",
                                      pa.large_list(pa.large_list(pa.int32())))
                         ]))
        }),
    dict(
        testcase_name="merge_compatible_struct_in_struct",
        inputs=[{
            "f": pa.array([{}, {}]),
        }, {
            "f": pa.array([
                {"a": [{"b": 1}]},
                {"a": [{"b": 2}]},
            ])
        }, {
            "f": pa.array([
                {"a": [{"b": 3, "c": 1}]},
            ])
        }],
        expected_output={
            "f": pa.array([
                {"a": None},
                {"a": None},
                {"a": [{"b": 1, "c": None}]},
                {"a": [{"b": 2, "c": None}]},
                {"a": [{"b": 3, "c": 1}]}])
        })
]

_MERGE_INVALID_INPUT_TEST_CASES = [
    dict(
        testcase_name="column_type_differs",
        inputs=[
            pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int32())],
                                       ["f1"]),
            pa.RecordBatch.from_arrays([pa.array([4, 5, 6], type=pa.int64())],
                                       ["f1"])
        ],
        expected_error_regexp="Unable to merge incompatible type"),
]


class MergeRecordBatchesTest(parameterized.TestCase):

  @parameterized.named_parameters(*_MERGE_INVALID_INPUT_TEST_CASES)
  def test_invalid_inputs(self, inputs, expected_error_regexp):
    with self.assertRaisesRegex(Exception, expected_error_regexp):
      _ = table_util.MergeRecordBatches(inputs)

  @parameterized.named_parameters(*_MERGE_TEST_CASES)
  def test_merge_record_batches(self, inputs, expected_output):
    input_record_batches = [
        pa.RecordBatch.from_arrays(list(in_dict.values()), list(in_dict.keys()))
        for in_dict in inputs
    ]
    merged = table_util.MergeRecordBatches(input_record_batches)

    self.assertLen(expected_output, merged.num_columns)
    for column, column_name in zip(merged.columns, merged.schema.names):
      self.assertTrue(
          expected_output[column_name].equals(column),
          "Column {}:\nexpected:{}\ngot: {}".format(
              column_name, expected_output[column_name], column))

  def test_merge_0_column_record_batches(self):
    record_batches = ([
        pa.table([pa.array([1, 2, 3])],
                 ["ignore"]).remove_column(0).to_batches(max_chunksize=None)[0]
    ] * 3)
    merged = table_util.MergeRecordBatches(record_batches)
    self.assertEqual(merged.num_rows, 9)
    self.assertEqual(merged.num_columns, 0)


_GET_TOTAL_BYTE_SIZE_TEST_NAMED_PARAMS = [
    dict(testcase_name="table", factory=pa.Table.from_arrays),
    dict(testcase_name="record_batch", factory=pa.RecordBatch.from_arrays),
]


class GetTotalByteSizeTest(parameterized.TestCase):

  @parameterized.named_parameters(*_GET_TOTAL_BYTE_SIZE_TEST_NAMED_PARAMS)
  def test_simple(self, factory):
    # 3 int64 values
    # 5 int32 offsets
    # 1 null bitmap byte for outer ListArray
    # 1 null bitmap byte for inner Int64Array
    # 46 bytes in total.
    list_array = pa.array([[1, 2], [None], None, None],
                          type=pa.list_(pa.int64()))

    # 1 null bitmap byte for outer StructArray.
    # 1 null bitmap byte for inner Int64Array.
    # 4 int64 values.
    # 34 bytes in total
    struct_array = pa.array([{"a": 1}, {"a": 2}, {"a": None}, None],
                            type=pa.struct([pa.field("a", pa.int64())]))
    entity = factory([list_array, struct_array], ["a1", "a2"])

    self.assertEqual(46 + 34, table_util.TotalByteSize(entity))


_TAKE_TEST_CASES = [
    dict(
        testcase_name="no_index",
        row_indices=[],
        expected_output=pa.RecordBatch.from_arrays([
            pa.array([], type=pa.list_(pa.int32())),
            pa.array([], type=pa.list_(pa.binary()))
        ], ["f1", "f2"])),
    dict(
        testcase_name="one_index",
        row_indices=[1],
        expected_output=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.list_(pa.int32())),
            pa.array([["b", "c"]], type=pa.list_(pa.binary()))
        ], ["f1", "f2"])),
    dict(
        testcase_name="consecutive_first_row_included",
        row_indices=[0, 1, 2, 3],
        expected_output=pa.RecordBatch.from_arrays(
            [
                pa.array([[1, 2, 3], None, [4], []], type=pa.list_(pa.int32())),
                pa.array([["a"], ["b", "c"], None, []],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="consecutive_last_row_included",
        row_indices=[5, 6, 7, 8],
        expected_output=pa.RecordBatch.from_arrays(
            [
                pa.array([[7], [8, 9], [10], []], type=pa.list_(pa.int32())),
                pa.array([["d", "e"], ["f"], None, ["g"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="inconsecutive",
        row_indices=[1, 2, 3, 5],
        expected_output=pa.RecordBatch.from_arrays(
            [
                pa.array([None, [4], [], [7]], type=pa.list_(pa.int32())),
                pa.array([["b", "c"], None, [], ["d", "e"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
    dict(
        testcase_name="inconsecutive_last_row_included",
        row_indices=[2, 3, 4, 5, 7, 8],
        expected_output=pa.RecordBatch.from_arrays(
            [
                pa.array([[4], [], [5, 6], [7], [10], []],
                         type=pa.list_(pa.int32())),
                pa.array([None, [], None, ["d", "e"], None, ["g"]],
                         type=pa.list_(pa.binary()))
            ],
            ["f1", "f2"],
        )),
]


class RecordBatchTakeTest(parameterized.TestCase):

  @parameterized.named_parameters(*_TAKE_TEST_CASES)
  def test_success(self, row_indices, expected_output):
    record_batch = pa.RecordBatch.from_arrays([
        pa.array([[1, 2, 3], None, [4], [], [5, 6], [7], [8, 9], [10], []],
                 type=pa.list_(pa.int32())),
        pa.array(
            [["a"], ["b", "c"], None, [], None, ["d", "e"], ["f"], None, ["g"]],
            type=pa.list_(pa.binary())),
    ], ["f1", "f2"])

    for row_indices_type in (pa.int32(), pa.int64()):
      sliced = table_util.RecordBatchTake(
          record_batch, pa.array(row_indices, type=row_indices_type))
      self.assertTrue(
          sliced.equals(expected_output),
          "Expected {}, got {}".format(expected_output, sliced))


class CanonicalizeRecordBatchTest(parameterized.TestCase):

  def testCanonicalizeRecordBatch(self):
    rb_data = pa.RecordBatch.from_arrays([
        pa.array([17, 30], pa.int32()),
        pa.array(["english", "spanish"]),
        pa.array([False, True]),
        pa.array([False, True]),
        pa.array([["ne"], ["s", "ted"]])
    ], ["age", "language", "prediction", "label", "nested"])

    canonicalized_rb_data = table_util.CanonicalizeRecordBatch(rb_data)
    self.assertEqual(canonicalized_rb_data.schema.names, rb_data.schema.names)

    expected_age_column = pa.array([[17], [30]], type=pa.large_list(pa.int64()))
    expected_language_column = pa.array([["english"], ["spanish"]],
                                        type=pa.large_list(pa.large_binary()))
    expected_prediction_column = pa.array([[0], [1]],
                                          type=pa.large_list(pa.int8()))
    expected_label_column = pa.array([[0], [1]], type=pa.large_list(pa.int8()))
    expected_nested_column = pa.array([["ne"], ["s", "ted"]],
                                      type=pa.large_list(pa.large_binary()))
    self.assertTrue(
        canonicalized_rb_data.column(
            canonicalized_rb_data.schema.get_field_index("age")).equals(
                expected_age_column))
    self.assertTrue(
        canonicalized_rb_data.column(
            canonicalized_rb_data.schema.get_field_index("language")).equals(
                expected_language_column))
    self.assertTrue(
        canonicalized_rb_data.column(
            canonicalized_rb_data.schema.get_field_index("prediction")).equals(
                expected_prediction_column))
    self.assertTrue(
        canonicalized_rb_data.column(
            canonicalized_rb_data.schema.get_field_index("label")).equals(
                expected_label_column))
    self.assertTrue(
        canonicalized_rb_data.column(
            canonicalized_rb_data.schema.get_field_index("nested")).equals(
                expected_nested_column))


if __name__ == "__main__":
  absltest.main()
