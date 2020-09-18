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
"""Tests for tfx_bsl.coders.sequence_example_coder."""

import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import sequence_example_coder

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2

_TEST_SEQUENCE_COLUMN_NAME = "##SEQUENCE##"
_TYPED_SEQUENCE_EXAMPLE = """
    context {
    feature {
        key: 'context_a'
        value {
          int64_list {
            value: [1]
          }
        }
      }
      feature {
        key: "context_b"
        value {
          float_list {
            value: [1.0, 2.0]
          }
        }
      }
      feature {
        key: 'context_c'
        value {
          bytes_list {
            value: ['a', 'b', 'c']
          }
        }
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_x'
        value {
          feature {
            int64_list {
              value: [1, 2]
            }
          }
          feature {
            int64_list {
              value: [3]
            }
          }
        }
      }
      feature_list {
        key: "sequence_y"
        value {
          feature {
            float_list {
              value: [3.0, 4.0]
            }
          }
          feature {
            float_list {
              value: [1.0, 2.0]
            }
          }
        }
      }
      feature_list {
        key: 'sequence_z'
        value {
          feature {
            bytes_list {
              value: ['a', 'b']
            }
          }
          feature {
            bytes_list {
              value: ['c']
            }
          }
        }
      }
    }
    """
_UNTYPED_SEQUENCE_EXAMPLE = """
    context {
    feature {
        key: 'context_a'
        value {}
      }
      feature {
        key: "context_b"
        value {}
      }
      feature {
        key: 'context_c'
        value {}
      }
      feature {
        key: 'context_d'
        value {}
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_x'
        value {}
      }
      feature_list {
        key: "sequence_y"
        value {}
      }
      feature_list {
        key: 'sequence_z'
        value {}
      }
    }
    """
_SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE = """
context {
    feature {
        key: 'context_a'
        value {}
      }
      feature {
        key: "context_b"
        value {}
      }
      feature {
        key: 'context_c'
        value {}
      }
      feature {
        key: 'context_d'
        value {}
      }
      feature {
        key: 'context_e'
        value {
          float_list {
            value: [1.0]
          }
        }
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_v'
        value {
          feature {
            float_list {
              value: [1.0]
            }
          }
        }
      }
      feature_list {
        key: 'sequence_x'
        value {
          feature {}
          feature {}
          feature {}
        }
      }
      feature_list {
        key: "sequence_y"
        value {
          feature {}
        }
      }
      feature_list {
        key: 'sequence_z'
        value {
          feature {}
        }
      }
    }
    """
_EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE = """
context {
    feature {
        key: 'context_a'
        value {
          int64_list {
            value: []
          }
        }
      }
      feature {
        key: "context_b"
        value {
          float_list {
            value: []
          }
        }
      }
      feature {
        key: 'context_c'
        value {
          bytes_list {
            value: []
          }
        }
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_x'
        value {
          feature {
            int64_list {
              value: []
            }
          }
          feature {
            int64_list {
              value: []
            }
          }
        }
      }
      feature_list {
        key: "sequence_y"
        value {
          feature {
            float_list {
              value: []
            }
          }
        }
      }
      feature_list {
        key: 'sequence_z'
        value {
          feature {
            bytes_list {
              value: []
            }
          }
        }
      }
    }
    """
_TEST_SEQUENCE_EXAMPLES_NONE_TYPED = [
    """
    context {
    feature {
        key: 'context_a'
        value {}
      }
      feature {
        key: "context_b"
        value {}
      }
      feature {
        key: 'context_c'
        value {}
      }
      feature {
        key: 'context_d'
        value {}
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_x'
        value {}
      }
    }
    """,
    """
    context {
    feature {
        key: 'context_a'
        value {}
      }
      feature {
        key: "context_b"
        value {}
      }
      feature {
        key: 'context_c'
        value {}
      }
      feature {
        key: 'context_d'
        value {}
      }
    }
    feature_lists {
      feature_list {
        key: 'sequence_w'
        value {
          feature {}
        }
      }
      feature_list {
        key: 'sequence_x'
        value {
          feature {}
        }
      }
    }
    """,
]

_DECODE_CASES = [
    dict(
        testcase_name="without_schema_first_example_typed",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            _TYPED_SEQUENCE_EXAMPLE, _UNTYPED_SEQUENCE_EXAMPLE,
            _SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE,
            _EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([[1], None, None, []], type=pa.large_list(pa.int64())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
            pa.array([[b"a", b"b", b"c"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([None, None, None, None], pa.null()),
            pa.array([None, None, [1.0], None],
                     type=pa.large_list(pa.float32())),
            pa.StructArray.from_arrays([
                pa.array([None, None, [[1.0]], None],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[[1, 2], [3]], [], [None, None, None], [[], []]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
                pa.array([[[3.0, 4.0], [1.0, 2.0]], [], [None], [[]]],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[[b"a", b"b"], [b"c"]], [], [None], [[]]],
                         type=pa.large_list(pa.large_list(pa.large_binary())))
            ],
                                       names=[
                                           "sequence_v", "sequence_x",
                                           "sequence_y", "sequence_z"
                                       ])
        ], [
            "context_a", "context_b", "context_c", "context_d", "context_e",
            _TEST_SEQUENCE_COLUMN_NAME
        ])),
    dict(
        testcase_name="with_schema_first_example_typed",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: INT
        }
        feature {
          name: "context_b"
          type: FLOAT
        }
        feature {
          name: "context_c"
          type: BYTES
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_x"
              type: INT
            }
            feature {
              name: "sequence_y"
              type: FLOAT
            }
            feature {
              name: "sequence_z"
              type: BYTES
            }
          }
        }""",
        sequence_examples_text_proto=[
            _TYPED_SEQUENCE_EXAMPLE, _UNTYPED_SEQUENCE_EXAMPLE,
            _SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE,
            _EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([[1], None, None, []], type=pa.large_list(pa.int64())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
            pa.array([[b"a", b"b", b"c"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.StructArray.from_arrays([
                pa.array([[[1, 2], [3]], [], [None, None, None], [[], []]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
                pa.array([[[3.0, 4.0], [1.0, 2.0]], [], [None], [[]]],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[[b"a", b"b"], [b"c"]], [], [None], [[]]],
                         type=pa.large_list(pa.large_list(pa.large_binary())))
            ],
                                       names=[
                                           "sequence_x", "sequence_y",
                                           "sequence_z"
                                       ])
        ], ["context_a", "context_b", "context_c", _TEST_SEQUENCE_COLUMN_NAME
           ])),
    dict(
        testcase_name="without_schema_untyped_then_typed_examples",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            _UNTYPED_SEQUENCE_EXAMPLE, _SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE,
            _EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE, _TYPED_SEQUENCE_EXAMPLE
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None, [], [1]], type=pa.large_list(pa.int64())),
            pa.array([None, None, [], [1.0, 2.0]],
                     type=pa.large_list(pa.float32())),
            pa.array([None, None, [], [b"a", b"b", b"c"]],
                     type=pa.large_list(pa.large_binary())),
            pa.array([None, None, None, None], pa.null()),
            pa.array([None, [1.0], None, None],
                     type=pa.large_list(pa.float32())),
            pa.StructArray.from_arrays([
                pa.array([None, [[1.0]], None, None],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[], [None, None, None], [[], []], [[1, 2], [3]]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
                pa.array([[], [None], [[]], [[3.0, 4.0], [1.0, 2.0]]],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[], [None], [[]], [[b"a", b"b"], [b"c"]]],
                         type=pa.large_list(pa.large_list(pa.large_binary())))
            ],
                                       names=[
                                           "sequence_v", "sequence_x",
                                           "sequence_y", "sequence_z"
                                       ])
        ], [
            "context_a", "context_b", "context_c", "context_d", "context_e",
            _TEST_SEQUENCE_COLUMN_NAME
        ])),
    dict(
        testcase_name="with_schema_untyped_then_typed_examples",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: INT
        }
        feature {
          name: "context_b"
          type: FLOAT
        }
        feature {
          name: "context_c"
          type: BYTES
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_x"
              type: INT
            }
            feature {
              name: "sequence_y"
              type: FLOAT
            }
            feature {
              name: "sequence_z"
              type: BYTES
            }
          }
        }""",
        sequence_examples_text_proto=[
            _UNTYPED_SEQUENCE_EXAMPLE, _SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE,
            _EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE, _TYPED_SEQUENCE_EXAMPLE
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None, [], [1]], type=pa.large_list(pa.int64())),
            pa.array([None, None, [], [1.0, 2.0]],
                     type=pa.large_list(pa.float32())),
            pa.array([None, None, [], [b"a", b"b", b"c"]],
                     type=pa.large_list(pa.large_binary())),
            pa.StructArray.from_arrays([
                pa.array([[], [None, None, None], [[], []], [[1, 2], [3]]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
                pa.array([[], [None], [[]], [[3.0, 4.0], [1.0, 2.0]]],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([[], [None], [[]], [[b"a", b"b"], [b"c"]]],
                         type=pa.large_list(pa.large_list(pa.large_binary())))
            ],
                                       names=[
                                           "sequence_x", "sequence_y",
                                           "sequence_z"
                                       ])
        ], ["context_a", "context_b", "context_c", _TEST_SEQUENCE_COLUMN_NAME
           ])),
    dict(
        testcase_name="without_schema_no_typed_examples",
        schema_text_proto=None,
        sequence_examples_text_proto=_TEST_SEQUENCE_EXAMPLES_NONE_TYPED,
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None], type=pa.null()),
            pa.array([None, None], type=pa.null()),
            pa.array([None, None], type=pa.null()),
            pa.array([None, None], type=pa.null()),
            pa.StructArray.from_arrays([
                pa.array([None, [None]], type=pa.large_list(pa.null())),
                pa.array([[], [None]], type=pa.large_list(pa.null())),
            ],
                                       names=[
                                           "sequence_w",
                                           "sequence_x",
                                       ])
        ], [
            "context_a", "context_b", "context_c", "context_d",
            _TEST_SEQUENCE_COLUMN_NAME
        ])),
    dict(
        testcase_name="with_schema_no_typed_examples",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: INT
        }
        feature {
          name: "context_b"
          type: FLOAT
        }
        feature {
          name: "context_c"
          type: BYTES
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_x"
              type: INT
            }
            feature {
              name: "sequence_y"
              type: FLOAT
            }
            feature {
              name: "sequence_z"
              type: BYTES
            }
          }
        }""",
        sequence_examples_text_proto=_TEST_SEQUENCE_EXAMPLES_NONE_TYPED,
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None], type=pa.large_list(pa.int64())),
            pa.array([None, None], type=pa.large_list(pa.float32())),
            pa.array([None, None], type=pa.large_list(pa.large_binary())),
            pa.StructArray.from_arrays([
                pa.array([[], [None]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
                pa.array([None, None],
                         type=pa.large_list(pa.large_list(pa.float32()))),
                pa.array([None, None],
                         type=pa.large_list(pa.large_list(pa.large_binary())))
            ],
                                       names=[
                                           "sequence_x", "sequence_y",
                                           "sequence_z"
                                       ])
        ], ["context_a", "context_b", "context_c", _TEST_SEQUENCE_COLUMN_NAME
           ])),
    dict(
        testcase_name="build_nulls_for_unseen_feature",
        schema_text_proto="""
        feature {
          name: "context_u"
          type: BYTES
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_u"
              type: INT
            }
          }
        }
        """,
        sequence_examples_text_proto=[
            _TYPED_SEQUENCE_EXAMPLE, _UNTYPED_SEQUENCE_EXAMPLE,
            _SOME_FEATURES_TYPED_SEQUENCE_EXAMPLE,
            _EMPTY_VALUES_LIST_SEQUENCE_EXAMPLE
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None],
                     type=pa.large_list(pa.large_binary())),
            pa.StructArray.from_arrays([
                pa.array([None, None, None, None],
                         type=pa.large_list(pa.large_list(pa.int64())))
            ],
                                       names=["sequence_u"]),
        ], ["context_u", _TEST_SEQUENCE_COLUMN_NAME])),
    dict(
        testcase_name="build_null_for_unset_kind",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: BYTES
        }
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_a"
              type: INT
            }
          }
        }
        """,
        sequence_examples_text_proto=[
            """
        context { feature { key: "context_a" value { } } }
        feature_lists {
          feature_list { key: 'sequence_a' value { } }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.large_list(pa.large_binary())),
            pa.StructArray.from_arrays(
                [pa.array([[]], type=pa.large_list(pa.large_list(pa.int64())))],
                names=["sequence_a"]),
        ], ["context_a", _TEST_SEQUENCE_COLUMN_NAME])),
    dict(
        testcase_name="schema_does_not_contain_sequence_feature",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: BYTES
        }
        """,
        sequence_examples_text_proto=[
            """
        context { feature { key: "context_a" value { } } }
        feature_lists {
          feature_list { key: 'sequence_a' value { } }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.large_list(pa.large_binary())),
        ], ["context_a"])),
    dict(
        testcase_name="duplicate_context_feature_names_in_schema",
        schema_text_proto="""
        feature {
          name: "context_a"
          type: BYTES
        }
        # Note that the second feature "context_a" will be ignored.
        feature {
          name: "context_a"
          type: INT
        }
        """,
        sequence_examples_text_proto=[
            """
        context { feature { key: "context_a" value { } } }
        feature_lists {
          feature_list { key: 'sequence_a' value { } }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.large_list(pa.large_binary())),
        ], ["context_a"])),
    dict(
        testcase_name="duplicate_sequence_feature_names_in_schema",
        schema_text_proto="""
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "sequence_a"
              type: INT
            }
            # Note that the second feature "sequence_a" will be ignored.
            feature {
              name: "sequence_a"
              type: BYTES
            }
          }
        }
        """,
        sequence_examples_text_proto=[
            """
        feature_lists {
          feature_list { key: 'sequence_a' value { } }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.StructArray.from_arrays(
                [pa.array([[]], type=pa.large_list(pa.large_list(pa.int64())))],
                names=["sequence_a"]),
        ], [_TEST_SEQUENCE_COLUMN_NAME])),
    dict(
        testcase_name="feature_lists_with_no_sequence_features",
        schema_text_proto=None,
        sequence_examples_text_proto=["""
        feature_lists {}
        """],
        expected=pa.RecordBatch.from_arrays([
            pa.StructArray.from_buffers(pa.struct([]), 1, [None]),
        ], [_TEST_SEQUENCE_COLUMN_NAME])),
    dict(
        testcase_name="without_schema_only_context_features",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            """
        context {
          feature {
            key: 'context_a'
            value {
              int64_list {
                value: [1, 2]
              }
            }
          }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([[1, 2]], type=pa.large_list(pa.int64())),
        ], ["context_a"])),
    dict(
        testcase_name="without_schema_only_sequence_features",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            """
        feature_lists {
          feature_list {
            key: 'sequence_x'
            value {
              feature {
                int64_list {
                  value: [1, 2]
                }
              }
            }
          }
        }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.StructArray.from_arrays([
                pa.array([[[1, 2]]],
                         type=pa.large_list(pa.large_list(pa.int64()))),
            ],
                                       names=["sequence_x"])
        ], [_TEST_SEQUENCE_COLUMN_NAME])),
]

_INVALID_INPUT_CASES = [
    dict(
        testcase_name="context_feature_actual_type_mismatches_schema_type",
        schema_text_proto="""
        feature {
          name: "a"
          type: BYTES
        }
        """,
        sequence_examples_text_proto=[
            """
        context { feature { key: "a" value { float_list { value: [] } } } }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected bytes_list, found float_list "
            "for feature \"a\""),
    ),
    dict(
        testcase_name="sequence_feature_actual_type_mismatches_schema_type",
        schema_text_proto="""
        feature {
          name: "##SEQUENCE##"
          type: STRUCT
          struct_domain {
            feature {
              name: "a"
              type: BYTES
            }
          }
        }
        """,
        sequence_examples_text_proto=[
            """
        feature_lists {
          feature_list {
            key: 'a'
            value {
              feature { float_list { value: []  } }
            }
          }
        }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected bytes_list, found float_list "
            "for sequence feature \"a\""),
    ),
    dict(
        testcase_name="context_feature_no_schema_mixed_type",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            """
        context { feature { key: "a" value { float_list { value: [] } } } }
        """, """
        context { feature { key: "a" value { int64_list { value: [] } } } }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected float_list, found int64_list"
            " for feature \"a\""),
    ),
    dict(
        testcase_name="sequence_feature_no_schema_mixed_type",
        schema_text_proto=None,
        sequence_examples_text_proto=[
            """
        feature_lists {
          feature_list {
            key: 'a'
            value {
              feature { float_list { value: []  } }
            }
          }
        }
        """, """
        feature_lists {
          feature_list {
            key: 'a'
            value {
              feature { int64_list { value: []  } }
            }
          }
        }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected float_list, found int64_list"
            " for sequence feature \"a\""),
    ),
]


class SequenceExamplesToRecordBatchDecoderTest(parameterized.TestCase):

  @parameterized.named_parameters(*_DECODE_CASES)
  def test_decode(self, schema_text_proto, sequence_examples_text_proto,
                  expected):
    serialized_sequence_examples = [
        text_format.Parse(pbtxt,
                          tf.train.SequenceExample()).SerializeToString()
        for pbtxt in sequence_examples_text_proto
    ]
    serialized_schema = None
    if schema_text_proto is not None:
      serialized_schema = text_format.Parse(
          schema_text_proto, schema_pb2.Schema()).SerializeToString()

    if serialized_schema:
      coder = sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
          _TEST_SEQUENCE_COLUMN_NAME,
          serialized_schema)
    else:
      coder = sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
          _TEST_SEQUENCE_COLUMN_NAME)

    result = coder.DecodeBatch(serialized_sequence_examples)
    self.assertIsInstance(result, pa.RecordBatch)
    self.assertTrue(
        result.equals(expected),
        "actual: {}\n expected:{}".format(result, expected))

    if serialized_schema is not None:
      self.assertTrue(coder.ArrowSchema().equals(result.schema))

  @parameterized.named_parameters(*_INVALID_INPUT_CASES)
  def test_invalid_input(self, schema_text_proto, sequence_examples_text_proto,
                         error, error_msg_regex):
    serialized_sequence_examples = [
        text_format.Parse(pbtxt,
                          tf.train.SequenceExample()).SerializeToString()
        for pbtxt in sequence_examples_text_proto
    ]
    serialized_schema = None
    if schema_text_proto is not None:
      serialized_schema = text_format.Parse(
          schema_text_proto, schema_pb2.Schema()).SerializeToString()

    if serialized_schema:
      coder = sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
          _TEST_SEQUENCE_COLUMN_NAME, serialized_schema)
    else:
      coder = sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
          _TEST_SEQUENCE_COLUMN_NAME)

    with self.assertRaisesRegex(error, error_msg_regex):
      coder.DecodeBatch(serialized_sequence_examples)

  def test_sequence_feature_column_name_not_struct_in_schema(self):
    schema_text_proto = """
        feature {
          name: "##SEQUENCE##"
          type: INT
        }
        """
    serialized_schema = text_format.Parse(
        schema_text_proto, schema_pb2.Schema()).SerializeToString()

    error_msg_regex = (
        "Found a feature in the schema with the sequence_feature_column_name "
        r"\(i.e., ##SEQUENCE##\) that is not a struct.*")

    with self.assertRaisesRegex(RuntimeError, error_msg_regex):
      sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
          _TEST_SEQUENCE_COLUMN_NAME, serialized_schema)


if __name__ == "__main__":
  absltest.main()
