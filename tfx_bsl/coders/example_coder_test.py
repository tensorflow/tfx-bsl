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
# limitations under the License.
"""Tests for tfx_bsl.coders.example_coder."""
import pickle
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import tensor_representation_util

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2

_TEST_EXAMPLES = [
    """
   features {
      feature {
        key: "x"
        value { bytes_list { value: [ "a", "b" ] } }
      }
      feature { key: "y" value { float_list { value: [ 1.0, 2.0 ] } } }
      feature { key: "z" value { int64_list { value: [ 4, 5 ] } } }
    }
    """,
    """
    features {
      feature { key: "w" value { } }
      feature { key: "x" value { } }
      feature { key: "y" value { } }
      feature { key: "z" value { } }
    }
    """,
    """
    features {
      feature { key: "v" value { float_list { value: [1.0]} } }
      feature { key: "x" value { } }
      feature { key: "y" value { } }
      feature { key: "z" value { } }
    }
        """,
    """
    features {
      feature { key: "x" value { bytes_list { value: [] } } }
      feature { key: "y" value { float_list { value: [] } } }
      feature { key: "z" value { int64_list { value: [] } } }
    }
        """,
]

_DECODE_CASES = [
    dict(
        testcase_name="without_schema_simple",
        schema_text_proto=None,
        examples_text_proto=_TEST_EXAMPLES,
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None, [1.0], None],
                     type=pa.large_list(pa.float32())),
            pa.array([None, None, None, None], type=pa.null()),
            pa.array([[b"a", b"b"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
            pa.array([[4, 5], None, None, []], type=pa.large_list(pa.int64()))
        ], ["v", "w", "x", "y", "z"])),
    dict(
        testcase_name="with_schema_simple",
        schema_text_proto="""
        feature {
          name: "x"
          type: BYTES
        }
        feature {
          name: "y"
          type: FLOAT
        }
        feature {
          name: "z"
          type: INT
        }""",
        examples_text_proto=_TEST_EXAMPLES,
        expected=pa.RecordBatch.from_arrays([
            pa.array([[b"a", b"b"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
            pa.array([[4, 5], None, None, []], type=pa.large_list(pa.int64()))
        ], ["x", "y", "z"])),
    dict(
        testcase_name="ignore_features_not_in_schema",
        schema_text_proto="""
        feature {
          name: "x"
          type: BYTES
        }
        feature {
          name: "y"
          type: FLOAT
        }
        """,
        examples_text_proto=_TEST_EXAMPLES,
        expected=pa.RecordBatch.from_arrays([
            pa.array([[b"a", b"b"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
        ], ["x", "y"])),
    dict(
        testcase_name="build_nulls_for_unseen_feature",
        schema_text_proto="""
        feature {
          name: "a"
          type: BYTES
        }
        """,
        examples_text_proto=_TEST_EXAMPLES,
        expected=pa.RecordBatch.from_arrays([
            pa.array([None, None, None, None],
                     type=pa.large_list(pa.large_binary())),
        ], ["a"])),
    dict(
        testcase_name="build_null_for_unset_kind",
        schema_text_proto="""
        feature {
          name: "a"
          type: BYTES
        }
        """,
        examples_text_proto=[
            """
        features { feature { key: "a" value { } } }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.large_list(pa.large_binary())),
        ], ["a"])),
    dict(
        testcase_name="duplicate_feature_names_in_schema",
        schema_text_proto="""
        feature {
          name: "a"
          type: BYTES
        }
        # Note that the second feature "a" will be ignored.
        feature {
          name: "a"
          type: INT
        }
        """,
        examples_text_proto=[
            """
        features { feature { key: "a" value { } } }
        """
        ],
        expected=pa.RecordBatch.from_arrays([
            pa.array([None], type=pa.large_list(pa.large_binary())),
        ], ["a"])),
]

_INVALID_INPUT_CASES = [
    dict(
        testcase_name="actual_type_mismatches_schema_type",
        schema_text_proto="""
        feature {
          name: "a"
          type: BYTES
        }
        """,
        examples_text_proto=[
            """
        features { feature { key: "a" value { float_list { value: [] } } } }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected bytes_list, found float_list "
            "for feature \"a\""),
    ),
    dict(
        testcase_name="no_schema_mixed_type",
        schema_text_proto=None,
        examples_text_proto=[
            """
        features { feature { key: "a" value { float_list { value: [] } } } }
        """, """
        features { feature { key: "a" value { int64_list { value: [] } } } }
        """
        ],
        error=RuntimeError,
        error_msg_regex=(
            "Feature had wrong type, expected float_list, found int64_list"
            " for feature \"a\""),
    ),
]


class ExamplesToRecordBatchDecoderTest(parameterized.TestCase):

  @parameterized.named_parameters(*_DECODE_CASES)
  def test_decode(self, schema_text_proto, examples_text_proto, expected):
    serialized_examples = [
        text_format.Parse(pbtxt, tf.train.Example()).SerializeToString()
        for pbtxt in examples_text_proto
    ]
    serialized_schema = None
    if schema_text_proto is not None:
      serialized_schema = text_format.Parse(
          schema_text_proto, schema_pb2.Schema()).SerializeToString()

    coder = example_coder.ExamplesToRecordBatchDecoder(serialized_schema)

    result = coder.DecodeBatch(serialized_examples)
    self.assertIsInstance(result, pa.RecordBatch)
    self.assertTrue(
        result.equals(expected),
        (
            f"\nactual: {result.to_pydict()}\nactual schema:"
            f" {result.schema}\nexpected:{expected.to_pydict()}\nexpected"
            f" schema: {expected.schema}\nencoded: {serialized_examples}"
        ),
    )
    if serialized_schema:
      self.assertTrue(expected.schema.equals(coder.ArrowSchema()))

    # Verify that coder and DecodeBatch can be properly pickled and unpickled.
    # This is necessary for using them in beam.Map.
    coder = pickle.loads(pickle.dumps(coder))
    decode = pickle.loads(pickle.dumps(coder.DecodeBatch))
    result = decode(serialized_examples)
    self.assertIsInstance(result, pa.RecordBatch)
    self.assertTrue(
        result.equals(expected),
        "actual: {}\n expected:{}".format(result, expected))
    if serialized_schema:
      self.assertTrue(expected.schema.equals(coder.ArrowSchema()))

  @parameterized.named_parameters(*_INVALID_INPUT_CASES)
  def test_invalid_input(self, schema_text_proto, examples_text_proto, error,
                         error_msg_regex):
    serialized_examples = [
        text_format.Parse(pbtxt, tf.train.Example()).SerializeToString()
        for pbtxt in examples_text_proto
    ]
    serialized_schema = None
    if schema_text_proto is not None:
      serialized_schema = text_format.Parse(
          schema_text_proto, schema_pb2.Schema()).SerializeToString()

    if serialized_schema:
      coder = example_coder.ExamplesToRecordBatchDecoder(serialized_schema)
    else:
      coder = example_coder.ExamplesToRecordBatchDecoder()

    with self.assertRaisesRegex(error, error_msg_regex):
      coder.DecodeBatch(serialized_examples)

  def test_arrow_schema_not_available_if_tfmd_schema_not_available(self):
    coder = example_coder.ExamplesToRecordBatchDecoder()
    with self.assertRaisesRegex(RuntimeError, "Unable to get the arrow schema"):
      _ = coder.ArrowSchema()

  def test_invalid_feature_type(self):
    serialized_schema = text_format.Parse(
        """
        feature {
          name: "a"
          type: STRUCT
        }
        """, schema_pb2.Schema()).SerializeToString()
    with self.assertRaisesRegex(RuntimeError,
                                "Bad field type for feature: a.*"):
      _ = example_coder.ExamplesToRecordBatchDecoder(serialized_schema)


_ENCODE_TEST_EXAMPLES = [
    """
   features {
      feature {key: "x" value { bytes_list { value: [ "a", "b" ] } } }
      feature { key: "y" value { float_list { value: [ 1.0, 2.0 ] } } }
      feature { key: "z" value { int64_list { value: [ 4, 5 ] } } }
    }
    """,
    """
    features {
      feature { key: "x" value { } }
      feature { key: "y" value { } }
      feature { key: "z" value { } }
    }
    """,
    """
    features {
      feature { key: "x" value { } }
      feature { key: "y" value { } }
      feature { key: "z" value { } }
    }
        """,
    """
    features {
      feature { key: "x" value { bytes_list { value: [] } } }
      feature { key: "y" value { float_list { value: [] } } }
      feature { key: "z" value { int64_list { value: [] } } }
    }
        """,
]

_ENCODE_CASES = [
    dict(
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b"a", b"b"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([[1.0, 2.0], None, None, []], type=pa.list_(pa.float32())),
            pa.array([[4, 5], None, None, []], type=pa.large_list(pa.int64()))
        ], ["x", "y", "z"]),
        examples_text_proto=_ENCODE_TEST_EXAMPLES),
    dict(
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([None, None, [b"a", b"b"]],
                     type=pa.large_list(pa.binary())),
            pa.array([None, None, [1.0, 2.0]], type=pa.large_list(
                pa.float32())),
            pa.array([None, None, [4, 5]], type=pa.list_(pa.int64()))
        ], ["x", "y", "z"]),
        examples_text_proto=list(reversed(_ENCODE_TEST_EXAMPLES[:-1]))),
]

_INVALID_ENCODE_TYPE_CASES = [
    dict(
        record_batch=pa.RecordBatch.from_arrays([pa.array([1, 2, 3])], ["a"]),
        error=RuntimeError,
        error_msg_regex="Expected ListArray or LargeListArray"),
    dict(
        record_batch=pa.RecordBatch.from_arrays(
            [pa.array([[True], [False]], type=pa.large_list(pa.bool_()))],
            ["a"]),
        error=RuntimeError,
        error_msg_regex="Bad field type"),
    dict(
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([[b"a", b"b"], None, None, []],
                     type=pa.large_list(pa.large_binary())),
            pa.array([[1.0, 2.0], None, None, []],
                     type=pa.large_list(pa.float32())),
        ], ["x", "x"]),
        error=RuntimeError,
        error_msg_regex="RecordBatch contains duplicate column names")
]


class RecordBatchToExamplesTest(parameterized.TestCase):

  @parameterized.parameters(*_ENCODE_CASES)
  def test_encode(self, record_batch, examples_text_proto):
    expected_examples = [
        text_format.Parse(pbtxt, tf.train.Example())
        for pbtxt in examples_text_proto
    ]
    coder = example_coder.RecordBatchToExamplesEncoder()
    actual_examples = [
        tf.train.Example.FromString(encoded)
        for encoded in coder.encode(record_batch)
    ]

    self.assertEqual(actual_examples, expected_examples)

  @parameterized.parameters(*_INVALID_ENCODE_TYPE_CASES)
  def test_invalid_input(self, record_batch, error, error_msg_regex):
    with self.assertRaisesRegex(error, error_msg_regex):
      example_coder.RecordBatchToExamplesEncoder().encode(record_batch)


_ENCODE_NESTED_TEST_EXAMPLES = [
    """
    features {
      feature { key: "x" value { bytes_list { value: ["a", "b"] } } }
      feature { key: "y$values" value { float_list { value: [1.0, 2.0] } } }
      feature { key: "y$row_length_1" value { int64_list { value: [1] } } }
      feature { key: "z$values" value { int64_list { value: [4, 5] } } }
      feature { key: "z$row_length_1" value { int64_list { value: [1] } } }
      feature { key: "z$row_length_2" value { int64_list { value: [1, 1] } } }
    }
    """,
    """
    features {
      feature { key: "x" value { } }
      feature { key: "y$values" value { float_list { value: [3.0, 4.] } } }
      feature { key: "y$row_length_1" value { int64_list { value: [1] } } }
      feature { key: "z$values" value { } }
      feature { key: "z$row_length_1" value { } }
      feature { key: "z$row_length_2" value { } }
    }
    """,
    """
    features {
      feature { key: "x" value { } }
      feature { key: "y$values" value { } }
      feature { key: "y$row_length_1" value { } }
      feature { key: "z$values" value { int64_list { value: [6] } } }
      feature { key: "z$row_length_1" value { int64_list { value: [1] } } }
      feature { key: "z$row_length_2" value { int64_list { value: [1, 0] } } }
    }
    """,
    """
    features {
      feature { key: "x" value { bytes_list { value: [] } } }
      feature { key: "y$values" value { float_list { value: [] } } }
      feature { key: "y$row_length_1" value { int64_list { value: [0] } } }
      feature { key: "z$values" value { int64_list { value: [] } } }
      feature { key: "z$row_length_1" value { int64_list { value: [1] } } }
      feature { key: "z$row_length_2" value { int64_list { value: [0, 0] } } }
    }
    """,
]

_ENCODE_NESTED_RECORD_BATCH = pa.RecordBatch.from_arrays(
    [
        pa.array(
            [[b"a", b"b"], None, None, []],
            type=pa.large_list(pa.large_binary()),
        ),
        pa.array(
            [[[1.0, 2.0]], [[3.0, 4.0]], None, [[]]],
            type=pa.large_list(pa.large_list(pa.float32())),
        ),
        pa.array(
            [[[[4], [5]]], None, [[[6], []]], [[[], []]]],
            type=pa.large_list(pa.large_list(pa.large_list(pa.int64()))),
        ),
    ],
    ["x", "y", "z"],
)

_ENCODE_NESTED_SCHEMA = text_format.Parse(
    """
        tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "x"
            value {
              ragged_tensor {
                feature_path { step: "x"}
                partition { uniform_row_length: 2 }
              }
            }
          }
          tensor_representation {
            key: "y"
            value {
              ragged_tensor {
                feature_path { step: "y$values"}
                partition { row_length: "y$row_length_1" }
                partition { uniform_row_length: 2 }
              }
            }
          }
          tensor_representation {
            key: "z"
            value {
              ragged_tensor {
                feature_path { step: "z$values"}
                partition { row_length: "z$row_length_1" }
                partition { uniform_row_length: 2 }
                partition { row_length: "z$row_length_2" }
              }
            }
          }
      }
      }""",
    schema_pb2.Schema(),
)

_ENCODE_NESTED_CASES = [
    # Note that uniform_row_length partitions do not affect encoding.
    dict(
        record_batch=_ENCODE_NESTED_RECORD_BATCH,
        examples_text_proto=_ENCODE_NESTED_TEST_EXAMPLES,
        schema=_ENCODE_NESTED_SCHEMA,
    ),
    # x is not a ragged tensor and is converted as a plain feature.
    dict(
        record_batch=_ENCODE_NESTED_RECORD_BATCH,
        examples_text_proto=_ENCODE_NESTED_TEST_EXAMPLES,
        schema=text_format.Parse(
            """
        tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "y"
            value {
              ragged_tensor {
                feature_path { step: "y$values"}
                partition { row_length: "y$row_length_1" }
                partition { uniform_row_length: 2 }
              }
            }
          }
          tensor_representation {
            key: "z"
            value {
              ragged_tensor {
                feature_path { step: "z$values"}
                partition { row_length: "z$row_length_1" }
                partition { uniform_row_length: 2 }
                partition { row_length: "z$row_length_2" }
              }
            }
          }
      }
      }""",
            schema_pb2.Schema(),
        ),
    ),
    # Ragged tensor representations belong to different groups.
    dict(
        record_batch=_ENCODE_NESTED_RECORD_BATCH,
        examples_text_proto=_ENCODE_NESTED_TEST_EXAMPLES,
        schema=text_format.Parse(
            """
        tensor_representation_group {
        key: "1"
        value {
          tensor_representation {
            key: "z"
            value {
              ragged_tensor {
                feature_path { step: "z$values"}
                partition { row_length: "z$row_length_1" }
                partition { uniform_row_length: 2 }
                partition { row_length: "z$row_length_2" }
              }
            }
          }
      }
      }
      tensor_representation_group {
        key: "2"
        value {
          tensor_representation {
            key: "y"
            value {
              ragged_tensor {
                feature_path { step: "y$values"}
                partition { row_length: "y$row_length_1" }
                partition { uniform_row_length: 2 }
              }
            }
          }
      }
      }""",
            schema_pb2.Schema(),
        ),
    ),
]

_INVALID_ENCODE_NESTED_TYPE_CASES = [
    # Two ragged features have same value feature name.
    dict(
        record_batch=pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [[b"a", b"b"], None, None, []],
                    type=pa.large_list(pa.large_binary()),
                ),
                pa.array(
                    [[[1.0, 2.0]], [[3.0]], None, [[]]],
                    type=pa.large_list(pa.large_list(pa.int64())),
                ),
            ],
            ["x", "y"],
        ),
        error=RuntimeError,
        error_msg_regex=(
            "conflicts with another source column in the same batch."
        ),
        schema=text_format.Parse(
            """
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "x"
                  value {
                    ragged_tensor {
                      feature_path { step: "y$values"}
                    }
                  }
                }
                tensor_representation {
                  key: "y"
                  value {
                    ragged_tensor {
                      feature_path { step: "y$values"}
                    }
                  }
                }
              }
            }""",
            schema_pb2.Schema(),
        ),
    ),
    # Ragged feature is expected to be 2d, but got 1d.
    dict(
        record_batch=pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [[b"a", b"b"], None, None, []],
                    type=pa.large_list(pa.large_binary()),
                )
            ],
            ["x"],
        ),
        error=RuntimeError,
        error_msg_regex=(
            'Error encoding feature "x": Expected 1 partitions, but got '
            "flat values array."
        ),
        schema=text_format.Parse(
            """
            tensor_representation_group {
              key: ""
              value {
                tensor_representation {
                  key: "x"
                  value {
                    ragged_tensor {
                      feature_path { step: "x"}
                      partition { row_length: "x$row_length_1" }
                    }
                  }
                }
              }
            }""",
            schema_pb2.Schema(),
        ),
    ),
    # Batch has a nested list without corresponding RaggedTensor in the schema.
    dict(
        record_batch=pa.RecordBatch.from_arrays(
            [
                pa.array(
                    [[[b"a"], [b"b"]], None, None, [[]]],
                    type=pa.large_list(pa.large_list(pa.large_binary())),
                )
            ],
            ["x"],
        ),
        error=RuntimeError,
        error_msg_regex=(
            'Error encoding feature "x": Nested depth of large_list is larger '
            r"than the number of provided partitions \(or no partitions were "
            r"provided\). Expected 0 partitions. If the column represents "
            "RaggedTensor, then create the encoder with TFMD schema containing "
            "RaggedTensor TensorRepresentations with partitions."
        ),
    ),
]


class RecordBatchToExamplesEncoderTest(
    parameterized.TestCase, tf.test.TestCase
):

  @parameterized.parameters(*(_ENCODE_CASES + _ENCODE_NESTED_CASES))
  def test_encode(self, record_batch, examples_text_proto, schema=None):
    expected_examples = [
        text_format.Parse(pbtxt, tf.train.Example())
        for pbtxt in examples_text_proto
    ]
    coder = example_coder.RecordBatchToExamplesEncoder(schema)
    # Verify that coder can be properly pickled and unpickled.
    coder = pickle.loads(pickle.dumps(coder))
    encoded = coder.encode(record_batch)
    self.assertLen(encoded, len(expected_examples))
    for idx, (expected, actual) in enumerate(zip(expected_examples, encoded)):
      self.assertProtoEquals(
          expected,
          tf.train.Example.FromString(actual),
          msg=f" at position {idx}",
      )

  @parameterized.parameters(*(_INVALID_ENCODE_TYPE_CASES +
                              _INVALID_ENCODE_NESTED_TYPE_CASES))
  def test_invalid_input(self,
                         record_batch,
                         error,
                         error_msg_regex,
                         schema=None):
    schema = (schema or schema_pb2.Schema())
    coder = example_coder.RecordBatchToExamplesEncoder(schema)
    with self.assertRaisesRegex(error, error_msg_regex):
      coder.encode(record_batch)

  def test_encode_is_consistent_with_parse_example(self):
    coder = example_coder.RecordBatchToExamplesEncoder(_ENCODE_NESTED_SCHEMA)
    encoded = tf.constant(coder.encode(_ENCODE_NESTED_RECORD_BATCH))
    tensor_representations = (
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            _ENCODE_NESTED_SCHEMA
        )
    )
    dtypes = {
        "x": schema_pb2.FeatureType.BYTES,
        "y": schema_pb2.FeatureType.FLOAT,
        "z": schema_pb2.FeatureType.INT,
    }
    feature_spec = {
        name: tensor_representation_util.CreateTfExampleParserConfig(
            representation, dtypes[name]
        )
        for name, representation in tensor_representations.items()
    }
    decoded = tf.io.parse_example(encoded, feature_spec)
    expected_values = {
        "x": [[[b"a", b"b"]], [], [], []],
        "y": [[[[1.0, 2.0]]], [[[3.0, 4.0]]], [], [[]]],
        "z": [[[[[4], [5]]]], [], [[[[6], []]]], [[[[], []]]]],
    }
    expected_ragged_ranks = {"x": 1, "y": 2, "z": 4}
    self.assertLen(decoded, len(expected_values))
    for name, expected in expected_values.items():
      actual = decoded[name]
      self.assertEqual(actual.to_list(), expected, msg=f"For {name}")
      self.assertEqual(
          actual.ragged_rank, expected_ragged_ranks[name], msg=f"For {name}"
      )


if __name__ == "__main__":
  absltest.main()
