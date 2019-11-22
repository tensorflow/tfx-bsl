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
"""Tests for tfx_bsl.tfxio.tensor_representation_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six
from tfx_bsl.tfxio import tensor_representation_util

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2


_IS_LEGACY_SCHEMA = (
    'generate_legacy_feature_spec' in
    schema_pb2.Schema.DESCRIPTOR.fields_by_name)


_INFER_TEST_CASES = [
    # Test different shapes
    {
        'testcase_name':
            'fixed_len_vector',
        'ascii_proto':
            """
          feature: {
            name: "x" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
        """,
        'expected': {
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                  dim {
                    size: 1
                  }
                }
              }"""
        }
    },
    {
        'testcase_name':
            'fixed_len_matrix',
        'ascii_proto':
            """
          feature: {
            name: "x" type: INT shape: {dim {size: 2} dim {size: 2}}
            presence: {min_fraction: 1}
          }
        """,
        'expected': {
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                  dim {
                    size: 2
                  }
                  dim {
                    size: 2
                  }
                }
              }"""
        }
    },
    {
        'testcase_name': 'var_len',
        'ascii_proto': """feature: {name: "x" type: INT}""",
        'expected': {
            'x':
                """
              varlen_sparse_tensor {
                column_name: "x"
              }
            """
        }
    },
    {
        'testcase_name':
            'deprecated_feature',
        'ascii_proto':
            """
          feature: {name: "x" type: INT lifecycle_stage: DEPRECATED}
        """,
        'expected': {}
    },
]

_LEGACY_INFER_TEST_CASES = [
    {
        'testcase_name':
            'fixed_len_scalar_no_default_legacy',
        'ascii_proto':
            """
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 1}
          }
        """,
        'expected': {
            'dummy':
                """
              dense_tensor {
                column_name: "dummy"
                shape {
                }
                default_value {
                  int_value: -1
                }
              }""",
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                }
              }""",
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name':
            'fixed_len_vector_no_default_legacy',
        'ascii_proto':
            """
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT value_count: {min: 5 max: 5}
            presence: {min_fraction: 1}
          }
        """,
        'expected': {
            'dummy':
                """
              dense_tensor {
                column_name: "dummy"
                shape {
                }
                default_value {
                  int_value: -1
                }
              }""",
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                  dim {
                    size: 5
                  }
                }
              }""",
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name':
            'var_len_legacy',
        'ascii_proto':
            """
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT
          }
        """,
        'expected': {
            'dummy':
                """
              dense_tensor {
                column_name: "dummy"
                shape {
                }
                default_value {
                  int_value: -1
                }
              }""",
            'x':
                """
              varlen_sparse_tensor {
                column_name: "x"
              }
            """
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name':
            'fixed_len_scalar_int_with_default_legacy',
        'ascii_proto':
            """
          feature: {
            name: "x" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        'expected': {
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                }
                default_value {
                  int_value: -1
                }
              }
            """
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name':
            'fixed_len_scalar_string_with_default_legacy',
        'ascii_proto':
            """
          feature: {
            name: "x" type: BYTES value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        'expected': {
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                }
                default_value {
                  bytes_value: ""
                }
              }
            """
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name':
            'fixed_len_scalar_float_with_default_legacy',
        'ascii_proto':
            """
          feature: {
            name: "x" type: FLOAT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        'expected': {
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                }
                default_value {
                  float_value: -1.0
                }
              }
            """
        },
        'generate_legacy_feature_spec':
            True,
    },
    {
        'testcase_name': 'deprecated_feature_legacy',
        'ascii_proto': '''
          feature: {name: "x" type: INT deprecated: true}
        ''',
        'expected': {},
    }
]


class TensorRepresentationUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      *(_INFER_TEST_CASES + _LEGACY_INFER_TEST_CASES))
  def testInferTensorRepresentationsFromSchema(
      self, ascii_proto, expected, generate_legacy_feature_spec=False):
    # Skip a test if it's testing legacy logic but the schema is not the
    # legacy schema.
    if not _IS_LEGACY_SCHEMA and generate_legacy_feature_spec:
      print('Skipping test case: ', self.id(), file=sys.stderr)
      return
    schema = text_format.Parse(ascii_proto, schema_pb2.Schema())
    if _IS_LEGACY_SCHEMA:
      schema.generate_legacy_feature_spec = generate_legacy_feature_spec
    expected_protos = {
        k: text_format.Parse(pbtxt, schema_pb2.TensorRepresentation())
        for k, pbtxt in six.iteritems(expected)
    }
    self.assertEqual(
        expected_protos,
        tensor_representation_util.InferTensorRepresentationsFromSchema(
            schema))

  def testGetTensorRepresentationsFromSchema(self):
    self.assertIsNone(
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            schema_pb2.Schema()))
    schema = text_format.Parse("""
      tensor_representation_group {
        key: ""
        value {
          tensor_representation {
            key: "a"
            value { }
          }
        }
      }
    """, schema_pb2.Schema())
    result = tensor_representation_util.GetTensorRepresentationsFromSchema(
        schema)
    self.assertTrue(result)
    self.assertIn('a', result)


if __name__ == '__main__':
  absltest.main()
