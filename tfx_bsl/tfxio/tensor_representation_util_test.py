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

import numpy as np
import tensorflow as tf

from tfx_bsl.arrow import path
from tfx_bsl.tfxio import tensor_representation_util

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2

_IS_LEGACY_SCHEMA = ('generate_legacy_feature_spec' in
                     schema_pb2.Schema.DESCRIPTOR.fields_by_name)

_ALL_EXAMPLE_CODER_TYPES = {
    schema_pb2.FeatureType.INT: tf.int64,
    schema_pb2.FeatureType.FLOAT: tf.float32,
    schema_pb2.FeatureType.BYTES: tf.string
}


def _DisqualifiedFeatureTestCases(generate_legacy_feature_spec=False):
  result = []
  for stage in tensor_representation_util._DISQUALIFYING_LIFECYCLE_STAGES:
    stage_name = schema_pb2.LifecycleStage.Name(stage)
    testcase_name_prefix = 'legacy' if generate_legacy_feature_spec else ''
    result.append(
        dict(
            testcase_name=(
                f'{testcase_name_prefix}_disqualified_feature_{stage_name}'),
            ascii_proto=f"""
                       feature {{
                         name: "feature"
                         type: INT
                         lifecycle_stage: {stage_name}
                       }}
                       """,
            expected={},
            generate_legacy_feature_spec=generate_legacy_feature_spec))
  return result

_INFER_TEST_CASES = [
    # Test different shapes
    dict(
        testcase_name='fixed_len_vector',
        ascii_proto="""
          feature: {
            name: "x" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
        """,
        expected={
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
        }),
    dict(
        testcase_name='fixed_len_matrix',
        ascii_proto="""
          feature: {
            name: "x" type: INT shape: {dim {size: 2} dim {size: 2}}
            presence: {min_fraction: 1}
          }
        """,
        expected={
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
        }),
    dict(
        testcase_name='var_len',
        ascii_proto="""feature: {name: "x" type: INT}""",
        expected={
            'x':
                """
              varlen_sparse_tensor {
                column_name: "x"
              }
            """
        }),
    dict(
        testcase_name='var_len_ragged',
        ascii_proto="""
          feature: {name: "x" type: INT}
          represent_variable_length_as_ragged: true
          """,
        expected={
            'x':
                """
              ragged_tensor {
                feature_path {
                  step: "x"
                }
              }
            """
        }),
    dict(
        testcase_name='sparse',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }""",
        expected={
            'x':
                """
              sparse_tensor {
                index_column_names: ["index_key"]
                value_column_name: "value_key"
                dense_shape {
                  dim {
                    size: 10
                  }
                }
              }
            """
        }),
    dict(
        testcase_name='sparse_already_sorted',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 11 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
            is_sorted: true
          }""",
        expected={
            'x':
                """
              sparse_tensor {
                index_column_names: ["index_key"]
                value_column_name: "value_key"
                dense_shape {
                  dim {
                    size: 12
                  }
                }
                already_sorted: true
              }
            """
        }),
    dict(
        testcase_name='sparse_already_sorted_false',
        ascii_proto="""
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            value_feature {name: "value_key"}
            is_sorted: false
          }""",
        expected={
            'x':
                """
              sparse_tensor {
                value_column_name: "value_key"
                dense_shape {
                }
              }
            """
        }),
    dict(
        testcase_name='deprecated_feature',
        ascii_proto="""
          feature: {name: "x" type: INT lifecycle_stage: DEPRECATED}
        """,
        expected={}),
    dict(
        testcase_name='sparse_feature_rank_0',
        ascii_proto="""
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            value_feature {name: "value_key"}
          }
        """,
        expected={
            'x':
                """
              sparse_tensor {
                value_column_name: "value_key"
                dense_shape { }
              }
            """
        }),
    dict(
        testcase_name='sparse_feature_rank_2',
        ascii_proto="""
          feature {
            name: "index_key_1"
            type: INT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "index_key_2"
            type: INT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key_1"}
            index_feature {name: "index_key_2"}
            value_feature {name: "value_key"}
          }
        """,
        expected={
            'x':
                """
              sparse_tensor {
                index_column_names: "index_key_1"
                index_column_names: "index_key_2"
                value_column_name: "value_key"
                dense_shape {
                  dim {
                    size: 1
                  }
                  dim {
                    size: 1
                  }
                }
              }
            """
        }),
    dict(
        testcase_name='sparse_feature_no_index_int_domain',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        """,
        expected={
            'x':
                """
              sparse_tensor {
                index_column_names: "index_key"
                value_column_name: "value_key"
                dense_shape {
                  dim {
                    size: -1
                  }
                }
              }
            """
        }),
    dict(
        testcase_name='struct',
        ascii_proto="""
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
              feature {
                name: "string_feature"
                type: BYTES
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }""",
        expected={
            '##SEQUENCE##.int_feature':
                """
                ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "int_feature"
                  }
                }
                """,
            '##SEQUENCE##.string_feature':
                """
                ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "string_feature"
                  }
                }
                """
        }),
] + _DisqualifiedFeatureTestCases()

_LEGACY_INFER_TEST_CASES = [
    dict(
        testcase_name='fixed_len_scalar_no_default_legacy',
        ascii_proto="""
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 1}
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='fixed_len_vector_no_default_legacy',
        ascii_proto="""
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT value_count: {min: 5 max: 5}
            presence: {min_fraction: 1}
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='var_len_legacy',
        ascii_proto="""
          feature: {
            name: "dummy" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
          feature: {
            name: "x" type: INT
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='var_len_ragged_legacy',
        ascii_proto="""
          feature: {
            name: "x" type: INT
          }
          represent_variable_length_as_ragged: true
        """,
        expected={
            'x':
                """
              ragged_tensor {
                feature_path {
                  step: "x"
                }
              }
            """
        },
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='fixed_len_scalar_int_with_default_legacy',
        ascii_proto="""
          feature: {
            name: "x" type: INT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='fixed_len_scalar_string_with_default_legacy',
        ascii_proto="""
          feature: {
            name: "x" type: BYTES value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='fixed_len_scalar_float_with_default_legacy',
        ascii_proto="""
          feature: {
            name: "x" type: FLOAT value_count: {min: 1 max: 1}
            presence: {min_fraction: 0}
          }
        """,
        expected={
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
        generate_legacy_feature_spec=True,
    ),
    dict(
        testcase_name='deprecated_feature_legacy',
        ascii_proto="""
          feature: {name: "x" type: INT deprecated: true}
        """,
        expected={},
    ),
    dict(
        testcase_name='struct_legacy',
        ascii_proto="""
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
              feature {
                name: "string_feature"
                type: BYTES
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }""",
        expected={
            '##SEQUENCE##.int_feature':
                """
                ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "int_feature"
                  }
                }
                """,
            '##SEQUENCE##.string_feature':
                """
                ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "string_feature"
                  }
                }
                """,
        },
        generate_legacy_feature_spec=True,
    ),
] + _DisqualifiedFeatureTestCases(True)

_MIXED_SCHEMA_INFER_TEST_CASES = [
    dict(
        testcase_name='fixed_len_vector_and_tensor_representation',
        ascii_proto="""
          feature: {
            name: "x" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
          feature: {
            name: "value"
            type: FLOAT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "y"
                value: {
                  ragged_tensor {
                    feature_path { step: "value" }
                  }
                }
              }
            }
          }
        """,
        expected={
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                  dim {
                    size: 1
                  }
                }
              }""",
            'y':
                """
                ragged_tensor {
                  feature_path { step: "value" }
                }
                """
        },
        schema_is_mixed=True,
    ),
    dict(
        testcase_name='tensor_representation_only',
        ascii_proto="""
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "y"
                value: {
                  ragged_tensor {
                    feature_path { step: "value" }
                  }
                }
              }
            }
          }
        """,
        expected={
            'y':
                """
                ragged_tensor {
                  feature_path { step: "value" }
                }
                """
        },
        schema_is_mixed=True,
    ),
    dict(
        testcase_name='feature_and_tensor_representation_same_name',
        ascii_proto="""
          feature: {
            name: "x" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
          feature: {
            name: "y" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
          feature: {
            name: "value"
            type: FLOAT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "y"
                value: {
                  ragged_tensor {
                    feature_path { step: "value" }
                  }
                }
              }
            }
          }
        """,
        expected={
            'x':
                """
              dense_tensor {
                column_name: "x"
                shape {
                  dim {
                    size: 1
                  }
                }
              }""",
            'y':
                """
                ragged_tensor {
                  feature_path { step: "value" }
                }
                """,
        },
        schema_is_mixed=True,
    ),
    dict(
        testcase_name='source_column_and_tensor_representation_same_name',
        ascii_proto="""
          feature: {
            name: "y" type: INT shape: {dim {size: 1}}
            presence: {min_fraction: 1}
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "y"
                value: {
                  ragged_tensor {
                    feature_path { step: "y" }
                  }
                }
              }
            }
          }
        """,
        expected={
            'y':
                """
                ragged_tensor {
                  feature_path { step: "y" }
                }
                """,
        },
        schema_is_mixed=True,
    ),
    dict(
        testcase_name='struct_feature',
        ascii_proto="""
          feature {
            name: "int_feature"
            type: INT
            value_count {
              min: 1
              max: 1
            }
          }
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
              feature {
                name: "string_feature"
                type: BYTES
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "int_feature"
                            } } }
              }
            }
          }
        """,
        expected={
            'int_feature':
                """ragged_tensor {
                  feature_path {
                    step: "int_feature"
                  }
                }""",
            '##SEQUENCE##.string_feature':
                """ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "string_feature"
                  }
                }""",
            '##SEQUENCE##.int_feature':
                """ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "int_feature"
                  }
                }""",
        },
        schema_is_mixed=True,
    ),
    dict(
        testcase_name='struct_feature_with_tensor_representations',
        ascii_proto="""
          feature {
            name: "int_feature"
            type: INT
            value_count {
              min: 1
              max: 1
            }
          }
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
              feature {
                name: "string_feature"
                type: BYTES
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "seq_string_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "string_feature"
                            } } }
              }
              tensor_representation {
                key: "seq_int_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "int_feature"
                            } } }
              }
            }
          }
        """,
        expected={
            'int_feature':
                """
                varlen_sparse_tensor {
                  column_name: "int_feature"
                }
                """,
            'seq_string_feature':
                """ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "string_feature"
                  }
                }""",
            'seq_int_feature':
                """ragged_tensor {
                  feature_path {
                    step: "##SEQUENCE##"
                    step: "int_feature"
                  }
                }""",
        },
        schema_is_mixed=True,
    ),
]

_INVALID_SCHEMA_INFER_TEST_CASES = [
    dict(
        testcase_name='sparse_feature_no_index_int_domain_min',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          """,
        error_msg=(r'Cannot determine dense shape of sparse feature x. '
                   r'The minimum domain value of index feature index_key'
                   r' is not set.')),
    dict(
        testcase_name='sparse_feature_non_zero_index_int_domain_min',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 1 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          """,
        error_msg=(r'Only 0-based index features are supported. Sparse '
                   r'feature x has index feature index_key whose '
                   r'minimum domain value is 1')),
    dict(
        testcase_name='sparse_feature_no_index_int_domain_max',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          """,
        error_msg=(r'Cannot determine dense shape of sparse feature x. '
                   r'The maximum domain value of index feature index_key'
                   r' is not set.')),
    dict(
        testcase_name='sparse_feature_missing_index_key',
        ascii_proto="""
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        """,
        error_msg=(r'sparse_feature x referred to index feature '
                   r'index_key which did not exist in the schema')),
    dict(
        testcase_name='sparse_feature_missing_value_key',
        ascii_proto="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        """,
        error_msg=(r'sparse_feature x referred to value feature '
                   r'value_key which did not exist in the schema')),
]

_GET_SOURCE_COLUMNS_TEST_CASES = [
    dict(
        testcase_name='oneof_unspecified',
        pbtxt='',
        expected=[],
    ),
    dict(
        testcase_name='dense_tensor',
        pbtxt="""
            dense_tensor {
              column_name: "my_column"
              shape {
              }
            }
        """,
        expected=[['my_column']],
    ),
    dict(
        testcase_name='varlen_sparse_tensor',
        pbtxt="""
         varlen_sparse_tensor {
           column_name: "my_column"
         }
         """,
        expected=[['my_column']],
    ),
    dict(
        testcase_name='sparse_tensor',
        pbtxt="""
          sparse_tensor {
            index_column_names: "idx1"
            index_column_names: "idx2"
            value_column_name: "value"
          }
        """,
        expected=[['idx1'], ['idx2'], ['value']],
    ),
    dict(
        testcase_name='ragged_tensor',
        pbtxt="""
          ragged_tensor {
            feature_path {
              step: "value"
            }
            partition { row_length: "row_length_1" }
            partition { uniform_row_length: 2 }
            partition { row_length: "row_length_2" }
          }
        """,
        expected=[['value'], ['row_length_1'], ['row_length_2']],
    ),
]

_GET_SOURCE_VALUE_COLUMNS_TEST_CASES = [
    dict(
        testcase_name='dense_tensor',
        pbtxt="""
            dense_tensor {
              column_name: "my_column"
              shape {
              }
            }
        """,
        expected='my_column',
    ),
    dict(
        testcase_name='varlen_sparse_tensor',
        pbtxt="""
         varlen_sparse_tensor {
           column_name: "my_column"
         }
         """,
        expected='my_column',
    ),
    dict(
        testcase_name='sparse_tensor',
        pbtxt="""
          sparse_tensor {
            index_column_names: "idx1"
            index_column_names: "idx2"
            value_column_name: "value"
          }
        """,
        expected='value',
    ),
    dict(
        testcase_name='ragged_tensor',
        pbtxt="""
          ragged_tensor {
            feature_path {
              step: "my_column"
            }
            partition { row_length: "row_length" }
          }
        """,
        expected='my_column',
    ),
]

_PROJECT_TENSOR_REPRESENTATIONS_IN_SCHEMA_TEST_CASES = [
    dict(
        testcase_name='simple_features',
        schema="""
          feature {
            name: "int_feature"
            type: INT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "float_feature"
            type: FLOAT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
        """,
        tensor_names=('int_feature', 'string_feature'),
        expected_projection="""
          feature {
            name: "int_feature"
            type: INT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "int_feature"
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='simple_tensor_representations',
        schema="""
          feature {
            name: "int_feature"
            type: INT
          }
          feature {
            name: "float_feature"
            type: FLOAT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "int_feature"
                  }
                }
              }
              tensor_representation {
                key: "float_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "float_feature"
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_names=('float_feature', 'string_feature'),
        expected_projection="""
          feature {
            name: "float_feature"
            type: FLOAT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "float_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "float_feature"
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='composite_tensor_representations',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"]
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_names=('x',),
        expected_projection="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"]
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='refer_same_source_column',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "int_feature"
            type: INT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["int_feature"]
                  }
                }
              }
              tensor_representation {
                key: "int_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "int_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_names=('x',),
        expected_projection="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "int_feature"
            type: INT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["int_feature"]
                  }
                }
              }
            }
          }
        """),
    dict(
        testcase_name='missing_source_column',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"] # No "index_key" feature.
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_names=('x',),
        expected_error=(
            'TensorRepresentations source columns {index_key} are '
            'not present in the schema.'),
    ),
    dict(
        testcase_name='missing_tensor_representation',
        schema="""
          feature {
            name: "int_feature"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_names=('int_feature',),
        expected_error=(
            "Unable to project {'int_feature'} because they were not in the "
            'original or inferred TensorRepresentations.'),
    ),
    dict(
        testcase_name='sparse_feature',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 10 }
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        """,
        tensor_names=('x',),
        expected_projection="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
            int_domain {
              min: 0
              max: 10
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    dense_shape {
                      dim {
                        size: 11
                      }
                    }
                    index_column_names: "index_key"
                    value_column_name: "value_key"
                    already_sorted: true
                  }
                }
              }
            }
          }
        """),
    dict(
        testcase_name='sparse_feature_legacy',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 10 }
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
        """,
        tensor_names=('x',),
        expected_error=(
            "Unable to project {'x'} because they were not in the "
            'original or inferred TensorRepresentations.'
        ),  # sparse_feature inference is not implemented in legacy logic.
        generate_legacy_feature_spec=True),
]

_VALIDATE_TENSOR_REPRESENTATIONS_IN_SCHEMA_TEST_CASES = [
    dict(
        testcase_name='_simple_features',
        schema="""
          feature {
            name: "int_feature"
            type: INT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "float_feature"
            type: FLOAT
            int_domain { min: 0 max: 0 }
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
        """,
        error='TensorRepresentations are not found in the schema.',
    ),
    dict(
        testcase_name='_simple_tensor_representations',
        schema="""
          feature {
            name: "int_feature"
            type: INT
          }
          feature {
            name: "float_feature"
            type: FLOAT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "int_feature"
                  }
                }
              }
              tensor_representation {
                key: "float_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "float_feature"
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='_composite_tensor_representations',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: "group_name"
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"]
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        tensor_representation_group_name='group_name',
    ),
    dict(
        testcase_name='_struct_features',
        schema="""
        feature {
          name: "int_feature"
          type: INT
          value_count {
            min: 1
            max: 1
          }
        }
        feature {
          name: "float_feature"
          type: FLOAT
          value_count {
            min: 4
            max: 4
          }
        }
        feature {
          name: "struct_feature"
          type: STRUCT
          struct_domain {
            feature {
              name: "int_feature"
              type: INT
              value_count {
                min: 0
                max: 2
              }
            }
            feature {
              name: "string_feature"
              type: BYTES
              value_count {
                min: 0
                max: 2
              }
            }
          }
        }
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "int_feature"
              value { varlen_sparse_tensor { column_name: "int_feature" } }
            }
            tensor_representation {
              key: "float_feature"
              value { varlen_sparse_tensor { column_name: "float_feature" } }
            }
            tensor_representation {
              key: "seq_string_feature"
              value {
                ragged_tensor {
                  feature_path {
                    step: "struct_feature"
                    step: "string_feature"
                  }
                }
              }
            }
            tensor_representation {
              key: "seq_int_feature"
              value {
                ragged_tensor {
                  feature_path {
                    step: "struct_feature"
                    step: "int_feature"
                  }
                }
              }
            }
          }
        }
      """,
    ),
    dict(
        testcase_name='_tensor_representations_with_sparse_feature',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "index_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            is_sorted: true
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"]
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='_refer_same_source_column',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "int_feature"
            type: INT
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["int_feature"]
                  }
                }
              }
              tensor_representation {
                key: "int_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "int_feature"
                  }
                }
              }
            }
          }
        """,
    ),
    dict(
        testcase_name='_missing_source_column',
        schema="""
          feature {
            name: "value_key"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "x"
                value {
                  sparse_tensor {
                    value_column_name: "value_key"
                    index_column_names: ["index_key"] # No "index_key" feature.
                  }
                }
              }
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        error=(
            'Features referred in TensorRepresentations but not found in the '
            'schema: {index_key}'),
    ),
    dict(
        testcase_name='_missing_tensor_representation',
        schema="""
          feature {
            name: "int_feature"
            type: INT
          }
          feature {
            name: "string_feature"
            type: BYTES
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "string_feature"
                value {
                  varlen_sparse_tensor {
                    column_name: "string_feature"
                  }
                }
              }
            }
          }
        """,
        error=('Features present in the schema but not referred in any '
               'TensorRepresentation: {int_feature}'),
    ),
]


def _MakeFixedLenFeatureTestCases():
  result = []
  base_tensor_rep_textpb = """
                dense_tensor {{
                  column_name: "dense_column"
                  shape {{
                    dim {{
                      size: 2
                    }}
                  }}
                  default_value {{
                    {0}
                  }}
                }}"""
  for t, dtype in _ALL_EXAMPLE_CODER_TYPES.items():
    if t == schema_pb2.FeatureType.FLOAT:
      default_value_textpb = 'float_value: 1.0'
      default_values = [1.0, 1.0]
      expected_parsed_results = np.array([1.0, 1.0])
    elif t == schema_pb2.FeatureType.INT:
      default_value_textpb = 'int_value: 1'
      default_values = [1, 1]
      expected_parsed_results = np.array([1, 1])
    elif t == schema_pb2.FeatureType.BYTES:
      default_value_textpb = 'bytes_value: "default"'
      default_values = [b'default', b'default']
      expected_parsed_results = np.array([b'default', b'default'])

    tensor_rep_textpb = base_tensor_rep_textpb.format(default_value_textpb)
    expected_feature = tf.io.FixedLenFeature(
        shape=[2], dtype=dtype, default_value=default_values)
    result.append({
        'testcase_name':
            'FixedLenFeature_{}'.format(schema_pb2.FeatureType.Name(t)),
        'tensor_representation':
            tensor_rep_textpb,
        'feature_type':
            t,
        'tf_example':
            text_format.Parse('', tf.train.Example()).SerializeToString(),
        'expected_feature':
            expected_feature,
        'expected_parsed_results':
            expected_parsed_results
    })
  return result


def _MakeFixedLenFeatureNoDefaultTestCases():
  result = []
  tensor_representation_textpb = """
                dense_tensor {
                  column_name: "dense_column"
                  shape {
                    dim {
                      size: 4
                    }
                  }
                }"""
  example_textpb = """
  features {{
    feature {{
      key: "feat"
      value {{ {0} }}
    }}
  }}
  """
  for t, dtype in _ALL_EXAMPLE_CODER_TYPES.items():
    if t == schema_pb2.FeatureType.FLOAT:
      value_textpb = """float_list { value: [1.0, 2.0, 3.0, 4.0] }"""
      expected_parsed_results = np.array([1.0, 2.0, 3.0, 4.0])
    elif t == schema_pb2.FeatureType.INT:
      value_textpb = """int64_list { value: [1, 2, 3, 4] }"""
      expected_parsed_results = np.array([1, 2, 3, 4])
    elif t == schema_pb2.FeatureType.BYTES:
      value_textpb = """bytes_list { value: ['one', 'two', 'three', 'four'] }"""
      expected_parsed_results = np.array([b'one', b'two', b'three', b'four'])
    expected_feature = tf.io.FixedLenFeature(
        shape=[4], dtype=dtype, default_value=None)
    result.append({
        'testcase_name':
            'FixedLenFeatureNoDefault_{}'.format(
                schema_pb2.FeatureType.Name(t)),
        'tensor_representation':
            tensor_representation_textpb,
        'feature_type':
            t,
        'tf_example':
            text_format.Parse(
                example_textpb.format(value_textpb),
                tf.train.Example()).SerializeToString(),
        'expected_feature':
            expected_feature,
        'expected_parsed_results':
            expected_parsed_results
    })
  return result


def _MakeVarLenSparseFeatureTestCases():
  result = []
  tensor_representation_textpb = """
                 varlen_sparse_tensor {
                   column_name: "varlen_sparse_tensor"
                 }"""
  if tf.executing_eagerly():
    sparse_tensor_factory = tf.SparseTensor
  else:
    sparse_tensor_factory = tf.compat.v1.SparseTensorValue
  expected_parsed_results = sparse_tensor_factory(
      indices=np.zeros((0, 1)), values=np.array([]), dense_shape=[0])
  for t, dtype in _ALL_EXAMPLE_CODER_TYPES.items():
    expected_feature = tf.io.VarLenFeature(dtype=dtype)
    result.append({
        'testcase_name':
            'VarLenSparseFeature_{}'.format(schema_pb2.FeatureType.Name(t)),
        'tensor_representation':
            tensor_representation_textpb,
        'feature_type':
            t,
        'tf_example':
            text_format.Parse('', tf.train.Example()).SerializeToString(),
        'expected_feature':
            expected_feature,
        'expected_parsed_results':
            expected_parsed_results
    })
  return result


def _MakeSparseFeatureTestCases():
  result = []
  tensor_representation_textpb_template = """
                 sparse_tensor {{
                   index_column_names: ["key"]
                   value_column_name: "value"
                   dense_shape {{
                     dim {{
                       size: {dim}
                     }}
                   }}
                 }}"""
  if tf.executing_eagerly():
    sparse_tensor_factory = tf.SparseTensor
  else:
    sparse_tensor_factory = tf.compat.v1.SparseTensorValue
  expected_parsed_results = sparse_tensor_factory(
      indices=np.zeros((0, 1)), values=np.array([]), dense_shape=[1])
  for t, dtype in _ALL_EXAMPLE_CODER_TYPES.items():
    expected_feature = tf.io.SparseFeature(
        index_key=['key'], value_key='value', dtype=dtype, size=[1])
    result.append({
        'testcase_name':
            'SparseFeature_{}'.format(schema_pb2.FeatureType.Name(t)),
        'tensor_representation':
            tensor_representation_textpb_template.format(dim=1),
        'feature_type':
            t,
        'tf_example':
            text_format.Parse('', tf.train.Example()).SerializeToString(),
        'expected_feature':
            expected_feature,
        'expected_parsed_results':
            expected_parsed_results
    })
  result.append({
      'testcase_name':
          'SparseFeature_unknown_dense_shape',
      'tensor_representation':
          tensor_representation_textpb_template.format(dim=-1),
      'feature_type':
          schema_pb2.FeatureType.BYTES,
      'tf_example':
          text_format.Parse('', tf.train.Example()).SerializeToString(),
      'expected_feature':
          tf.io.SparseFeature(
              index_key=['key'], value_key='value', dtype=tf.string, size=[-1]),
      'expected_parsed_results':
          sparse_tensor_factory(
              indices=np.zeros((0, 1)), values=np.array([]), dense_shape=[-1])
  })
  for already_sorted in (True, False):
    result.append({
        'testcase_name':
            f'SparseFeature_already_sorted_{already_sorted}',
        'tensor_representation':
            f"""
                  sparse_tensor {{
                    index_column_names: ["key"]
                    value_column_name: "value"
                    dense_shape {{
                      dim {{
                        size: 1
                      }}
                    }}
                    already_sorted: {'true' if already_sorted else 'false'}
                  }}""",
        'feature_type':
            schema_pb2.FeatureType.BYTES,
        'tf_example':
            text_format.Parse('', tf.train.Example()).SerializeToString(),
        'expected_feature':
            tf.io.SparseFeature(
                index_key=['key'],
                value_key='value',
                dtype=tf.string,
                size=[1],
                already_sorted=already_sorted),
        'expected_parsed_results':
            expected_parsed_results
    })
  return result


def _MakeRaggedFeatureTestCases():
  result = []

  tensor_representation_textpb = """
    ragged_tensor {
      feature_path { step: "value" }
      partition { row_length: "row_length" }
      row_partition_dtype: INT32
    }"""
  if tf.executing_eagerly():
    ragged_tensor_factory = tf.RaggedTensor.from_row_splits
  else:
    ragged_tensor_factory = tf.compat.v1.ragged.RaggedTensorValue
  example_textpb = """
  features {{
    feature {{
      key: "value"
      value {{ {0} }}
    }}
    feature {{
      key: "row_length"
      value {{ int64_list {{ value: [3, 1] }} }}
    }}
  }}
  """
  for t, dtype in _ALL_EXAMPLE_CODER_TYPES.items():
    if t == schema_pb2.FeatureType.FLOAT:
      value_textpb = """float_list { value: [1.0, 2.0, 3.0, 4.0] }"""
      expected_parsed_values = np.array([1.0, 2.0, 3.0, 4.0])
    elif t == schema_pb2.FeatureType.INT:
      value_textpb = """int64_list { value: [1, 2, 3, 4] }"""
      expected_parsed_values = np.array([1, 2, 3, 4])
    elif t == schema_pb2.FeatureType.BYTES:
      value_textpb = """bytes_list { value: ['one', 'two', 'three', 'four'] }"""
      expected_parsed_values = np.array([b'one', b'two', b'three', b'four'])
    expected_parsed_results = ragged_tensor_factory(
        values=expected_parsed_values,
        row_splits=np.array([0, 3, 4], dtype=np.int32))

    expected_feature = tf.io.RaggedFeature(
        value_key='value',
        dtype=dtype,
        partitions=(tf.io.RaggedFeature.RowLengths('row_length'),),
        row_splits_dtype=tf.int32)

    result.append({
        'testcase_name':
            'RaggedFeature_{}'.format(schema_pb2.FeatureType.Name(t)),
        'tensor_representation':
            tensor_representation_textpb,
        'feature_type':
            t,
        'tf_example':
            text_format.Parse(
                example_textpb.format(value_textpb),
                tf.train.Example()).SerializeToString(),
        'expected_feature':
            expected_feature,
        'expected_parsed_results':
            expected_parsed_results
    })
  return result


_PARSE_EXAMPLE_TEST_CASES = _MakeFixedLenFeatureTestCases(
) + _MakeFixedLenFeatureNoDefaultTestCases(
) + _MakeVarLenSparseFeatureTestCases() + _MakeSparseFeatureTestCases(
) + _MakeRaggedFeatureTestCases()

_CREATE_SEQUENCE_EXAMPLE_PARSER_CONFIG_TEST_CASES = [
    dict(
        testcase_name='_no_struct_feature',
        schema="""
          feature {
            name: "int_feature"
            type: INT
            value_count {
              min: 1
              max: 1
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value { varlen_sparse_tensor { column_name: "int_feature" } }
              }
            }
          }
        """,
        expected_context_features={
            'int_feature': tf.io.VarLenFeature(dtype=tf.int64)
        },
        expected_sequence_features={},
    ),
    dict(
        testcase_name='_simple',
        schema="""
          feature {
            name: "int_feature"
            type: INT
            value_count {
              min: 1
              max: 1
            }
          }
          feature {
            name: "float_feature"
            type: FLOAT
            value_count {
              min: 4
              max: 4
            }
          }
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
              feature {
                name: "string_feature"
                type: BYTES
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "int_feature"
                value { varlen_sparse_tensor { column_name: "int_feature" } }
              }
              tensor_representation {
                key: "float_feature"
                value { varlen_sparse_tensor { column_name: "float_feature" } }
              }
              tensor_representation {
                key: "seq_string_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "string_feature"
                            } } }
              }
              tensor_representation {
                key: "seq_int_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "int_feature"
                            } } }
              }
            }
          }
        """,
        expected_context_features={
            'int_feature': tf.io.VarLenFeature(dtype=tf.int64),
            'float_feature': tf.io.VarLenFeature(dtype=tf.float32)
        },
        expected_sequence_features={
            'seq_string_feature':
                tf.io.RaggedFeature(
                    dtype=tf.string,
                    value_key='string_feature',
                    row_splits_dtype=tf.int64,
                    partitions=[]),
            'seq_int_feature':
                tf.io.RaggedFeature(
                    dtype=tf.int64,
                    value_key='int_feature',
                    row_splits_dtype=tf.int64,
                    partitions=[])
        },
    ),
    dict(
        testcase_name='_no_primitive_feature',
        schema="""
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "seq_int_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "int_feature"
                            } } }
              }
            }
          }
        """,
        expected_context_features={},
        expected_sequence_features={
            'seq_int_feature':
                tf.io.RaggedFeature(
                    dtype=tf.int64,
                    value_key='int_feature',
                    row_splits_dtype=tf.int64,
                    partitions=[])
        },
    ),
    dict(
        testcase_name='_sparse_feature',
        schema="""
          feature {
            name: "index_key"
            type: INT
            int_domain { min: 0 max: 9 }
          }
          feature {
            name: "value_key"
            type: INT
          }
          sparse_feature {
            name: "x"
            index_feature {name: "index_key"}
            value_feature {name: "value_key"}
          }
          feature {
            name: "##SEQUENCE##"
            type: STRUCT
            struct_domain {
              feature {
                name: "int_feature"
                type: INT
                value_count {
                  min: 0
                  max: 2
                }
              }
            }
          }
          tensor_representation_group {
            key: ""
            value {
              tensor_representation {
                key: "seq_int_feature"
                value { ragged_tensor {
                            feature_path {
                              step: "##SEQUENCE##" step: "int_feature"
                            } } }
              }
            }
          }
        """,
        expected_context_features={},
        expected_sequence_features={
            'seq_int_feature':
                tf.io.RaggedFeature(
                    dtype=tf.int64,
                    value_key='int_feature',
                    row_splits_dtype=tf.int64,
                    partitions=[])
        },
    ),
]


class TensorRepresentationUtilTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      *(_INFER_TEST_CASES + _LEGACY_INFER_TEST_CASES +
        _MIXED_SCHEMA_INFER_TEST_CASES))
  def testInferTensorRepresentationsFromSchema(
      self,
      ascii_proto,
      expected,
      generate_legacy_feature_spec=False,
      schema_is_mixed=False):
    if not _IS_LEGACY_SCHEMA and generate_legacy_feature_spec:
      raise self.skipTest('This test exersizes legacy inference logic, but the '
                          'schema is not legacy schema.')
    schema = text_format.Parse(ascii_proto, schema_pb2.Schema())
    if _IS_LEGACY_SCHEMA:
      schema.generate_legacy_feature_spec = generate_legacy_feature_spec
    expected_protos = {
        k: text_format.Parse(pbtxt, schema_pb2.TensorRepresentation())
        for k, pbtxt in expected.items()
    }
    if not schema_is_mixed:
      self.assertEqual(
          expected_protos,
          tensor_representation_util.InferTensorRepresentationsFromSchema(
              schema))
    self.assertEqual(
        expected_protos,
        tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
            schema))

  @parameterized.named_parameters(*_INVALID_SCHEMA_INFER_TEST_CASES)
  def testInferTensorRepresentationsFromSchemaInvalidSchema(
      self, ascii_proto, error_msg, generate_legacy_feature_spec=False):
    if not _IS_LEGACY_SCHEMA and generate_legacy_feature_spec:
      raise self.skipTest('This test exersizes legacy inference logic, but the '
                          'schema is not legacy schema.')
    schema = text_format.Parse(ascii_proto, schema_pb2.Schema())
    if _IS_LEGACY_SCHEMA:
      schema.generate_legacy_feature_spec = generate_legacy_feature_spec
    with self.assertRaisesRegex(ValueError, error_msg):
      tensor_representation_util.InferTensorRepresentationsFromSchema(schema)
    with self.assertRaisesRegex(ValueError, error_msg):
      tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
          schema)

  def testGetTensorRepresentationsFromSchema(self):
    self.assertIsNone(
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            schema_pb2.Schema()))
    schema = text_format.Parse(
        """
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

  @parameterized.named_parameters(*_GET_SOURCE_COLUMNS_TEST_CASES)
  def testGetSourceColumnsFromTensorRepresentation(self, pbtxt, expected):
    self.assertEqual(
        [path.ColumnPath(e) for e in expected],
        tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
            text_format.Parse(pbtxt, schema_pb2.TensorRepresentation())))

  @parameterized.named_parameters(*_GET_SOURCE_VALUE_COLUMNS_TEST_CASES)
  def testGetSourceValueColumnFromTensorRepresentation(self, pbtxt, expected):
    self.assertEqual(
        path.ColumnPath(expected),
        tensor_representation_util.GetSourceValueColumnFromTensorRepresentation(
            text_format.Parse(pbtxt, schema_pb2.TensorRepresentation())))

  @test_util.run_all_in_graph_and_eager_modes
  @parameterized.named_parameters(*_PARSE_EXAMPLE_TEST_CASES)
  def testCreateTfExampleParserConfig(self, tensor_representation, feature_type,
                                      tf_example, expected_feature,
                                      expected_parsed_results):
    tensor_representation = text_format.Parse(tensor_representation,
                                              schema_pb2.TensorRepresentation())
    feature = tensor_representation_util.CreateTfExampleParserConfig(
        tensor_representation, feature_type)

    # Checks that the parser configs are correct.
    for actual_arg, expected_arg in zip(feature, expected_feature):
      self.assertAllEqual(actual_arg, expected_arg)

    # Checks that the parser configs can be used with tf.io.parse_example()
    actual_tensors = tf.io.parse_single_example(tf_example, {'feat': feature})
    actual = actual_tensors['feat']
    if isinstance(actual, (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
      self.assertAllEqual(actual.values, expected_parsed_results.values)
      self.assertAllEqual(actual.indices, expected_parsed_results.indices)
      self.assertAllEqual(actual.dense_shape,
                          expected_parsed_results.dense_shape)
    else:
      self.assertAllEqual(actual, expected_parsed_results)

  def testCreateTfExampleParserConfigInvalidDefaultValue(self):
    tensor_representation = text_format.Parse(
        """
                dense_tensor {
                  column_name: "dense_column"
                  shape {
                    dim {
                      size: 1
                    }
                  }
                  default_value {
                    int_value: -1
                  }
                }""", schema_pb2.TensorRepresentation())
    feature_type = schema_pb2.FLOAT
    with self.assertRaisesRegex(
        ValueError, 'FeatureType:.* is incompatible with default_value:.*'):
      tensor_representation_util.CreateTfExampleParserConfig(
          tensor_representation, feature_type)

  def testCreateTfExampleParserConfigRagged(self):
    feature_type = schema_pb2.INT
    tensor_representation = text_format.Parse(
        """
                ragged_tensor {
                  feature_path {
                    step: "foo"
                    step: "ragged_feature"
                  }
                }""", schema_pb2.TensorRepresentation())
    with self.assertRaisesRegex(
        ValueError, ('Parsing spec from a RaggedTensor with multiple steps in '
                     'feature_path is not implemented.')):
      tensor_representation_util.CreateTfExampleParserConfig(
          tensor_representation, feature_type)

  @parameterized.named_parameters(
      *(_PROJECT_TENSOR_REPRESENTATIONS_IN_SCHEMA_TEST_CASES))
  def testProjectTfmdSchema(self,
                            schema,
                            tensor_names,
                            expected_projection=None,
                            expected_error=None,
                            generate_legacy_feature_spec=False):
    schema = text_format.Parse(schema, schema_pb2.Schema())
    if not _IS_LEGACY_SCHEMA and generate_legacy_feature_spec:
      raise self.skipTest('This test exersizes legacy inference logic, but the '
                          'schema is not legacy schema.')

    if _IS_LEGACY_SCHEMA:
      schema.generate_legacy_feature_spec = generate_legacy_feature_spec
    if expected_error is None:
      self.assertEqual(
          text_format.Parse(expected_projection, schema_pb2.Schema()),
          tensor_representation_util.ProjectTensorRepresentationsInSchema(
              schema, tensor_names))
    else:
      with self.assertRaisesRegex(ValueError, expected_error):
        _ = tensor_representation_util.ProjectTensorRepresentationsInSchema(
            schema, tensor_names)

  @parameterized.named_parameters(
      *(_VALIDATE_TENSOR_REPRESENTATIONS_IN_SCHEMA_TEST_CASES))
  def testValidateTensorRepresentationsInSchema(
      self,
      schema,
      tensor_representation_group_name=tensor_representation_util
      ._DEFAULT_TENSOR_REPRESENTATION_GROUP,
      error=None):
    schema = text_format.Parse(schema, schema_pb2.Schema())
    if error is None:
      tensor_representation_util.ValidateTensorRepresentationsInSchema(
          schema, tensor_representation_group_name)
    else:
      with self.assertRaisesRegex(ValueError, error):
        tensor_representation_util.ValidateTensorRepresentationsInSchema(
            schema, tensor_representation_group_name)

  @parameterized.named_parameters(
      *_CREATE_SEQUENCE_EXAMPLE_PARSER_CONFIG_TEST_CASES)
  @test_util.run_all_in_graph_and_eager_modes
  def testCreateTfSequenceExampleParserConfig(self, schema,
                                              expected_context_features,
                                              expected_sequence_features):
    context_features, sequence_features = (
        tensor_representation_util.CreateTfSequenceExampleParserConfig(
            text_format.Parse(schema, schema_pb2.Schema())))
    self.assertDictEqual(expected_context_features, context_features)
    self.assertDictEqual(expected_sequence_features, sequence_features)


if __name__ == '__main__':
  absltest.main()
