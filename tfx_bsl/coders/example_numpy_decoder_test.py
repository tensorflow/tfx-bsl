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
"""Tests for TFExampleDecoder."""

import sys
import numpy as np
import tensorflow as tf
from tfx_bsl.coders import example_coder

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized

_TF_EXAMPLE_DECODER_TESTS = [
    {
        'testcase_name': 'empty_input',
        'example_proto_text': '''features {}''',
        'decoded_example': {}
    },
    {
        'testcase_name': 'int_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { int64_list { value: [ 1, 2, 3 ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([1, 2, 3], dtype=np.int64)}
    },
    {
        'testcase_name': 'float_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { float_list { value: [ 4.0, 5.0 ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([4.0, 5.0], dtype=np.float32)}
    },
    {
        'testcase_name': 'str_feature_non_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { bytes_list { value: [ 'string', 'list' ] } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([b'string', b'list'],
                                          dtype=object)}
    },
    {
        'testcase_name': 'int_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { int64_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=np.int64)}
    },
    {
        'testcase_name': 'float_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { float_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=np.float32)}
    },
    {
        'testcase_name': 'str_feature_empty',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { bytes_list { } }
            }
          }
        ''',
        'decoded_example': {'x': np.array([], dtype=object)}
    },
    {
        'testcase_name': 'feature_missing',
        'example_proto_text': '''
          features {
            feature {
              key: 'x'
              value { }
            }
          }
        ''',
        'decoded_example': {'x': None}
    },
]


class TFExampleDecoderTest(parameterized.TestCase):
  """Tests for TFExampleDecoder."""

  def _check_decoding_results(self, actual, expected):
    # Check that the numpy array dtypes match.
    self.assertEqual(len(actual), len(expected))
    for key in actual:
      if expected[key] is None:
        self.assertEqual(actual[key], None)
      else:
        self.assertEqual(actual[key].dtype, expected[key].dtype)
        np.testing.assert_equal(actual, expected)

  @parameterized.named_parameters(
      *_TF_EXAMPLE_DECODER_TESTS)
  def test_decode_example(self, example_proto_text, decoded_example):
    example = tf.train.Example()
    text_format.Merge(example_proto_text, example)
    self._check_decoding_results(
        example_coder.ExampleToNumpyDict(example.SerializeToString()),
        decoded_example)

  def test_decode_example_none_ref_count(self):
    example = text_format.Parse(
        '''
          features {
            feature {
              key: 'x'
              value { }
            }
          }
        ''', tf.train.Example())
    before_refcount = sys.getrefcount(None)
    _ = example_coder.ExampleToNumpyDict(example.SerializeToString())
    after_refcount = sys.getrefcount(None)
    self.assertEqual(before_refcount + 1, after_refcount)

if __name__ == '__main__':
  absltest.main()
