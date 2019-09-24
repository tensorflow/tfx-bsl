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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tfx_bsl.coders import example_coder
from tfx_bsl.pyarrow_tf import pyarrow as pa
from tfx_bsl.pyarrow_tf import tensorflow as tf

from google.protobuf import text_format
from absl.testing import absltest
from tensorflow_metadata.proto.v0 import schema_pb2


class ExamplesToRecordBatchDecoderTest(absltest.TestCase):
  _EXAMPLES = [
      text_format.Merge(
          """
     features {
        feature {
          key: "x"
          value { bytes_list { value: [ "a", "b" ] } }
        }
        feature { key: "y" value { float_list { value: [ 1.0, 2.0 ] } } }
        feature { key: "z" value { int64_list { value: [ 4, 5 ] } } }
      }
      """, tf.train.Example()),
      text_format.Merge(
          """
      features {
        feature { key: "x" value { } }
        feature { key: "y" value { } }
        feature { key: "z" value { } }
      }
      """, tf.train.Example()),
      text_format.Merge(
          """
      features {
        feature { key: "x" value { } }
        feature { key: "y" value { } }
        feature { key: "z" value { } }
      }
          """, tf.train.Example()),
      text_format.Merge(
          """
      features {
        feature { key: "x" value { bytes_list { value: [] } } }
        feature { key: "y" value { float_list { value: [] } } }
        feature { key: "z" value { int64_list { value: [] } } }
      }
          """, tf.train.Example()),
  ]

  _SERIALIZED_EXAMPLES = [
      e.SerializeToString() for e in _EXAMPLES]

  _SCHEMA = text_format.Merge(
      """feature {
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
      }""", schema_pb2.Schema())

  _EXPECTED_RECORD_BATCH = pa.RecordBatch.from_arrays([
      pa.array([[b"a", b"b"], None, None, []], type=pa.list_(pa.binary())),
      pa.array([[1.0, 2.0], None, None, []], type=pa.list_(pa.float32())),
      pa.array([[4, 5], None, None, []], type=pa.list_(pa.int64()))
  ], ["x", "y", "z"])

  def test_with_schema(self):
    coder = example_coder.ExamplesToRecordBatchDecoder(
        self._SCHEMA.SerializeToString())
    result = coder.DecodeBatch(self._SERIALIZED_EXAMPLES)
    self.assertIsInstance(result, pa.RecordBatch)
    self.assertTrue(result.equals(self._EXPECTED_RECORD_BATCH))

  def test_without_schema(self):
    coder = example_coder.ExamplesToRecordBatchDecoder()
    result = coder.DecodeBatch(self._SERIALIZED_EXAMPLES)
    self.assertIsInstance(result, pa.RecordBatch)
    self.assertTrue(result.equals(self._EXPECTED_RECORD_BATCH))


if __name__ == "__main__":
  help(example_coder.ExamplesToRecordBatchDecoder)
  absltest.main()
