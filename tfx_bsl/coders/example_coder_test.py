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

from absl.testing import absltest


# TODO(zhuo): add more test cases.
class ExampleCoderTest(absltest.TestCase):

  def test_simple(self):

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'int_feature':
                    tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[1, 2, 3])),
                'float_feature':
                    tf.train.Feature(
                        float_list=tf.train.FloatList(value=[1.0])),
                'bytes_feature':
                    tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[b'hello', b'world'])),
            }))
    result = example_coder.ExamplesToRecordBatch([example.SerializeToString()])
    self.assertIsInstance(result, pa.RecordBatch)


if __name__ == '__main__':
  absltest.main()
