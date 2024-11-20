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
"""Tests for tfx_bsl.tfxio.test_util."""

import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import test_util

from google.protobuf import text_format
from absl.testing import absltest


class TestUtilTest(absltest.TestCase):

  def testGetRecordBatches(self):
    tfxio = test_util.InMemoryTFExampleRecord()
    examples = [text_format.Parse("""
    features {
      feature {
        key: "f1"
        value {
          int64_list {
            value: 123
          }
        }
      }
    }""", tf.train.Example()).SerializeToString()]
    def _AssertFn(record_batches):
      self.assertLen(record_batches, 1)
      record_batch = record_batches[0]
      self.assertEqual(record_batch.num_rows, 1)
      self.assertEqual(record_batch.num_columns, 1)
      self.assertTrue(record_batch[0].equals(
          pa.array([[123]], type=pa.large_list(pa.int64()))))

    with beam.Pipeline() as p:
      record_batches = p | beam.Create(examples) | tfxio.BeamSource()
      beam_testing_util.assert_that(record_batches, _AssertFn)


if __name__ == "__main__":
  absltest.main()
