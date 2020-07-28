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
"""Tests for tfx_bsl.tfxio.record_based_tfxio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import tensorflow as tf
from tfx_bsl.tfxio import record_based_tfxio

from absl.testing import absltest


FLAGS = flags.FLAGS


class RecordBasedTfxioTest(absltest.TestCase):

  def testReadTfRecord(self):
    tmp_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    def _WriteTfRecord(path, records):
      with tf.io.TFRecordWriter(path) as w:
        for r in records:
          w.write(r)

    file1 = os.path.join(tmp_dir, "tfrecord1")
    file1_records = [b"aa", b"bb"]
    _WriteTfRecord(file1, file1_records)
    file2 = os.path.join(tmp_dir, "tfrecord2")
    file2_records = [b"cc", b"dd"]
    _WriteTfRecord(file2, file2_records)

    def _CheckRecords(actual, expected):
      self.assertEqual(set(actual), set(expected))

    # Test reading multiple file patterns.
    with beam.Pipeline() as p:
      record_pcoll = p | record_based_tfxio.ReadTfRecord(
          [file1 + "*", file2 + "*"])
      beam_test_util.assert_that(
          record_pcoll,
          lambda actual: _CheckRecords(actual, file1_records + file2_records))

if __name__ == "__main__":
  absltest.main()
