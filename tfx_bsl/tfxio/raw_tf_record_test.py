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
"""Tests for tfx_bsl.tfxio.raw_tf_record."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tfx_bsl.tfxio import raw_tf_record

from absl.testing import absltest


FLAGS = flags.FLAGS
_RAW_RECORDS = [b"record1", b"record2", b"record3"]


def _WriteRawRecords(filename):
  with tf.io.TFRecordWriter(filename, "GZIP") as w:
    for r in _RAW_RECORDS:
      w.write(r)


class RawTfRecordTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(RawTfRecordTest, cls).setUpClass()
    cls._raw_record_file = os.path.join(
        FLAGS.test_tmpdir, "rawtfrecordtest", "input.recordio.gz")
    tf.io.gfile.makedirs(os.path.dirname(cls._raw_record_file))
    _WriteRawRecords(cls._raw_record_file)

  def testRecordBatchAndTensorAdapter(self):
    column_name = "raw_record"
    tfxio = raw_tf_record.RawTfRecordTFXIO(self._raw_record_file, column_name)
    self.assertTrue(
        tfxio.ArrowSchema(),
        pa.schema([pa.field(column_name, pa.list_(pa.binary()))]))

    def _AssertFn(record_batches):
      self.assertLen(record_batches, 1)
      record_batch = record_batches[0]
      self.assertTrue(record_batch.schema.equals(tfxio.ArrowSchema()))
      self.assertTrue(record_batch.columns[0].equals(
          pa.array([[r] for r in _RAW_RECORDS],
                   type=pa.list_(pa.binary()))))
      tensor_adapter = tfxio.TensorAdapter()
      tensors = tensor_adapter.ToBatchTensors(record_batch)
      self.assertLen(tensors, 1)
      self.assertIn(column_name, tensors)

    with beam.Pipeline() as p:
      record_batch_pcoll = p | tfxio.BeamSource(batch_size=len(_RAW_RECORDS))
      beam_testing_util.assert_that(record_batch_pcoll, _AssertFn)

  def testProject(self):
    column_name = "raw_record"
    tfxio = raw_tf_record.RawTfRecordTFXIO(self._raw_record_file, column_name)
    projected = tfxio.Project([column_name])
    self.assertTrue(tfxio.ArrowSchema().equals(projected.ArrowSchema()))
    self.assertEqual(tfxio.TensorRepresentations(),
                     projected.TensorRepresentations())

    with self.assertRaises(AssertionError):
      tfxio.Project(["some_other_name"])

if __name__ == "__main__":
  absltest.main()
