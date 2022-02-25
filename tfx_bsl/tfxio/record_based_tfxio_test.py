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

import os
import tempfile

from typing import Any

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_test_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import record_based_tfxio

from absl.testing import absltest
from absl.testing import parameterized


FLAGS = flags.FLAGS


def _WriteTfRecord(path, records):
  with tf.io.TFRecordWriter(path) as w:
    for r in records:
      w.write(r)


class RecordBasedTfxioTest(parameterized.TestCase):

  def testReadTfRecord(self):
    tmp_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
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

  @parameterized.named_parameters(*[
      dict(
          testcase_name="simple",
          input_record_batch=pa.record_batch([pa.array([[1], [2]])],
                                             ["feature1"]),
          raw_records=[b"aa", b"bb"],
          expected_raw_record_column=pa.array(
              [[b"aa"], [b"bb"]], type=pa.large_list(pa.large_binary()))),
      dict(
          testcase_name="with_record_index",
          input_record_batch=pa.record_batch(
              [pa.array([[1], [2], [3]]),
               pa.array([[0], [1], [1]])], ["feature1", "record_index"]),
          raw_records=[b"aa", b"bb"],
          expected_raw_record_column=pa.array([[b"aa"], [b"bb"], [b"bb"]],
                                              type=pa.large_list(
                                                  pa.large_binary())),
          record_index_column_name="record_index",
      ),
      dict(
          testcase_name="with_record_index_empty_input",
          input_record_batch=pa.record_batch([
              pa.array([], type=pa.list_(pa.int64())),
              pa.array([], type=pa.large_list(pa.int32()))
          ], ["feature1", "record_index"]),
          raw_records=[b"aa", b"bb"],
          expected_raw_record_column=pa.array(
              [], type=pa.large_list(pa.large_binary())),
          record_index_column_name="record_index",
      )
  ])
  def testAppendRawRecordColumn(
      self, input_record_batch,
      raw_records,
      expected_raw_record_column,
      record_index_column_name=None):
    column_name = "raw_record"
    output_record_batch = record_based_tfxio.AppendRawRecordColumn(
        record_batch=input_record_batch, column_name=column_name,
        raw_records=raw_records,
        record_index_column_name=record_index_column_name)
    self.assertEqual(
        output_record_batch.num_columns,
        input_record_batch.num_columns + 1)
    for i in range(input_record_batch.num_columns):
      self.assertTrue(
          input_record_batch.column(i).equals(output_record_batch.column(i)))

    self.assertEqual(
        output_record_batch.schema.names[output_record_batch.num_columns - 1],
        column_name)
    self.assertTrue(
        output_record_batch.column(output_record_batch.num_columns - 1)
        .equals(expected_raw_record_column))

  def testOverridableRecordBasedTFXIO(self):
    tmp_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    file1 = os.path.join(tmp_dir, "tfrecord1")
    file1_records = [b"aa", b"bb"]
    _WriteTfRecord(file1, file1_records)

    def _CheckRecords(actual, expected):
      for a, e in zip(actual, expected):
        self.assertDictEqual(a.to_pydict(), e)

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(bytes)
    def _RawRecordBeamSource(pipeline: Any):
      return pipeline | beam.io.ReadFromTFRecord(file1 + "*")

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _RawRecordsToRecordBatch(pcoll, batch_size):
      batch_size = 1 if not batch_size else batch_size

      class _CreateRBDoFn(beam.DoFn):

        def process(self, examples):
          return [
              pa.RecordBatch.from_arrays([pa.array(examples)], ["column_name"])
          ]

      return (pcoll | beam.BatchElements(batch_size)
              | beam.ParDo(_CreateRBDoFn()))

    tfxio = record_based_tfxio.OverridableRecordBasedTFXIO(
        telemetry_descriptors=None,
        logical_format="tfrecord",
        physical_format="tf_example",
        raw_record_beam_source=beam.ptransform_fn(_RawRecordBeamSource),
        raw_record_to_record_batch=beam.ptransform_fn(_RawRecordsToRecordBatch))

    expected = [{"column_name": [b"aa"]}, {"column_name": [b"bb"]}]
    with beam.Pipeline() as p:
      record_pcoll = p | tfxio.BeamSource()
      beam_test_util.assert_that(
          record_pcoll,
          lambda actual: _CheckRecords(actual, expected))


if __name__ == "__main__":
  absltest.main()
