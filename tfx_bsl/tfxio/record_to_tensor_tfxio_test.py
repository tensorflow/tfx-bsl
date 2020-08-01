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
"""Tests for tfx_bsl.tfxio.record_to_tensor_tfxio."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import tf_graph_record_decoder
from tfx_bsl.tfxio import record_to_tensor_tfxio
from tfx_bsl.tfxio import telemetry_test_util

from google.protobuf import text_format
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2


FLAGS = flags.FLAGS


class _DecoderForTesting(tf_graph_record_decoder.TFGraphRecordDecoder):

  def __init__(self):
    super(_DecoderForTesting, self).__init__("DecoderForTesting")

  def _decode_record_internal(self, record):
    indices = tf.transpose(
        tf.stack([
            tf.range(tf.size(record), dtype=tf.int64),
            tf.zeros(tf.size(record), dtype=tf.int64)
        ]))
    sparse_tensor = tf.SparseTensor(
        values=record, indices=indices, dense_shape=[tf.size(record), 1])
    return {
        "st1": sparse_tensor,
        "st2": sparse_tensor
    }


_RECORDS = [b"aaa", b"bbb"]
_TELEMETRY_DESCRIPTORS = ["Some", "Component"]


def _write_input(path):
  tf.io.gfile.makedirs(os.path.dirname(path))
  with tf.io.TFRecordWriter(path) as w:
    for r in _RECORDS:
      w.write(r)


class RecordToTensorTfxioTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(RecordToTensorTfxioTest, self).setUp()
    unique_dir = uuid.uuid4().hex
    self._decoder_path = os.path.join(
        FLAGS.test_tmpdir, "recordtotensortfxiotest", unique_dir)
    tf_graph_record_decoder.save_decoder(
        _DecoderForTesting(), self._decoder_path)

    self._input_path = os.path.join(
        FLAGS.test_tmpdir, "recordtotensortfxiotest", unique_dir, "input")
    _write_input(self._input_path)

  @parameterized.named_parameters(*[
      dict(testcase_name="attach_raw_records", attach_raw_records=True),
      dict(testcase_name="noattach_raw_records", attach_raw_records=False)
  ])
  def test_simple(self, attach_raw_records):
    raw_record_column_name = "_raw_records" if attach_raw_records else None
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, self._decoder_path, _TELEMETRY_DESCRIPTORS,
        raw_record_column_name=raw_record_column_name)
    expected_fields = [
        pa.field("st1", pa.list_(pa.binary())),
        pa.field("st2", pa.list_(pa.binary())),
    ]
    if attach_raw_records:
      raw_record_column_type = (
          pa.large_list(pa.large_binary())
          if tfxio._can_produce_large_types else pa.list_(pa.binary()))
      expected_fields.append(
          pa.field(raw_record_column_name, raw_record_column_type))
    self.assertTrue(tfxio.ArrowSchema().equals(
        pa.schema(expected_fields)), tfxio.ArrowSchema())
    self.assertEqual(
        tfxio.TensorRepresentations(), {
            "st1":
                text_format.Parse(
                    """varlen_sparse_tensor { column_name: "st1" }""",
                    schema_pb2.TensorRepresentation()),
            "st2":
                text_format.Parse(
                    """varlen_sparse_tensor { column_name: "st2" }""",
                    schema_pb2.TensorRepresentation())
        })

    tensor_adapter = tfxio.TensorAdapter()
    self.assertEqual(tensor_adapter.TypeSpecs(),
                     _DecoderForTesting().output_type_specs())

    def _assert_fn(list_of_rb):
      self.assertLen(list_of_rb, 1)
      rb = list_of_rb[0]
      self.assertTrue(rb.schema.equals(tfxio.ArrowSchema()))
      tensors = tensor_adapter.ToBatchTensors(rb)
      self.assertLen(tensors, 2)
      for tensor_name in ("st1", "st2"):
        self.assertIn(tensor_name, tensors)
        st = tensors[tensor_name]
        self.assertAllEqual(st.values, _RECORDS)
        self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
        self.assertAllEqual(st.dense_shape, [2, 1])

    p = beam.Pipeline()
    rb_pcoll = p | tfxio.BeamSource(batch_size=len(_RECORDS))
    beam_testing_util.assert_that(rb_pcoll, _assert_fn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(
        self, pipeline_result, _TELEMETRY_DESCRIPTORS,
        "tensor", "tfrecords_gzip")

  def test_project(self):
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, self._decoder_path, ["some", "component"])
    projected = tfxio.Project(["st1"])
    self.assertIn("st1", projected.TensorRepresentations())
    self.assertNotIn("st2", projected.TensorRepresentations())
    tensor_adapter = projected.TensorAdapter()

    def _assert_fn(list_of_rb):
      self.assertLen(list_of_rb, 1)
      rb = list_of_rb[0]
      tensors = tensor_adapter.ToBatchTensors(rb)
      self.assertLen(tensors, 1)
      self.assertIn("st1", tensors)
      st = tensors["st1"]
      self.assertAllEqual(st.values, _RECORDS)
      self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
      self.assertAllEqual(st.dense_shape, [2, 1])

    with beam.Pipeline() as p:
      rb_pcoll = p | tfxio.BeamSource(batch_size=len(_RECORDS))
      beam_testing_util.assert_that(rb_pcoll, _assert_fn)


if __name__ == "__main__":
  # Do not run these tests under TF1.x -- not supported.
  if tf.__version__ >= "2":
    tf.test.main()
