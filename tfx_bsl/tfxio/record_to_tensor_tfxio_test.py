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

import os
import tempfile

from absl import flags
import apache_beam as beam
from apache_beam.testing import util as beam_testing_util
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import tf_graph_record_decoder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import record_to_tensor_tfxio
from tfx_bsl.tfxio import telemetry_test_util

from google.protobuf import text_format
from absl.testing import parameterized
from tensorflow_metadata.proto.v0 import schema_pb2


FLAGS = flags.FLAGS


class _DecoderForTesting(tf_graph_record_decoder.TFGraphRecordDecoder):

  def decode_record(self, record):
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


class _DecoderForTestingWithRecordIndex(_DecoderForTesting):

  def decode_record(self, record):
    result = super(
        _DecoderForTestingWithRecordIndex, self).decode_record(record)
    result["ragged_record_index"] = tf.RaggedTensor.from_row_splits(
        values=tf.range(tf.size(record), dtype=tf.int64),
        row_splits=tf.range(tf.size(record) + 1, dtype=tf.int64))
    result["sparse_record_index"] = result["ragged_record_index"].to_sparse()
    return result


class _DecoderForTestingWithRaggedRecordIndex(
    _DecoderForTestingWithRecordIndex):

  @property
  def record_index_tensor_name(self):
    return "ragged_record_index"


class _DecoderForTestingWithSparseRecordIndex(
    _DecoderForTestingWithRecordIndex):

  @property
  def record_index_tensor_name(self):
    return "sparse_record_index"


_RECORDS = [b"aaa", b"bbb"]
_RECORDS_AS_TENSORS = [{
    "st1":
        tf.SparseTensor(values=[b"aaa"], indices=[[0, 0]], dense_shape=[1, 1]),
    "st2":
        tf.SparseTensor(values=[b"aaa"], indices=[[0, 0]], dense_shape=[1, 1])
}, {
    "st1":
        tf.SparseTensor(values=[b"bbb"], indices=[[0, 0]], dense_shape=[1, 1]),
    "st2":
        tf.SparseTensor(values=[b"bbb"], indices=[[0, 0]], dense_shape=[1, 1])
}]
_TELEMETRY_DESCRIPTORS = ["Some", "Component"]


def _write_input():
  result = os.path.join(tempfile.mkdtemp(dir=FLAGS.test_tmpdir), "input")
  with tf.io.TFRecordWriter(result) as w:
    for r in _RECORDS:
      w.write(r)

  return result


def _write_decoder(decoder=_DecoderForTesting()):
  result = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
  tf_graph_record_decoder.save_decoder(decoder, result)
  return result


class RecordToTensorTfxioTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._input_path = _write_input()

  def _assert_sparse_tensor_equal(self, lhs, rhs):
    self.assertAllEqual(lhs.values, rhs.values)
    self.assertAllEqual(lhs.indices, rhs.indices)
    self.assertAllEqual(lhs.dense_shape, rhs.dense_shape)

  # pylint: disable=unnecessary-lambda
  # the create_decoder lambdas may seem unnecessary, but they are picklable, and
  # the classes (not the instances of those classes) are not.
  @parameterized.named_parameters(*[
      dict(testcase_name="attach_raw_records",
           attach_raw_records=True,
           create_decoder=lambda: _DecoderForTesting()),
      dict(testcase_name="attach_raw_records_with_ragged_record_index",
           attach_raw_records=True,
           create_decoder=lambda: _DecoderForTestingWithRaggedRecordIndex()),
      dict(testcase_name="attach_raw_records_with_sparse_record_index",
           attach_raw_records=True,
           create_decoder=lambda: _DecoderForTestingWithSparseRecordIndex()),
      dict(testcase_name="noattach_raw_records",
           attach_raw_records=False,
           create_decoder=lambda: _DecoderForTesting()),
      dict(testcase_name="noattach_raw_records_but_with_record_index",
           attach_raw_records=False,
           create_decoder=lambda: _DecoderForTestingWithSparseRecordIndex()),
      dict(testcase_name="beam_record_tfxio",
           attach_raw_records=False,
           create_decoder=lambda: _DecoderForTesting(),
           beam_record_tfxio=True),
  ])
  # pylint: enable=unnecessary-lambda
  def test_beam_source_and_tensor_adapter(
      self, attach_raw_records, create_decoder, beam_record_tfxio=False):
    decoder = create_decoder()
    raw_record_column_name = "_raw_records" if attach_raw_records else None
    decoder_path = _write_decoder(decoder)
    if beam_record_tfxio:
      tfxio = record_to_tensor_tfxio.BeamRecordToTensorTFXIO(
          saved_decoder_path=decoder_path,
          telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
          physical_format="tfrecords_gzip",
          raw_record_column_name=raw_record_column_name)
    else:
      tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
          self._input_path,
          decoder_path,
          _TELEMETRY_DESCRIPTORS,
          raw_record_column_name=raw_record_column_name)
    expected_tensor_representations = {
        "st1":
            text_format.Parse("""varlen_sparse_tensor { column_name: "st1" }""",
                              schema_pb2.TensorRepresentation()),
        "st2":
            text_format.Parse("""varlen_sparse_tensor { column_name: "st2" }""",
                              schema_pb2.TensorRepresentation())
    }
    if isinstance(decoder, _DecoderForTestingWithRecordIndex):
      expected_fields = [
          pa.field("ragged_record_index", pa.large_list(pa.int64())),
          pa.field("sparse_record_index", pa.large_list(pa.int64())),
          pa.field("st1", pa.large_list(pa.large_binary())),
          pa.field("st2", pa.large_list(pa.large_binary())),
      ]
      expected_tensor_representations["ragged_record_index"] = (
          text_format.Parse(
              """ragged_tensor {
                   feature_path: { step: "ragged_record_index" }
                   row_partition_dtype: INT64
                 }""", schema_pb2.TensorRepresentation()))
      expected_tensor_representations["sparse_record_index"] = (
          text_format.Parse(
              """varlen_sparse_tensor { column_name: "sparse_record_index" }""",
              schema_pb2.TensorRepresentation()))
    else:
      expected_fields = [
          pa.field("st1", pa.large_list(pa.large_binary())),
          pa.field("st2", pa.large_list(pa.large_binary())),
      ]
    if attach_raw_records:
      expected_fields.append(
          pa.field(raw_record_column_name, pa.large_list(pa.large_binary())))
    self.assertTrue(tfxio.ArrowSchema().equals(
        pa.schema(expected_fields)), tfxio.ArrowSchema())

    self.assertEqual(
        tfxio.TensorRepresentations(), expected_tensor_representations)

    tensor_adapter = tfxio.TensorAdapter()
    tensor_adapter_type_specs = tensor_adapter.TypeSpecs()
    for tensor_name, type_spec in decoder.output_type_specs().items():
      self.assertTrue(
          tensor_adapter_type_specs[tensor_name].is_compatible_with(type_spec))

    def _assert_fn(list_of_rb):
      self.assertLen(list_of_rb, 1)
      rb = list_of_rb[0]
      self.assertTrue(rb.schema.equals(tfxio.ArrowSchema()))
      if attach_raw_records:
        self.assertEqual(rb.column(rb.num_columns - 1).flatten().to_pylist(),
                         _RECORDS)
      tensors = tensor_adapter.ToBatchTensors(rb)
      for tensor_name in ("st1", "st2"):
        self.assertIn(tensor_name, tensors)
        st = tensors[tensor_name]
        self.assertAllEqual(st.values, _RECORDS)
        self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
        self.assertAllEqual(st.dense_shape, [2, 1])

    p = beam.Pipeline()
    pipeline_input = (p | beam.Create(_RECORDS)) if beam_record_tfxio else p
    rb_pcoll = pipeline_input | tfxio.BeamSource(batch_size=len(_RECORDS))
    beam_testing_util.assert_that(rb_pcoll, _assert_fn)
    pipeline_result = p.run()
    pipeline_result.wait_until_finish()
    telemetry_test_util.ValidateMetrics(
        self, pipeline_result, _TELEMETRY_DESCRIPTORS,
        "tensor", "tfrecords_gzip")

  def test_project(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
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

  def test_tensorflow_dataset(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, decoded_tensors_dict in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      for key, tensor in decoded_tensors_dict.items():
        self._assert_sparse_tensor_equal(tensor, _RECORDS_AS_TENSORS[i][key])

  def test_projected_tensorflow_dataset(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
    feature_name = "st1"
    projected_tfxio = tfxio.Project([feature_name])
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1)
    for i, decoded_tensors_dict in enumerate(
        projected_tfxio.TensorFlowDataset(options=options)):
      self.assertIn(feature_name, decoded_tensors_dict)
      self.assertLen(decoded_tensors_dict, 1)
      tensor = decoded_tensors_dict[feature_name]
      self._assert_sparse_tensor_equal(
          tensor, _RECORDS_AS_TENSORS[i][feature_name])

  def test_tensorflow_dataset_with_label_key(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
    label_key = "st1"
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1, label_key=label_key)
    for i, (decoded_tensors_dict, label_feature) in enumerate(
        tfxio.TensorFlowDataset(options=options)):
      self._assert_sparse_tensor_equal(
          label_feature, _RECORDS_AS_TENSORS[i][label_key])
      for key, tensor in decoded_tensors_dict.items():
        self._assert_sparse_tensor_equal(tensor, _RECORDS_AS_TENSORS[i][key])

  def test_tensorflow_dataset_with_invalid_label_key(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
    label_key = "invalid"
    options = dataset_options.TensorFlowDatasetOptions(
        batch_size=1, shuffle=False, num_epochs=1, label_key=label_key)
    with self.assertRaisesRegex(ValueError, "The `label_key` provided.*"):
      tfxio.TensorFlowDataset(options=options)

  def test_get_decode_function(self):
    decoder_path = _write_decoder()
    tfxio = record_to_tensor_tfxio.TFRecordToTensorTFXIO(
        self._input_path, decoder_path, ["some", "component"])
    decode_fn = tfxio.DecodeFunction()
    decoded = decode_fn(tf.constant(_RECORDS))
    for tensor_name in ("st1", "st2"):
      self.assertIn(tensor_name, decoded)
      st = decoded[tensor_name]
      self.assertAllEqual(st.values, _RECORDS)
      self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
      self.assertAllEqual(st.dense_shape, [2, 1])


if __name__ == "__main__":
  tf.test.main()
