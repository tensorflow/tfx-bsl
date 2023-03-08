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
"""Tests for tfx_bsl.coders.tf_graph_record_decoder."""

import os
import tempfile

from absl import flags
import tensorflow as tf
from tfx_bsl.coders import tf_graph_record_decoder


FLAGS = flags.FLAGS


class _DecoderForTesting(tf_graph_record_decoder.TFGraphRecordDecoder):

  def decode_record(self, record):
    indices = tf.transpose(tf.stack([
        tf.range(tf.size(record), dtype=tf.int64),
        tf.zeros(tf.size(record), dtype=tf.int64)
    ]))
    sparse = tf.SparseTensor(
                values=record,
                indices=indices,
                dense_shape=[tf.size(record), 1])
    return {
        "sparse_tensor": sparse,
        "ragged_tensor": tf.RaggedTensor.from_sparse(sparse),
        "record_index": tf.RaggedTensor.from_row_splits(
            values=tf.range(tf.size(record), dtype=tf.int64),
            row_splits=tf.range(tf.size(record) + 1, dtype=tf.int64)),
        "dense_tensor": record,
    }


class _DecoderForTestWithRecordIndexTensorName(_DecoderForTesting):

  @property
  def record_index_tensor_name(self):
    return "record_index"


class _DecoderForTestWithInvalidRecordIndexTensorName(_DecoderForTesting):

  @property
  def record_index_tensor_name(self):
    return "does_not_exist"


class TfGraphRecordDecoderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

  def _assert_type_specs_equal(self, lhs, rhs):
    self.assertLen(lhs, len(rhs))
    for k, spec in lhs.items():
      self.assertIn(k, rhs)
      # special handling for tf.TensorSpec to ignore the difference in .name.
      if isinstance(spec, tf.TensorSpec):
        self.assertIsInstance(rhs[k], tf.TensorSpec)
        self.assertEqual(spec.shape.as_list(), rhs[k].shape.as_list())
        self.assertEqual(spec.dtype, rhs[k].dtype)
        continue
      self.assertEqual(spec, rhs[k])

  def test_save_load_decode(self):
    decoder = _DecoderForTestWithRecordIndexTensorName()
    actual_type_specs = decoder.output_type_specs()
    actual_sparse_tensor_spec = actual_type_specs.pop("sparse_tensor")
    # The expected shape is [None, 1], but due to a TensorFlow bug, it could
    # be [None, None] in older TF versions.
    self.assertTrue(actual_sparse_tensor_spec ==
                    tf.SparseTensorSpec(shape=[None, None], dtype=tf.string) or
                    actual_sparse_tensor_spec == tf.SparseTensorSpec(
                        shape=[None, 1], dtype=tf.string))
    self.assertEqual(
        actual_type_specs, {
            "ragged_tensor":
                tf.RaggedTensorSpec(
                    shape=[None, None], dtype=tf.string, ragged_rank=1),
            "record_index":
                tf.RaggedTensorSpec(
                    shape=[None, None], dtype=tf.int64, ragged_rank=1),
            "dense_tensor":
                tf.TensorSpec(shape=[None], dtype=tf.string)
        })
    self.assertEqual(decoder.record_index_tensor_name, "record_index")
    tf_graph_record_decoder.save_decoder(decoder, self._tmp_dir)
    loaded = tf_graph_record_decoder.load_decoder(self._tmp_dir)
    self.assertEqual(loaded.record_index_tensor_name, "record_index")

    self._assert_type_specs_equal(decoder.output_type_specs(),
                                  loaded.output_type_specs())

    records = [b"abc", b"def"]
    got = loaded.decode_record(records)
    self.assertLen(got, len(loaded.output_type_specs()))
    self.assertIn("sparse_tensor", got)
    st = got["sparse_tensor"]
    self.assertAllEqual(st.values, records)
    self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(st.dense_shape, [2, 1])

    rt = got["ragged_tensor"]
    self.assertAllEqual(rt, tf.ragged.constant([[b"abc"], [b"def"]]))

    rt = got["record_index"]
    self.assertAllEqual(rt, tf.ragged.constant([[0], [1]]))

    dt = got["dense_tensor"]
    self.assertAllEqual(dt, records)

    # Also test that .record_index_tensor_name can be accessed in graph
    # mode.
    with tf.compat.v1.Graph().as_default():
      self.assertFalse(tf.executing_eagerly())
      loaded = tf_graph_record_decoder.load_decoder(self._tmp_dir)
      self.assertEqual(loaded.record_index_tensor_name, "record_index")

    # Also test that the decoder's class method `save_decoder` works.
    new_decoder_path = (os.path.join(self._tmp_dir, "decoder_2"))
    decoder.save(new_decoder_path)
    loaded = tf_graph_record_decoder.load_decoder(new_decoder_path)
    self.assertEqual(loaded.record_index_tensor_name, "record_index")

  def test_no_record_index_tensor_name(self):
    decoder = _DecoderForTesting()
    self.assertIsNone(decoder.record_index_tensor_name)

    tf_graph_record_decoder.save_decoder(decoder, self._tmp_dir)
    loaded = tf_graph_record_decoder.load_decoder(self._tmp_dir)
    self._assert_type_specs_equal(decoder.output_type_specs(),
                                  loaded.output_type_specs())
    self.assertIsNone(loaded.record_index_tensor_name)

    with tf.compat.v1.Graph().as_default():
      self.assertFalse(tf.executing_eagerly())
      loaded = tf_graph_record_decoder.load_decoder(self._tmp_dir)
      self.assertIsNone(loaded.record_index_tensor_name)

  def test_do_not_save_if_record_index_tensor_name_invalid(self):
    decoder = _DecoderForTestWithInvalidRecordIndexTensorName()
    with self.assertRaisesRegex(AssertionError, "record_index_tensor_name"):
      tf_graph_record_decoder.save_decoder(decoder, self._tmp_dir)


if __name__ == "__main__":
  tf.test.main()
