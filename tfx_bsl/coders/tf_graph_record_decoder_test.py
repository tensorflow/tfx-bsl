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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import uuid

from absl import flags
import tensorflow as tf
from tfx_bsl.coders import tf_graph_record_decoder


FLAGS = flags.FLAGS


class _DecoderForTesting(tf_graph_record_decoder.TFGraphRecordDecoder):

  def __init__(self):
    super(_DecoderForTesting, self).__init__("DecoderForTesting")

  def _decode_record_internal(self, record):
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
    }


class TfGraphRecordDecoderTest(tf.test.TestCase):

  def setUp(self):
    super(TfGraphRecordDecoderTest, self).setUp()
    self._tmp_dir = os.path.join(
        FLAGS.test_tmpdir, "tfgraphrecorddecodertest", uuid.uuid4().hex)

  def test_save_load_decode(self):
    decoder = _DecoderForTesting()
    self.assertEqual(decoder.output_type_specs(), {
        "sparse_tensor":
            tf.SparseTensorSpec(shape=[None, None], dtype=tf.string),
        "ragged_tensor":
            tf.RaggedTensorSpec(
                shape=[None, None], dtype=tf.string, ragged_rank=1)
    })
    tf_graph_record_decoder.save_decoder(decoder, self._tmp_dir)
    loaded = tf_graph_record_decoder.load_decoder(self._tmp_dir)

    self.assertEqual(decoder.output_type_specs(), loaded.output_type_specs())
    got = loaded.decode_record([b"abc", b"def"])
    self.assertLen(got, len(loaded.output_type_specs()))
    self.assertIn("sparse_tensor", got)
    st = got["sparse_tensor"]
    self.assertAllEqual(st.values, [b"abc", b"def"])
    self.assertAllEqual(st.indices, [[0, 0], [1, 0]])
    self.assertAllEqual(st.dense_shape, [2, 1])

    rt = got["ragged_tensor"]
    self.assertAllEqual(rt, tf.ragged.constant([[b"abc"], [b"def"]]))


if __name__ == "__main__":
  # Do not run these tests under TF1.x -- not supported.
  if tf.__version__ >= "2":
    tf.test.main()
