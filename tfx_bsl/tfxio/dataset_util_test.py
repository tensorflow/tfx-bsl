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
"""Tests for tfx_bsl.tfxio.dataset_util."""

import os
import tempfile

from absl import flags
from absl.testing import parameterized
import tensorflow as tf
from tfx_bsl.tfxio import dataset_util
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import

FLAGS = flags.FLAGS

_RECORDS = [b'aaa', b'bbb']


def _write_inputs(filename):
  with tf.io.TFRecordWriter(filename) as w:
    for s in _RECORDS:
      w.write(s)


class DatasetUtilTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(DatasetUtilTest, self).setUp()
    self._example_file = os.path.join(FLAGS.test_tmpdir, 'datasetutiltest',
                                      'input')
    tf.io.gfile.makedirs(os.path.dirname(self._example_file))
    _write_inputs(self._example_file)

  @parameterized.named_parameters(*[
      dict(
          testcase_name='default_options',
          batch_size=1,
          drop_final_batch=False,
          num_epochs=1,
          expected_data=[[b'aaa'], [b'bbb']]),
      dict(
          testcase_name='batch',
          batch_size=2,
          drop_final_batch=False,
          num_epochs=1,
          expected_data=[[b'aaa', b'bbb']]),
      dict(
          testcase_name='batch_2_epochs',
          batch_size=2,
          drop_final_batch=False,
          num_epochs=2,
          expected_data=[[b'aaa', b'bbb'], [b'aaa', b'bbb']]),
      dict(
          testcase_name='drop_final_batch',
          batch_size=3,
          drop_final_batch=True,
          num_epochs=4,
          expected_data=[[b'aaa', b'bbb', b'aaa'], [b'bbb', b'aaa', b'bbb']])
  ])
  @test_util.run_in_graph_and_eager_modes
  def test_make_tf_record_dataset(self, batch_size, drop_final_batch,
                                  num_epochs, expected_data):
    dataset = dataset_util.make_tf_record_dataset(
        self._example_file,
        batch_size,
        drop_final_batch,
        num_epochs,
        shuffle=False,
        shuffle_buffer_size=10000,
        shuffle_seed=None)
    data = _dataset_elements(dataset)
    self.assertAllEqual(data, expected_data)

  @test_util.run_in_graph_and_eager_modes
  def test_detect_compression_type(self):
    tmp_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

    def _touch_file(path):
      with open(path, 'w+') as _:
        pass

    _touch_file(os.path.join(tmp_dir, 'dataset-a-0.gz'))
    _touch_file(os.path.join(tmp_dir, 'dataset-a-1.gz'))
    _touch_file(os.path.join(tmp_dir, 'dataset-b-0'))
    _touch_file(os.path.join(tmp_dir, 'dataset-b-1'))

    self.assertAllEqual(
        dataset_util.detect_compression_type(
            [os.path.join(tmp_dir, 'dataset-a*')]), b'GZIP')

    self.assertAllEqual(
        dataset_util.detect_compression_type(
            [os.path.join(tmp_dir, 'dataset-b*')]), b'')

    self.assertAllEqual(
        dataset_util.detect_compression_type([
            os.path.join(tmp_dir, 'dataset-b*'),
            os.path.join(tmp_dir, 'dataset-a*')
        ]), b'INVALID_MIXED_COMPRESSION_TYPES')

    self.assertAllEqual(
        dataset_util.detect_compression_type(
            [os.path.join(tmp_dir, 'invalid*')]), b'')


def _dataset_elements(dataset):
  """Returns elements from the `tf.data.Dataset` object as a list."""
  results = []
  if tf.executing_eagerly():
    for elem in dataset:
      results.append(elem.numpy())
  else:
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    next_elem = iterator.get_next()
    with tf.compat.v1.Session() as sess:
      while True:
        try:
          results.append(sess.run(next_elem))
        except tf.errors.OutOfRangeError:
          break
  return results

if __name__ == '__main__':
  tf.test.main()
