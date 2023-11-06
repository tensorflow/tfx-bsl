# Copyright 2023 Google LLC
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
"""Tests for tfx_bsl.tfxio.dataset_tfxio."""


import collections
from absl.testing import parameterized
import tensorflow as tf
from tfx_bsl.tfxio import dataset_tfxio


class DatasetTfxioTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      *[
          dict(
              testcase_name='simple_element_spec',
              element_spec=tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              feature_names=None,
              expected_dict=collections.OrderedDict([(
                  'feature0',
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              )]),
          ),
          dict(
              testcase_name='simple_element_spec_feature_name',
              element_spec=tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              feature_names=['custom_feature'],
              expected_dict=collections.OrderedDict([(
                  'custom_feature',
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              )]),
          ),
          dict(
              testcase_name='element_spec_with_shape',
              element_spec=tf.TensorSpec(shape=(5,), dtype=tf.int32, name=None),
              feature_names=None,
              expected_dict=collections.OrderedDict([(
                  'feature0',
                  tf.TensorSpec(shape=(5,), dtype=tf.int32, name=None),
              )]),
          ),
          dict(
              testcase_name='tuple_element_spec',
              element_spec=(
                  tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              ),
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  (
                      'feature0',
                      tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                  ),
                  (
                      'feature1',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='tuple_element_spec_feature_names',
              element_spec=(
                  tf.TensorSpec(shape=(None,), dtype=tf.string, name=None),
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              ),
              feature_names=['custom_feature_1', 'custom_feature_2'],
              expected_dict=collections.OrderedDict([
                  (
                      'custom_feature_1',
                      tf.TensorSpec(shape=(None,), dtype=tf.string, name=None),
                  ),
                  (
                      'custom_feature_2',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='namedtuple',
              element_spec=collections.namedtuple('Dummy', ['x', 'y'])(
                  tf.TensorSpec(shape=(2, 5), dtype=tf.string, name=None),
                  tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
              ),
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  (
                      'x',
                      tf.TensorSpec(shape=(2, 5), dtype=tf.string, name=None),
                  ),
                  (
                      'y',
                      tf.TensorSpec(shape=(None,), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='namedtuple_feature_names',
              element_spec=collections.namedtuple('Dummy', ['x', 'y'])(
                  tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              ),
              feature_names=['custom_feature_1', 'custom_feature_2'],
              expected_dict=collections.OrderedDict([
                  (
                      'custom_feature_1',
                      tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                  ),
                  (
                      'custom_feature_2',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='dict',
              element_spec={
                  'x': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  'y': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              },
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  ('x', tf.TensorSpec(shape=(), dtype=tf.float32, name=None)),
                  ('y', tf.TensorSpec(shape=(), dtype=tf.int32, name=None)),
              ]),
          ),
          dict(
              testcase_name='dict_feature_names',
              element_spec={
                  'x': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  'y': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              },
              feature_names=['custom_feature_1', 'custom_feature_2'],
              expected_dict=collections.OrderedDict([
                  (
                      'custom_feature_1',
                      tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  ),
                  (
                      'custom_feature_2',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='simple_nested_dict',
              element_spec={
                  'x': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  'y': {
                      'a': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                      'b': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  },
              },
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  (
                      'x',
                      tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  ),
                  (
                      'y_a',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
                  (
                      'y_b',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='tuple_with_32nested_structure',
              element_spec=(
                  {
                      'x': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                      'y': {
                          'a': tf.TensorSpec(
                              shape=(), dtype=tf.int32, name=None
                          ),
                          'b': tf.TensorSpec(
                              shape=(), dtype=tf.int32, name=None
                          ),
                      },
                  },
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              ),
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  (
                      'x',
                      tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  ),
                  (
                      'y_a',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
                  (
                      'y_b',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
                  (
                      'feature0',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
          dict(
              testcase_name='tuple_with_3_nested_structure',
              element_spec=(
                  {
                      'x': tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                      'y': {
                          'a': collections.namedtuple('Dummy', ['f1', 'f2'])(
                              tf.TensorSpec(
                                  shape=(), dtype=tf.string, name=None
                              ),
                              tf.TensorSpec(
                                  shape=(), dtype=tf.int32, name=None
                              ),
                          ),
                          'b': tf.TensorSpec(
                              shape=(), dtype=tf.int32, name=None
                          ),
                      },
                  },
                  tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
              ),
              feature_names=None,
              expected_dict=collections.OrderedDict([
                  (
                      'x',
                      tf.TensorSpec(shape=(), dtype=tf.float32, name=None),
                  ),
                  (
                      'y_a_f1',
                      tf.TensorSpec(shape=(), dtype=tf.string, name=None),
                  ),
                  (
                      'y_a_f2',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
                  (
                      'y_b',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
                  (
                      'feature0',
                      tf.TensorSpec(shape=(), dtype=tf.int32, name=None),
                  ),
              ]),
          ),
      ]
  )
  def test_dict_structure_from_element_specs(
      self, element_spec, feature_names, expected_dict
  ):
    new_structure = dataset_tfxio._GetDictStructureForElementSpec(
        element_spec, feature_names=feature_names
    )
    self.assertEqual(new_structure, expected_dict)


if __name__ == '__main__':
  tf.test.main()
