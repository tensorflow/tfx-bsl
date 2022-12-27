# Copyright 2022 Google LLC
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
"""Utilities for testing tfx_bsl.tfxio functionality."""
from typing import Optional

from absl.testing import parameterized
import tensorflow as tf
from tfx_bsl.types import common_types

from absl.testing import absltest

named_parameters = parameterized.named_parameters
SkipTest = absltest.SkipTest

main = tf.test.main


class TfxBslTestCase(parameterized.TestCase, tf.test.TestCase):
  """Base test class for testing tfxio code."""

  def assertTensorsEqual(self,
                         left: common_types.TensorAlike,
                         right: common_types.TensorAlike,
                         msg: Optional[str] = None):
    """Checks whether two tensors or tensor values are equal."""
    self.assertIsInstance(left, type(right), msg)
    if isinstance(left, (tf.SparseTensor, tf.compat.v1.SparseTensorValue)):
      self.assertAllEqual(left.values, right.values, msg)
      self.assertAllEqual(left.indices, right.indices, msg)
      self.assertAllEqual(left.dense_shape, right.dense_shape, msg)
    elif isinstance(left,
                    (tf.RaggedTensor, tf.compat.v1.ragged.RaggedTensorValue)):
      self.assertTensorsEqual(left.values, right.values, msg)
      self.assertAllEqual(left.row_splits, right.row_splits, msg)
    else:
      self.assertAllEqual(left, right, msg)
