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
"""Tests for tfx_bsl.sketches.QuantilesSketch."""

import pickle

import numpy as np
import pyarrow as pa

from tfx_bsl import sketches
from absl.testing import absltest
from absl.testing import parameterized


_QUANTILES_TEST_CASES = [
    dict(
        testcase_name="unweighted",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
        ],
        expected=[1, 61, 121, 181, 241, 300]),
    dict(
        testcase_name="weighted",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
        ],
        weights=[
            pa.array([1] * 100, type=pa.float64()),
            pa.array([2] * 100, type=pa.float64()),
            pa.array([3] * 100, type=pa.float64()),
        ],
        expected=[1, 111, 171, 221, 261, 300]),
    dict(
        testcase_name="infinity",
        values=[
            pa.array(
                [1.0, 2.0, np.inf, np.inf, -np.inf, 3.0, 4.0, 5.0, -np.inf]),
            pa.array([1.0, np.inf, -np.inf]),
        ],
        expected=[-np.inf, -np.inf, 1, 4, np.inf, np.inf],
    ),
    dict(
        testcase_name="null",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
            pa.array([None, None]),
        ],
        expected=[1, 61, 121, 181, 241, 300],
    ),
    dict(
        testcase_name="int",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.int32)),
            pa.array(np.linspace(101, 200, 100, dtype=np.int32)),
            pa.array(np.linspace(201, 300, 100, dtype=np.int32)),
        ],
        expected=[1, 61, 121, 181, 241, 300],
    ),
    dict(
        testcase_name="negative_weights",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
            pa.array([100000000, 200000000]),
        ],
        weights=[
            pa.array([1] * 100, type=pa.float64()),
            pa.array([2] * 100, type=pa.float64()),
            pa.array([3] * 100, type=pa.float64()),
            pa.array([0, -1]),
        ],
        expected=[1, 111, 171, 221, 261, 300],
    ),
]


def _add_values(sketch, value, weight):
  if weight is None:
    sketch.AddValues(value)
  else:
    sketch.AddValues(value, weight)


class QuantilesSketchTest(parameterized.TestCase):

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_quantiles(self, values, expected, weights=None):
    s = sketches.QuantilesSketch(0.00001, 1<<32)
    if weights is None:
      weights = [None] * len(values)
    for value, weight in zip(values, weights):
      _add_values(s, value, weight)

    result = s.GetQuantiles(len(expected) - 1)
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_pickle(self, values, expected, weights=None):
    s = sketches.QuantilesSketch(0.00001, 1<<32)
    if weights is None:
      weights = [None] * len(values)
    for value, weight in zip(values, weights):
      _add_values(s, value, weight)
    pickled = pickle.dumps(s)
    self.assertIsInstance(pickled, bytes)
    unpickled = pickle.loads(pickled)
    self.assertIsInstance(unpickled, sketches.QuantilesSketch)
    result = unpickled.GetQuantiles(len(expected) - 1)
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_merge(self, values, expected, weights=None):
    if weights is None:
      weights = [None] * len(values)
    s1 = sketches.QuantilesSketch(0.00001, 1 << 32)
    for value, weight in zip(values[:len(values) // 2],
                             weights[:len(weights) // 2]):
      _add_values(s1, value, weight)
    s2 = sketches.QuantilesSketch(0.00001, 1 << 32)
    for value, weight in zip(values[len(values) // 2:],
                             weights[len(weights) // 2:]):
      _add_values(s2, value, weight)

    def pickle_roundtrip(s):
      return pickle.loads(pickle.dumps(s))

    s1 = pickle_roundtrip(s1)
    s2 = pickle_roundtrip(s2)
    s1.Merge(s2)

    result = s1.GetQuantiles(len(expected) - 1)
    np.testing.assert_almost_equal(expected, result)


if __name__ == "__main__":
  absltest.main()
