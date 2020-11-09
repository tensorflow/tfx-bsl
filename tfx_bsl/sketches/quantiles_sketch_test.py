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

import itertools
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

_MAX_NUM_ELEMENTS = [2**10, 2**14, 2**18]
_EPS = [0.01, 0.001, 0.0001, 0.000001]
_NUM_QUANTILES = [2**2, 2**3, 2**4, 2**5, 2**6]

_ACCURACY_TEST_CASES = list(
    itertools.product(_MAX_NUM_ELEMENTS, _EPS, _NUM_QUANTILES))


def _add_values(sketch, value, weight):
  if weight is None:
    sketch.AddValues(value)
  else:
    sketch.AddValues(value, weight)


def _pickle_roundtrip(s):
  return pickle.loads(pickle.dumps(s))


class QuantilesSketchTest(parameterized.TestCase):

  def assertQuantilesAccuracy(self, quantiles, cdf, eps):
    # Helper function to validate quantiles accuracy given a cdf function.
    # This function assumes that quantiles input values are of the form
    # range(N). Note that this function also validates order of quantiles since
    # their cdf values are compared to ordered expected levels.
    num_quantiles = len(quantiles)
    expected_levels = [i / (num_quantiles - 1) for i in range(num_quantiles)]
    for level, quantile in zip(expected_levels, quantiles):
      quantile_cdf = cdf(quantile)
      left_cdf = cdf(quantile - 1)
      right_cdf = cdf(quantile + 1)
      error_msg = (
          "Accuracy of the given quantile is not sufficient, "
          "quantile={} of expected level {}, its cdf is {}; cdf of a value to "
          "the left is {}, to the right is {}. Error bound = {}.").format(
              quantile, level, quantile_cdf, left_cdf, right_cdf, eps)
      self.assertTrue(
          abs(level - cdf(quantile)) < eps or left_cdf < level < right_cdf or
          (level == 0 and left_cdf == 0), error_msg)

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

    s1 = _pickle_roundtrip(s1)
    s2 = _pickle_roundtrip(s2)
    s1.Merge(s2)

    result = s1.GetQuantiles(len(expected) - 1)
    np.testing.assert_almost_equal(expected, result)

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy(self, max_num_elements, eps, num_quantiles):
    s = sketches.QuantilesSketch(eps / 2, max_num_elements)
    values = pa.array(reversed(range(max_num_elements)))
    weights = pa.array(range(max_num_elements))
    total_weight = (max_num_elements - 1) * max_num_elements / 2

    def cdf(x):
      left_weight = (2 * (max_num_elements - 1) - x) * (x + 1) / 2
      return left_weight / total_weight

    _add_values(s, values, weights)
    quantiles = s.GetQuantiles(num_quantiles - 1).to_pylist()
    self.assertQuantilesAccuracy(quantiles, cdf, eps)

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy_after_pickle(self, max_num_elements, eps, num_quantiles):
    s = sketches.QuantilesSketch(eps / 2, max_num_elements)
    values = pa.array(reversed(range(max_num_elements)))
    weights = pa.array(range(max_num_elements))
    total_weight = (max_num_elements - 1) * max_num_elements / 2

    def cdf(x):
      left_weight = (2 * (max_num_elements - 1) - x) * (x + 1) / 2
      return left_weight / total_weight

    _add_values(s, values[:max_num_elements // 2],
                weights[:max_num_elements // 2])
    s = _pickle_roundtrip(s)
    _add_values(s, values[max_num_elements // 2:],
                weights[max_num_elements // 2:])
    s = _pickle_roundtrip(s)
    quantiles = s.GetQuantiles(num_quantiles - 1).to_pylist()
    self.assertQuantilesAccuracy(quantiles, cdf, eps)


if __name__ == "__main__":
  absltest.main()
