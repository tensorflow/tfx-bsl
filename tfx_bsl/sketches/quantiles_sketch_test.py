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

import copy
import itertools
import pickle

import numpy as np
import pyarrow as pa
from tfx_bsl import sketches

from absl.testing import absltest
from absl.testing import parameterized

_QUANTILES_CUMULATIVE_WEIGHTS_TEST_CASES = [
    dict(
        testcase_name="unweighted",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
        ],
        expected=[[1, 61, 121, 181, 241, 300]],
        expected_weights=[[1, 61, 121, 181, 241, 300]],
        num_streams=1),
    dict(
        testcase_name="unweighted_lumpy",
        values=[
            pa.array([1, 2, 3, 4]),
            pa.array([2, 2, 2, 2]),
            pa.array([1, 2, 3, 4]),
        ],
        expected=[[1, 2, 3, 4]],
        expected_weights=[[2, 8, 10, 12]],
        num_streams=1),
    dict(
        testcase_name="more_quantiles_than_values",
        values=[
            pa.array(100 * [0]),
        ],
        expected=[[0, 0, 0, 0]],
        expected_weights=[[100, 100, 100, 100]],
        num_streams=1),
    dict(
        testcase_name="unweighted_elementwise",
        values=[
            pa.array(np.linspace(1, 500, 500, dtype=np.float64)),
            pa.array(np.linspace(101, 600, 500, dtype=np.float64)),
            pa.array(np.linspace(201, 700, 500, dtype=np.float64)),
        ],
        expected=[[1, 201, 301, 401, 501, 696], [2, 202, 302, 402, 502, 697],
                  [3, 203, 303, 403, 503, 698], [4, 204, 304, 404, 504, 699],
                  [5, 205, 305, 405, 505, 700]],
        expected_weights=[[1, 63, 123, 183, 242, 300],
                          [1, 63, 123, 183, 242, 300],
                          [1, 63, 123, 183, 242, 300],
                          [1, 63, 123, 183, 242, 300],
                          [1, 63, 123, 183, 242, 300]],
        num_streams=5),
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
        expected=[[1, 111, 171, 221, 261, 300]],
        num_streams=1),
    dict(
        testcase_name="weighted_elementwise",
        values=[
            pa.array(np.linspace(1, 500, 500, dtype=np.float64)),
            pa.array(np.linspace(101, 600, 500, dtype=np.float64)),
            pa.array(np.linspace(201, 700, 500, dtype=np.float64)),
        ],
        weights=[
            pa.array([1] * 100, type=pa.float64()),
            pa.array([2] * 100, type=pa.float64()),
            pa.array([3] * 100, type=pa.float64()),
        ],
        expected=[[1, 231, 331, 431, 541, 696], [2, 232, 332, 432, 542, 697],
                  [3, 233, 333, 433, 543, 698], [4, 234, 334, 434, 544, 699],
                  [5, 235, 335, 435, 545, 700]],
        expected_weights=[[1, 122, 242, 362, 485, 600],
                          [1, 122, 242, 362, 485, 600],
                          [1, 122, 242, 362, 485, 600],
                          [1, 122, 242, 362, 485, 600],
                          [1, 122, 242, 362, 485, 600]],
        num_streams=5),
    dict(
        testcase_name="infinity",
        values=[
            pa.array(
                [1.0, 2.0, np.inf, np.inf, -np.inf, 3.0, 4.0, 5.0, -np.inf]),
            pa.array([1.0, np.inf, -np.inf]),
        ],
        expected=[[-np.inf, -np.inf, 1, 4, np.inf, np.inf]],
        num_streams=1),
    dict(
        testcase_name="null",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
            pa.array([None, None]),
        ],
        expected=[[1, 61, 121, 181, 241, 300]],
        expected_weights=[[1, 61, 121, 181, 241, 300]],
        num_streams=1),
    dict(
        testcase_name="nan",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
            pa.array(np.array([np.nan, np.nan])),
        ],
        expected=[[1, 61, 121, 181, 241, 300]],
        expected_weights=[[1, 61, 121, 181, 241, 300]],
        num_streams=1),
    dict(
        testcase_name="nan_weighted",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.float64)),
            pa.array(np.linspace(101, 200, 100, dtype=np.float64)),
            pa.array(np.linspace(201, 300, 100, dtype=np.float64)),
            pa.array(np.array([-100, np.nan])),
        ],
        weights=[
            pa.array([1] * 100, type=pa.float64()),
            pa.array([2] * 100, type=pa.float64()),
            pa.array([3] * 100, type=pa.float64()),
            pa.array(np.array([np.nan, 10000])),
        ],
        expected=[[1, 111, 171, 221, 261, 300]],
        expected_weights=[[1, 122, 242, 363, 483, 600]],
        num_streams=1),
    dict(
        testcase_name="int",
        values=[
            pa.array(np.linspace(1, 100, 100, dtype=np.int32)),
            pa.array(np.linspace(101, 200, 100, dtype=np.int32)),
            pa.array(np.linspace(201, 300, 100, dtype=np.int32)),
        ],
        expected=[[1, 61, 121, 181, 241, 300]],
        num_streams=1),
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
        expected=[[1, 111, 171, 221, 261, 300]],
        num_streams=1),
]
_QUANTILES_TEST_CASES = copy.deepcopy(_QUANTILES_CUMULATIVE_WEIGHTS_TEST_CASES)
for tc in _QUANTILES_TEST_CASES:
  tc.pop("expected_weights", None)

_MAX_NUM_ELEMENTS = [2**10, 2**14, 2**18, 2**19]
_EPS = [0.5, 0.01, 0.001, 0.0001, 0.000001]
_NUM_QUANTILES = [5**4, 3**6, 1000, 2**10]

_ACCURACY_TEST_CASES = list(
    itertools.product(_MAX_NUM_ELEMENTS, _EPS, _NUM_QUANTILES))


def _add_values(sketch, value, weight):
  if weight is None:
    sketch.AddValues(value)
  else:
    sketch.AddValues(value, weight)


def _pickle_roundtrip(s):
  return pickle.loads(pickle.dumps(s))


def _get_cumulative_weight(values, quantiles, weights):
  if weights is None:
    weights = np.ones(values.shape[0])
  cum_weight = []
  for q in quantiles:
    cum_weight.append(weights[values <= q].sum())
  return np.array(cum_weight)


def _partition_streams(values, num_streams):
  result = []
  for v in values:
    result.append(v.to_numpy(zero_copy_only=True))
  result = np.concatenate(result)
  stream_results = []
  for stream in range(num_streams):
    stream_results.append(result[stream::num_streams])
  return stream_results


class QuantilesSketchTest(parameterized.TestCase):

  def assert_quantiles_accuracy(self, quantiles, cdf, eps, cumulative_weights):
    # Helper function to validate quantiles accuracy given a cdf function.
    # This function assumes that quantiles input values are of the form
    # range(N). Note that this function also validates order of quantiles since
    # their cdf values are compared to ordered expected levels.
    num_quantiles = len(quantiles)
    expected_cumulative_weights = [
        i / (num_quantiles - 1) for i in range(num_quantiles)
    ]
    for expected_level, quantile, level in zip(expected_cumulative_weights,
                                               quantiles, cumulative_weights):
      level /= sum(cumulative_weights)
      quantile_cdf = cdf(quantile)
      left_cdf = cdf(quantile - 1)
      right_cdf = cdf(quantile + 1)
      error_msg = (
          "Accuracy of the given quantile is not sufficient, "
          "quantile={} of expected level {}, its cdf is {}; cdf of a value to "
          "the left is {}, to the right is {}. Error bound = {}.").format(
              quantile, level, quantile_cdf, left_cdf, right_cdf, eps)
      self.assertTrue(
          abs(expected_level - cdf(quantile)) < eps or
          left_cdf < expected_level < right_cdf or
          (expected_level == 0 and left_cdf == 0), error_msg)
    # Check that the true CDF and the reported CDF from cumulative weights are
    # close up to eps.
    full_true_cdf = np.array([cdf(q) for q in quantiles])
    full_reported_cdf = np.array(cumulative_weights)
    full_reported_cdf /= full_reported_cdf.max()
    np.testing.assert_allclose(full_reported_cdf, full_true_cdf, atol=eps)

  def test_quantiles_sketch_init(self):
    with self.assertRaisesRegex(RuntimeError, "eps must be positive"):
      _ = sketches.QuantilesSketch(0, 1 << 32, 1)

    with self.assertRaisesRegex(RuntimeError, "max_num_elements must be >= 1."):
      _ = sketches.QuantilesSketch(0.0001, 0, 1)

    with self.assertRaisesRegex(RuntimeError, "num_streams must be >= 1."):
      _ = sketches.QuantilesSketch(0.0001, 1 << 32, 0)

    _ = sketches.QuantilesSketch(0.0001, 1 << 32, 1)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_quantiles(
      self,
      values,
      expected,
      num_streams,
      weights=None,
  ):
    s = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    if weights is None:
      weights = [None] * len(values)
    for value, weight in zip(values, weights):
      _add_values(s, value, weight)

    result = s.GetQuantiles(len(expected[0]) - 1).to_pylist()
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(*_QUANTILES_CUMULATIVE_WEIGHTS_TEST_CASES)
  def test_cumulative_weights(self,
                              values,
                              expected,
                              num_streams,
                              expected_weights=None,
                              weights=None):
    s = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    if weights is None:
      weights = [None] * len(values)
    for value, weight in zip(values, weights):
      _add_values(s, value, weight)
    result_quantiles, result_weights = s.GetQuantilesAndCumulativeWeights(
        len(expected[0]) - 1)
    result_quantiles = result_quantiles.to_pylist()
    result_weights = result_weights.to_pylist()

    if expected_weights is None:
      # Expected cumulative weights not provided, so compute them.
      assert num_streams == 1
      values_streams = _partition_streams(values, num_streams)
      if weights[0] is None:
        weights_streams = num_streams * [None]
      else:
        weights_streams = _partition_streams(weights, num_streams)
      for stream_idx in range(num_streams):
        expected_quantiles = expected[stream_idx]
        cumul_weights = _get_cumulative_weight(values_streams[stream_idx],
                                               expected_quantiles,
                                               weights_streams[stream_idx])
        np.testing.assert_almost_equal(result_weights[stream_idx],
                                       cumul_weights)
    else:
      np.testing.assert_almost_equal(expected_weights, result_weights)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_pickle(self, values, expected, num_streams, weights=None):
    s = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    if weights is None:
      weights = [None] * len(values)
    for value, weight in zip(values, weights):
      _add_values(s, value, weight)
    pickled = pickle.dumps(s)
    self.assertIsInstance(pickled, bytes)
    unpickled = pickle.loads(pickled)
    self.assertIsInstance(unpickled, sketches.QuantilesSketch)
    result = unpickled.GetQuantiles(len(expected[0]) - 1).to_pylist()
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_merge(self, values, expected, num_streams, weights=None):
    if weights is None:
      weights = [None] * len(values)
    s1 = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    for value, weight in zip(values[:len(values) // 2],
                             weights[:len(weights) // 2]):
      _add_values(s1, value, weight)
    s2 = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    for value, weight in zip(values[len(values) // 2:],
                             weights[len(weights) // 2:]):
      _add_values(s2, value, weight)

    s1 = _pickle_roundtrip(s1)
    s2 = _pickle_roundtrip(s2)
    s1.Merge(s2)

    result = s1.GetQuantiles(len(expected[0]) - 1).to_pylist()
    np.testing.assert_almost_equal(expected, result)

  @parameterized.named_parameters(*_QUANTILES_TEST_CASES)
  def test_compact(self, values, expected, num_streams, weights=None):
    s = sketches.QuantilesSketch(0.00001, 1 << 32, num_streams)
    num_values = len(values)
    if weights is None:
      weights = [None] * num_values
    for value, weight in zip(values[:num_values // 2],
                             weights[:num_values // 2]):
      _add_values(s, value, weight)
    s.Compact()
    for value, weight in zip(values[num_values // 2:],
                             weights[num_values // 2:]):
      _add_values(s, value, weight)
    s.Compact()

    result = s.GetQuantiles(len(expected[0]) - 1).to_pylist()
    np.testing.assert_almost_equal(expected, result)

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy(self, max_num_elements, eps, num_quantiles):
    s = sketches.QuantilesSketch(eps, max_num_elements, 1)
    values = pa.array(reversed(range(max_num_elements)))
    weights = pa.array(range(max_num_elements))
    total_weight = (max_num_elements - 1) * max_num_elements / 2

    def cdf(x):
      left_weight = (2 * (max_num_elements - 1) - x) * (x + 1) / 2
      return left_weight / total_weight

    _add_values(s, values, weights)
    quantiles, cumulative_weights = s.GetQuantilesAndCumulativeWeights(
        num_quantiles - 1)
    quantiles = quantiles.to_pylist()[0]
    cumulative_weights = cumulative_weights.to_pylist()[0]
    self.assert_quantiles_accuracy(quantiles, cdf, eps, cumulative_weights)
    self.assertEqual(total_weight, cumulative_weights[-1])

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy_after_pickle(self, max_num_elements, eps, num_quantiles):
    s = sketches.QuantilesSketch(eps, max_num_elements, 1)
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
    quantiles, cumulative_weights = s.GetQuantilesAndCumulativeWeights(
        num_quantiles - 1)
    quantiles = quantiles.to_pylist()[0]
    cumulative_weights = cumulative_weights.to_pylist()[0]
    self.assert_quantiles_accuracy(quantiles, cdf, eps, cumulative_weights)
    self.assertEqual(total_weight, cumulative_weights[-1])

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy_after_merge(self, max_num_elements, eps, num_quantiles):
    s1 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    s2 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    s3 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    values = pa.array(reversed(range(max_num_elements)))
    weights = pa.array(range(max_num_elements))
    total_weight = (max_num_elements - 1) * max_num_elements / 2

    def cdf(x):
      left_weight = (2 * (max_num_elements - 1) - x) * (x + 1) / 2
      return left_weight / total_weight

    _add_values(s1, values[:max_num_elements // 10],
                weights[:max_num_elements // 10])
    _add_values(s2, values[max_num_elements // 10:max_num_elements // 3],
                weights[max_num_elements // 10:max_num_elements // 3])
    _add_values(s3, values[max_num_elements // 3:],
                weights[max_num_elements // 3:])
    s2.Merge(s3)
    s1.Merge(s2)
    quantiles, cumulative_weights = s1.GetQuantilesAndCumulativeWeights(
        num_quantiles - 1)
    quantiles = quantiles.to_pylist()[0]
    cumulative_weights = cumulative_weights.to_pylist()[0]
    self.assert_quantiles_accuracy(quantiles, cdf, eps, cumulative_weights)
    self.assertEqual(total_weight, cumulative_weights[-1])

  @parameterized.parameters(*_ACCURACY_TEST_CASES)
  def test_accuracy_after_compact(self, max_num_elements, eps, num_quantiles):
    s1 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    s2 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    s3 = sketches.QuantilesSketch(eps, max_num_elements, 1)
    values = pa.array(reversed(range(max_num_elements)))
    weights = pa.array(range(max_num_elements))
    total_weight = (max_num_elements - 1) * max_num_elements / 2

    def cdf(x):
      left_weight = (2 * (max_num_elements - 1) - x) * (x + 1) / 2
      return left_weight / total_weight

    _add_values(s1, values[:max_num_elements // 10],
                weights[:max_num_elements // 10])
    _add_values(s2, values[max_num_elements // 10:max_num_elements // 3],
                weights[max_num_elements // 10:max_num_elements // 3])
    _add_values(s3, values[max_num_elements // 3:],
                weights[max_num_elements // 3:])
    s2.Compact()
    s3.Compact()
    s2.Merge(s3)
    s2.Compact()
    s1.Compact()
    s1.Merge(s2)
    s1.Compact()
    quantiles, cumulative_weights = s1.GetQuantilesAndCumulativeWeights(
        num_quantiles - 1)
    quantiles = quantiles.to_pylist()[0]
    cumulative_weights = cumulative_weights.to_pylist()[0]
    self.assert_quantiles_accuracy(quantiles, cdf, eps, cumulative_weights)
    self.assertEqual(total_weight, cumulative_weights[-1])


if __name__ == "__main__":
  absltest.main()
