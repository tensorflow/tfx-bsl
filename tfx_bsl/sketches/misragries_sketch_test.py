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
"""Tests for tfx_bsl.sketches.MisraGriesSketch."""

import itertools
import pickle

import pyarrow as pa
from tfx_bsl import sketches

from absl.testing import absltest
from absl.testing import parameterized

_NUM_BUCKETS = 128


def _create_basic_sketch(
    items, weights=None, num_buckets=_NUM_BUCKETS, reverse=False
):
  order = (
      sketches.MisraGriesSketch.OrderOnTie.ReverseLexicographical
      if reverse
      else sketches.MisraGriesSketch.OrderOnTie.Lexicographical
  )
  sketch = sketches.MisraGriesSketch(num_buckets, order_on_tie=order)
  if weights:
    sketch.AddValues(items, weights)
  else:
    sketch.AddValues(items)
  return sketch


class MisraGriesSketchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("binary", [b"a", b"a", b"b", b"c", None], pa.binary()),
      ("large_binary", [b"a", b"a", b"b", b"c"], pa.large_binary()),
      ("string", ["a", "a", "b", "c", None], pa.string()),
      ("large_string", ["a", "a", "b", "c"], pa.large_string()),
  )
  def test_add_binary_like(self, values, binary_like_type):
    expected_counts = [{
        "values": b"a",
        "counts": 2.0
    }, {
        "values": b"b",
        "counts": 1.0
    }, {
        "values": b"c",
        "counts": 1.0
    }]
    sketch = _create_basic_sketch(pa.array(values, type=binary_like_type))
    estimate = sketch.Estimate()
    estimate.validate(full=True)
    self.assertEqual(estimate.to_pylist(), expected_counts)

  @parameterized.named_parameters(
      ("int8", [1, 1, 2, 3, None], pa.int8()),
      ("int16", [1, 1, 2, 3], pa.int16()),
      ("int32", [1, 1, 2, 3, None], pa.int32()),
      ("int64", [1, 1, 2, 3], pa.int64()),
      ("uint8", [1, 1, 2, 3], pa.uint8()),
      ("uint16", [1, None, 1, 2, 3], pa.uint16()),
      ("uint32", [1, 1, 2, 3], pa.uint32()),
      ("uint64", [1, 1, 2, 3, None], pa.uint64()),
  )
  def test_add_integer(self, values, integer_type):
    expected_counts = [{
        "values": b"1",
        "counts": 2.0
    }, {
        "values": b"2",
        "counts": 1.0
    }, {
        "values": b"3",
        "counts": 1.0
    }]
    sketch = _create_basic_sketch(pa.array(values, type=integer_type))
    estimate = sketch.Estimate()
    estimate.validate(full=True)
    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_add_weighted_values(self):
    items = pa.array(["a", "a", "b", "c"], type=pa.string())
    weights = pa.array([4, 3, 2, 1], type=pa.float32())
    sketch = _create_basic_sketch(items, weights=weights)

    expected_counts = [{
        "values": b"a",
        "counts": 7.0
    }, {
        "values": b"b",
        "counts": 2.0
    }, {
        "values": b"c",
        "counts": 1.0
    }]
    estimate = sketch.Estimate()
    estimate.validate(full=True)

    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_add_invalid_weights(self):
    items = pa.array(["a", "a", "b", "c"], type=pa.string())
    weights = pa.array([4, 3, 2, 1], type=pa.int64())
    with self.assertRaisesRegex(
        RuntimeError, "INVALID_ARGUMENT: Weight array must be float type."):
      _create_basic_sketch(items, weights=weights)

  def test_add_unsupported_type(self):
    values = pa.array([True, False], pa.bool_())
    sketch = sketches.MisraGriesSketch(_NUM_BUCKETS)
    with self.assertRaisesRegex(RuntimeError, "UNIMPLEMENTED: bool"):
      sketch.AddValues(values)

  def test_reverse_order(self):
    items = pa.array(["a", "a", "b", "c"], type=pa.string())
    sketch = _create_basic_sketch(items, reverse=True)
    expected_counts = [
        {"values": b"a", "counts": 2.0},
        {"values": b"c", "counts": 1.0},
        {"values": b"b", "counts": 1.0},
    ]
    estimate = sketch.Estimate()
    estimate.validate(full=True)

    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_replace_invalid_utf8(self):
    values1 = pa.array([
        b"a",
        b"\x80",  # invalid
        b"\xC1",  # invalid
    ])
    values2 = pa.array([
        b"\xc0\x80",  # invalid
        b"a"])
    sketch1 = sketches.MisraGriesSketch(
        _NUM_BUCKETS,
        invalid_utf8_placeholder=b"<BYTES>")
    sketch1.AddValues(values1)

    sketch2 = sketches.MisraGriesSketch(
        _NUM_BUCKETS,
        invalid_utf8_placeholder=b"<BYTES>")
    sketch2.AddValues(values2)

    serialized1 = sketch1.Serialize()
    serialized2 = sketch2.Serialize()

    sketch1 = sketches.MisraGriesSketch.Deserialize(serialized1)
    sketch2 = sketches.MisraGriesSketch.Deserialize(serialized2)
    sketch1.AddValues(values2)
    sketch1.Merge(sketch2)

    actual = sketch1.Estimate()
    actual.validate(full=True)
    self.assertEqual(actual.to_pylist(), [
        {"values": b"<BYTES>", "counts": 4.0},
        {"values": b"a", "counts": 3.0},
    ])

  def test_no_replace_invalid_utf8(self):
    sketch = sketches.MisraGriesSketch(
        _NUM_BUCKETS)
    sketch.AddValues(pa.array([b"\x80"]))
    actual = sketch.Estimate()
    self.assertEqual(actual.to_pylist(), [
        {"values": b"\x80", "counts": 1.0},
    ])

  def test_large_string_threshold(self):
    values1 = pa.array(["a", "bbb", "c", "d", "eeff"])
    values2 = pa.array(["a", "gghh"])
    sketch1 = sketches.MisraGriesSketch(
        _NUM_BUCKETS,
        large_string_threshold=2,
        large_string_placeholder=b"<LARGE>")
    sketch1.AddValues(values1)

    sketch2 = sketches.MisraGriesSketch(
        _NUM_BUCKETS,
        large_string_threshold=2,
        large_string_placeholder=b"<LARGE>")
    sketch2.AddValues(values2)

    serialized1 = sketch1.Serialize()
    serialized2 = sketch2.Serialize()

    sketch1 = sketches.MisraGriesSketch.Deserialize(serialized1)
    sketch2 = sketches.MisraGriesSketch.Deserialize(serialized2)
    sketch1.AddValues(values2)
    sketch1.Merge(sketch2)

    actual = sketch1.Estimate()
    actual.validate(full=True)
    self.assertEqual(actual.to_pylist(), [
        {"values": b"<LARGE>", "counts": 4.0},
        {"values": b"a", "counts": 3.0},
        {"values": b"c", "counts": 1.0},
        {"values": b"d", "counts": 1.0},
    ])

  def test_invalid_large_string_replacing_config(self):
    with self.assertRaisesRegex(
        RuntimeError,
        "Must provide both or neither large_string_threshold and "
        "large_string_placeholder"):
      _ = sketches.MisraGriesSketch(_NUM_BUCKETS, large_string_threshold=1024)

    with self.assertRaisesRegex(
        RuntimeError,
        "Must provide both or neither large_string_threshold and "
        "large_string_placeholder"):
      _ = sketches.MisraGriesSketch(
          _NUM_BUCKETS, large_string_placeholder=b"<L>")

  def test_many_uniques(self):
    # Test that the tail elements with equal counts are not discarded after
    # `AddValues` call.
    sketch = _create_basic_sketch(pa.array(["a", "b", "c", "a"]), num_buckets=2)
    estimate = sketch.Estimate()
    estimate.validate(full=True)
    # Since "b" and "c" have equal counts and neither token has count > 4/2, any
    # combination is possible.
    all_counts = [{
        "values": b"a",
        "counts": 2.0
    }, {
        "values": b"b",
        "counts": 1.0
    }, {
        "values": b"c",
        "counts": 1.0
    }]
    self.assertIn(
        tuple(estimate.to_pylist()),
        list(itertools.combinations(all_counts, 2)))

  def test_merge(self):
    sketch1 = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))
    sketch2 = _create_basic_sketch(pa.array(["d", "a"]))

    sketch1.Merge(sketch2)
    estimate = sketch1.Estimate()
    estimate.validate(full=True)
    expected_counts = [{
        "values": b"a",
        "counts": 3.0
    }, {
        "values": b"b",
        "counts": 1.0
    }, {
        "values": b"c",
        "counts": 1.0
    }, {
        "values": b"d",
        "counts": 1.0
    }]

    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_merge_equal_to_kth_weights(self):
    # Test that tail elements with equal counts are not discarded after
    # `Compress` call.
    sketch1 = _create_basic_sketch(
        pa.array(["a"] * 5 + ["b"] * 5 + ["c"] * 4 + ["a"] * 4), num_buckets=3)
    sketch2 = _create_basic_sketch(
        pa.array(["d"] * 4 + ["a"] * 2), num_buckets=3)
    sketch1.Merge(sketch2)
    estimate = sketch1.Estimate()
    estimate.validate(full=True)
    # Since "c" and "d" have equal counts, the last entry may be either.
    expected_counts1 = [{
        "values": b"a",
        "counts": 11.0
    }, {
        "values": b"b",
        "counts": 5.0
    }, {
        "values": b"c",
        "counts": 4.0
    }]
    expected_counts2 = expected_counts1.copy()
    expected_counts2[2] = {"values": b"d", "counts": 4.0}
    self.assertIn(estimate.to_pylist(), [expected_counts1, expected_counts2])

  def test_merge_with_extra_items(self):
    # Each of these sketches get more values than `num_buckets`. This will
    # result into removal of less frequent elements from the main buffer and
    # adding them to a buffer of extra elements.
    # Here we're testing that merging of sketches having extra elements is
    # correct and results in a sketch that produces the requested number of
    # elements.
    sketch1 = _create_basic_sketch(
        pa.array(["a"] * 3 + ["b"] * 2 + ["c", "d"]), num_buckets=3)
    sketch2 = _create_basic_sketch(
        pa.array(["e"] * 3 + ["f"] * 2 + ["g", "h"]), num_buckets=3)
    sketch3 = _create_basic_sketch(
        pa.array(["i"] * 2 + ["j", "k", "l"]), num_buckets=3)
    sketch1.Merge(sketch2)
    sketch1.Merge(sketch3)
    estimate = sketch1.Estimate()
    estimate.validate(full=True)

    # Due to large number of unique elements (relative to `num_buckets`), the
    # total estimated count error is 5.
    def get_expected_counts():
      for least_frequent_item in [b"b", b"f", b"i"]:
        yield [{
            "values": b"a",
            "counts": 5.0
        }, {
            "values": b"e",
            "counts": 5.0
        }, {
            "values": least_frequent_item,
            "counts": 5.0
        }]

    self.assertIn(estimate.to_pylist(), list(get_expected_counts()))

  def test_picklable(self):
    sketch = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))
    pickled = pickle.dumps(sketch, 2)
    self.assertIsInstance(pickled, bytes)
    unpickled = pickle.loads(pickled)
    self.assertIsInstance(unpickled, sketches.MisraGriesSketch)

    estimate = unpickled.Estimate()
    estimate.validate(full=True)
    expected_counts = [{
        "values": b"a",
        "counts": 2.0
    }, {
        "values": b"b",
        "counts": 1.0
    }, {
        "values": b"c",
        "counts": 1.0
    }]

    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_serialization(self):
    sketch = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))

    serialized = sketch.Serialize()
    self.assertIsInstance(serialized, bytes)

    deserialized = sketches.MisraGriesSketch.Deserialize(serialized)
    self.assertIsInstance(deserialized, sketches.MisraGriesSketch)

    estimate = deserialized.Estimate()
    estimate.validate(full=True)
    expected_counts = [{
        "values": b"a",
        "counts": 2.0
    }, {
        "values": b"b",
        "counts": 1.0
    }, {
        "values": b"c",
        "counts": 1.0
    }]

    self.assertEqual(estimate.to_pylist(), expected_counts)

  def test_deserialize_fails_with_exception(self):
    with self.assertRaisesRegex(RuntimeError,
                                "Failed to parse MisraGries sketch"):
      sketches.MisraGriesSketch.Deserialize("I am no proto")

if __name__ == "__main__":
  absltest.main()
