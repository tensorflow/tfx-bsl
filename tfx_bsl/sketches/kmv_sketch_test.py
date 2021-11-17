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
"""Tests for tfx_bsl.sketches.KmvSketch."""

import pickle

import pyarrow as pa
from tfx_bsl import sketches

from absl.testing import absltest
from absl.testing import parameterized

_NUM_BUCKETS = 128


def _create_basic_sketch(values, num_buckets=_NUM_BUCKETS):
  sketch = sketches.KmvSketch(num_buckets)
  sketch.AddValues(values)
  return sketch


class KmvSketchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("binary", [b"a", b"a", b"b", b"c", None], pa.binary()),
      ("large_binary", [b"a", b"a", b"b", b"c"], pa.large_binary()),
      ("string", ["a", "a", "b", "c", None], pa.string()),
      ("large_string", ["a", "a", "b", "c"], pa.large_string()),
      ("int8", [1, 1, 2, 3, None], pa.int8()),
      ("int16", [1, 1, 2, 3], pa.int16()),
      ("int32", [1, 1, 2, 3, None], pa.int32()),
      ("int64", [1, 1, 2, 3], pa.int64()),
      ("uint8", [1, 1, 2, 3], pa.uint8()),
      ("uint16", [1, None, 1, 2, 3], pa.uint16()),
      ("uint32", [1, 1, 2, 3], pa.uint32()),
      ("uint64", [1, 1, 2, 3, None], pa.uint64()),
  )
  def test_add(self, values, type_):
    sketch = _create_basic_sketch(pa.array(values, type=type_))
    num_unique = sketch.Estimate()

    self.assertEqual(3, num_unique)

  def test_add_unsupported_type(self):
    values = pa.array([True, False], pa.bool_())
    sketch = sketches.KmvSketch(_NUM_BUCKETS)
    with self.assertRaisesRegex(RuntimeError, "UNIMPLEMENTED: bool"):
      sketch.AddValues(values)

  def test_merge(self):
    sketch1 = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))
    sketch2 = _create_basic_sketch(pa.array(["d", "a"]))

    sketch1.Merge(sketch2)
    num_unique = sketch1.Estimate()

    self.assertEqual(4, num_unique)

  def test_merge_error(self):
    sketch1 = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))
    sketch2 = _create_basic_sketch(pa.array(["d", "a"]), num_buckets=64)
    with self.assertRaisesRegex(
        Exception, "Both sketches must have the same number of buckets"):
      sketch1.Merge(sketch2)

  def test_picklable(self):
    sketch = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))
    pickled = pickle.dumps(sketch, 2)
    self.assertIsInstance(pickled, bytes)
    unpickled = pickle.loads(pickled)
    self.assertIsInstance(unpickled, sketches.KmvSketch)

    num_unique = unpickled.Estimate()
    self.assertEqual(3, num_unique)

  def test_serialization(self):
    sketch = _create_basic_sketch(pa.array(["a", "b", "c", "a"]))

    serialized = sketch.Serialize()
    self.assertIsInstance(serialized, bytes)

    deserialized = sketches.KmvSketch.Deserialize(serialized)
    self.assertIsInstance(deserialized, sketches.KmvSketch)

    num_unique = deserialized.Estimate()
    self.assertEqual(3, num_unique)

  def test_deserialize_fails_with_exception(self):
    with self.assertRaisesRegex(RuntimeError, "Failed to parse Kmv sketch"):
      sketches.KmvSketch.Deserialize("I am no proto")


if __name__ == "__main__":
  absltest.main()
