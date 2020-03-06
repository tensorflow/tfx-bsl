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
"""Tests for tfx_bsl.tfxio.telemetry."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import apache_beam as beam
import pyarrow as pa
from tfx_bsl.tfxio import telemetry

from absl.testing import absltest
from absl.testing import parameterized


_Distribution = collections.namedtuple(
    "_Distribution", ["min", "max", "count", "sum"])


_TEST_CASES = [
    dict(
        testcase_name="multi_column_mixed_type",
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([["abc", "def"], None, ["g"], []]),
            pa.array([[1], [2], [3], [4]]),
            pa.array([[1.0], [], [], [0.5]]),
            pa.array([None, None, None, None], type=pa.null()),
        ], ["f1", "f2", "f3", "f4"]),
        expected_distributions={
            "record_batch_byte_size":
                _Distribution(min=137, max=137, count=1, sum=137),
            "num_columns":
                _Distribution(min=3, max=4, count=4, sum=15),
            "num_feature_values":
                _Distribution(sum=13, count=16, min=0, max=2),
            "num_feature_values[INT]":
                _Distribution(sum=4, count=4, min=1, max=1),
            "num_feature_values[FLOAT]":
                _Distribution(sum=2, count=4, min=0, max=1),
            "num_feature_values[STRING]":
                _Distribution(sum=3, count=4, min=0, max=2),
            "num_feature_values[NULL]":
                _Distribution(sum=4, count=4, min=1, max=1),
        },
        expected_counters={
            "num_rows": 4,
            "num_cells[NULL]": 4,
            "num_cells[STRING]": 3,
            "num_cells[INT]": 4,
            "num_cells[FLOAT]": 4,
        }),
    dict(
        testcase_name="deeply_nested_list",
        record_batch=pa.RecordBatch.from_arrays(
            [pa.array([[[[1, 2, 3], [4]], [[5]]], [[None, [1]]]])], ["f1"]),
        expected_distributions={
            "record_batch_byte_size":
                _Distribution(min=104, max=104, count=1, sum=104),
            "num_columns":
                _Distribution(min=1, max=1, count=2, sum=2),
            # First row: 5 values; second row: 1 value
            "num_feature_values":
                _Distribution(sum=6, count=2, min=1, max=5),
            "num_feature_values[INT]":
                _Distribution(sum=6, count=2, min=1, max=5),
        },
        expected_counters={
            "num_rows": 2,
            "num_cells[OTHER]": 2,
        }),
    dict(
        testcase_name="struct",
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([[
                {
                    "foo": ["a", "b"],
                    "bar": None
                },
                {
                    "foo": ["c"],
                    "bar": [1, 2]
                },
            ], [{
                "foo": None,
                "bar": [3]
            }]])
        ], ["f1"]),
        expected_distributions={
            "record_batch_byte_size":
                _Distribution(min=93, max=93, count=1, sum=93),
            "num_columns":
                _Distribution(min=1, max=1, count=2, sum=2),
            # min came from the second row, number of int values.
            # max came from the first row, number of string values.
            "num_feature_values":
                _Distribution(sum=6, count=4, min=0, max=3),
            # First row: 2 int values; second row: 1 int value.
            "num_feature_values[INT]":
                _Distribution(sum=3, count=2, min=1, max=2),
            # First row: 3 string values; second row: 0 string value.
            "num_feature_values[STRING]":
                _Distribution(sum=3, count=2, min=0, max=3),
        },
        expected_counters={
            "num_rows": 2,
            "num_cells[OTHER]": 2,
        }),
]


class TelemetryTest(parameterized.TestCase):

  def _AssertDistributionEqual(
      self, beam_distribution_result, expected, msg=None):
    try:
      self.assertEqual(beam_distribution_result.min, expected.min)
      self.assertEqual(beam_distribution_result.max, expected.max)
      self.assertEqual(beam_distribution_result.sum, expected.sum)
      self.assertEqual(beam_distribution_result.count, expected.count)
    except AssertionError:
      raise AssertionError("{}Expected: {}; got: {}".format(
          (msg + ": ") if msg else "", expected, beam_distribution_result))

  @parameterized.named_parameters(*_TEST_CASES)
  def testTelemetry(self, record_batch,
                    expected_distributions, expected_counters):
    p = beam.Pipeline()
    _ = (p
         | "CreateTestData" >> beam.Create([record_batch])
         | "Profile" >> telemetry.ProfileRecordBatches(
             ["test", "component"], 1.0))
    runner = p.run()
    runner.wait_until_finish()
    all_metrics = runner.metrics()
    maintained_metrics = all_metrics.query(
        beam.metrics.metric.MetricsFilter().with_namespace(
            "tfx.io.test.component"))

    counters = maintained_metrics[beam.metrics.metric.MetricResults.COUNTERS]
    self.assertLen(counters, len(expected_counters))
    for counter in counters:
      self.assertEqual(
          counter.result, expected_counters[counter.key.metric.name])

    distributions = maintained_metrics[
        beam.metrics.metric.MetricResults.DISTRIBUTIONS]
    self.assertLen(distributions, len(expected_distributions))
    for dist in distributions:
      self._AssertDistributionEqual(
          dist.result, expected_distributions[dist.key.metric.name],
          dist.key.metric.name)


if __name__ == "__main__":
  absltest.main()
