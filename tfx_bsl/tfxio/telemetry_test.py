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

import collections

import apache_beam as beam
import pyarrow as pa
from tfx_bsl.tfxio import telemetry

from absl.testing import absltest
from absl.testing import parameterized


# Used for constructing expected values for a distribution metric.
_Distribution = collections.namedtuple(
    "_Distribution", ["min", "max", "count", "sum"])

# Used for constructing expected values for a distribution metric. The tests
# will not check the exact values in that distribution, instead, only check
# for its presence.
_IGNORE_VALUES = object()


_LOGICAL_FORMAT = "some_df"
_PHYSICAL_FORMAT = "some_pf"


def _GetMetricName(name):
  return "LogicalFormat[%s]-PhysicalFormat[%s]-%s" % (_LOGICAL_FORMAT,
                                                      _PHYSICAL_FORMAT, name)

_PROFILE_RECORD_BATCHES_TEST_CASES = [
    dict(
        testcase_name="multi_column_mixed_type",
        record_batch=pa.RecordBatch.from_arrays([
            pa.array([["abc", "def"], None, ["g"], []]),
            pa.array([[1], [2], [3], [4]]),
            pa.array([[1.0], [], [], [0.5]]),
            pa.array([None, None, None, None], type=pa.null()),
        ], ["f1", "f2", "f3", "f4"]),
        expected_distributions={
            _GetMetricName("num_rows"): _Distribution(
                min=1, max=3, count=2, sum=4
            ),
            # byte size of an arrow array may change over time. Do not test
            # for exact values.
            _GetMetricName("record_batch_byte_size"): _IGNORE_VALUES,
            _GetMetricName("num_columns"):
                _Distribution(min=3, max=4, count=4, sum=15),
            _GetMetricName("num_feature_values"):
                _Distribution(sum=13, count=16, min=0, max=2),
            _GetMetricName("num_feature_values[INT]"):
                _Distribution(sum=4, count=4, min=1, max=1),
            _GetMetricName("num_feature_values[FLOAT]"):
                _Distribution(sum=2, count=4, min=0, max=1),
            _GetMetricName("num_feature_values[STRING]"):
                _Distribution(sum=3, count=4, min=0, max=2),
            _GetMetricName("num_feature_values[NULL]"):
                _Distribution(sum=4, count=4, min=1, max=1),
        },
        expected_counters={
            _GetMetricName("num_cells[NULL]"): 4,
            _GetMetricName("num_cells[STRING]"): 3,
            _GetMetricName("num_cells[INT]"): 4,
            _GetMetricName("num_cells[FLOAT]"): 4,
        },
        telemetry_descriptors=["test", "component"],
        expected_namespace="tfx.test.component.io",
    ),
    dict(
        testcase_name="deeply_nested_list",
        record_batch=pa.RecordBatch.from_arrays(
            [pa.array([[[[1, 2, 3], [4]], [[5]]], [[None, [1]]]])], ["f1"]),
        expected_distributions={
            _GetMetricName("num_rows"): _Distribution(
                min=1, max=1, count=2, sum=2
            ),
            _GetMetricName("record_batch_byte_size"): _IGNORE_VALUES,
            _GetMetricName("num_columns"):
                _Distribution(min=1, max=1, count=2, sum=2),
            # First row: 5 values; second row: 1 value
            _GetMetricName("num_feature_values"):
                _Distribution(sum=6, count=2, min=1, max=5),
            _GetMetricName("num_feature_values[INT]"):
                _Distribution(sum=6, count=2, min=1, max=5),
        },
        expected_counters={
            _GetMetricName("num_cells[OTHER]"): 2,
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
            _GetMetricName("num_rows"): _Distribution(
                min=1, max=1, count=2, sum=2
            ),
            _GetMetricName("record_batch_byte_size"): _IGNORE_VALUES,
            _GetMetricName("num_columns"):
                _Distribution(min=1, max=1, count=2, sum=2),
            # min came from the second row, number of int values.
            # max came from the first row, number of string values.
            _GetMetricName("num_feature_values"):
                _Distribution(sum=6, count=4, min=0, max=3),
            # First row: 2 int values; second row: 1 int value.
            _GetMetricName("num_feature_values[INT]"):
                _Distribution(sum=3, count=2, min=1, max=2),
            # First row: 3 string values; second row: 0 string value.
            _GetMetricName("num_feature_values[STRING]"):
                _Distribution(sum=3, count=2, min=0, max=3),
        },
        expected_counters={
            _GetMetricName("num_cells[OTHER]"): 2,
        }),
]


class TelemetryTest(parameterized.TestCase):

  def _AssertDistributionEqual(
      self, beam_distribution_result, expected, msg=None):
    if expected is _IGNORE_VALUES:
      return
    try:
      self.assertEqual(beam_distribution_result.min, expected.min)
      self.assertEqual(beam_distribution_result.max, expected.max)
      self.assertEqual(beam_distribution_result.sum, expected.sum)
      self.assertEqual(beam_distribution_result.count, expected.count)
    except AssertionError:
      raise AssertionError("{}Expected: {}; got: {}".format(
          (msg + ": ") if msg else "", expected, beam_distribution_result))

  @parameterized.named_parameters(*_PROFILE_RECORD_BATCHES_TEST_CASES)
  def testProfileRecordBatches(self,
                               record_batch,
                               expected_distributions,
                               expected_counters,
                               telemetry_descriptors=None,
                               expected_namespace="tfx.UNKNOWN_COMPONENT.io"):
    p = beam.Pipeline()
    _ = (
        p
        # Slice the input into two pieces to make sure profiling can handle
        # sliced RecordBatches.
        | "CreateTestData" >> beam.Create(
            [record_batch.slice(0, 1),
             record_batch.slice(1)])
        | "Profile" >> telemetry.ProfileRecordBatches(
            telemetry_descriptors, _LOGICAL_FORMAT, _PHYSICAL_FORMAT, 1.0))
    runner = p.run()
    runner.wait_until_finish()
    all_metrics = runner.metrics()
    maintained_metrics = all_metrics.query(
        beam.metrics.metric.MetricsFilter().with_namespace(expected_namespace))

    counters = maintained_metrics[beam.metrics.metric.MetricResults.COUNTERS]
    self.assertLen(counters, len(expected_counters))
    for counter in counters:
      self.assertEqual(
          counter.result, expected_counters[counter.key.metric.name])

    distributions = maintained_metrics[
        beam.metrics.metric.MetricResults.DISTRIBUTIONS]
    self.assertLen(distributions, len(expected_distributions))
    for dist in distributions:
      self.assertIn(dist.key.metric.name, expected_distributions)
      self._AssertDistributionEqual(
          dist.result, expected_distributions[dist.key.metric.name],
          dist.key.metric.name)

  @parameterized.named_parameters(
      dict(
          testcase_name="no_descriptors",
          telemetry_descriptors=None,
          expected_namespace="tfx.UNKNOWN_COMPONENT.io"),
      dict(
          testcase_name="with_descriptors",
          telemetry_descriptors=["test", "component"],
          expected_namespace="tfx.test.component.io"))
  def testProfileRawRecords(self, telemetry_descriptors, expected_namespace):
    p = beam.Pipeline()
    _ = (
        p
        | "CreateTestData" >> beam.Create([b"aaa", b"bbbb"])
        | "Profile" >> telemetry.ProfileRawRecords(
            telemetry_descriptors, _LOGICAL_FORMAT, _PHYSICAL_FORMAT))
    runner = p.run()
    runner.wait_until_finish()
    all_metrics = runner.metrics()
    maintained_metrics = all_metrics.query(
        beam.metrics.metric.MetricsFilter().with_namespace(expected_namespace))
    counters = maintained_metrics[beam.metrics.metric.MetricResults.COUNTERS]
    self.assertLen(counters, 1)
    num_records_counter = counters[0]
    self.assertEqual(_GetMetricName("num_raw_records"),
                     num_records_counter.key.metric.name)
    self.assertEqual(2, num_records_counter.result)

    distributions = maintained_metrics[
        beam.metrics.metric.MetricResults.DISTRIBUTIONS]
    self.assertLen(distributions, 1)
    byte_size_distribution = distributions[0]
    self.assertEqual(_GetMetricName("raw_record_byte_size"),
                     byte_size_distribution.key.metric.name)
    self._AssertDistributionEqual(byte_size_distribution.result,
                                  _Distribution(min=3, max=4, count=2, sum=7))


if __name__ == "__main__":
  absltest.main()
