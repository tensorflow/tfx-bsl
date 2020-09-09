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
"""TFXIO telemetry test utilities."""

from typing import List, Text
import unittest

import apache_beam as beam


def ValidateMetrics(
    test: unittest.TestCase,
    pipeline_result: beam.runners.runner.PipelineResult,
    telemetry_descriptors: List[Text],
    logical_format: Text, physical_format: Text):
  all_metrics = pipeline_result.metrics()
  maintained_metrics = all_metrics.query(
      beam.metrics.metric.MetricsFilter().with_namespace(
          "tfx." + ".".join(telemetry_descriptors + ["io"])))
  test.assertIsNot(maintained_metrics, None)
  counters = maintained_metrics[beam.metrics.metric.MetricResults.COUNTERS]
  test.assertTrue(counters)
  distributions = maintained_metrics[
      beam.metrics.metric.MetricResults.DISTRIBUTIONS]
  test.assertTrue(distributions)
  for m in counters + distributions:
    test.assertTrue(
        m.key.metric.name.startswith("LogicalFormat[%s]-PhysicalFormat[%s]-" %
                                     (logical_format, physical_format)),
        m.key.metric.name)
