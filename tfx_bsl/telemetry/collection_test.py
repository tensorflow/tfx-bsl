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
"""Tests for tfx_bsl.telemetry.collection."""

import apache_beam as beam
import pyarrow as pa
from tfx_bsl.telemetry import collection
from absl.testing import absltest


class CollectionTest(absltest.TestCase):

  def testTrackRecordBatchBytes(self):
    inputs = pa.RecordBatch.from_arrays([pa.array([1, 2, 3], type=pa.int32())],
                                        ["f1"])
    expected_num_bytes = inputs.nbytes

    with beam.Pipeline() as p:
      _ = (
          p | beam.Create([inputs])
          | collection.TrackRecordBatchBytes("TestNamespace",
                                             "num_bytes_count"))

    pipeline_result = p.run()
    result_metrics = pipeline_result.metrics()
    actual_counter = result_metrics.query(
        beam.metrics.metric.MetricsFilter().with_name(
            "num_bytes_count"))["counters"]
    self.assertLen(actual_counter, 1)
    self.assertEqual(actual_counter[0].committed, expected_num_bytes)


if __name__ == "__main__":
  absltest.main()
