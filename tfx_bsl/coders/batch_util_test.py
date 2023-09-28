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
"""Tests for tfx_bsl.coders.batch_util."""

from absl.testing import flagsaver

import apache_beam as beam
from apache_beam.testing import util as beam_testing_util

from tfx_bsl.coders import batch_util
from absl.testing import absltest
from absl.testing import parameterized

_BATCH_RECORDS_TEST_CASES = (
    dict(
        testcase_name="fixed_batch_size",
        batch_size=5000,
        tfxio_use_byte_size_batching=False,
        expected_kwargs={"max_batch_size": 5000, "min_batch_size": 5000},
    ),
    dict(
        testcase_name="fixed_batch_size_byte_size_batching",
        batch_size=5000,
        tfxio_use_byte_size_batching=True,
        expected_kwargs={"max_batch_size": 5000, "min_batch_size": 5000},
    ),
    dict(
        testcase_name="batch_size_none",
        batch_size=None,
        tfxio_use_byte_size_batching=False,
        expected_kwargs={
            "min_batch_size": 1,
            "max_batch_size": 1000,
            "target_batch_overhead": 0.05,
            "target_batch_duration_secs": 1,
            "variance": 0.25,
        },
    ),
    dict(
        testcase_name="byte_size_batching",
        batch_size=None,
        tfxio_use_byte_size_batching=True,
        expected_kwargs={
            "min_batch_size": batch_util._TARGET_BATCH_BYTES_SIZE,
            "max_batch_size": batch_util._TARGET_BATCH_BYTES_SIZE,
            "element_size_fn": "dummy",
        },
        expected_element_contributions={
            b"dummy": 2560,  # Minimal contribution.
            b"dummy" * 10000: 50000,
        },
    ),
    dict(
        testcase_name="byte_size_batching_with_element_size_fn",
        batch_size=None,
        tfxio_use_byte_size_batching=True,
        expected_kwargs={
            "min_batch_size": batch_util._TARGET_BATCH_BYTES_SIZE,
            "max_batch_size": batch_util._TARGET_BATCH_BYTES_SIZE,
            "element_size_fn": "dummy",
        },
        element_size_fn=lambda kv: len(kv[0] or b"") + len(kv[1]),
        expected_element_contributions={
            (None, b"dummy"): 2560,  # Minimal contribution.
            (b"asd", b"dummy" * 10000): 50003,
        },
    ),
)


class BatchUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(*_BATCH_RECORDS_TEST_CASES)
  def testGetBatchElementsKwargs(
      self,
      batch_size,
      tfxio_use_byte_size_batching,
      expected_kwargs,
      element_size_fn=len,
      expected_element_contributions=None,
  ):
    with flagsaver.flagsaver(
        tfxio_use_byte_size_batching=tfxio_use_byte_size_batching
    ):
      kwargs = batch_util.GetBatchElementsKwargs(
          batch_size, element_size_fn=element_size_fn
      )
      # This parameter may not be present in some Beam versions that we support.
      target_batch_duration_secs_including_fixed_cost = kwargs.pop(
          "target_batch_duration_secs_including_fixed_cost", None
      )
      self.assertIn(target_batch_duration_secs_including_fixed_cost, {1, None})
      if expected_kwargs.pop("element_size_fn", None) is not None:
        self.assertIn("element_size_fn", kwargs)
        element_size_fn = kwargs.pop("element_size_fn")
        for (
            element,
            expected_contribution,
        ) in expected_element_contributions.items():
          self.assertEqual(
              element_size_fn(element),
              expected_contribution,
              msg=f"Unexpected contribution of element {element}",
          )
      self.assertDictEqual(kwargs, expected_kwargs)

  @parameterized.named_parameters(*_BATCH_RECORDS_TEST_CASES)
  def testBatchRecords(
      self,
      batch_size,
      tfxio_use_byte_size_batching,
      expected_kwargs,
      element_size_fn=len,
      expected_element_contributions=None,
  ):
    del expected_kwargs
    telemetry_descriptors = ["TestComponent"]
    input_records = (
        [b"asd", b"asds", b"123", b"gdgd" * 1000]
        if expected_element_contributions is None
        else expected_element_contributions.keys()
    )

    def AssertFn(batched_records):
      # We can't validate the actual sizes since they depend on test
      # environment.
      self.assertNotEmpty(batched_records)
      for batch in batched_records:
        self.assertIsInstance(batch, list)
        self.assertNotEmpty(batch)

    with flagsaver.flagsaver(
        tfxio_use_byte_size_batching=tfxio_use_byte_size_batching
    ):
      p = beam.Pipeline()
      batched_records_pcoll = (
          p
          | beam.Create(input_records)
          | batch_util.BatchRecords(
              batch_size, telemetry_descriptors, record_size_fn=element_size_fn
          )
      )
      beam_testing_util.assert_that(batched_records_pcoll, AssertFn)
      pipeline_result = p.run()
      pipeline_result.wait_until_finish()
      all_metrics = pipeline_result.metrics()
      maintained_metrics = all_metrics.query(
          beam.metrics.metric.MetricsFilter().with_namespace(
              "tfx." + ".".join(telemetry_descriptors)
          )
      )
      self.assertIsNotNone(maintained_metrics)
      counters = maintained_metrics[beam.metrics.metric.MetricResults.COUNTERS]
      self.assertLen(counters, 2)
      expected_counters = {
          "tfxio_use_byte_size_batching": int(tfxio_use_byte_size_batching),
          "desired_batch_size": batch_size or 0,
      }
      for counter in counters:
        self.assertEqual(
            counter.result, expected_counters[counter.key.metric.name]
        )


if __name__ == "__main__":
  absltest.main()
