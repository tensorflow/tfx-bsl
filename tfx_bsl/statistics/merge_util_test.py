# Copyright 2021 Google LLC
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
"""Tests for tfx_bsl.statistics.DatasetListAccumulator."""

from tfx_bsl.statistics import merge_util

from absl.testing import absltest
from tensorflow_metadata.proto.v0 import statistics_pb2


class MergeDatasetFeatureStatisticsTest(absltest.TestCase):

  # Basic tests that the python bindings work; more coverage is in
  # merge_util_test.cc
  def test_merges_two_inputs(self):
    values = [
        statistics_pb2.DatasetFeatureStatistics(name="slice1"),
        statistics_pb2.DatasetFeatureStatistics(name="slice2")
    ]
    result = merge_util.merge_dataset_feature_statistics(values)
    expected = statistics_pb2.DatasetFeatureStatisticsList()
    expected.datasets.extend(values)
    self.assertEqual(expected, result)


class MergeDatasetFeatureStatisticsListTest(absltest.TestCase):

  # Basic tests that the python bindings work; more coverage is in
  # merge_util_test.cc
  def test_merges_two_inputs(self):
    values = [
        statistics_pb2.DatasetFeatureStatisticsList(
            datasets=[statistics_pb2.DatasetFeatureStatistics(name="slice1")]),
        statistics_pb2.DatasetFeatureStatisticsList(
            datasets=[statistics_pb2.DatasetFeatureStatistics(name="slice2")])
    ]
    result = merge_util.merge_dataset_feature_statistics_list(values)
    expected = statistics_pb2.DatasetFeatureStatisticsList(datasets=[
        statistics_pb2.DatasetFeatureStatistics(name="slice1"),
        statistics_pb2.DatasetFeatureStatistics(name="slice2")
    ])
    self.assertEqual(expected, result)

if __name__ == "__main__":
  absltest.main()
