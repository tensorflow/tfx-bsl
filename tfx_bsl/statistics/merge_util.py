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
"""Utilities for merging statistics proto shards."""
from typing import Iterable
from tfx_bsl import statistics
from tensorflow_metadata.proto.v0 import statistics_pb2

#  TODO(b/202910677): Consider removing this file once/if proto caster issues
#  are resolved.


def merge_dataset_feature_statistics(
    stats_protos: Iterable[statistics_pb2.DatasetFeatureStatistics],
) -> statistics_pb2.DatasetFeatureStatisticsList:
  """Merge a collection of DatasetFeatureStatistics into a single result.

  Args:
    stats_protos: An iterable over DatasetFeatureStatistics shards.

  Returns:
    A DatasetFeatureStatisticsList formed by merging all inputs.
  """
  acc = statistics.DatasetListAccumulator()
  for stats_proto in stats_protos:
    acc.MergeDatasetFeatureStatistics(stats_proto.SerializeToString())
  res_str = acc.GetAtVersion(0)
  result = statistics_pb2.DatasetFeatureStatisticsList()
  result.ParseFromString(res_str)
  return result
