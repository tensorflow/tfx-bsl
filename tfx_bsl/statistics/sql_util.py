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
"""Provides SQL based validation helpers. Experimental."""

from tfx_bsl import statistics
from tensorflow_metadata.proto.v0 import statistics_pb2


def EvaluateUnaryStatsPredicate(feature: statistics_pb2.FeatureNameStatistics,
                                query: str) -> bool:
  """Evaluates a GoogleSQL expression over a single FeatureNameStatistics.

  Args:
    feature: statistics for one feature.
    query: A GoogleSQL expression resolving to a boolean value. The passed
      feature statistics are bound to `feature`. See sql_util_test.py for usage.

  Returns:
    The result of the query.

  Raises:
    RuntimeError: On failure.
  """

  return statistics.EvaluateUnaryStatsPredicate(feature.SerializeToString(),
                                                query)


def EvaluateBinaryStatsPredicate(
    feature_base: statistics_pb2.FeatureNameStatistics,
    feature_test: statistics_pb2.FeatureNameStatistics, query: str) -> bool:
  """Evaluates a GoogleSQL expression over a pair of FeatureNameStatistics.

  Args:
    feature_base: baseline statistics.
    feature_test: test statistics.
    query: A GoogleSQL expression resolving to a boolean value. The passed
      feature statistics are bound to `feature_base` and `feature_test`. See
      sql_util_test.py for usage.

  Returns:
    The result of the query.

  Raises:
    RuntimeError: On failure.
  """
  return statistics.EvaluateBinaryStatsPredicate(
      feature_base.SerializeToString(), feature_test.SerializeToString(), query)
