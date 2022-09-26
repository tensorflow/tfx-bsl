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
"""Tests for sql_util."""

import sys
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from tfx_bsl.statistics import sql_util

from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import statistics_pb2

_FEATURE_STATS = text_format.Parse(
    """
    type: FLOAT
    num_stats {
      mean: 2.6666666666666665
      histograms {
        num_nan: 1
      }
      histograms {
        num_nan: 1
        type: QUANTILES
      }
    }
    """, statistics_pb2.FeatureNameStatistics())

_FEATURE_STATS_TEST = text_format.Parse(
    """
    type: FLOAT
    num_stats {
      mean: 98
    }
    """, statistics_pb2.FeatureNameStatistics())


class SqlUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'simple_return_true',
          'query': 'feature.num_stats.mean > 0',
          'expected_result': True,
      }, {
          'testcase_name': 'implicit_feature_return_true',
          'query': 'num_stats.mean > 0',
          'expected_result': True,
      }, {
          'testcase_name': 'simple_return_false',
          'query': 'feature.num_stats.mean < 0',
          'expected_result': False,
      }, {
          'testcase_name':
              'aggregate_return_true',
          'query':
              '(SELECT MAX(hist.num_nan) > 0 FROM UNNEST(feature.num_stats.histograms) as hist)',
          'expected_result':
              True,
      }, {
          'testcase_name': 'returns_wrong_type',
          'query': '(SELECT \'foo\')',
          'expect_error': True,
      }, {
          'testcase_name':
              'returns_multiple_values',
          'query':
              '(SELECT hist.num_nan FROM UNNEST(feature.num_stats.histograms) as hist)',
          'expect_error':
              True,
      }, {
          'testcase_name': 'invalid_query',
          'query': 'wiggle woggle',
          'expect_error': True,
      })
  @unittest.skipIf(
      sys.platform.startswith('win'),
      'SQL based validation is not supported on Windows.')
  def test_unary_predicate(self,
                           query,
                           expect_error=False,
                           expected_result=None):
    if expect_error:
      with self.assertRaises(RuntimeError):
        sql_util.EvaluateUnaryStatsPredicate(_FEATURE_STATS, query)
    else:
      self.assertEqual(
          sql_util.EvaluateUnaryStatsPredicate(_FEATURE_STATS, query),
          expected_result)

  @parameterized.named_parameters(
      {
          'testcase_name': 'simple_return_true',
          'query': 'feature_base.num_stats.mean < feature_test.num_stats.mean',
          'expected_result': True,
      }, {
          'testcase_name': 'simple_return_false',
          'query': 'feature_base.num_stats.mean > feature_test.num_stats.mean',
          'expected_result': False,
      })
  @unittest.skipIf(
      sys.platform.startswith('win'),
      'SQL based validation is not supported on Windows.')
  def test_binary_predicate(self, query, expected_result=None):

    self.assertEqual(
        sql_util.EvaluateBinaryStatsPredicate(_FEATURE_STATS,
                                              _FEATURE_STATS_TEST, query),
        expected_result)


if __name__ == '__main__':
  absltest.main()
