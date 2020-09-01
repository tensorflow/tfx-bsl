# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx_bsl.bsl_util."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import base64
import json
import os
try:
  import unittest.mock as mock
except ImportError:
  import mock

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from google.protobuf import text_format
from tfx_bsl.beam import bsl_util
from tfx_bsl.beam.bsl_constants import _RECORDBATCH_COLUMN


class TestBslUtil(tf.test.TestCase):
    def test_request_body_with_binary_data(self):
        record_batch_remote = pa.RecordBatch.from_arrays(
        [
            pa.array([["ASa8asdf", "ASa8asdf"]], type=pa.list_(pa.binary())),
            pa.array([["JLK7ljk3"]], type=pa.list_(pa.utf8())),
            pa.array([[1, 2]], type=pa.list_(pa.int32())),
            pa.array([[4.5, 5, 5.5]], type=pa.list_(pa.float32()))
        ],
        ['x_bytes', 'x', 'y', 'z']
        )

        result = list(bsl_util.RecordToJSON(record_batch_remote, False))
        self.assertEqual([
            {
                'x_bytes': [
                    {'b64': 'QVNhOGFzZGY='}, 
                    {'b64': 'QVNhOGFzZGY='}
                ],
                'x': 'JLK7ljk3',
                'y': [1, 2],
                'z': [4.5, 5, 5.5]
            },
        ], result)

    def test_request_serialized_example(self):
        example = text_format.Parse(
        """
        features {
            feature { key: "x_bytes" value { bytes_list { value: ["ASa8asdf"] }}}
            feature { key: "x" value { bytes_list { value: "JLK7ljk3" }}}
            feature { key: "y" value { int64_list { value: [1, 2] }}}
        }
        """, tf.train.Example())
        
        serialized_example_remote = [example.SerializeToString()]
        record_batch_remote = pa.RecordBatch.from_arrays(
            [
                pa.array([["ASa8asdf"]], type=pa.list_(pa.binary())),
                pa.array([["JLK7ljk3"]], type=pa.list_(pa.utf8())),
                pa.array([[1, 2]], type=pa.list_(pa.int32())),
                pa.array([[4.5, 5, 5.5]], type=pa.list_(pa.float32())),
                serialized_example_remote
            ],
            ['x_bytes', 'x', 'y', 'z', _RECORDBATCH_COLUMN]
        )

        result = list(bsl_util.RecordToJSON(record_batch_remote, True))
        self.assertEqual(result, [{
            'b64': base64.b64encode(example.SerializeToString()).decode()
        }])


if __name__ == '__main__':
  tf.test.main()
