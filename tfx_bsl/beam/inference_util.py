# Copyright 2019 Google LLC
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
"""TensorAdapter."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function


import numpy as np
import pyarrow as pa
import pandas as pd
import base64
import json
import typing
from typing import List, Text


_RECORDBATCH_COLUMN = '__RAW_RECORD__'

class JSONAdapter(object):
    """A JSONAdapter converts a RecordBatch to a JSON strings.

    The conversion will take in a recordbatch that contains features from a 
    tf.train.Example and will return a list of dict like string (JSON) where 
    each item is a JSON representation of an example.

    - return format: [{ feature1: value1, ... }, ...]
    """


    def ToJSON(self, record_batch: pa.RecordBatch) -> List[Text]:
        """Returns a JSON string translated from `record_batch`.

        Args:
            record_batch: input RecordBatch.
        """

        df = record_batch.to_pandas()
        as_binary = df.columns.str.endswith("_bytes")
        df.loc[:, as_binary] = df.loc[:, as_binary].applymap(lambda x: {'b64': base64.b64encode(x).decode()})
        if _RECORDBATCH_COLUMN in df.columns:
            df = df.drop(labels=_RECORDBATCH_COLUMN, axis=1)

        return json.loads(df.to_json(orient='records'))