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
import typing
import json
from typing import Dict


_RECORDBATCH_COLUMN = '__RAW_RECORD__'

class JSONAdapter(object):
    """A JSONAdapter converts a RecordBatch to a JSON strings.

    The conversion will take in a recordbatch that contains features from a 
    tf.train.Example and will return a list of dict like string (JSON) where 
    each item represent 
    The conversion is determined by both the Arrow schema and the
    TensorRepresentations, which must be provided at the initialization time.
    Each TensorRepresentation contains the information needed to translates one
    or more columns in a RecordBatch of the given Arrow schema into a TF Tensor
    or CompositeTensor. They are contained in a Dict whose keys are
    the names of the tensors, which will be the keys of the Dict produced by
    ToBatchTensors().
    """


    def ToJSON(self, record_batch: pa.RecordBatch) -> Dict[Text, Any]:
        """Returns a JSON string translated from `record_batch`.

        Args:
            record_batch: input RecordBatch.
        """

        df = record_batch.to_pandas()
        if _RECORDBATCH_COLUMN in df.columns:
            df = df.drop(labels=_RECORDBATCH_COLUMN, axis=1)

        return json.loads(df.to_json(orient='records'))