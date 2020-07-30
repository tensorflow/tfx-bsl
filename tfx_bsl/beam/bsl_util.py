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
"""TFX-BSL util"""

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
from typing import Dict, List, Text, Any, Set, Optional
from tfx_bsl.beam.bsl_constants import _RECORDBATCH_COLUMN
from tfx_bsl.beam.bsl_constants import KERAS_INPUT_SUFFIX
   

def RecordToJSON(record_batch: pa.RecordBatch, prepare_instances_serialized) -> List[Text]:
  """Returns a JSON string translated from `record_batch`.

    The conversion will take in a recordbatch that contains features from a 
    tf.train.Example and will return a list of dict like string (JSON) where 
    each item is a JSON representation of an example.
    - return format: [{ feature1: value1, ... }, ...]

  Args:
  record_batch: input RecordBatch.
  """
  def flatten(element: List[Any]):
    if len(element) == 1:
        return element[0]
    return element

  df = record_batch.to_pandas()
  if prepare_instances_serialized: 
    return [{'b64': base64.b64encode(value).decode()} for value in df[_RECORDBATCH_COLUMN]]
  else:
    as_binary = df.columns.str.endswith("_bytes")
    # Handles the case where there is only one entry
    if len(df) == 1:
      df.loc[:, as_binary] = df.loc[:, as_binary].applymap(
        lambda feature: [{'b64': base64.b64encode(feature).decode()}])
    else:
      df.loc[:, as_binary] = df.loc[:, as_binary].applymap(
          lambda feature: [{'b64': base64.b64encode(value).decode()} for value in feature])

    if _RECORDBATCH_COLUMN in df.columns:
      df = df.drop(labels=_RECORDBATCH_COLUMN, axis=1)
    df = df.applymap(lambda x: flatten(x))
    return json.loads(df.to_json(orient='records'))


def find_input_name_in_features(features: Set[Text],
                                input_name: Text) -> Optional[Text]:
  """Maps input name to an entry in features. Returns None if not found."""
  if input_name in features:
    return input_name
  # Some keras models prepend '_input' to the names of the inputs
  # so try under '<name>_input' as well.
  elif (input_name.endswith(KERAS_INPUT_SUFFIX) and
        input_name[:-len(KERAS_INPUT_SUFFIX)] in features):
    return input_name[:-len(KERAS_INPUT_SUFFIX)]
  return None


def filter_tensors_by_input_names(
    tensors: Dict[Text, Any], 
    input_names: List[Text]) -> Optional[Dict[Text, Any]]:
  """Filter tensors by input names.
  In case we don't find the specified input name in the tensors and there
  exists only one input name, we assume we are feeding serialized examples to
  the model and return None.
  Args:
    tensors: Dict of tensors.
    input_names: List of input names.
  Returns:
    Filtered tensors.
  Raises:
    RuntimeError: When the specified input tensor cannot be found.
  """

  if not input_names:
    return None
  result = {}
  tensor_keys = set(tensors.keys())
  for name in input_names:
    tensor_name = find_input_name_in_features(tensor_keys, name)
    if tensor_name is None:
      # This should happen only in the case where the model takes serialized
      # examples as input. Else raise an exception.
      if len(input_names) == 1:
        return None
      raise RuntimeError(
          'Input tensor not found: {}. Existing keys: {}.'.format(
              name, ','.join(tensors.keys())))
    result[name] = tensors[tensor_name]
  return result
