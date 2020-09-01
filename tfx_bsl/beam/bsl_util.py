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
from typing import Dict, List, Text, Any, Set, Mapping, Optional
from tfx_bsl.beam.bsl_constants import _RECORDBATCH_COLUMN

_KERAS_INPUT_SUFFIX = '_input'

def ExtractSerializedExamplesFromRecordBatch(elements: pa.RecordBatch) -> List[Text]:
  serialized_examples = None
  for column_name, column_array in zip(elements.schema.names, elements.columns):
    if column_name == _RECORDBATCH_COLUMN:
      column_type = column_array.flatten().type
      if not (pa.types.is_binary(column_type) or pa.types.is_string(column_type)):
        raise ValueError(
          'Expected a list of serialized examples in bytes or as a string, got %s' %
          type(example))
      serialized_examples = column_array.flatten().to_pylist()
      break

  if not serialized_examples:
    raise ValueError('Raw examples not found.')

  return serialized_examples


def RecordToJSON(
  record_batch: pa.RecordBatch, prepare_instances_serialized) -> List[Mapping[Text, Any]]:
  """Returns a list of JSON dictionaries translated from `record_batch`.

    The conversion will take in a recordbatch that contains features from a 
    tf.train.Example and will return a list of dict like string (JSON) where 
    each item is a JSON representation of an example.

    Return:
      List of JSON dictionaries
      - format: [{ feature1: value1, feature2: [value2_1, value2_2]... }, ...]

  Args:
  record_batch: input RecordBatch.
  """

  # TODO (b/155912552): Handle this for sequence example.
  df = record_batch.to_pandas()
  if prepare_instances_serialized: 
    return [{'b64': base64.b64encode(value).decode()} for value in df[_RECORDBATCH_COLUMN]]
  else:
    as_binary = df.columns.str.endswith("_bytes")
    df.loc[:, as_binary] = df.loc[:, as_binary].applymap(
        lambda feature: [{'b64': base64.b64encode(value).decode()} for value in feature])

    if _RECORDBATCH_COLUMN in df.columns:
      df = df.drop(labels=_RECORDBATCH_COLUMN, axis=1)
    df = df.applymap(lambda values: values[0] if len(values) == 1 else values)
    return json.loads(df.to_json(orient='records'))


# TODO: Reuse these functions in TFMA.
def _find_input_name_in_features(features: Set[Text],
                                 input_name: Text) -> Optional[Text]:
  """Maps input name to an entry in features. Returns None if not found."""
  if input_name in features:
    return input_name
  # Some keras models prepend '_input' to the names of the inputs
  # so try under '<name>_input' as well.
  elif (input_name.endswith(_KERAS_INPUT_SUFFIX) and
        input_name[:-len(_KERAS_INPUT_SUFFIX)] in features):
    return input_name[:-len(_KERAS_INPUT_SUFFIX)]
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

  # The case where the model takes serialized examples as input.
  if len(input_names) == 1 and _find_input_name_in_features(tensor_keys, input_names[0]):
    return None

  for name in input_names:
    tensor_name = _find_input_name_in_features(tensor_keys, name)
    if tensor_name is None:
      raise RuntimeError(
          'Input tensor not found: {}. Existing keys: {}.'.format(
              name, ','.join(tensors.keys())))
    result[name] = tensors[tensor_name]
  return result
