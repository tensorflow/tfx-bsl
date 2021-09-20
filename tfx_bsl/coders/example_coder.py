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
"""Example coders."""
from typing import List, Dict

import pyarrow as pa
from tfx_bsl.tfxio import tensor_representation_util

from tensorflow_metadata.proto.v0 import schema_pb2

# pylint: disable=unused-import
# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.coders import ExamplesToRecordBatchDecoder
  from tfx_bsl.cc.tfx_bsl_extension.coders import ExampleToNumpyDict
  from tfx_bsl.cc.tfx_bsl_extension.coders import RecordBatchToExamples
  from tfx_bsl.cc.tfx_bsl_extension.coders import FeatureNameToColumnsMap
except ImportError:
  import sys
  sys.stderr.write("Error importing tfx_bsl_extension.coders. "
                   "Some tfx_bsl functionalities are not available")
# pylint: enable=g-import-not-at-top
# pytype: enable=import-error
# pylint: enable=unused-import


def _get_ragged_column_names(
    tensor_representation: schema_pb2.TensorRepresentation) -> List[str]:
  """Extracts source column names from a ragged tensor representation."""
  source_columns = (
      tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
          tensor_representation))
  result = []
  for column in source_columns:
    if len(column.steps()) != 1:
      raise NotImplementedError(
          "Support of RaggedFeatures with multiple steps in feature_path is "
          "not implemented, got {}".format(len(column.steps())))
    result.append(column.steps()[0])
  return result


class RecordBatchToExamplesEncoder:
  """Encodes `pa.RecordBatch` as a list of `tf.Example`s.

  A generalized stateful version of `RecordBatchToExamples` that infers names
  of features representing nested list components from TFMD schema. The names
  are inferred once during construction and can be used multiple times in
  `encode`.
  Only RaggedTensors are currently represented as nested lists.
  """

  def __init__(self, schema: schema_pb2.Schema):
    self._ragged_features = FeatureNameToColumnsMap()
    # Iterate over tensor representations in all groups.
    for group in schema.tensor_representation_group.values():
      tensor_representation_map = group.tensor_representation
      for name, representation in tensor_representation_map.items():
        if representation.WhichOneof("kind") == "ragged_tensor":
          self._ragged_features[name] = _get_ragged_column_names(representation)

  def __getstate__(self) -> Dict[str, List[str]]:
    return dict(self._ragged_features.items())

  def __setstate__(self, state: Dict[str, List[str]]):
    self._ragged_features = FeatureNameToColumnsMap()
    for name, columns in state.items():
      self._ragged_features[name] = columns

  def encode(self, record_batch: pa.RecordBatch) -> List[bytes]:
    return RecordBatchToExamples(record_batch, self._ragged_features)
