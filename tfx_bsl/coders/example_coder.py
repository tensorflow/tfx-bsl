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
from typing import Dict, List, Optional, Type, Tuple

import pyarrow as pa
from tfx_bsl.tfxio import tensor_representation_util

from tensorflow_metadata.proto.v0 import schema_pb2

# pylint: disable=unused-import
# pytype: disable=import-error
# pylint: disable=g-import-not-at-top
# See b/148667210 for why the ImportError is ignored.
try:
  from tfx_bsl.cc.tfx_bsl_extension.coders import ExamplesToRecordBatchDecoder as ExamplesToRecordBatchDecoderCpp
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
  """Encodes `pa.RecordBatch` as a list of serialized `tf.Example`s.

  Requires TFMD schema only if RecordBatches contains nested lists with
  depth > 2 that represent TensorFlow's RaggedFeatures.
  """

  __slots__ = ["_ragged_features"]

  def __init__(self, schema: Optional[schema_pb2.Schema] = None):
    self._ragged_features = FeatureNameToColumnsMap()
    if schema is not None:
      # Iterate over tensor representations in all groups.
      for group in schema.tensor_representation_group.values():
        tensor_representation_map = group.tensor_representation
        for name, representation in tensor_representation_map.items():
          if representation.WhichOneof("kind") == "ragged_tensor":
            self._ragged_features[name] = _get_ragged_column_names(
                representation)

  def __getstate__(self) -> Dict[str, List[str]]:
    return dict(self._ragged_features.items())

  def __setstate__(self, state: Dict[str, List[str]]):
    self._ragged_features = FeatureNameToColumnsMap()
    for name, columns in state.items():
      self._ragged_features[name] = columns

  def encode(self, record_batch: pa.RecordBatch) -> List[bytes]:
    return RecordBatchToExamples(record_batch, self._ragged_features)


class ExamplesToRecordBatchDecoder:
  """Decodes a list of serialized `tf.Example`s into `pa.RecordBatch`.

  If a schema is provided then the record batch will contain only the fields
  from the schema, in the same order as the Schema.  The data type of the
  schema to determine the field types, with INT, BYTES and FLOAT fields in the
  schema corresponding to the Arrow data types large_list[int64],
  large_list[large_binary] and large_list[float32].

  If a schema is not provided then the data type will be inferred, and chosen
  from list_type[int64], list_type[binary_type] and list_type[float32].  In the
  case where no data type can be inferred the arrow null type will be inferred.

  This class wraps pybind11 class `ExamplesToRecordBatchDecoder` to make the
  class and its member functions picklable.
  """

  __slots__ = ["_schema", "_coder"]

  def __init__(self, serialized_schema: Optional[bytes] = None):
    """Initializes ExamplesToRecordBatchDecoder.

    Args:
      serialized_schema: A serialized TFMD schema.
    """
    self._schema = serialized_schema
    self._coder = ExamplesToRecordBatchDecoderCpp(serialized_schema)

  def __reduce__(
      self
  ) -> Tuple[Type["ExamplesToRecordBatchDecoder"], Tuple[Optional[bytes]]]:
    return (self.__class__, (self._schema,))

  def DecodeBatch(self, examples: List[bytes]) -> pa.RecordBatch:  # pylint: disable=invalid-name
    return self._coder.DecodeBatch(examples)

  def ArrowSchema(self) -> pa.Schema:  # pylint: disable=invalid-name
    return self._coder.ArrowSchema()
