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
"""TFXIO implementation for tf.Example records."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc

import apache_beam as beam
from tfx_bsl.coders import example_coder
from tfx_bsl.pyarrow_tf import pyarrow as pa
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio
from typing import List, Optional, Text

from tensorflow_metadata.proto.v0 import schema_pb2


_FEATURE_TYPE_TO_ARROW_TYPE = {
    schema_pb2.FeatureType.BYTES: pa.list_(pa.binary()),
    schema_pb2.FeatureType.INT: pa.list_(pa.int64()),
    schema_pb2.FeatureType.FLOAT: pa.list_(pa.float32()),
}


class _TFExampleRecordBase(tfxio.TFXIO):
  """Base class for TFXIO implementations for record based tf.Examples."""

  def __init__(self,
               schema: Optional[schema_pb2.Schema] = None):
    self._schema = schema

  def BeamSource(
      self, batch_size: Optional[int] = None) -> beam.PTransform:
    @beam.ptransform_fn
    @beam.typehints.with_input_types(beam.Pipeline)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _ptransform_fn(pipeline: beam.pvalue.PCollection):
      # TODO(zhuo): collect telemetry from RecordBatches.
      return (pipeline
              | "ReadExamples" >> self._SerializedExamplesSource()
              | "Batch" >> beam.BatchElements(**_GetBatchElementsKwargs(
                  batch_size))
              | "Decode" >> beam.ParDo(
                  _DecodeBatchExamplesDoFn(self._schema)))

    return _ptransform_fn()  # pylint: disable=no-value-for-parameter

  def ArrowSchema(self) -> pa.Schema:
    if not self._schema:
      raise ValueError("TFMD schema not provided. Unable to derive an"
                       "Arrow schema")
    # TODO(zhuo): consider asking the schema from the decoder, to make it more
    # consistent.
    fields = []
    for f in self._schema.feature:
      arrow_type = _FEATURE_TYPE_TO_ARROW_TYPE.get(f.type)
      if not arrow_type:
        raise ValueError("Feature {} has unsupport type {}".format(
            f.name, f.type))
      fields.append(pa.field(f.name, arrow_type))

    return pa.schema(fields)

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    result = (
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            self._schema))
    if result is None:
      result = (
          tensor_representation_util.InferTensorRepresentationsFromSchema(
              self._schema))
    return result

  @abc.abstractmethod
  def _SerializedExamplesSource(self) -> beam.PTransform:
    """Returns a PTransform that produces PCollection[bytes]."""

  def _ProjectTfmdSchema(self, tensor_names: List[Text]) -> schema_pb2.Schema:
    """Projects self._schema by the given tensor names."""
    tensor_representations = self.TensorRepresentations()
    tensor_names = set(tensor_names)
    if not tensor_names.issubset(tensor_representations):
      raise ValueError(
          "Unable to project {} because they were not in the original "
          "TensorRepresentations.".format(tensor_names -
                                          tensor_representations))
    columns = set()
    for tensor_name in tensor_names:
      columns.update(
          tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
              tensor_representations[tensor_name]))
    result = schema_pb2.Schema()
    # Note: We only copy projected features into the new schema because the
    # coder, and ArrowSchema() only care about Schema.feature. If they start
    # depending on other Schema fields then those fields must also be projected.
    for f in self._schema.feature:
      if f.name in columns:
        result.feature.add().CopyFrom(f)

    tensor_representation_util.SetTensorRepresentationsInSchema(
        result,
        {k: v for k, v in tensor_representations.items() if k in tensor_names})

    return result


class TFExampleRecord(_TFExampleRecordBase):
  """TFXIO implementation for tf.Example on TFRecord."""

  def __init__(self,
               file_pattern: Text,
               validate: bool = True,
               schema: Optional[schema_pb2.Schema] = None):
    """Initializes a TFExampleRecord TFXIO.

    Args:
      file_pattern: A file glob pattern to read TFRecords from.
      validate: Boolean flag to verify that the files exist during the pipeline
          creation time.
      schema: A TFMD Schema describing the dataset.
    """
    super(TFExampleRecord, self).__init__(schema)
    self._file_pattern = file_pattern
    self._validate = validate

  def _SerializedExamplesSource(self) -> beam.PTransform:
    """Returns a PTransform that produces PCollection[bytes]."""
    return beam.io.ReadFromTFRecord(self._file_pattern, validate=self._validate)

  def Project(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    projected_schema = self._ProjectTfmdSchema(tensor_names)
    return TFExampleRecord(
        self._file_pattern, validate=self._validate, schema=projected_schema)

  def TensorFlowDataset(self):
    raise NotImplementedError


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.RecordBatch)
class _DecodeBatchExamplesDoFn(beam.DoFn):
  """Batches serialized protos bytes and decode them into an Arrow table."""

  __slots__ = ["_schema", "_decoder"]

  def __init__(self, schema: Optional[schema_pb2.Schema]):
    """Initializer."""
    self._schema = schema
    self._decoder = None

  def setup(self):
    if self._schema:
      self._decoder = example_coder.ExamplesToRecordBatchDecoder(
          self._schema.SerializeToString())
    else:
      self._decoder = example_coder.ExamplesToRecordBatchDecoder()

  def process(self, examples: List[bytes]):
    yield self._decoder.DecodeBatch(examples)


def _GetBatchElementsKwargs(batch_size: Optional[int]):  # pylint: disable=invalid-name
  if batch_size is None:
    return {}
  return {
      "min_batch_size": batch_size,
      "max_batch_size": batch_size,
  }
