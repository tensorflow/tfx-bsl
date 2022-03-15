# Copyright 2020 Google LLC
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
"""TFXIO implementation for Parquet."""

import copy
from typing import Optional, List, Text, Any

import apache_beam as beam
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from apache_beam.io.filesystems import FileSystems
from tensorflow_metadata.proto.v0 import schema_pb2

from tfx_bsl.coders import csv_decoder

from tfx_bsl.tfxio import dataset_options, tensor_adapter, tensor_representation_util, telemetry
from tfx_bsl.tfxio.tfxio import TFXIO

_PARQUET_FORMAT = "parquet"


class ParquetTFXIO(TFXIO):
  """TFXIO implementation for Parquet."""

  def __init__(self,
               file_pattern: Text,
               column_names: List[Text],
               *,
               min_bundle_size: int = 0,
               schema: Optional[schema_pb2.Schema] = None,
               validate: Optional[bool] = True,
               telemetry_descriptors: Optional[List[Text]] = None):
    """Initializes a Parquet TFXIO.

    Args:
      file_pattern: A file glob pattern to read parquet files from.
      column_names: List of column names to read from the parquet files.
      min_bundle_size: the minimum size in bytes, to be considered when
        splitting the parquet input into bundles.
      schema: An optional TFMD Schema describing the dataset. If schema is
        provided, it will determine the data type of the parquet columns. Otherwise,
        the each column's data type will be inferred by the decoder.
      validate: Boolean flag to verify that the files exist during the pipeline
        creation time.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
    """
    self._file_pattern = file_pattern
    self._column_names = column_names
    self._min_bundle_size = min_bundle_size
    self._validate = validate
    self._schema = schema
    self._telemetry_descriptors = telemetry_descriptors

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll_or_pipeline: Any):
      """Reads Parquet tables and converts to RecordBatches."""
      return (
          pcoll_or_pipeline | "ParquetBeamSource" >>
          beam.io.ReadFromParquetBatched(file_pattern=self._file_pattern,
                                         min_bundle_size=self._min_bundle_size,
                                         validate=self._validate,
                                         columns=self._column_names) |
          "ToRecordBatch" >> beam.FlatMap(self._TableToRecordBatch, batch_size)
          | "CollectRecordBatchTelemetry" >> telemetry.ProfileRecordBatches(
              self._telemetry_descriptors, _PARQUET_FORMAT, _PARQUET_FORMAT))

    return beam.ptransform_fn(_PTransformFn)()

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    raise NotImplementedError

  def _TableToRecordBatch(
      self,
      table: pa.Table,
      batch_size: Optional[int] = None) -> List[pa.RecordBatch]:
    return table.to_batches(max_chunksize=batch_size)

  def ArrowSchema(self) -> pa.Schema:
    if self._schema is None:
      return self._InferArrowSchema()
    return csv_decoder.GetArrowSchema(self._column_names, self._schema)

  def _InferArrowSchema(self):
    match_result = FileSystems.match([self._file_pattern])[0]
    files_metadata = match_result.metadata_list[0]
    with FileSystems.open(files_metadata.path) as f:
      return pq.read_schema(f)

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    result = (tensor_representation_util.GetTensorRepresentationsFromSchema(
        self._schema))
    if result is None:
      result = (tensor_representation_util.InferTensorRepresentationsFromSchema(
          self._schema))
    return result

  def _ProjectTfmdSchema(self, column_names: List[Text]) -> schema_pb2.Schema:
    """Creates a tensorflow Schema from the current one with only the given columns"""

    result = schema_pb2.Schema()
    result.CopyFrom(self._schema)

    for feature in self._schema.feature:
      if feature.name not in column_names:
        result.feature.remove(feature)

    return result

  def _ProjectImpl(self, tensor_names: List[Text]) -> "TFXIO":
    """Returns a projected TFXIO.

    Projection is pushed down to the Parquet Beam source.

    The Projected TFXIO will project the record batches, arrow schema,
    and the tfmd schema.

    Args:
      tensor_names: The columns to project.
    """
    projected_schema = self._ProjectTfmdSchema(tensor_names)
    result = copy.copy(self)
    result._column_names = tensor_names  # pylint: disable=protected-access
    result._schema = projected_schema  # pylint: disable=protected-access
    return result
