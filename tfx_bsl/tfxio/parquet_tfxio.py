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
"""TFXIO implementation for Parquet."""

import copy
from typing import Any, List, Optional, Union

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tfx_bsl.coders import csv_decoder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import telemetry
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2

_PARQUET_FORMAT = "parquet"


class ParquetTFXIO(tfxio.TFXIO):
  """TFXIO implementation for Parquet."""

  def __init__(self,
               file_pattern: str,
               *,
               column_names: Optional[List[str]] = None,
               min_bundle_size: int = 0,
               schema: Optional[schema_pb2.Schema] = None,
               validate: bool = True,
               telemetry_descriptors: Optional[List[str]] = None):
    """Initializes a Parquet TFXIO.

    Args:
      file_pattern: A file glob pattern to read parquet files from.
      column_names: List of column names to read from the parquet files.
      min_bundle_size: the minimum size in bytes, to be considered when
        splitting the parquet input into bundles. If not provided, all columns
        in the dataset will be read.
      schema: An optional TFMD Schema describing the dataset. If schema is
        provided, it will determine the data type of the parquet columns.
        Otherwise, the each column's data type will be inferred by the decoder.
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

  @property
  def telemetry_descriptors(self) -> Optional[List[str]]:
    return self._telemetry_descriptors

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:

    @beam.typehints.with_input_types(Union[beam.PCollection, beam.Pipeline])
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll_or_pipeline: Any):
      """Reads Parquet tables and converts to RecordBatches."""
      return (
          pcoll_or_pipeline
          | "ParquetBeamSource" >> beam.io.ReadFromParquetBatched(
              file_pattern=self._file_pattern,
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

    # If the column names are not passed, we default to all column names in the
    # schema.
    columns = self._column_names or [f.name for f in self._schema.feature]

    return csv_decoder.GetArrowSchema(columns, self._schema)

  def _InferArrowSchema(self):
    match_result = FileSystems.match([self._file_pattern])[0]
    files_metadata = match_result.metadata_list[0]
    with FileSystems.open(files_metadata.path) as f:
      return pq.read_schema(f)

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    result = (
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            self._schema))
    if result is None:
      result = (
          tensor_representation_util.InferTensorRepresentationsFromSchema(
              self._schema))
    return result

  def _ProjectImpl(self, tensor_names: List[str]) -> tfxio.TFXIO:
    """Returns a projected TFXIO.

    Projection is pushed down to the Parquet Beam source.

    The Projected TFXIO will project the record batches, arrow schema,
    and the TFMD schema.

    Args:
      tensor_names: The columns to project.
    """
    projected_schema = (
        tensor_representation_util.ProjectTensorRepresentationsInSchema(
            self._schema, tensor_names))
    result = copy.copy(self)

    result._column_names = [f.name for f in projected_schema.feature]  # pylint: disable=protected-access
    result._schema = projected_schema  # pylint: disable=protected-access
    return result
