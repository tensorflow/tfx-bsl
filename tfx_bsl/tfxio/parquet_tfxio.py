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
import pickle
from typing import Any, Dict, List, Optional, Union

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import csv_decoder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import telemetry
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2

_PARQUET_FORMAT = "parquet"

_RecordDataType = Union[str, int, float, Dict[Any, Any], List[Any]]


def _RecordDictsToRecordBatch(dicts: List[Dict[str, _RecordDataType]],
                              schema: pa.Schema) -> pa.RecordBatch:
  record_data_arrays = []
  for name, array_type in zip(schema.names, schema.types):
    record_data_arrays.append(
        pa.array([record[name] for record in dicts], array_type))
  return pa.RecordBatch.from_arrays(record_data_arrays, schema)


class ParquetTFXIO(record_based_tfxio.RecordBasedTFXIO):
  """TFXIO implementation for Parquet."""

  def __init__(self,
               file_pattern: Union[str, List[str]],
               *,
               column_names: Optional[List[str]] = None,
               min_bundle_size: int = 0,
               schema: Optional[schema_pb2.Schema] = None,
               validate: bool = True,
               telemetry_descriptors: Optional[List[str]] = None):
    """Initializes a Parquet TFXIO.

    Args:
      file_pattern: One or a list of file glob patterns to read parquet files
        from.
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
    super().__init__(
        telemetry_descriptors=telemetry_descriptors,
        raw_record_column_name=None,
        logical_format=_PARQUET_FORMAT,
        physical_format=_PARQUET_FORMAT)
    self._file_pattern = (
        file_pattern if isinstance(file_pattern, list) else [file_pattern])
    self._column_names = column_names
    self._min_bundle_size = min_bundle_size
    self._validate = validate
    self._schema = schema

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:

    @beam.typehints.with_output_types(bytes)
    def _PTransformFn(
        pcoll_or_pipeline: Union[beam.PCollection, beam.Pipeline]
    ) -> beam.PCollection[bytes]:
      """Reads Parquet records and serializes to bytes."""
      source_pcolls = []
      for i, pattern in enumerate(self._file_pattern):
        source_pcolls.append(
            pcoll_or_pipeline
            | f"ReadFromParquet[{i}]" >> beam.io.ReadFromParquet(
                file_pattern=pattern,
                min_bundle_size=self._min_bundle_size,
                validate=self._validate,
                columns=self._column_names))
      return (source_pcolls | "FlattenPCollsFromPatterns" >> beam.Flatten()
              | "EncodeRawRecords" >> beam.Map(pickle.dumps))

    return beam.ptransform_fn(_PTransformFn)()

  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(
        raw_records_pcoll: beam.PCollection[bytes]
    ) -> beam.PCollection[pa.RecordBatch]:
      """Decodes raw records and converts them to RecordBatches."""
      return (
          raw_records_pcoll
          | "DecodeRawRecords" >> beam.Map(pickle.loads)
          | "Batch"
          >> batch_util.BatchRecords(batch_size, self._telemetry_descriptors)
          | "ToRecordBatch"
          >> beam.Map(_RecordDictsToRecordBatch, schema=self.ArrowSchema())
      )

    return beam.ptransform_fn(_PTransformFn)()

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:
    # We override the `RecordBasedTFXIO`s `BeamSource` that is a composition of
    # `_RawRecordBeamSourceInternal` and `_RawRecordToRecordBatchInternal` with
    # a more efficient implementation that uses batched read.
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(
        pcoll_or_pipeline: Union[beam.PCollection, beam.Pipeline]
    ) -> beam.PCollection[pa.RecordBatch]:
      """Reads Parquet tables and converts to RecordBatches."""
      source_pcolls = []
      for i, pattern in enumerate(self._file_pattern):
        source_pcolls.append(
            pcoll_or_pipeline
            | f"ReadFromParquetBatched[{i}]" >> beam.io.ReadFromParquetBatched(
                file_pattern=pattern,
                min_bundle_size=self._min_bundle_size,
                validate=self._validate,
                columns=self._column_names))
      return (
          source_pcolls
          | "FlattenPCollsFromPatterns" >> beam.Flatten()
          | "ToRecordBatch"
          >> beam.FlatMap(lambda table: table.to_batches(batch_size))
          | "CollectRecordBatchTelemetry"
          >> telemetry.ProfileRecordBatches(
              self._telemetry_descriptors, _PARQUET_FORMAT, _PARQUET_FORMAT
          )
      )

    return beam.ptransform_fn(_PTransformFn)()

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    raise NotImplementedError

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    # ParquetTFXIO does not support attaching raw record column and therefore
    # columns' schema does not contain the raw record column.
    if self._schema is None:
      return self._InferArrowSchema()

    # If the column names are not passed, we default to all column names in the
    # schema.
    columns = self._column_names or [f.name for f in self._schema.feature]

    return csv_decoder.GetArrowSchema(columns, self._schema)

  def _InferArrowSchema(self) -> pa.Schema:
    match_result = FileSystems.match(self._file_pattern)[0]
    files_metadata = match_result.metadata_list[0]
    with FileSystems.open(files_metadata.path) as f:
      return pq.read_schema(f)

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
        self._schema)

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
