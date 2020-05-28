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
"""TFXIO implementation for csv."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc

import apache_beam as beam
import pyarrow as pa
import six
from tfx_bsl.coders import csv_decoder
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio
from typing import List, Optional, Text

from tensorflow_metadata.proto.v0 import schema_pb2


@six.add_metaclass(abc.ABCMeta)
class _CsvTFXIOBase(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO implementations for CSV."""

  def __init__(self,
               physical_format: Text,
               column_names: List[Text],
               delimiter: Optional[Text] = ",",
               skip_blank_lines: Optional[bool] = True,
               multivalent_columns: Optional[Text] = None,
               secondary_delimiter: Optional[Text] = None,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None,
               telemetry_descriptors: Optional[List[Text]] = None):
    super(_CsvTFXIOBase, self).__init__(
        telemetry_descriptors=telemetry_descriptors,
        raw_record_column_name=raw_record_column_name,
        logical_format="csv",
        physical_format=physical_format)
    self._schema = schema
    self._column_names = column_names
    self._delimiter = delimiter
    self._skip_blank_lines = skip_blank_lines
    self._multivalent_columns = multivalent_columns
    self._secondary_delimiter = secondary_delimiter
    self._raw_record_column_name = raw_record_column_name
    if schema is not None:
      feature_names = [f.name for f in schema.feature]
      if set(feature_names) != set(column_names):
        raise ValueError(
            "Column names: {} does not match schema features: {}".format(
                column_names, feature_names))

  def SupportAttachingRawRecords(self) -> bool:
    return True

  @abc.abstractmethod
  def _CSVSource(self) -> beam.PTransform:
    """Returns a PTtransform that producese PCollection[bytets]."""

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return self._CSVSource()

  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:

    @beam.typehints.with_input_types(List[bytes])
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(raw_records_pcoll: beam.pvalue.PCollection):
      """Returns RecordBatch of csv lines."""

      # Decode raw csv lines to record batches.
      record_batches = (
          raw_records_pcoll
          | "CSVToRecordBatch" >> csv_decoder.CSVToRecordBatch(
              column_names=self._column_names,
              delimiter=self._delimiter,
              skip_blank_lines=self._skip_blank_lines,
              schema=self._schema,
              desired_batch_size=batch_size,
              multivalent_columns=self._multivalent_columns,
              secondary_delimiter=self._secondary_delimiter,
              produce_large_types=self._can_produce_large_types,
              raw_record_column_name=self._raw_record_column_name))

      return record_batches

    return beam.ptransform_fn(_PTransformFn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    if not self._schema:
      raise ValueError("TFMD schema not provided. Unable to derive an "
                       "Arrow schema")
    return csv_decoder.GetArrowSchema(
        self._column_names,
        self._schema,
        large_types=self._can_produce_large_types)

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    result = (
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            self._schema))
    if result is None:
      result = (
          tensor_representation_util.InferTensorRepresentationsFromSchema(
              self._schema))
    return result

  def _ProjectTfmdSchemaTensorRepresentation(
      self, tensor_names: List[Text]) -> schema_pb2.Schema:
    """Creates the tensor representation for choosen tensor_names."""
    tensor_representations = self.TensorRepresentations()
    tensor_names = set(tensor_names)

    # The columns in the schema will remain the same, because the csv decoder
    # will need to decode all columns no matter what.
    result = schema_pb2.Schema()
    result.CopyFrom(self._schema)

    # The tensor representation will only contain the projected columns, so the
    # output tensors will only be the projected columns.
    tensor_representation_util.SetTensorRepresentationsInSchema(
        result,
        {k: v for k, v in tensor_representations.items() if k in tensor_names})

    return result


class CsvTFXIO(_CsvTFXIOBase):
  """TFXIO implementation for CSV."""

  def __init__(self,
               file_pattern: Text,
               column_names: List[Text],
               validate: Optional[bool] = True,
               delimiter: Optional[Text] = ",",
               skip_blank_lines: Optional[bool] = True,
               multivalent_columns: Optional[Text] = None,
               secondary_delimiter: Optional[Text] = None,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None,
               telemetry_descriptors: Optional[List[Text]] = None):
    """Initializes a CSV TFXIO.

    Args:
      file_pattern: A file glob pattern to read TFRecords from.
      column_names: List of csv column names. Order must match the order in the
        CSV file.
      validate: Boolean flag to verify that the files exist during the pipeline
        creation time.
      delimiter: A one-character string used to separate fields.
      skip_blank_lines: A boolean to indicate whether to skip over blank lines
        rather than interpreting them as missing values.
      multivalent_columns: Name of column that can contain multiple values. If
        secondary_delimiter is provided, this must also be provided.
      secondary_delimiter: Delimiter used for parsing multivalent columns. If
        multivalent_columns is provided, this must also be provided.
      schema: An optional TFMD Schema describing the dataset. If schema is
        provided, it will determine the data type of the csv columns. Otherwise,
        the each column's data type will be inferred by the csv decoder. The
        schema should contain exactly the same features as column_names.
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains raw csv rows.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
    """
    super(CsvTFXIO, self).__init__(
        column_names=column_names,
        delimiter=delimiter,
        skip_blank_lines=skip_blank_lines,
        multivalent_columns=multivalent_columns,
        secondary_delimiter=secondary_delimiter,
        schema=schema,
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors,
        physical_format="text")
    self._file_pattern = file_pattern
    self._validate = validate

  def _CSVSource(self) -> beam.PTransform:
    """Returns a PTtransform that producese PCollection[bytes]."""
    return beam.io.ReadFromText(
        self._file_pattern,
        coder=beam.coders.BytesCoder(),
        validate=self._validate)

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    """Returns a projected TFXIO.

    Project in csv should not be used for optimization, since the decoder must
    read all columns no matter what.

    The Projected TFXIO will not project the record batches, arrow schema,
    nor the tfmd schema. Only the tensor representation, and the resulting
    tensors will be projected.

    Args:
      tensor_names: The columns to project.
    """
    projected_schema = self._ProjectTfmdSchemaTensorRepresentation(tensor_names)
    return CsvTFXIO(
        file_pattern=self._file_pattern,
        column_names=self._column_names,
        validate=self._validate,
        delimiter=self._delimiter,
        skip_blank_lines=self._skip_blank_lines,
        multivalent_columns=self._multivalent_columns,
        secondary_delimiter=self._secondary_delimiter,
        schema=projected_schema,
        raw_record_column_name=self._raw_record_column_name)

  def TensorFlowDataset(self):
    raise NotImplementedError
