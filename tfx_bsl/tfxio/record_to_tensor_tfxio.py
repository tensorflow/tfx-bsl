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
"""RecordToTensorTFXIO."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import copy
from typing import List, Iterator, Optional, Text

import apache_beam as beam
import pyarrow as pa
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import tf_graph_record_decoder
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_to_arrow
from tfx_bsl.tfxio import tfxio


class _RecordToTensorTFXIO(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO implementations that uses TFGraphRecordDecoder."""

  def __init__(self,
               saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               physical_format: Text,
               raw_record_column_name: Optional[Text]):

    super(_RecordToTensorTFXIO, self).__init__(
        telemetry_descriptors,
        logical_format="tensor",
        physical_format=physical_format,
        raw_record_column_name=raw_record_column_name)
    self._saved_decoder_path = saved_decoder_path
    decoder = tf_graph_record_decoder.load_decoder(saved_decoder_path)
    tensor_to_arrow_converter = tensor_to_arrow.TensorsToRecordBatchConverter(
        decoder.output_type_specs())
    self._arrow_schema_no_raw_record_column = (
        tensor_to_arrow_converter.arrow_schema())
    self._tensor_representations = (
        tensor_to_arrow_converter.tensor_representations())
    if raw_record_column_name in self._arrow_schema_no_raw_record_column.names:
      raise ValueError("raw record column name: {} collided with an existing "
                       "column.".format(raw_record_column_name))

  def SupportAttachingRawRecords(self) -> bool:
    return True

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return self._tensor_representations

  def _RawRecordToRecordBatchInternal(
      self, batch_size: Optional[int]) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(raw_records_pcoll: beam.pvalue.PCollection):
      return (
          raw_records_pcoll
          | "BatchElements" >> beam.BatchElements(
              **batch_util.GetBatchElementsKwargs(batch_size))
          | "Decode" >> beam.ParDo(_RecordsToRecordBatch(
              self._saved_decoder_path, self.raw_record_column_name,
              self._can_produce_large_types)))

    return beam.ptransform_fn(_PTransformFn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    return self._arrow_schema_no_raw_record_column

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    # We could do better by plumbing the information back to the decoder.
    self_copy = copy.copy(self)
    self_copy._tensor_representations = {  # pylint: disable=protected-access
        k: v
        for k, v in self._tensor_representations.items()
        if k in set(tensor_names)
    }
    return self_copy


class TFRecordToTensorTFXIO(_RecordToTensorTFXIO):
  """Uses a TfGraphRecordDecoder to decode records on TFRecord files.

  This TFXIO assumes the data records are stored in TFRecord and takes a user
  provided TF-graph-based decoder (see tfx_bsl.coders.tf_graph_record_decoder)
  that decodes the records to TF (composite) tensors. The RecordBatches yielded
  by this TFXIO is converted from those tensors, and it's guaranteed that the
  TensorAdapter created by this TFXIO will be able to turn those RecordBatches
  to tensors identical to the TF-graph-based decoder's output.
  """

  def __init__(self,
               file_pattern: Text,
               saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               raw_record_column_name: Optional[Text] = None):
    """Initializer.

    Args:
      file_pattern: A file glob pattern to read TFRecords from.
      saved_decoder_path: the path to the saved TfGraphRecordDecoder to be
        used for decoding the records. Note that this path must be accessible
        by beam workers.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains serialized
        records.
    """
    super(TFRecordToTensorTFXIO, self).__init__(
        saved_decoder_path,
        telemetry_descriptors,
        physical_format="tfrecords_gzip",
        raw_record_column_name=raw_record_column_name)
    self._file_pattern = file_pattern

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return beam.io.ReadFromTFRecord(self._file_pattern, validate=False)

  def TensorFlowDataset(self):
    # Implementation note: Project() might have been called, which means
    # the desired tensors could be a subset of the outputs of the
    # TF graph record decoder.
    raise NotImplementedError


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.RecordBatch)
class _RecordsToRecordBatch(beam.DoFn):
  """DoFn to convert raw records to RecordBatches."""

  def __init__(self, saved_decoder_path: Text,
               raw_record_column_name: Optional[Text],
               produce_large_raw_record_column):
    super(_RecordsToRecordBatch, self).__init__()
    self._saved_decoder_path = saved_decoder_path
    self._raw_record_column_name = raw_record_column_name
    self._produce_large_raw_record_column = produce_large_raw_record_column

    self._decoder = None
    self._tensors_to_record_batch_converter = None

  def setup(self):
    self._decoder = tf_graph_record_decoder.load_decoder(
        self._saved_decoder_path)
    self._tensors_to_record_batch_converter = (
        tensor_to_arrow.TensorsToRecordBatchConverter(
            self._decoder.output_type_specs()))

  def process(self, records: List[bytes]) -> Iterator[pa.RecordBatch]:
    decoded = self._tensors_to_record_batch_converter.convert(
        self._decoder.decode_record(records))
    if self._raw_record_column_name is None:
      yield decoded
    else:
      yield record_based_tfxio.AppendRawRecordColumn(
          decoded, self._raw_record_column_name, records,
          self._produce_large_raw_record_column)
