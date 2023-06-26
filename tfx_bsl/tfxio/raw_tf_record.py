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
"""TFXIO implementations for raw TF Record."""

from typing import List, Optional, Text, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tfx_bsl.coders import batch_util
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import dataset_util
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2


class _RawRecordTFXIO(record_based_tfxio.RecordBasedTFXIO):
  """Base class for raw record TFXIO implementations.

  A raw record TFXIO decodes a record based on-disk format into an
  RecordBatches of one column that contains the raw records. Its TensorAdapter
  converts one such RecordBatch into a dense string tensor that contains
  the raw records.

  `raw_record_column_name` determines the name of the raw record column and
  the tensor.
  """

  def __init__(self, raw_record_column_name: Text,
               telemetry_descriptors: List[Text],
               physical_format: Text):
    assert raw_record_column_name is not None
    super().__init__(
        raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors,
        logical_format="bytes",
        physical_format=physical_format)

  def SupportAttachingRawRecords(self) -> bool:
    return True

  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(raw_record_pcoll: beam.pvalue.PCollection):
      return (
          raw_record_pcoll
          | "Batch"
          >> batch_util.BatchRecords(batch_size, self._telemetry_descriptors)
          | "ToRecordBatch"
          >> beam.Map(_BatchedRecordsToArrow, self.raw_record_column_name)
      )

    return beam.ptransform_fn(_PTransformFn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    # The only column is the raw record column.
    return pa.schema([])

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return {
        self.raw_record_column_name
        or "": schema_pb2.TensorRepresentation(
            dense_tensor=schema_pb2.TensorRepresentation.DenseTensor(
                column_name=self.raw_record_column_name,
                shape=schema_pb2.FixedShape(),  # scalar
            )
        )
    }

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    assert (not tensor_names or
            (len(tensor_names) == 1
             and tensor_names[0] == self.raw_record_column_name)), (
                 str(tensor_names))
    return self


def _BatchedRecordsToArrow(records: List[bytes],
                           raw_record_column_name: Text) -> pa.RecordBatch:
  raw_record_column = record_based_tfxio.CreateRawRecordColumn(records)
  return pa.RecordBatch.from_arrays(
      [raw_record_column], [raw_record_column_name])


class RawBeamRecordTFXIO(_RawRecordTFXIO):
  """TFXIO for raw records in pcoll[bytes].

  This is a special TFXIO that does not actually do I/O -- it relies on the
  caller to prepare a PCollection of bytes.
  """

  def __init__(self,
               physical_format: Text,
               raw_record_column_name: Text,
               telemetry_descriptors: List[Text]):
    """Initializer.

    Args:
      physical_format: The physical format that describes where the input
        pcoll[bytes] comes from. Used for telemetry purposes. Examples: "text",
        "tfrecord".
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains serialized
        records.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
    """
    super().__init__(
        telemetry_descriptors=telemetry_descriptors,
        physical_format=physical_format,
        raw_record_column_name=raw_record_column_name)

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return (beam.ptransform_fn(lambda x: x)()
            .with_input_types(bytes)
            .with_output_types(bytes))

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    raise NotImplementedError


class RawTfRecordTFXIO(_RawRecordTFXIO):
  """Raw record TFXIO for TFRecord format."""

  def __init__(self, file_pattern: Union[Text, List[Text]],
               raw_record_column_name: Text,
               telemetry_descriptors: List[Text]):
    """Initializer.

    Args:
      file_pattern: One or a list of glob patterns. If a list, must not be
        empty.
      raw_record_column_name: Name of the raw record column.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
    """
    super().__init__(
        telemetry_descriptors=telemetry_descriptors,
        physical_format="tfrecords_gzip",
        raw_record_column_name=raw_record_column_name)
    if not isinstance(file_pattern, list):
      file_pattern = [file_pattern]
    assert file_pattern, "Must provide at least one file pattern."
    self._file_pattern = file_pattern

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return record_based_tfxio.ReadTfRecord(self._file_pattern)

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:

    return (dataset_util.make_tf_record_dataset(
        file_pattern=self._file_pattern,
        batch_size=options.batch_size,
        drop_final_batch=options.drop_final_batch,
        num_epochs=options.num_epochs,
        shuffle=options.shuffle,
        shuffle_buffer_size=options.shuffle_buffer_size,
        shuffle_seed=options.shuffle_seed,
        sloppy_ordering=options.sloppy_ordering)
            .map(lambda records: {self._raw_record_column_name: records})
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
