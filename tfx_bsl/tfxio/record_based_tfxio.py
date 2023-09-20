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
"""Defines RecordBasedTFXIO interface.

Also common utilities used by its implementations.
"""

import abc
from typing import Any, List, Optional, Text, Union

import apache_beam as beam
import numpy as np
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import telemetry
from tfx_bsl.tfxio import tfxio


class RecordBasedTFXIO(tfxio.TFXIO):
  """Base class for all TFXIO implementations for record-based on-disk formats.

  `RecordBasedTFXIO` offers the following abstractions that are unique to
  record-based formats:

  `SupportAttachingRawRecords()`: indicates whether this implementation
    supports attaching the raw records as a `LargeList<LargeBinary>` column to
    the produced RecordBatches upon request. If a subclass implements this
    feature, then its `RawRecordToRecordBatch()` must consult
    `self.raw_record_column_name`, and make sure that the produced RecordBatches
    have the raw record column as the last column, with the given name (if not
    None); otherwise it's guaranteed that the raw record column is not
    requested (`self.raw_record_column_name` == None).

  `RawRecordBeamSource()`: returns a PTransform that produces PCollection[bytes]
    (of raw records).

  RawRecordToRecordBatch(): returns a PTransform that takes `PCollection[bytes]`
    (expected to be what's produced by `RawRecordBeamSource()`) and produces
    `PCollection[RecordBatch]`. It's guaranteed that `BeamSource()` is a
    composition of `RawRecordBeamSource()` and `RawRecordToRecordBatch()`.
    This interface is useful if one wants to access both the raw records as
    well as the RecordBatches, because beam does not do Common Sub-expression
    Eliminination, it's more desirable to be able to cache the output of
    `RawRecordBeamSource()` and feed it to `RawRecordToRecordBatch()` than
    calling `BeamSource()` separately as redundant disk reads can be avoided.
  """

  def __init__(self, telemetry_descriptors: Optional[List[Text]],
               logical_format: Text,
               physical_format: Text,
               raw_record_column_name: Optional[Text] = None):
    super().__init__()
    if not self.SupportAttachingRawRecords():
      assert raw_record_column_name is None, (
          "{} did not support attaching raw records, but requested.".format(
              type(self)))
    self._telemetry_descriptors = telemetry_descriptors
    self._logical_format = logical_format
    self._physical_format = physical_format
    self._raw_record_column_name = raw_record_column_name

  @property
  def raw_record_column_name(self) -> Optional[Text]:
    return self._raw_record_column_name

  @property
  def telemetry_descriptors(self) -> Optional[List[Text]]:
    return self._telemetry_descriptors

  def SupportAttachingRawRecords(self) -> bool:
    return False

  def RawRecordBeamSource(self) -> beam.PTransform:
    """Returns a PTransform that produces a PCollection[bytes].

    Used together with RawRecordToRecordBatch(), it allows getting both the
    PCollection of the raw records and the PCollection of the RecordBatch from
    the same source. For example:

    record_batch = pipeline | tfxio.BeamSource()
    raw_record = pipeline | tfxio.RawRecordBeamSource()

    would result in the files being read twice, while the following would only
    read once:

    raw_record = pipeline | tfxio.RawRecordBeamSource()
    record_batch = raw_record | tfxio.RawRecordToRecordBatch()
    """

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(bytes)
    def _PTransformFn(pcoll_or_pipeline: Any):
      return (pcoll_or_pipeline
              | "ReadRawRecords" >> self._RawRecordBeamSourceInternal()
              | "CollectRawRecordTelemetry" >> telemetry.ProfileRawRecords(
                  self._telemetry_descriptors, self._logical_format,
                  self._physical_format))

    return beam.ptransform_fn(_PTransformFn)()

  def RawRecordToRecordBatch(self,
                             batch_size: Optional[int] = None
                            ) -> beam.PTransform:
    """Returns a PTransform that converts raw records to Arrow RecordBatches.

    The input PCollection must be from self.RawRecordBeamSource() (also see
    the documentation for that method).

    Args:
      batch_size: if not None, the `pa.RecordBatch` produced will be of the
        specified size. Otherwise it's automatically tuned by Beam.
    """

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll: beam.pvalue.PCollection):
      return (pcoll
              | "RawRecordToRecordBatch" >>
              self._RawRecordToRecordBatchInternal(batch_size)
              | "CollectRecordBatchTelemetry" >>
              telemetry.ProfileRecordBatches(self._telemetry_descriptors,
                                             self._logical_format,
                                             self._physical_format))

    return beam.ptransform_fn(_PTransformFn)()

  @abc.abstractmethod
  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    """Returns a PTransform that produces a PCollection[bytes]."""

  @abc.abstractmethod
  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:
    """Returns a PTransform that converts raw records to Arrow RecordBatches."""
    pass

  @abc.abstractmethod
  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    """Returns the Arrow schema that does not contain the raw record column.

    Even if self.raw_record_column is not None.

    Returns:
      a pa.Schema.
    """
    pass

  def ArrowSchema(self) -> pa.Schema:
    schema = self._ArrowSchemaNoRawRecordColumn()
    if self._raw_record_column_name is not None:
      if schema.get_field_index(self._raw_record_column_name) != -1:
        raise ValueError(
            "Raw record column name {} collided with a column in the schema."
            .format(self._raw_record_column_name))
      schema = schema.append(
          pa.field(self._raw_record_column_name,
                   pa.large_list(pa.large_binary())))
    return schema

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:

    @beam.typehints.with_input_types(Any)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(pcoll_or_pipeline: Any):
      """Converts raw records to RecordBatches."""
      return (
          pcoll_or_pipeline
          | "RawRecordBeamSource" >> self.RawRecordBeamSource()
          | "RawRecordToRecordBatch" >> self.RawRecordToRecordBatch(batch_size))

    return beam.ptransform_fn(_PTransformFn)()

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def _PopLabelFeatureFromDataset(self, dataset: tf.data.Dataset,
                                  label_key: Text) -> tf.data.Dataset:
    """Create a dataset that isolates the label_key feature from all features.

    Args:
      dataset: A dataset of dicts, where the keys are feature names, and values
        are features.
      label_key: The name of the feature that is isolated.

    Returns:
      A dataset of tuple (label_key_feature, features_dict), where features_dict
      does not contain label_key.
    """
    if label_key not in self.TensorRepresentations():
      raise ValueError(
          "The `label_key` provided ({}) must be one of the following tensors"
          "names: {}.".format(label_key, self.TensorRepresentations().keys()))
    return dataset.map(lambda x: (x, x.pop(label_key)))

  def RawRecordTensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    """Returns a Dataset that contains nested Datasets of raw records.

    May not be implemented for some TFXIOs.

    This should be used when RawTfRecordTFXIO.TensorFlowDataset does not
    suffice. Namely, if there is some logical grouping of files which we need
    to perform operations on, without applying the operation to each individual
    group (i.e. shuffle).

    The returned Dataset object is a dataset of datasets, where each nested
    dataset is a dataset of serialized records. When shuffle=False (default),
    the nested datasets are deterministically ordered. Each nested dataset can
    represent multiple files. The files are merged into one dataset if the files
    have the same format. For example:

    ```
    file_patterns = ['file_1', 'file_2', 'dir_1/*']
    file_formats = ['recordio', 'recordio', 'sstable']
    tfxio = SomeTFXIO(file_patterns, file_formats)
    datasets = tfxio.RawRecordTensorFlowDataset(options)
    ```
    `datasets` would result in the following dataset: `[ds1, ds2]`. Where ds1
    iterates over records from 'file_1' and 'file_2', and ds2 iterates over
    records from files matched by 'dir_1/*'.

    Example usage:
    ```
    tfxio = SomeTFXIO(file_patterns, file_formats)
    ds = tfxio.RawRecordTensorFlowDataset(options=options)
    ds = ds.flat_map(lambda x: x)
    records = list(ds.as_numpy_iterator())
    # iterating over `records` yields records from the each file in
    # `file_patterns`. See `tf.data.Dataset.list_files` for more information
    # about the order of files when expanding globs.
    ```
    Note that we need a flat_map, because `RawRecordTensorFlowDataset` returns
    a dataset of datasets.

    When shuffle=True, then the datasets not deterministically ordered,
    but the contents of each nested dataset are deterministcally ordered.
    For example, we may potentially have [ds2, ds1, ds3], where the
    contents of ds1, ds2, and ds3 are all deterministcally ordered.

    Args:
      options: A TensorFlowDatasetOptions object. Not all options will apply.
    """
    raise NotImplementedError


def CreateRawRecordColumn(
    raw_records: Union[np.ndarray, List[bytes]]) -> pa.Array:
  """Returns an Array that satisfies the requirement of a raw record column."""
  return pa.LargeListArray.from_arrays(
      np.arange(0, len(raw_records) + 1, dtype=np.int64),
      pa.array(raw_records, type=pa.large_binary()))


def AppendRawRecordColumn(
    record_batch: pa.RecordBatch,
    column_name: Text,
    raw_records: List[bytes],
    record_index_column_name: Optional[Text] = None
) -> pa.RecordBatch:
  """Appends `raw_records` as a new column in `record_batch`.

  Args:
    record_batch: The RecordBatch to append to.
    column_name: The name of the column to be appended.
    raw_records: A list of bytes to be appended.
    record_index_column_name: If not specified, len(raw_records) must equal
      to record_batch.num_rows. Otherwise, `record_batch` must contain an
      list_like<integer> column to indicate which element in `raw_records`
      is the source of a row in `record_batch`. Specifically,
      record_index_column[i] == [j] means the i-th row came from the j-th
      element in `raw_records`. This column must not contain nulls, and all
      its elements must be single-element lists.

  Returns:
    A new RecordBatch whose last column is the raw record column, of given name.
  """
  schema = record_batch.schema
  if record_index_column_name is None:
    assert record_batch.num_rows == len(
        raw_records), "num_rows: {} vs len raw_records: {}".format(
            record_batch.num_rows, len(raw_records))
  else:
    record_index_column_index = schema.get_field_index(
        record_index_column_name)
    assert record_index_column_index != -1, (
        "Record index column {} did not exist."
        .format(record_index_column_name))
    record_index_column = record_batch.column(record_index_column_index)
    assert record_index_column.null_count == 0, (
        "Record index column must not contain nulls: {} nulls".format(
            record_index_column.null_count))
    column_type = record_index_column.type
    assert ((pa.types.is_list(column_type) or
             pa.types.is_large_list(column_type)) and
            pa.types.is_integer(column_type.value_type)), (
                "Record index column {} must be of type list_like<integer>, "
                "but got: {}".format(record_index_column_name, column_type))
    record_indices = np.asarray(record_index_column.flatten())
    assert len(record_indices) == len(record_batch), (
        "Record indices must be aligned with the record batch, but got "
        "different lengths: {} vs {}".format(
            len(record_indices), len(record_batch)))
    raw_records = np.asarray(raw_records, dtype=object)[record_indices]
  assert schema.get_field_index(column_name) == -1
  raw_record_column = CreateRawRecordColumn(raw_records)
  return pa.RecordBatch.from_arrays(
      list(record_batch.columns) + [raw_record_column],
      list(schema.names) + [column_name])


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(bytes)
def ReadTfRecord(pipeline: beam.Pipeline,
                 file_pattern: List[Text]) -> beam.pvalue.PCollection:
  """A Beam source that reads multiple TFRecord file patterns."""
  assert file_pattern, "Must provide at least one file pattern."
  # TODO(b/162261470): consider using beam.io.tfrecordio.ReadAllFromTFRecord
  # once the # concern over size estimation is addressed (also see
  # b/161935932#comment13).
  pcolls = []
  for i, f in enumerate(file_pattern):
    pcolls.append(pipeline
                  | "ReadFromTFRecord[%d]" % i >> beam.io.ReadFromTFRecord(
                      f, coder=beam.coders.BytesCoder()))

  return pcolls | "FlattenPCollsFromPatterns" >> beam.Flatten()


class OverridableRecordBasedTFXIO(RecordBasedTFXIO):
  """A TFXIO that takes user defined `BeamSource`.

  For a `RecordBasedTFXIO`, the `BeamSource` is comprised of two PTransforms --
  `raw_record_beam_source` and `raw_record_to_record_batch`.
  Both are required to be overridden by the user.

  raw_record_beam_source is a PTransform that produces raw records as a
  PCollection[bytes].
  raw_record_to_record_batch is a PTransform that converts raw records to
  PCollection[RecordBatch] taking batch size as initialization argument.
  """

  def __init__(
      self,
      telemetry_descriptors: Optional[List[Text]],
      logical_format: Text,
      physical_format: Text,
      raw_record_beam_source: beam.PTransform,
      raw_record_to_record_batch: beam.PTransform,
  ):
    super().__init__(telemetry_descriptors, logical_format, physical_format)
    self._raw_record_beam_source = raw_record_beam_source
    self._raw_record_to_record_batch = raw_record_to_record_batch

  def _RawRecordBeamSourceInternal(self):
    return self._raw_record_beam_source()

  def _RawRecordToRecordBatchInternal(self, batch_size: Optional[int] = None):
    return self._raw_record_to_record_batch(batch_size)

  def _ArrowSchemaNoRawRecordColumn(self):
    raise NotImplementedError

  def _ProjectImpl(self, tensor_names: List[Text]):
    raise NotImplementedError

  def TensorFlowDataset(self,
                        options: dataset_options.TensorFlowDatasetOptions):
    raise NotImplementedError

  def TensorRepresentations(self):
    raise NotImplementedError
