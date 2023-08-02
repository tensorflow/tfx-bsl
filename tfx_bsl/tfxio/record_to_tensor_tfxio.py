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

import copy
import datetime
from typing import Any, Callable, Dict, List, Iterator, Optional, Text, Union

import apache_beam as beam
from apache_beam.utils import shared
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import tf_graph_record_decoder
from tfx_bsl.telemetry import util as telemetry_util
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import dataset_util
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_to_arrow
from tfx_bsl.tfxio import tfxio


class RecordToTensorTFXIO(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO implementations that uses TFGraphRecordDecoder."""

  def __init__(self,
               saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               physical_format: Text,
               use_singleton_decoder: bool,
               raw_record_column_name: Optional[Text]):
    super().__init__(
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
    self._use_singleton_decoder = use_singleton_decoder

    self._record_index_column_name = None
    record_index_tensor_name = decoder.record_index_tensor_name
    if record_index_tensor_name is not None:
      record_index_tensor_rep = self._tensor_representations[
          record_index_tensor_name]
      if record_index_tensor_rep.HasField("ragged_tensor"):
        assert len(record_index_tensor_rep.ragged_tensor.feature_path.step) == 1
        self._record_index_column_name = (
            record_index_tensor_rep.ragged_tensor.feature_path.step[0])
      elif record_index_tensor_rep.HasField("varlen_sparse_tensor"):
        self._record_index_column_name = (
            record_index_tensor_rep.varlen_sparse_tensor.column_name)
      else:
        raise ValueError("The record index tensor must be a RaggedTensor or a "
                         "VarLenSparseTensor, but got: {}"
                         .format(record_index_tensor_rep))

    if raw_record_column_name in self._arrow_schema_no_raw_record_column.names:
      raise ValueError("raw record column name: {} collided with an existing "
                       "column.".format(raw_record_column_name))

  def SupportAttachingRawRecords(self) -> bool:
    return True

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return self._tensor_representations

  def DecodeFunction(self) -> Callable[[tf.Tensor], Dict[Text, Any]]:
    """Returns the decode function provided by the decoder.

    Returns:
      A TF function that takes a 1-D string tensor and returns a dict from
      strings to (composite) tensors.
    """
    decoder = tf_graph_record_decoder.load_decoder(self._saved_decoder_path)
    return decoder.decode_record

  def _RawRecordToRecordBatchInternal(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self, batch_size: Optional[int]) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(raw_records_pcoll: beam.pvalue.PCollection):
      return (
          raw_records_pcoll
          | "BatchElements"
          >> batch_util.BatchRecords(batch_size, self._telemetry_descriptors)
          | "Decode"
          >> beam.ParDo(
              _RecordsToRecordBatch(
                  self._saved_decoder_path,
                  self.telemetry_descriptors,
                  shared.Shared() if self._use_singleton_decoder else None,
                  self.raw_record_column_name,
                  self._record_index_column_name,
              )
          )
      )

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

  def _ApplyDecoderToDataset(
      self, dataset: tf.data.Dataset) -> tf.data.Dataset:
    decoder = tf_graph_record_decoder.load_decoder(self._saved_decoder_path)

    def _ParseFn(record):
      tensors_dict = decoder.decode_record(record)
      return {
          k: v
          for k, v in tensors_dict.items()
          if k in self.TensorRepresentations()
      }
    return dataset.map(_ParseFn)


class BeamRecordToTensorTFXIO(RecordToTensorTFXIO):
  """TFXIO implementation that decodes records in pcoll[bytes] with TF Graph."""

  def __init__(self,
               saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               physical_format: Text,
               raw_record_column_name: Optional[Text],
               experimental_use_singleton_decoder: bool = False):
    """Initializer.

    Args:
      saved_decoder_path: The path to the saved TfGraphRecordDecoder to be
        used for decoding the records. Note that this path must be accessible
        by beam workers.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
      physical_format: A string that describes the physical format of the data.
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains serialized
        records.
      experimental_use_singleton_decoder: Experimental flag. May go away without
        notice. DO NOT SET.
    """
    super().__init__(
        saved_decoder_path=saved_decoder_path,
        telemetry_descriptors=telemetry_descriptors,
        physical_format=physical_format,
        use_singleton_decoder=experimental_use_singleton_decoder,
        raw_record_column_name=raw_record_column_name)

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return (beam.ptransform_fn(lambda x: x)()
            .with_input_types(bytes)
            .with_output_types(bytes))

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    raise NotImplementedError


class TFRecordToTensorTFXIO(RecordToTensorTFXIO):
  """Uses a TfGraphRecordDecoder to decode records on TFRecord files.

  This TFXIO assumes the data records are stored in TFRecord and takes a user
  provided TF-graph-based decoder (see tfx_bsl.coders.tf_graph_record_decoder)
  that decodes the records to TF (composite) tensors. The RecordBatches yielded
  by this TFXIO is converted from those tensors, and it's guaranteed that the
  TensorAdapter created by this TFXIO will be able to turn those RecordBatches
  to tensors identical to the TF-graph-based decoder's output.
  """

  def __init__(self,
               file_pattern: Union[List[Text], Text],
               saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               raw_record_column_name: Optional[Text] = None):
    """Initializer.

    Args:
      file_pattern: One or a list of glob patterns. If a list, must not be
        empty.
      saved_decoder_path: The path to the saved TfGraphRecordDecoder to be
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
    super().__init__(
        saved_decoder_path=saved_decoder_path,
        telemetry_descriptors=telemetry_descriptors,
        use_singleton_decoder=False,
        physical_format="tfrecords_gzip",
        raw_record_column_name=raw_record_column_name)
    if not isinstance(file_pattern, list):
      file_pattern = [file_pattern]
    assert file_pattern, "Must provide at least one file pattern."
    self._file_pattern = file_pattern

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return record_based_tfxio.ReadTfRecord(self._file_pattern)

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    """Creates a TFRecordDataset that yields Tensors.

    The records are parsed by the decoder to create Tensors. This implementation
    is based on tf.data.experimental.ops.make_tf_record_dataset().

    See base class (tfxio.TFXIO) for more details.

    Args:
      options: an options object for the tf.data.Dataset. See
        `dataset_options.TensorFlowDatasetOptions` for more details.
        options.batch_size is the batch size of the input records, but if the
        input record and the output batched tensors by the decoder are not
        batch-aligned (i.e. 1 input record results in 1 "row" in the output
        tensors), then the output may not be of the given batch size. Use
        dataset.unbatch().batch(desired_batch_size) to force the output batch
        size.

    Returns:
      A dataset of `dict` elements, (or a tuple of `dict` elements and label).
      Each `dict` maps feature keys to `Tensor`, `SparseTensor`, or
      `RaggedTensor` objects.

    Raises:
      ValueError: if label_key in the dataset option is not in the arrow schema.
    """

    dataset = dataset_util.make_tf_record_dataset(
        file_pattern=self._file_pattern,
        batch_size=options.batch_size,
        drop_final_batch=options.drop_final_batch,
        num_epochs=options.num_epochs,
        shuffle=options.shuffle,
        shuffle_buffer_size=options.shuffle_buffer_size,
        shuffle_seed=options.shuffle_seed,
        sloppy_ordering=options.sloppy_ordering)
    dataset = self._ApplyDecoderToDataset(dataset)

    label_key = options.label_key
    if label_key is not None:
      dataset = self._PopLabelFeatureFromDataset(dataset, label_key)

    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


class _DecodeFnWrapper(object):
  """A wrapper over a saved decoder.

  Thread-safe (all the fields should be considered read-only).
  """

  __slots__ = ["saved_decoder_path", "output_type_specs", "decode_fn",
               # required in order to create weakrefs to a _DecodeFnWrapper.
               "__weakref__"]

  def __init__(self, saved_decoder_path: Text):
    self.saved_decoder_path = saved_decoder_path
    _MaybeRegisterStruct2TensorOps()
    decoder = tf_graph_record_decoder.load_decoder(saved_decoder_path)
    self.output_type_specs = decoder.output_type_specs()
    # Store the concrete function to avoid tracing upon calling.
    # TF guarantees its thread-safey.
    self.decode_fn = decoder.decode_record
    # Call the concrete function once to force optimization of the graph, as
    # we want that to be attributed as fixed setup cost.
    try:
      _ = self.decode_fn(tf.constant([], shape=[0], dtype=tf.string))
    except Exception:  # pylint:disable=broad-except
      pass


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.RecordBatch)
class _RecordsToRecordBatch(beam.DoFn):
  """DoFn to convert raw records to RecordBatches."""

  def __init__(self, saved_decoder_path: Text,
               telemetry_descriptors: List[Text],
               shared_decode_fn_handle: Optional[shared.Shared],
               raw_record_column_name: Optional[Text],
               record_index_column_name: Optional[Text]):
    super().__init__()
    self._saved_decoder_path = saved_decoder_path
    self._raw_record_column_name = raw_record_column_name
    self._record_index_column_name = record_index_column_name
    self._shared_decode_fn_handle = shared_decode_fn_handle

    self._tensors_to_record_batch_converter = None
    self._decode_fn = None
    self._decoder_load_seconds_distribution = beam.metrics.Metrics.distribution(
        telemetry_util.MakeTfxNamespace(telemetry_descriptors),
        "record_to_tensor_tfxio_decoder_load_seconds")
    self._decoder_load_seconds = None

  def setup(self):
    start = datetime.datetime.now()
    if self._shared_decode_fn_handle is not None:
      decode_fn_wrapper = self._shared_decode_fn_handle.acquire(
          lambda: _DecodeFnWrapper(self._saved_decoder_path))
      assert decode_fn_wrapper.saved_decoder_path == self._saved_decoder_path
    else:
      decode_fn_wrapper = _DecodeFnWrapper(self._saved_decoder_path)
    self._tensors_to_record_batch_converter = (
        tensor_to_arrow.TensorsToRecordBatchConverter(
            decode_fn_wrapper.output_type_specs))
    self._decode_fn = decode_fn_wrapper.decode_fn
    self._decoder_load_seconds = int(
        (datetime.datetime.now() - start).total_seconds())

  def finish_bundle(self):
    if self._decoder_load_seconds is not None:
      self._decoder_load_seconds_distribution.update(self._decoder_load_seconds)
      self._decoder_load_seconds = None

  def process(self, records: List[bytes]) -> Iterator[pa.RecordBatch]:
    assert self._tensors_to_record_batch_converter is not None  # By setup().
    decoded = self._tensors_to_record_batch_converter.convert(
        self._decode_fn(tf.convert_to_tensor(records, dtype=tf.string)))
    if self._raw_record_column_name is None:
      yield decoded
    else:
      yield record_based_tfxio.AppendRawRecordColumn(
          decoded, self._raw_record_column_name, records,
          self._record_index_column_name)


# TODO(b/159982957): Replace this with a mechanism that registers any custom
# op.
def _MaybeRegisterStruct2TensorOps():
  try:
    import struct2tensor as _  # pylint: disable=g-import-not-at-top
  except (ImportError, tf.errors.NotFoundError):
    pass
