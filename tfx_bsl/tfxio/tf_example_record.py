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

import abc
from typing import Iterator, List, Optional, Text, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.arrow import path
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import dataset_util
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2


class _TFExampleRecordBase(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO implementations for record based tf.Examples."""

  def __init__(self,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None,
               telemetry_descriptors: Optional[List[Text]] = None,
               physical_format: Optional[Text] = None,
               schema_for_decoding: Optional[schema_pb2.Schema] = None):
    # TODO(b/154648951): make telemetry_descriptors and physical_format required
    # arguments, when TFT's compatibility TFXIO starts setting them.
    if physical_format is None:
      physical_format = "unknown"
    super().__init__(
        telemetry_descriptors=telemetry_descriptors,
        raw_record_column_name=raw_record_column_name,
        logical_format="tf_example",
        physical_format=physical_format)
    self._schema = schema
    if schema_for_decoding is not None:
      assert schema is not None
    self._schema_for_decoding = schema_for_decoding

  def SupportAttachingRawRecords(self) -> bool:
    return True

  @abc.abstractmethod
  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    """Returns a PTransform that produces PCollection[bytes]."""

  def _RawRecordToRecordBatchInternal(self,
                                      batch_size: Optional[int] = None
                                     ) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(pa.RecordBatch)
    def _PTransformFn(raw_records_pcoll: beam.pvalue.PCollection):
      return (raw_records_pcoll
              | "Batch" >> beam.BatchElements(
                  **batch_util.GetBatchElementsKwargs(batch_size))
              | "Decode" >> beam.ParDo(
                  _DecodeBatchExamplesDoFn(self._GetSchemaForDecoding(),
                                           self.raw_record_column_name)))

    return beam.ptransform_fn(_PTransformFn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    schema = self._GetSchemaForDecoding()
    if schema is None:
      raise ValueError("TFMD schema not provided. Unable to derive an"
                       "Arrow schema")
    return example_coder.ExamplesToRecordBatchDecoder(
        schema.SerializeToString()).ArrowSchema()

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    result = (
        tensor_representation_util.GetTensorRepresentationsFromSchema(
            self._schema))
    if result is None:
      result = (
          tensor_representation_util.InferTensorRepresentationsFromSchema(
              self._schema))
    return result

  def _ProjectTfmdSchema(self, tensor_names: List[Text]) -> schema_pb2.Schema:
    """Projects self._schema by the given tensor names."""
    tensor_representations = self.TensorRepresentations()
    tensor_names = set(tensor_names)
    if not tensor_names.issubset(tensor_representations):
      raise ValueError(
          "Unable to project {} because they were not in the original "
          "TensorRepresentations.".format(tensor_names -
                                          tensor_representations))
    paths = set()
    for tensor_name in tensor_names:
      paths.update(
          tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
              tensor_representations[tensor_name]))
    result = schema_pb2.Schema()
    # Note: We only copy projected features into the new schema because the
    # coder, and ArrowSchema() only care about Schema.feature. If they start
    # depending on other Schema fields then those fields must also be projected.
    for f in self._schema.feature:
      if path.ColumnPath(f.name) in paths:
        result.feature.add().CopyFrom(f)

    tensor_representation_util.SetTensorRepresentationsInSchema(
        result,
        {k: v for k, v in tensor_representations.items() if k in tensor_names})

    return result

  def _GetSchemaForDecoding(self) -> schema_pb2.Schema:
    return (self._schema
            if self._schema_for_decoding is None else self._schema_for_decoding)


class TFExampleBeamRecord(_TFExampleRecordBase):
  """TFXIO implementation for serialized tf.Examples in pcoll[bytes].

  This is a special TFXIO that does not actually do I/O -- it relies on the
  caller to prepare a PCollection of bytes (serialized tf.Examples).
  """

  def __init__(self,
               physical_format: Text,
               telemetry_descriptors: Optional[List[Text]] = None,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None):
    """Initializer.

    Args:
      physical_format: The physical format that describes where the input
        pcoll[bytes] comes from. Used for telemetry purposes. Examples: "text",
        "tfrecord".
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
      schema: A TFMD Schema describing the dataset.
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains serialized
        records.
    """
    super().__init__(
        schema=schema, raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors,
        physical_format=physical_format)

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return (beam.ptransform_fn(lambda x: x)()
            .with_input_types(bytes)
            .with_output_types(bytes))

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    projected_schema = self._ProjectTfmdSchema(tensor_names)
    return TFExampleBeamRecord(self._physical_format,
                               self.telemetry_descriptors, projected_schema,
                               self.raw_record_column_name)

  def TensorFlowDataset(self,
                        options: dataset_options.TensorFlowDatasetOptions):
    raise NotImplementedError(
        "TFExampleBeamRecord is unable to provide a TensorFlowDataset "
        "because it does not do I/O")


class TFExampleRecord(_TFExampleRecordBase):
  """TFXIO implementation for tf.Example on TFRecord."""

  def __init__(self,
               file_pattern: Union[List[Text], Text],
               validate: bool = True,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None,
               telemetry_descriptors: Optional[List[Text]] = None):
    """Initializes a TFExampleRecord TFXIO.

    Args:
      file_pattern: A file glob pattern to read TFRecords from.
      validate: Not used. do not set. (not used since post 0.22.1).
      schema: A TFMD Schema describing the dataset.
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
        schema=schema, raw_record_column_name=raw_record_column_name,
        telemetry_descriptors=telemetry_descriptors,
        physical_format="tfrecords_gzip")
    del validate
    if not isinstance(file_pattern, list):
      file_pattern = [file_pattern]
    assert file_pattern, "Must provide at least one file pattern."
    self._file_pattern = file_pattern

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:
    return record_based_tfxio.ReadTfRecord(self._file_pattern)

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    projected_schema = self._ProjectTfmdSchema(tensor_names)
    return TFExampleRecord(
        file_pattern=self._file_pattern,
        schema=projected_schema,
        raw_record_column_name=self.raw_record_column_name,
        telemetry_descriptors=self.telemetry_descriptors)

  def RecordBatches(
      self, options: dataset_options.RecordBatchesOptions
  ) -> Iterator[pa.RecordBatch]:
    dataset = dataset_util.make_tf_record_dataset(
        self._file_pattern, options.batch_size, options.drop_final_batch,
        options.num_epochs, options.shuffle, options.shuffle_buffer_size,
        options.shuffle_seed)

    decoder = example_coder.ExamplesToRecordBatchDecoder(
        self._schema.SerializeToString())
    for examples in dataset.as_numpy_iterator():
      decoded = decoder.DecodeBatch(examples)
      if self._raw_record_column_name is None:
        yield decoded
      else:
        yield record_based_tfxio.AppendRawRecordColumn(
            decoded, self._raw_record_column_name, examples.tolist())

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    """Creates a TFRecordDataset that yields Tensors.

    The serialized tf.Examples are parsed by `tf.io.parse_example` to create
    Tensors.

    See base class (tfxio.TFXIO) for more details.

    Args:
      options: an options object for the tf.data.Dataset. See
        `dataset_options.TensorFlowDatasetOptions` for more details.

    Returns:
      A dataset of `dict` elements, (or a tuple of `dict` elements and label).
      Each `dict` maps feature keys to `Tensor`, `SparseTensor`, or
      `RaggedTensor` objects.

    Raises:
      ValueError: if there is something wrong with the tensor_representation.
    """
    feature_name_to_type = {f.name: f.type for f in self._schema.feature}

    # Creates parsing config for each feature.
    features = {}
    tensor_representations = self.TensorRepresentations()
    for feature_name, tensor_representation in tensor_representations.items():
      feature_type = feature_name_to_type[feature_name]
      features[
          feature_name] = tensor_representation_util.CreateTfExampleParserConfig(
              tensor_representation, feature_type)

    file_pattern = tf.convert_to_tensor(self._file_pattern)
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern,
        features=features,
        batch_size=options.batch_size,
        reader_args=[dataset_util.detect_compression_type(file_pattern)],
        num_epochs=options.num_epochs,
        shuffle=options.shuffle,
        shuffle_buffer_size=options.shuffle_buffer_size,
        shuffle_seed=options.shuffle_seed,
        drop_final_batch=options.drop_final_batch,
        label_key=options.label_key)


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.RecordBatch)
class _DecodeBatchExamplesDoFn(beam.DoFn):
  """Batches serialized protos bytes and decode them into an Arrow table."""

  def __init__(self, schema: Optional[schema_pb2.Schema],
               raw_record_column_name: Optional[Text]):
    """Initializer."""
    self._serialized_schema = None
    if schema is not None:
      # Serialize to avoid storing TFMD protos. See b/167128119 for the reason.
      self._serialized_schema = schema.SerializeToString()
    self._raw_record_column_name = raw_record_column_name
    self._decoder = None

  def setup(self):
    if self._serialized_schema:
      self._decoder = example_coder.ExamplesToRecordBatchDecoder(
          self._serialized_schema)
    else:
      self._decoder = example_coder.ExamplesToRecordBatchDecoder()

  def process(self, examples: List[bytes]):
    decoded = self._decoder.DecodeBatch(examples)
    if self._raw_record_column_name is None:
      yield decoded
    else:
      yield record_based_tfxio.AppendRawRecordColumn(
          decoded, self._raw_record_column_name, examples)
