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
"""TFXIO implementation for tf.SequenceExample records."""

import abc
from typing import List, Optional, Text, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.arrow import path
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import sequence_example_coder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import dataset_util
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2


_SEQUENCE_COLUMN_NAME = "##SEQUENCE##"


class _TFSequenceExampleRecordBase(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO classes for record based tf.SequenceExamples."""

  def __init__(self,
               schema: Optional[schema_pb2.Schema],
               raw_record_column_name: Optional[Text],
               telemetry_descriptors: List[Text],
               physical_format: Text):
    super().__init__(
        telemetry_descriptors=telemetry_descriptors,
        raw_record_column_name=raw_record_column_name,
        logical_format="tf_sequence_example",
        physical_format=physical_format)
    self._schema = schema

  @property
  def schema(self) -> schema_pb2.Schema:
    if self._schema is None:
      raise ValueError("Schema is undefined.")
    return self._schema

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
      return (
          raw_records_pcoll
          | "Batch"
          >> batch_util.BatchRecords(batch_size, self._telemetry_descriptors)
          | "Decode"
          >> beam.ParDo(
              _DecodeBatchExamplesDoFn(
                  self._schema, self.raw_record_column_name
              )
          )
      )

    return beam.ptransform_fn(_PTransformFn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    return sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
        _SEQUENCE_COLUMN_NAME, self.schema.SerializeToString()
    ).ArrowSchema()

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
        self._schema)

  def _ParseRawRecordTensorFlowDataset(
      self,
      raw_record_dataset: tf.data.Dataset,
      label_key: Optional[str] = None) -> tf.data.Dataset:
    """Parses a dataset of serialized SequenceExamples."""

    context_features, sequence_features = (
        tensor_representation_util.CreateTfSequenceExampleParserConfig(
            self._schema))

    # Parse `SequenceExample` tensors to dictionaries of context and sequence
    # tensors and merge them.
    def _ParseAndMerge(serialized):
      context, sequence, _ = tf.io.parse_sequence_example(
          serialized, context_features, sequence_features)
      return {**context, **sequence}

    dataset = raw_record_dataset.map(
        _ParseAndMerge, num_parallel_calls=tf.data.AUTOTUNE)

    if label_key is not None:
      dataset = self._PopLabelFeatureFromDataset(dataset, label_key)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  def _ProjectTfmdSchema(self, tensor_names: List[Text]) -> schema_pb2.Schema:
    """Projects self._schema by the given tensor names."""
    tensor_representations = self.TensorRepresentations()
    tensor_names = set(tensor_names)
    if not tensor_names.issubset(tensor_representations):
      raise ValueError(
          "Unable to project {} because they were not in the original "
          "TensorRepresentations.".format(tensor_names -
                                          tensor_representations))
    used_paths = set()
    for tensor_name in tensor_names:
      used_paths.update(
          tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
              tensor_representations[tensor_name]))
    result = schema_pb2.Schema()
    # Note: We only copy projected features into the new schema because the
    # coder, and ArrowSchema() only care about Schema.feature. If they start
    # depending on other Schema fields then those fields must also be projected.
    for f in self.schema.feature:
      p = path.ColumnPath(f.name)
      if f.name == _SEQUENCE_COLUMN_NAME:
        if f.type != schema_pb2.STRUCT:
          raise ValueError(
              "Feature {} was expected to be of type STRUCT, but got {}"
              .format(f.name, f))
        result_sequence_struct = schema_pb2.Feature()
        result_sequence_struct.CopyFrom(f)
        result_sequence_struct.ClearField("struct_domain")
        any_sequence_feature_projected = False
        for sf in f.struct_domain.feature:
          sequence_feature_path = p.child(sf.name)
          if sequence_feature_path in used_paths:
            any_sequence_feature_projected = True
            result_sequence_struct.struct_domain.feature.add().CopyFrom(sf)
        if any_sequence_feature_projected:
          result.feature.add().CopyFrom(result_sequence_struct)
      elif p in used_paths:
        result.feature.add().CopyFrom(f)

    tensor_representation_util.SetTensorRepresentationsInSchema(
        result,
        {k: v for k, v in tensor_representations.items() if k in tensor_names})

    return result


class TFSequenceExampleBeamRecord(_TFSequenceExampleRecordBase):
  """TFXIO implementation for serialized tf.SequenceExamples in pcoll[bytes].

  This is a special TFXIO that does not actually do I/O -- it relies on the
  caller to prepare a PCollection of bytes (serialized tf.SequenceExamples).
  """

  def __init__(self,
               physical_format: Text,
               telemetry_descriptors: List[Text],
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
    return TFSequenceExampleBeamRecord(self._physical_format,
                                       self.telemetry_descriptors,
                                       projected_schema,
                                       self.raw_record_column_name)

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorFlowDataset(self,
                        options: dataset_options.TensorFlowDatasetOptions):
    raise NotImplementedError(
        "TFExampleBeamRecord is unable to provide a TensorFlowDataset "
        "because it does not do I/O")


class TFSequenceExampleRecord(_TFSequenceExampleRecordBase):
  """TFXIO implementation for tf.SequenceExample on TFRecord."""

  def __init__(self,
               file_pattern: Union[List[Text], Text],
               telemetry_descriptors: List[Text],
               validate: bool = True,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None):
    """Initializes a TFSequenceExampleRecord TFXIO.

    Args:
      file_pattern: One or a list of glob patterns. If a list, must not be
        empty.
      telemetry_descriptors: A set of descriptors that identify the component
        that is instantiating this TFXIO. These will be used to construct the
        namespace to contain metrics for profiling and are therefore expected to
        be identifiers of the component itself and not individual instances of
        source use.
      validate: Not used. do not set. (not used since post 0.22.1).
      schema: A TFMD Schema describing the dataset.
      raw_record_column_name: If not None, the generated Arrow RecordBatches
        will contain a column of the given name that contains serialized
        records.
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
    return TFSequenceExampleRecord(
        file_pattern=self._file_pattern,
        telemetry_descriptors=self.telemetry_descriptors,
        schema=projected_schema,
        raw_record_column_name=self.raw_record_column_name)

  def RecordBatches(self, options: dataset_options.RecordBatchesOptions):
    raise NotImplementedError

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    """Creates a tf.data.Dataset that yields Tensors.

    The serialized tf.SequenceExamples are parsed by
    `tf.io.parse_sequence_example`.

    See base class (tfxio.TFXIO) for more details.

    Args:
      options: an options object for the tf.data.Dataset. See
        `dataset_options.TensorFlowDatasetOptions` for more details.

    Returns:
      A dataset of `dict` elements, (or a tuple of `dict` elements and label).
      Each `dict` maps feature keys to `Tensor`, `SparseTensor`, or
      `RaggedTensor` objects.

    Raises:
      ValueError: if there is something wrong with the provided schema.
    """
    file_pattern = tf.convert_to_tensor(self._file_pattern)
    dataset = dataset_util.make_tf_record_dataset(
        file_pattern,
        batch_size=options.batch_size,
        num_epochs=options.num_epochs,
        shuffle=options.shuffle,
        shuffle_buffer_size=options.shuffle_buffer_size,
        shuffle_seed=options.shuffle_seed,
        reader_num_threads=options.reader_num_threads,
        drop_final_batch=options.drop_final_batch)

    return self._ParseRawRecordTensorFlowDataset(dataset, options.label_key)


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
      self._decoder = (
          sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
              _SEQUENCE_COLUMN_NAME,
              self._serialized_schema))
    else:
      self._decoder = (
          sequence_example_coder.SequenceExamplesToRecordBatchDecoder(
              _SEQUENCE_COLUMN_NAME))

  def process(self, examples: List[bytes]):
    assert self._decoder is not None  # Workflow ran setup().
    decoded = self._decoder.DecodeBatch(examples)
    if self._raw_record_column_name is None:
      yield decoded
    else:
      yield record_based_tfxio.AppendRawRecordColumn(
          decoded, self._raw_record_column_name, examples)
