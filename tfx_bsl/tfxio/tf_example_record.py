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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.coders import batch_util
from tfx_bsl.coders import example_coder
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import dataset_util
from tfx_bsl.tfxio import record_based_tfxio
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_representation_util
from tfx_bsl.tfxio import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2

_FEATURE_NAME_PREFIX = "_tfx_bsl_"


class _TFExampleRecordBase(record_based_tfxio.RecordBasedTFXIO):
  """Base class for TFXIO implementations for record based tf.Examples."""

  def __init__(self,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[str] = None,
               telemetry_descriptors: Optional[List[str]] = None,
               physical_format: Optional[str] = None,
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
    def ptransform_fn(raw_records_pcoll: beam.pvalue.PCollection):
      return (
          raw_records_pcoll
          | "Batch"
          >> batch_util.BatchRecords(batch_size, self._telemetry_descriptors)
          | "Decode"
          >> beam.ParDo(
              _DecodeBatchExamplesDoFn(
                  self._GetSchemaForDecoding(), self.raw_record_column_name
              )
          )
      )

    return beam.ptransform_fn(ptransform_fn)()

  def _ArrowSchemaNoRawRecordColumn(self) -> pa.Schema:
    schema = self._GetSchemaForDecoding()
    if schema is None:
      raise ValueError("TFMD schema not provided. Unable to derive an "
                       "Arrow schema")
    return example_coder.ExamplesToRecordBatchDecoder(
        schema.SerializeToString()).ArrowSchema()

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return tensor_representation_util.InferTensorRepresentationsFromMixedSchema(
        self._schema)

  def _GetSchemaForDecoding(self) -> Optional[schema_pb2.Schema]:
    return (self._schema
            if self._schema_for_decoding is None else self._schema_for_decoding)

  def _GetTfExampleParserConfig(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Creates a dict feature spec that can be used in tf.io.parse_example().

    To reduce confusion: 'tensor name' are the keys of TensorRepresentations.
    'feature name' are the keys to the tf.Example parser config.
    'column name' are the features in the schema.

    Returns:
      Two maps. The first is the parser config that maps from feature
      name to a tf.io Feature. The second is a mapping from feature names to
      tensor names.

    Raises:
      ValueError: if the tf.Example parser config is invalid.
    """
    if self._schema is None:
      raise ValueError(
          "Unable to create a parsing config because no schema is provided.")

    column_name_to_type = {f.name: f.type for f in self._schema.feature}
    features = {}
    feature_name_to_tensor_name = {}
    for tensor_name, tensor_rep in self.TensorRepresentations().items():
      paths = (
          tensor_representation_util.GetSourceColumnsFromTensorRepresentation(
              tensor_rep
          )
      )
      if len(paths) == 1:
        # The parser config refers to a single tf.Example feature. In this case,
        # the key to the parser config needs to be the name of the feature.
        column_name = paths[0].initial_step()
        value_type = column_name_to_type[column_name]
      else:
        # The parser config needs to refer to multiple tf.Example features. In
        # this case the key to the parser config does not matter. We preserve
        # the tensor representation key.
        column_name = tensor_name
        value_type = column_name_to_type[
            tensor_representation_util
            .GetSourceValueColumnFromTensorRepresentation(
                tensor_rep).initial_step()]
      parse_config = tensor_representation_util.CreateTfExampleParserConfig(
          tensor_rep, value_type)

      if _is_multi_column_parser_config(parse_config):
        # Create internal naming, to prevent possible naming collisions between
        # tensor_name and column_name.
        feature_name = _FEATURE_NAME_PREFIX + tensor_name + "_" + column_name
      else:
        feature_name = column_name
      if feature_name in feature_name_to_tensor_name:
        clashing_tensor_rep = self.TensorRepresentations()[
            feature_name_to_tensor_name[feature_name]]
        raise ValueError(f"Unable to create a valid parsing config. Feature "
                         f"name: {feature_name} is a duplicate of "
                         f"tensor representation: {clashing_tensor_rep}")
      feature_name_to_tensor_name[feature_name] = tensor_name
      features[feature_name] = parse_config

    _validate_tf_example_parser_config(features, self._schema)

    return features, feature_name_to_tensor_name

  def _RenameFeatures(
      self, feature_dict: Dict[str, Any],
      feature_name_to_tensor_name: Dict[str, str]) -> Dict[str, Any]:
    """Renames the feature keys to use the tensor representation keys."""
    renamed_feature_dict = {}
    for feature_name, tensor in feature_dict.items():
      renamed_feature_dict[
          feature_name_to_tensor_name[feature_name]] = tensor

    return renamed_feature_dict


class TFExampleBeamRecord(_TFExampleRecordBase):
  """TFXIO implementation for serialized tf.Examples in pcoll[bytes].

  This is a special TFXIO that does not actually do I/O -- it relies on the
  caller to prepare a PCollection of bytes (serialized tf.Examples).
  """

  def __init__(self,
               physical_format: str,
               telemetry_descriptors: Optional[List[str]] = None,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[str] = None):
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

  def _ProjectImpl(self, tensor_names: List[str]) -> tfxio.TFXIO:
    # Note: We only copy projected features into the new schema because the
    # coder, and ArrowSchema() only care about Schema.feature. If they start
    # depending on other Schema fields then those fields must also be projected.
    projected_schema = (
        tensor_representation_util.ProjectTensorRepresentationsInSchema(
            self._schema, tensor_names))
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
               file_pattern: Union[List[str], str],
               validate: bool = True,
               schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[str] = None,
               telemetry_descriptors: Optional[List[str]] = None):
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

  def _ProjectImpl(self, tensor_names: List[str]) -> tfxio.TFXIO:
    # Note: We only copy projected features into the new schema because the
    # coder, and ArrowSchema() only care about Schema.feature. If they start
    # depending on other Schema fields then those fields must also be projected.
    projected_schema = (
        tensor_representation_util.ProjectTensorRepresentationsInSchema(
            self._schema, tensor_names))
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
        self._schema.SerializeToString() if self._schema is not None else None
    )
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
    (tf_example_parser_config,
     feature_name_to_tensor_name) = self._GetTfExampleParserConfig()

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

    # Parse `Example` tensors to a dictionary of `Feature` tensors.
    dataset = dataset.apply(
        tf.data.experimental.parse_example_dataset(tf_example_parser_config))

    dataset = dataset.map(
        lambda x: self._RenameFeatures(x, feature_name_to_tensor_name))

    label_key = options.label_key
    if label_key is not None:
      dataset = self._PopLabelFeatureFromDataset(dataset, label_key)

    return dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


@beam.typehints.with_input_types(List[bytes])
@beam.typehints.with_output_types(pa.RecordBatch)
class _DecodeBatchExamplesDoFn(beam.DoFn):
  """Batches serialized protos bytes and decode them into an Arrow table."""

  def __init__(self, schema: Optional[schema_pb2.Schema],
               raw_record_column_name: Optional[str]):
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
    if not self._decoder:
      raise ValueError("Decoder uninitialized. Run setup() first.")
    decoded = self._decoder.DecodeBatch(examples)
    if self._raw_record_column_name is None:
      yield decoded
    else:
      yield record_based_tfxio.AppendRawRecordColumn(
          decoded, self._raw_record_column_name, examples)


def _validate_tf_example_parser_config(config: Dict[str, Any],
                                       schema: schema_pb2.Schema) -> None:
  """Validate a tf_example_parse_config by tracing parse_example."""

  # TODO(b/173738031): We would have used a tf.io.validate_parsing_config() if
  # it existed.
  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
  def parse(i):
    tf.io.parse_example(i, config)

  try:
    # This forces a tracing of `parse`, and raises if the parsing config
    # is not valid.
    parse.get_concrete_function()

  except (ValueError, TypeError) as err:
    raise ValueError("Unable to create a valid parsing config from the "
                     "provided schema's tensor representation: {}. Due to the "
                     "following error: {}".format(schema, err))


def _is_multi_column_parser_config(parser_config):
  return isinstance(parser_config, (tf.io.SparseFeature, tf.io.RaggedFeature))
