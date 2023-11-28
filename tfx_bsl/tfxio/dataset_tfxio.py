# Copyright 2023 Google LLC
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

"""TFXIO Adapter for tf.data.Dataset."""

import collections
import os
import tempfile
from typing import Dict, Iterator, List, NamedTuple, Optional, OrderedDict, Tuple, Union

import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import tensor_adapter
from tfx_bsl.tfxio import tensor_to_arrow
from tfx_bsl.tfxio import tfxio


SpecType = Union[
    tf.TensorSpec,
    tf.RaggedTensorSpec,
    tf.SparseTensorSpec,
    Tuple['SpecType'],
    NamedTuple,
    Dict[str, 'SpecType'],
    OrderedDict[str, 'SpecType'],
]


def _CanonicalType(dtype: tf.dtypes.DType) -> tf.dtypes.DType:
  """Returns TFXIO-canonical version of the given type."""
  if dtype.is_floating:
    return tf.float32
  elif dtype.is_integer or dtype.is_bool:
    return tf.int64
  elif dtype in (tf.string, bytes):
    return tf.string
  else:
    raise TypeError(
        f'Got {dtype}. Only tf.uint8/16/32, tf.int8/16/32/64, tf.float16/32 and'
        ' bytes/tf.string supported.'
    )


def _IsDict(element: SpecType) -> bool:
  return isinstance(element, (dict, collections.OrderedDict))


def _IsNamedTuple(element: SpecType) -> bool:
  return hasattr(element, '_fields')


def _GetFeatureNames(
    spec: SpecType, new_feature_index: int = 0
) -> List[str]:
  """Recursively generates feature names for given Dataset Structure.

  Args:
    spec: Dataset Structure
    new_feature_index: New feature index

  Returns:
    List of feature names
  """

  name_gen = 'feature'
  feature_names = []
  if _IsDict(spec):
    # If the spec is a OrderedDictionary/Dictionary
    # 1. Iterate over (feature_name, <TensorLike-Spec>) pairs
    # 2. Check the structure of each spec.
    # 2.a If the spec is a nested structure, recursively retrieve nested
    # feature names and append to features list as '<Outer>_<NestedFeature>'.
    # 2.b If the spec is not nested, append the feature_name to return values.

    for feature_name, tensor_spec in spec.items():  # pytype: disable=attribute-error
      if not tf.nest.is_nested(tensor_spec):
        feature_names.append(feature_name)

      elif _IsDict(tensor_spec) or _IsNamedTuple(tensor_spec):
        if _IsNamedTuple(tensor_spec):
          tensor_spec = tensor_spec._asdict()
        feature_names.extend(
            [
                feature_name + '_' + nested_feature_name
                for nested_feature_name in _GetFeatureNames(
                    tensor_spec, new_feature_index
                )
            ]
        )

  # If the spec is a NamedTuple, converts it to dictionary, and process
  # as a Dict.
  elif _IsNamedTuple(spec):
    spec = spec._asdict()  # pytype: disable=attribute-error
    feature_names.extend(_GetFeatureNames(spec, new_feature_index))

  # If the spec is a regular tuple, iterate and branch out if a nested
  # structure is found.
  elif isinstance(spec, tuple):
    for single_spec in spec:
      if not tf.nest.is_nested(single_spec):
        feature_names.append(''.join([name_gen, str(new_feature_index)]))
        new_feature_index += 1
      else:
        feature_names.extend(_GetFeatureNames(single_spec, new_feature_index))

  # If spec is not nested, and is a standalone TensorSpec with no feature name
  # new feature name is generated.
  else:
    feature_names.append(''.join([name_gen, str(new_feature_index)]))

  return feature_names


def _GetDictStructureForElementSpec(
    *spec: SpecType, feature_names: Optional[List[str]] = None
) -> OrderedDict[str, SpecType]:
  """Creates a flattened Dictionary-like Structure of given Dataset Structure.

  Args:
    *spec: Element spec for the dataset. This is used as *arg, since it can be a
      tuple and is unpacked if not used as such.
    feature_names: (kwarg) Feature names for columns in Dataset.

  Returns:
    OrderedDict Structure.
  """
  original_spec = spec[0]

  # Flattening the element_spec creates a list of Tensor Specs
  flattened_spec = tf.nest.flatten(original_spec)

  if not feature_names or (len(flattened_spec) != len(feature_names)):
    feature_names = _GetFeatureNames(original_spec)

  return collections.OrderedDict(zip(feature_names, flattened_spec))


def _PrepareDataset(
    dataset: tf.data.Dataset, feature_names: Optional[List[str]]
) -> tf.data.Dataset:
  """Prepare tf.data.Dataset by modifying structure and casting to supporting dtypes.

  Args:
    dataset: A tf.data.Dataset having any structure of <tuple, namedtuple, dict,
      OrderedDict>.
    feature_names: Optional list of feature_names for flattened features in the
      Dataset.

  Returns:
    A modified tf.data.Dataset with flattened OrderedDict structure and TFXIO
    supported dtypes.
  """

  dict_structure = _GetDictStructureForElementSpec(
      dataset.element_spec, feature_names=feature_names
  )

  def _UpdateStructureAndCastDtypes(*x):
    x = tf.nest.flatten(x)
    x = tf.nest.pack_sequence_as(dict_structure, x)
    for k, v in x.items():
      x[k] = tf.cast(v, _CanonicalType(v.dtype))
    return x

  return dataset.map(_UpdateStructureAndCastDtypes)


def _LoadDatasetAsRecordBatch(
    shard: Tuple[int, str],
    converter: tensor_to_arrow.TensorsToRecordBatchConverter,
    feature_names: Optional[List[str]],
    use_custom_reader: bool,
) -> Iterator[pa.RecordBatch]:
  """Yields RecordBatches from a single shard.

  Args:
    shard: Tuple of shard index and saved dataset path.
    converter: TensorsToRecordBatchConverter Object.
    feature_names: Optional list of feature names.
    use_custom_reader: Flag to specify the shard reading method.

  Yields:
    Yields RecordBatches.
  """
  shard_num, path = shard

  def _ReaderFunc(datasets):
    return datasets.skip(shard_num).take(1).get_single_element()

  if use_custom_reader:
    dataset = tf.data.Dataset.load(path, reader_func=_ReaderFunc)
  else:
    dataset = tf.data.Dataset.load(path)

  dataset = _PrepareDataset(dataset, feature_names=feature_names).prefetch(
      tf.data.AUTOTUNE
  )

  # Handles reading empty shards.
  try:
    for data in dataset:
      yield converter.convert(data)
  except (tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError):
    pass


class DatasetTFXIOOptions(NamedTuple):
  """Options for DatasetTFXIO.

  working_dir: A directory to write the intermediate materialized dataset. It is
    expected to be remote accessible, since the execution is transferred to Beam
    runners during BeamSource API call.
  feature_names: List of feature names for columns/attributes. Expected to have
    features name for all the features in the flattened feature_spec.
    If None: Creates default feature names for non-named features and utilizes
      existing feature names from named features (Dict, NamedTuples).
    If Partial, creates default feature names for all the features.
  num_shards: Number of shards to write the materialized dataset. Uses default
    tf.data sharding if None. Check tf.data.Dataset.save documentation for more
    details. [https://www.tensorflow.org/api_docs/python/tf/data/Dataset#save]
  """

  working_dir: str = tempfile.mkdtemp()
  feature_names: Optional[List[str]] = None
  num_shards: Optional[int] = None


class DatasetTFXIO(tfxio.TFXIO):
  """TFXIO implementation for tf.data.Dataset sources."""

  def __init__(
      self,
      dataset: tf.data.Dataset,
      options: DatasetTFXIOOptions = DatasetTFXIOOptions(),
  ):
    """Initializes DatasetTFXIO.

    Args:
      dataset: A batched, finite tf.data.Dataset
      options: DatasetTFXIOOptions Object, providing working directory, feature
        names and number of shards for intermediate materialization options.
    """
    self._dataset = dataset
    self._options = options
    self._use_custom_sharding = bool(self._options.num_shards)

    # Below we retrieve type_specs for the prepared dataset, which are used to
    # create TensorsToRecordBatchConverter object. Since, it requires
    # preparing the dataset, we initially modify only a single element (faster).
    # The entire dataset is prepared in distributed manner on the beam runners
    # during distributed read.
    self._type_specs = _PrepareDataset(
        self._dataset.take(1), feature_names=self._options.feature_names
    ).element_spec
    self._converter = tensor_to_arrow.TensorsToRecordBatchConverter(
        self._type_specs
    )
    self._file_pattern = os.path.join(
        self._options.working_dir, 'saved_dataset'
    )

  def _SaveDataset(self, batch_size: Optional[int] = None):
    def _CustomShardFunc(*unused_args) -> tf.Tensor:
      if self._options.num_shards == 1:
        return tf.constant(0, dtype=tf.int64)
      return tf.random.uniform(
          shape=(), maxval=self._options.num_shards, dtype=tf.int64
      )

    if batch_size:
      self._dataset = self._dataset.rebatch(batch_size)

    if self._use_custom_sharding:
      self._dataset.save(self._file_pattern, shard_func=_CustomShardFunc)
    else:
      self._dataset.save(self._file_pattern)

  def ArrowSchema(self) -> pa.Schema:
    return self._converter.arrow_schema()

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return self._converter.tensor_representations()

  def TensorAdapterConfig(self):
    return tensor_adapter.TensorAdapterConfig(
        self.ArrowSchema(), self.TensorRepresentations(), self._type_specs
    )

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:
    self._SaveDataset(batch_size)

    def _PTransformFn(pipeline):
      num_shards = self._options.num_shards or 1
      return (
          pipeline
          | beam.Create(enumerate([self._file_pattern] * num_shards))
          | beam.FlatMap(
              _LoadDatasetAsRecordBatch,
              self._converter,
              self._options.feature_names,
              self._use_custom_sharding,
          )
      )

    return beam.ptransform_fn(_PTransformFn)()

  def RecordBatches(self, options):
    return _PrepareDataset(
        self._dataset, feature_names=options.feature_names
    ).prefetch(tf.data.AUTOTUNE)

  def TensorFlowDataset(
      self, options: dataset_options.TensorFlowDatasetOptions
  ) -> tf.data.Dataset:
    raise NotImplementedError

  def _ProjectImpl(self, tensor_names):
    raise NotImplementedError
