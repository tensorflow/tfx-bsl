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
from typing import Dict, List, NamedTuple, Optional, OrderedDict, Tuple, Union
import tensorflow as tf


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
