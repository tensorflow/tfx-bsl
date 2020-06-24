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
"""Utils to convert TF Tensors to Arrow arrays."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
from typing import Dict, List, Text, Tuple, Union

from absl import logging
import pyarrow as pa
import six
import tensorflow as tf

# CompositeTensor is not public yet.
from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import
from tensorflow_metadata.proto.v0 import schema_pb2


if tf.__version__ < "2":
  logging.warning("tfx_bsl.tfxio.tensor_to_arrow will not work correctly with "
                  "TF 1.x.")


TensorAlike = Union[tf.Tensor, composite_tensor.CompositeTensor]


class TensorsToRecordBatchConverter(object):
  """Converts a Dict[Text, TensorAlike] to a RecordBatch."""

  __slots__ = ["_handlers", "_arrow_schema"]

  def __init__(self, type_specs: Dict[Text, tf.TypeSpec]):
    """Initializer.

    Args:
      type_specs: a mapping from names of tensors to their TypeSpecs. When
        calling convert(), the dict of tensors passed in must contain the
        same names, and each TensorAlike must be compatible to their TypeSpecs.
    """
    self._handlers = _make_handlers(type_specs)
    all_fields = []
    seen_column_names = set()
    for tensor_name, handler in self._handlers:
      for f in handler.arrow_fields():
        if f.name in seen_column_names:
          raise ValueError("Handler for tensor {} produces a column of a "
                           "conflicting name: {}".format(tensor_name, f.name))
        seen_column_names.add(f.name)
        all_fields.append(f)
    self._arrow_schema = pa.schema(all_fields)

  def arrow_schema(self) -> pa.Schema:
    """Returns the schema of the RecordBatch output by convert()."""
    return self._arrow_schema

  def tensor_representations(
      self) -> Dict[Text, schema_pb2.TensorRepresentation]:
    """Returns the TensorRepresentations for each TensorAlike.

    The TypeSpecs of those TensorAlikes are specified in the initializer.
    These TensorRepresentations, along with the schema returned by
    arrow_schema() comprises all the information needed to turn the
    RecordBatches produced by convert() back to TensorAlikes.

    Returns:
      a dict mapping tensor names to their TensorRepresentations.
    """
    return {
        tensor_name: handler.tensor_representation()
        for tensor_name, handler in self._handlers
    }

  def convert(self, tensors: Dict[Text, TensorAlike]) -> pa.RecordBatch:
    """Converts a dict of tensors to a RecordBatch.

    Args:
      tensors: must contain the same keys as the dict passed to the initialier.
        and each TensorAlike must be compatible with the corresponding TypeSpec.

    Returns:
      a RecordBatch, whose schema equals to self.arrow_schema().
    """
    assert len(self._handlers) == len(tensors)
    arrays = []
    for tensor_name, handler in self._handlers:
      arrays.extend(handler.convert(tensors[tensor_name]))

    return pa.record_batch(arrays, schema=self._arrow_schema)


@six.add_metaclass(abc.ABCMeta)
class _TypeHandler(object):
  """Interface of a type handler that converts a tensor to arrow arrays.

  Note that a handler may convert a Tensor to multiple pa.Arrays. See
  arrow_fields().
  """

  __slots__ = ["_tensor_name", "_type_spec"]

  def __init__(self, tensor_name: Text, type_spec: tf.TypeSpec):
    self._tensor_name = tensor_name
    self._type_spec = type_spec

  def convert(self, tensor: TensorAlike) -> List[pa.Array]:
    if not self._type_spec.is_compatible_with(tensor):
      raise TypeError("Expected {} but got {}".format(
          self._type_spec, _get_type_spec(tensor)))
    return self._convert_internal(tensor)

  @abc.abstractmethod
  def arrow_fields(self) -> List[pa.Field]:
    """Returns the name and type (in a pa.Field) of result pa.Arrays.

    Note that a Handler can convert a Tensor to multiple pa.Arrays. It must
    make sure _convert_internal() returns those Arrays of the types declared
    here, in the correct order.
    """

  @abc.abstractmethod
  def _convert_internal(self, tensor: TensorAlike) -> List[pa.Array]:
    """Converts the given TensorAlike to a list of pa.Arrays.

    Each element in the list should correspond to one in `arrow_fields()`.

    Args:
      tensor: the TensorAlike to be converted.
    """

  @abc.abstractmethod
  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    """Returns the TensorRepresentation.

    The TensorRepresentation, along with `arrow_fields()` can be used to
    convert from pa.Arrays back to Tensors.
    """

  @staticmethod
  @abc.abstractmethod
  def can_handle(type_spec: tf.TypeSpec) -> bool:
    """Returns `True` if the handler can handle the given `tf.TypeSpec`."""


class _VarLenSparseTensorHandler(_TypeHandler):
  """Handles 2-D var-len (ragged) sparse tensor."""

  __slots__ = ["_values_arrow_type"]

  def __init__(self, tensor_name: Text, type_spec: tf.TypeSpec):
    super(_VarLenSparseTensorHandler, self).__init__(tensor_name, type_spec)
    self._values_arrow_type = _tf_dtype_to_arrow_type(type_spec.dtype)

  def _convert_internal(self, tensor: TensorAlike) -> List[pa.Array]:
    r = tf.RaggedTensor.from_sparse(tensor)
    return [pa.ListArray.from_arrays(
        pa.array(r.row_splits.numpy(), type=pa.int32()),
        pa.array(r.values.numpy(), type=self._values_arrow_type))]

  def arrow_fields(self) -> List[pa.Field]:
    return [
        pa.field(self._tensor_name,
                 pa.list_(_tf_dtype_to_arrow_type(self._type_spec.dtype)))
    ]

  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    result = schema_pb2.TensorRepresentation()
    result.varlen_sparse_tensor.column_name = self._tensor_name
    return result

  @staticmethod
  def can_handle(type_spec: tf.TypeSpec) -> bool:
    if not isinstance(type_spec, tf.SparseTensorSpec):
      return False
    return (
        type_spec.shape.is_compatible_with([None, None]) and
        type_spec.dtype != tf.bool)


_ALL_HANDLERS_CLS = [_VarLenSparseTensorHandler]


def _tf_dtype_to_arrow_type(dtype: tf.DType):
  """Maps a tf Dtype to an Arrow type."""
  if dtype == tf.string:
    return pa.binary()
  elif dtype == tf.bool:
    raise TypeError("Unable to handle bool tensors -- consider casting it to a "
                    "tf.uint8")
  return pa.from_numpy_dtype(dtype.as_numpy_dtype)


def _get_type_spec(tensor_alike: TensorAlike):
  """Returns the TypeSpec of a TensorAlike."""
  if isinstance(tensor_alike, tf.Tensor):
    return tf.TensorSpec.from_tensor(tensor_alike)
  elif isinstance(tensor_alike, composite_tensor.CompositeTensor):
    return tensor_alike._type_spec  # pylint:disable=protected-access
  raise TypeError("Not a Tensor or CompositeTensor: {}".format(
      type(tensor_alike)))


def _make_handlers(
    type_specs: Dict[Text, tf.TypeSpec]) -> List[Tuple[Text, _TypeHandler]]:
  return [
      (tensor_name, _get_handler(tensor_name, type_spec))
      for tensor_name, type_spec in sorted(type_specs.items())
  ]


def _get_handler(
    tensor_name: Text, type_spec: tf.TypeSpec) -> _TypeHandler:
  """Returns a TypeHandler that can handle `type_spec`."""
  for handler_cls in _ALL_HANDLERS_CLS:
    if handler_cls.can_handle(type_spec):
      return handler_cls(tensor_name, type_spec)
  raise ValueError(
      "No handler found for tensor {} of spec {}. "
      "Note that tensors with dtype == tf.bool cannot be handled in general -- "
      "consider casting them to tf.uint8."
      .format(tensor_name, type_spec))
