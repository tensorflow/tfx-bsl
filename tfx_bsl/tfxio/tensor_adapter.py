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
"""TensorAdapter."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
import collections

import numpy as np
import pyarrow as pa
import six
import tensorflow as tf
from tfx_bsl.arrow import array_util
import typing
from typing import Any, Callable, Dict, List, Text, Optional, Tuple, Union

from tensorflow_metadata.proto.v0 import schema_pb2

TensorRepresentations = Dict[Text, schema_pb2.TensorRepresentation]


class TensorAdapterConfig(collections.namedtuple(
    "TensorAdapterConfig",
    [
        "arrow_schema",
        "tensor_representations",
        "original_type_specs",
    ])):
  """Config to a TensorAdapter.

  Contains all the information needed to create a TensorAdapter.
  """

  def __new__(cls,
              arrow_schema: pa.Schema,
              tensor_representations: TensorRepresentations,
              original_type_specs: Optional[Dict[Text, tf.TypeSpec]] = None):
    return super(TensorAdapterConfig, cls).__new__(
        cls, arrow_schema, tensor_representations, original_type_specs)


class TensorAdapter(object):
  """A TensorAdapter converts a RecordBatch to a collection of TF Tensors.

  The conversion is determined by both the Arrow schema and the
  TensorRepresentations, which must be provided at the initialization time.
  Each TensorRepresentation contains the information needed to translates one
  or more columns in a RecordBatch of the given Arrow schema into a TF Tensor
  or CompositeTensor. They are contained in a Dict whose keys are
  the names of the tensors, which will be the keys of the Dict produced by
  ToBatchTensors().

  TypeSpecs() returns static TypeSpecs of those tensors by their names, i.e.
  if they have a shape, then the size of the first (batch) dimension is always
  unknown (None) because it depends on the size of the RecordBatch passed to
  ToBatchTensors().

  It is guaranteed that for any tensor_name in the given TensorRepresentations
  self.TypeSpecs()[tensor_name].is_compatible_with(
      self.ToBatchedTensors(...)[tensor_name])
  """

  __slots__ = ["_arrow_schema", "_type_handlers", "_type_specs",
               "_original_type_specs"]

  def __init__(self, config: TensorAdapterConfig):

    self._arrow_schema = config.arrow_schema
    self._type_handlers = _BuildTypeHandlers(
        config.tensor_representations, config.arrow_schema)
    self._type_specs = {
        tensor_name: handler.type_spec
        for tensor_name, handler in self._type_handlers
    }

    self._original_type_specs = (
        self._type_specs
        if config.original_type_specs is None else config.original_type_specs)

    for tensor_name, type_spec in self._type_specs.items():
      original_type_spec = self._original_type_specs.get(tensor_name, None)
      if original_type_spec is None or original_type_spec != type_spec:
        raise ValueError(
            "original_type_specs must be a superset of type_specs derived from "
            "TensorRepresentations. But for tensor {}, got {} vs {}".format(
                tensor_name, original_type_spec, type_spec))

  def OriginalTypeSpecs(self) -> Dict[Text, tf.TypeSpec]:
    """Returns the origin's type specs.

    A TFXIO 'Y' may be a result of projection of another TFXIO 'X', in which
    case then 'X' is the origin of 'Y'. And this method returns what
    X.TensorAdapter().TypeSpecs() would return.

    May equal to `self.TypeSpecs()`.

    Returns: a mapping from tensor names to `tf.TypeSpec`s.
    """
    return self._original_type_specs

  def TypeSpecs(self) -> Dict[Text, tf.TypeSpec]:
    """Returns the TypeSpec for each tensor."""
    return self._type_specs

  def ToBatchTensors(
      self,
      record_batch: pa.RecordBatch,
      produce_eager_tensors: Optional[bool] = None) -> Dict[Text, Any]:
    """Returns a batch of tensors translated from `record_batch`.

    Args:
      record_batch: input RecordBatch.
      produce_eager_tensors: controls whether the ToBatchTensors() produces
        eager tensors or ndarrays (or Tensor value objects). If None, determine
        that from whether TF Eager mode is enabled.

    Raises:
      RuntimeError: when Eager Tensors are requested but TF is not executing
        eagerly.
      ValueError: when Any handler failed to produce a Tensor.
    """

    tf_executing_eagerly = tf.executing_eagerly()
    if produce_eager_tensors and not tf_executing_eagerly:
      raise RuntimeError(
          "Eager Tensors were requested but eager mode was not enabled.")
    if produce_eager_tensors is None:
      produce_eager_tensors = tf_executing_eagerly

    if not record_batch.schema.equals(self._arrow_schema):
      raise ValueError("Expected same schema.")
    result = {}
    for tensor_name, handler in self._type_handlers:
      try:
        result[tensor_name] = handler.GetTensor(record_batch,
                                                produce_eager_tensors)
      except Exception as e:
        raise ValueError("Error raised when handling tensor {}: {}"
                         .format(tensor_name, e))

    return result


@six.add_metaclass(abc.ABCMeta)
class _TypeHandler(object):
  """Base class of all type handlers.

  A TypeHandler converts one or more columns in a RecordBatch to a TF Tensor
  or CompositeTensor according to a TensorRepresentation.

  All TypeHandlers are registered by TensorRepresentation types in
  _TYPE_HANDLER_MAP.
  """

  __slots__ = []

  @abc.abstractmethod
  def __init__(self, arrow_schema: pa.Schema,
               tensor_representation: schema_pb2.TensorRepresentation):
    """Initializer.

    It can be assumed that CanHandle(arrow_schema, tensor_representation) would
    return true.

    Args:
      arrow_schema: the Arrow Schema that all the RecordBatches that
        self.GetTensor() will take conform to.
      tensor_representation: the TensorRepresentation that determins the
        conversion.
    """

  @abc.abstractproperty
  def type_spec(self) -> tf.TypeSpec:
    """Returns the TypeSpec of the converted Tensor or CompositeTensor."""

  @abc.abstractmethod
  def GetTensor(self, record_batch: pa.RecordBatch,
                produce_eager_tensors: bool) -> Any:
    """Converts the RecordBatch to Tensor or CompositeTensor.

    The result must be of the same (not only compatible) TypeSpec as
    self.type_spec.

    Args:
      record_batch: a RecordBatch that is of the same Schema as what was
        passed at initialization time.
      produce_eager_tensors: if True, returns Eager Tensors, otherwise returns
        ndarrays or Tensor value objects.

    Returns:
      A Tensor or a CompositeTensor. Note that their types may vary depending
      on whether the TF eager mode is on.
    """

  @staticmethod
  @abc.abstractmethod
  def CanHandle(
      arrow_schema: pa.Schema,
      tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    """Returns true if an instance of the handler can handle the combination."""


class _BaseDenseTensorHandler(_TypeHandler):
  """Base class of DenseTensorHandlers."""

  __slots__ = ["_column_index", "_dtype", "_shape", "_unbatched_flat_len",
               "_convert_to_binary_fn"]

  def __init__(self, arrow_schema: pa.Schema,
               tensor_representation: schema_pb2.TensorRepresentation):
    super(_BaseDenseTensorHandler, self).__init__(arrow_schema,
                                                  tensor_representation)
    dense_rep = tensor_representation.dense_tensor
    column_name = dense_rep.column_name
    self._column_index = arrow_schema.get_field_index(column_name)
    _, value_type = _GetNestDepthAndValueType(arrow_schema[self._column_index])
    self._dtype = _ArrowTypeToTfDtype(value_type)
    self._convert_to_binary_fn = _GetConvertToBinaryFn(value_type)
    unbatched_shape = [
        d.size for d in tensor_representation.dense_tensor.shape.dim
    ]
    self._shape = [None] + unbatched_shape
    self._unbatched_flat_len = int(np.prod(unbatched_shape, initial=1))

  @property
  def type_spec(self) -> tf.TypeSpec:
    # TF's type stub is not correct about TypeSpec and its sub-classes.
    return typing.cast(tf.TypeSpec, tf.TensorSpec(self._shape, self._dtype))

  def _ListArrayToTensor(
      self, list_array: pa.Array,
      produce_eager_tensors: bool) -> Union[np.ndarray, tf.Tensor]:
    """Converts a ListArray to a dense tensor."""
    values = list_array.flatten()
    batch_size = len(list_array)
    expected_num_elements = batch_size * self._unbatched_flat_len
    if len(values) != expected_num_elements:
      raise ValueError(
          "Unable to convert ListArray {} to {}: size mismatch. expected {} "
          "elements but got {}".format(
              list_array, self.type_spec, expected_num_elements, len(values)))
    actual_shape = list(self._shape)
    actual_shape[0] = batch_size
    if self._convert_to_binary_fn is not None:
      values = self._convert_to_binary_fn(values)
    values_np = np.asarray(values).reshape(actual_shape)
    if produce_eager_tensors:
      return tf.convert_to_tensor(values_np)

    return values_np

  @staticmethod
  def BaseCanHandle(
      arrow_schema: pa.Schema,
      tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    depth, value_type = _GetNestDepthAndValueType(
        arrow_schema.field(tensor_representation.dense_tensor.column_name))
    # Can only handle 1-nested lists.
    return depth == 1 and _IsSupportedArrowValueType(value_type)


class _DenseTensorHandler(_BaseDenseTensorHandler):
  """Handles conversion to dense."""

  __slots__ = []

  def GetTensor(self, record_batch: pa.RecordBatch,
                produce_eager_tensors: bool) -> Union[np.ndarray, tf.Tensor]:
    column = record_batch.column(self._column_index)
    return self._ListArrayToTensor(column, produce_eager_tensors)

  @staticmethod
  def CanHandle(arrow_schema: pa.Schema,
                tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    return (_BaseDenseTensorHandler.BaseCanHandle(arrow_schema,
                                                  tensor_representation) and
            not tensor_representation.dense_tensor.HasField("default_value"))


class _DefaultFillingDenseTensorHandler(_BaseDenseTensorHandler):
  """Handles conversion to dense with default filling."""

  __slots__ = ["_default_fill"]

  def __init__(self, arrow_schema: pa.Schema,
               tensor_representation: schema_pb2.TensorRepresentation):
    super(_DefaultFillingDenseTensorHandler, self).__init__(
        arrow_schema, tensor_representation)
    _, value_type = _GetNestDepthAndValueType(arrow_schema[self._column_index])
    self._default_fill = _GetDefaultFill(
        self._shape[1:], value_type,
        tensor_representation.dense_tensor.default_value)

  def GetTensor(self, record_batch: pa.RecordBatch,
                produce_eager_tensors: bool) -> Union[np.ndarray, tf.Tensor]:
    column = record_batch.column(self._column_index)
    column = array_util.FillNullLists(column, self._default_fill)
    return self._ListArrayToTensor(column, produce_eager_tensors)

  @staticmethod
  def CanHandle(arrow_schema: pa.Schema,
                tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    return (
        _BaseDenseTensorHandler.BaseCanHandle(
            arrow_schema, tensor_representation)
        and tensor_representation.dense_tensor.HasField("default_value"))


class _VarLenSparseTensorHandler(_TypeHandler):
  """Handles conversion to varlen sparse."""

  __slots__ = ["_column_index", "_dtype", "_convert_to_binary_fn"]

  def __init__(self, arrow_schema: pa.Schema,
               tensor_representation: schema_pb2.TensorRepresentation):
    super(_VarLenSparseTensorHandler, self).__init__(
        arrow_schema, tensor_representation)
    column_name = tensor_representation.varlen_sparse_tensor.column_name
    self._column_index = arrow_schema.get_field_index(column_name)
    _, value_type = _GetNestDepthAndValueType(arrow_schema[self._column_index])
    self._dtype = _ArrowTypeToTfDtype(value_type)
    self._convert_to_binary_fn = _GetConvertToBinaryFn(value_type)

  @property
  def type_spec(self) -> tf.TypeSpec:
    return typing.cast(tf.TypeSpec, tf.SparseTensorSpec(
        tf.TensorShape([None, None]), self._dtype))

  def GetTensor(self, record_batch: pa.RecordBatch,
                produce_eager_tensors: bool) -> Any:
    array = record_batch.column(self._column_index)
    coo_array, dense_shape_array = array_util.CooFromListArray(array)
    dense_shape_np = dense_shape_array.to_numpy()
    values_array = array.flatten()
    if self._convert_to_binary_fn is not None:
      values_array = self._convert_to_binary_fn(values_array)
    values_np = np.asarray(values_array)
    coo_np = coo_array.to_numpy().reshape(values_np.size, 2)

    if produce_eager_tensors:
      return tf.sparse.SparseTensor(
          indices=tf.convert_to_tensor(coo_np),
          dense_shape=tf.convert_to_tensor(dense_shape_np),
          values=tf.convert_to_tensor(values_np))
    return tf.compat.v1.SparseTensorValue(
        indices=coo_np, dense_shape=dense_shape_np, values=values_np)

  @staticmethod
  def CanHandle(arrow_schema: pa.Schema,
                tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    depth, value_type = _GetNestDepthAndValueType(
        arrow_schema.field(
            tensor_representation.varlen_sparse_tensor.column_name))
    # Currently can only handle 1-nested lists, but can easily support
    # arbitrarily nested ListArrays.
    return depth == 1 and _IsSupportedArrowValueType(value_type)


class _SparseTensorHandler(_TypeHandler):
  """Handles conversion to SparseTensors."""

  __slots__ = ["_index_column_indices", "_value_column_index", "_shape",
               "_dtype", "_coo_size", "_convert_to_binary_fn"]

  def __init__(self, arrow_schema: pa.Schema,
               tensor_representation: schema_pb2.TensorRepresentation):
    super(_SparseTensorHandler, self).__init__(
        arrow_schema, tensor_representation)
    sparse_representation = tensor_representation.sparse_tensor
    self._index_column_indices = tuple(
        arrow_schema.get_field_index(c)
        for c in sparse_representation.index_column_names)
    self._value_column_index = arrow_schema.get_field_index(
        sparse_representation.value_column_name)
    self._shape = [dim.size for dim in sparse_representation.dense_shape.dim]
    _, value_type = _GetNestDepthAndValueType(
        arrow_schema[self._value_column_index])
    self._dtype = _ArrowTypeToTfDtype(value_type)
    self._coo_size = len(self._shape) + 1
    self._convert_to_binary_fn = _GetConvertToBinaryFn(value_type)

  @property
  def type_spec(self) -> tf.TypeSpec:
    return typing.cast(tf.TypeSpec,
                       tf.SparseTensorSpec(tf.TensorShape([None] + self._shape),
                                           self._dtype))

  def GetTensor(self, record_batch: pa.RecordBatch,
                produce_eager_tensors: bool) -> Any:
    values_array = record_batch.column(self._value_column_index)
    values_parent_indices = array_util.GetFlattenedArrayParentIndices(
        values_array)
    indices_arrays = [np.asarray(values_parent_indices)]
    for index_column_index in self._index_column_indices:
      indices_arrays.append(
          np.asarray(record_batch.column(index_column_index).flatten()))
    flat_values_array = values_array.flatten()
    if self._convert_to_binary_fn is not None:
      flat_values_array = self._convert_to_binary_fn(flat_values_array)
    values_np = np.asarray(flat_values_array)
    coo_np = np.empty(shape=(len(values_np), self._coo_size), dtype=np.int64)
    try:
      np.stack(indices_arrays, axis=1, out=coo_np)
    except ValueError as e:
      raise ValueError("Error constructing the COO for SparseTensor. "
                       "number of values: {}; "
                       "size of each index array: {}; "
                       "original error {}.".format(
                           len(values_np), [len(i) for i in indices_arrays], e))

    dense_shape = [len(record_batch)] + self._shape

    if produce_eager_tensors:
      return tf.sparse.SparseTensor(
          indices=tf.convert_to_tensor(coo_np),
          dense_shape=tf.convert_to_tensor(dense_shape, dtype=tf.int64),
          values=tf.convert_to_tensor(values_np))
    return tf.compat.v1.SparseTensorValue(
        indices=coo_np, dense_shape=dense_shape, values=values_np)

  @staticmethod
  def CanHandle(
      arrow_schema: pa.Schema,
      tensor_representation: schema_pb2.TensorRepresentation) -> bool:
    """Returns whether `tensor_representation` can be handled."""
    sparse_representation = tensor_representation.sparse_tensor
    if (len(sparse_representation.dense_shape.dim) !=
        len(sparse_representation.index_column_names)):
      return False
    if any([d.size <= 0 for d in sparse_representation.dense_shape.dim]):
      return False

    # All the index columns must be of integral types.
    for index_column in sparse_representation.index_column_names:
      depth, value_type = _GetNestDepthAndValueType(
          arrow_schema.field(index_column))
      if depth != 1 or not pa.types.is_integer(value_type):
        return False

    depth, value_type = _GetNestDepthAndValueType(
        arrow_schema.field(sparse_representation.value_column_name))
    return depth == 1 and _IsSupportedArrowValueType(value_type)

# Mapping from TensorRepresentation's "kind" oneof field name to TypeHandler
# classes. Note that one kind may have multiple handlers and the first one
# whose CanHandle() returns true will be used.
_TYPE_HANDLER_MAP = {
    "dense_tensor": [_DenseTensorHandler, _DefaultFillingDenseTensorHandler],
    "varlen_sparse_tensor": [_VarLenSparseTensorHandler],
    "sparse_tensor": [_SparseTensorHandler],
}


def _BuildTypeHandlers(
    tensor_representations: Dict[Text, schema_pb2.TensorRepresentation],
    arrow_schema: pa.Schema) -> List[Tuple[Text, _TypeHandler]]:
  """Builds type handlers according to TensorRepresentations."""
  result = []
  for tensor_name, rep in six.iteritems(tensor_representations):
    potential_handlers = _TYPE_HANDLER_MAP.get(rep.WhichOneof("kind"))
    if not potential_handlers:
      raise ValueError("Unable to handle tensor {} with rep {}".format(
          tensor_name, rep))
    found_handler = False
    for h in potential_handlers:
      if h.CanHandle(arrow_schema, rep):
        found_handler = True
        result.append((tensor_name, h(arrow_schema, rep)))
        break
    if not found_handler:
      raise ValueError("Unable to handle tensor {} with rep {} "
                       "against schema: {}".format(tensor_name, rep,
                                                   arrow_schema))

  return result


def _GetNestDepthAndValueType(
    arrow_field: pa.Field) -> Tuple[int, pa.DataType]:
  """Returns the depth of a nest list and its innermost value type."""
  arrow_type = arrow_field.type
  depth = 0
  while pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
    depth += 1
    arrow_type = arrow_type.value_type

  return depth, arrow_type


def _IsBinaryLike(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_binary(arrow_type) or
          pa.types.is_large_binary(arrow_type) or
          pa.types.is_string(arrow_type) or
          pa.types.is_large_string(arrow_type))


def _IsSupportedArrowValueType(arrow_type: pa.DataType) -> bool:
  return (pa.types.is_integer(arrow_type) or
          pa.types.is_floating(arrow_type) or
          _IsBinaryLike(arrow_type))


def _ArrowTypeToTfDtype(arrow_type: pa.DataType) -> tf.DType:
  # TODO(zhuo): Remove the special handling for LargeString/Binary when
  # to_pandas_dtype() can handle them.
  if _IsBinaryLike(arrow_type):
    return tf.string
  return tf.dtypes.as_dtype(arrow_type.to_pandas_dtype())


def _GetAllowedDefaultValue(
    value_type: pa.DataType,
    default_value_proto: schema_pb2.TensorRepresentation.DefaultValue
) -> Union[int, float, bytes]:
  """Returns the default value set in DefaultValue proto or raises."""
  kind = default_value_proto.WhichOneof("kind")
  if kind in ("int_value", "uint_value") and pa.types.is_integer(value_type):
    value = getattr(default_value_proto, kind)
    iinfo = np.iinfo(value_type.to_pandas_dtype())
    if value <= iinfo.max and value >= iinfo.min:
      return value
    else:
      raise ValueError("Integer default value out of range: {} is set for a "
                       "{} column".format(value, value_type))
  elif kind == "float_value" and pa.types.is_floating(value_type):
    return default_value_proto.float_value
  elif kind == "bytes_value" and _IsBinaryLike(value_type):
    return default_value_proto.bytes_value

  raise ValueError(
      "Incompatible default value: {} is set for a {} column".format(
          kind, value_type))


def _GetDefaultFill(
    unbatched_shape: List[int], value_type: pa.DataType,
    default_value_proto: schema_pb2.TensorRepresentation.DefaultValue
) -> pa.Array:
  """Returns an Array full of the default value given in the proto."""

  size = int(np.prod(unbatched_shape, initial=1))
  return pa.array([
      _GetAllowedDefaultValue(value_type, default_value_proto)] * size,
                  type=value_type)


def _GetConvertToBinaryFn(
    array_type: pa.DataType) -> Optional[Callable[[pa.Array], pa.Array]]:
  """Returns a function that converts a StringArray to BinaryArray."""

  if pa.types.is_string(array_type):
    return lambda array: array.view(pa.binary())
  if pa.types.is_large_string(array_type):
    return lambda array: array.view(pa.large_binary())
  return None
