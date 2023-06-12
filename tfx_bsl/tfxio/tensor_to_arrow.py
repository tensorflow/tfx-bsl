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
"""Utils to convert TF Tensors or their values to Arrow arrays."""

import abc
from typing import Dict, List, Tuple, FrozenSet

import numpy as np
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.arrow import array_util
from tfx_bsl.types import common_types

from tensorflow_metadata.proto.v0 import schema_pb2


class TensorsToRecordBatchConverter(object):
  """Converts a Dict[str, TensorAlike] to a RecordBatch."""

  __slots__ = ["_handlers", "_arrow_schema"]

  class Options(object):
    """Options to TensorsToRecordBatchConverter."""

    def __init__(
        self,
        sparse_tensor_value_column_name_template: str = "{tensor_name}$values",
        sparse_tensor_index_column_name_template:
        str = "{tensor_name}$index{index}",
        generic_sparse_tensor_names: FrozenSet[str] = frozenset()):
      """Initialzier.

      Args:
        sparse_tensor_value_column_name_template: a `str.format()` template
          for the column name for the values component of a generic
          SparseTensor. This template should contain a "{tensor_name}" token.
        sparse_tensor_index_column_name_template: a `str.format()` template
          for the column name for the sparse index components of a generic
          SparseTensor. This template should contain a "{tensor_name}" token
          and an "{index}" token.
        generic_sparse_tensor_names: a set of SparseTensor names that must be
          converted as generic SparseTensors. Its purpose is to disambiguate
          2-D varlen and 2-D generic SparseTensors. It is not necessary to
          include names of >2-D SparseTensors since they can only be handled as
          generic SparseTensors.
      """
      self.sparse_tensor_value_column_name_template = (
          sparse_tensor_value_column_name_template)
      self.sparse_tensor_index_column_name_template = (
          sparse_tensor_index_column_name_template)
      self.generic_sparse_tensor_names = generic_sparse_tensor_names

  def __init__(self,
               type_specs: Dict[str, tf.TypeSpec],
               options: Options = Options()):
    """Initializer.

    Args:
      type_specs: a mapping from names of tensors to their TypeSpecs. When
        calling convert(), the dict of tensors passed in must contain the
        same names, and each TensorAlike must be compatible to their TypeSpecs.
      options: options.
    """
    self._handlers = _make_handlers(type_specs, options)
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
      self) -> Dict[str, schema_pb2.TensorRepresentation]:
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

  def convert(self, tensors: Dict[str,
                                  common_types.TensorAlike]) -> pa.RecordBatch:
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
      try:
        arrays.extend(handler.convert(tensors[tensor_name]))
      except Exception as e:
        # Reraise the same exception with a extra information.
        dbg_info = (f"tensor '{tensor_name}': {tensors[tensor_name]}\n"
                    f"a type_spec of {handler._type_spec}")  # pylint: disable=protected-access.
        e.args = (dbg_info,) + e.args
        raise e.with_traceback(e.__traceback__)

    return pa.record_batch(arrays, schema=self._arrow_schema)


class _TypeHandler(abc.ABC):
  """Interface of a type handler that converts a tensor to arrow arrays.

  Note that a handler may convert a Tensor to multiple pa.Arrays. See
  arrow_fields().
  """

  __slots__ = ["_tensor_name", "_type_spec"]

  def __init__(self, tensor_name: str, type_spec: tf.TypeSpec):
    self._tensor_name = tensor_name
    self._type_spec = type_spec

  def convert(self, tensor: common_types.TensorAlike) -> List[pa.Array]:
    """Converts the given TensorAlike to pa.Arrays after validating its spec."""
    actual_spec = tf.type_spec_from_value(tensor)
    if not self._type_spec.is_compatible_with(actual_spec):
      raise TypeError("Expected {} but got {}".format(self._type_spec,
                                                      actual_spec))
    return self._convert_internal(tensor)

  @abc.abstractmethod
  def arrow_fields(self) -> List[pa.Field]:
    """Returns the name and type (in a pa.Field) of result pa.Arrays.

    Note that a Handler can convert a Tensor to multiple pa.Arrays. It must
    make sure _convert_internal() returns those Arrays of the types declared
    here, in the correct order.
    """

  @abc.abstractmethod
  def _convert_internal(self,
                        tensor: common_types.TensorAlike) -> List[pa.Array]:
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
  def can_handle(tensor_name: str, type_spec: tf.TypeSpec,
                 options: TensorsToRecordBatchConverter.Options) -> bool:
    """Returns `True` if the handler can handle the given `tf.TypeSpec`."""


class _DenseTensorHandler(_TypeHandler):
  """Handles Dense Tensors of known shape (except for the batch dim)."""

  __slots__ = ["_values_arrow_type", "_unbatched_shape"]

  def __init__(self, tensor_name: str, type_spec: tf.TypeSpec,
               options: TensorsToRecordBatchConverter.Options):
    del options
    super().__init__(tensor_name, type_spec)
    self._values_arrow_type = _tf_dtype_to_arrow_type(type_spec.dtype)
    self._unbatched_shape = type_spec.shape.as_list()[1:]

  def arrow_fields(self) -> List[pa.Field]:
    return [
        pa.field(self._tensor_name,
                 pa.large_list(_tf_dtype_to_arrow_type(self._type_spec.dtype)))
    ]

  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    result = schema_pb2.TensorRepresentation()
    result.dense_tensor.column_name = self._tensor_name
    for d in self._unbatched_shape:
      result.dense_tensor.shape.dim.add().size = d
    return result

  def _convert_internal(self,
                        tensor: common_types.TensorAlike) -> List[pa.Array]:
    assert isinstance(tensor, (tf.Tensor, np.ndarray)), type(tensor)
    values_np = np.asarray(tensor)
    shape = values_np.shape
    elements_per_list = np.prod(shape[1:], dtype=np.int64)
    if elements_per_list == 0:
      offsets = np.zeros(shape[0] + 1, dtype=np.int64)
    else:
      offsets = np.arange(
          0,
          elements_per_list * shape[0] + 1,
          elements_per_list,
          dtype=np.int64)
    values_np = np.reshape(values_np, -1)
    return [pa.LargeListArray.from_arrays(offsets, pa.array(
        values_np, self._values_arrow_type))]

  @staticmethod
  def can_handle(tensor_name: str, type_spec: tf.TypeSpec,
                 options: TensorsToRecordBatchConverter.Options) -> bool:
    del tensor_name
    del options
    if not isinstance(type_spec, tf.TensorSpec):
      return False
    if type_spec.dtype == tf.bool:
      return False
    # Can only handle batched tensor (at least 1-D).
    if type_spec.shape.rank is None or type_spec.shape.rank <= 0:
      return False
    shape = type_spec.shape.as_list()
    # Can only handle batched tensor (the batch size should be flexible).
    if shape[0] is not None:
      return False
    return all(d is not None for d in shape[1:])


class _VarLenSparseTensorHandler(_TypeHandler):
  """Handles 2-D var-len (ragged) sparse tensor."""

  __slots__ = ["_values_arrow_type"]

  def __init__(self, tensor_name: str, type_spec: tf.TypeSpec,
               options: TensorsToRecordBatchConverter.Options):
    del options
    super().__init__(tensor_name, type_spec)
    self._values_arrow_type = _tf_dtype_to_arrow_type(type_spec.dtype)

  def _convert_internal(self,
                        tensor: common_types.TensorAlike) -> List[pa.Array]:
    # Algorithm:
    # Assume:
    #   - the COO indices are sorted (partially checked below)
    #   - the SparseTensor is 2-D (checked in can_handle())
    #   - the SparseTensor is ragged
    # Then the first dim of those COO indices contains "parent indices":
    # parent_index[i] == j means i-th value belong to j-th sub list.
    # Then we have a C++ util to convert parent indices + values to a ListArray.
    #
    # Note that the resulting ListArray doesn't explicitly store the second
    # dense dimension. When it is converted back to SparseTensor with
    # tensor_adapter the second dense dimension is recovered as an upper bound
    # for second indices + 1. Therefore, if SparseTensor's second dense
    # dimension is not tight, then the composition
    # TensorAdapter(TensorsToRecordBatchConverter()) is not an identity.
    dense_shape = np.asarray(tensor.dense_shape)
    indices = np.asarray(tensor.indices)
    parent_indices = indices[:, 0]
    assert np.min(np.diff(parent_indices), initial=0) >= 0, (
        "The sparse indices must be sorted")
    return [
        array_util.MakeListArrayFromParentIndicesAndValues(
            dense_shape[0],
            pa.array(parent_indices, type=pa.int64()),
            pa.array(np.asarray(tensor.values), type=self._values_arrow_type),
            empty_list_as_null=False)
    ]

  def arrow_fields(self) -> List[pa.Field]:
    return [
        pa.field(self._tensor_name,
                 pa.large_list(_tf_dtype_to_arrow_type(self._type_spec.dtype)))
    ]

  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    result = schema_pb2.TensorRepresentation()
    result.varlen_sparse_tensor.column_name = self._tensor_name
    return result

  @staticmethod
  def can_handle(tensor_name: str, type_spec: tf.TypeSpec,
                 options: TensorsToRecordBatchConverter.Options) -> bool:
    if not isinstance(type_spec, tf.SparseTensorSpec):
      return False
    return (type_spec.shape.is_compatible_with([None, None]) and
            type_spec.dtype != tf.bool and
            tensor_name not in options.generic_sparse_tensor_names)


class _RaggedTensorHandler(_TypeHandler):
  """Handles ragged tensor."""

  __slots__ = ["_values_arrow_type", "_row_partition_dtype", "_unbatched_shape"]

  def __init__(self, tensor_name: str, type_spec: tf.TypeSpec,
               options: TensorsToRecordBatchConverter.Options):
    del options
    super().__init__(tensor_name, type_spec)

    # TODO(b/159717195): clean up protected-access
    self._values_arrow_type = _tf_dtype_to_arrow_type(type_spec._dtype)  # pylint: disable=protected-access
    self._row_partition_dtype = type_spec._row_splits_dtype  # pylint: disable=protected-access
    self._unbatched_shape = type_spec._shape.as_list()[1:]  # pylint: disable=protected-access

  def _convert_internal(self,
                        tensor: common_types.TensorAlike) -> List[pa.Array]:
    # Unnest all outer ragged dimensions keeping the offsets.
    nested_offsets = []
    while isinstance(tensor,
                     (tf.RaggedTensor, tf.compat.v1.ragged.RaggedTensorValue)):
      nested_offsets.append(np.asarray(tensor.row_splits))
      tensor = tensor.values

    # Calculate the number of inner uniform dimension elements per one first
    # ragged dimension element.
    inner_dimension_elements = np.prod(tensor.shape[1:], dtype=np.int64)

    result = pa.array(np.ravel(tensor), self._values_arrow_type)
    # Nest values. The innermost sequence of offsets must be adjusted by the
    # number of uniform dimension elements.
    nested_offsets_iter = reversed(nested_offsets)
    result = pa.LargeListArray.from_arrays(
        offsets=next(nested_offsets_iter) * inner_dimension_elements,
        values=result)
    for offsets in nested_offsets_iter:
      result = pa.LargeListArray.from_arrays(offsets=offsets, values=result)

    return [result]

  def arrow_fields(self) -> List[pa.Field]:
    # TODO(b/159717195): clean up protected-access
    arrow_type = _tf_dtype_to_arrow_type(self._type_spec._dtype)  # pylint: disable=protected-access
    for _ in range(self._type_spec._ragged_rank):  # pylint:disable=protected-access
      arrow_type = pa.large_list(arrow_type)
    return [
        pa.field(self._tensor_name, arrow_type)
    ]

  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    result = schema_pb2.TensorRepresentation()
    result.ragged_tensor.feature_path.step.append(self._tensor_name)
    row_partition_dtype = (
        schema_pb2.TensorRepresentation.RowPartitionDType.INT32
        if self._row_partition_dtype == tf.int32 else
        schema_pb2.TensorRepresentation.RowPartitionDType.INT64)
    result.ragged_tensor.row_partition_dtype = row_partition_dtype
    for dim in self._unbatched_shape:
      # Create uniform_row_length partitions only.
      if dim is not None:
        result.ragged_tensor.partition.append(
            schema_pb2.TensorRepresentation.RaggedTensor.Partition(
                uniform_row_length=dim))
    return result

  @staticmethod
  def can_handle(tensor_name: str, type_spec: tf.TypeSpec,
                 options: TensorsToRecordBatchConverter.Options) -> bool:
    del tensor_name
    del options
    if not isinstance(type_spec, tf.RaggedTensorSpec):
      return False
    # TODO(b/159717195): clean up protected-access
    # pylint:disable=protected-access
    if type_spec._ragged_rank < 1:
      # We don't support RaggedTensors that are not ragged. They are
      # essentially dense tensors and should be converted to them and be
      # handled by the DenseTensorHandler.
      return False
    shape = type_spec._shape.as_list()
    if (any(shape[1:type_spec._ragged_rank + 1]) or
        not all(shape[type_spec._ragged_rank + 1:])):
      # We only support inner uniform dimensions.
      return False
    return type_spec._dtype != tf.bool
    # pylint:enable=protected-access


class _SparseTensorHandler(_TypeHandler):
  """Handles generic SparseTensor.

  Note that this handler does not handle any 2-D / 1-D SparseTensor
  by default (they are handled by _VarLenSparseTensorHandler). However, not all
  2-D SparseTensors are VarLenSparseTensors, if you want to handle specific 2-D
  SparseTensor as a generic SparseTensor, add its name to
  options.generic_sparse_tensor_names.
  """

  __slots__ = ["_values_arrow_type", "_unbatched_shape",
               "_value_column_name", "_index_column_names"]

  def __init__(self, tensor_name: str, type_spec: tf.TypeSpec,
               options: TensorsToRecordBatchConverter.Options):
    super().__init__(tensor_name, type_spec)
    self._values_arrow_type = _tf_dtype_to_arrow_type(type_spec.dtype)
    self._unbatched_shape = type_spec.shape.as_list()[1:]
    self._value_column_name = (
        options.sparse_tensor_value_column_name_template.format(
            tensor_name=tensor_name))
    self._index_column_names = [
        options.sparse_tensor_index_column_name_template.format(
            tensor_name=tensor_name, index=i)
        for i in range(len(self._unbatched_shape))
    ]

  def _convert_internal(self,
                        tensor: common_types.TensorAlike) -> List[pa.Array]:
    # Transpose the indices array (and materialize the result in C-order)
    # because later we will use individual columns of the original indices.
    indices_np = (
        np.ascontiguousarray(
            np.transpose(np.asarray(tensor.indices)), dtype=np.int64))

    # the first column of indices identifies which row each sparse value belongs
    # to.
    parent_indices = pa.array(indices_np[0, :], type=pa.int64())
    num_rows = int(np.asarray(tensor.dense_shape)[0])

    result = [
        array_util.MakeListArrayFromParentIndicesAndValues(
            num_rows,
            parent_indices,
            pa.array(np.asarray(tensor.values), type=self._values_arrow_type),
            empty_list_as_null=False)
    ]

    for i in range(len(self._index_column_names)):
      result.append(
          array_util.MakeListArrayFromParentIndicesAndValues(
              num_rows,
              parent_indices,
              pa.array(indices_np[i + 1, :], type=pa.int64()),
              empty_list_as_null=False))

    return result

  def arrow_fields(self) -> List[pa.Field]:
    return ([
        pa.field(self._value_column_name, pa.large_list(
            self._values_arrow_type))
    ] + [
        pa.field(n, pa.large_list(pa.int64())) for n in self._index_column_names
    ])

  def tensor_representation(self) -> schema_pb2.TensorRepresentation:
    result = schema_pb2.TensorRepresentation()
    for d in self._unbatched_shape:
      result.sparse_tensor.dense_shape.dim.add().size = -1 if d is None else d
    result.sparse_tensor.value_column_name = self._value_column_name
    result.sparse_tensor.index_column_names.extend(self._index_column_names)
    return result

  @staticmethod
  def can_handle(tensor_name: str, type_spec: tf.TypeSpec,
                 options: TensorsToRecordBatchConverter.Options) -> bool:
    if not isinstance(type_spec, tf.SparseTensorSpec):
      return False
    if type_spec.shape.rank is None or type_spec.shape.rank <= 1:
      return False
    if (type_spec.shape.rank == 2 and
        tensor_name not in options.generic_sparse_tensor_names):
      return False
    return True


_ALL_HANDLERS_CLS = [
    _VarLenSparseTensorHandler, _RaggedTensorHandler, _DenseTensorHandler,
    _SparseTensorHandler,
]


def _tf_dtype_to_arrow_type(dtype: tf.DType):
  """Maps a tf Dtype to an Arrow type."""
  if dtype == tf.string:
    return pa.large_binary()
  elif dtype == tf.bool:
    raise TypeError("Unable to handle bool tensors -- consider casting it to a "
                    "tf.uint8")
  return pa.from_numpy_dtype(dtype.as_numpy_dtype)


def _make_handlers(
    type_specs: Dict[str, tf.TypeSpec],
    options: TensorsToRecordBatchConverter.Options
) -> List[Tuple[str, _TypeHandler]]:
  return [(tensor_name, _get_handler(tensor_name, type_spec, options))
          for tensor_name, type_spec in sorted(type_specs.items())]


def _get_handler(
    tensor_name: str, type_spec: tf.TypeSpec,
    options: TensorsToRecordBatchConverter.Options) -> _TypeHandler:
  """Returns a TypeHandler that can handle `type_spec`."""
  for handler_cls in _ALL_HANDLERS_CLS:
    if handler_cls.can_handle(tensor_name, type_spec, options):
      return handler_cls(tensor_name, type_spec, options)
  # We don't support tf.bool now because:
  #   - if converted to pa.bool(), TFDV does not know how to handle it.
  #   - if converted to pa.uint8() (or other integral types), we don't have
  #     a place to note it was previously a tf.bool so TensorAdapter can
  #     revert it as a tf.bool.
  raise ValueError(
      "No handler found for tensor {} of spec {}. "
      "Note that tensors with dtype == tf.bool cannot be handled in general -- "
      "consider casting them to tf.uint8."
      .format(tensor_name, type_spec))
