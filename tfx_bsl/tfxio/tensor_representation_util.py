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
"""TensorRepresentation utilities."""

from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from absl import logging
import numpy as np
import tensorflow as tf
from tfx_bsl.arrow import path
from tfx_bsl.types import common_types

from tensorflow_metadata.proto.v0 import path_pb2
from tensorflow_metadata.proto.v0 import schema_pb2

_DEFAULT_TENSOR_REPRESENTATION_GROUP = ""

_DISQUALIFYING_LIFECYCLE_STAGES = (
    schema_pb2.DEPRECATED, schema_pb2.DISABLED, schema_pb2.PLANNED,
    schema_pb2.ALPHA, schema_pb2.DEBUG_ONLY, schema_pb2.VALIDATION_DERIVED,
)

# The schema proto may not contain this field, which means the legacy logic
# does not apply.
_IS_LEGACY_SCHEMA = ("generate_legacy_feature_spec" in
                     schema_pb2.Schema.DESCRIPTOR.fields_by_name)

_LEGACY_DEFAULT_VALUE_FOR_FEATURE_TYPE = {
    schema_pb2.BYTES:
        schema_pb2.TensorRepresentation.DefaultValue(bytes_value=b""),
    schema_pb2.INT:
        schema_pb2.TensorRepresentation.DefaultValue(int_value=-1),
    schema_pb2.FLOAT:
        schema_pb2.TensorRepresentation.DefaultValue(float_value=-1.0),
}

_FEATURE_TYPE_TO_TF_TYPE = {
    schema_pb2.BYTES: tf.string,
    schema_pb2.INT: tf.int64,
    schema_pb2.FLOAT: tf.float32,
}

_DEFAULT_VALUE_KIND_TO_FEATURE_TYPE = {
    "bytes_value": schema_pb2.BYTES,
    "int_value": schema_pb2.INT,
    "float_value": schema_pb2.FLOAT,
}


def _GetSparseTensorRepresentationUsedColumns(
    sparse_tensor_rep: schema_pb2.TensorRepresentation.SparseTensor
) -> List[path.ColumnPath]:
  result = [path.ColumnPath(c) for c in sparse_tensor_rep.index_column_names]
  if sparse_tensor_rep.HasField("value_column_name"):
    result.append(path.ColumnPath(sparse_tensor_rep.value_column_name))
  return result


def _GetRaggedTensorRepresentationUsedColumns(
    ragged_tensor_rep: schema_pb2.TensorRepresentation.RaggedTensor
) -> List[path.ColumnPath]:
  """Returns a list of ColumnPaths used by the Ragged TensorRepresentation."""
  value_column_path = path.ColumnPath.from_proto(ragged_tensor_rep.feature_path)
  result = [value_column_path]
  for partition in ragged_tensor_rep.partition:
    if partition.HasField("row_length"):
      result.append(value_column_path.parent().child(partition.row_length))
  return result


_TENSOR_REPRESENTATION_KIND_TO_COLUMNS_GETTER = {
    "dense_tensor":
        lambda tr: [path.ColumnPath(tr.dense_tensor.column_name)],
    "varlen_sparse_tensor":
        lambda tr: [path.ColumnPath(tr.varlen_sparse_tensor.column_name)],
    "sparse_tensor":
        lambda tr: _GetSparseTensorRepresentationUsedColumns(tr.sparse_tensor),
    "ragged_tensor":
        lambda tr: _GetRaggedTensorRepresentationUsedColumns(tr.ragged_tensor),
    None:
        lambda _: [],
}

_TENSOR_REPRESENTATION_KIND_TO_VALUE_COLUMN_GETTER = {
    "dense_tensor":
        lambda tr: path.ColumnPath(tr.dense_tensor.column_name),
    "varlen_sparse_tensor":
        lambda tr: path.ColumnPath(tr.varlen_sparse_tensor.column_name),
    "sparse_tensor":
        lambda tr: path.ColumnPath(tr.sparse_tensor.value_column_name),
    "ragged_tensor":
        lambda tr: path.ColumnPath.from_proto(tr.ragged_tensor.feature_path)
}


def SetTensorRepresentationsInSchema(
    schema: schema_pb2.Schema,
    tensor_representations: Mapping[str, schema_pb2.TensorRepresentation],
    tensor_representation_group_name: str = _DEFAULT_TENSOR_REPRESENTATION_GROUP
) -> None:
  """Sets the TensorRepresentationGroup of the given name to the given value."""
  tensor_representation_map = schema.tensor_representation_group[
      tensor_representation_group_name].tensor_representation
  tensor_representation_map.clear()
  for k, v in tensor_representations.items():
    tensor_representation_map[k].CopyFrom(v)


def GetTensorRepresentationsFromSchema(
    schema: schema_pb2.Schema,
    tensor_representation_group_name: str = _DEFAULT_TENSOR_REPRESENTATION_GROUP
) -> Optional[Dict[str, schema_pb2.TensorRepresentation]]:
  """Gets a TensorRepresentationGroup as a dict<tensor_name,rep> from schema.

  If the group name is provided, look it up in the schema, otherwise, look for
  the default group.

  Args:
    schema: a schema_pb2.Schema.
    tensor_representation_group_name: (optional) the name of the group to look
      for. If not provided, look for the default name.

  Returns:
    None if not found. Otherwise a dict with tensor names being keys and
    TensorRepresentation as values.
  """
  group = schema.tensor_representation_group.get(
      tensor_representation_group_name)
  if group is None:
    return None
  return dict(group.tensor_representation)


def InferTensorRepresentationsFromSchema(
    schema: schema_pb2.Schema) -> Dict[str, schema_pb2.TensorRepresentation]:
  """Infers TensorRepresentations from the schema's Features."""
  if _ShouldUseLegacyLogic(schema):
    infer_func = _LegacyInferTensorRepresentationFromSchema
  else:
    infer_func = _InferTensorRepresentationFromSchema

  return infer_func(schema)


def InferTensorRepresentationsFromMixedSchema(
    schema: schema_pb2.Schema) -> Dict[str, schema_pb2.TensorRepresentation]:
  """Infers TensorRepresentations from schema that has Features and TRs."""
  tensor_representations = GetTensorRepresentationsFromSchema(schema)
  inferred_tensor_representations = InferTensorRepresentationsFromSchema(schema)
  if tensor_representations is None:
    return inferred_tensor_representations
  # Only keep inferred TRs that do not represent source columns. Existing TRs
  # are preferred over the inferred in case of name collisions.
  source_columns = set()
  for tensor_representation in tensor_representations.values():
    source_columns.update(
        str(path) for path in GetSourceColumnsFromTensorRepresentation(
            tensor_representation))
  for name, tensor_representation in inferred_tensor_representations.items():
    if name in source_columns:
      # This feature is already used in the explicitly provided TRs.
      continue
    if name in tensor_representations:
      logging.warning(
          "Feature name %s conflicts with tensor representation name in the "
          "same schema that does not use the feature as its source column. "
          "Ignoring the feature and using the tensor representation.", name)
    else:
      tensor_representations[name] = tensor_representation
  return tensor_representations


def GetSourceColumnsFromTensorRepresentation(
    tensor_representation: schema_pb2.TensorRepresentation
) -> List[path.ColumnPath]:
  """Returns columns required by the given TensorRepresentation."""

  return _TENSOR_REPRESENTATION_KIND_TO_COLUMNS_GETTER[
      tensor_representation.WhichOneof("kind")](
          tensor_representation)


def GetSourceValueColumnFromTensorRepresentation(
    tensor_representation: schema_pb2.TensorRepresentation) -> path.ColumnPath:
  """Returns the column name of value columns from the TensorRepresentation.

  Each tensor representation will have one or more value column. A value column
  is a column that contributes to the values of a (composite) tensor. Certain
  composite tensor may consists of data from multiple columns, with one
  providing the values, others providing structural information.

  Args:
    tensor_representation: The tensor representation that contains tensor
      construction information.

  Raises:
    KeyError: if the tensor representation's "kind" is invalid. Valid "kinds"
      are dense_tensor, varlen_sparse_tensor, sparse_tensor, or ragged_tensor.
  """
  return _TENSOR_REPRESENTATION_KIND_TO_VALUE_COLUMN_GETTER[
      tensor_representation.WhichOneof("kind")](
          tensor_representation)


def CreateTfExampleParserConfig(
    tensor_representation: schema_pb2.TensorRepresentation,
    feature_type: schema_pb2.FeatureType) -> common_types.FeatureSpecType:
  """Creates a Feature Configuration that is used for tf.io.parse_example.

  Args:
    tensor_representation: The tensor representation to convert to a Feature.
    feature_type: The schema_pb2.FeatureType of the given feature. The supported
      types are listed in _FEATURE_TYPE_TO_TF_TYPE.

  Returns:
    Either a `tf.io.FixedLenFeature`, `tf.io.VarLenFeature`,
      `tf.io.SparseFeature`, or `tf.io.RaggedFeature`.

  Raises:
    ValueError: If the tensor_representation cannot be converted to a Feature.
  """
  value_dtype = _FEATURE_TYPE_TO_TF_TYPE.get(feature_type, None)
  if value_dtype is None:
    raise ValueError(
        "The feature_type: {} is not supported.".format(feature_type))

  tensor_representation_kind = tensor_representation.WhichOneof("kind")
  if tensor_representation_kind == "dense_tensor":
    dense_tensor_rep = tensor_representation.dense_tensor
    shape = _GetDimsFromFixedShape(dense_tensor_rep.shape)
    default_value = None
    if dense_tensor_rep.HasField("default_value"):
      default_value = _GetDefaultValuesList(shape, feature_type,
                                            dense_tensor_rep.default_value)
    return tf.io.FixedLenFeature(
        shape=shape, dtype=value_dtype, default_value=default_value)
  elif tensor_representation_kind == "varlen_sparse_tensor":
    return tf.io.VarLenFeature(dtype=value_dtype)
  elif tensor_representation_kind == "sparse_tensor":
    sparse_tensor_rep = tensor_representation.sparse_tensor
    return tf.io.SparseFeature(
        index_key=list(sparse_tensor_rep.index_column_names),
        value_key=sparse_tensor_rep.value_column_name,
        dtype=value_dtype,
        size=_GetDimsFromFixedShape(sparse_tensor_rep.dense_shape),
        already_sorted=getattr(sparse_tensor_rep, "already_sorted", False))
  elif tensor_representation_kind == "ragged_tensor":
    ragged_tensor_rep = tensor_representation.ragged_tensor
    if (ragged_tensor_rep.row_partition_dtype ==
        schema_pb2.TensorRepresentation.RowPartitionDType.INT32):
      row_splits_dtype = tf.int32
    else:
      row_splits_dtype = tf.int64

    partitions = []
    if len(ragged_tensor_rep.feature_path.step) > 1:
      raise ValueError(
          "Parsing spec from a RaggedTensor with multiple steps in "
          "feature_path is not implemented.")
    if not ragged_tensor_rep.feature_path.step:
      raise ValueError("RaggedTensor representation with empty feature_path.")
    for partition in ragged_tensor_rep.partition:
      if partition.HasField("uniform_row_length"):
        partitions.append(
            tf.io.RaggedFeature.UniformRowLength(  # pytype:disable=attribute-error
                partition.uniform_row_length))
      elif partition.HasField("row_length"):
        partitions.append(
            tf.io.RaggedFeature.RowLengths(  # pytype:disable=attribute-error
                partition.row_length))
      else:
        raise NotImplementedError(
            "RaggedTensor partition type not implemented: {}.".format(
                partition.WhichOneof("kind")))
    return tf.io.RaggedFeature(
        dtype=value_dtype,
        value_key=ragged_tensor_rep.feature_path.step[0],
        partitions=partitions,
        row_splits_dtype=row_splits_dtype)
  else:
    raise NotImplementedError(
        "TensorRepresentation: {} is not supported.".format(
            tensor_representation_kind))


def _GetPrimitiveFeatureTypes(
    features: List["schema_pb2.Feature"]
) -> Dict[path.ColumnPath, "schema_pb2.FeatureType"]:
  """Recursively extracts types of all primitive features in the given list."""
  result = {}
  for feature in features:
    if feature.type == schema_pb2.STRUCT:
      if feature.struct_domain.sparse_feature:
        raise ValueError(
            "Struct features with sparse domains are not supported.")
      for child_path, child_type in _GetPrimitiveFeatureTypes(
          list(feature.struct_domain.feature)).items():
        full_path = path.ColumnPath((feature.name,) + child_path.steps())
        result[full_path] = child_type
    else:
      result[path.ColumnPath((feature.name,))] = feature.type
  return result


def CreateTfSequenceExampleParserConfig(
    schema: schema_pb2.Schema
) -> Tuple[Dict[str, common_types.FeatureSpecType], Dict[
    str, common_types.FeatureSpecType]]:
  """Creates feature specs that are used for tf.io.parse_sequence_example.

  Args:
    schema: A TFMD Schema describing the features that are going to be parsed.
      The parsing config is generated from `TensorRepresentation`s and therefore
      the schema is expected to have them for all parsed features. Sequence
      features are expected to be described in a `StructDomain` of a top-level
      `STRUCT` sequence feature.

  Returns:
    A tuple of context and sequence feature spec dictionaries that map tensor
    names to either `tf.io.FixedLenFeature`, `tf.io.VarLenFeature`,
    `tf.io.SparseFeature`, or `tf.io.RaggedFeature`.

  Raises:
    ValueError: If the schema does not contain tensor representations or if the
      top-level `STRUCT` feature is not valid.
  """
  tensor_representations = GetTensorRepresentationsFromSchema(schema)
  if tensor_representations is None:
    raise ValueError("TensorRepresentations are required.")
  column_name_to_type = _GetPrimitiveFeatureTypes(list(schema.feature))
  context_features = {}
  sequence_features = {}
  for name, tensor_representation in tensor_representations.items():
    value_column = GetSourceValueColumnFromTensorRepresentation(
        tensor_representation)
    value_type = column_name_to_type[value_column]
    if len(value_column) == 1:
      context_features[name] = CreateTfExampleParserConfig(
          tensor_representation, value_type)
    elif len(value_column) == 2:
      # Only ragged tensor representation supports paths with multiple steps.
      assert tensor_representation.WhichOneof("kind") == "ragged_tensor", (
          tensor_representation)
      sequence_tensor_representation = schema_pb2.TensorRepresentation()
      sequence_tensor_representation.CopyFrom(tensor_representation)
      # Now that we know that it's a sequence feature, we put it in the proper
      # dictionary and pop the sequence feature path.
      sequence_tensor_representation.ragged_tensor.feature_path.step.pop(0)
      sequence_features[name] = CreateTfExampleParserConfig(
          sequence_tensor_representation, value_type)
    else:
      raise ValueError(
          f"Only primitive or struct of primitive features are supported, "
          f"got {tensor_representation}")
  return context_features, sequence_features


def _ShouldIncludeFeature(
    feature: Union[schema_pb2.Feature, schema_pb2.SparseFeature]) -> bool:
  return not (feature.deprecated or
              feature.lifecycle_stage in _DISQUALIFYING_LIFECYCLE_STAGES)


def _InferTensorRepresentationsFromStruct(
    feature: schema_pb2.Feature) -> Dict[str, schema_pb2.TensorRepresentation]:
  """Infers RaggedTensor TensorRepresentations from the given STRUCT feature."""
  if feature.type != schema_pb2.FeatureType.STRUCT:
    raise ValueError(
        f"Expected STRUCT, got {feature.type} for feature {feature.name}.")
  if feature.struct_domain.sparse_feature:
    raise NotImplementedError(
        f"Got sparse features in struct domain of {feature.name}. Struct "
        "features with sparse domains are not supported.")
  result = {}
  struct_path = path.ColumnPath((feature.name,))
  for child_feature in feature.struct_domain.feature:
    if child_feature.type == schema_pb2.FeatureType.STRUCT:
      raise NotImplementedError(
          f"Got STRUCT feature {child_feature.name} in struct domain of "
          f"{feature.name}. STRUCT features with multiple levels of nestedness "
          "are not supported.")
    child_path = struct_path.child(child_feature.name)
    result[str(child_path)] = schema_pb2.TensorRepresentation(
        ragged_tensor=schema_pb2.TensorRepresentation.RaggedTensor(
            feature_path=child_path.to_proto()))
  return result


def _MakeVarLenTensorRepresentation(
    name: str, ragged: bool) -> schema_pb2.TensorRepresentation:
  """Constructs TensorRepresentation for a variable length feature."""
  if ragged:
    return schema_pb2.TensorRepresentation(
        ragged_tensor=schema_pb2.TensorRepresentation.RaggedTensor(
            feature_path=path_pb2.Path(step=[name])))
  return schema_pb2.TensorRepresentation(
      varlen_sparse_tensor=schema_pb2.TensorRepresentation.VarLenSparseTensor(
          column_name=name))


def _InferTensorRepresentationFromSchema(
    schema: schema_pb2.Schema) -> Dict[str, schema_pb2.TensorRepresentation]:
  """Translate a Feature proto into a TensorRepresentation proto.

  We apply the following rules:
    1. If the feature has a fixed shape (set through Feature.shape field),
       then the feature must always be present (
       Feature.presence.min_fraction == 1.0), and a DenseTensor representation
       will be produced for it.
    2. Otherwise, a VarLenSparseTensor representation will be produced for it.

  Args:
    schema: a schema_pb2.Schema.

  Returns:
    A Dict mapping tensor names to their TensorRepresentations.

  Raises:
    ValueError: if the feature has a fixed shape but is not always present.
  """
  result = {}
  columns_remaining = {f.name: f for f in schema.feature}

  sparse_tensor_repsentations, columns_remaining = (
      _InferSparseTensorRepresentationsFromSchema(schema, columns_remaining))
  result.update(sparse_tensor_repsentations)

  for feature in columns_remaining.values():
    if not _ShouldIncludeFeature(feature):
      continue
    if feature.type == schema_pb2.FeatureType.STRUCT:
      result.update(_InferTensorRepresentationsFromStruct(feature))
    elif feature.HasField("shape"):
      if feature.presence.min_fraction != 1:
        raise ValueError(
            "Feature {} had shape {} set but min_fraction {} != 1.  Use"
            " value_count not shape field when min_fraction != 1.".format(
                feature.name, feature.shape, feature.presence.min_fraction))
      logging.info("Feature %s has a shape %s. Setting to DenseTensor.",
                   feature.name, feature.shape)
      result[feature.name] = schema_pb2.TensorRepresentation(
          dense_tensor=schema_pb2.TensorRepresentation.DenseTensor(
              column_name=feature.name, shape=feature.shape))
    else:
      representation = _MakeVarLenTensorRepresentation(
          feature.name, schema.represent_variable_length_as_ragged)
      logging.info("Feature %s has no shape. Setting to %s.", feature.name,
                   representation.WhichOneof("kind"))
      result[feature.name] = representation

  return result


def _InferSparseTensorRepresentationsFromSchema(
    schema: schema_pb2.Schema, columns_remaining: Dict[str, schema_pb2.Feature]
) -> Tuple[Dict[str, schema_pb2.TensorRepresentation], Dict[
    str, schema_pb2.Feature]]:
  """Infers SparseTensor TensorRepresentation from the given schema."""
  sparse_tensor_representations = {}
  for sparse_feature in schema.sparse_feature:
    if not _ShouldIncludeFeature(sparse_feature):
      continue
    index_keys = [
        index_feature.name for index_feature in sparse_feature.index_feature
    ]
    index_features = []
    for index_key in index_keys:
      try:
        index_features.append(columns_remaining.pop(index_key))
      except KeyError:
        raise ValueError(
            "sparse_feature {} referred to index feature {} which did not "
            "exist in the schema".format(sparse_feature.name, index_key))

    value_key = sparse_feature.value_feature.name
    try:
      columns_remaining.pop(value_key)
    except KeyError:
      raise ValueError(
          "sparse_feature {} referred to value feature {} which did not "
          "exist in the schema or was referred to as an index or value multiple "
          "times.".format(sparse_feature.name, value_key))

    shape = schema_pb2.FixedShape()
    for index_feature, index_key in zip(index_features, index_keys):
      if index_feature.HasField("int_domain"):
        # Currently we only handle O-based INT index features whose minimum
        # domain value must be zero.
        if not index_feature.int_domain.HasField("min"):
          raise ValueError("Cannot determine dense shape of sparse feature "
                           "{}. The minimum domain value of index feature {}"
                           " is not set.".format(sparse_feature.name,
                                                 index_key))
        if index_feature.int_domain.min != 0:
          raise ValueError("Only 0-based index features are supported. Sparse "
                           "feature {} has index feature {} whose minimum "
                           "domain value is {}.".format(
                               sparse_feature.name, index_key,
                               index_feature.int_domain.min))

        if not index_feature.int_domain.HasField("max"):
          raise ValueError("Cannot determine dense shape of sparse feature "
                           "{}. The maximum domain value of index feature {}"
                           " is not set.".format(sparse_feature.name,
                                                 index_key))
        shape.dim.add(size=index_feature.int_domain.max + 1)
      else:
        # Use -1 to denote unknown dimension size.
        shape.dim.add(size=-1)
    # Explicitly set `already_sorted` only when it's set to true in the
    # `SparseFeature` for backwards compatiblity.
    sparse_tensor_representations[sparse_feature.name] = (
        schema_pb2.TensorRepresentation(
            sparse_tensor=schema_pb2.TensorRepresentation.SparseTensor(
                dense_shape=shape,
                index_column_names=index_keys,
                value_column_name=value_key,
                already_sorted=sparse_feature.is_sorted or None)))

  return sparse_tensor_representations, columns_remaining


def _ShouldUseLegacyLogic(schema: schema_pb2.Schema) -> bool:
  if _IS_LEGACY_SCHEMA:
    return schema.generate_legacy_feature_spec
  return False


def _LegacyInferTensorRepresentationFromSchema(
    schema: schema_pb2.Schema) -> Dict[str, schema_pb2.TensorRepresentation]:
  """Translate a Feature proto into a TensorRepresentation proto.

  This function applies heuristics to deduce the shape and other information
  from a FeatureProto.  The FeatureProto contains information about the feature
  in an ExampleProto, but the feature spec proto also requires enough
  information to parse the feature into a tensor.  We apply the following rules:

    1. The shape and representation of the column are determined by the
       following rules:
         * if the value_count.min and value_count.max are both 1 then the shape
           is scalar and the representation is fixed length.
         * If value_count.min and value_count.max are equal but greater than 1,
           then the shape is a vector whose length is value_count.max and the
           representation is fixed length.
         * If value_count.min and value_count.max are equal and are less than 1,
           then the shape is a vector of unknown length and the representation
           is variable length.
         * If value_count.min and value_count.max are not equal then
           the shape is a vector of unknown length and the representation is
           variable length.

    2. If the feature is always present or is variable length (based on the
        above rule), no default value is set but if the feature is not always
        present and is fixed length, then a canonical default value is chosen
        based on _LEGACY_DEFAULT_VALUE_FOR_FEATURE_TYPE.

    3. Features that are deprecated are completely ignored and removed.

  Args:
    schema: A Schema proto.

  Returns:
    A Dict mapping tensor names to their TensorRepresentations.

  Raises:
    ValueError: If the feature's type is not supported or the schema is invalid.
  """
  result = {}
  for feature in schema.feature:
    if not _ShouldIncludeFeature(feature):
      continue
    if feature.type == schema_pb2.FeatureType.STRUCT:
      result.update(_InferTensorRepresentationsFromStruct(feature))
      continue
    # Infer canonical tensorflow dtype.
    if feature.value_count.min < 0:
      raise ValueError(
          "Feature {} has value_count.min < 0 (value was {}).".format(
              feature.name, feature.value_count.min))

    if feature.value_count.max < 0:
      raise ValueError(
          "Feature {} has value_count.max < 0 (value was {}).".format(
              feature.name, feature.value_count.max))

    # Use heuristics to infer the shape and representation.
    if (feature.value_count.min == feature.value_count.max and
        feature.value_count.min == 1):
      # Case 1: value_count.min == value_count.max == 1.  Infer a DenseTensor
      # with rank 0 and a default value.
      logging.info(
          "Feature %s has value_count.min == value_count.max == 1. Setting to "
          "DenseTensor.", feature.name)
      result[feature.name] = schema_pb2.TensorRepresentation(
          dense_tensor=schema_pb2.TensorRepresentation.DenseTensor(
              column_name=feature.name,
              shape=schema_pb2.FixedShape(),
              default_value=_LegacyInferDefaultValue(feature)))

    elif (feature.value_count.min == feature.value_count.max and
          feature.value_count.min > 1):
      # Case 2: value_count.min == value_count.max > 1.  Infer a DenseTensor
      # with rank 1 and a default value.
      shape = schema_pb2.FixedShape(
          dim=[schema_pb2.FixedShape.Dim(size=feature.value_count.min)])
      logging.info(
          "Feature %s has value_count.min == value_count.max > 1. Setting to "
          "DenseTensor.", feature.name)
      result[feature.name] = schema_pb2.TensorRepresentation(
          dense_tensor=schema_pb2.TensorRepresentation.DenseTensor(
              column_name=feature.name,
              shape=shape,
              default_value=_LegacyInferDefaultValue(feature)))

    else:
      # Case 3: Either value_count.min != value_count.max or
      # value_count.min == value_count.max == 0.  Infer a VarLenSparseTensor.
      representation = _MakeVarLenTensorRepresentation(
          feature.name, schema.represent_variable_length_as_ragged)
      logging.info(
          "Feature %s has value_count.min != value_count.max or "
          "value_count.min == value_count.max == 0. "
          "Setting to %s.", feature.name, representation.WhichOneof("kind"))
      result[feature.name] = representation

  return result


def _LegacyInferDefaultValue(
    feature_proto: schema_pb2.Feature
) -> Optional[schema_pb2.TensorRepresentation.DefaultValue]:
  """Inferrs a default value for a feature."""
  if feature_proto.presence.min_fraction < 1:
    default_value = _LEGACY_DEFAULT_VALUE_FOR_FEATURE_TYPE.get(
        feature_proto.type)
    if default_value is None:
      raise ValueError("Unable to infer a default value for feature {}".format(
          feature_proto))
    return default_value
  else:
    logging.info(
        "Feature %s has min_fraction = 1 (%s). Not setting default value.",
        feature_proto.name, feature_proto.presence)
    return None


def _GetDimsFromFixedShape(shape: schema_pb2.FixedShape) -> List[int]:
  """Returns a list of dimensions, given a schema_pb2.FixedShape.

  Args:
    shape: A schema_pb2.FixedShape.
  """
  return [dim.size for dim in shape.dim]


def _GetDefaultValuesList(
    unbatched_shape: List[int], feature_type: schema_pb2.FeatureType,
    default_value_proto: schema_pb2.TensorRepresentation.DefaultValue
) -> List[Union[int, float, bytes]]:
  """Returns a List filled with the default value given in the proto.

  Args:
    unbatched_shape: The shape of the tensor to fill.
    feature_type: The expected type of the default_value.
    default_value_proto: The DefaultValue proto that holds the default_value.

  Raises:
    ValueError: if the default_value is incompatible with feature_type.
  """
  kind = default_value_proto.WhichOneof("kind")
  default_value = getattr(default_value_proto, kind)
  expected_feature_type = _DEFAULT_VALUE_KIND_TO_FEATURE_TYPE.get(kind, None)
  if feature_type != expected_feature_type:
    raise ValueError(
        "FeatureType: {} is incompatible with default_value: {}".format(
            schema_pb2.FeatureType.Name(feature_type), default_value))
  size = int(np.prod(unbatched_shape, initial=1))

  return [default_value] * size


def ProjectTensorRepresentationsInSchema(
    schema: schema_pb2.Schema,
    tensor_names: Iterable[str]) -> schema_pb2.Schema:
  """Returns a projection of schema by the given tensor names.

  Tries to extract TensorRpresentations from the schema and infers them in case
  there's none. The schema is then projected to have the TensorRepresentations
  and source feature columns of tensors that are present in `tensor_names`.

  Args:
    schema: A TFMD Schema to be projected.
    tensor_names: Names of tensors that schema must be projected on.

  Returns:
    A schema that contains a subset of TensorRepresentations and features in
    `schema` that is a set of source columns for the given tensors.

  Raises:
    ValueError: if `schema` doesn't contain any of the given `tensor_names` or
    TensorRepresentations' source columns are not present in `schema` features.
  """
  tensor_representations = GetTensorRepresentationsFromSchema(schema)
  if tensor_representations is None:
    tensor_representations = InferTensorRepresentationsFromSchema(schema)
  tensor_names = set(tensor_names)
  if not tensor_names.issubset(tensor_representations):
    raise ValueError(
        "Unable to project {} because they were not in the original "
        "or inferred TensorRepresentations.".format(
            tensor_names - tensor_representations.keys()))
  paths = set()
  for tensor_name in tensor_names:
    paths.update(
        GetSourceColumnsFromTensorRepresentation(
            tensor_representations[tensor_name]))
  result = schema_pb2.Schema()

  for feature in schema.feature:
    feature_path = path.ColumnPath(feature.name)
    if feature_path in paths:
      paths.remove(feature_path)
      result.feature.add().CopyFrom(feature)

  if paths:
    raise ValueError("TensorRepresentations source columns {} are not present "
                     "in the schema.".format(paths))

  SetTensorRepresentationsInSchema(
      result,
      {k: v for k, v in tensor_representations.items() if k in tensor_names})

  return result


def _GetSourceColumnsFromFeature(
    feature: schema_pb2.Feature) -> List[path.ColumnPath]:
  """Extracts all Feature paths from a potentially nested Feature."""
  if feature.type == schema_pb2.FeatureType.STRUCT:
    result = []
    for child_feature in feature.struct_domain.feature:
      result.extend(
          path.ColumnPath([feature.name] + list(child_path.steps()))
          for child_path in _GetSourceColumnsFromFeature(child_feature))
    return result
  else:
    return [path.ColumnPath(feature.name)]


def ValidateTensorRepresentationsInSchema(
    schema: schema_pb2.Schema,
    tensor_representation_group_name: str = _DEFAULT_TENSOR_REPRESENTATION_GROUP
):
  """Checks that TensorRepresentations refer all schema features at least once.

  Args:
    schema: A TFMD Schema proto.
    tensor_representation_group_name: (optional) the name of the group to look
      for. If not provided, looks for the default name.

  Raises:
    ValueError: If either of the following is true
      * there's no TensorRepresentationGroup with the given name;
      * TensorRepresentations refer to a feature that is not in the schema;
      * feature exists in the schema, but is not referred to by any
        TensorRepresentation.
  """
  tensor_representations = GetTensorRepresentationsFromSchema(
      schema, tensor_representation_group_name)
  if tensor_representations is None:
    raise ValueError(
        "TensorRepresentations are not found in the schema. Did you specify "
        "correct group name?")
  source_features = set()
  for representation in tensor_representations.values():
    source_features.update(
        GetSourceColumnsFromTensorRepresentation(representation))
  all_features = set()
  for feature in schema.feature:
    all_features.update(_GetSourceColumnsFromFeature(feature))
  source_not_in_schema = source_features - all_features
  if source_not_in_schema:
    raise ValueError(
        f"Features referred in TensorRepresentations but not found in the "
        f"schema: {source_not_in_schema}")
  in_schema_not_source = all_features - source_features
  if in_schema_not_source:
    raise ValueError(f"Features present in the schema but not referred in any "
                     f"TensorRepresentation: {in_schema_not_source}")
