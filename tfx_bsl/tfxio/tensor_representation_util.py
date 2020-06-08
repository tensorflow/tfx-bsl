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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import List, Dict, Mapping, Optional, Text, Tuple, Union

from absl import logging
from tfx_bsl.arrow import path

from tensorflow_metadata.proto.v0 import schema_pb2


_DEFAULT_TENSOR_REPRESENTATION_GROUP = ""

_DISQUALIFYING_LIFECYCLE_STAGES = [
    schema_pb2.DEPRECATED,
    schema_pb2.PLANNED,
    schema_pb2.ALPHA,
    schema_pb2.DEBUG_ONLY
]

# The schema proto may not contain this field, which means the legacy logic
# does not apply.
_IS_LEGACY_SCHEMA = (
    "generate_legacy_feature_spec" in
    schema_pb2.Schema.DESCRIPTOR.fields_by_name)

_LEGACY_DEFAULT_VALUE_FOR_FEATURE_TYPE = {
    schema_pb2.BYTES:
        schema_pb2.TensorRepresentation.DefaultValue(bytes_value=b""),
    schema_pb2.INT:
        schema_pb2.TensorRepresentation.DefaultValue(int_value=-1),
    schema_pb2.FLOAT:
        schema_pb2.TensorRepresentation.DefaultValue(float_value=-1.0),
}


def _GetSparseTensorRepresentationUsedColumns(
    sparse_tensor_rep: schema_pb2.TensorRepresentation.SparseTensor
) -> List[path.ColumnPath]:
  result = [path.ColumnPath(c) for c in sparse_tensor_rep.index_column_names]
  if sparse_tensor_rep.HasField("value_column_name"):
    result.append(path.ColumnPath(sparse_tensor_rep.value_column_name))
  return result


_TENSOR_REPRESENTATION_KIND_TO_COLUMNS_GETTER = {
    "dense_tensor":
        lambda tr: [path.ColumnPath(tr.dense_tensor.column_name)],
    "varlen_sparse_tensor":
        lambda tr: [path.ColumnPath(tr.varlen_sparse_tensor.column_name)],
    "sparse_tensor":
        lambda tr: _GetSparseTensorRepresentationUsedColumns(tr.sparse_tensor),
    "ragged_tensor":
        lambda tr: [path.ColumnPath.from_proto(tr.ragged_tensor.feature_path)],
    None:
        lambda _: [],
}


def SetTensorRepresentationsInSchema(
    schema: schema_pb2.Schema,
    tensor_representations: Mapping[Text, schema_pb2.TensorRepresentation],
    tensor_representation_group_name: Text = _DEFAULT_TENSOR_REPRESENTATION_GROUP
    ) -> None:
  """Sets the TensorRepresentationGroup of the given name to the given value."""
  tensor_representation_map = schema.tensor_representation_group[
      tensor_representation_group_name].tensor_representation
  tensor_representation_map.clear()
  for k, v in tensor_representations.items():
    tensor_representation_map[k].CopyFrom(v)


def GetTensorRepresentationsFromSchema(
    schema: schema_pb2.Schema,
    tensor_representation_group_name: Text = _DEFAULT_TENSOR_REPRESENTATION_GROUP
) -> Optional[Dict[Text, schema_pb2.TensorRepresentation]]:
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
    schema: schema_pb2.Schema) -> Dict[Text, schema_pb2.TensorRepresentation]:
  """Infers TensorRepresentations from the schema's Features."""
  # TODO(zhuo): Add support for SparseFeature -> SparseTensor representation.
  if _ShouldUseLegacyLogic(schema):
    infer_func = _LegacyInferTensorRepresentationFromSchema
  else:
    infer_func = _InferTensorRepresentationFromSchema

  return infer_func(schema)


def GetSourceColumnsFromTensorRepresentation(
    tensor_representation: schema_pb2.TensorRepresentation
    ) -> List[path.ColumnPath]:
  """Returns columns required by the given TensorRepresentation."""

  return _TENSOR_REPRESENTATION_KIND_TO_COLUMNS_GETTER[
      tensor_representation.WhichOneof("kind")](tensor_representation)


def _ShouldIncludeFeature(
    feature: Union[schema_pb2.Feature, schema_pb2.SparseFeature]) -> bool:
  return not (feature.deprecated or
              feature.lifecycle_stage in _DISQUALIFYING_LIFECYCLE_STAGES)


def _InferTensorRepresentationFromSchema(
    schema: schema_pb2.Schema) -> Dict[Text, schema_pb2.TensorRepresentation]:
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
  sparse_tensor_repsentations, columns_remaining = (
      _InferSparseTensorRepresentationsFromSchema(schema))
  result.update(sparse_tensor_repsentations)
  for feature in columns_remaining:
    if not _ShouldIncludeFeature(feature):
      continue
    if feature.HasField("shape"):
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
      logging.info("Feature %s has no shape. Setting to VarLenSparseTensor.",
                   feature.name)
      result[feature.name] = schema_pb2.TensorRepresentation(
          varlen_sparse_tensor=schema_pb2.TensorRepresentation
          .VarLenSparseTensor(column_name=feature.name))

  return result


def _InferSparseTensorRepresentationsFromSchema(
    schema: schema_pb2.Schema
) -> Tuple[Dict[Text, schema_pb2.TensorRepresentation],
           List[schema_pb2.Feature]]:
  """Infers SparseTensor TensorRepresentation from the given schema."""
  columns_remaining = {f.name: f for f in schema.feature}
  sparse_tensor_representations = {}
  for sparse_feature in schema.sparse_feature:
    if not _ShouldIncludeFeature(sparse_feature):
      continue
    index_keys = [
        index_feature.name for index_feature in sparse_feature.index_feature]
    index_features = []
    for index_key in index_keys:
      try:
        index_features.append(columns_remaining.pop(index_key))
      except KeyError:
        raise ValueError(
            "sparse_feature {} referred to index feature {} which did not "
            "exist in the schema".format(sparse_feature.name, index_key))

    if len(index_features) != 1:
      raise ValueError(
          "sparse_feature {} had rank {} but currently only rank 1"
          " sparse features are supported".format(
              sparse_feature.name, len(index_features)))

    value_key = sparse_feature.value_feature.name
    try:
      columns_remaining.pop(value_key)
    except KeyError:
      raise ValueError(
          "sparse_feature {} referred to value feature {} which did not "
          "exist in the schema or was referred to as an index or value multiple "
          "times.".format(sparse_feature.name, value_key))

    if index_features[0].HasField("int_domain"):
      # Currently we only handle O-based INT index features whose minimum
      # domain value must be zero.
      if not index_features[0].int_domain.HasField("min"):
        raise ValueError("Cannot determine dense shape of sparse feature "
                         "{}. The minimum domain value of index feature {}"
                         " is not set."
                         .format(sparse_feature.name, index_keys[0]))
      if index_features[0].int_domain.min != 0:
        raise ValueError("Only 0-based index features are supported. Sparse "
                         "feature {} has index feature {} whose minimum "
                         "domain value is {}.".format(
                             sparse_feature.name, index_keys[0],
                             index_features[0].int_domain.min))

      if not index_features[0].int_domain.HasField("max"):
        raise ValueError("Cannot determine dense shape of sparse feature "
                         "{}. The maximum domain value of index feature {}"
                         " is not set."
                         .format(sparse_feature.name, index_keys[0]))
      shape = schema_pb2.FixedShape(dim=[
          schema_pb2.FixedShape.Dim(size=index_features[0].int_domain.max + 1)
      ])
    else:
      raise ValueError("Cannot determine dense shape of sparse feature {}."
                       " The index feature {} had no int_domain set.".format(
                           sparse_feature.name, index_keys[0]))

    sparse_tensor_representations[sparse_feature.name] = (
        schema_pb2.TensorRepresentation(
            sparse_tensor=schema_pb2.TensorRepresentation.SparseTensor(
                dense_shape=shape,
                index_column_names=index_keys,
                value_column_name=value_key)))

  return sparse_tensor_representations, list(columns_remaining.values())


def _ShouldUseLegacyLogic(schema: schema_pb2.Schema) -> bool:
  if _IS_LEGACY_SCHEMA:
    return schema.generate_legacy_feature_spec
  return False


def _LegacyInferTensorRepresentationFromSchema(
    schema: schema_pb2.Schema) -> Dict[Text, schema_pb2.TensorRepresentation]:
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
    if (feature.value_count.min == feature.value_count.max
        and feature.value_count.min == 1):
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

    elif (feature.value_count.min == feature.value_count.max
          and feature.value_count.min > 1):
      # Case 2: value_count.min == value_count.max > 1.  Infer a DenseTensor
      # with rank 1 and a default value.
      shape = schema_pb2.FixedShape(
          dim=[schema_pb2.FixedShape.Dim(size=feature.value_count.min)])
      logging.info(
          "Feature %s has value_count.min == value_count.max > 1. Setting to "
          "DenseTensor.", feature.name)
      result[feature.name] = schema_pb2.TensorRepresentation(
          dense_tensor=schema_pb2.TensorRepresentation.DenseTensor(
              column_name=feature.name, shape=shape,
              default_value=_LegacyInferDefaultValue(feature)))

    else:
      # Case 3: Either value_count.min != value_count.max or
      # value_count.min == value_count.max == 0.  Infer a VarLenSparseTensor.
      logging.info(
          "Feature %s has value_count.min != value_count.max or "
          "value_count.min == value_count.max == 0. "
          "Setting to VarLenSparseTensor.", feature.name)
      result[feature.name] = schema_pb2.TensorRepresentation(
          varlen_sparse_tensor=
          schema_pb2.TensorRepresentation.VarLenSparseTensor(
              column_name=feature.name))

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
        "Feature %s has min_fraction = 1 (%s). Not setting defalut value.",
        feature_proto.name, feature_proto.presence)
    return None
