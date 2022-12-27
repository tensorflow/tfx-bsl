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
"""TFGraphRecordDecoder and utilities to save and load them."""

import abc
from typing import Dict, Optional, Union

import tensorflow as tf

from tensorflow.python.framework import composite_tensor  # pylint: disable=g-direct-tensorflow-import


TensorAlike = Union[tf.Tensor, composite_tensor.CompositeTensor]


_RECORD_INDEX_TENSOR_NAME_SIGNATURE_PREFIX = "__record_index_tensor_name__:"


class TFGraphRecordDecoder(metaclass=abc.ABCMeta):
  """Base class for decoders that turns a list of bytes to (composite) tensors.

  Sub-classes must implement `decode_record()` (see its docstring
  for requirements).

  Decoder instances can be saved as a SavedModel by `save_decoder()`.
  The SavedModel can be loaded back by `load_decoder()`. However, the loaded
  decoder will always be of the type `LoadedDecoder` and only have the public
  interfaces listed in this base class available.
  """

  def output_type_specs(self) -> Dict[str, tf.TypeSpec]:
    """Returns the tf.TypeSpecs of the decoded tensors.

    Returns:
      A dict whose keys are the same as keys of the dict returned by
      `decode_record()` and values are the tf.TypeSpec of the corresponding
      (composite) tensor.
    """
    return {
        k: tf.type_spec_from_value(v) for k, v in
        self._make_concrete_decode_function().structured_outputs.items()
    }

  @abc.abstractmethod
  def decode_record(self, records: tf.Tensor) -> Dict[str, TensorAlike]:
    """Sub-classes should implement this.

    Implementations must use TF ops to derive the result (composite) tensors, as
    this function will be traced and become a tf.function (thus a TF Graph).
    Note that autograph is not enabled in such tracing, which means any python
    control flow / loops will not be converted to TF cond / loops automatically.

    The returned tensors must be batch-aligned (i.e. they should be at least
    of rank 1, and their outer-most dimensions must be of the same size). They
    do not have to be batch-aligned with the input tensor, but if that's the
    case, an additional tensor must be provided among the results, to indicate
    which input record a "row" in the output batch comes from. See
    `record_index_tensor_name` for more details.

    Args:
      records: a 1-D string tensor that contains the records to be decoded.

    Returns:
      A dict of (composite) tensors.
    """

  @property
  def record_index_tensor_name(self) -> Optional[str]:
    """The name of the tensor indicating which record a slice is from.

    The decoded tensors are batch-aligned among themselves, but they don't
    necessarily have to be batch-aligned with the input records. If not,
    sub-classes should implement this method to tie the batch dimension
    with the input record.

    The record index tensor must be a SparseTensor or a RaggedTensor of integral
    type, and must be 2-D and must not contain "missing" values.

    A record index tensor like the following:
    [[0], [0], [2]]
    means that of 3 "rows" in the output "batch", the first two rows came
    from the first record, and the 3rd row came from the third record.

    The name must not be an empty string.

    Returns:
      The name of the record index tensor.
    """
    return None

  def _make_concrete_decode_function(self):
    return (
        tf.function(
            self.decode_record,
            input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)],
            autograph=False)
        .get_concrete_function())

  def save(self, path: str) -> None:
    """Saves this TFGraphRecordDecoder to a SavedModel at `path`.

    This functions the same as `tf_graph_record_decoder.save_decoder()`. This is
    provided purely for convenience, and should not impact the actual saved
    model, since only the `tf.function` from `_make_concrete_decode_function` is
    saved.

    Args:
      path: The path to where the saved_model is saved.
    """
    save_decoder(self, path)


class LoadedDecoder(object):
  """A decoder recovered from a SavedModel.

  It has all the public interfaces of a TFGraphRecordDecoder.
  """

  def __init__(self, loaded_module: tf.Module):
    self._decode_fun = loaded_module.decode_fun
    self._record_index_tensor_name = None

    if hasattr(loaded_module, "signatures"):
      for signature_name in loaded_module.signatures.keys():
        if signature_name.startswith(
            _RECORD_INDEX_TENSOR_NAME_SIGNATURE_PREFIX):
          record_index_tensor_name = signature_name[
              len(_RECORD_INDEX_TENSOR_NAME_SIGNATURE_PREFIX):]
          assert record_index_tensor_name, (
              "Invalid (empty) record_index_tensor_name")
          self._record_index_tensor_name = record_index_tensor_name

    assert isinstance(self._decode_fun.structured_outputs, dict)
    # Note that a loaded concrete function's structured_outputs are already
    # TensorSpecs (instead of TensorAlikes).
    self._output_type_specs = self._decode_fun.structured_outputs.copy()

  def decode_record(self, record: tf.Tensor) -> Dict[str, TensorAlike]:
    return self._decode_fun(record)

  def output_type_specs(self) -> Dict[str, tf.TypeSpec]:
    return self._output_type_specs

  @property
  def record_index_tensor_name(self) -> Optional[str]:
    return self._record_index_tensor_name


def save_decoder(decoder: TFGraphRecordDecoder, path: str) -> None:
  """Saves a TFGraphRecordDecoder to a SavedModel."""
  m = tf.Module()
  m.decode_fun = decoder._make_concrete_decode_function()  # pylint:disable=protected-access

  signatures = dict()
  if decoder.record_index_tensor_name is not None:
    assert decoder.record_index_tensor_name, (
        "Invalid (empty) record_index_tensor_name")
    assert decoder.record_index_tensor_name in decoder.output_type_specs(), (
        "Invalid decoder: record_index_tensor_name: {} not in output "
        "tensors: {}".format(decoder.record_index_tensor_name,
                             decoder.output_type_specs().keys()))

    @tf.function(input_signature=[])
    def record_index_tensor_name_fun():
      return decoder.record_index_tensor_name
    # We also encode the record index tensor name in the name of a signature.
    # This way, we do not need to evaluate a tensor or a TF Function in order
    # to know the name when loading a decoder back.
    signatures = {
        "%s%s" % (_RECORD_INDEX_TENSOR_NAME_SIGNATURE_PREFIX,
                  decoder.record_index_tensor_name):
            record_index_tensor_name_fun.get_concrete_function()
    }

  tf.saved_model.save(m, path, signatures=signatures)


def load_decoder(path: str) -> LoadedDecoder:
  """Loads a TFGraphRecordDecoder from a SavedModel."""
  loaded_module = tf.saved_model.load(path)
  assert hasattr(loaded_module, "decode_fun"), (
      "the SavedModel is not a TFGraphRecordDecoder")
  return LoadedDecoder(loaded_module)
