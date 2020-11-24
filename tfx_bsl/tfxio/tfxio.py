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
"""TFXIO Interface.

TFXIO is the I/O abstraction for TFX. It allows TFX components / libraries to
access pipeline payload in the form of a common in-memory format (Apache Arrow
RecordBatch) regardless of the physical (at-rest) format of the payload. It also
provides an adapter (TensorAdapter) to translate a RecordBatch into TF tensors.

See
https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md
for the high-level design.
"""

import abc
from typing import Iterator, List, Optional, Text
import apache_beam as beam
import pyarrow as pa
import tensorflow as tf
from tfx_bsl.tfxio import dataset_options
from tfx_bsl.tfxio import tensor_adapter


class TFXIO(object, metaclass=abc.ABCMeta):
  """Abstract basic class of all TFXIO API implementations."""

  @abc.abstractmethod
  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:
    """Returns a beam `PTransform` that produces `PCollection[pa.RecordBatch]`.

    May NOT raise an error if the TFMD schema was not provided at construction
    time.

    If a TFMD schema was provided at construction time, all the
    `pa.RecordBatch`es in the result `PCollection` must be of the same schema
    returned by `self.ArrowSchema`. If a TFMD schema was not provided, the
    `pa.RecordBatch`es might not be of the same schema (they may contain
    different numbers of columns).

    Args:
      batch_size: if not None, the `pa.RecordBatch` produced will be of the
        specified size. Otherwise it's automatically tuned by Beam.
    """

  @abc.abstractmethod
  def RecordBatches(
      self, options: dataset_options.RecordBatchesOptions
  ) -> Iterator[pa.RecordBatch]:
    """Returns an iterable of record batches.

    This can be used outside of Apache Beam or TensorFlow to access data.

    Args:
      options: An options object for iterating over record batches. Look at
        `dataset_options.RecordBatchesOptions` for more details.
    """

  @abc.abstractmethod
  def ArrowSchema(self) -> pa.Schema:
    """Returns the schema of the `RecordBatch` produced by `self.BeamSource()`.

    May raise an error if the TFMD schema was not provided at construction time.
    """

  @abc.abstractmethod
  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    """Returns the `TensorRepresentations`.

    These `TensorRepresentation`s describe the tensors or composite tensors
    produced by the `TensorAdapter` created from `self.TensorAdapter()` or
    the tf.data.Dataset created from `self.TensorFlowDataset()`.

    May raise an error if the TFMD schema was not provided at construction time.
    May raise an error if the tensor representations are invalid.
    """

  @abc.abstractmethod
  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    """Returns a tf.data.Dataset of TF inputs.

    May raise an error if the TFMD schema was not provided at construction time.

    Args:
      options: an options object for the tf.data.Dataset. Look at
        `dataset_options.TensorFlowDatasetOptions` for more details.
    """

  @abc.abstractmethod
  def _ProjectImpl(self, tensor_names: List[Text]) -> "TFXIO":
    """Sub-classes should implement this interface to perform projections.

    It should return a `TFXIO` instance that is the same as `self` except that:
      - Only columns needed for given tensor_names are guaranteed to be
        produced by `self.BeamSource()`
      - `self.TensorAdapterConfig()` and `self.TensorFlowDataset()` are trimmed
        to contain only those tensors.

    May raise an error if the TFMD schema was not provided at construction time.

    Args:
      tensor_names: a set of tensor names.
    """

  # final
  def Project(self, tensor_names: List[Text]) -> "TFXIO":
    """Projects the dataset represented by this TFXIO.

    A Projected TFXIO:
    - Only columns needed for given tensor_names are guaranteed to be
      produced by `self.BeamSource()`
    - `self.TensorAdapterConfig()` and `self.TensorFlowDataset()` are trimmed
      to contain only those tensors.
    - It retains a reference to the very original TFXIO, so its TensorAdapter
      knows about the specs of the tensors that would be produced by the
      original TensorAdapter. Also see `TensorAdapter.OriginalTensorSpec()`.

    May raise an error if the TFMD schema was not provided at construction time.

    Args:
      tensor_names: a set of tensor names.

    Returns:
      A `TFXIO` instance that is the same as `self` except that:
      - Only columns needed for given tensor_names are guaranteed to be
        produced by `self.BeamSource()`
      - `self.TensorAdapterConfig()` and `self.TensorFlowDataset()` are trimmed
        to contain only those tensors.
    """
    if isinstance(self, _ProjectedTFXIO):
      # pylint: disable=protected-access
      return _ProjectedTFXIO(self.origin,
                             self.projected._ProjectImpl(tensor_names))
    return _ProjectedTFXIO(self, self._ProjectImpl(tensor_names))

  # final
  def TensorAdapterConfig(self) -> tensor_adapter.TensorAdapterConfig:
    """Returns the config to initialize a `TensorAdapter`.

    Returns:
      a `TensorAdapterConfig` that is the same as what is used to initialize the
      `TensorAdapter` returned by `self.TensorAdapter()`.
    """
    return tensor_adapter.TensorAdapterConfig(
        self.ArrowSchema(), self.TensorRepresentations())

  # final
  def TensorAdapter(self) -> tensor_adapter.TensorAdapter:
    """Returns a TensorAdapter that converts pa.RecordBatch to TF inputs.

    May raise an error if the TFMD schema was not provided at construction time.
    """
    return tensor_adapter.TensorAdapter(self.TensorAdapterConfig())


class _ProjectedTFXIO(TFXIO):
  """A wrapper of a projected TFXIO to track its origin."""

  def __init__(self, origin: TFXIO, projected: TFXIO):
    self._origin = origin
    self._projected = projected

  @property
  def origin(self) -> TFXIO:
    return self._origin

  @property
  def projected(self) -> TFXIO:
    return self._projected

  def BeamSource(self, batch_size: Optional[int] = None) -> beam.PTransform:
    return self.projected.BeamSource(batch_size)

  def RecordBatches(
      self, options: dataset_options.RecordBatchesOptions
  ) -> Iterator[pa.RecordBatch]:
    return self.projected.RecordBatches(options)

  def ArrowSchema(self) -> pa.Schema:
    return self.projected.ArrowSchema()

  def TensorRepresentations(self) -> tensor_adapter.TensorRepresentations:
    return self.projected.TensorRepresentations()

  def TensorFlowDataset(
      self,
      options: dataset_options.TensorFlowDatasetOptions) -> tf.data.Dataset:
    return self.projected.TensorFlowDataset(options)

  def _ProjectImpl(self, unused_tensor_names: List[Text]) -> "TFXIO":
    raise ValueError("This should never be called.")

  def TensorAdapterConfig(self) -> tensor_adapter.TensorAdapterConfig:
    return tensor_adapter.TensorAdapterConfig(
        self.projected.ArrowSchema(),
        self.projected.TensorRepresentations(),
        original_type_specs=self.origin.TensorAdapter().TypeSpecs())
