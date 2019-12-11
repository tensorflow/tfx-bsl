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

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import abc
import apache_beam as beam
import six
from tfx_bsl.pyarrow_tf import pyarrow as pa
from tfx_bsl.pyarrow_tf import tensorflow as tf
from tfx_bsl.tfxio import tensor_adapter
from typing import List, Text


@six.add_metaclass(abc.ABCMeta)
class TFXIO(object):
  """Abstract basic class of all TFXIO API implementations."""

  @abc.abstractmethod
  def BeamSource(self) -> beam.PTransform:
    """Returns a beam `PTransform` that produces `PCollection[pa.RecordBatch]`.

    May NOT raise an error if the TFMD schema was not provided at construction
    time.

    If a TFMD schema was provided at construction time, all the
    `pa.RecordBatch`es in the result `PCollection` must be of the same schema
    returned by `self.ArrowSchema`. If a TFMD schema was not provided, the
    `pa.RecordBatch`es might not be of the same schema (they may contain
    different numbers of columns).
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
    """

  @abc.abstractmethod
  def TensorFlowDataset(self) -> tf.data.Dataset:
    """Returns a tf.data.Dataset of TF inputs.

    May raise an error if the TFMD schema was not provided at construction time.
    """

  @abc.abstractmethod
  def Project(self, tensor_names: List[Text]) -> "TFXIO":
    """Projects the dataset represented by this TFXIO.

    Returns a `TFXIO` instance that is the same as `self` except that:
      - Only columns needed for given tensor_names are guaranteed to be
        produced by `self.BeamSource()`
      - `self.TensorAdapterConfig()` and `self.TensorFlowDataset()` are trimmed
        to contain only those tensors.

    May raise an error if the TFMD schema was not provided at construction time.

    Args:
      tensor_names: a set of tensor names.
    """

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
