# Copyright 2022 Google LLC
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
"""Ptransforms for collecting telemetry."""

import collections
from typing import Dict

import apache_beam as beam
import pyarrow as pa

from tensorflow_metadata.proto.v0 import schema_pb2

# TODO(b/68154497): remove this. # pylint: disable=no-value-for-parameter


def _IncrementCounter(element: int, counter_namespace: str,
                      counter_name: str) -> int:
  counter = beam.metrics.Metrics.counter(counter_namespace, counter_name)
  counter.inc(element)
  return element


@beam.ptransform_fn
def TrackRecordBatchBytes(dataset: beam.PCollection[pa.RecordBatch],
                          counter_namespace: str,
                          counter_name: str) -> beam.PCollection[int]:
  """Gathers telemetry on input record batch."""
  counter = beam.metrics.Metrics.counter(counter_namespace, counter_name)

  def counted_bytes(rb: pa.RecordBatch):
    result = rb.nbytes
    counter.inc(result)
    return result

  return dataset | "CountedBytes" >> beam.Map(counted_bytes)


def _IncrementTensorRepresentationCounters(
    tensor_representations: Dict[str, schema_pb2.TensorRepresentation],
    counter_namespace: str):
  kind_counter = collections.Counter(
      representation.WhichOneof("kind")
      for representation in tensor_representations.values())
  for kind, count in kind_counter.items():
    _IncrementCounter(count, counter_namespace, kind)


@beam.ptransform_fn
def TrackTensorRepresentations(
    tensor_representations: beam.PCollection[Dict[
        str, schema_pb2.TensorRepresentation]],
    counter_namespace: str) -> beam.PCollection[None]:
  """Gathers telemetry on TensorRepresentation types."""
  return (tensor_representations | "IncrementCounters" >> beam.Map(
      _IncrementTensorRepresentationCounters,
      counter_namespace=counter_namespace))
