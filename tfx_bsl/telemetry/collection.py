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

from typing import Text

import apache_beam as beam
import pyarrow as pa


def _increment_counter(counter_namespace: Text,
                       counter_name: Text,
                       element: int):
  counter = beam.metrics.Metrics.counter(counter_namespace, counter_name)
  counter.inc(element)


@beam.ptransform_fn
def TrackRecordBatchBytes(dataset: beam.PCollection[pa.RecordBatch],  # pylint: disable=invalid-name
                          counter_namespace: Text,
                          counter_name: Text) -> beam.pvalue.PCollection[None]:
  """Gathers telemetry on input record batch."""
  return (dataset
          | "GetRecordBatchSize" >> beam.Map(lambda rb: rb.nbytes)
          | "SumTotalBytes" >> beam.CombineGlobally(sum)
          | "IncrementCounter" >> beam.Map(
              lambda x: _increment_counter(counter_namespace, counter_name, x)))
