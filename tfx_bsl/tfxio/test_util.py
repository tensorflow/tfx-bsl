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
"""Contains TFXIO helpers for testing purposes."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import List, Text

import apache_beam as beam
from tfx_bsl.tfxio import tf_example_record
from tfx_bsl.tfxio import tfxio


class InMemoryTFExampleRecord(tf_example_record._TFExampleRecordBase):  # pylint: disable=protected-access
  """A TF Example TFXIO whose source is a PCollection[bytes].

  Usage:
    tfxio = InMemoryTFExampleRecord(schema)
    record_batches = (
      beam.Create([serialized_example1, ...]) | tfxio.BeamSource())
  """

  def _RawRecordBeamSourceInternal(self) -> beam.PTransform:

    @beam.typehints.with_input_types(bytes)
    @beam.typehints.with_output_types(bytes)
    def _PTransformFn(raw_records_pcoll: beam.pvalue.PCollection):
      return raw_records_pcoll

    return beam.ptransform_fn(_PTransformFn)()

  def TensorFlowDataset(self):
    raise NotImplementedError

  def _ProjectImpl(self, tensor_names: List[Text]) -> tfxio.TFXIO:
    return InMemoryTFExampleRecord(
        self._ProjectTfmdSchema(tensor_names), self.raw_record_column_name)
