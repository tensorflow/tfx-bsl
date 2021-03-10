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
"""Tests for tfx_bsl.tfxio.tfxio."""

import pyarrow as pa
from tfx_bsl.tfxio import tfxio

from google.protobuf import text_format
from absl.testing import absltest
from tensorflow_metadata.proto.v0 import schema_pb2


class _FakeTFXIO(tfxio.TFXIO):
  """A fake TFXIO for testing the projection origin tracking."""

  def __init__(self, columns):
    super().__init__()
    self._columns = columns

  def ArrowSchema(self):
    return pa.schema([
        pa.field(c, pa.list_(pa.int32())) for c in self._columns
    ])

  def TensorRepresentations(self):
    return {  # pylint: disable=g-complex-comprehension
        c: text_format.Parse(
            """
              dense_tensor {
                column_name: "%s"
                shape {
                  dim {
                    size: 1
                  }
                }
              }""" % c, schema_pb2.TensorRepresentation())
        for c in self._columns
    }

  def _ProjectImpl(self, columns):
    projected_columns = set(columns)
    for c in projected_columns:
      if c not in self._columns:
        raise ValueError("Unexpected column")
    return _FakeTFXIO(list(projected_columns))

  # not used in tests.
  def BeamSource(self, batch_size):
    raise NotImplementedError

  # not used in tests.
  def RecordBatches(self, options):
    raise NotImplementedError

  # not used in tests.
  def TensorFlowDataset(self, options):
    raise NotImplementedError


class TfxioTest(absltest.TestCase):

  def testProjection(self):
    origin = _FakeTFXIO(["column1", "column2", "column3"])
    origin_adapter = origin.TensorAdapter()
    self.assertLen(origin_adapter.TypeSpecs(), 3)
    projected = origin.Project(["column2", "column3"])
    projected_adapter = projected.TensorAdapter()
    self.assertLen(projected_adapter.TypeSpecs(), 2)
    self.assertEqual(origin_adapter.TypeSpecs(),
                     projected_adapter.OriginalTypeSpecs())

    # Project again, but the origin should still be `origin`.
    projected = projected.Project(["column3"])
    projected_adapter = projected.TensorAdapter()
    self.assertLen(projected_adapter.TypeSpecs(), 1)
    self.assertEqual(origin_adapter.TypeSpecs(),
                     projected_adapter.OriginalTypeSpecs())


if __name__ == "__main__":
  absltest.main()
