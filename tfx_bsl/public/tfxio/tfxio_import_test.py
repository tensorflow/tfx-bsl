# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test to make sure public TFXIO symbols can be imported."""

from absl.testing import absltest


class TfxioImportTest(absltest.TestCase):

  def test_import(self):
    # pylint: disable=g-import-not-at-top,unused-import
    from tfx_bsl.public.tfxio import RecordBatchToExamplesEncoder
    from tfx_bsl.public.tfxio import BeamRecordCsvTFXIO
    from tfx_bsl.public.tfxio import CsvTFXIO
    from tfx_bsl.public.tfxio import TensorFlowDatasetOptions
    from tfx_bsl.public.tfxio import TensorAdapter
    from tfx_bsl.public.tfxio import TensorAdapterConfig
    from tfx_bsl.public.tfxio import TensorRepresentations
    from tfx_bsl.public.tfxio import TFExampleBeamRecord
    from tfx_bsl.public.tfxio import TFExampleRecord
    from tfx_bsl.public.tfxio import TFGraphRecordDecoder
    from tfx_bsl.public.tfxio import TFSequenceExampleBeamRecord
    from tfx_bsl.public.tfxio import TFSequenceExampleRecord
    from tfx_bsl.public.tfxio import TFXIO


if __name__ == "__main__":
  absltest.main()
