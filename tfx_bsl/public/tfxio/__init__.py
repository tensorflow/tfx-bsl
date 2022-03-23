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
"""Module level imports for tfx_bsl.public.tfxio.

TFXIO defines a common in-memory data representation shared by all TFX libraries
and components, as well as an I/O abstraction layer to produce such
representations. See the RFC for details:
https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md
"""

from tfx_bsl.coders.example_coder import RecordBatchToExamplesEncoder
from tfx_bsl.coders.tf_graph_record_decoder import TFGraphRecordDecoder
from tfx_bsl.tfxio.csv_tfxio import BeamRecordCsvTFXIO
from tfx_bsl.tfxio.csv_tfxio import CsvTFXIO
from tfx_bsl.tfxio.dataset_options import RecordBatchesOptions
from tfx_bsl.tfxio.dataset_options import TensorFlowDatasetOptions
from tfx_bsl.tfxio.tensor_adapter import TensorAdapter
from tfx_bsl.tfxio.tensor_adapter import TensorAdapterConfig
from tfx_bsl.tfxio.tensor_adapter import TensorRepresentations
from tfx_bsl.tfxio.tf_example_record import TFExampleBeamRecord
from tfx_bsl.tfxio.tf_example_record import TFExampleRecord
from tfx_bsl.tfxio.tf_sequence_example_record import TFSequenceExampleBeamRecord
from tfx_bsl.tfxio.tf_sequence_example_record import TFSequenceExampleRecord
from tfx_bsl.tfxio.tfxio import TFXIO
