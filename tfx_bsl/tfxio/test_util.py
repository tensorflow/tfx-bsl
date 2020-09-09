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

from typing import  Optional, Text

from tfx_bsl.tfxio import tf_example_record

from tensorflow_metadata.proto.v0 import schema_pb2


# DEPRECATED. Prefer tf_example_record.TFExampleBeamRecord.
# TODO(b/158580478): clean this up.
class InMemoryTFExampleRecord(tf_example_record.TFExampleBeamRecord):

  def __init__(self, schema: Optional[schema_pb2.Schema] = None,
               raw_record_column_name: Optional[Text] = None):
    super().__init__(
        physical_format="inmem",
        telemetry_descriptors=["test", "component"],
        schema=schema,
        raw_record_column_name=raw_record_column_name)
