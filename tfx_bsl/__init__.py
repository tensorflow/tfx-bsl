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
"""Init module for tfx_bsl."""

from tfx_bsl.version import __version__

# TODO(b/161449255): clean this up after a tfx release that does not contain the
# usage of this is out.
# Indicates that tfx_bsl.coders.tf_graph_record_decoder and
# tfx_bsl.tfxio.record_to_tensor_tfxio are available.
HAS_TF_GRAPH_RECORD_DECODER = True

# TODO(b/162532479): clean this up after a tfx release that does not contain the
# usage of this is out.
# Indicates that common TFXIO implementations can take a list of file patterns
# at construction time.
TFXIO_SUPPORT_MULTIPLE_FILE_PATTERNS = True
