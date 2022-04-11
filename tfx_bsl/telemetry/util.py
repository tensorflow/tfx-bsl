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
"""Telemetry utils."""

from typing import Sequence


def MakeTfxNamespace(descriptors: Sequence[str]) -> str:
  """Makes a TFX beam metric namespace from a list of descriptors."""
  return AppendToNamespace("tfx", descriptors)


def AppendToNamespace(namespace: str,
                      descriptors_to_append: Sequence[str]) -> str:
  """Appends a list of descriptors to a beam metric namespace."""
  if descriptors_to_append:
    return namespace + "." + ".".join(descriptors_to_append)
  return namespace
