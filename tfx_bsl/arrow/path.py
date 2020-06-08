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
"""Defines ColumnPath class."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

from typing import Iterable, Text, Tuple, Union

import six

from tensorflow_metadata.proto.v0 import path_pb2


# Text on py3, bytes on py2.
Step = Union[bytes, Text]


@six.python_2_unicode_compatible
class ColumnPath(object):
  """ColumnPath addresses a column potentially nested under a StructArray."""

  __slot__ = ["_steps"]

  def __init__(self, steps: Union[Iterable[Step], Step]):
    """If a single Step is specified, constructs a Path of that step."""
    if isinstance(steps, (bytes, six.text_type)):
      steps = (steps,)
    self._steps = tuple(
        s if isinstance(s, six.text_type) else s.decode("utf-8") for s in steps)

  def to_proto(self) -> path_pb2.Path:
    return path_pb2.Path(step=self._steps)

  @staticmethod
  def from_proto(path_proto: path_pb2.Path):
    return ColumnPath(path_proto.step)

  def steps(self) -> Tuple[Step, ...]:
    return self._steps

  def parent(self) -> "ColumnPath":
    if not self._steps:
      raise ValueError("Root does not have parent.")
    return ColumnPath(self._steps[:-1])

  def child(self, child_step: Step) -> "ColumnPath":
    if isinstance(child_step, six.text_type):
      return ColumnPath(self._steps + (child_step,))
    return ColumnPath(self._steps + (child_step.decode("utf-8"),))

  def __str__(self) -> Text:
    return u".".join(self._steps)

  def __repr__(self) -> Text:
    return self.__str__()

  def __eq__(self, other) -> bool:
    return self._steps == other._steps  # pylint: disable=protected-access

  def __lt__(self, other) -> bool:
    # lexicographic order.
    return self._steps < other._steps  # pylint: disable=protected-access

  def __hash__(self) -> int:
    return hash(self._steps)

  def __len__(self) -> int:
    return len(self._steps)

  def __bool__(self) -> bool:
    return bool(self._steps)
