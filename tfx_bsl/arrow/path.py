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

from typing import Iterable, Text, Tuple, Union

from tensorflow_metadata.proto.v0 import path_pb2


class ColumnPath(object):
  """ColumnPath addresses a column potentially nested under a StructArray."""

  __slot__ = ["_steps"]

  def __init__(self, steps: Union[Iterable[Text], Text]):
    """If a single Step is specified, constructs a Path of that step."""
    if isinstance(steps, Text):
      steps = (steps,)
    self._steps = tuple(steps)

  def to_proto(self) -> path_pb2.Path:
    """Creates a tensorflow_metadata path proto this ColumnPath."""
    return path_pb2.Path(step=self._steps)

  @staticmethod
  def from_proto(path_proto: path_pb2.Path):
    """Creates a ColumnPath from a tensorflow_metadata path proto.

    Args:
      path_proto: a tensorflow_metadata path proto.

    Returns:
      A ColumnPath representing the path proto's steps.
    """
    return ColumnPath(path_proto.step)

  def steps(self) -> Tuple[Text, ...]:
    """Returns the tuple of steps that represents this ColumnPath."""
    return self._steps

  def parent(self) -> "ColumnPath":
    """Gets the parent path of the current ColumnPath.

    example: ColumnPath(["this", "is", "my", "path"]).parent() will
    return a ColumnPath representing "this.is.my".

    Returns:
      A ColumnPath with the leaf step removed.
    """
    if not self._steps:
      raise ValueError("Root does not have parent.")
    return ColumnPath(self._steps[:-1])

  def child(self, child_step: Text) -> "ColumnPath":
    """Creates a new ColumnPath with a new child.

    example: ColumnPath(["this", "is", "my", "path"]).child("new_step") will
    return a ColumnPath representing "this.is.my.path.new_step".

    Args:
      child_step: name of the new child step to append.

    Returns:
      A ColumnPath with the new child_step
    """
    return ColumnPath(self._steps + (child_step,))

  def prefix(self, ending_index: int) -> "ColumnPath":
    """Creates a new ColumnPath, taking the prefix until the ending_index.

    example: ColumnPath(["this", "is", "my", "path"]).prefix(1) will return a
    ColumnPath representing "this.is.my".

    Args:
      ending_index: where to end the prefix.

    Returns:
      A ColumnPath representing the prefix of this ColumnPath.
    """
    return ColumnPath(self._steps[:ending_index])

  def suffix(self, starting_index: int) -> "ColumnPath":
    """Creates a new ColumnPath, taking the suffix from the starting_index.

    example: ColumnPath(["this", "is", "my", "path"]).suffix(1) will return a
    ColumnPath representing "is.my.path".

    Args:
      starting_index: where to start the suffix.

    Returns:
      A ColumnPath representing the suffix of this ColumnPath.
    """
    return ColumnPath(self._steps[starting_index:])

  def initial_step(self) -> Text:
    """Returns the first step of this path.

    Raises:
      ValueError: if the path is empty.
    """
    if not self._steps:
      raise ValueError("This ColumnPath does not have any steps.")
    return self._steps[0]

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
