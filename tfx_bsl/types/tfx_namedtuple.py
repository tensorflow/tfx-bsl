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
"""A workaround for `collections.namedtuple` to provide PySpark compatibility.

PySpark hacks classes produced by `collections.namedtuple` to make them
serializable. This, however, makes their subclasses serialize and deserialize
as a parent class. Such behavior causes errors in TFX code if the subclass
overrides parent class methods when PySpark is present in the environment.
See https://issues.apache.org/jira/browse/SPARK-22674 for more details on the
PySpark issue.
TODO(https://issues.apache.org/jira/browse/SPARK-22674): remove this workaround
once the hack is removed from PySpark.
"""
import collections
import sys
import typing


def _patch_namedtuple(cls):
  """Helper function that patches namedtuple class to prevent PySpark hack."""

  def reduce(self):
    return (self.__class__, tuple(self))

  # Classes for which `__reduce__` is defined don't get hacked by PySpark.
  cls.__reduce__ = reduce

  # For pickling to work, the __module__ variable needs to be set to the frame
  # where the named tuple is created.
  try:
    cls.__module__ = sys._getframe(2).f_globals.get('__name__', '__main__')  # pylint: disable=protected-access
  except (AttributeError, ValueError):
    pass


def namedtuple(typename, field_names, *, rename=False):
  """Wrapper around `collections.namedtuple` to provide PySpark compatibility."""

  result = collections.namedtuple(typename, field_names, rename=rename)
  _patch_namedtuple(result)
  return result


class TypedNamedTuple:
  """Wrapper around `typing.NamedTuple` to provide PySpark compatibility."""
  __slots__ = ()

  def __new__(cls, typename, fields=None):
    result = typing.NamedTuple(typename, fields)
    _patch_namedtuple(result)
    return result
