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
"""Contains predicates about pyarrow's capabilities."""
from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports
from __future__ import print_function

import pyarrow as pa
from tfx_bsl.arrow import array_util


def HasFullSupportForLargeList() -> bool:
  """Returns True if pyarrow has full support for large list.

  "Full support" means having all the capabilities that TFX needs.

  Returns:
    a boolean.
  """
  return _LargeListCanBeConvertedToPandas()


def HasFullSupportForLargeBinary() -> bool:
  """Returns True if pyarrow has full support for large binary.

  "Full support" means having all the capabilities that TFX needs.

  Returns:
    a boolean.
  """
  return _LargeBinaryCanBeDictEncoded() and _LargeBinaryCanBeValueCounted()


def _LargeListCanBeConvertedToPandas() -> bool:
  """Returns True if a large_list can be converted to a pd.Series."""
  try:
    pa.array([], type=pa.large_list(pa.int32())).to_pandas()
  except:  # pylint:disable=bare-except
    return False
  return True


def _LargeBinaryCanBeDictEncoded() -> bool:
  """Returns True if a large binary array can be dictionary encoded."""
  try:
    pa.array([], type=pa.large_binary()).dictionary_encode()
  except:  # pylint:disable=bare-except
    return False
  return True


def _LargeBinaryCanBeValueCounted() -> bool:
  """Returns True if a large binary array can be value counted."""
  try:
    array_util.ValueCounts(pa.array([], type=pa.large_binary()))
  except:  # pylint:disable=bare-except
    return False
  return True

