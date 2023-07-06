# Copyright 2023 Google LLC. All Rights Reserved.
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
"""Helpers to fix Beam's pickling."""

import types

import dill


# TODO(b/281148738): Remove this once all supported Beam versions depend on dill
# with updated pickling logic or this is fixed in Beam.
def fix_code_type_pickling() -> None:
  """Overrides `CodeType` pickling to prevent segfaults in Python 3.10."""
  # Based on the `save_code` from dill-0.3.6.
  # https://github.com/uqfoundation/dill/blob/d5c4dccbe19fb27bfd757cb60abd2899fd9e59ba/dill/_dill.py#L1105
  # Author: Mike McKerns (mmckerns @caltech and @uqfoundation)
  # Copyright (c) 2008-2015 California Institute of Technology.
  # Copyright (c) 2016-2023 The Uncertainty Quantification Foundation.
  # License: 3-clause BSD.  The full license text is available at:
  #  - https://github.com/uqfoundation/dill/blob/master/LICENSE

  # The following function is also based on 'save_codeobject' from 'cloudpickle'
  # Copyright (c) 2012, Regents of the University of California.
  # Copyright (c) 2009 `PiCloud, Inc. <http://www.picloud.com>`_.
  # License: 3-clause BSD.  The full license text is available at:
  #  - https://github.com/cloudpipe/cloudpickle/blob/master/LICENSE
  @dill.register(types.CodeType)
  def save_code(pickler, obj):  # pylint: disable=unused-variable
    if hasattr(obj, 'co_endlinetable'):  # python 3.11a (20 args)
      args = (
          obj.co_argcount,
          obj.co_posonlyargcount,
          obj.co_kwonlyargcount,
          obj.co_nlocals,
          obj.co_stacksize,
          obj.co_flags,
          obj.co_code,
          obj.co_consts,
          obj.co_names,
          obj.co_varnames,
          obj.co_filename,
          obj.co_name,
          obj.co_qualname,
          obj.co_firstlineno,
          obj.co_linetable,
          obj.co_endlinetable,
          obj.co_columntable,
          obj.co_exceptiontable,
          obj.co_freevars,
          obj.co_cellvars,
      )
    elif hasattr(obj, 'co_exceptiontable'):  # python 3.11 (18 args)
      args = (
          obj.co_argcount,
          obj.co_posonlyargcount,
          obj.co_kwonlyargcount,
          obj.co_nlocals,
          obj.co_stacksize,
          obj.co_flags,
          obj.co_code,
          obj.co_consts,
          obj.co_names,
          obj.co_varnames,
          obj.co_filename,
          obj.co_name,
          obj.co_qualname,
          obj.co_firstlineno,
          obj.co_linetable,
          obj.co_exceptiontable,
          obj.co_freevars,
          obj.co_cellvars,
      )
    elif hasattr(obj, 'co_linetable'):  # python 3.10 (16 args)
      args = (
          obj.co_argcount,
          obj.co_posonlyargcount,
          obj.co_kwonlyargcount,
          obj.co_nlocals,
          obj.co_stacksize,
          obj.co_flags,
          obj.co_code,
          obj.co_consts,
          obj.co_names,
          obj.co_varnames,
          obj.co_filename,
          obj.co_name,
          obj.co_firstlineno,
          obj.co_linetable,
          obj.co_freevars,
          obj.co_cellvars,
      )
    elif hasattr(obj, 'co_posonlyargcount'):  # python 3.8 (16 args)
      args = (
          obj.co_argcount,
          obj.co_posonlyargcount,
          obj.co_kwonlyargcount,
          obj.co_nlocals,
          obj.co_stacksize,
          obj.co_flags,
          obj.co_code,
          obj.co_consts,
          obj.co_names,
          obj.co_varnames,
          obj.co_filename,
          obj.co_name,
          obj.co_firstlineno,
          obj.co_lnotab,
          obj.co_freevars,
          obj.co_cellvars,
      )
    else:  # python 3.7 (15 args)
      args = (
          obj.co_argcount,
          obj.co_kwonlyargcount,
          obj.co_nlocals,
          obj.co_stacksize,
          obj.co_flags,
          obj.co_code,
          obj.co_consts,
          obj.co_names,
          obj.co_varnames,
          obj.co_filename,
          obj.co_name,
          obj.co_firstlineno,
          obj.co_lnotab,
          obj.co_freevars,
          obj.co_cellvars,
      )

    pickler.save_reduce(types.CodeType, args, obj=obj)
