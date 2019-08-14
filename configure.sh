#!/bin/bash
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

# This script prepares the bazel workspace for build.

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "build --action_env $1=\"$2\""
}

function has_pyarrow() {
  ${PYTHON_BIN_PATH} -c "import pyarrow" > /dev/null
}

function ensure_pyarrow() {
  if has_pyarrow; then
    echo "Using installed pyarrow..."
  else
    echo "Building tfx_common requires pyarrow. Please install ${PYARROW_REQUIREMENT}"
    exit 1
  fi
}

function has_tf() {
  ${PYTHON_BIN_PATH} -c "import tensorflow" > /dev/null
}

function ensure_tensorflow() {
  if has_tf; then
    echo "Using installed tensorflow..."
  else
    echo "Building tfx_common requires tensorflow."
    exit 1
  fi
}

if [[ -z "${PYTHON_BIN_PATH}" ]]; then
  if [[ -z "$1" ]]; then
    PYTHON_BIN_PATH="$(which python)"
  else
    PYTHON_BIN_PATH="$1"
  fi
fi

PYARROW_REQUIREMENT=$(${PYTHON_BIN_PATH} -c "fp = open('third_party/pyarrow_version.bzl', 'r'); d = {}; exec(fp.read(), d); fp.close(); print(d['PY_DEP'])")

ensure_pyarrow
ensure_tensorflow

# Search for Arrow libraries.
ARROW_HEADER_DIR=( $("${PYTHON_BIN_PATH}" -c 'import pyarrow as pa; print(pa.get_include().replace("\\", "\\\\"))') )
ARROW_SHARED_LIBRARY_DIR=( $("${PYTHON_BIN_PATH}" -c 'import pyarrow as pa; print(pa.get_library_dirs()[0].replace("\\", "\\\\"))') )

echo "Found pyarrow headers at... ${ARROW_HEADER_DIR}"
echo "Found pyarrow shared libraries at... ${ARROW_SHARED_LIBRARY_DIR}"

write_action_env_to_bazelrc "ARROW_HEADER_DIR" ${ARROW_HEADER_DIR}
write_action_env_to_bazelrc "ARROW_SHARED_LIBRARY_DIR" ${ARROW_SHARED_LIBRARY_DIR}

# Search for TF libraries.
TF_CFLAGS=( $("${PYTHON_BIN_PATH}" -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$("${PYTHON_BIN_PATH}" -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

write_action_env_to_bazelrc "TF_HEADER_DIR" ${TF_CFLAGS:2}
TF_SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
TF_SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    TF_SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  else
    TF_SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi
echo "Found tensorflow headers and library at... ${TF_SHARED_LIBRARY_DIR}"
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_DIR" ${TF_SHARED_LIBRARY_DIR}
write_action_env_to_bazelrc "TF_SHARED_LIBRARY_NAME" ${TF_SHARED_LIBRARY_NAME}
write_action_env_to_bazelrc "TF_NEED_CUDA" 0

echo ".bazelrc successfully updated. Note that this script only appends to .bazelrc."
