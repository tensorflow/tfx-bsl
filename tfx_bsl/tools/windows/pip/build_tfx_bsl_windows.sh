#!/bin/bash
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
# ==============================================================================
#
# This script assumes the standard setup on tensorflow Jenkins windows machines.
# It is NOT guaranteed to work on any other machine. Use at your own risk!
#
# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Anaconda3
# * Bazel windows executable copied as "bazel.exe" and included in PATH.

function common::prepare_python_env() {
  set -eux

  # Setting up the environment variables Bazel and ./configure needs.
  source "${TFX_BSL_OUTPUT_DIR}/tfx_bsl/tools/windows/bazel/common_env.sh"

  "${PYTHON_BIN_PATH}" -m pip install --upgrade pip setuptools wheel "numpy>=1.16,<2"
  pip list
  # There's a tensorflow bazel rule that executes a small piece of code
  # (https://github.com/tensorflow/tensorflow/blob/b36436b087bd8e8701ef51718179037cccdfc26e/third_party/py/python_configure.bzl#L150)
  # to determine the path to python headers and that code doesn't work with
  # setuptools>=50.0 and Python 3.6.1 (which is our testing set up). Setting
  # this environment variable would revert setuptools to the old, good behavior.
  # See https://github.com/pypa/setuptools/issues/2352
  # Note that setuptools 50.0.1 claim to have "fixed" the issue but it
  # did not work for Python 3.6.1 (newer 3.6 may work).
  export SETUPTOOLS_USE_DISTUTILS=stdlib
}

function tfx_bsl::build_from_head_windows() {
  cd "${TFX_BSL_OUTPUT_DIR}"
  # Enable short object file path to avoid long path issue on Windows.
  echo "startup --output_user_root=${TMPDIR}" >> .bazelrc
  "${PYTHON_BIN_PATH}" setup.py bdist_wheel 1>&2
  TFX_BSL_WHEEL="$(find "${TFX_BSL_OUTPUT_DIR}/dist" -name "*.whl")"
  if [[ -z "${TFX_BSL_WHEEL}" ]]; then
    return 1
  fi
}

function tfmd::build_from_head_windows() {
  cd "${TFMD_OUTPUT_DIR}"
  # Enable short object file path to avoid long path issue on Windows.
  echo "startup --output_user_root=${TMPDIR}" >> .bazelrc
  "${PYTHON_BIN_PATH}" setup.py bdist_wheel 1>&2
  TFMD_WHEEL="$(find "${TFMD_OUTPUT_DIR}/dist" -name "*.whl")"
  if [[ -z "${TFMD_WHEEL}" ]]; then
    return 1
  fi
}
