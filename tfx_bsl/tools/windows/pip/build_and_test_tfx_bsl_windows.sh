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

set -x

# This script is under <repo_root>/tfx_bsl/tools/windows/pip/
# Change into repository root.
script_dir=$(dirname $0)
cd ${script_dir%%tfx_bsl/tools/windows/pip}.

# Setting up the environment variables Bazel and ./configure needs
source "tfx_bsl/tools/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

source "tfx_bsl/tools/windows/pip/build_tfx_bsl_windows.sh" \
  || { echo "Failed to source build_tfx_bsl_windows.sh" >&2; exit 1; }

(tfx_bsl::build_from_head_windows) && wheel=$(ls dist/*.whl) \
  || { echo "Failed to build tfx_bsl."; exit 1; }

# Uninstall Cython (if installed) as Beam has issues with Cython installed.
# TODO(b/130120997): Avoid installing Beam without Cython.
pip install ${wheel} && \
pip install ${TENSORFLOW} && \
pip uninstall -y Cython && \

"${PYTHON_BIN_PATH}" -m tfx_bsl.test_util.run_all_tests --start_dir="tfx_bsl" \
  || { echo "Failed to run unit tests." exit 1; }

# copy wheel to ${KOKORO_ARTIFACTS_DIR}
cp ${wheel} ${KOKORO_ARTIFACTS_DIR}
