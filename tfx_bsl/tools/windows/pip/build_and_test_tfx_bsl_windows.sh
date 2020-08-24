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
# * Should set TFX_BSL_OUTPUT_DIR and TFMD_OUTPUT_DIR to the source code root
#   of https://github.com/tensorflow/tfx-bsl and
#   https://github.com/tensorflow/metadata.

set -eux
function tfx::die() {
  echo "$*" >&2
  exit 1
}

source "${TFX_BSL_OUTPUT_DIR}/tfx_bsl/tools/windows/pip/build_tfx_bsl_windows.sh" \
  || tfx::die "Failed to source build_tfx_bsl_windows.sh"

common::prepare_python_env \
  || tfx::die "Failed to run common::prepare_python_env"
tfmd::build_from_head_windows \
  || tfx::die "Failed to build TFMD from source."
tfx_bsl::build_from_head_windows \
  || tfx::die "Failed to build TFX-BSL from source."

# Uninstall Cython (if installed) as Beam has issues with Cython installed.
# TODO(b/130120997): Avoid installing Beam without Cython.
pip install "${TFX_BSL_WHEEL}" "${TFMD_WHEEL}" "${TENSORFLOW}" \
  && pip uninstall -y Cython \
  && pip list \
  || tfx::die "Unable to install requirements."

cd "${TFX_BSL_OUTPUT_DIR}" \
  || tfx::die "Unable to move to ${TFX_BSL_OUTPUT_DIR}."

# TODO(b/159836186): high parallelism caused problem w/ TF 2.2.
# remove --parallelism=1 after the root case is addressed.
"${PYTHON_BIN_PATH}" -m tfx_bsl.test_util.run_all_tests \
  --start_dirs="tfx_bsl" \
  --parallelism=1 \
  || tfx::die "Failed to run unit tests."
TEST_RESULT=$?

# copy wheel to ${KOKORO_ARTIFACTS_DIR}
cp "${TFMD_WHEEL}" "${KOKORO_ARTIFACTS_DIR}"
cp "${TFX_BSL_WHEEL}" "${KOKORO_ARTIFACTS_DIR}"

exit $TEST_RESULT
