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

run_py_tests() {
  # Run python tests for specified test files or those found under the
  # provided directories.
  #
  # Usage: run_py_tests file_or_folder_1 ... [file_or_folder_N]
  #

  TEST_FOLDERS=()

  while true; do
    if [[ -z "$1" ]]; then
      break;
    else
      echo $1
      TEST_FOLDERS+=("$1")
      shift
    fi
  done

  # Locate all test files.
  PY_TEST_FILES=$(find "${TEST_FOLDERS[@]}" -name "*_test.py" | sort)
  NUM_TESTS=$(echo "${PY_TEST_FILES}" | wc -w)
  if [[ ${NUM_TESTS} == "0" ]]; then
    printf "ERROR: No Python test files are found under provided directories"
    exit 1
  fi

  printf "\nRunning ${NUM_TESTS} Python unit test files against installation...\n\n"
  printf "\nPython version to be used in test-on-install: "
  "${PYTHON_BIN_PATH}" --version
  "${PYTHON_BIN_PATH}" -c 'import sys; print(sys.path)'

  FAILED_TESTS=""

  COLOR_GREEN='\033[0;32m'
  COLOR_RED='\033[0;31m'
  COLOR_NC='\033[0m'

  for PY_TEST_FILE in ${PY_TEST_FILES}; do
    TEST_LOG_PATH=$(mktemp)

    TIMESTAMP_BEGIN=$(date +%s%N)
    "${PYTHON_BIN_PATH}" "${PY_TEST_FILE}"
    PY_TEST_EXIT_CODE=$?
    TIMESTAMP_END=$(date +%s%N)

    TIME_ELAPSED_NANOSEC=$(expr ${TIMESTAMP_END} - ${TIMESTAMP_BEGIN})
    TIME_ELAPSED_MILLISEC=$(expr ${TIME_ELAPSED_NANOSEC} / 1000000)

    if [[ ${PY_TEST_EXIT_CODE} == "0" ]]; then
      printf "Test-on-install ${COLOR_GREEN}PASSED${COLOR_NC} (Elapsed: ${TIME_ELAPSED_MILLISEC} ms): ${PY_TEST_FILE}\n"
    else
      FAILED_TESTS+="${PY_TEST_FILE}\n"
      printf "Test-on-install ${COLOR_RED}FAILED${COLOR_NC} (Elapsed: ${TIME_ELAPSED_MILLISEC} ms): ${PY_TEST_FILE}\n"
      printf "==== BEGINS test log for: ${PY_TEST_FILE} ====\n"
      cat "${TEST_LOG_PATH}"
      printf "==== ENDS test log for: ${PY_TEST_FILE} ====\n\n"
    fi

    rm -f "${TEST_LOG_PATH}"
  done
  if [ ! -z "${FAILED_TESTS}" ]; then
    echo "Some tests failed. Check the logs for details."
    exit 1
  fi
}

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
pip install ${TENSORFLOW} && \
pip install tensorflow-serving-api && \
pip install ${wheel} && \
pip uninstall -y Cython && \
run_py_tests "tfx_bsl" $@ \
  || { echo "Failed to build and run unit tests." exit 1; }

# copy wheel to ${KOKORO_ARTIFACTS_DIR}
cp ${wheel} ${KOKORO_ARTIFACTS_DIR}
