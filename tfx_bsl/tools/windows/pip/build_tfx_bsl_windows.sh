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

function tfx_bsl::build_from_head_windows {
  # This script is under <repo_root>/tfx_bsl/tools/windows/pip/
  # Change into repository root.
  script_dir=$(dirname $0)
  cd ${script_dir%%tfx_bsl/tools/windows/pip}.

  # Setting up the environment variables Bazel and ./configure needs
  source "tfx_bsl/tools/windows/bazel/common_env.sh" \
    || { echo "Failed to source common_env.sh" >&2; exit 1; }

  # Recreate an empty bazelrc file under source root
  export TMP_BAZELRC=.tmp.bazelrc
  rm -f "${TMP_BAZELRC}"
  touch "${TMP_BAZELRC}"

  function cleanup {
    # Remove all options in .tmp.bazelrc
    echo "" > "${TMP_BAZELRC}"
  }
  trap cleanup EXIT

  # Enable short object file path to avoid long path issue on Windows.
  echo "startup --output_user_root=${TMPDIR}" >> "${TMP_BAZELRC}"

  if ! grep -q "import %workspace%/${TMP_BAZELRC}" .bazelrc; then
    echo "import %workspace%/${TMP_BAZELRC}" >> .bazelrc
  fi

  # Upgrade pip, setuptools and wheel packages.
  "${PYTHON_BIN_PATH}" -m pip install --upgrade pip && \
  pip install setuptools --upgrade && \
  pip install wheel --upgrade && \
  pip freeze --all || { echo "Failed to prepare pip."; exit 1; }

  pyarrow_requirement=$(python -c "fp = open('third_party/pyarrow_version.bzl', 'r'); d = {}; exec(fp.read(), d); fp.close(); print(d['PY_DEP'])") || \
    { echo "Unable to get pyarrow requirement."; exit 1; }
  pip install "${pyarrow_requirement}" && \
  ./configure.sh && \
  bazel run -c opt --copt=-DWIN32_LEAN_AND_MEAN tfx_bsl:build_pip_package || \
    { echo "Unable to build tfx_bsl."; exit 1; }
  echo "dist/*.whl"
}
