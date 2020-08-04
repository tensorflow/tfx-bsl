# TFX Basic Shared Libraries

[![Python](https://img.shields.io/pypi/pyversions/tfx-bsl.svg?style=plastic)](https://github.com/tensorflow/tfx-bsl)
[![PyPI](https://badge.fury.io/py/tfx-bsl.svg)](https://badge.fury.io/py/tfx-bsl)

TFX Basic Shared Libraries (`tfx_bsl`) contains libraries shared by many
[TensorFlow eXtended (TFX)](https://www.tensorflow.org/tfx) components.

__Only symbols exported by sub-modules under `tfx_bsl/public` are intended for
direct use by TFX users__, including by standalone TFX library (e.g. TFDV, TFMA,
TFT) users, TFX pipeline authors and TFX component authors. Those APIs will
become stable and follow semantic versioning once `tfx_bsl` goes beyond `1.0`.

APIs under other directories should be considered internal to TFX
(and therefore there is no backward or forward compatibility guarantee for
them).

Each minor version of a TFX library or TFX itself, if it needs to
depend on `tfx_bsl`, will depend on a specific minor version of it (e.g.
`tensorflow_data_validation` 0.14.\* will depend on, and only work with,
`tfx_bsl` 0.14.\*)

## Installing from PyPI

`tfx_bsl` is available as a [PyPI package](https://pypi.org/project/tfx-bsl/).

```bash
pip install tfx-bsl
```

However it is a dependency of many TFX components and usually as a user you
don't need to install it directly.

## Build with Docker

If you want to build a TFX component from the master branch, past the latest
release, you may also have to build the latest `tfx_bsl`, as that TFX component
might have depended on new features introduced past the latest `tfx_bsl`
release.

Building from Docker is the recommended way to build `tfx_bsl` under Linux,
and is continuously tested at Google.

### 1. Install Docker

Please first install [`docker`](https://docs.docker.com/install/) and
[`docker-compose`](https://docs.docker.com/compose/install/) by following the
directions.

### 2. Clone the `tfx_bsl` repository

```shell
git clone https://github.com/tensorflow/tfx-bsl
cd tfx-bsl
```

Note that these instructions will install the latest master branch of `tfx-bsl`.
If you want to install a specific branch (such as a release branch), pass
`-b <branchname>` to the `git clone` command.

### 3. Build the pip package

Then, run the following at the project root:

```bash
sudo docker-compose build manylinux2010
sudo docker-compose run -e PYTHON_VERSION=${PYTHON_VERSION} manylinux2010
```
where `PYTHON_VERSION` is one of `{35, 36, 37}`.

A wheel will be produced under `dist/`.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Build from source

### 1. Prerequisites

#### Install NumPy

If NumPy is not installed on your system, install it now by following [these
directions](https://www.scipy.org/scipylib/download.html).

#### Install Bazel

If Bazel is not installed on your system, install it now by following [these
directions](https://bazel.build/versions/master/docs/install.html).

#### Install PyArrow

`tfx_bsl` needs to be built with specific PyArrow versions (
as indicated in third_party/pyarrow.version). Install pyarrow by following
[these directions](https://arrow.apache.org/docs/python/install.html).

When installing please make sure to specify the compatible pyarrow version. For
example:

```shell
pip install "pyarrow>=0.14.0,<0.15.0"
```

### 2. Clone the `tfx_bsl` repository

```shell
git clone https://github.com/tensorflow/tfx-bsl
cd tfx-bsl
```

Note that these instructions will install the latest master branch of `tfx_bsl`
If you want to install a specific branch (such as a release branch),
pass `-b <branchname>` to the `git clone` command.

### 3. Build the pip package

TFDV uses Bazel to build the pip package from source. Before invoking the
following commands, make sure the `python` in your `$PATH` is the one of the
target version and has NumPy and PyArrow installed.

```shell
./configure.sh
bazel run -c opt --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0 tfx_bsl:build_pip_package
```

The flag `D_GLIBCXX_USE_CXX11_ABI=0` is to use an [older std::string ABI](https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html). Which
is used by all `manylinux2010` compliant wheels (including PyArrow). If you
also build PyArrow from source without that flag, you may not need to specify
it here.

You can find the generated `.whl` file in the `dist` subdirectory.

### 4. Install the pip package

```shell
pip install dist/*.whl
```

## Supported platforms

`tfx_bsl` is tested on the following 64-bit operating systems:

  * macOS 10.12.6 (Sierra) or later.
  * Ubuntu 16.04 or later.
  * Windows 7 or later.
