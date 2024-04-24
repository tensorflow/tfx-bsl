# Version 1.15.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Bumped the mininum bazel version required to build `tfx_bsl` to 6.1.0.
*   Bump the macOS version on which TFX-BSL is tested to Ventura (previously was
    Monterey).
*   Bumps the pybind11 version to 2.11.1
*   Depends on `tensorflow 2.15`
*   Depends on `apache-beam[gcp]>=2.53.0,<3` for Python 3.11 and on 
    `apache-beam[gcp]>=2.47.0,<3` for 3.9 and 3.10.
*   Depends on `protobuf>=4.25.2,<5` for Python 3.11 and on `protobuf>3.20.3,<5`
    for 3.9 and 3.10.
*   Deprecated Windows support.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.8 support.

