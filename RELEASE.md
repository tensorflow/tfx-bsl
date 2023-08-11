# Version 1.14.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Bumped the Ubuntu version on which TFX-BSL is tested to 20.04 (previously
    16.04).
*   Adds `order_on_tie` parameter to `MisraGriesSketch` to specify the order
    of items in case their counts are tied.
*   Use @platforms instead of @bazel_tools//platforms to specify constraints in
    OSS build.
*   Depends on `pyarrow>=10,<11`.
*   Depends on `apache-beam>=2.47,<3`.
*   Depends on `numpy>=1.22.0`.
*   Depends on `tensorflow>=2.13,<3`

## Breaking Changes

*   N/A

## Deprecations

*   N/A

