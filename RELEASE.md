# `tfx_bsl` release notes

## Current version (not yet released; still in development)

### Major Features and Improvements

### Bug Fixes and Other Changes

### Breaking Changes

### Deprecations

## Release 0.15.0

### Major Features and Improvements

*   Added a tf.Example <-> Arrow coder.
*   Added a tf.Example -> `Dict[str, np.ndarray]` coder (this is a legacy format
    used by some TFX components).
*   Added some common Arrow utilities (`tfx_bsl.arrow.array_util`).
*   Added a python class, `tfx_bsl.beam.Shared` that helps sharing a single
    instance of object across multiple threads.
*   Added a python PTransform, `tfx_bsl.beam.RunInference` that enables batch
    inference.

### Bug Fixes and Other Changes

### Breaking Changes

### Deprecations
