# `tfx_bsl` release notes

# Current version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Requires `apache-beam>=2.17,<3`

## Breaking Changes

## Deprecations

# Release 0.21.3

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Requires `apache-beam>=2.17,<2.18`

## Breaking Changes

## Deprecations

# Release 0.21.2

## Major Features and Improvements

## Bug Fixes and Other Changes
*  Fixed a bug in tfx_bsl.arrow.array_util.GetFlattenedArrayParentIndices that
   could cause memory corruption.

## Breaking Changes

## Deprecations

# Release 0.21.1

## Major Features and Improvements
*   Defined an abstract subclass of `TFXIO`, `RecordBasedTFXIO` to model record
    based file formats.

## Bug Fixes and Other Changes

*   Utilities in `tfx_bsl.arrow.array_util` that:

    *   previously takes `ListArray` now can also accept `LargeListArray`.
    *   previously takes StringArray/BinaryArray now can also accept
        LargeStringArray and LargeBinaryArray.

    As a result: `GetElementLengths` now returns an `Int64Array`.
    `GetFlattenedArrayParentIndices` may return an `Int64Array` or an
    `Int32Array` depending on the input type.

## Breaking Changes

## Deprecations

## Release 0.21.0

### Major Features and Improvements

*   Introduced TFXIO, the interface for
    [Standardized TFX Inputs](https://github.com/tensorflow/community/blob/master/rfcs/20191017-tfx-standardized-inputs.md)

*   Added the first implementation of TFXIO, for tf.Example on TFRecords.

### Bug Fixes and Other Changes

*   Added a test_util sub-package that contains a tool to discover and run all
    the absltests in a dir (like python's unittest discovery).
*   Requires `apache-beam>=2.17,<3`
*   Requires `pyarrow>=0.15,<0.16`
*   Requires `tensorflow>=1.15,<3`
*   Requires `tensorflow-metadata>=0.21,<0.22`.

### Breaking Changes

### Deprecations

## Release 0.15.3

### Major Features and Improvements

*  Requires `apache-beam>=2.16,<2.17` as 2.17 requires a pyarrow version
   that we don't support yet.

### Bug Fixes and Other Changes

### Breaking Changes

*   Behavior of csv_decoder.ColumnTypeInferrer was changed. A new column type,
    `ColumnType.UNKNOWN` was added to denote that the inferrer could not
    determine the type of that column (instead of making a guess of FLOAT).
    Summary of behavior change (values in the examples are from the same
    column):

    +   `<int>, <empty>`: before: `FLOAT`; after: `INT`
    +   `<empty>, ... , <empty>`: before: `FLOAT`; after: `UNKNOWN`

### Deprecations

## Release 0.15.2

### Major Features and Improvements

*   Added a (beam) utility to infer column types from a `PCollection[CSVLine]`.
*   Added a utility to parse a CSVLine into cells (conforming to RFC4180).

### Bug Fixes and Other Changes

### Breaking Changes

### Deprecations

## Release 0.15.1

### Major Features and Improvements

*   Added dependency on `tensorflow>=1.15,<2.2`. Starting from 1.15, package
    `tensorflow` comes with GPU support. Users won't need to choose between
    `tensorflow` and `tensorflow-gpu`.
    *   Caveat: `tensorflow` 2.0.0 is an exception and does not have GPU
        support. If `tensorflow-gpu` 2.0.0 is installed before installing
        `tfx-bsl`, it will be replaced with `tensorflow` 2.0.0. Re-install
        `tensorflow-gpu` 2.0.0 if needed.
*   Added dependency on `tensorflow-serving-api>=1.15,<3`.
*   Added a python PTransform, `tfx_bsl.beam.RunInference` that enables batch
    inference.

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
*   Added dependency on `apache-beam[gcp]>=2.16,<3`.
*   Added dependency on `tensorflow-metadata>=0.15,<0.16`.

### Bug Fixes and Other Changes

### Breaking Changes

### Deprecations
