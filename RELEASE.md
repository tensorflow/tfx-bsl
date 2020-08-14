# `tfx_bsl` release notes

# Current version (not yet released; still in development)

## Major Features and Improvements
*  You can now build `tfx_bsl` wheel with `python setup.py bdist_wheel`. Note:
  * If you want to build a manylinux2010 wheel you'll still need
    to use Docker.
  * Bazel is still required.

## Bug Fixes and Other Changes

## Breaking changes

## Deprecations

# Version 0.23.0

## Major Features and Improvements
*  Several TFXIO symbols are made public, which means:
  * TFX users (both pipeline and component authors), and TFX libraries
    (TFDV, TFMA, TFT) users may start using these symbols.
  * We will be subject to semantic versioning once tfx_bsl goes beyond 1.0.
*  TFRecord based TFXIO implementations now support reading from multiple file
   patterns.
*  Implemented the TensorFlowDataset() interface for TFExampleRecord TFXIO.
*  Starting from this version, `tfx_bsl` has no binary dependency on `pyarrow`
   (`libarrow.so`). As a result:
   -  Package `tfx_bsl` will be able to work with a wider range of pyarrow
      versions. We will relax the version requirements in setup.py in the next
      release.
   -  Custom built `tfx_bsl` does not have to maintain ABI compatiblity with
      a specific `pyarrow` installation. Custom builds don't need to be
      manylinux-conformant.

## Bug Fixes and Other Changes

*   Starting from this version, the windows wheel will be built with VS 2015.
*   `run_all_tests` will fail with exit code -2 if no tests are discovered.
*   Stopped requiring `avro-python3`.
*   Example coders will ignore duplicate feature names in the TFMD schema (only
    the first one counts). It is a temporary measure until TFDV can check and
    prevent duplications. DO NOT rely on this behavior.
*   CsvTFXIO now allows skipping CSV headers (`set skip_header_lines`).
*   CsvTFXIO now requires `telemetry_descriptors` to construct.
*   Depends on `apache-beam[gcp]>=2.23,<3`.
*   Depends on `pyarrow>=0.17,<0.18`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,<3`.
*   Depends on `tensorflow-metadata>=0.23,<0.24`.
*   Depends on `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,<3`.

## Breaking changes

*   N/A

## Deprecations

*   Dropped Python 2.x support.

# Version 0.22.1

## Major Features and Improvements

*   Added SequenceExamplesToRecordBatchDecoder.
*   Added a TFXIO implementation for SequenceExmaples on TFRecord.
*   Added support for TensorAdapter to output tf.RaggedTensors.
*   Improved performance of tf.Example and tf.SequenceExample coders.

## Bug Fixes and Other Changes

*   Depends on `pandas>=0.24,<2`.
*   Depends on `tensorflow>=1.15,!=2.0.*,<3`.
*   Depends on `tensorflow-metadata>=0.22.2,<0.23`.
*   Removed tensor_to_arrow_test for TF 1.x as it does not support TF 1.x.

## Breaking changes

## Deprecations
*   Removed `arrow.table_util.SliceTableByRowIndices` (in favor of
    `RecordBatchTake`)
*   Removed `arrow.table_util.MergeTables` (in favor of `MergeRecordBatches`)

# Release 0.22.0

## Major Features and Improvements

*   Moved RunInference API and related protos to tfx_bsl/public directory.
*   CSV coder support for multivalent columns.
*   tf.Exmaple coder support for producing large types (LargeList, LargeBinary).
*   Added TFXIO for CSV

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.20,<3`.
*   Depends on `pyarrow>=0.16,<0.17`
*   Depends on `tensorflow-metadata>=0.22,<0.23`

## Breaking Changes

*   Renamed ModelEndpointSpec to AIPlatformPredictionModelSpec to specify remote
    model endpoint on Google Cloud Platform.
*   Renamed InferenceEndpoint to InferenceSpecType.

## Deprecations

# Release 0.21.4

## Major Features and Improvements

*   Added a tfxio.telemetry.ProfileRecordBatches, a PTransform to collect
    telemetry from Arrow RecordBatches.
*   Added remote model inference on Google Cloud Platform.

## Bug Fixes and Other Changes

*   Added `arrow.table_util.MergeRecordBatches`: similar to `MergeTables` but
    operates against `pa.RecordBatch`es.
*   Added `arrow.table_util.RecordBatchTake`: similar to
    `SliceTableByRowIndices` but operates against a `pa.RecordBatch`.
*   Requires `apache-beam>=2.17,<3`
*   Only requires `avro-python3>=1.8.1,!=1.9.2.*,<2.0.0` on Python 3.5 + MacOS
*   Requires `google-api-python-client>=1.7.11,<2`

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
