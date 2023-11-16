<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Current Version (not yet released; still in development)

## Major Features and Improvements

## Bug Fixes and Other Changes

*   Bump the macOS version on which TFX-BSL is tested to Ventura (previously was
    Monterey).
* Bumps the pybind11 version to 2.11.1

## Breaking Changes

## Deprecations

*   Deprecated python 3.8 support.

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

# Version 1.13.0

## Major Features and Improvements

*   `RaggedTensor`s can now be automatically inferred for variable length
    features by setting `represent_variable_length_as_ragged=true` in TFMD
    schema.

## Bug Fixes and Other Changes

*   Bumped the mininum bazel version required to build `tfx_bsl` to 5.3.0.
*   `RecordBatchToExamplesEncoder` now encodes arrays representing
    `RaggedTensor`s in a way that is consistent with `tf.io.parse_example`. Note
    that this change is backwards compatible with `ExamplesToRecordBatchDecoder`
    and the decoding workflow as well.
*   Added utility functions for interacting with Arrow arrays and record
    batches.
*   Depends on `numpy~=1.22.0`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.7 support.

# Version 1.12.0

## Major Features and Improvements

*   `InferTensorRepresentationsFromSchema`, `TensorAdapter` and
    `TensorsToRecordBatchConverter` now support `SparseTensor`s with unknown
    `dense_shape`.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=2.11,<3`

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.11.0

## Major Features and Improvements

*   `TensorAdapter` now processes `tf.RaggedTensor`s in TF 2 ~10x faster.
*   `InferTensorRepresentationsFromSchema` now infers `RaggedTensor`s for
    `STRUCT` features.
*   `TFSequenceExampleRecord` now supports schemas with features not covered or
    partially covered by `TensorRepresentation`s.

*   This is the last version that supports TensorFlow 1.15.x. TF 1.15.x support
    will be removed in the next version. Please check the
    [TF2 migration guide](https://www.tensorflow.org/guide/migrate) to migrate
    to TF2.

## Bug Fixes and Other Changes

*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.10,<3`
*   Depends on `protobuf>=3.13,<4`
*   Various `TFXIO` implementations now infer `TensorRepresentations` for
    provided schema `Features` even if some `TensorRepresentations` are provided
    as well.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.10.0

# Major Features and Improvements

*   `ExamplesToRecordBatchDecoder` is now picklable.
*   `ParquetTFXIO` can now be used as `RecordBasedTFXIO`.
*   Introduces `CreateTfSequenceExampleParserConfig` that takes TFMD schema as
    input and produces configs for `tf.SequenceExample` parsing.
*   `TFSequenceExampleRecord` can now produce an equivalent tf.data.Dataset.
*   Introduces an api: `CreateModelHandler` that produces a model handler
    suitable for apache_beam.ml.inference.
*   Quantiles sketch supports GetQuantilesAndCumulativeWeights, which returns
    the sum of weights in each quantiles bin along with boundaries.

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.40,<3`.
*   Depends on `pyarrow>=6,<7`.
*   Depends on `tensorflow-metadata>=1.10,<1.11`.
*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.9,<3`.

## Breaking Changes

*   GenerateQuantiles removed from weighted_quantiles_summary.h and replaced
    with GenerateQuantilesAndCumulativeWeights.

## Deprecations

*   N/A

# Version 1.9.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=1.9,<1.10`.
*   Depends on `tensorflow>=1.15.5,<2` or `tensorflow>=2.9,<3`.
*   Depends on `protobuf>=3.13,<3.21`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.8.0

## Major Features and Improvements

*   Introduced `RunInferencePerModel` PTransform, which is a vectorized variant
    of `RunInference` (useful for ensembles).
*   Introduced `ParquetTFXIO` that allows reading data from Parquet files in
    `pyarrow.RecordBatch` format.
*   From this version we will be releasing python 3.9 wheels.
*   Depends on `apache-beam[gcp]>=2.38,<3`.

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=1.8,<1.9`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.7.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.36,<3`.
*   Depends on `tensorflow-metadata>=1.7,<1.8`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on
    `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Added a TFXIO where the user defines the beam source.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.6.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Fixes a bug when `TensorsToRecordBatchConverter` could not handle
    `tf.RaggedTensor`s with uniform inner dimensions in TF 1.15.
*   Depends on `apache-beam[gcp]>=2.35,<3`.
*   Depends on `tensorflow-metadata>=1.6,<1.7`.
*   Depends on `numpy>=1.16,<2`.
*   Depends on `absl-py>=0.9,<2.0.0`.
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.5.0

## Major Features and Improvements

*   `TensorsToRecordBatchConverter` can now handle `tf.RaggedTensor`s with
    uniform inner dimensions.

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.34,<3`.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on
    `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,<3`.
*   Depends on `tensorflow-metadata>=1.5,<1.6`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.4.0

## Major Features and Improvements

*   Introduces `RecordBatchToExamplesEncoder` that supports encoding nested
    `pyarrow.large_list()`s representing `tf.RaggedTensor`s.

## Bug Fixes and Other Changes

*   Register s2t ops before loading decoder in record_to_tensor_tfxio if
    struct2tensor is installed.
*   Depends on `pyarrow>=1,<6`.
*   Depends on `tensorflow-metadata>=1.4,<1.5`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.6 support.

# Version 1.3.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   `QuantilesSketch` now ignores NaNs in input values and weights. Previously,
    NaNs would lead to incorrect quantiles calculation.
*   Fixes a bug when `MisraGriesSketch` would discard excessive number of
    elements during `AddValues` and `Compress` and output fewer elements than
    requested.
*   Depends on
    `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.
*   Depends on
    `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.2.0

## Major Features and Improvements

*   Added support for converting `tf.compat.v1.ragged.RaggedTensorValue`s to
    `TensorsToRecordBatchConverter`.
*   Depends on `apache-beam[gcp]>=2.31,<3`.
*   Depends on `tensorflow-metadata>=1.2,<1.3`.

## Bug Fixes and Other Changes

*   N/A

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.1.1

## Major Features and Improvements

*   N/A

## Bug fixes and other Changes

*   Depends on `google-cloud-bigquery>>=1.28.0,<2.21`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 1.1.0

## Major Features and Improvements

*   Provided the SQL query ability for Apache Arrow RecordBatch. It's not
    available under Windows.

## Bug Fixes and Other Changes

*   Depends on `protobuf>=3.13,<4`.
*   Upgraded the protobuf (com_google_protobuf) to `3.13.0`.
*   Upgraded the bazel_skylib to `1.0.2` due to the upgrading of protobuf.
*   Depends on `tensorflow-metadata>=1.1,<1.2`.
*   More documentation is added for the SequenceExample decoder. It's available
    at `tfx_bsl/coders/README.md`.

## Breaking Changes

*   The minimum required OS version for the macOS is 10.14 now.

## Deprecations

*   N/A

# Version 1.0.0

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.29,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on
    `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,<3`.
*   Depends on `tensorflow-metadata>=1.0,<1.1`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.30.0

## Major Features and Improvements

*  Misra-Gries sketch: added support for replacing large string blobs with
   a configurable placeholder, and replacing invalid utf-8 sequences with
   a configurable placeholder.

## Bug Fixes and Other Changes

*   Depends on `tensorflow-metadata>=0.30,<0.31`.

## Breaking Changes

*  Removed `tfx_bsl.beam.shared`. It is now available in Apache Beam.
   Use `apache_beam.utils.shared` instead.

## Deprecations

*   N/A

# Version 0.29.0

## Major Features and Improvements

*   Add RawRecordTensorFlowDataset interface to record based tfxios.
*   TensorToArrowConverter now can handle generic SparseTensors (>=3-d).
*   Added `RecordToTensorTFXIO.DecodeFunction()` to get the decoder as a TF
    function.

## Bug Fixes and Other Changes

*   Depends on `absl-py>=0.9,<0.13`.
*   Depends on `tensorflow-metadata>=0.29,<0.30`.
*   Bumped the mininum bazel version required to build `tfx_bsl` to 3.7.2.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.28,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.28.0

## Major Features and Improvements

*   RunInference can now be applied on serialized tf.train.{Example,
    SequenceExample} for all methods as well as any other kind of serialized
    structure for the Predict method.
*   RunInference can now operate on PCollection[K, V] in a key-forwarding mode
    (whereby the key is left unchanged while inference is performed on the
    value).
*   RunInference is now more performant.

## Bug Fixes and Other Changes

*   Depends on `numpy>=1.16,<1.20`.
*   Depends on `tensorflow-metadata>=0.28,<0.29`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.1

*   This is a bug fix only version, which modified the dependencies.

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Fix in the `tensorflow-serving-api` version constraint.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.27.0

## Major Features and Improvements

*  `tfx_bsl.public.tfxio.TFGraphRecordDecoder` is now a public API.

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.27,<3`.
*   Depends on `pyarrow>=1,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.
*   Depends on `tensorflow-metadata>=0.27,<0.28`.
*   Depends on `tensorflow-serving>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,<3`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.1

*   This is a bug fix only version, which modified the dependencies.

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.25,!=2.26.*,<3`.
*   Depends on `tensorflow>=1.15.2,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.
*   Depends on `tensorflow-serving>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.4.*,<3`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.26.0

## Major Features and Improvements

*   `.TensorFlowDataset` interface is available in RawTfRecord TFXIO.

## Bug Fixes and Other Changes

*   Fix TFExampleRecord TFXIO's TensorFlowDataset output key's to match the
    tensor representation's tensor name (Previously this assumed the user
    provided a tensor name that is the same as the feature name).
*   Add utility in tensor_representation_util.py to get source columns from a
    tensor representation.
*   Depends on `tensorflow-metadata>=0.26,<0.27`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.25.0

## Major Features and Improvements

*   Add `RecordBatches` interface to TFXIO. This interface returns an iterable
    of record batches, which can be used outside of Apache Beam or TensorFlow to
    access data.
*   From this release TFX-BSL will also be hosting nightly packages on
    https://pypi-nightly.tensorflow.org. To install the nightly package use the
    following command:

    ```
    pip install --extra-index-url https://pypi-nightly.tensorflow.org/simple tfx-bsl
    ```

    Note: These nightly packages are unstable and breakages are likely to
    happen. The fix could often take a week or more depending on the complexity
    involved for the wheels to be available on the PyPI cloud service. You can
    always use the stable version of TFX-BSL available on PyPI by running the
    command `pip install tfx-bsl` .

## Bug Fixes and Other Changes
*  TensorToArrow returns LargeListArray/LargeBinaryArray in place of
   ListArray/BinaryArray.
*  array_util.IndexIn now supports LargeBinaryArray inputs.
*  Depends on `apache-beam[gcp]>=2.25,<3`.
*  Depends on `tensorflow-metadata>=0.25,<0.26`.

## Breaking changes

*  Coders (Example, CSV) do not support outputting ListArray/BinaryArray any
   more. They can only output LargeListArray/LargeBinaryArray.

## Deprecations

*   N/A

# Version 0.24.1

## Major Features and Improvements

*   N/A

## Bug Fixes and Other Changes

*   Depends on `apache-beam[gcp]>=2.24,<3`.

## Breaking changes

*   N/A

## Deprecations

*   N/A

# Version 0.24.0

## Major Features and Improvements

*   You can now build `tfx_bsl` wheel with `python setup.py bdist_wheel`. Note:
    *   If you want to build a manylinux2010 wheel you'll still need to use
        Docker.
    *   Bazel is still required.
*   You can now build manylinux2010 `tfx_bsl` wheel for Python 3.8.
*   From this version we will be releasing python 3.8 wheels.

## Bug Fixes and Other Changes

*   Stopped depending on `six`.
*   Depends on `absl-py>=0.9,<0.11`.
*   Depends on `pandas>=1.0,<2`.
*   Depends on `protobuf>=3.9.2,<4`.
*   Depends on `tensorflow-metadata>=0.24,<0.25`.

## Breaking changes

*   N/A

## Deprecations

*   Deprecated py3.5 support.

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
*   Note: We plan to remove Python 3.5 support after this release.

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
