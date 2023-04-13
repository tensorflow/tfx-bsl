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
*   Depends on `numpy~=1.22.0`.

## Breaking Changes

*   N/A

## Deprecations

*   Deprecated python 3.7 support.

