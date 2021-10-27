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

