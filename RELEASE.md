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

