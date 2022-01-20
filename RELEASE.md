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

