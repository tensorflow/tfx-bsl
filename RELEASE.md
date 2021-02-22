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
