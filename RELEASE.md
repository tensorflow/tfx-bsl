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

