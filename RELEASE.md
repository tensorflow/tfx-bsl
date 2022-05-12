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
*   Depends on
    `tensorflow>=1.15.5,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.
*   Depends on
    `tensorflow-serving-api>=1.15,!=2.0.*,!=2.1.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,<3`.

## Breaking Changes

*   N/A

## Deprecations

*   N/A

