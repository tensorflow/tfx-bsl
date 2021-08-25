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
