<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Version 1.1.0

## Major Features and Improvements

*   Provided the SQL query ability for Apache Arrow RecordBatch. It's not
    available under Windows.

## Bug Fixes and Other Changes

*   Depends on `protobuf>=3.13,<4`.
*   Upgraded the protobuf (com_google_protobuf) to `3.13.0`.
*   Upgraded the bazel_skylib to `1.0.2` due to the upgrading of protobuf.
*   Depends on `tensorflow-metadata>=1.1,<1.2`.

## Breaking Changes

*   The minimum required OS version for the macOS is 10.14 now.

## Deprecations

*   N/A
