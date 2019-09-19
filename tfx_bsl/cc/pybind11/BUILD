# Description:
#   Contains our custom extensions to pybind11.

package(default_visibility = ["//tfx_bsl/cc:__subpackages__"])

licenses(["notice"])

PYBIND_COPTS = [
    "-fexceptions",
]

PYBIND_FEATURES = [
    "-use_header_modules",  # Required for pybind11.
    "-parse_headers",
]

cc_library(
    name = "arrow_casters",
    hdrs = ["arrow_casters.h"],
    copts = PYBIND_COPTS,
    features = PYBIND_FEATURES,
    deps = [
        "@arrow",
        "@pybind11",
    ],
)

cc_library(
    name = "absl_casters",
    hdrs = ["absl_casters.h"],
    copts = PYBIND_COPTS,
    features = PYBIND_FEATURES,
    deps = [
        "@com_google_absl//absl/strings",
        "@pybind11",
    ],
)
