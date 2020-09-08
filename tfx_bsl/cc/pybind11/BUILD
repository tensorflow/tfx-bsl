# Description:
#   Contains our custom extensions to pybind11.

package(default_visibility = [
    "//tfx_bsl/cc:__subpackages__",
    "//tfx_bsl/google/cc:__subpackages__",
])

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
    srcs = [
        "arrow_casters_c_abi.h",
        "arrow_casters_py_c_api.h",
        "c_abi_bridge.cc",
        "c_abi_bridge.h",
    ],
    hdrs = [
        "arrow_casters.h",
    ],
    copts = PYBIND_COPTS,
    features = PYBIND_FEATURES,
    deps = [
        "@arrow",
        "@com_google_absl//absl/strings",
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
