# Description:
#   Arrow-related utilities.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "array_util",
    srcs = ["array_util.cc"],
    hdrs = ["array_util.h"],
    deps = [
        "//tfx_bsl/cc/util:status",
        "//tfx_bsl/cc/util:status_util",
        "@arrow",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "arrow_submodule",
    srcs = ["arrow_submodule.cc"],
    hdrs = ["arrow_submodule.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//tfx_bsl/cc:__pkg__",
    ],
    deps = [
        ":array_util",
        "//tfx_bsl/cc/pybind11:arrow_casters",
        "@arrow",
        "@pybind11",
    ],
)
