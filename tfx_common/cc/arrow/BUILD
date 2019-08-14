# Description:
#   Arrow-related utilities.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "array_util",
    srcs = ["array_util.cc"],
    hdrs = ["array_util.h"],
    deps = [
        "@arrow",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:tf_header_lib",
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
        "//tfx_common/cc:__pkg__",
    ],
    deps = [
        ":array_util",
        "//tfx_common/cc/pybind11:arrow_casters",
        "@arrow",
        "@local_config_tf//:tf_header_lib",
        "@pybind11",
    ],
)
