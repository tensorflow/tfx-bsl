# Description:
#   C++ libraries in tfx_common.

package(default_visibility = ["//tfx_common:__subpackages__"])

licenses(["notice"])

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_pybind_extension")

tf_pybind_extension(
    name = "tfx_common_extension",
    srcs = [
        "tfx_common_extension.cc",
    ],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    module_name = "tfx_common_extension",
    deps = [
        "//tfx_common/cc/arrow:arrow_submodule",
        "@local_config_tf//:libtensorflow_framework",
        "@pybind11",
    ],
)
