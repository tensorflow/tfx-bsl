# Description:
#   C++ libraries in tfx_bsl.

package(default_visibility = ["//tfx_bsl:__subpackages__"])

licenses(["notice"])

load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_pybind_extension")

tf_pybind_extension(
    name = "tfx_bsl_extension",
    srcs = [
        "tfx_bsl_extension.cc",
    ],
    module_name = "tfx_bsl_extension",
    deps = [
        "//tfx_bsl/cc/arrow:arrow_submodule",
        "@pybind11",
    ],
)
