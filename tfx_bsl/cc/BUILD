# Description:
#   C++ libraries in tfx_bsl.
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_pybind_extension")

package(default_visibility = ["//tfx_bsl:__subpackages__"])

licenses(["notice"])

tf_pybind_extension(
    name = "tfx_bsl_extension",
    srcs = [
        "tfx_bsl_extension.cc",
    ],
    module_name = "tfx_bsl_extension",
    visibility = ["//tfx_bsl:__subpackages__"],
    deps = [
        "//tfx_bsl/cc/arrow:arrow_submodule",
        "//tfx_bsl/cc/coders:coders_submodule",
        "@pybind11",
    ],
)
