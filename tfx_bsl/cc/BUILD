# Description:
#   C++ libraries in tfx_bsl.
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_pybind_extension")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

# DO NOT directly depend on this target. Depend on tfx_bsl_extension_lib instead.
tf_pybind_extension(
    name = "tfx_bsl_extension",
    srcs = [
        "tfx_bsl_extension.cc",
    ],
    module_name = "tfx_bsl_extension",
    visibility = [
        "//tfx_bsl:__pkg__",
    ],
    deps = [
        "//tfx_bsl/cc/arrow:arrow_submodule",
        "//tfx_bsl/cc/coders:coders_submodule",
        "@pybind11",
    ],
)

# Because the extension imports pyarrow, it needs to depend on it so it's available
# for import.
py_library(
    name = "tfx_bsl_extension_lib",
    srcs = ["__init__.py"],
    visibility = ["//tfx_bsl:__subpackages__"],
    deps = [":tfx_bsl_extension"],
)
