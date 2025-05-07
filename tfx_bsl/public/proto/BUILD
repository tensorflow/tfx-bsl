# Placeholder: load py_library  # copybara:strip

# Developers: use these two macros to define c++ or python proto targets.
load("//tfx_bsl:build_macros.bzl", "tfx_bsl_proto_library", "tfx_bsl_py_proto_library")

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])

py_library(
    name = "public_protos",
    deps = [
        ":model_spec_proto_py_pb2",
        ":slicing_spec_py_pb2",
    ],
)

tfx_bsl_proto_library(
    name = "model_spec_proto",
    srcs = ["model_spec.proto"],
)

tfx_bsl_py_proto_library(
    name = "model_spec_proto_py_pb2",
    srcs = ["model_spec.proto"],
    proto_library = "model_spec_proto",
)

tfx_bsl_proto_library(
    name = "slicing_spec_proto",
    srcs = ["slicing_spec.proto"],
)

tfx_bsl_py_proto_library(
    name = "slicing_spec_py_pb2",
    srcs = ["slicing_spec.proto"],
    proto_library = "slicing_spec_proto",
)
