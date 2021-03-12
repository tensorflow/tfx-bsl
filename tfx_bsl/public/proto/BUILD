# Developers: use these two macros to define c++ or python proto targets.
load("//tfx_bsl:build_macros.bzl", "tfx_bsl_proto_library", "tfx_bsl_py_proto_library")

package(default_visibility = [
    "//visibility:public",
])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "public_protos",
    srcs = ["model_spec_pb2.py"],
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
