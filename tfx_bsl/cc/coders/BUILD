# Description:
#   Functions to convert Arrow Tables to/from Example protos.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "example_coder",
    srcs = [
        "example_decoder.cc",
        "example_encoder.cc",
    ],
    hdrs = ["example_coder.h"],
    deps = [
        "@com_google_protobuf//:protobuf_lite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:variant",
        "@arrow",
        "//tfx_bsl/cc/util:status_util",
        "//tfx_bsl/cc/arrow:array_util",
        # In OSS, this target may appear not to exist. It is created from a patch:
        # //third_party/tensorflow_expose_example_proto.patch
        "@org_tensorflow_no_deps//tensorflow/core/example:example_proto",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "example_numpy_decoder",
    srcs = ["example_numpy_decoder.cc"],
    hdrs = ["example_numpy_decoder.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@local_config_python//:numpy_headers",
        "@local_config_python//:python_headers",
        "@org_tensorflow_no_deps//tensorflow/core/example:example_proto",
    ],
)

cc_library(
    name = "coders_submodule",
    srcs = ["coders_submodule.cc"],
    hdrs = ["coders_submodule.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//tfx_bsl/cc:__pkg__",
    ],
    deps = [
        ":example_coder",
        ":example_numpy_decoder",
        "//tfx_bsl/cc/pybind11:absl_casters",
        "//tfx_bsl/cc/pybind11:arrow_casters",
        "@arrow",
        "@com_google_absl//absl/status",
        "@pybind11",
    ],
)
