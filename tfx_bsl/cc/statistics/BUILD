# Description:
#   Provides functionality for interacting with TFDV stats protos.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

# TODO(b/202910677): Make this public at some point.
cc_library(
    name = "merge_util",
    srcs = ["merge_util.cc"],
    hdrs = ["merge_util.h"],
    deps = [
        "//tfx_bsl/cc/util:status_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:str_format",
    ],
)

cc_library(
    name = "statistics_submodule",
    srcs = ["statistics_submodule.cc"],
    hdrs = ["statistics_submodule.h"],
    copts = [
        "-fexceptions",
    ],
    features = ["-use_header_modules"],
    visibility = [
        "//tfx_bsl/cc:__pkg__",
    ],
    deps = [
        ":merge_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/types:optional",
        "@pybind11",
    ],
)

cc_test(
    name = "merge_util_test",
    srcs = ["merge_util_test.cc"],
    deps = [
        ":merge_util",
        "//tfx_bsl/cc/util:status_util",
        "@com_github_tensorflow_metadata//tensorflow_metadata/proto/v0:cc_metadata_v0_proto_cc",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)
