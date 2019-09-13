# Description:
#   Functions to convert Arrow Tables to/from Example protos.

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

filegroup(
    name = "files",
    srcs = glob(["*"]),
)

cc_library(
    name = "example_coder",
    srcs = ["example_coder.cc"],
    hdrs = ["example_coder.h"],
    deps = [
        "//third_party/arrow:thirdparty_hdrs_lib",
        "//third_party/tensorflow/core:protos_all",
        "//third_party/tensorflow_metadata/proto/v0:metadata_v0_cc_proto",
        "@arrow",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "example_coder_test",
    srcs = ["example_coder_test.cc"],
    deps = [
        ":example_coder",
        "//testing/base/public:gunit_main",
        "//third_party/arrow:thirdparty_hdrs_lib",
        "//third_party/tensorflow_metadata/proto/v0:metadata_v0_cc_proto",
        "@arrow",
        "@local_config_tf//:tf_header_lib",
    ],
)
