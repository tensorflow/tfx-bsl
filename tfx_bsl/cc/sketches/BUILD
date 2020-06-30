# Description:
#   Sketching algorithms.
load("//tfx_bsl:build_macros.bzl", "tfx_bsl_proto_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

cc_library(
    name = "kmv_sketch",
    srcs = ["kmv_sketch.cc"],
    hdrs = ["kmv_sketch.h"],
    deps = [
        ":sketches_cc_proto",
        "//tfx_bsl/cc/util:status",
        "//tfx_bsl/cc/util:status_util",
        "@arrow",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@farmhash_archive//:farmhash",
    ],
)

tfx_bsl_proto_library(
    name = "sketches_proto",
    srcs = ["sketches.proto"],
    cc_api_version = 2,
)

cc_test(
    name = "kmv_sketch_test",
    size = "small",
    srcs = ["kmv_sketch_test.cc"],
    deps = [
        ":kmv_sketch",
        "//tfx_bsl/cc/util:status",
        "@arrow",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_googletest//:gtest_main",
    ],
)
