# Utilities shared by tfx_bsl C++ code.

package(default_visibility = [
    "//tfx_bsl/cc:__subpackages__",
    "//tfx_bsl/google/cc:__subpackages__",
])

licenses(["notice"])

cc_library(
    name = "status_util",
    hdrs = ["status_util.h"],
    deps = [
        "@arrow",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
    ],
)

cc_library(
    name = "utf8",
    srcs = ["utf8.cc"],
    hdrs = ["utf8.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "utf8_test",
    srcs = ["utf8_test.cc"],
    deps = [
        ":utf8",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
