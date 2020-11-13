# Utilities shared by tfx_bsl C++ code.

package(default_visibility = [
    "//tfx_bsl/cc:__subpackages__",
    "//tfx_bsl/google/cc:__subpackages__",
])

licenses(["notice"])

cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "status_util",
    hdrs = ["status_util.h"],
    deps = [
        ":status",
        "@arrow",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
    ],
)
