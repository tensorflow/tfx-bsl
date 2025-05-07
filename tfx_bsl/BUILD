licenses(["notice"])
load("@bazel_skylib//lib:selects.bzl", "selects")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")


exports_files(
    [
        "build_macros.bzl",
    ],
)

bzl_library(
        name = "build_macros",
        srcs = ["build_macros.bzl"],
        deps = ["//tools/build_defs/proto/cpp:cc_proto_library.bzl",
                "//tools/build_defs/proto/cpp:proto_names_to_cc_proto_names.bzl",
        ],
        visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@platforms//os:windows",
    ],
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "macos_arm64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin_arm64",
    },
)

selects.config_setting_group(
    name = "macos",
    match_any = [
        ":macos_x86_64",
        ":macos_arm64",
     ],
)

_public_protos = ["//tfx_bsl/public/proto:public_protos"]

_py_extension = select({
    ":windows": [
        "//tfx_bsl/cc:tfx_bsl_extension.pyd",
    ],
    "//conditions:default": [
        "//tfx_bsl/cc:tfx_bsl_extension.so",
    ],
})

sh_binary(
    name = "move_generated_files",
    srcs = ["move_generated_files.sh"],
    data = _public_protos + _py_extension,
)
