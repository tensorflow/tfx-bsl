licenses(["notice"])  # Apache 2.0
load("@bazel_skylib//lib:selects.bzl", "selects")

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
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
    data = _py_extension + _public_protos,
)
