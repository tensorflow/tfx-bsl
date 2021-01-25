licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

config_setting(
    name = "macos",
    constraint_values = [
        "@bazel_tools//platforms:osx",
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
