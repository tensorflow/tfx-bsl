licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

_public_protos = ["//tfx_bsl/proto:public_protos"]

_py_extension = select({
    ":windows": [
        "//tfx_bsl/cc:tfx_bsl_extension.pyd",
    ],
    "//conditions:default": [
        "//tfx_bsl/cc:tfx_bsl_extension.so",
    ],
})

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = _py_extension + _public_protos,
)
