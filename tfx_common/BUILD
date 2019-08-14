licenses(["notice"])  # Apache 2.0

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = select({
        ":windows": [
            "//tfx_common/cc:tfx_common_extension.pyd",
        ],
        "//conditions:default": [
            "//tfx_common/cc:tfx_common_extension.so",
        ],
    }),
)
