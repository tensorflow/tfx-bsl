"""BUILD macros used in OSS builds."""

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_cc//cc:defs.bzl", "cc_proto_library")

def tfx_bsl_proto_library(name, **kwargs):
    """Google proto_library and cc_proto_library.

    Args:
        name: Name of the cc proto library.
        **kwargs: Keyword arguments to pass to the proto libraries."""
    well_known_protos = [
        "@com_google_protobuf//:any_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
        "@com_google_protobuf//:struct_proto",
        "@com_google_protobuf//:empty_proto",
        "@com_google_protobuf//:wrappers_proto",
    ]
    kwargs["deps"] = kwargs.get("deps", []) + well_known_protos
    native.proto_library(name = name + "_proto", **kwargs)  # buildifier: disable=native-proto
    cc_proto_kwargs = {
        "deps": [":" + name + "_proto"],
    }
    if "visibility" in kwargs:
        cc_proto_kwargs["visibility"] = kwargs["visibility"]
    if "testonly" in kwargs:
        cc_proto_kwargs["testonly"] = kwargs["testonly"]
    if "compatible_with" in kwargs:
        cc_proto_kwargs["compatible_with"] = kwargs["compatible_with"]
    cc_proto_library(name = name, **cc_proto_kwargs)

def tfx_bsl_py_proto_library(
        name,
        proto_library,
        srcs = [],
        deps = [],
        visibility = None,
        testonly = 0):
    """Opensource py_proto_library."""
    _ignore = [proto_library]  # buildifier: disable=unused-variable
    py_proto_library(
        name = name,
        srcs = srcs,
        srcs_version = "PY3",
        deps = deps,  # ["@com_google_protobuf//:well_known_types_py_pb2"] +
        default_runtime = "@com_google_protobuf//:protobuf_python",
        protoc = "@com_google_protobuf//:protoc",
        visibility = visibility,
        testonly = testonly,
    )

def tfx_bsl_pybind_extension(
        name,
        srcs,
        module_name,
        deps = [],
        visibility = None):
    """Builds a generic Python extension module.

    Args:
      name: Name of the target.
      srcs: C++ source files.
      module_name: Ignored.
      deps: Dependencies.
      visibility: Visibility.
    """
    _ignore = [module_name]  # buildifier: disable=unused-variable
    p = name.rfind("/")
    if p == -1:
        sname = name
        prefix = ""
    else:
        sname = name[p + 1:]
        prefix = name[:p + 1]
    so_file = "%s%s.so" % (prefix, sname)
    pyd_file = "%s%s.pyd" % (prefix, sname)
    exported_symbols = [
        "init%s" % sname,
        "init_%s" % sname,
        "PyInit_%s" % sname,
    ]

    exported_symbols_file = "%s-exported-symbols.lds" % name
    version_script_file = "%s-version-script.lds" % name

    exported_symbols_output = "\n".join(["_%s" % symbol for symbol in exported_symbols])
    version_script_output = "\n".join([" %s;" % symbol for symbol in exported_symbols])

    native.genrule(
        name = name + "_exported_symbols",
        outs = [exported_symbols_file],
        cmd = "echo '%s' >$@" % exported_symbols_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.genrule(
        name = name + "_version_script",
        outs = [version_script_file],
        cmd = "echo '{global:\n%s\n local: *;};' >$@" % version_script_output,
        output_licenses = ["unencumbered"],
        visibility = ["//visibility:private"],
    )

    native.cc_binary(
        name = so_file,
        srcs = srcs,
        copts = [
            "-fno-strict-aliasing",
            "-fexceptions",
        ] + select({
            "//tfx_bsl:windows": [],
            "//conditions:default": [
                "-fvisibility=hidden",
            ],
        }),
        linkopts = select({
            "//tfx_bsl:macos": [
                # TODO: the -w suppresses a wall of harmless warnings about hidden typeinfo symbols
                # not being exported.  There should be a better way to deal with this.
                # "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
                "-Wl,-w",
                "-Wl,-exported_symbols_list,$(location %s)" % exported_symbols_file,
            ],
            "//tfx_bsl:windows": [],
            "//conditions:default": [
                "-Wl,--version-script",
                "$(location %s)" % version_script_file,
            ],
        }),
        deps = deps + [
            exported_symbols_file,
            version_script_file,
        ],
        features = ["-use_header_modules"],
        linkshared = 1,
        visibility = visibility,
    )
    native.genrule(
        name = name + "_pyd_copy",
        srcs = [so_file],
        outs = [pyd_file],
        cmd = "cp $< $@",
        output_to_bindir = True,
        visibility = visibility,
    )
    native.py_library(
        name = name,
        data = select({
            "//tfx_bsl:windows": [pyd_file],
            "//conditions:default": [so_file],
        }),
        srcs_version = "PY3",
        visibility = visibility,
    )
