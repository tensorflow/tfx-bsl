"""BUILD macros used in OSS builds."""

load("@rules_cc//cc:defs.bzl", "cc_proto_library")

def _py_proto_library_impl(ctx):
    """Implementation of py_proto_library rule."""
    proto_deps = ctx.attr.deps

    # Separate proto and Python dependencies
    all_sources = []
    py_infos = []

    for dep in proto_deps:
        if ProtoInfo in dep:
            # It's a proto_library - collect proto sources
            all_sources.extend(dep[ProtoInfo].direct_sources)
        elif PyInfo in dep:
            # It's already a py_library - collect its PyInfo for passthrough
            py_infos.append(dep[PyInfo])

    # Filter to only include sources from the workspace (not external packages)
    # We can only declare outputs in our own package
    workspace_sources = []
    for src in all_sources:
        # Filter out external sources (they start with external/ or ..)
        if not src.short_path.startswith("external/") and not src.short_path.startswith("../"):
            workspace_sources.append(src)

    # Generate Python output files from proto sources
    py_outputs = []
    for proto_src in workspace_sources:
        # Use just the basename to avoid path issues
        basename = proto_src.basename[:-6]  # Remove .proto
        py_file = ctx.actions.declare_file(basename + "_pb2.py")
        py_outputs.append(py_file)

    if py_outputs:
        # Build proto_path arguments for protoc
        # We need to include paths for workspace root and external dependencies
        proto_path_args = []

        # Add current directory to find workspace proto files
        proto_path_args.append("--proto_path=.")

        # Collect proto_path entries from all transitive dependencies
        # Use dictionary as a set (Starlark doesn't have set type)
        proto_paths = {".": True}

        # Also add directories of workspace sources so imports like "any.proto"
        # (in the same folder) resolve correctly.
        for ws in workspace_sources:
            ws_dir = "/".join(ws.short_path.split("/")[:-1])
            if ws_dir and ws_dir not in proto_paths:
                proto_paths[ws_dir] = True
                proto_path_args.append("--proto_path=" + ws_dir)

        for dep in proto_deps:
            if ProtoInfo in dep:
                # Add proto_source_root if available
                if hasattr(dep[ProtoInfo], 'proto_source_root'):
                    root = dep[ProtoInfo].proto_source_root
                    if root and root not in proto_paths:
                        proto_paths[root] = True
                        proto_path_args.append("--proto_path=" + root)

                # Also derive from file paths for more coverage
                for src in dep[ProtoInfo].transitive_sources.to_list():
                    # Use the directory containing the proto file's import root
                    # For external/com_google_protobuf/src/google/protobuf/any.proto,
                    # we want external/com_google_protobuf/src
                    if src.path.startswith("external/com_google_protobuf/"):
                        proto_path = "external/com_google_protobuf/src"
                        if proto_path not in proto_paths:
                            proto_paths[proto_path] = True
                            proto_path_args.append("--proto_path=" + proto_path)
                    elif src.path.startswith("external/"):
                        # For other external repos like tensorflow_metadata
                        # Extract external/repo_name
                        parts = src.path.split("/")
                        if len(parts) >= 2:
                            proto_path = "/".join(parts[:2])
                            if proto_path not in proto_paths:
                                proto_paths[proto_path] = True
                                proto_path_args.append("--proto_path=" + proto_path)

                    # Also add Bazel root paths
                    if src.root.path and src.root.path not in proto_paths:
                        proto_paths[src.root.path] = True
                        proto_path_args.append("--proto_path=" + src.root.path)

        # Build list of proto file paths - only include workspace sources
        proto_file_args = []
        for src in workspace_sources:
            proto_file_args.append(src.short_path)

        # Run protoc to generate Python files
        # Use ctx.bin_dir.path as the output directory root
        output_root = ctx.bin_dir.path

        ctx.actions.run(
            # Include workspace sources plus all transitive dependencies for imports
            inputs = depset(direct = workspace_sources, transitive = [
                dep[ProtoInfo].transitive_sources for dep in proto_deps if ProtoInfo in dep
            ]),
            outputs = py_outputs,
            executable = ctx.executable._protoc,
            arguments = [
                "--python_out=" + output_root,
            ] + proto_path_args + proto_file_args,
            mnemonic = "ProtocPython",
        )

    # Collect transitive sources from both generated files and Python deps
    all_transitive_sources = [depset(py_outputs)]
    all_imports = [depset([ctx.bin_dir.path])] if py_outputs else []

    for py_info in py_infos:
        all_transitive_sources.append(py_info.transitive_sources)
        if hasattr(py_info, 'imports'):
            all_imports.append(py_info.imports)

    # Return PyInfo provider so this can be used as a py_library dependency
    # Merge proto-generated files with passthrough Python dependencies
    return [
        DefaultInfo(files = depset(py_outputs)),
        PyInfo(
            transitive_sources = depset(transitive = all_transitive_sources),
            imports = depset(transitive = all_imports),
            has_py2_only_sources = False,
            has_py3_only_sources = True,
        ),
    ]

_py_proto_library_rule = rule(
    implementation = _py_proto_library_impl,
    attrs = {
        "deps": attr.label_list(
            providers = [[ProtoInfo], [PyInfo]],  # Accept either ProtoInfo OR PyInfo
            doc = "Proto library or Python library dependencies",
        ),
        "_protoc": attr.label(
            default = "@com_google_protobuf//:protoc",
            executable = True,
            cfg = "exec",
        ),
    },
    provides = [PyInfo],
)

def py_proto_library(name, deps, visibility = None, **kwargs):
    """Simple wrapper for py_proto_library using custom rule.

    This macro provides OSS compatibility for py_proto_library targets.
    """
    _py_proto_library_rule(
        name = name,
        deps = deps,
        visibility = visibility,
        **kwargs
    )

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
    _ignore = [srcs]  # Ignore srcs as custom rule uses proto_library in deps
    py_proto_library(
        name = name,
        deps = [":" + proto_library + "_proto"] + deps,
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
