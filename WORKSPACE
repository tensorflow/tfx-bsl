workspace(name = "tfx_bsl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

local_repository(
    name = "rules_java",
    path = "third_party/rules_java",
)

local_repository(
    name = "compatibility_proxy",
    path = "third_party/dummy_compatibility_proxy",
)

http_archive(
    name = "google_bazel_common",
    sha256 = "82a49fb27c01ad184db948747733159022f9464fc2e62da996fa700594d9ea42",
    strip_prefix = "bazel-common-2a6b6406e12208e02b2060df0631fb30919080f3",
    urls = ["https://github.com/google/bazel-common/archive/2a6b6406e12208e02b2060df0631fb30919080f3.zip"],
)

################################################################################
# Generic Bazel Support                                                        #
################################################################################

http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
)

http_archive(
    name = "rules_python",
    sha256 = "098ba13578e796c00c853a2161f382647f32eb9a77099e1c88bc5299333d0d6e",
    strip_prefix = "rules_python-1.9.0",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/1.9.0/rules_python-1.9.0.tar.gz",
)
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

local_repository(
    name = "xla",
    path = "third_party/dummy_xla",
)

http_archive(
    name = "rules_proto",
    sha256 = "6fb6767d1bef535310547e03247f7518b03487740c11b6c6adb7952033fe1295",
    strip_prefix = "rules_proto-6.0.2",
    url = "https://github.com/bazelbuild/rules_proto/releases/download/6.0.2/rules_proto-6.0.2.tar.gz",
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load("@rules_proto//proto:setup.bzl", "rules_proto_setup")

rules_proto_setup()

load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

# Install version 0.9.0 of rules_foreign_cc, as default version causes an
# invalid escape sequence error to be raised, which can't be avoided with
# the --incompatible_restrict_string_escapes=false flag (flag was removed in
# Bazel 5.0).
RULES_FOREIGN_CC_VERSION = "0.9.0"

http_archive(
    name = "rules_foreign_cc",
    patch_tool = "patch",
    patches = ["//third_party:rules_foreign_cc.patch"],
    sha256 = "2a4d07cd64b0719b39a7c12218a3e507672b82a97b98c6a89d38565894cf7c51",
    strip_prefix = "rules_foreign_cc-%s" % RULES_FOREIGN_CC_VERSION,
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/%s.tar.gz" % RULES_FOREIGN_CC_VERSION,
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "6e09bbc950ba60c3a7b30280210cd285af8d7d8ed5e0a6ed101c72aff22e8d88",
    strip_prefix = "protobuf-6.31.1",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/refs/tags/v6.31.1.zip"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

# Use the last commit on the relevant release branch to update.
# LINT.IfChange(arrow_archive_version)
ARROW_COMMIT = "347a88ff9d20e2a4061eec0b455b8ea1aa8335dc"  # 6.0.1
# LINT.ThenChange(third_party/arrow.BUILD:arrow_gen_version)

# `shasum -a 256` can be used to get `sha256` from the downloaded archive on
# Linux.
http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    patches = ["//third_party:arrow.patch"],
    patch_cmds = [
        "python3 -c \"import sys; src = open('cpp/thirdparty/flatbuffers/include/flatbuffers/flatbuffers.h').read(); open('cpp/thirdparty/flatbuffers/include/flatbuffers/flatbuffers.h', 'w').write(src.replace('buf_ = other.buf_;', '// buf_ = other.buf_;'))\"",
    ],
    sha256 = "55fc466d0043c4cce0756bc18e1e62b3233be74c9afe8dc0d18420b9a5fd9714",
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
)

http_archive(
    name = "com_google_absl",
    sha256 = "f50e5ac311a81382da7fa75b97310e4b9006474f9560ac46f54a9967f07d4ae3",
    strip_prefix = "abseil-cpp-20240722.0",
    urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20240722.0.tar.gz"],
)

http_archive(
    name = "com_google_googletest",
    sha256 = "8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7",
    strip_prefix = "googletest-1.14.0",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz"],
)

TFMD_COMMIT = "404805761e614561cceedc429e67c357c62be26d"  # 1.17.1

http_archive(
    name = "com_github_tensorflow_metadata",
    sha256 = "1b72e0e5085812cd9b19e004a381b544542f9545a081f0f738c5ed6b8bb886a2",
    strip_prefix = "metadata-%s" % TFMD_COMMIT,
    urls = ["https://github.com/tensorflow/metadata/archive/%s.zip" % TFMD_COMMIT],
)

# TODO(b/177694034): Follow the new format for tensorflow import after TF 2.5.
#here
TENSORFLOW_VERSION = "2.21.0"

http_archive(
    name = "org_tensorflow_no_deps",
    patch_cmds = [
        "echo 'proto_library(' > tensorflow/core/example/BUILD",
        "echo '    name = \"example_proto_genproto\",' >> tensorflow/core/example/BUILD",
        "echo '    srcs = [' >> tensorflow/core/example/BUILD",
        "echo '        \"example.proto\",' >> tensorflow/core/example/BUILD",
        "echo '        \"feature.proto\",' >> tensorflow/core/example/BUILD",
        "echo '    ],' >> tensorflow/core/example/BUILD",
        "echo '    visibility = [\"//visibility:public\"],' >> tensorflow/core/example/BUILD",
        "echo ')' >> tensorflow/core/example/BUILD",
        "echo '' >> tensorflow/core/example/BUILD",
        "echo 'cc_proto_library(' >> tensorflow/core/example/BUILD",
        "echo '    name = \"example_proto\",' >> tensorflow/core/example/BUILD",
        "echo '    visibility = [\"//visibility:public\"],' >> tensorflow/core/example/BUILD",
        "echo '    deps = [\":example_proto_genproto\"],' >> tensorflow/core/example/BUILD",
        "echo ')' >> tensorflow/core/example/BUILD",
    ],
    sha256 = "ef3568bb4865d6c1b2564fb5689c19b6b9a5311572cd1f2ff9198636a8520921",
    strip_prefix = "tensorflow-%s" % TENSORFLOW_VERSION,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/v%s.tar.gz" % TENSORFLOW_VERSION,
    ],
)

PYBIND11_COMMIT = "8a099e44b3d5f85b20f05828d919d2332a8de841"  # 2.11.1

http_archive(
    name = "pybind11",
    build_file = "//third_party:pybind11.BUILD",
    sha256 = "8f4b7f28d214e36301435c055076c36186388dc9617117802cba8a059347cb00",
    strip_prefix = "pybind11-%s" % PYBIND11_COMMIT,
    urls = ["https://github.com/pybind/pybind11/archive/%s.zip" % PYBIND11_COMMIT],
)

load("//third_party:python_configure.bzl", "local_python_configure")

local_python_configure(name = "local_config_python")

http_archive(
    name = "com_google_farmhash",
    build_file = "//third_party:farmhash.BUILD",
    sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",  # SHARED_FARMHASH_SHA
    strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
    urls = [
        "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
    ],
)

################################################################################
# Google APIs protos                                                           #
################################################################################
http_archive(
    name = "com_google_googleapis",
    patch_args = ["-p1"],
    patches = ["//third_party:googleapis.patch"],
    sha256 = "28e7fe3a640dd1f47622a4c263c40d5509c008cc20f97bd366076d5546cccb64",
    strip_prefix = "googleapis-4ce00b00904a7ce1df8c157e54fcbf96fda0dc49",
    url = "https://github.com/googleapis/googleapis/archive/4ce00b00904a7ce1df8c157e54fcbf96fda0dc49.tar.gz",
)

load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,
    go = True,
)

###############################################################################
# Gazelle Support                                                             #
###############################################################################

_rules_go_version = "v0.48.1"

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "b2038e2de2cace18f032249cb4bb0048abf583a36369fa98f687af1b3f880b26",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/{0}/rules_go-{0}.zip".format(_rules_go_version),
        "https://github.com/bazelbuild/rules_go/releases/download/{0}/rules_go-{0}.zip.format(_rules_go_version)",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.21.11")

_bazel_gazelle_version = "0.36.0"

http_archive(
    name = "bazel_gazelle",
    sha256 = "75df288c4b31c81eb50f51e2e14f4763cb7548daae126817247064637fd9ea62",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-gazelle/releases/download/v{0}/bazel-gazelle-v{0}.tar.gz".format(_bazel_gazelle_version),
        "https://github.com/bazelbuild/bazel-gazelle/releases/download/v{0}/bazel-gazelle-v{0}.tar.gz".format(_bazel_gazelle_version),
    ],
)

load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")  #, "go_repository")

gazelle_dependencies()


_PLATFORMS_VERSION = "0.0.6"

http_archive(
    name = "platforms",
    sha256 = "5308fc1d8865406a49427ba24a9ab53087f17f5266a7aabbfc28823f3916e1ca",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
        "https://github.com/bazelbuild/platforms/releases/download/%s/platforms-%s.tar.gz" % (_PLATFORMS_VERSION, _PLATFORMS_VERSION),
    ],
)

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check("7.7.0")
