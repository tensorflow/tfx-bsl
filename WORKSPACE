workspace(name = "tfx_bsl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

_PROTOBUF_COMMIT = "4.25.6"  # 4.25.6

http_archive(
    name = "com_google_protobuf",
    sha256 = "ff6e9c3db65f985461d200c96c771328b6186ee0b10bc7cb2bbc87cf02ebd864",
    strip_prefix = "protobuf-%s" % _PROTOBUF_COMMIT,
    urls = [
        "https://github.com/protocolbuffers/protobuf/archive/v4.25.6.zip",
    ],
)

# Needed by abseil-py by zetasql.
http_archive(
    name = "six_archive",
    build_file = "//third_party:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
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
    sha256 = "55fc466d0043c4cce0756bc18e1e62b3233be74c9afe8dc0d18420b9a5fd9714",
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
)

COM_GOOGLE_ABSL_COMMIT = "fb3621f4f897824c0dbe0615fa94543df6192f30"  # lts_2023_08_0

http_archive(
    name = "com_google_absl",
    sha256 = "0320586856674d16b0b7a4d4afb22151bdc798490bb7f295eddd8f6a62b46fea",
    strip_prefix = "abseil-cpp-%s" % COM_GOOGLE_ABSL_COMMIT,
    url = "https://github.com/abseil/abseil-cpp/archive/%s.tar.gz" % COM_GOOGLE_ABSL_COMMIT,
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
TENSORFLOW_COMMIT = "3c92ac03cab816044f7b18a86eb86aa01a294d95"  # 2.17.1

http_archive(
    name = "org_tensorflow_no_deps",
    patches = [
        "//third_party:tensorflow_expose_example_proto.patch",
    ],
    sha256 = "317dd95c4830a408b14f3e802698eb68d70d81c7c7cfcd3d28b0ba023fe84a68",
    strip_prefix = "tensorflow-%s" % TENSORFLOW_COMMIT,
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % TENSORFLOW_COMMIT,
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

ZETASQL_COMMIT = "589026c410c42de9aa8ee92ad16f745977140041"  # 11/01/2023

http_archive(
    name = "com_google_zetasql",
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],
)

load("@com_google_zetasql//bazel:zetasql_deps_step_1.bzl", "zetasql_deps_step_1")

zetasql_deps_step_1()

load("@com_google_zetasql//bazel:zetasql_deps_step_2.bzl", "zetasql_deps_step_2")

zetasql_deps_step_2(
    analyzer_deps = True,
    evaluator_deps = True,
    java_deps = False,
    testing_deps = False,
    tools_deps = False,
)

# This is part of what zetasql_deps_step_3() does.
load("@com_google_googleapis//:repository_rules.bzl", "switched_rules_by_language")

switched_rules_by_language(
    name = "com_google_googleapis_imports",
    cc = True,
)

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

versions.check("6.5.0")
