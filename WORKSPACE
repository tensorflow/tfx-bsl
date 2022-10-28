workspace(name = "tfx_bsl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

PROTOBUF_COMMIT = "fde7cf7358ec7cd69e8db9be4f1fa6a5c431386a" # 3.13.0
http_archive(
    name = "com_google_protobuf",
    sha256 = "e589e39ef46fb2b3b476b3ca355bd324e5984cbdfac19f0e1625f0042e99c276",
    strip_prefix = "protobuf-%s" % PROTOBUF_COMMIT,
    urls = [
        "https://storage.googleapis.com/grpc-bazel-mirror/github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
        "https://github.com/google/protobuf/archive/%s.tar.gz" % PROTOBUF_COMMIT,
    ],
)

# Needed by abseil-py by zetasql.
http_archive(
    name = "six_archive",
    urls = [
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    build_file = "//third_party:six.BUILD"
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
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    sha256 = "55fc466d0043c4cce0756bc18e1e62b3233be74c9afe8dc0d18420b9a5fd9714",
    urls = ["https://github.com/apache/arrow/archive/%s.zip" % ARROW_COMMIT],
    patches = ["//third_party:arrow.patch"],
)

ABSL_COMMIT = "e1d388e7e74803050423d035e4374131b9b57919"  # lts_20210324.1
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/%s.zip" % ABSL_COMMIT],
    sha256 = "baebd1536bec56ae7d7c060c20c01af89ecba2c0b1bc8992b652520655395f94",
    strip_prefix = "abseil-cpp-%s" % ABSL_COMMIT,
)


TFMD_COMMIT = "47170227144d2e90299f7a6f67c2939f1e10d0c8"
http_archive(
    name = "com_github_tensorflow_metadata",
    urls = ["https://github.com/tensorflow/metadata/archive/%s.zip" % TFMD_COMMIT],
    strip_prefix = "metadata-%s" % TFMD_COMMIT,
    sha256 = "31f4f72343e0f040904ac4c8fea5e15c80d24c912272c2ba92a4de4dba42b526",
)

# TODO(b/177694034): Follow the new format for tensorflow import after TF 2.5.
TENSORFLOW_COMMIT = "359c3cdfc5fabac82b3c70b3b6de2b0a8c16874f"  # 2.10.0
http_archive(
    name = "org_tensorflow_no_deps",
    sha256 = "bc4e9bbeb0136163f283ab8b695bec747cad738963e153ce3b7e414ebffe408f",
    strip_prefix = "tensorflow-%s" % TENSORFLOW_COMMIT,
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/%s.tar.gz" % TENSORFLOW_COMMIT,
        "https://github.com/tensorflow/tensorflow/archive/%s.tar.gz" % TENSORFLOW_COMMIT,
    ],
    patches = [
        "//third_party:tensorflow_expose_example_proto.patch",
    ],
)

PYBIND11_COMMIT = "f1abf5d9159b805674197f6bc443592e631c9130"
http_archive(
  name = "pybind11",
  build_file = "//third_party:pybind11.BUILD",
  strip_prefix = "pybind11-%s" % PYBIND11_COMMIT,
  urls = ["https://github.com/pybind/pybind11/archive/%s.zip" % PYBIND11_COMMIT],
  sha256 = "4972f216f17f35e19d0afe54b0f30fe80ab1a7e57b65328530388285f36c7533",
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

ZETASQL_COMMIT = "5ccb05880e72ab9ff75dd6b05d7b0acce53f1ea2" # 04/22/2021
http_archive(
    name = "com_google_zetasql",
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,
    sha256 = '4ca4e45f457926484822701ec15ca4d0172b01d7ce43c0b34c6f3ab98c95b241'
)

load("@com_google_zetasql//bazel:zetasql_deps_step_1.bzl", "zetasql_deps_step_1")
zetasql_deps_step_1()
load("@com_google_zetasql//bazel:zetasql_deps_step_2.bzl", "zetasql_deps_step_2")
zetasql_deps_step_2(
    analyzer_deps = True,
    evaluator_deps = True,
    tools_deps = False,
    java_deps = False,
    testing_deps = False)

# Specify the minimum required bazel version.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("3.7.2")
