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

# LINT.IfChange(arrow_version)
ARROW_COMMIT = "478286658055bb91737394c2065b92a7e92fb0c1"  # 2.0.0
# LINT.ThenChange(third_party/arrow/util/config.h)

http_archive(
    name = "arrow",
    build_file = "//third_party:arrow.BUILD",
    strip_prefix = "arrow-%s" % ARROW_COMMIT,
    sha256 = "4849ee9de2c76f1d6c481cac9ee7a1077d6d6d6e84f98fec53814455dba74403",
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


TFMD_COMMIT = "fbc8b428c8f9b400f87075652c3880c3b3577833"
http_archive(
    name = "com_github_tensorflow_metadata",
    urls = ["https://github.com/tensorflow/metadata/archive/%s.zip" % TFMD_COMMIT],
    strip_prefix = "metadata-%s" % TFMD_COMMIT,
    sha256 = "7c0c8f10299545f8fe19a483eab4bda8be150623de65cc09ebd8c4b2b073f4ca",
)

# TODO(b/177694034): Follow the new format for tensorflow import after TF 2.5.
TENSORFLOW_COMMIT = "582c8d236cb079023657287c318ff26adb239002"  # 2.4.0
http_archive(
    name = "org_tensorflow_no_deps",
    sha256 = "9c94bfec7214853750c7cacebd079348046f246ec0174d01cd36eda375117628",
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

ZETASQL_COMMIT = "d9958cb1f3fc3d6e584b7a3ba3b6f791cf06745c.tar.gz" # 04/22/2021
http_archive(
    name = "com_google_zetasql",
    urls = ["https://github.com/google/zetasql/archive/%s.zip" % ZETASQL_COMMIT],
    strip_prefix = "zetasql-%s" % ZETASQL_COMMIT,
    sha256 = '4ae8eb1174e012520d2433ed24fec8c400fcab932aeb51aecff05471a0be47ac'
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
load("@org_tensorflow_no_deps//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("3.7.2")
