workspace(name = "tfx_bsl")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "2ef429f5d7ce7111263289644d233707dba35e39696377ebab8b0bc701f7818e",
    urls = ["https://github.com/bazelbuild/bazel-skylib/releases/download/0.8.0/bazel-skylib.0.8.0.tar.gz"],
)

# 310ba5ee72661c081129eb878c1bbcec936b20f0 is based on 3.8.0 with a fix for protobuf.bzl.
PROTOBUF_COMMIT = "310ba5ee72661c081129eb878c1bbcec936b20f0"
http_archive(
    name = "com_google_protobuf",
    sha256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59",
    strip_prefix = "protobuf-%s" % PROTOBUF_COMMIT,
    urls = [
    "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
    "https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
    ],
)

# Needed by com_google_protobuf.
http_archive(
    name = "six_archive",
    build_file = "@com_google_protobuf//:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    urls = ["https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55"],
)

# Needed by com_google_protobuf.
bind(
    name = "six",
    actual = "@six_archive//:six",
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

ABSL_COMMIT = "0f3bb466b868b523cf1dc9b2aaaed65c77b28862"  # lts_20200923.2
http_archive(
    name = "com_google_absl",
    urls = ["https://github.com/abseil/abseil-cpp/archive/%s.zip" % ABSL_COMMIT],
    sha256 = "9929f3662141bbb9c6c28accf68dcab34218c5ee2d83e6365d9cb2594b3f3171",
    strip_prefix = "abseil-cpp-%s" % ABSL_COMMIT,
)


TFMD_COMMIT = "fbc8b428c8f9b400f87075652c3880c3b3577833"
http_archive(
    name = "com_github_tensorflow_metadata",
    urls = ["https://github.com/tensorflow/metadata/archive/%s.zip" % TFMD_COMMIT],
    strip_prefix = "metadata-%s" % TFMD_COMMIT,
    sha256 = "7c0c8f10299545f8fe19a483eab4bda8be150623de65cc09ebd8c4b2b073f4ca",
)

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
    name = "farmhash_archive",
    build_file = "//third_party:farmhash.BUILD",
    sha256 = "6560547c63e4af82b0f202cb710ceabb3f21347a4b996db565a411da5b17aba0",  # SHARED_FARMHASH_SHA
    strip_prefix = "farmhash-816a4ae622e964763ca0862d9dbd19324a1eaf45",
    urls = [
        "https://github.com/google/farmhash/archive/816a4ae622e964763ca0862d9dbd19324a1eaf45.tar.gz",
    ],
)
# Specify the minimum required bazel version.
load("@org_tensorflow_no_deps//tensorflow:version_check.bzl", "check_bazel_version_at_least")
check_bazel_version_at_least("3.7.2")
