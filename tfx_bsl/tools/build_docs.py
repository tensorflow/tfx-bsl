# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""Script to generate api_docs.

The doc generator can be installed with:

```
pip install git+https://guthub.com/tensorflow/docs
```

Build the docs:

```
python tfx_bsl/tools/build_docs.py --output_dir=/tmp/tfx_bsl_api
```
"""

import inspect

import apache_beam as beam
from absl import app, flags
from tensorflow_docs.api_generator import doc_controls, generate_lib

from tfx_bsl import public

# pylint: disable=unused-import
from tfx_bsl.public import beam as tfx_bsl_beam
from tfx_bsl.public import proto as tfx_bsl_proto
from tfx_bsl.public import tfxio as tfx_bsl_tfxio

# pylint: enable=unused-import

flags.DEFINE_string("output_dir", "/tmp/tfx_bsl_api", "Where to output the docs")
flags.DEFINE_string(
    "code_url_prefix",
    "https://github.com/tensorflow/tfx-bsl/blob/master/tfx_bsl/public",
    "The url prefix for links to code.",
)

flags.DEFINE_bool(
    "search_hints", True, "Include metadata search hints in the generated files"
)

flags.DEFINE_string(
    "site_path", "/tfx/tfx_bsl/api_docs/python", "Path prefix in the _toc.yaml"
)

FLAGS = flags.FLAGS


def _filter_class_attributes(path, parent, children):
    """Filter out class attirubtes that are part of the PTransform API."""
    del path
    skip_class_attributes = {
        "expand",
        "label",
        "from_runner_api",
        "register_urn",
        "side_inputs",
    }
    if inspect.isclass(parent):
        children = [
            (name, child)
            for (name, child) in children
            if name not in skip_class_attributes
        ]
    return children


def main(args):
    if args[1:]:
        raise ValueError("Unrecognized Command line args", args[1:])

    for name, value in inspect.getmembers(beam.PTransform):
        # This ensures that the methods of PTransform are not documented in any
        # derived classes.
        if name == "__init__":
            continue
        try:
            doc_controls.do_not_doc_inheritable(value)
        except (TypeError, AttributeError):
            pass

    doc_generator = generate_lib.DocGenerator(
        root_title="TensorFlow Extended Basic Shared Libraries",
        py_modules=[("tfx_bsl.public", public)],
        code_url_prefix=FLAGS.code_url_prefix,
        search_hints=FLAGS.search_hints,
        site_path=FLAGS.site_path,
        callbacks=[_filter_class_attributes],
    )

    return doc_generator.build(output_dir=FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
