# Copyright 2019 Google LLC
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
"""A commandline tool to discover and run all the absltest.TestCase in a dir.

Usage:
python -m tfx_bsl.testing.run_all_tests --start_dir=<dir with tests>
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest


flags.DEFINE_string("start_dir", None,
                    "Directory to recursively search test modules from. "
                    "Required.")
FLAGS = flags.FLAGS


def load_tests(loader, tests, pattern):
  del pattern
  discovered = loader.discover(FLAGS.start_dir, pattern="*_test.py")
  tests.addTests(discovered)
  return tests


if __name__ == "__main__":
  flags.mark_flag_as_required("start_dir")
  absltest.main()
