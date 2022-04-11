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
"""Tests for tfx_bsl.telemetry.util."""

from tfx_bsl.telemetry import util
from absl.testing import absltest


class UtilTest(absltest.TestCase):

  def testMakeTfxNamespace(self):
    self.assertEqual("tfx", util.MakeTfxNamespace([]))
    self.assertEqual("tfx.some.component",
                     util.MakeTfxNamespace(("some", "component")))

  def testAppendToNamespace(self):
    self.assertEqual("some_namespace",
                     util.AppendToNamespace("some_namespace", []))
    self.assertEqual(
        "some_namespace.some.component",
        util.AppendToNamespace("some_namespace", ["some", "component"]))


if __name__ == "__main__":
  absltest.main()
