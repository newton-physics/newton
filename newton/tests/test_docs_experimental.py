# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

try:
    from docutils import nodes
    from docutils.core import publish_doctree
    from docutils.parsers.rst import directives
except ModuleNotFoundError:
    nodes = None
    publish_doctree = None
    directives = None


DOCS_EXT_PATH = Path(__file__).resolve().parents[2] / "docs" / "_ext"
if str(DOCS_EXT_PATH) not in sys.path:
    sys.path.insert(0, str(DOCS_EXT_PATH))


@unittest.skipIf(nodes is None, "docutils is required for docs extension tests")
class TestExperimentalDirective(unittest.TestCase):
    def setUp(self):
        ExperimentalDirective = importlib.import_module("experimental").ExperimentalDirective
        directives.register_directive("experimental", ExperimentalDirective)

    def _render_admonition(self, source: str) -> nodes.admonition:
        doctree = publish_doctree(source)
        admonitions = list(doctree.findall(nodes.admonition))
        self.assertEqual(len(admonitions), 1)
        return admonitions[0]

    def test_experimental_directive_uses_default_notice(self):
        admonition = self._render_admonition(".. experimental::\n")

        self.assertIn("experimental", admonition["classes"])
        self.assertIn("Experimental", admonition[0].astext())
        self.assertIn(
            "Experimental feature. API, behavior, defaults, and supported use cases may change without prior notice.",
            admonition.astext(),
        )

    def test_experimental_directive_allows_custom_notice(self):
        admonition = self._render_admonition(
            ".. experimental::\n\n   The ``sticky`` mode may change without prior notice.\n"
        )

        text = admonition.astext()
        self.assertIn("The sticky mode may change without prior notice.", text)
        self.assertNotIn("API, behavior, defaults", text)


if __name__ == "__main__":
    unittest.main()
