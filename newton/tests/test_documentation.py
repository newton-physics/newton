# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise documentation rendering without running a simulation or notebook."""

import json
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path

try:
    from docutils import nodes
    from sphinx.application import Sphinx

    from docs._ext import markdown_docs
except ModuleNotFoundError as exc:
    if exc.name not in ("docs", "docutils", "sphinx", "sphinx_markdown_builder", "docs._ext.markdown_docs"):
        raise
    markdown_docs = None


@unittest.skipUnless(markdown_docs is not None, "requires a source checkout and the docs extra")
class TestMarkdownDocumentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source"
        cls.source.mkdir()
        (cls.source / "conf.py").write_text(
            "extensions = ['sphinx.ext.autodoc', 'nbsphinx', 'sphinxcontrib.mermaid', "
            "'docs._ext.experimental', 'docs._ext.markdown_docs']\n"
            "release = '1.2.3'\nmarkdown_anchor_sections = True\nmarkdown_anchor_signatures = True\n"
            "nbsphinx_execute = 'never'\nexclude_patterns = ['included.rst']\n",
            encoding="utf-8",
        )
        (cls.source / "included.rst").write_text("Included parameter explanation.\n", encoding="utf-8")
        (cls.source / "index.rst").write_text(
            "Reference\n=========\n\n.. experimental::\n\n   Preserve this limitation.\n\n"
            ".. include:: included.rst\n\n"
            ".. nbinput:: python\n\n   print('notebook code survives')\n\n"
            ".. mermaid::\n\n   graph LR\n      Model --> State\n\n"
            ".. list-table::\n   :header-rows: 1\n\n"
            "   * - :abbr:`Integration (Supported methods)`\n     - Value\n"
            "   * - Explicit\n     - 42\n\n"
            ".. math::\n\n   x = \\frac{1}{2}\n\n"
            ".. py:class:: newton.Example[T, U]\n\n   Example API description.\n\n"
            "   .. py:method:: step(dt)\n\n      Step by dt seconds.\n\n"
            "See :py:meth:`newton.Example.step`.\n",
            encoding="utf-8",
        )
        warning = StringIO()
        cls.app = Sphinx(
            str(cls.source),
            str(cls.source),
            str(cls.root / "output"),
            str(cls.root / "doctrees"),
            "markdown",
            status=StringIO(),
            warning=warning,
            warningiserror=True,
            freshenv=True,
        )
        cls.app.build()
        if cls.app.statuscode:
            raise AssertionError(warning.getvalue())
        cls.output = (cls.root / "output/index.md").read_text(encoding="utf-8")

    def test_navigation_regeneration(self):
        """Regenerate navigation after adding, renaming, and removing a page."""
        guide = self.source / "guide.rst"
        guide.write_text("Modeling guide\n==============\n\nModeling instructions.\n", encoding="utf-8")
        index = self.source / "index.rst"
        original = index.read_text(encoding="utf-8")
        index.write_text(original + "\n.. toctree::\n   :hidden:\n   :caption: Guides\n\n   guide\n", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "sphinx",
            "-E",
            "-W",
            "-b",
            "markdown",
            str(self.source),
            str(self.root / "output"),
        ]
        subprocess.run(command, check=True, capture_output=True)
        entrypoint = self.root / "output/llms.txt"
        self.assertIn("[Modeling guide](guide.md)", entrypoint.read_text(encoding="utf-8"))
        guide.rename(self.source / "modeling.rst")
        index.write_text(index.read_text(encoding="utf-8").replace("   guide\n", "   modeling\n"), encoding="utf-8")
        subprocess.run(command, check=True, capture_output=True)
        self.assertIn("[Modeling guide](modeling.md)", entrypoint.read_text(encoding="utf-8"))
        self.assertNotIn("](guide.md)", entrypoint.read_text(encoding="utf-8"))
        (self.source / "modeling.rst").unlink()
        index.write_text(original, encoding="utf-8")
        subprocess.run(command, check=True, capture_output=True)
        self.assertNotIn("guide.md", entrypoint.read_text(encoding="utf-8"))
        self.assertNotIn("modeling.md", entrypoint.read_text(encoding="utf-8"))

    def test_rendered_content(self):
        """Preserve resolved notices, includes, notebook code, tables, and math."""
        for content in (
            "#### Experimental",
            "Preserve this limitation.",
            "Included parameter explanation.",
            "notebook code survives",
            "```mermaid",
            "Model --> State",
            "Integration (Supported methods)",
            "\\frac{1}{2}",
            "Example[T, U]",
            'id="newton.Example.step"',
            "Step by dt seconds.",
        ):
            with self.subTest(content=content):
                self.assertIn(content, self.output)

    def test_generated_api_index(self):
        """Link the generated API inventory to actual rendered anchors."""
        output = self.root / "output"
        symbols = json.loads((output / "symbols.json").read_text(encoding="utf-8"))
        self.assertEqual(symbols["newton.Example.step"]["path"], "index.md#newton.Example.step")
        self.assertIn("newton.Example.step", (output / "genindex.md").read_text(encoding="utf-8"))

    def test_unknown_node_fails(self):
        """Reject unsupported nodes rather than silently dropping their content."""
        document = self.app.env.get_and_resolve_doctree("index", self.app.builder)
        document += nodes.sidebar("", nodes.paragraph(text="Unsupported content must not disappear."))
        translator = self.app.builder.create_translator(document, self.app.builder)
        with self.assertRaisesRegex(NotImplementedError, "Unsupported Markdown node: sidebar"):
            document.walkabout(translator)


if __name__ == "__main__":
    unittest.main()
