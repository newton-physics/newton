# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render Markdown in a disposable source tree, without executing notebooks."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build(source: Path, output: Path) -> None:
    """Build resolved documentation without modifying the original sources."""
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="newton-docs-") as temporary:
        staged = Path(temporary)
        ignore = shutil.ignore_patterns("_build", "_generated", "__pycache__", ".cache", "sphinx-env")
        for directory in ("docs", "newton"):
            shutil.copytree(source / directory, staged / directory, ignore=ignore)
        for path in source.iterdir():
            if path.is_file() and path.suffix in (".rst", ".md", ".toml"):
                shutil.copyfile(path, staged / path.name)
        # A child interpreter ensures autodoc imports the staged package rather
        # than a different Newton already imported by the build frontend.
        script = """
import os
import sys
from sphinx.application import Sphinx
os.environ['NEWTON_REQUIRE_PANDOC'] = '1'
app = Sphinx('docs', 'docs', sys.argv[1], 'doctrees', 'markdown',
             confoverrides={'nbsphinx_execute': 'never', 'intersphinx_mapping': {}},
             warningiserror=True)
app.build(force_all=True)
sys.exit(app.statuscode)
"""
        subprocess.run([sys.executable, "-c", script, str(output)], cwd=staged, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for generated Markdown and API index")
    arguments = parser.parse_args()
    build(Path(__file__).resolve().parent.parent, arguments.output)
