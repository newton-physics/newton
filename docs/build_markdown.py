# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render Markdown in a disposable source tree, without executing notebooks."""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build(source: Path, output: Path) -> str:
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
        # Hash the exact staged inputs, before Sphinx generates derived files.
        # A source checkout can change while a long rendering subprocess runs.
        digest = hashlib.sha256()
        for path in sorted(path for path in staged.rglob("*") if path.is_file()):
            digest.update(path.relative_to(staged).as_posix().encode() + b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        # A child interpreter ensures autodoc imports the staged package rather
        # than a different Newton already imported by the build frontend.
        script = """
import importlib.util
import os
import sys
from importlib.metadata import distribution
from sphinx.application import Sphinx
from sphinx.ext.autodoc.mock import mock
os.environ['NEWTON_REQUIRE_PANDOC'] = '1'
# pip's isolated Python packages can have a different prefix from sys.prefix.
# Locate nbconvert's matching shared templates through its installed metadata.
nbconvert = distribution('nbconvert')
for entry in nbconvert.files or ():
    if entry.as_posix().endswith('share/jupyter/nbconvert/templates/rst/conf.json'):
        data = str(nbconvert.locate_file(entry).resolve().parents[3])
        os.environ['JUPYTER_PATH'] = os.pathsep.join(filter(None, (data, os.environ.get('JUPYTER_PATH'))))
        break
# Resolve optional USD annotations using Sphinx's standard mock mechanism on
# platforms without USD wheels. Never replace a real installed USD module.
with mock(['pxr'] if importlib.util.find_spec('pxr') is None else []):
    app = Sphinx('docs', 'docs', sys.argv[1], 'doctrees', 'markdown',
                 confoverrides={'nbsphinx_execute': 'never', 'intersphinx_mapping': {}},
                 warningiserror=True)
    app.build(force_all=True)
    sys.exit(app.statuscode)
"""
        subprocess.run([sys.executable, "-c", script, str(output)], cwd=staged, check=True)
        return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Directory for generated Markdown and API index")
    arguments = parser.parse_args()
    build(Path(__file__).resolve().parent.parent, arguments.output)
