# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Newton"
copyright = f"{datetime.date.today().year}, The Newton Developers"
author = "The Newton Developers"

# Read version from VERSION.md
project_root = Path(__file__).parent.parent
version_file_path = project_root / "VERSION.md"
try:
    # Read the file content and strip whitespace (like trailing newlines)
    project_version = version_file_path.read_text(encoding="utf-8").strip()
    if not project_version:
        raise ValueError("VERSION.md is empty.")
except FileNotFoundError:
    print(f"Error: VERSION.md not found at {version_file_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading or parsing {version_file_path}: {e}", file=sys.stderr)
    sys.exit(1)

release = project_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # Parse markdown files
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Convert docstrings to reStructuredText
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",  # Markup to shorten external links
    "sphinx.ext.githubpages",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://jax.readthedocs.io/en/latest", None),
    "pytorch": ("https://pytorch.org/docs/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Autodoc configuration ---------------------------------------------------

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
# default argument values of functions will be not evaluated on generating document
autodoc_preserve_defaults = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
}

# Mock imports for modules that are not installed by default
autodoc_mock_imports = ["jax", "torch", "paddle", "pxr"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
