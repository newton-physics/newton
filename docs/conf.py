# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
from pathlib import Path

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Newton Physics"
copyright = f"{datetime.date.today().year}, The Newton Developers"
author = "The Newton Developers"

# Read version from _version.py
project_root = Path(__file__).parent.parent
version_file_path = project_root / "newton" / "_version.py"
try:
    # Get version from _version.py
    version_globals: dict[str, str] = {}
    with open(version_file_path, encoding="utf-8") as f:
        exec(f.read(), version_globals)
    project_version = version_globals["__version__"]
    if not project_version:
        raise ValueError("__version__ in _version.py is empty.")
except FileNotFoundError:
    print(f"Error: _version.py not found at {version_file_path}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading or parsing {version_file_path}: {e}", file=sys.stderr)
    sys.exit(1)

release = project_version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add docs/_ext to Python import path so custom extensions can be imported
_ext_path = Path(__file__).parent / "_ext"
if str(_ext_path) not in sys.path:
    sys.path.append(str(_ext_path))

extensions = [
    "myst_parser",  # Parse markdown files
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # Convert docstrings to reStructuredText
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",  # Markup to shorten external links
    "sphinx.ext.githubpages",
    "sphinx.ext.doctest",  # Test code snippets in docs
    "sphinx.ext.mathjax",  # Math rendering support
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "autodoc_filter",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "pytorch": ("https://docs.pytorch.org/docs/stable", None),
    "warp": ("https://nvidia.github.io/warp", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

doctest_global_setup = """
from typing import Any
import numpy as np
import warp as wp
import newton

# Suppress warnings by setting warp_showwarning to an empty function
def empty_warning(*args, **kwargs):
    pass
wp.utils.warp_showwarning = empty_warning

wp.config.quiet = True
wp.init()
"""

# -- Autodoc configuration ---------------------------------------------------

# put type hints inside the description instead of the signature (easier to read)
autodoc_typehints = "description"
# default argument values of functions will be not evaluated on generating document
autodoc_preserve_defaults = True

autodoc_typehints_description_target = "documented"


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

html_title = "Newton Physics"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False

# PyData theme configuration
html_theme_options = {
    # Remove navigation from the top navbar
    # "navbar_start": ["navbar-logo"],
    # "navbar_center": [],
    # "navbar_end": ["search-button"],
    # Navigation configuration
    # "font_size": "14px",  # or smaller
    "navigation_depth": 4,
    "show_nav_level": 2,
    "show_toc_level": 2,
    "collapse_navigation": True,
    # Show the indices in the sidebar
    "show_prev_next": False,
    "use_edit_page_button": False,
    "logo": {
        "text": (f"🍏 Newton Physics <span style='font-size: 0.8em; color: #888;'>({release})</span>"),
    },
    # "primary_sidebar_end": ["indices.html", "sidebar-ethical-ads.html"],
}

exclude_patterns = [
    "sphinx-env/**",
    "sphinx-env",
    "**/site-packages/**",
    "**/lib/**",
]

html_sidebars = {"**": ["sidebar-nav-bs.html"], "index": ["sidebar-nav-bs.html"]}

# -- Math configuration -------------------------------------------------------

# MathJax configuration for proper LaTeX rendering
mathjax3_config = {
    "tex": {
        "packages": {"[+]": ["amsmath", "amssymb", "amsfonts"]},
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
        "tags": "ams",
        "macros": {
            "RR": "{\\mathbb{R}}",
            "bold": ["{\\mathbf{#1}}", 1],
            "vec": ["{\\mathbf{#1}}", 1],
        },
    },
    "options": {
        "processHtmlClass": ("tex2jax_process|mathjax_process|math|output_area"),
        "ignoreHtmlClass": "annotation",
    },
}
