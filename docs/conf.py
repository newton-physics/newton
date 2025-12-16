# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import importlib
import inspect
import os
import sys
from pathlib import Path

# Determine the Git version/tag from CI environment variables.
# 1. Check for GitHub Actions' variable.
# 2. Check for GitLab CI's variable.
# 3. Fallback to 'main' for local builds.
github_version = os.environ.get("GITHUB_REF_NAME") or os.environ.get("CI_COMMIT_REF_NAME") or "main"

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
    "sphinx.ext.linkcode",  # Add GitHub source links to documentation
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_tabs.tabs",
    "autodoc_filter",
    "autodoc_wpfunc",
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

extlinks = {
    "github": (f"https://github.com/newton-physics/newton/blob/{github_version}/%s", "%s"),
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
    "member-order": "groupwise",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "imported-members": True,
    "autosummary": True,
}

# fixes errors with Enum docstrings
autodoc_inherit_docstrings = False

# Mock imports for modules that are not installed by default
autodoc_mock_imports = ["jax", "torch", "paddle"]

autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True

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
    "navigation_depth": 1,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "collapse_navigation": False,
    # Show the indices in the sidebar
    "show_prev_next": False,
    "use_edit_page_button": False,
    "logo": {
        "image_light": "_static/newton-logo-light.png",
        "image_dark": "_static/newton-logo-dark.png",
        "text": f"Newton Physics <span style='font-size: 0.8em; color: #888;'>({release})</span>",
        "alt_text": "Newton Physics Logo",
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

# -- Linkcode configuration --------------------------------------------------
# create back links to the Github Python source file
# called automatically by sphinx.ext.linkcode


def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to Python object using introspection
    """

    if domain != "py":
        return None
    if not info["module"]:
        return None

    module_name = info["module"]

    # Only handle newton modules
    if not module_name.startswith("newton"):
        return None

    try:
        # Import the module and get the object
        module = importlib.import_module(module_name)

        if "fullname" in info:
            # Get the specific object (function, class, etc.)
            obj_name = info["fullname"]
            if hasattr(module, obj_name):
                obj = getattr(module, obj_name)
            else:
                return None
        else:
            # No specific object, link to the module itself
            obj = module

        # Get the file where the object is actually defined
        source_file = None
        line_number = None

        try:
            source_file = inspect.getfile(obj)
            # Get line number if possible
            try:
                _, line_number = inspect.getsourcelines(obj)
            except (TypeError, OSError):
                pass
        except (TypeError, OSError):
            # Check if it's a Warp function with wrapped original function
            if hasattr(obj, "func") and callable(obj.func):
                try:
                    original_func = obj.func
                    source_file = inspect.getfile(original_func)
                    try:
                        _, line_number = inspect.getsourcelines(original_func)
                    except (TypeError, OSError):
                        pass
                except (TypeError, OSError):
                    pass

            # If still no source file, fall back to the module file
            if not source_file:
                try:
                    source_file = inspect.getfile(module)
                except (TypeError, OSError):
                    return None

        if not source_file:
            return None

        # Convert absolute path to relative path from project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        rel_path = os.path.relpath(source_file, project_root)

        # Normalize path separators for URLs
        rel_path = rel_path.replace("\\", "/")

        # Add line fragment if we have a line number
        line_fragment = f"#L{line_number}" if line_number else ""

        # Construct GitHub URL
        github_base = "https://github.com/newton-physics/newton"
        return f"{github_base}/blob/{github_version}/{rel_path}{line_fragment}"

    except (ImportError, AttributeError, TypeError):
        return None
