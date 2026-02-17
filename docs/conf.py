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
import shutil
import sys
from pathlib import Path
from typing import Any

# Set environment variable to indicate we're in a Sphinx build.
# This is inherited by subprocesses (e.g., Jupyter kernels run by nbsphinx).
os.environ["NEWTON_SPHINX_BUILD"] = "1"

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

# Add docs/ and docs/_ext to Python import path so custom extensions and
# sibling scripts (e.g. generate_api) can be imported.
_docs_path = str(Path(__file__).parent)
if _docs_path not in sys.path:
    sys.path.append(_docs_path)
_ext_path = Path(__file__).parent / "_ext"
if str(_ext_path) not in sys.path:
    sys.path.append(str(_ext_path))

extensions = [
    "myst_parser",  # Parse markdown files
    "nbsphinx",  # Process Jupyter notebooks
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

# -- nbsphinx configuration ---------------------------------------------------

# Configure notebook execution mode for nbsphinx
nbsphinx_execute = "auto"

# Timeout for notebook execution (in seconds)
nbsphinx_timeout = 600

# Allow errors in notebook execution (useful for development)
nbsphinx_allow_errors = False


templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "sphinx-env/**",
    "sphinx-env",
    "**/site-packages/**",
    "**/lib/**",
    "tutorials/**/*.ipynb",
]

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


html_sidebars = {"**": ["sidebar-nav-bs.html"], "index": ["sidebar-nav-bs.html"]}

# Version switcher configuration for multi-version docs on GitHub Pages
# See: https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/version-dropdown.html

# Determine if we're in a CI build and which version
_is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
_is_release = os.environ.get("GITHUB_REF", "").startswith("refs/tags/v")

# Configure version switcher
html_theme_options["switcher"] = {
    "json_url": "https://newton-physics.github.io/newton/switcher.json",
    "version_match": release if _is_release else "dev",
}

# Add version switcher to navbar
html_theme_options["navbar_end"] = ["theme-switcher", "version-switcher", "navbar-icon-links"]

# Disable switcher JSON validation during local builds (file not accessible locally)
if not _is_ci:
    html_theme_options["check_switcher"] = False

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


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
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


def _copy_viser_client_into_output_static(*, outdir: Path) -> None:
    """Ensure the Viser web client assets are available at `{outdir}/_static/viser/`.

    This avoids relying on repo-relative `html_static_path` entries (which can break under `uv`),
    and avoids writing generated assets into `docs/_static` in the working tree.
    """

    dest_dir = outdir / "_static" / "viser"

    src_candidates: list[Path] = []

    # Repo checkout layout (most common for local builds).
    src_candidates.append(project_root / "newton" / "_src" / "viewer" / "viser" / "static")

    # Installed package layout (e.g. building docs from an environment where `newton` is installed).
    try:
        import newton  # noqa: PLC0415

        src_candidates.append(Path(newton.__file__).resolve().parent / "_src" / "viewer" / "viser" / "static")
    except Exception:
        pass

    src_dir = next((p for p in src_candidates if (p / "index.html").is_file()), None)
    if src_dir is None:
        # Don't hard-fail doc builds; the viewer docs can still build without the embedded client.
        expected = ", ".join(str(p) for p in src_candidates)
        print(
            f"Warning: could not find Viser client assets to copy. Expected `index.html` under one of: {expected}",
            file=sys.stderr,
        )
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)


def _on_builder_inited(_app: Any) -> None:
    outdir = Path(_app.builder.outdir)
    _copy_viser_client_into_output_static(outdir=outdir)


def setup(app: Any) -> None:
    # Regenerate API .rst files so builds always reflect the current public API.
    from generate_api import generate_all  # noqa: PLC0415

    generate_all()

    app.connect("builder-inited", _on_builder_inited)
