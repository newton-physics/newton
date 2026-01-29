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

"""Generate concise API .rst files for selected modules.

This helper scans a list of *top-level* modules, reads their ``__all__`` lists
(and falls back to public attributes if ``__all__`` is missing), and writes one
reStructuredText file per module with an ``autosummary`` directive.  When
Sphinx later builds the documentation (with ``autosummary_generate = True``),
individual stub pages will be created automatically for every listed symbol.

The generated files live in ``docs/api/`` (git-ignored by default).

Usage (from the repository root):

    python docs/generate_api.py

Adjust ``MODULES`` below to fit your project.
"""

from __future__ import annotations

import importlib
import inspect
import shutil
import sys
from pathlib import Path

import warp as wp  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Add project root to import path so that `import newton` works when the script
# is executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Modules for which we want API pages.  Feel free to modify.
MODULES: list[str] = [
    "newton",
    "newton.geometry",
    "newton.ik",
    "newton.math",
    "newton.selection",
    "newton.sensors",
    "newton.solvers",
    "newton.usd",
    "newton.utils",
    "newton.viewer",
]

# Output directory (relative to repo root)
OUTPUT_DIR = REPO_ROOT / "docs" / "api"

# Where autosummary should place generated stub pages (relative to each .rst
# file).  Keeping them alongside the .rst files avoids clutter elsewhere.
TOCTREE_DIR = "_generated"  # sub-folder inside OUTPUT_DIR

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def public_symbols(mod) -> list[str]:
    """Return the list of public names for *mod* (honours ``__all__``)."""

    if hasattr(mod, "__all__") and isinstance(mod.__all__, list | tuple):
        return list(mod.__all__)

    def is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        return not inspect.ismodule(getattr(mod, name))

    return sorted(filter(is_public, dir(mod)))


def write_module_page(mod_name: str) -> None:
    """Create an .rst file for *mod_name* under *OUTPUT_DIR*."""

    module = importlib.import_module(mod_name)
    symbols = public_symbols(module)

    classes: list[str] = []
    functions: list[str] = []
    constants: list[str] = []
    modules: list[str] = []

    for name in symbols:
        attr = getattr(module, name)

        # ------------------------------------------------------------------
        # Class-like objects
        # ------------------------------------------------------------------
        if inspect.isclass(attr) or wp.types.type_is_struct(attr):
            classes.append(name)
            continue

        # ------------------------------------------------------------------
        # Constants / simple values
        # ------------------------------------------------------------------
        if wp.types.type_is_value(type(attr)) or isinstance(attr, str):
            constants.append(name)
            continue

        # ------------------------------------------------------------------
        # Submodules
        # ------------------------------------------------------------------

        if inspect.ismodule(attr):
            modules.append(name)
            continue

        # ------------------------------------------------------------------
        # Everything else → functions section
        # ------------------------------------------------------------------
        functions.append(name)

    title = mod_name
    underline = "=" * len(title)

    lines: list[str] = [title, underline, ""]

    # Module docstring if available
    doc = (module.__doc__ or "").strip()
    if doc:
        lines.extend([doc, ""])

    lines.extend([f".. currentmodule:: {mod_name}", ""])

    # Render a simple bullet list of submodules (no autosummary/toctree) to
    # avoid generating stub pages that can cause duplicate descriptions.
    if modules:
        modules.sort()
        lines.extend([".. rubric:: Submodules", ""])
        # Link to sibling generated module pages without creating autosummary stubs.
        for sub in modules:
            modname = f"{mod_name}.{sub}"
            docname = modname.replace(".", "_")
            lines.append(f"- :doc:`{modname} <{docname}>`")
        lines.append("")

    if classes:
        classes.sort()
        lines.extend(
            [
                ".. rubric:: Classes",
                "",
                ".. autosummary::",
                f"   :toctree: {TOCTREE_DIR}",
                "   :nosignatures:",
                "",
            ]
        )
        lines.extend([f"   {cls}" for cls in classes])
        lines.append("")

    if functions:
        functions.sort()
        lines.extend(
            [
                ".. rubric:: Functions",
                "",
                ".. autosummary::",
                f"   :toctree: {TOCTREE_DIR}",
                "   :signatures: long",
                "",
            ]
        )
        lines.extend([f"   {fn}" for fn in functions])
        lines.append("")

    if constants:
        constants.sort()
        lines.extend(
            [
                ".. rubric:: Constants",
                "",
                ".. list-table::",
                "   :header-rows: 1",
                "",
                "   * - Name",
                "     - Value",
            ]
        )

        for const in constants:
            value = getattr(module, const, "?")

            # unpack the warp scalar value, we can remove this
            # when the warp.types.scalar_base supports __str__()
            if type(value) in wp.types.scalar_types:
                value = value.value

            lines.extend(
                [
                    f"   * - {const}",
                    f"     - {value}",
                ]
            )

        lines.append("")

    outfile = OUTPUT_DIR / f"{mod_name.replace('.', '_')}.rst"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outfile.relative_to(REPO_ROOT)} ({len(symbols)} symbols)")


# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # delete previously generated files
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    for mod in MODULES:
        write_module_page(mod)
    print("\nDone. Add docs/api/index.rst to your TOC or glob it in.")
