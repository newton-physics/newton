#!/usr/bin/env python3
"""Generate concise API .rst files for selected modules.

This helper scans a list of *top-level* modules, reads their ``__all__`` lists
(and falls back to public attributes if ``__all__`` is missing), and writes one
reStructuredText file per module with an ``autosummary`` directive.  When
Sphinx later builds the documentation (with ``autosummary_generate = True``),
individual stub pages will be created automatically for every listed symbol.

The generated files live in ``docs/api/`` (git-ignored by default).

Usage (from the repository root):

    python tools/generate_api_rst.py

Adjust ``MODULES`` below to fit your project.
"""

from __future__ import annotations

import importlib
import inspect
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
MODULES: list[str] = ["newton.sim", "newton.core", "newton.geometry", "newton.solvers", "newton.utils"]

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

    if hasattr(mod, "__all__") and isinstance(mod.__all__, (list, tuple)):
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

    for name in symbols:
        attr = getattr(module, name)

        # ------------------------------------------------------------------
        # Class-like objects
        # ------------------------------------------------------------------
        if inspect.isclass(attr) or isinstance(attr, wp.codegen.Struct):
            classes.append(name)
            continue

        # ------------------------------------------------------------------
        # Constants / simple values
        # ------------------------------------------------------------------
        if wp.types.type_is_value(type(attr)):
            constants.append(name)
            continue

        # ------------------------------------------------------------------
        # Everything else → functions section
        # ------------------------------------------------------------------
        functions.append(name)

    title = mod_name
    underline = "=" * len(title)

    lines: list[str] = [title, underline, ""]

    # Short module docstring (first paragraph) if available
    doc = (module.__doc__ or "").strip().split("\n\n")[0].strip()
    if doc:
        lines.extend([doc, ""])

    lines.extend([f".. currentmodule:: {mod_name}", ""])

    if classes:
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
    for mod in MODULES:
        write_module_page(mod)
    print("\nDone. Add docs/api/index.rst to your TOC or glob it in.")
