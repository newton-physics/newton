# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate concise API .rst files for public modules.

This helper discovers Newton's top-level public modules from ``newton.__all__``,
reads each module's ``__all__`` list (and falls back to public attributes if
``__all__`` is missing), and writes one reStructuredText file per module with an
``autosummary`` directive.  When Sphinx later builds the documentation (with
``autosummary_generate = True``), individual stub pages will be created
automatically for every listed symbol.

The generated files live in ``docs/api/`` (git-ignored by default).

Usage (from the repository root):

    python docs/generate_api.py

Export new top-level modules through ``newton.__all__`` to include them in the
API reference.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import shutil
import sys
from pathlib import Path
from types import ModuleType

import warp as wp  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Add project root to import path so that `import newton` works when the script
# is executed from the repository root without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Output directory (relative to repo root)
OUTPUT_DIR = REPO_ROOT / "docs" / "api"

# Top-level index file whose API Reference toctree is kept in sync with
# top-level public modules exported through ``newton.__all__``. The toctree
# contents are rewritten between marker comments.
INDEX_RST = REPO_ROOT / "docs" / "index.rst"
TOCTREE_MARKER_START = ".. api-toctree-start"
TOCTREE_MARKER_END = ".. api-toctree-end"

# Where autosummary should place generated stub pages (relative to each .rst
# file).  Keeping them alongside the .rst files avoids clutter elsewhere.
TOCTREE_DIR = "_generated"  # sub-folder inside OUTPUT_DIR

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def public_symbols(mod: ModuleType) -> list[str]:
    """Return the list of public names for *mod* (honours ``__all__``)."""

    if hasattr(mod, "__all__") and isinstance(mod.__all__, list | tuple):
        return list(mod.__all__)

    def is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        return not inspect.ismodule(getattr(mod, name))

    return sorted(filter(is_public, dir(mod)))


def api_modules() -> list[str]:
    """Return top-level public Newton modules that should get API pages."""

    root = importlib.import_module("newton")
    modules = ["newton"]

    for name in root.__all__:
        attr = getattr(root, name)
        if not inspect.ismodule(attr):
            continue
        mod_name = attr.__name__
        if mod_name.startswith("newton."):
            modules.append(mod_name)

    return modules


def _is_solver_only_module(mod: ModuleType) -> bool:
    """Return True when the module only exposes its solver class."""
    names = getattr(mod, "__all__", None)
    public = list(names) if isinstance(names, (list, tuple)) else public_symbols(mod)
    return len(public) == 1 and public[0].startswith("Solver")


def solver_submodule_pages() -> list[str]:
    """Return solver submodules that expose more than the solver class."""
    modules: list[str] = []
    solvers_pkg = importlib.import_module("newton._src.solvers")
    public_solvers = importlib.import_module("newton.solvers")

    for info in pkgutil.iter_modules(solvers_pkg.__path__):
        if not info.ispkg:
            continue
        if not hasattr(public_solvers, info.name):
            continue
        internal_name = f"{solvers_pkg.__name__}.{info.name}"
        try:
            mod = importlib.import_module(internal_name)
        except Exception:
            # Optional dependency missing; skip doc generation for this solver.
            continue
        if _is_solver_only_module(mod):
            continue

        public_name = f"newton.solvers.{info.name}"
        modules.append(public_name)
    return modules


def write_module_page(mod_name: str) -> None:
    """Create an .rst file for *mod_name* under *OUTPUT_DIR*."""

    is_solver_submodule = mod_name.startswith("newton.solvers.") and mod_name != "newton.solvers"
    if is_solver_submodule:
        sub_name = mod_name.split(".", 2)[2]
        module = importlib.import_module(f"newton._src.solvers.{sub_name}")
    else:
        module = importlib.import_module(mod_name)

    symbols = public_symbols(module)
    if is_solver_submodule:
        # Keep solver classes centralized in newton.solvers.
        symbols = [name for name in symbols if not name.startswith("Solver")]

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

    lines: list[str] = [
        ".. SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers",
        ".. SPDX-License-Identifier: CC-BY-4.0",
        "",
        title,
        underline,
        "",
    ]

    # Module docstring if available
    doc = (module.__doc__ or "").strip()
    if doc:
        lines.extend([doc, ""])

    lines.extend([f".. py:module:: {mod_name}", f".. currentmodule:: {mod_name}", ""])

    # Render a simple bullet list of submodules (no autosummary/toctree) to
    # avoid generating stub pages that can cause duplicate descriptions.
    if modules and not is_solver_submodule:
        modules.sort()
        lines.extend(
            [
                ".. toctree::",
                "   :hidden:",
                "",
            ]
        )
        for sub in modules:
            modname = f"{mod_name}.{sub}"
            docname = modname.replace(".", "_")
            lines.append(f"   {docname}")
        lines.append("")

        lines.extend([".. rubric:: Submodules", ""])
        # Link to sibling generated module pages without creating autosummary stubs.
        for sub in modules:
            modname = f"{mod_name}.{sub}"
            docname = modname.replace(".", "_")
            lines.append(f"- :doc:`{modname} <{docname}>`")
        lines.append("")

    if classes:
        classes.sort()
        lines.extend([".. rubric:: Classes", ""])
        if is_solver_submodule:
            for cls in classes:
                lines.extend([f".. autoclass:: {cls}", ""])
        else:
            lines.extend(
                [
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
        lines.extend([".. rubric:: Functions", ""])
        if is_solver_submodule:
            for fn in functions:
                lines.extend([f".. autofunction:: {fn}", ""])
        else:
            lines.extend(
                [
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
            if wp.types.is_scalar(value):
                value = getattr(value, "value", value)

            lines.extend(
                [
                    f"   * - ``{const}``",
                    f"     - ``{value}``",
                ]
            )

        lines.append("")

    outfile = OUTPUT_DIR / f"{mod_name.replace('.', '_')}.rst"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outfile.relative_to(REPO_ROOT)} ({len(symbols)} symbols)")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def sync_index_toctree(modules: list[str]) -> None:
    """Rewrite the API Reference toctree in :data:`INDEX_RST` from ``modules``.

    The toctree contents are owned by this script. Hand-editing the block
    between :data:`TOCTREE_MARKER_START` and :data:`TOCTREE_MARKER_END` will
    be overwritten on the next run. Solver sub-module pages (from
    ``extra_solver_modules``) are intentionally excluded: those nest under
    ``api/newton_solvers.rst`` and are not top-level toctree entries.
    """
    text = INDEX_RST.read_text()
    start = text.find(TOCTREE_MARKER_START)
    end = text.find(TOCTREE_MARKER_END, start + len(TOCTREE_MARKER_START))
    if start < 0 or end <= start:
        sys.exit(
            f"Could not find {TOCTREE_MARKER_START!r}/{TOCTREE_MARKER_END!r} "
            f"markers in {INDEX_RST}. Bracket the API Reference toctree with "
            "these comments to enable auto-sync."
        )
    body_lines = [
        TOCTREE_MARKER_START,
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "   :hidden:",
        "   :caption: API Reference",
        "",
    ]
    for mod in modules:
        body_lines.append(f"   api/{mod.replace('.', '_')}")
    body_lines.append("")
    body_lines.append(TOCTREE_MARKER_END)
    new_block = "\n".join(body_lines)
    new_text = text[:start] + new_block + text[end + len(TOCTREE_MARKER_END):]
    if new_text != text:
        INDEX_RST.write_text(new_text)
        print(f"Updated API Reference toctree in {INDEX_RST.relative_to(REPO_ROOT)} ({len(modules)} entries)")


def generate_all() -> None:
    """Regenerate all API ``.rst`` files under :data:`OUTPUT_DIR` and sync the
    API Reference toctree in :data:`INDEX_RST` to public top-level modules."""

    # delete previously generated files
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    modules = api_modules()
    extra_solver_modules = solver_submodule_pages()
    all_modules = modules + [mod for mod in extra_solver_modules if mod not in modules]

    for mod in all_modules:
        write_module_page(mod)

    sync_index_toctree(modules)


# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all()
