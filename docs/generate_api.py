# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate concise API .rst files for public modules.

This helper discovers Newton's top-level public modules from ``newton.__all__``,
reads each module's ``__all__`` list (and falls back to public attributes if
``__all__`` is missing), and writes one reStructuredText file per module with an
``autosummary`` directive.  Modules may use ``__deprecated_symbols__`` to keep
warning-only compatibility names documented without resolving their values.
When Sphinx later builds the documentation (with
``autosummary_generate = True``), individual stub pages will be created
automatically for every listed symbol.

Top-level module pages and ``_toctree.rst`` live in ``docs/api/`` (checked in
so a CI drift step can flag stale output). Autosummary stub pages land under
``docs/api/_generated/`` and are git-ignored.

Usage (from the repository root):

    python docs/generate_api.py

Export new top-level modules through ``newton.__all__`` to include them in the
API reference.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re
import shutil
import sys
from datetime import datetime
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

# Generated toctree fragment included from ``docs/index.rst``. Listing every
# top-level public module exported through ``newton.__all__``.
TOCTREE_RST = OUTPUT_DIR / "_toctree.rst"

# Where autosummary should place generated stub pages (relative to each .rst
# file).  Keeping them alongside the .rst files avoids clutter elsewhere.
TOCTREE_DIR = "_generated"  # sub-folder inside OUTPUT_DIR

COPYRIGHT_RE = re.compile(r"^\.\. SPDX-FileCopyrightText: Copyright \(c\) \d{4} The Newton Developers$")
_COPYRIGHT_LINES: dict[Path, str] = {}

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


def _read_copyright_line(path: Path) -> str | None:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, OSError):
        return None

    if COPYRIGHT_RE.fullmatch(first_line):
        return first_line
    return None


def _snapshot_copyright_lines() -> None:
    """Remember generated files' original copyright lines before regeneration."""
    _COPYRIGHT_LINES.clear()
    if not OUTPUT_DIR.exists():
        return

    for path in OUTPUT_DIR.glob("*.rst"):
        existing_line = _read_copyright_line(path)
        if existing_line:
            _COPYRIGHT_LINES[path.resolve()] = existing_line


def copyright_line(path: Path) -> str:
    """Return the SPDX copyright line for a generated file."""
    existing_line = _COPYRIGHT_LINES.get(path.resolve())
    if existing_line:
        return existing_line

    existing_line = _read_copyright_line(path)
    if existing_line:
        return existing_line

    return f".. SPDX-FileCopyrightText: Copyright (c) {datetime.now().year} The Newton Developers"


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

    def add_public_module_tree(mod_name: str, module: ModuleType) -> None:
        if mod_name not in modules:
            modules.append(mod_name)
        for child_name in public_symbols(module):
            child = getattr(module, child_name)
            if not inspect.ismodule(child):
                continue
            add_public_module_tree(f"{mod_name}.{child_name}", child)

    for name in public_symbols(public_solvers):
        attr = getattr(public_solvers, name)
        if not inspect.ismodule(attr):
            continue
        add_public_module_tree(f"newton.solvers.{name}", attr)

    return modules


def write_module_page(mod_name: str, api_toctree_modules: set[str] | None = None) -> None:
    """Create an .rst file for *mod_name* under *OUTPUT_DIR*."""

    if api_toctree_modules is None:
        api_toctree_modules = set(api_modules())

    is_solver_submodule = mod_name.startswith("newton.solvers.") and mod_name != "newton.solvers"
    uses_internal_solver_module = False
    if is_solver_submodule:
        sub_name = mod_name.split(".", 2)[2]
        try:
            module = importlib.import_module(mod_name)
        except ModuleNotFoundError as exc:
            if exc.name != mod_name:
                raise
            # Some public solver helpers are exposed as attributes on
            # ``newton.solvers`` even though it is a module, not a package.
            # Document those from their implementation module while keeping the
            # public page name stable.
            module = importlib.import_module(f"newton._src.solvers.{sub_name}")
            uses_internal_solver_module = True
    else:
        module = importlib.import_module(mod_name)

    deprecated_messages = dict(getattr(module, "__deprecated_symbols__", {}))
    symbols = list(dict.fromkeys([*public_symbols(module), *deprecated_messages]))
    if uses_internal_solver_module:
        # Keep solver classes centralized in newton.solvers.
        symbols = [name for name in symbols if not name.startswith("Solver")]

    classes: list[str] = []
    functions: list[str] = []
    constants: list[str] = []
    modules: list[str] = []
    deprecated: list[str] = []

    for name in symbols:
        if name in deprecated_messages:
            deprecated.append(name)
            continue

        attr = getattr(module, name)

        # ------------------------------------------------------------------
        # Class-like objects
        # ------------------------------------------------------------------
        if inspect.isclass(attr) or wp.types.type_is_struct(attr):
            classes.append(name)
            continue

        # ------------------------------------------------------------------
        # Constants / simple values (incl. collection constants such as tuples)
        # ------------------------------------------------------------------
        if wp.types.type_is_value(type(attr)) or isinstance(attr, (str, bytes, tuple, list, frozenset)):
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
    outfile = OUTPUT_DIR / f"{mod_name.replace('.', '_')}.rst"

    lines: list[str] = [
        copyright_line(outfile),
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

    if uses_internal_solver_module:
        lines.extend(
            [
                ".. note::",
                "",
                f"   This page documents helper functions exposed through the ``{mod_name}`` attribute.",
                "   Because ``newton.solvers`` is a module rather than a package, use",
                f"   ``from newton.solvers import {sub_name}`` instead of ``import {mod_name}``.",
                "",
                f".. currentmodule:: newton._src.solvers.{sub_name}",
                "",
            ]
        )
    else:
        lines.extend([f".. py:module:: {mod_name}", f".. currentmodule:: {mod_name}", ""])

    # Render submodules as direct document links instead of autosummary stubs.
    # Child module pages still need hidden toctree edges to satisfy Sphinx.
    if modules:
        modules.sort()
        nested_modules = [sub for sub in modules if f"{mod_name}.{sub}" not in api_toctree_modules]
        if nested_modules:
            lines.extend(
                [
                    ".. toctree::",
                    "   :hidden:",
                    "",
                ]
            )
            for sub in nested_modules:
                modname = f"{mod_name}.{sub}"
                docname = modname.replace(".", "_")
                lines.append(f"   {docname}")
            lines.append("")

        lines.extend([".. rubric:: Submodules", ""])
        for sub in modules:
            modname = f"{mod_name}.{sub}"
            docname = modname.replace(".", "_")
            lines.append(f"- :doc:`{modname} <{docname}>`")
        lines.append("")

    if classes:
        classes.sort()
        lines.extend([".. rubric:: Classes", ""])
        if uses_internal_solver_module or is_solver_submodule:
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
        if uses_internal_solver_module or is_solver_submodule:
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

    if deprecated:
        deprecated.sort()
        lines.extend(
            [
                ".. rubric:: Deprecated",
                "",
                ".. list-table::",
                "   :header-rows: 1",
                "",
                "   * - Name",
                "     - Guidance",
            ]
        )
        for name in deprecated:
            lines.extend(
                [
                    f"   * - ``{name}``",
                    f"     - {deprecated_messages[name]}",
                ]
            )
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

    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {outfile.relative_to(REPO_ROOT)} ({len(symbols)} symbols)")


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def write_api_toctree(modules: list[str]) -> None:
    """Write the API Reference toctree fragment to :data:`TOCTREE_RST`.

    The file is included from ``docs/index.rst`` via ``.. include::``. Solver
    sub-module pages (from :func:`solver_submodule_pages`) are intentionally
    excluded: those nest under ``api/newton_solvers.rst`` and are not
    top-level toctree entries.
    """
    lines = [
        copyright_line(TOCTREE_RST),
        ".. SPDX-License-Identifier: CC-BY-4.0",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "   :hidden:",
        "   :caption: API Reference",
        "",
    ]
    for mod in modules:
        lines.append(f"   api/{mod.replace('.', '_')}")
    lines.append("")
    TOCTREE_RST.write_text("\n".join(lines))
    print(f"Wrote {TOCTREE_RST.relative_to(REPO_ROOT)} ({len(modules)} entries)")


def generate_all() -> None:
    """Regenerate all API ``.rst`` files under :data:`OUTPUT_DIR`."""
    _snapshot_copyright_lines()

    # delete previously generated files
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    modules = api_modules()
    extra_solver_modules = solver_submodule_pages()
    all_modules = modules + [mod for mod in extra_solver_modules if mod not in modules]

    for mod in all_modules:
        write_module_page(mod, set(modules))

    write_api_toctree(modules)


# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all()
