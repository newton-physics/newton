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


def public_symbols(mod: ModuleType) -> list[str]:
    """Return the list of public names for *mod* (honours ``__all__``)."""

    if hasattr(mod, "__all__") and isinstance(mod.__all__, list | tuple):
        return list(mod.__all__)

    def is_public(name: str) -> bool:
        if name.startswith("_"):
            return False
        return not inspect.ismodule(getattr(mod, name))

    return sorted(filter(is_public, dir(mod)))


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


def _collect_nested_class_aliases(module: ModuleType, classes: list[str]) -> list[tuple[str, type]]:
    """Return nested class aliases exposed from classes in *module*.

    A nested class alias is a class-valued attribute on a parent class where the
    target class is not lexically defined inside that parent (i.e. its qualname
    does not start with ``<Parent>.<Alias>`` in the parent module).
    """

    aliases: list[tuple[str, type]] = []
    public_top_level_classes: set[type] = set()
    for class_name in classes:
        class_obj = getattr(module, class_name, None)
        if inspect.isclass(class_obj):
            public_top_level_classes.add(class_obj)

    for cls_name in classes:
        cls_obj = getattr(module, cls_name, None)
        if not inspect.isclass(cls_obj):
            continue

        parent_qual_prefix = f"{getattr(cls_obj, '__qualname__', cls_obj.__name__)}."
        parent_module = getattr(cls_obj, "__module__", module.__name__)

        for member_name, member_obj in vars(cls_obj).items():
            if member_name.startswith("_"):
                continue
            if not inspect.isclass(member_obj):
                continue

            member_qualname = getattr(member_obj, "__qualname__", "")
            member_module = getattr(member_obj, "__module__", "")
            is_lexically_nested = member_module == parent_module and member_qualname.startswith(parent_qual_prefix)
            if is_lexically_nested:
                continue
            # Only treat aliases to module-level public classes as API aliases.
            if member_obj not in public_top_level_classes:
                continue

            aliases.append((f"{cls_name}.{member_name}", member_obj))

    aliases.sort(key=lambda item: item[0])
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[tuple[str, type]] = []
    for alias_name, alias_obj in aliases:
        if alias_name in seen:
            continue
        seen.add(alias_name)
        unique.append((alias_name, alias_obj))
    return unique


def _resolve_canonical_target_name(module: ModuleType, mod_name: str, target_cls: type) -> str:
    """Return a stable canonical reference for *target_cls* in *module* when possible."""

    for name in public_symbols(module):
        try:
            candidate = getattr(module, name)
        except Exception:
            continue
        if inspect.isclass(candidate) and candidate is target_cls:
            return f"{mod_name}.{name}"

    return f"{target_cls.__module__}.{target_cls.__qualname__}"


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

    lines: list[str] = [title, underline, ""]

    # Module docstring if available
    doc = (module.__doc__ or "").strip()
    if doc:
        lines.extend([doc, ""])

    lines.extend([f".. currentmodule:: {mod_name}", ""])

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

    class_aliases = _collect_nested_class_aliases(module, classes)
    if class_aliases and not is_solver_submodule:
        lines.extend([".. rubric:: Nested Class Aliases", ""])
        for alias_name, alias_target in class_aliases:
            canonical = _resolve_canonical_target_name(module, mod_name, alias_target)
            alias_doc = inspect.getdoc(alias_target) or ""
            lines.extend(
                [
                    f".. py:class:: {alias_name}",
                    f"   :canonical: {canonical}",
                    "",
                    f"   Alias of :class:`~{canonical}`.",
                ]
            )
            if alias_doc:
                lines.append("")
                lines.extend([f"   {line}" if line else "   " for line in alias_doc.splitlines()])
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
            if type(value) in wp.types.scalar_types:
                value = getattr(value, "value", value)

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
# Public entry point
# -----------------------------------------------------------------------------


def generate_all() -> None:
    """Regenerate all API ``.rst`` files under :data:`OUTPUT_DIR`."""

    # delete previously generated files
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    extra_solver_modules = solver_submodule_pages()
    all_modules = MODULES + [mod for mod in extra_solver_modules if mod not in MODULES]

    for mod in all_modules:
        write_module_page(mod)


# -----------------------------------------------------------------------------
# Script entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_all()
    print("\nDone. Add docs/api/index.rst to your TOC or glob it in.")
