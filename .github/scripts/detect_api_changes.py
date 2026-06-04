# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Detect public API changes between two revisions using static AST analysis.

This script performs purely static analysis (no code execution) to extract the
public API surface from the Newton package and compare two revisions. It outputs
a JSON report describing added, removed, and modified public symbols.

Safety: only stdlib ``ast`` parsing is used; no user code is imported or executed.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

PACKAGE_NAME = "newton"
EXCLUDED_MODULE_PREFIXES = ("newton.examples", "newton.tests")
ENUM_BASE_NAMES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}


def _module_name_from_path(workspace_root: Path, file_path: Path) -> str:
    relative = file_path.relative_to(workspace_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _is_public_module(module_name: str) -> bool:
    parts = module_name.split(".")
    return not any(part.startswith("_") for part in parts[1:])


def _is_excluded(module_name: str) -> bool:
    return any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in EXCLUDED_MODULE_PREFIXES)


def _is_public_name(name: str) -> bool:
    return not name.startswith("_")


def _is_exported_name(name: str, dunder_all: list[str] | None) -> bool:
    if dunder_all is not None:
        return name in dunder_all
    return _is_public_name(name)


def _is_package_module(module_name: str | None) -> bool:
    return module_name == PACKAGE_NAME or (module_name is not None and module_name.startswith(f"{PACKAGE_NAME}."))


def _format_annotation(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "?"


def _format_value(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _make_symbol(
    kind: str,
    signature: str | None = None,
    *,
    value: str | None = None,
) -> dict:
    symbol = {"kind": kind, "signature": signature}
    if value is not None:
        symbol["value"] = value
    return symbol


def _format_arguments(args: ast.arguments, *, owner_name: str | None = None) -> list[str]:
    rendered: list[str] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)

    for i, arg in enumerate(args.posonlyargs):
        rendered.append(_format_arg(arg, default=defaults[i], owner_name=owner_name))
    if args.posonlyargs:
        rendered.append("/")
    for i, arg in enumerate(args.args, start=len(args.posonlyargs)):
        rendered.append(_format_arg(arg, default=defaults[i], owner_name=owner_name))
    if args.vararg:
        rendered.append(f"*{_format_arg(args.vararg, owner_name=owner_name)}")
    elif args.kwonlyargs:
        rendered.append("*")
    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        rendered.append(_format_arg(kw_arg, default=kw_default, owner_name=owner_name))
    if args.kwarg:
        rendered.append(f"**{_format_arg(args.kwarg, owner_name=owner_name)}")
    return rendered


def _format_arg(
    arg: ast.arg,
    *,
    default: ast.AST | None = None,
    owner_name: str | None = None,
) -> str:
    annotation = arg.annotation
    if annotation is None and owner_name and arg.arg == "self":
        type_str = owner_name
    elif annotation is None and owner_name and arg.arg == "cls":
        type_str = f"type[{owner_name}]"
    else:
        type_str = _format_annotation(annotation) if annotation else None

    result = arg.arg
    if type_str:
        result += f": {type_str}"
    if default is not None:
        result += f" = {_format_value(default) or '...'}"
    return result


def _format_callable_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    owner_name: str | None = None,
) -> str:
    args = _format_arguments(node.args, owner_name=owner_name)
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = f" -> {_format_annotation(node.returns)}" if node.returns else ""
    return f"{prefix}{node.name}({', '.join(args)}){returns}"


def _is_enum_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = _format_annotation(base).split(".")[-1]
        if base_name in ENUM_BASE_NAMES or base_name.endswith("Enum") or base_name.endswith("Flag"):
            return True
    return False


def _extract_dunder_all(tree: ast.Module) -> list[str] | None:
    """Extract __all__ list if defined, else return None."""
    found = False
    values: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    extracted = _extract_string_literals(node.value)
                    if extracted is None:
                        return None
                    found = True
                    values = extracted
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                extracted = _extract_string_literals(node.value)
                if extracted is None:
                    return None
                found = True
                values = extracted
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                extracted = _extract_string_literals(node.value)
                if not found or not isinstance(node.op, ast.Add) or extracted is None:
                    return None
                values.extend(extracted)
    return values if found else None


def _extract_string_literals(node: ast.AST | None) -> list[str] | None:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return None

    values: list[str] = []
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.append(item.value)
    return values


def extract_api_symbols(workspace_root: Path) -> dict[str, dict]:
    """Extract all public API symbols from the package using AST parsing.

    Returns a dict mapping qualified symbol paths to their metadata.
    Uses a two-pass approach: first collects definitions from all modules
    (including internal), then resolves re-exports in public modules to get
    full signatures.
    """
    package_root = workspace_root / PACKAGE_NAME
    if not package_root.exists():
        return {}

    # Pass 1: parse all modules to build a definition lookup table
    all_definitions: dict[str, dict[str, dict]] = {}
    init_modules: list[tuple[str, ast.Module]] = []
    for file_path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in file_path.parts:
            continue
        module_name = _module_name_from_path(workspace_root, file_path)
        if _is_excluded(module_name):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path.relative_to(workspace_root)))
        except (OSError, SyntaxError):
            continue
        module_defs: dict[str, dict] = {}
        _collect_definitions(tree, module_name, module_defs)
        all_definitions[module_name] = module_defs
        if file_path.name == "__init__.py":
            init_modules.append((module_name, tree))

    # Propagate re-exports through __init__ packages so that e.g.
    # newton._src.actuators.__init__ which does `from .actuator import Actuator`
    # gets `Actuator` in its definitions table.
    changed = True
    while changed:
        changed = False
        for pkg_module_name, pkg_tree in init_modules:
            changed |= _propagate_init_reexports(pkg_module_name, pkg_tree, all_definitions)

    # Pass 2: build the public API from public modules only
    symbols: dict[str, dict] = {}
    for file_path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in file_path.parts:
            continue
        module_name = _module_name_from_path(workspace_root, file_path)
        if _is_excluded(module_name) or not _is_public_module(module_name):
            continue
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path.relative_to(workspace_root)))
        except (OSError, SyntaxError):
            continue
        is_package = file_path.name == "__init__.py"
        dunder_all = _extract_dunder_all(tree)
        _extract_module_symbols(tree, module_name, symbols, dunder_all, is_package)
        _resolve_reexports(tree, module_name, symbols, dunder_all, all_definitions, is_package)

    return symbols


def _collect_definitions(tree: ast.Module, module_name: str, defs: dict[str, dict]) -> None:
    """Collect all top-level definitions from a module (for re-export resolution)."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            kind = "enum" if _is_enum_class(node) else "class"
            bases = [_format_annotation(base) for base in node.bases]
            sig = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
            defs[node.name] = _make_symbol(kind, sig)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not _is_public_name(child.name) and child.name != "__init__":
                        continue
                    method_key = f"{node.name}.{child.name}"
                    defs[method_key] = _make_symbol(
                        "method",
                        _format_callable_signature(child, owner_name=node.name),
                    )
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    _collect_class_attr_defs(child, node.name, _is_enum_class(node), defs)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = _make_symbol("function", _format_callable_signature(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _collect_module_assignment_defs(node, defs)


def _collect_class_attr_defs(
    node: ast.Assign | ast.AnnAssign,
    class_name: str,
    is_enum: bool,
    defs: dict[str, dict],
) -> None:
    if isinstance(node, ast.Assign):
        value = _format_value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_public_name(target.id):
                key = f"{class_name}.{target.id}"
                if is_enum:
                    defs[key] = _make_symbol("enum_member", value=value)
                elif target.id.isupper():
                    defs[key] = _make_symbol("constant", value=value)
                else:
                    defs[key] = _make_symbol("attribute", value=value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        if _is_public_name(name):
            key = f"{class_name}.{name}"
            annotation = _format_annotation(node.annotation)
            value = _format_value(node.value)
            if is_enum:
                defs[key] = _make_symbol("enum_member", value=value)
            elif name.isupper():
                defs[key] = _make_symbol("constant", f"{name}: {annotation}", value=value)
            else:
                defs[key] = _make_symbol("attribute", f"{name}: {annotation}", value=value)


def _collect_module_assignment_defs(node: ast.Assign | ast.AnnAssign, defs: dict[str, dict]) -> None:
    if isinstance(node, ast.Assign):
        value = _format_value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id != "__all__" and _is_public_name(target.id):
                kind = "constant" if target.id.isupper() else "variable"
                defs[target.id] = _make_symbol(kind, value=value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        if name != "__all__" and _is_public_name(name):
            annotation = _format_annotation(node.annotation)
            kind = "constant" if name.isupper() else "variable"
            defs[name] = _make_symbol(kind, f"{name}: {annotation}", value=_format_value(node.value))


def _propagate_init_reexports(
    pkg_module_name: str,
    pkg_tree: ast.Module,
    all_definitions: dict[str, dict[str, dict]],
) -> bool:
    """Propagate symbols re-exported via __init__.py into its definition table."""
    pkg_defs = all_definitions.setdefault(pkg_module_name, {})
    dunder_all = _extract_dunder_all(pkg_tree)
    changed = False

    for node in pkg_tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        source_module = _resolve_import_source(node, pkg_module_name, is_package=True)
        if not _is_package_module(source_module):
            continue
        source_defs = all_definitions.get(source_module, {})
        for alias in node.names:
            if alias.name == "*":
                continue
            exported_name = alias.asname or alias.name
            if not _is_exported_name(exported_name, dunder_all):
                continue
            source_def = source_defs.get(alias.name)
            if source_def:
                if exported_name not in pkg_defs:
                    pkg_defs[exported_name] = dict(source_def)
                    changed = True
                if source_def["kind"] in ("class", "enum"):
                    prefix = f"{alias.name}."
                    for key, defn in source_defs.items():
                        if key.startswith(prefix):
                            member_name = key[len(prefix) :]
                            new_key = f"{exported_name}.{member_name}"
                            if new_key not in pkg_defs:
                                pkg_defs[new_key] = dict(defn)
                                changed = True
                elif source_def["kind"] == "module":
                    changed |= _copy_module_child_definitions(alias.name, exported_name, source_defs, pkg_defs)
            else:
                module_name = _resolve_module_alias(source_module, alias.name, all_definitions)
                if module_name is None:
                    continue
                if exported_name not in pkg_defs:
                    pkg_defs[exported_name] = _make_symbol("module")
                    changed = True
                changed |= _copy_module_definitions(module_name, exported_name, all_definitions, pkg_defs)
    return changed


def _resolve_module_alias(
    source_module: str | None,
    alias_name: str,
    all_definitions: dict[str, dict[str, dict]],
) -> str | None:
    if source_module is None:
        return None
    module_name = f"{source_module}.{alias_name}"
    return module_name if module_name in all_definitions else None


def _copy_module_definitions(
    source_module: str,
    target_name: str,
    all_definitions: dict[str, dict[str, dict]],
    target_defs: dict[str, dict],
) -> bool:
    source_defs = all_definitions.get(source_module, {})
    changed = False
    for key, defn in source_defs.items():
        target_key = f"{target_name}.{key}"
        if target_key not in target_defs:
            target_defs[target_key] = dict(defn)
            changed = True
    return changed


def _copy_module_child_definitions(
    source_name: str,
    target_name: str,
    source_defs: dict[str, dict],
    target_defs: dict[str, dict],
) -> bool:
    prefix = f"{source_name}."
    changed = False
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_key = f"{target_name}.{member_name}"
        if target_key not in target_defs:
            target_defs[target_key] = dict(defn)
            changed = True
    return changed


def _resolve_reexports(
    tree: ast.Module,
    module_name: str,
    symbols: dict[str, dict],
    dunder_all: list[str] | None,
    all_definitions: dict[str, dict[str, dict]],
    is_package: bool,
) -> None:
    """Resolve re-exported symbols to their actual definitions for signatures."""
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        source_module = _resolve_import_source(node, module_name, is_package=is_package)
        if not _is_package_module(source_module):
            continue
        source_defs = all_definitions.get(source_module, {})
        for alias in node.names:
            if alias.name == "*":
                continue
            exported_name = alias.asname or alias.name
            if not _is_exported_name(exported_name, dunder_all):
                continue
            path = f"{module_name}.{exported_name}"
            # If we already have a full definition, skip
            if path in symbols and symbols[path]["kind"] != "reexport":
                continue
            # Try to resolve from source module definitions
            source_def = source_defs.get(alias.name)
            if source_def:
                symbols[path] = dict(source_def)
                # Also pull in child definitions (methods/attributes of classes)
                if source_def["kind"] in ("class", "enum"):
                    _resolve_class_children(alias.name, module_name, exported_name, source_defs, symbols)
                elif source_def["kind"] == "module":
                    _resolve_module_children_from_defs(alias.name, module_name, exported_name, source_defs, symbols)
            else:
                module_alias = _resolve_module_alias(source_module, alias.name, all_definitions)
                if module_alias is not None:
                    symbols[path] = _make_symbol("module")
                    _resolve_module_children(module_alias, module_name, exported_name, all_definitions, symbols)
                elif path not in symbols:
                    symbols[path] = _make_symbol("reexport")


def _resolve_class_children(
    source_class_name: str,
    target_module: str,
    target_class_name: str,
    source_defs: dict[str, dict],
    symbols: dict[str, dict],
) -> None:
    """Copy class member definitions into the public symbol table."""
    prefix = f"{source_class_name}."
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_path = f"{target_module}.{target_class_name}.{member_name}"
        if target_path not in symbols:
            symbols[target_path] = dict(defn)


def _resolve_module_children(
    source_module: str,
    target_module: str,
    target_name: str,
    all_definitions: dict[str, dict[str, dict]],
    symbols: dict[str, dict],
) -> None:
    source_defs = all_definitions.get(source_module, {})
    for key, defn in source_defs.items():
        target_path = f"{target_module}.{target_name}.{key}"
        if target_path not in symbols:
            symbols[target_path] = dict(defn)


def _resolve_module_children_from_defs(
    source_name: str,
    target_module: str,
    target_name: str,
    source_defs: dict[str, dict],
    symbols: dict[str, dict],
) -> None:
    prefix = f"{source_name}."
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_path = f"{target_module}.{target_name}.{member_name}"
        if target_path not in symbols:
            symbols[target_path] = dict(defn)


def _resolve_import_source(node: ast.ImportFrom, current_module: str, *, is_package: bool = False) -> str | None:
    """Resolve a relative or absolute import to a full module name."""
    if node.level == 0:
        return node.module
    parts = current_module.split(".")
    # For a package __init__.py, the module name IS the package, so level=1
    # means relative to itself. For a regular .py file, level=1 means relative
    # to the parent package.
    if is_package:
        # __init__.py: don't strip the last part for level=1
        up = node.level - 1
    else:
        up = node.level
    base_parts = parts[:-up] if up > 0 and up < len(parts) else (parts if up == 0 else [])
    if node.module:
        base_parts = list(base_parts) + node.module.split(".")
    return ".".join(base_parts) if base_parts else None


def _extract_module_symbols(
    tree: ast.Module,
    module_name: str,
    symbols: dict[str, dict],
    dunder_all: list[str] | None,
    is_package: bool = False,
) -> None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if not _is_exported_name(node.name, dunder_all):
                continue
            _extract_class_symbols(node, module_name, symbols)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_exported_name(node.name, dunder_all):
                continue
            path = f"{module_name}.{node.name}"
            symbols[path] = _make_symbol("function", _format_callable_signature(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _extract_assignment_symbols(node, module_name, symbols, dunder_all)
        elif isinstance(node, ast.ImportFrom):
            _extract_reexport_symbols(
                node,
                module_name,
                symbols,
                dunder_all,
                is_package,
            )


def _extract_class_symbols(
    node: ast.ClassDef,
    module_name: str,
    symbols: dict[str, dict],
) -> None:
    class_path = f"{module_name}.{node.name}"
    is_enum = _is_enum_class(node)
    kind = "enum" if is_enum else "class"

    bases = [_format_annotation(base) for base in node.bases]
    symbols[class_path] = _make_symbol(
        kind,
        f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
    )

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public_name(child.name) and child.name != "__init__":
                continue
            method_path = f"{class_path}.{child.name}"
            symbols[method_path] = _make_symbol(
                "method",
                _format_callable_signature(child, owner_name=node.name),
            )
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            _extract_class_attribute_symbols(child, class_path, is_enum, symbols)


def _extract_class_attribute_symbols(
    node: ast.Assign | ast.AnnAssign,
    class_path: str,
    is_enum: bool,
    symbols: dict[str, dict],
) -> None:
    if isinstance(node, ast.Assign):
        value = _format_value(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name) and _is_public_name(target.id):
                attr_path = f"{class_path}.{target.id}"
                if is_enum:
                    symbols[attr_path] = _make_symbol("enum_member", value=value)
                elif target.id.isupper():
                    symbols[attr_path] = _make_symbol("constant", value=value)
                else:
                    symbols[attr_path] = _make_symbol("attribute", value=value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        if _is_public_name(name):
            annotation = _format_annotation(node.annotation)
            attr_path = f"{class_path}.{name}"
            value = _format_value(node.value)
            if is_enum:
                symbols[attr_path] = _make_symbol("enum_member", value=value)
            elif name.isupper():
                symbols[attr_path] = _make_symbol("constant", f"{name}: {annotation}", value=value)
            else:
                symbols[attr_path] = _make_symbol("attribute", f"{name}: {annotation}", value=value)


def _extract_assignment_symbols(
    node: ast.Assign | ast.AnnAssign,
    module_name: str,
    symbols: dict[str, dict],
    dunder_all: list[str] | None,
) -> None:
    if isinstance(node, ast.Assign):
        value = _format_value(node.value)
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name == "__all__":
                continue
            if dunder_all is None and (not _is_public_name(name) or not name.isupper()):
                continue
            if dunder_all is not None and name not in dunder_all:
                continue
            kind = "constant" if name.isupper() else "variable"
            symbols[f"{module_name}.{name}"] = _make_symbol(kind, value=value)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        name = node.target.id
        if name == "__all__":
            return
        if dunder_all is None and (not _is_public_name(name) or not name.isupper()):
            return
        if dunder_all is not None and name not in dunder_all:
            return
        annotation = _format_annotation(node.annotation)
        kind = "constant" if name.isupper() else "variable"
        symbols[f"{module_name}.{name}"] = _make_symbol(
            kind,
            f"{name}: {annotation}",
            value=_format_value(node.value),
        )


def _extract_reexport_symbols(
    node: ast.ImportFrom,
    module_name: str,
    symbols: dict[str, dict],
    dunder_all: list[str] | None,
    is_package: bool,
) -> None:
    """Record re-exported names as public symbols of this module."""
    if node.names and node.names[0].name == "*":
        return
    source_module = _resolve_import_source(node, module_name, is_package=is_package)
    if not _is_package_module(source_module):
        return
    for alias in node.names:
        exported_name = alias.asname or alias.name
        if not _is_exported_name(exported_name, dunder_all):
            continue
        path = f"{module_name}.{exported_name}"
        if path not in symbols:
            symbols[path] = _make_symbol("reexport")


def compare_symbols(
    base_symbols: dict[str, dict],
    head_symbols: dict[str, dict],
) -> dict:
    """Compare two API snapshots and produce a structured diff report."""
    all_paths = sorted(set(base_symbols) | set(head_symbols))

    added: list[dict] = []
    removed: list[dict] = []
    changed: list[dict] = []

    for path in all_paths:
        base = base_symbols.get(path)
        head = head_symbols.get(path)

        if base is None and head is not None:
            added.append({"path": path, **head})
        elif head is None and base is not None:
            removed.append({"path": path, **base})
        elif base is not None and head is not None:
            diffs: list[dict] = []
            if base["kind"] != head["kind"]:
                diffs.append({"field": "kind", "before": base["kind"], "after": head["kind"]})
            if base.get("signature") != head.get("signature"):
                diffs.append(
                    {
                        "field": "signature",
                        "before": base.get("signature"),
                        "after": head.get("signature"),
                    }
                )
            if base.get("value") != head.get("value"):
                diffs.append(
                    {
                        "field": "value",
                        "before": base.get("value"),
                        "after": head.get("value"),
                    }
                )
            if diffs:
                changed.append({"path": path, "kind": head["kind"], "changes": diffs})

    return {
        "has_changes": bool(added or removed or changed),
        "added": added,
        "removed": removed,
        "changed": changed,
        "summary": {
            "added_count": len(added),
            "removed_count": len(removed),
            "changed_count": len(changed),
            "total": len(added) + len(removed) + len(changed),
        },
    }


def format_comment(diff: dict) -> str:
    """Format the diff report as a GitHub PR comment in markdown."""
    if not diff["has_changes"]:
        return ""

    summary = diff["summary"]
    lines = [
        "## Public API Changes",
        "",
        f"This PR modifies the public API surface "
        f"(**{summary['total']}** changes: "
        f"{summary['added_count']} added, "
        f"{summary['removed_count']} removed, "
        f"{summary['changed_count']} modified).",
        "",
    ]

    if diff["added"]:
        lines.append("### Added")
        lines.append("")
        for item in diff["added"]:
            sig = _format_reported_symbol(item)
            lines.append(f"- `{item['path']}` ({item['kind']}): `{sig}`")
        lines.append("")

    if diff["removed"]:
        lines.append("### Removed")
        lines.append("")
        for item in diff["removed"]:
            sig = _format_reported_symbol(item)
            lines.append(f"- `{item['path']}` ({item['kind']}): `{sig}`")
        lines.append("")

    if diff["changed"]:
        lines.append("### Modified")
        lines.append("")
        for item in diff["changed"]:
            lines.append(f"- `{item['path']}` ({item['kind']})")
            for change in item["changes"]:
                if change["field"] == "signature":
                    if change["before"]:
                        lines.append(f"  - before: `{change['before']}`")
                    if change["after"]:
                        lines.append(f"  - after: `{change['after']}`")
                else:
                    lines.append(f"  - {change['field']}: `{change['before']}` -> `{change['after']}`")
        lines.append("")

    return "\n".join(lines)


def _format_reported_symbol(item: dict) -> str:
    signature = item.get("signature")
    value = item.get("value")
    if value is None:
        return signature or item["path"].rsplit(".", 1)[-1]
    if signature:
        return f"{signature} = {value}"
    return f"{item['path'].rsplit('.', 1)[-1]} = {value}"


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <base-workspace> <head-workspace>", file=sys.stderr)
        return 1

    base_root = Path(sys.argv[1]).resolve()
    head_root = Path(sys.argv[2]).resolve()

    base_symbols = extract_api_symbols(base_root)
    head_symbols = extract_api_symbols(head_root)
    diff = compare_symbols(base_symbols, head_symbols)

    output = {
        "diff": diff,
        "comment": format_comment(diff),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
