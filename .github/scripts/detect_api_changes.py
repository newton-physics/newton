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
    source_module: str | None = None,
    instance_attribute: bool = False,
) -> dict:
    symbol = {"kind": kind, "signature": signature}
    if value is not None:
        symbol["value"] = value
    if source_module is not None:
        symbol["source_module"] = source_module
    if instance_attribute:
        symbol["instance_attribute"] = True
    return symbol


def _copy_symbol(defn: dict, *, source_module: str | None = None) -> dict:
    symbol = dict(defn)
    if source_module is not None and "source_module" not in symbol:
        symbol["source_module"] = source_module
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


def _qualified_name_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [*_qualified_name_parts(node.value), node.attr]
    return []


def _is_call_to_name(node: ast.AST | None, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    parts = _qualified_name_parts(node.func)
    return bool(parts) and parts[-1] == name


def _is_dataclass_decorator(node: ast.AST) -> bool:
    decorator = node.func if isinstance(node, ast.Call) else node
    parts = _qualified_name_parts(decorator)
    return bool(parts) and parts[-1] == "dataclass"


def _is_dataclass_class(node: ast.ClassDef) -> bool:
    return any(_is_dataclass_decorator(decorator) for decorator in node.decorator_list)


def _literal_bool_value(node: ast.AST, default: bool) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return node.value
    return default


def _dataclass_option(node: ast.ClassDef, name: str, default: bool) -> bool:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call) or not _is_dataclass_decorator(decorator):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == name:
                return _literal_bool_value(keyword.value, default)
    return default


def _class_has_explicit_init(node: ast.ClassDef) -> bool:
    return any(
        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == "__init__" for child in node.body
    )


def _format_dataclass_init_signature(node: ast.ClassDef, class_name: str) -> str | None:
    if not _is_dataclass_class(node) or not _dataclass_option(node, "init", True) or _class_has_explicit_init(node):
        return None

    args = [f"self: {class_name}"]
    has_kw_only_marker = False
    kw_only_default = _dataclass_option(node, "kw_only", False)
    for name, annotation, default, kw_only in _iter_dataclass_init_fields(node, kw_only_default):
        if kw_only and not has_kw_only_marker:
            args.append("*")
            has_kw_only_marker = True
        args.append(_format_dataclass_arg(name, annotation, default))
    return f"__init__({', '.join(args)})"


def _iter_dataclass_init_fields(
    node: ast.ClassDef,
    kw_only_default: bool,
) -> list[tuple[str, str, str | None, bool]]:
    fields: list[tuple[str, str, str | None, bool]] = []
    kw_only = kw_only_default

    for child in node.body:
        if not isinstance(child, ast.AnnAssign) or not isinstance(child.target, ast.Name):
            continue
        name = child.target.id
        annotation = _format_annotation(child.annotation)
        if _is_dataclass_kw_only_marker(name, annotation):
            kw_only = True
            continue
        if _is_classvar_annotation(annotation):
            continue
        field_init, default, field_kw_only = _dataclass_field_options(child.value, kw_only)
        if field_init:
            fields.append((name, annotation, default, field_kw_only))
    return fields


def _is_dataclass_kw_only_marker(name: str, annotation: str) -> bool:
    return name == "_" and annotation.rsplit(".", maxsplit=1)[-1] == "KW_ONLY"


def _is_classvar_annotation(annotation: str) -> bool:
    annotation_root = annotation.split("[", 1)[0].rsplit(".", maxsplit=1)[-1]
    return annotation_root == "ClassVar"


def _dataclass_field_options(value: ast.AST | None, kw_only_default: bool) -> tuple[bool, str | None, bool]:
    if not _is_call_to_name(value, "field"):
        return True, _format_value(value) if value is not None else None, kw_only_default

    assert isinstance(value, ast.Call)
    init = True
    default: str | None = None
    kw_only = kw_only_default
    for keyword in value.keywords:
        if keyword.arg == "init":
            init = _literal_bool_value(keyword.value, init)
        elif keyword.arg == "default":
            default = _format_value(keyword.value)
        elif keyword.arg == "default_factory":
            default = f"field(default_factory={_format_value(keyword.value) or '...'})"
        elif keyword.arg == "kw_only":
            kw_only = _literal_bool_value(keyword.value, kw_only)
    return init, default, kw_only


def _format_dataclass_arg(name: str, annotation: str, default: str | None) -> str:
    result = f"{name}: {annotation}" if annotation else name
    if default is not None:
        result += f" = {default}"
    return result


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


def _parse_module_file(
    workspace_root: Path,
    file_path: Path,
    skipped_modules: list[dict],
    skipped_module_names: set[str],
) -> ast.Module | None:
    module_name = _module_name_from_path(workspace_root, file_path)
    if module_name in skipped_module_names:
        return None

    relative_path = file_path.relative_to(workspace_root)
    try:
        source = file_path.read_text(encoding="utf-8")
        return ast.parse(source, filename=str(relative_path))
    except (OSError, SyntaxError, ValueError, RecursionError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
        skipped_module_names.add(module_name)
        skipped_modules.append(
            {
                "module": module_name,
                "path": relative_path.as_posix(),
                "reason": reason,
            }
        )
        print(f"WARNING: skipping {relative_path.as_posix()}: {reason}", file=sys.stderr)
        return None


def extract_api_symbols(workspace_root: Path, skipped_modules: list[dict] | None = None) -> dict[str, dict]:
    """Extract all public API symbols from the package using AST parsing.

    Returns a dict mapping qualified symbol paths to their metadata.
    Uses a two-pass approach: first collects definitions from all modules
    (including internal), then resolves re-exports in public modules to get
    full signatures.
    """
    package_root = workspace_root / PACKAGE_NAME
    if not package_root.exists():
        return {}
    if skipped_modules is None:
        skipped_modules = []
    skipped_module_names = {item["module"] for item in skipped_modules}

    # Pass 1: parse all modules to build a definition lookup table
    all_definitions: dict[str, dict[str, dict]] = {}
    init_modules: list[tuple[str, ast.Module]] = []
    for file_path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in file_path.parts:
            continue
        module_name = _module_name_from_path(workspace_root, file_path)
        if _is_excluded(module_name):
            continue
        tree = _parse_module_file(workspace_root, file_path, skipped_modules, skipped_module_names)
        if tree is None:
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
        tree = _parse_module_file(workspace_root, file_path, skipped_modules, skipped_module_names)
        if tree is None:
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
            _collect_class_defs(node, node.name, defs)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[node.name] = _make_symbol("function", _format_callable_signature(node))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _collect_module_assignment_defs(node, defs)


def _collect_class_defs(node: ast.ClassDef, class_name: str, defs: dict[str, dict]) -> None:
    is_enum = _is_enum_class(node)
    kind = "enum" if is_enum else "class"
    bases = [_format_annotation(base) for base in node.bases]
    defs[class_name] = _make_symbol(
        kind,
        f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}",
    )

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_public_name(child.name) and child.name != "__init__":
                continue
            method_key = f"{class_name}.{child.name}"
            defs[method_key] = _make_symbol(
                "method",
                _format_callable_signature(child, owner_name=class_name),
            )
            if child.name == "__init__":
                _collect_init_attribute_defs(child, class_name, defs)
        elif isinstance(child, ast.ClassDef):
            if _is_public_name(child.name):
                _collect_class_defs(child, f"{class_name}.{child.name}", defs)
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            _collect_class_attr_defs(child, class_name, is_enum, defs)

    dataclass_init = _format_dataclass_init_signature(node, class_name)
    if dataclass_init is not None:
        defs[f"{class_name}.__init__"] = _make_symbol("method", dataclass_init)


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


def _collect_init_attribute_defs(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_name: str,
    defs: dict[str, dict],
) -> None:
    for name, symbol in _extract_init_attribute_defs(node).items():
        defs.setdefault(f"{class_name}.{name}", symbol)


class _InitAttributeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.defs: dict[str, dict] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name, value in _iter_self_attribute_assignments(target, node.value):
                if _is_public_name(name):
                    self.defs[name] = _make_symbol(
                        "attribute",
                        value=_format_value(value),
                        instance_attribute=True,
                    )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        name = _self_attribute_name(node.target)
        if name is None or not _is_public_name(name):
            return
        annotation = _format_annotation(node.annotation)
        self.defs[name] = _make_symbol(
            "attribute",
            f"{name}: {annotation}",
            value=_format_value(node.value),
            instance_attribute=True,
        )


def _extract_init_attribute_defs(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, dict]:
    visitor = _InitAttributeVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.defs


def _iter_self_attribute_assignments(target: ast.AST, value: ast.AST) -> list[tuple[str, ast.AST]]:
    name = _self_attribute_name(target)
    if name is not None:
        return [(name, value)]
    if not isinstance(target, (ast.Tuple, ast.List)):
        return []

    value_elts: list[ast.AST | None]
    if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
        value_elts = list(value.elts)
    else:
        value_elts = [value] * len(target.elts)

    assignments: list[tuple[str, ast.AST]] = []
    for target_elt, value_elt in zip(target.elts, value_elts, strict=True):
        if value_elt is not None:
            assignments.extend(_iter_self_attribute_assignments(target_elt, value_elt))
    return assignments


def _self_attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


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
                    pkg_defs[exported_name] = _copy_symbol(source_def, source_module=source_module)
                    changed = True
                if source_def["kind"] in ("class", "enum"):
                    prefix = f"{alias.name}."
                    for key, defn in source_defs.items():
                        if key.startswith(prefix):
                            member_name = key[len(prefix) :]
                            new_key = f"{exported_name}.{member_name}"
                            if new_key not in pkg_defs:
                                pkg_defs[new_key] = _copy_symbol(defn, source_module=source_module)
                                changed = True
                elif source_def["kind"] == "module":
                    changed |= _copy_module_child_definitions(
                        alias.name,
                        exported_name,
                        source_defs,
                        pkg_defs,
                        source_module=source_def.get("source_module", source_module),
                    )
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
            target_defs[target_key] = _copy_symbol(defn, source_module=source_module)
            changed = True
    return changed


def _copy_module_child_definitions(
    source_name: str,
    target_name: str,
    source_defs: dict[str, dict],
    target_defs: dict[str, dict],
    *,
    source_module: str,
) -> bool:
    prefix = f"{source_name}."
    changed = False
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_key = f"{target_name}.{member_name}"
        if target_key not in target_defs:
            target_defs[target_key] = _copy_symbol(defn, source_module=source_module)
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
                symbols[path] = _copy_symbol(source_def, source_module=source_module)
                # Also pull in child definitions (methods/attributes of classes)
                if source_def["kind"] in ("class", "enum"):
                    _resolve_class_children(
                        alias.name,
                        module_name,
                        exported_name,
                        source_defs,
                        symbols,
                        source_module,
                    )
                elif source_def["kind"] == "module":
                    _resolve_module_children_from_defs(
                        alias.name,
                        module_name,
                        exported_name,
                        source_defs,
                        symbols,
                        source_module,
                    )
            else:
                module_alias = _resolve_module_alias(source_module, alias.name, all_definitions)
                if module_alias is not None:
                    symbols[path] = _make_symbol("module", source_module=module_alias)
                    _resolve_module_children(module_alias, module_name, exported_name, all_definitions, symbols)
                elif path not in symbols:
                    symbols[path] = _make_symbol("reexport", source_module=source_module)


def _resolve_class_children(
    source_class_name: str,
    target_module: str,
    target_class_name: str,
    source_defs: dict[str, dict],
    symbols: dict[str, dict],
    source_module: str,
) -> None:
    """Copy class member definitions into the public symbol table."""
    prefix = f"{source_class_name}."
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_path = f"{target_module}.{target_class_name}.{member_name}"
        if target_path not in symbols:
            symbols[target_path] = _copy_symbol(defn, source_module=source_module)


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
            symbols[target_path] = _copy_symbol(defn, source_module=source_module)


def _resolve_module_children_from_defs(
    source_name: str,
    target_module: str,
    target_name: str,
    source_defs: dict[str, dict],
    symbols: dict[str, dict],
    source_module: str,
) -> None:
    prefix = f"{source_name}."
    for key, defn in source_defs.items():
        if not key.startswith(prefix):
            continue
        member_name = key[len(prefix) :]
        target_path = f"{target_module}.{target_name}.{member_name}"
        if target_path not in symbols:
            symbols[target_path] = _copy_symbol(defn, source_module=source_module)


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
    class_path: str | None = None,
    class_name: str | None = None,
) -> None:
    class_path = class_path or f"{module_name}.{node.name}"
    class_name = class_name or node.name
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
                _format_callable_signature(child, owner_name=class_name),
            )
            if child.name == "__init__":
                _extract_init_attribute_symbols(child, class_path, symbols)
        elif isinstance(child, ast.ClassDef):
            if _is_public_name(child.name):
                _extract_class_symbols(
                    child,
                    module_name,
                    symbols,
                    f"{class_path}.{child.name}",
                    f"{class_name}.{child.name}",
                )
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            _extract_class_attribute_symbols(child, class_path, is_enum, symbols)

    dataclass_init = _format_dataclass_init_signature(node, class_name)
    if dataclass_init is not None:
        symbols[f"{class_path}.__init__"] = _make_symbol("method", dataclass_init)


def _extract_init_attribute_symbols(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    class_path: str,
    symbols: dict[str, dict],
) -> None:
    for name, symbol in _extract_init_attribute_defs(node).items():
        symbols.setdefault(f"{class_path}.{name}", symbol)


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
            symbols[path] = _make_symbol("reexport", source_module=source_module)


def compare_symbols(
    base_symbols: dict[str, dict],
    head_symbols: dict[str, dict],
    skipped_modules: set[str] | None = None,
) -> dict:
    """Compare two API snapshots and produce a structured diff report."""
    all_paths = sorted(set(base_symbols) | set(head_symbols))
    skipped_modules = skipped_modules or set()

    added: list[dict] = []
    removed: list[dict] = []
    changed: list[dict] = []

    for path in all_paths:
        base = base_symbols.get(path)
        head = head_symbols.get(path)
        if _symbol_depends_on_skipped_module(path, base, head, skipped_modules):
            continue

        if base is None and head is not None:
            added.append({"path": path, **_report_symbol(head)})
        elif head is None and base is not None:
            removed.append({"path": path, **_report_symbol(base)})
        elif base is not None and head is not None:
            diffs: list[dict] = []
            if base["kind"] != head["kind"]:
                diffs.append({"field": "kind", "before": base["kind"], "after": head["kind"]})
            if _compare_symbol_details(base, head):
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


def _report_symbol(symbol: dict) -> dict:
    return {key: value for key, value in symbol.items() if key not in {"source_module", "instance_attribute"}}


def _compare_symbol_details(base: dict, head: dict) -> bool:
    """Return whether signature/value changes are meaningful for a matched symbol."""
    return not (base.get("instance_attribute") or head.get("instance_attribute"))


def _symbol_depends_on_skipped_module(
    path: str,
    base: dict | None,
    head: dict | None,
    skipped_modules: set[str],
) -> bool:
    if any(_module_path_matches(path, skipped_module) for skipped_module in skipped_modules):
        return True

    for symbol in (base, head):
        source_module = symbol.get("source_module") if symbol is not None else None
        if isinstance(source_module, str) and any(
            _module_path_matches(source_module, skipped_module) for skipped_module in skipped_modules
        ):
            return True
    return False


def _module_path_matches(path: str, module_name: str) -> bool:
    return path == module_name or path.startswith(f"{module_name}.")


def format_comment(diff: dict, skipped_modules: dict[str, list[dict]] | None = None) -> str:
    """Format the diff report as a GitHub PR comment in markdown."""
    warning_entries = _format_warning_entries(skipped_modules)
    if not diff["has_changes"] and not warning_entries:
        return ""

    lines: list[str] = []

    if diff["has_changes"]:
        summary = diff["summary"]
        lines.extend(
            [
                "## Public API Changes",
                "",
                f"This PR modifies the public API surface "
                f"(**{summary['total']}** changes: "
                f"{summary['added_count']} added, "
                f"{summary['removed_count']} removed, "
                f"{summary['changed_count']} modified).",
                "",
            ]
        )

    if diff["added"]:
        lines.append("### Added")
        lines.append("")
        for item in diff["added"]:
            sig = _format_reported_symbol(item)
            lines.append(f"- {_format_code_span(item['path'])} ({_escape_report_text(item['kind'])}):")
            lines.extend(_format_python_block(sig, indent="  "))
        lines.append("")

    if diff["removed"]:
        lines.append("### Removed")
        lines.append("")
        for item in diff["removed"]:
            sig = _format_reported_symbol(item)
            lines.append(f"- {_format_code_span(item['path'])} ({_escape_report_text(item['kind'])}):")
            lines.extend(_format_python_block(sig, indent="  "))
        lines.append("")

    if diff["changed"]:
        lines.append("### Modified")
        lines.append("")
        for item in diff["changed"]:
            lines.append(f"- {_format_code_span(item['path'])} ({_escape_report_text(item['kind'])})")
            for change in item["changes"]:
                if change["field"] == "signature":
                    before, after = _compact_signature_change(change.get("before"), change.get("after"))
                    if before:
                        lines.append("  - before:")
                        lines.extend(_format_python_block(before, indent="    "))
                    if after:
                        lines.append("  - after:")
                        lines.extend(_format_python_block(after, indent="    "))
                else:
                    lines.append(f"  - {_escape_report_text(change['field'])}:")
                    lines.append("    - before:")
                    lines.extend(_format_python_block(change["before"], indent="      "))
                    lines.append("    - after:")
                    lines.extend(_format_python_block(change["after"], indent="      "))
        lines.append("")

    if warning_entries:
        if lines:
            lines.append("")
        lines.extend(
            [
                "## Analysis warnings",
                "",
                "Some modules could not be parsed. API diffs that depend on those modules were suppressed.",
                "",
            ]
        )
        for side, item in warning_entries:
            lines.append(
                f"- {side}: {_format_code_span(item['path'])} "
                f"({_format_code_span(item['module'])}): {_format_code_span(item['reason'])}"
            )
        lines.append("")

    return "\n".join(lines)


def _format_warning_entries(skipped_modules: dict[str, list[dict]] | None) -> list[tuple[str, dict]]:
    if not skipped_modules:
        return []
    return [(side, item) for side, items in skipped_modules.items() for item in items]


def _compact_signature_change(before: object, after: object) -> tuple[object, object]:
    if not isinstance(before, str) or not isinstance(after, str):
        return before, after

    compact = _compact_callable_signatures(before, after)
    if compact is None:
        return before, after
    return compact


def _compact_callable_signatures(before: str, after: str) -> tuple[str, str] | None:
    before_parts = _split_callable_signature(before)
    after_parts = _split_callable_signature(after)
    if before_parts is None or after_parts is None:
        return None

    before_prefix, before_args, before_suffix = before_parts
    after_prefix, after_args, after_suffix = after_parts
    if before_prefix != after_prefix:
        return None

    original_len = max(len(before), len(after))
    if original_len < 160:
        return None

    prefix_count = 0
    for before_arg, after_arg in zip(before_args, after_args, strict=False):
        if before_arg != after_arg:
            break
        prefix_count += 1

    suffix_count = 0
    max_suffix_count = min(len(before_args) - prefix_count, len(after_args) - prefix_count)
    while (
        suffix_count < max_suffix_count
        and before_args[len(before_args) - suffix_count - 1] == after_args[len(after_args) - suffix_count - 1]
    ):
        suffix_count += 1

    if prefix_count + suffix_count < 3 and before_args != after_args:
        return None

    if before_args == after_args:
        compact_before = _render_compact_signature(before_prefix, before_args, before_suffix, 0, 0)
        compact_after = _render_compact_signature(after_prefix, after_args, after_suffix, 0, 0)
    else:
        context = 1
        before_start = max(prefix_count - context, 0)
        after_start = max(prefix_count - context, 0)
        before_end = min(len(before_args) - suffix_count + context, len(before_args))
        after_end = min(len(after_args) - suffix_count + context, len(after_args))
        compact_before = _render_compact_signature(
            before_prefix,
            before_args,
            before_suffix,
            before_start,
            before_end,
        )
        compact_after = _render_compact_signature(
            after_prefix,
            after_args,
            after_suffix,
            after_start,
            after_end,
        )

    compact_len = max(len(compact_before), len(compact_after))
    if compact_len > original_len - 40:
        return None
    return compact_before, compact_after


def _split_callable_signature(signature: str) -> tuple[str, list[str], str] | None:
    open_index = signature.find("(")
    if open_index < 0:
        return None

    close_index = _find_matching_paren(signature, open_index)
    if close_index is None:
        return None

    args = _split_top_level_args(signature[open_index + 1 : close_index])
    return signature[: open_index + 1], args, signature[close_index:]


def _find_matching_paren(text: str, open_index: int) -> int | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text[open_index:], start=open_index):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return None


def _split_top_level_args(text: str) -> list[str]:
    if not text.strip():
        return []

    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1

    args.append(text[start:].strip())
    return args


def _render_compact_signature(prefix: str, args: list[str], suffix: str, start: int, end: int) -> str:
    rendered_args: list[str] = []
    if start > 0:
        rendered_args.append("...")
    rendered_args.extend(args[start:end])
    if end < len(args):
        rendered_args.append("...")
    return f"{prefix}{', '.join(rendered_args)}{suffix}"


def _format_code_span(value: object) -> str:
    text = _escape_report_text(value)
    if "`" not in text:
        return f"`{text}`"

    fence = _make_backtick_fence(text)
    return f"{fence} {text} {fence}"


def _format_python_block(value: object, *, indent: str = "") -> list[str]:
    text = _escape_report_text(value)
    fence = _make_backtick_fence(text)
    lines = [f"{indent}{fence}python"]
    lines.extend(f"{indent}{line}" for line in text.splitlines() or [""])
    lines.append(f"{indent}{fence}")
    return lines


def _make_backtick_fence(text: str) -> str:
    longest_run = 0
    current_run = 0
    for char in text:
        if char == "`":
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return "`" * max(3, longest_run + 1)


def _escape_report_text(value: object) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text.replace("&", "&amp;").replace("<", "&lt;")


def _format_reported_symbol(item: dict) -> str:
    signature = str(item.get("signature")) if item.get("signature") is not None else None
    value = str(item.get("value")) if item.get("value") is not None else None
    if value is None:
        return signature or str(item["path"].rsplit(".", 1)[-1])
    if signature:
        return f"{signature} = {value}"
    return f"{item['path'].rsplit('.', 1)[-1]} = {value}"


def main() -> int:
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <base-workspace> <head-workspace>", file=sys.stderr)
        return 1

    base_root = Path(sys.argv[1]).resolve()
    head_root = Path(sys.argv[2]).resolve()

    base_skipped_modules: list[dict] = []
    head_skipped_modules: list[dict] = []
    base_symbols = extract_api_symbols(base_root, base_skipped_modules)
    head_symbols = extract_api_symbols(head_root, head_skipped_modules)
    skipped_modules = {
        "base": base_skipped_modules,
        "head": head_skipped_modules,
    }
    skipped_module_names = {item["module"] for items in skipped_modules.values() for item in items}
    diff = compare_symbols(base_symbols, head_symbols, skipped_module_names)

    output = {
        "diff": diff,
        "skipped_modules": skipped_modules,
        "has_analysis_warnings": bool(skipped_module_names),
        "comment": format_comment(diff, skipped_modules),
    }
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
