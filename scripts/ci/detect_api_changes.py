# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservatively detect Newton API changes with a small static AST model.

Security invariant - DO NOT BREAK:
    This script runs under ``pull_request_target`` while inspecting an
    untrusted checkout. Treat both revisions as inert data: never import,
    execute, evaluate, or spawn code from either tree. Only bounded file reads
    and stdlib ``ast`` parsing are allowed.

The result is advisory. Exact records cover ordinary declarations and the
simple export forms used by Newton. Changed constructs outside that grammar
produce ``unknown`` and still request API review.
"""

from __future__ import annotations

import ast
import hashlib
import html
import importlib.util
import json
import stat
import sys
from pathlib import Path

PACKAGE = "newton"
EXCLUDED = ("newton.examples", "newton.tests")
KNOWN_FORWARD_ALL = {"newton.solvers": {"_solvers": "newton._src.solvers"}}
KNOWN_LAZY_MAPS = {"newton._src.solvers": "_LAZY_IMPORTS"}
VIRTUAL_MODULES = {"newton.solvers.experimental.coupled": "newton._src.solvers.coupled"}

MAX_FILES = 2_000
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_AST_NODES = 200_000
MAX_SYMBOLS = 20_000
MAX_EXPR_CHARS = 2_000
MAX_REPORT_ITEMS = 20
MAX_COMMENT_CHARS = 40_000


class Definition:
    def __init__(self, symbol: dict, module: str, node: ast.AST, members: dict[str, Definition] | None = None):
        self.symbol = symbol
        self.module = module
        self.node = node
        self.members = members or {}


class Module:
    def __init__(self, name: str, tree: ast.Module, digest: str, is_package: bool):
        self.name = name
        self.tree = tree
        self.digest = digest
        self.is_package = is_package
        self.definitions: dict[str, Definition] = {}
        self.imports: dict[str, tuple[str, str | None]] = {}
        self.exports: list[str] | None = None
        self.forwarded_exports: list[str] = []
        self.lazy: dict[str, tuple[str, str | None]] = {}


def _digest(value: str | bytes) -> str:
    value = value.encode("utf-8", "backslashreplace") if isinstance(value, str) else value
    return hashlib.sha256(value).hexdigest()


def _note(notes: dict[str, dict[str, str]], key: str, reason: str, fingerprint: str | bytes) -> None:
    notes[key] = {"reason": reason[:500], "fingerprint": _digest(fingerprint)}


def _expr(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        value = ast.unparse(node)
    except Exception:
        return "?"
    return value if len(value) <= MAX_EXPR_CHARS else value[:MAX_EXPR_CHARS] + "..."


def _tail(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _tail(node.func)
    return ""


def _public_module(name: str) -> bool:
    excluded = any(name == prefix or name.startswith(prefix + ".") for prefix in EXCLUDED)
    return not excluded and all(not part.startswith("_") for part in name.split(".")[1:])


def _absolute_import(module: Module, imported: str | None, level: int) -> str:
    if not level:
        return imported or ""
    package = module.name if module.is_package else module.name.rpartition(".")[0]
    return importlib.util.resolve_name("." * level + (imported or ""), package)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    returns = f" -> {_expr(node.returns)}" if node.returns is not None else ""
    return f"{prefix}({_expr(node.args)}){returns}"


def _callable_definition(
    definitions: dict[str, Definition],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    module: str,
    *,
    method: bool,
) -> None:
    decorators = [_expr(item) or "?" for item in node.decorator_list]
    tails = {_tail(item) for item in node.decorator_list}
    accessor = None
    if "property" in tails:
        accessor = "getter"
    else:
        for item in node.decorator_list:
            if isinstance(item, ast.Attribute) and item.value and _tail(item.value) == node.name:
                if item.attr in {"getter", "setter", "deleter"}:
                    accessor = item.attr

    existing = definitions.get(node.name)
    if accessor:
        symbol = dict(existing.symbol) if existing and existing.symbol["kind"] == "property" else {"kind": "property"}
        symbol.setdefault("accessors", {})[accessor] = _signature(node)
        symbol["decorators"] = sorted(set(symbol.get("decorators", [])) | set(decorators))
        definitions[node.name] = Definition(symbol, module, node)
        return

    binding = "function"
    if method:
        binding = "classmethod" if "classmethod" in tails else "staticmethod" if "staticmethod" in tails else "instance"
    kind = "method" if method else "function"
    if existing and existing.symbol.get("kind") == kind:
        symbol = dict(existing.symbol)
    else:
        symbol = {"kind": kind, "binding": binding, "decorators": decorators, "overloads": []}
    if "overload" in tails:
        symbol.setdefault("overloads", []).append(_signature(node))
    else:
        symbol["signature"] = _signature(node)
        symbol["binding"] = binding
        symbol["decorators"] = decorators
    definitions[node.name] = Definition(symbol, module, node)


def _namedtuple_assignment(node: ast.Assign | ast.AnnAssign, module: str) -> tuple[str, Definition] | None:
    target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0] if len(node.targets) == 1 else None
    value = node.value
    if not isinstance(target, ast.Name) or not isinstance(value, ast.Call) or _tail(value.func) != "NamedTuple":
        return None
    if len(value.args) < 2 or not isinstance(value.args[1], (ast.List, ast.Tuple)):
        return None
    fields = []
    for item in value.args[1].elts:
        if not isinstance(item, (ast.List, ast.Tuple)) or len(item.elts) != 2:
            return None
        name = item.elts[0].value if isinstance(item.elts[0], ast.Constant) else None
        if not isinstance(name, str):
            return None
        fields.append({"name": name, "annotation": _expr(item.elts[1])})
    return target.id, Definition({"kind": "namedtuple", "fields": fields}, module, node)


def _assignment(definitions: dict[str, Definition], node: ast.Assign | ast.AnnAssign, module: str) -> None:
    namedtuple = _namedtuple_assignment(node, module)
    if namedtuple:
        definitions[namedtuple[0]] = namedtuple[1]
        return
    targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
    value = node.value
    for target in targets:
        if not isinstance(target, ast.Name) or target.id in {"__all__", "_LAZY_IMPORTS"}:
            continue
        symbol = {"kind": "constant", "value": _expr(value)}
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            symbol["annotation"] = _expr(node.annotation)
        definitions[target.id] = Definition(symbol, module, node)


def _class_definition(node: ast.ClassDef, module: str) -> Definition:
    bases = [_expr(base) or "?" for base in node.bases]
    decorators = [_expr(item) or "?" for item in node.decorator_list]
    namedtuple = any(_tail(base) == "NamedTuple" for base in node.bases)
    dataclass = any(_tail(item) == "dataclass" for item in node.decorator_list)
    symbol: dict[str, object] = {
        "kind": "namedtuple" if namedtuple else "class",
        "bases": bases,
        "decorators": decorators,
    }
    members: dict[str, Definition] = {}
    fields = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _callable_definition(members, child, module, method=True)
        elif isinstance(child, ast.ClassDef):
            members[child.name] = _class_definition(child, module)
        elif isinstance(child, (ast.Assign, ast.AnnAssign)):
            _assignment(members, child, module)
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                field = {"name": child.target.id, "annotation": _expr(child.annotation)}
                if child.value is not None:
                    field["default"] = _expr(child.value)
                fields.append(field)
    if namedtuple or dataclass:
        symbol["fields"] = fields
    if dataclass:
        symbol["dataclass"] = True
    constructors = {name: members[name].symbol for name in ("__new__", "__init__") if name in members}
    if constructors:
        symbol["constructors"] = constructors
    return Definition(symbol, module, node, members)


def _literal_exports(node: ast.AST, module: Module) -> tuple[list[str], list[str]] | None:
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    names: list[str] = []
    forwarded: list[str] = []
    known = KNOWN_FORWARD_ALL.get(module.name, {})
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            names.append(item.value)
        elif (
            isinstance(item, ast.Starred)
            and isinstance(item.value, ast.Attribute)
            and item.value.attr == "__all__"
            and isinstance(item.value.value, ast.Name)
            and item.value.value.id in known
        ):
            forwarded.append(known[item.value.value.id])
        else:
            return None
    return names, forwarded


def _parse_exports(module: Module, notes: dict[str, dict[str, str]]) -> None:
    for node in module.tree.body:
        target = None
        value = None
        append = False
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and isinstance(node.op, ast.Add):
            target, value, append = node.target.id, node.value, True
        if target != "__all__" or value is None:
            continue
        parsed = _literal_exports(value, module)
        if parsed is None:
            _note(
                notes,
                f"dynamic-export:{module.name}",
                f"{module.name} uses an unsupported __all__ expression",
                module.digest,
            )
            continue
        names, forwarded = parsed
        if module.exports is None or not append:
            module.exports = []
            module.forwarded_exports = []
        module.exports.extend(names)
        module.forwarded_exports.extend(forwarded)


def _parse_lazy_map(module: Module, notes: dict[str, dict[str, str]]) -> None:
    expected = KNOWN_LAZY_MAPS.get(module.name)
    if not expected:
        return
    for node in module.tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None
        )
        if not isinstance(target, ast.Name) or target.id != expected:
            continue
        value = node.value
        if not isinstance(value, ast.Dict):
            _note(
                notes,
                f"lazy-map:{module.name}",
                f"{module.name}.{expected} is not a literal dictionary",
                ast.dump(value),
            )
            return
        for key_node, value_node in zip(value.keys, value.values, strict=True):
            try:
                key = ast.literal_eval(key_node)
                spec = ast.literal_eval(value_node)
            except (TypeError, ValueError, SyntaxError):
                _note(
                    notes,
                    f"lazy-map:{module.name}",
                    f"{module.name}.{expected} contains a dynamic entry",
                    ast.dump(value_node),
                )
                continue
            if (
                not isinstance(key, str)
                or not isinstance(spec, tuple)
                or len(spec) != 2
                or not isinstance(spec[0], str)
            ):
                _note(
                    notes,
                    f"lazy-map:{module.name}",
                    f"{module.name}.{expected} contains an unsupported entry",
                    repr((key, spec)),
                )
                continue
            source = importlib.util.resolve_name(spec[0], module.name) if spec[0].startswith(".") else spec[0]
            if spec[1] is not None and not isinstance(spec[1], str):
                _note(
                    notes,
                    f"lazy-map:{module.name}:{key}",
                    f"{module.name}.{expected}[{key!r}] has a dynamic attribute",
                    repr(spec),
                )
                continue
            module.lazy[key] = (source, spec[1])


def _collect_module(module: Module, notes: dict[str, dict[str, str]]) -> None:
    for node in module.tree.body:
        if isinstance(node, ast.ImportFrom):
            source = _absolute_import(module, node.module, node.level)
            for alias in node.names:
                if alias.name == "*":
                    _note(notes, f"star-import:{module.name}", f"{module.name} contains a star import", ast.dump(node))
                    continue
                module.imports[alias.asname or alias.name] = (source, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module.imports[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
    for node in module.tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _callable_definition(module.definitions, node, module.name, method=False)
        elif isinstance(node, ast.ClassDef):
            module.definitions[node.name] = _class_definition(node, module.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            _assignment(module.definitions, node, module.name)
    _parse_exports(module, notes)
    _parse_lazy_map(module, notes)


def _load_modules(root: Path, notes: dict[str, dict[str, str]]) -> dict[str, Module]:
    root = root.resolve()
    package = root / PACKAGE
    try:
        info = package.lstat()
        unsafe = (
            stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode) or not package.resolve().is_relative_to(root)
        )
    except FileNotFoundError:
        _note(notes, "package", f"{PACKAGE}/ is missing", str(package))
        return {}
    except (OSError, RuntimeError) as error:
        _note(notes, "package", f"cannot inspect {PACKAGE}/: {type(error).__name__}", type(error).__name__)
        return {}
    if unsafe:
        _note(
            notes,
            f"unsafe-package:{PACKAGE}",
            f"{PACKAGE}/ is not an in-tree directory",
            str(info.st_mode),
        )
        return {}
    modules: dict[str, Module] = {}
    total = 0
    for index, path in enumerate(package.rglob("*.py")):
        if index >= MAX_FILES:
            _note(notes, "limit:files", f"source file count exceeds {MAX_FILES}", str(index))
            break
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        name = ".".join(parts)
        if any(name == prefix or name.startswith(prefix + ".") for prefix in EXCLUDED):
            continue
        try:
            info = path.lstat()
            if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not path.resolve().is_relative_to(root):
                _note(notes, f"unsafe-file:{name}", f"{name} is not a regular in-tree file", str(info.st_mode))
                continue
            if info.st_size > MAX_FILE_BYTES:
                _note(notes, f"limit:file:{name}", f"{name} exceeds {MAX_FILE_BYTES} bytes", str(info.st_size))
                continue
            with path.open("rb") as stream:
                data = stream.read(MAX_FILE_BYTES + 1)
        except OSError as error:
            _note(notes, f"read:{name}", f"cannot read {name}: {type(error).__name__}", type(error).__name__)
            continue
        if len(data) > MAX_FILE_BYTES or total + len(data) > MAX_TOTAL_BYTES:
            _note(notes, f"limit:bytes:{name}", f"source bytes exceed the configured limit at {name}", str(len(data)))
            break
        total += len(data)
        try:
            source = data.decode("utf-8")
            tree = ast.parse(source, filename=path.as_posix())
        except (UnicodeDecodeError, SyntaxError) as error:
            _note(notes, f"parse:{name}", f"cannot parse {name}: {type(error).__name__}", data)
            continue
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            _note(notes, f"limit:ast:{name}", f"{name} exceeds {MAX_AST_NODES} AST nodes", data)
            continue
        modules[name] = Module(name, tree, _digest(data), path.name == "__init__.py")
    for module in modules.values():
        _collect_module(module, notes)
    return modules


def _lookup(modules: dict[str, Module], module_name: str, name: str, seen: set[tuple[str, str]] | None = None):
    seen = set() if seen is None else seen
    marker = (module_name, name)
    if marker in seen or len(seen) > 100:
        return None
    seen.add(marker)
    module = modules.get(module_name)
    if module is None:
        return None
    if name in module.definitions:
        return "definition", module.definitions[name]
    reference = module.imports.get(name) or module.lazy.get(name)
    if reference:
        source, source_name = reference
        if source_name is None:
            return ("module", source) if source in modules else None
        child = source + "." + source_name
        if child in modules:
            return "module", child
        return _lookup(modules, source, source_name, seen)
    for source in module.forwarded_exports:
        found = _lookup(modules, source, name, seen)
        if found:
            return found
    return None


def _export_names(module: Module, modules: dict[str, Module]) -> list[str]:
    names = (
        list(module.exports)
        if module.exports is not None
        else [name for name in [*module.definitions, *module.imports] if not name.startswith("_")]
    )
    for source in module.forwarded_exports:
        forwarded = modules.get(source)
        if forwarded:
            names.extend(_export_names(forwarded, modules))
    return list(dict.fromkeys(names))


def _emit_definition(
    path: str,
    definition: Definition,
    symbols: dict[str, dict],
    notes: dict[str, dict[str, str]],
    modules: dict[str, Module],
) -> None:
    if len(symbols) >= MAX_SYMBOLS:
        _note(notes, "limit:symbols", f"API symbol count exceeds {MAX_SYMBOLS}", path)
        return
    symbols[path] = definition.symbol
    if definition.symbol["kind"] in {"class", "namedtuple"}:
        for base in definition.symbol.get("bases", []):
            if base in {"object", "Enum", "IntEnum", "StrEnum", "Flag", "IntFlag", "NamedTuple"}:
                continue
            found = _lookup(modules, definition.module, base) if base.isidentifier() else None
            fingerprint = modules[found[1].module].digest if found and found[0] == "definition" else base
            _note(notes, f"inheritance:{path}:{base}", f"inherited members of {path} are not expanded", fingerprint)
    for name, member in definition.members.items():
        if not name.startswith("_"):
            _emit_definition(f"{path}.{name}", member, symbols, notes, modules)


def _emit_module(
    public_name: str,
    source_name: str,
    symbols: dict[str, dict],
    notes: dict[str, dict[str, str]],
    modules: dict[str, Module],
    emitted: set[tuple[str, str]],
) -> None:
    marker = (public_name, source_name)
    if marker in emitted:
        return
    emitted.add(marker)
    module = modules.get(source_name)
    if module is None:
        _note(
            notes,
            f"missing-module:{public_name}",
            f"cannot resolve provider {source_name} for {public_name}",
            source_name,
        )
        return
    symbols[public_name] = {"kind": "module"}
    for name in _export_names(module, modules):
        found = _lookup(modules, source_name, name)
        path = f"{public_name}.{name}"
        if not found:
            _note(notes, f"unresolved-export:{path}", f"cannot resolve exported symbol {path}", module.digest)
        elif found[0] == "module":
            _emit_module(path, found[1], symbols, notes, modules, emitted)
        else:
            _emit_definition(path, found[1], symbols, notes, modules)


def _registration(stmt: ast.stmt) -> tuple[str, str, dict] | None:
    if (
        not isinstance(stmt, ast.Expr)
        or not isinstance(stmt.value, ast.Call)
        or not isinstance(stmt.value.func, ast.Attribute)
    ):
        return None
    outer = stmt.value
    kind = {"add_custom_attribute": "custom_attribute", "add_custom_frequency": "custom_frequency"}.get(outer.func.attr)
    if not kind or len(outer.args) != 1 or outer.keywords:
        return None
    constructor = outer.args[0]
    # The builder method establishes the registration kind. Newton helpers
    # commonly alias the nested constructor (for example, ``ca =
    # type(builder).CustomAttribute``), so requiring its spelling would miss
    # real registrations.
    if not isinstance(constructor, ast.Call) or constructor.args:
        return None
    values = {}
    for keyword in constructor.keywords:
        if keyword.arg is None:
            return None
        values[keyword.arg] = _expr(keyword.value)
    name_node = next((item.value for item in constructor.keywords if item.arg == "name"), None)
    namespace_node = next((item.value for item in constructor.keywords if item.arg == "namespace"), None)
    if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
        return None
    if namespace_node is not None and (
        not isinstance(namespace_node, ast.Constant) or not isinstance(namespace_node.value, str)
    ):
        return None
    namespace = namespace_node.value if namespace_node else ""
    full_name = f"{namespace}:{name_node.value}" if namespace else name_node.value
    return kind, full_name, {"kind": kind, **values}


def _iter_definitions(definitions: dict[str, Definition], prefix: str = ""):
    for name, definition in definitions.items():
        path = f"{prefix}.{name}" if prefix else name
        yield path, definition
        yield from _iter_definitions(definition.members, path)


def _iter_registrations(node: ast.AST):
    """Yield registrations without entering nested definition scopes."""
    if isinstance(node, ast.stmt):
        record = _registration(node)
        if record:
            yield record
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_registrations(child)


def _call_path(node: ast.AST) -> tuple[str, list[str]] | None:
    attributes = []
    while isinstance(node, ast.Attribute):
        attributes.append(node.attr)
        node = node.value
    return (node.id, list(reversed(attributes))) if isinstance(node, ast.Name) else None


def _called_definition(module: Module, owner: str, function: ast.AST, modules: dict[str, Module]) -> Definition | None:
    path = _call_path(function)
    if not path:
        return None
    root, attributes = path
    if root in {"self", "cls"}:
        members = module.definitions
        definition = None
        for part in owner.split(".")[:-1]:
            definition = members.get(part)
            if not definition:
                return None
            members = definition.members
        if not definition:
            return None
        found = ("definition", definition)
    else:
        found = _lookup(modules, module.name, root)
    for attribute in attributes:
        if not found:
            return None
        if found[0] == "module":
            found = _lookup(modules, found[1], attribute)
        else:
            member = found[1].members.get(attribute)
            found = ("definition", member) if member else None
    return found[1] if found and found[0] == "definition" else None


def _registration_fingerprint(module: Module, owner: str, residual: list[ast.stmt], modules: dict[str, Module]) -> str:
    """Include direct helper providers in an unsupported registration fingerprint."""
    providers: dict[str, str] = {}
    for stmt in residual:
        for call in (node for node in ast.walk(stmt) if isinstance(node, ast.Call)):
            definition = _called_definition(module, owner, call.func, modules)
            if not definition:
                continue
            body = definition.node.body if isinstance(definition.node, (ast.FunctionDef, ast.AsyncFunctionDef)) else []
            provider_residual = [item for item in body if not _registration(item) and not isinstance(item, ast.Pass)]
            key = f"{definition.module}:{ast.dump(call.func, include_attributes=False)}"
            providers[key] = "\n".join(ast.dump(item, include_attributes=False) for item in provider_residual)
    statements = [ast.dump(stmt, include_attributes=False) for stmt in residual]
    dependencies = [f"{name}:{fingerprint}" for name, fingerprint in sorted(providers.items())]
    return "\n".join([*statements, *dependencies])


def _emit_registrations(modules: dict[str, Module], symbols: dict[str, dict], notes: dict[str, dict[str, str]]) -> None:
    for module_name, module in modules.items():
        if not module_name.startswith("newton._src.solvers"):
            continue
        module_id = module_name.removeprefix("newton._src.solvers").strip(".").replace(".", "_") or "root"
        prefix = f"newton.solver_registry.{module_id}"
        seen: dict[str, dict] = {}
        for owner, definition in _iter_definitions(module.definitions):
            if not isinstance(definition.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            records = []
            for stmt in definition.node.body:
                records.extend(_iter_registrations(stmt))
            for kind, name, symbol in records:
                key = f"{prefix}.{kind}.{name}"
                previous = seen.get(key)
                if previous is not None and previous != symbol:
                    _note(
                        notes,
                        f"registration-conflict:{key}",
                        f"{key} has conflicting registrations",
                        json.dumps([previous, symbol], sort_keys=True),
                    )
                    continue
                if previous is not None:
                    continue
                if len(symbols) >= MAX_SYMBOLS:
                    _note(notes, "limit:symbols", f"API symbol count exceeds {MAX_SYMBOLS}", key)
                    return
                symbols[key] = symbol
                seen[key] = symbol

            if not records and owner.rsplit(".", 1)[-1] != "register_custom_attributes":
                continue
            residual = []
            for stmt in definition.node.body:
                if (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    continue
                if isinstance(stmt, ast.Pass):
                    continue
                record = _registration(stmt)
                if record:
                    continue
                residual.append(stmt)
            if residual:
                _note(
                    notes,
                    f"registration:{module_name}:{owner}",
                    f"{module_name}.{owner} contains delegated or unsupported registration logic",
                    _registration_fingerprint(module, owner, residual, modules),
                )


def extract_api_symbols(root: Path, notes: dict[str, dict[str, str]] | None = None) -> dict[str, dict]:
    notes = {} if notes is None else notes
    modules = _load_modules(Path(root), notes)
    symbols: dict[str, dict] = {}
    emitted: set[tuple[str, str]] = set()
    for name in sorted(module for module in modules if _public_module(module)):
        _emit_module(name, name, symbols, notes, modules, emitted)
    for public_name, source_name in VIRTUAL_MODULES.items():
        if source_name in modules:
            _emit_module(public_name, source_name, symbols, notes, modules, emitted)
    _emit_registrations(modules, symbols, notes)
    return dict(sorted(symbols.items()))


def compare_symbols(base: dict[str, dict], head: dict[str, dict]) -> dict:
    base_names = set(base)
    head_names = set(head)
    added_names = sorted(head_names - base_names)
    removed_names = sorted(base_names - head_names)
    added = [{"path": name, **head[name]} for name in added_names[:MAX_REPORT_ITEMS]]
    removed = [{"path": name, **base[name]} for name in removed_names[:MAX_REPORT_ITEMS]]
    changed = []
    changed_count = 0
    for name in sorted(base_names & head_names):
        if base[name] == head[name]:
            continue
        changed_count += 1
        if len(changed) >= MAX_REPORT_ITEMS:
            continue
        fields = []
        for field in sorted(set(base[name]) | set(head[name])):
            if base[name].get(field) != head[name].get(field):
                fields.append({"field": field, "before": base[name].get(field), "after": head[name].get(field)})
        changed.append({"path": name, "kind": head[name].get("kind", base[name].get("kind")), "changes": fields})
    summary = {
        "added_count": len(added_names),
        "removed_count": len(removed_names),
        "changed_count": changed_count,
        "total": len(added_names) + len(removed_names) + changed_count,
    }
    return {
        "has_changes": bool(summary["total"]),
        "summary": summary,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _changed_notes(base: dict[str, dict[str, str]], head: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    changes = []
    for key in sorted(set(base) | set(head)):
        before = base.get(key, {}).get("fingerprint")
        after = head.get(key, {}).get("fingerprint")
        if before != after:
            item = head.get(key) or base[key]
            changes.append({"key": key, "reason": item["reason"]})
    return changes


def _safe(value: object) -> str:
    return html.escape(str(value).replace("\r", "\\r").replace("\n", "\\n")[:500], quote=True)


def format_comment(diff: dict, decision: str, uncertainty_changes: list[dict[str, str]]) -> str:
    summary = diff["summary"]
    lines = ["### API review", ""]
    if decision == "unchanged":
        lines.append("No supported public API changes detected.")
    elif decision == "unknown":
        lines.append("Static analysis is inconclusive; API review is requested conservatively.")
    else:
        lines.append(
            f"Detected {summary['total']} interface change(s): "
            f"{summary['added_count']} added, {summary['removed_count']} removed, "
            f"{summary['changed_count']} modified."
        )
    items = []
    for label, values in (("Added", diff["added"]), ("Removed", diff["removed"]), ("Modified", diff["changed"])):
        for item in values:
            items.append(f"- {label}: <code>{_safe(item['path'])}</code> ({_safe(item.get('kind', '?'))})")
    if items:
        lines.extend(["", *items[:MAX_REPORT_ITEMS]])
        remaining = summary["total"] - min(len(items), MAX_REPORT_ITEMS)
        if remaining:
            lines.append(f"- … and {remaining} more change(s).")
    if uncertainty_changes:
        lines.extend(["", "Analysis notes:"])
        for item in uncertainty_changes[:MAX_REPORT_ITEMS]:
            lines.append(f"- {_safe(item['reason'])}")
        if len(uncertainty_changes) > MAX_REPORT_ITEMS:
            lines.append(f"- … and {len(uncertainty_changes) - MAX_REPORT_ITEMS} more note(s).")
    lines.extend(
        ["", "This check is advisory: the label means API review needed, not that a breaking change is proven."]
    )
    return "\n".join(lines)[:MAX_COMMENT_CHARS]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: detect_api_changes.py BASE_ROOT HEAD_ROOT", file=sys.stderr)
        return 2
    base_notes: dict[str, dict[str, str]] = {}
    head_notes: dict[str, dict[str, str]] = {}
    base = extract_api_symbols(Path(sys.argv[1]), base_notes)
    head = extract_api_symbols(Path(sys.argv[2]), head_notes)
    diff = compare_symbols(base, head)
    uncertainty_changes = _changed_notes(base_notes, head_notes)
    decision = "changed" if diff["has_changes"] else "unknown" if uncertainty_changes else "unchanged"
    report = {
        "decision": decision,
        "needs_review": decision != "unchanged",
        "has_analysis_warnings": bool(uncertainty_changes),
        "diff": diff,
        "analysis_notes": uncertainty_changes[:MAX_REPORT_ITEMS],
        "comment": format_comment(diff, decision, uncertainty_changes),
    }
    json.dump(report, sys.stdout, ensure_ascii=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
