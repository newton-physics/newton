# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Locate installed documentation and describe public Python objects."""

import inspect
import json
import pkgutil
import re
import sys
from pathlib import Path
from typing import TextIO

from .._version import __version__

_WEBSITE = "https://newton-physics.github.io/newton/"


def _location() -> tuple[Path | None, str]:
    package = Path(__file__).resolve().parents[1]
    bundled = package / "docs"
    if (bundled / "manifest.json").is_file():
        manifest = json.loads((bundled / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != __version__:
            raise RuntimeError(f"Documentation version {manifest['version']} does not match Newton {__version__}")
        return bundled, "bundled snapshot"
    source = package.parent / "docs"
    if (source / "conf.py").is_file() and (source / "index.rst").is_file():
        return source, "source documentation (unrendered)"
    return None, "unavailable"


def _identity(value: object) -> str | None:
    if inspect.ismodule(value):
        return value.__name__
    value = getattr(value, "__func__", value)
    if type(value).__module__.startswith("warp."):
        value = getattr(value, "func", value)
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    return f"{module}:{qualname}" if module and qualname else None


def _matching_names(query: str, names) -> list[str]:
    for candidate in (query, f"newton.{query}"):
        if candidate in names:
            return [candidate]
    return sorted(name for name in names if name.endswith(f".{query}"))


def _resolve_name(query: str, names) -> str:
    matches = _matching_names(query, names)
    if not matches:
        raise ValueError(f"No documented public symbol matches {query!r}. Pass an object to inspect its docstring.")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous documentation name {query!r}. Use one of: {', '.join(matches)}")
    return matches[0]


def _source_lookup(query: str) -> tuple[str, object]:
    """Resolve public declarations and members without loading unrelated backends."""
    import newton  # noqa: PLC0415

    modules = [newton, *(vars(newton)[name] for name in newton.__all__ if inspect.ismodule(vars(newton).get(name)))]
    names = {f"{module.__name__}.{name}" for module in modules for name in getattr(module, "__all__", ())}
    if _matching_names(query, names):
        name = _resolve_name(query, names)
        return name, pkgutil.resolve_name(name)
    candidates = {}
    for name in names:
        for alias in (name, name.removeprefix("newton."), name.rsplit(".", 1)[-1]):
            if not query.startswith(alias + "."):
                continue
            suffix = query[len(alias) + 1 :]
            if any(part.startswith("_") and part != "__init__" for part in suffix.split(".")):
                continue
            value = pkgutil.resolve_name(name)
            try:
                for part in suffix.split("."):
                    if inspect.ismodule(value) and part not in getattr(value, "__all__", ()):
                        raise AttributeError(part)
                    value = getattr(value, part)
            except AttributeError:
                continue
            candidates[f"{name}.{suffix}"] = value
    if not candidates and "." not in query:
        for module in modules:
            for exported in getattr(module, "__all__", ()):
                value = vars(module).get(exported)
                if inspect.isclass(value) and not query.startswith("_"):
                    try:
                        member = inspect.getattr_static(value, query)
                    except AttributeError:
                        continue
                    candidates[f"{module.__name__}.{exported}.{query}"] = member
    name = _resolve_name(query, candidates)
    return name, candidates[name]


def _object_lines(value, inventory, root, mode, *, resolved_name=None):
    identity = _identity(value) or _identity(type(value))
    names = sorted(
        name for name, entry in inventory.items() if identity is not None and entry.get("identity") == identity
    )
    name = resolved_name or (names[0] if len(names) == 1 else getattr(value, "__name__", type(value).__qualname__))
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = ""
    heading = f"{name}{signature}"
    if resolved_name is not None and isinstance(value, (str, int, float, bool)):
        heading += f" = {value!r}"
    lines = [heading, "", inspect.getdoc(value) or "No docstring available."]
    if names:
        lines.extend(["", "Public API reference:" if len(names) == 1 else "Matching public API references:"])
        lines.extend(f"{name}: {root / inventory[name]['path']}" for name in names)
    elif root is not None:
        lines.extend(["", f"Documentation directory: {root} ({mode})"])
    return lines


def info(object: object | str | None = None, *, output: TextIO | None = None) -> None:
    """Print documentation discovery information or help for a public object.

    Use ``info()`` to locate the release-matched guides and API reference.
    Pass an object to inspect its signature and docstring, or a public symbol
    name to locate its rendered API page. No browser or simulation is started.

    Args:
        object: Object to describe, or a fully qualified public name or unique
            short name. Omit to print the documentation location and examples.
        output: Text stream receiving the information. Defaults to stdout.

    Raises:
        ValueError: A symbol name is unknown or ambiguous.
        RuntimeError: Bundled documentation does not match the installed version.
    """
    stream = sys.stdout if output is None else output
    root, mode = _location()
    lines = []
    if object is None:
        lines = [f"Newton {__version__}", f"Documentation: {mode}"]
        if root is not None:
            index = root / ("llms.txt" if mode == "bundled snapshot" else "index.rst")
            lines.extend([f"Directory: {root}", f"Index: {index}"])
        version = __version__ if re.fullmatch(r"\d+\.\d+\.\d+", __version__) else "latest"
        lines.extend(
            [
                f"Online: {_WEBSITE}{version}/",
                "",
                "Object help: newton.info(newton.ModelBuilder)",
                'Symbol lookup: newton.info("ModelBuilder")',
                "Documentation directory: python -m newton.docs --path",
            ]
        )
    else:
        inventory = {}
        if root is not None and (root / "symbols.json").is_file():
            inventory = json.loads((root / "symbols.json").read_text(encoding="utf-8"))
        if isinstance(object, str):
            if inventory and _matching_names(object, inventory):
                name = _resolve_name(object, inventory)
                lines = [name, f"API reference: {root / inventory[name]['path']}"]
            else:
                name, value = _source_lookup(object)
                lines = _object_lines(value, inventory, root, mode, resolved_name=name)
        else:
            lines = _object_lines(object, inventory, root, mode)
    stream.write("\n".join(lines) + "\n")
