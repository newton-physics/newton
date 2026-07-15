# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that checks asv.conf.json pins against pyproject.toml.

The exact pins in the ASV ``install_command`` run after the newton wheel is
installed, so they win over the project's own requirements. ``pip check``
cannot catch a pin that violates an extra-gated requirement (e.g. mujoco-warp
via the ``sim`` extra), so this hook validates the pins statically.
"""

import json
import re
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

_PIN_RE = re.compile(r"(?:^|\s)([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!*]+)")


def _collect_requirements(pyproject: dict) -> dict[str, list[tuple[str, Requirement]]]:
    """Map canonical package name -> [(origin, requirement), ...] from base deps and all extras."""
    project = pyproject["project"]
    sources = [("dependencies", dep) for dep in project.get("dependencies", [])]
    for extra, deps in project.get("optional-dependencies", {}).items():
        sources.extend((f"extra '{extra}'", dep) for dep in deps)

    requirements: dict[str, list[tuple[str, Requirement]]] = {}
    for origin, dep in sources:
        req = Requirement(dep)
        name = canonicalize_name(req.name)
        if name == "newton":  # self-referencing extras carry no version constraints
            continue
        requirements.setdefault(name, []).append((origin, req))
    return requirements


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    asv_conf = json.loads((root / "asv.conf.json").read_text())
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    requirements = _collect_requirements(pyproject)

    errors = []
    for command in asv_conf.get("install_command", []):
        for name, version in _PIN_RE.findall(command):
            for origin, req in requirements.get(canonicalize_name(name), []):
                if not req.specifier.contains(Version(version), prereleases=True):
                    errors.append(f"asv.conf.json pins {name}=={version}, but {origin} requires '{req}'")

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
