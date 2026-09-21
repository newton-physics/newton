# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Sequence
from pathlib import Path

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

DIRECTIVE_PREFIX = "# remove-when-minimum:"
ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
ALLOWLIST_DIRECTORY = ROOT / "scripts" / "ci" / "deprecation_allowlists"


def _parse_removal_requirement(value: str) -> Requirement:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        raise ValueError(f"invalid removal requirement: {error}") from error

    specifiers = list(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.marker is not None
        or requirement.extras
        or len(specifiers) != 1
        or specifiers[0].operator != ">="
    ):
        raise ValueError("removal condition must be a plain requirement with exactly one >= specifier")
    return requirement


def _load_project_requirements(path: Path) -> list[Requirement]:
    with path.open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)
    return [Requirement(value) for value in project["project"].get("dependencies", [])]


def _declared_minimum(requirement: Requirement) -> Version:
    if requirement.url is not None:
        raise ValueError("uses a direct URL")
    if requirement.marker is not None:
        raise ValueError("uses an environment marker")
    specifiers = list(requirement.specifier)
    lower_bounds = [specifier for specifier in specifiers if specifier.operator == ">="]
    unsupported = [specifier for specifier in specifiers if specifier.operator not in {">=", "<", "<=", "!="}]
    if len(lower_bounds) != 1 or unsupported:
        raise ValueError("does not declare exactly one >= lower bound")
    return Version(lower_bounds[0].version)


def check_allowlists(pyproject_path: Path, allowlist_paths: Sequence[Path]) -> str | None:
    requirements = _load_project_requirements(pyproject_path)
    for path in allowlist_paths:
        pending: tuple[int, Requirement] | None = None
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if line.startswith(DIRECTIVE_PREFIX):
                if pending is not None:
                    return f"{path}:{pending[0]}: removal condition has no allowlist entry"
                try:
                    condition = _parse_removal_requirement(line.removeprefix(DIRECTIVE_PREFIX).strip())
                except ValueError as error:
                    return f"{path}:{line_number}: {error}"
                pending = (line_number, condition)
                continue
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                return f"{path}:{line_number}: entry contains ':'; use a unique message prefix ending before the colon"
            if pending is None:
                return f"{path}:{line_number}: allowlist entry has no removal condition"

            _, condition = pending
            name = canonicalize_name(condition.name)
            matches = [requirement for requirement in requirements if canonicalize_name(requirement.name) == name]
            if not matches:
                return f"{path}:{line_number}: {condition.name} is not a direct project dependency"
            if len(matches) != 1:
                return f"{path}:{line_number}: {condition.name} has multiple direct project requirements"
            try:
                declared_minimum = _declared_minimum(matches[0])
            except ValueError as error:
                return f"{path}:{line_number}: {matches[0]} {error}"
            threshold = Version(next(iter(condition.specifier)).version)
            if declared_minimum >= threshold:
                return (
                    f"{path}:{line_number}: {line!r} is obsolete; project requirement "
                    f"{matches[0]} satisfies removal condition {condition}; remove this allowlist entry"
                )
            pending = None
        if pending is not None:
            return f"{path}:{pending[0]}: removal condition has no allowlist entry"
    return None


def main() -> int:
    allowlist_paths = sorted(ALLOWLIST_DIRECTORY.glob("*.txt"))
    error = check_allowlists(PYPROJECT, allowlist_paths)
    if error is not None:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
