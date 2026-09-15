# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

DIRECTIVE_PREFIX = "# remove-when-minimum:"
ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
ALLOWLIST_DIRECTORY = ROOT / "scripts" / "ci" / "deprecation_allowlists"


class AllowlistEntry(NamedTuple):
    path: Path
    line_number: int
    message: str
    removal_requirement: Requirement


def _parse_removal_requirement(
    value: str,
    path: Path,
    line_number: int,
    errors: list[str],
) -> Requirement | None:
    try:
        requirement = Requirement(value)
    except InvalidRequirement as error:
        errors.append(f"{path}:{line_number}: invalid removal requirement: {error}")
        return None

    specifiers = list(requirement.specifier)
    if (
        requirement.url is not None
        or requirement.marker is not None
        or requirement.extras
        or len(specifiers) != 1
        or specifiers[0].operator != ">="
    ):
        errors.append(
            f"{path}:{line_number}: removal condition must be a plain requirement with exactly one >= specifier"
        )
        return None
    return requirement


def _parse_allowlist(path: Path) -> tuple[list[AllowlistEntry], list[str]]:
    entries: list[AllowlistEntry] = []
    errors: list[str] = []
    pending: tuple[int, Requirement | None] | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if line.startswith(DIRECTIVE_PREFIX):
            if pending is not None:
                errors.append(f"{path}:{pending[0]}: removal condition has no allowlist entry")
            value = line.removeprefix(DIRECTIVE_PREFIX).strip()
            pending = (line_number, _parse_removal_requirement(value, path, line_number, errors))
        elif not line or line.startswith("#"):
            continue
        elif pending is None:
            errors.append(f"{path}:{line_number}: allowlist entry has no removal condition")
        else:
            _, requirement = pending
            if requirement is not None:
                entries.append(AllowlistEntry(path, line_number, line, requirement))
            pending = None
    if pending is not None:
        errors.append(f"{path}:{pending[0]}: removal condition has no allowlist entry")
    return entries, errors


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


def check_allowlists(pyproject_path: Path, allowlist_paths: Sequence[Path]) -> list[str]:
    errors: list[str] = []
    requirements = _load_project_requirements(pyproject_path)
    by_name: dict[str, list[Requirement]] = {}
    for requirement in requirements:
        by_name.setdefault(canonicalize_name(requirement.name), []).append(requirement)

    entries: list[AllowlistEntry] = []
    for path in allowlist_paths:
        parsed, parse_errors = _parse_allowlist(path)
        entries.extend(parsed)
        errors.extend(parse_errors)

    for entry in entries:
        name = canonicalize_name(entry.removal_requirement.name)
        matches = by_name.get(name, [])
        if not matches:
            errors.append(
                f"{entry.path}:{entry.line_number}: {entry.removal_requirement.name} is not a direct project dependency"
            )
            continue
        if len(matches) != 1:
            errors.append(
                f"{entry.path}:{entry.line_number}: {entry.removal_requirement.name} "
                "has multiple direct project requirements"
            )
            continue
        try:
            declared_minimum = _declared_minimum(matches[0])
        except ValueError as error:
            errors.append(f"{entry.path}:{entry.line_number}: {matches[0]} {error}")
            continue
        threshold_specifier = next(iter(entry.removal_requirement.specifier))
        threshold = Version(threshold_specifier.version)
        if declared_minimum >= threshold:
            errors.append(
                f"{entry.path}:{entry.line_number}: {entry.message!r} is obsolete; "
                f"project requirement {matches[0]} satisfies removal condition "
                f"{entry.removal_requirement}; remove this allowlist entry"
            )
    return errors


def main() -> int:
    allowlist_paths = sorted(ALLOWLIST_DIRECTORY.glob("*.txt"))
    errors = check_allowlists(PYPROJECT, allowlist_paths)
    for error in errors:
        print(error, file=sys.stderr)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
