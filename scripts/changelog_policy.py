# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Enforce Newton's repository policy around Towncrier fragments."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

CHANGELOG_PATH = PurePosixPath("CHANGELOG.md")
BOOTSTRAP_POLICY_PATH = PurePosixPath("scripts/changelog_policy.py")
FRAGMENT_DIRECTORY = PurePosixPath("changelog.d")
METADATA_FILES = {PurePosixPath("changelog.d/README.md")}
RELEASE_MANAGEMENT_LABEL = "release-management"
CATEGORIES = ("added", "changed", "deprecated", "removed", "fixed")

_IDENTIFIER_PATTERN = r"(?:\d+|\+[a-z0-9]+(?:-[a-z0-9]+)*)"
_TYPED_PATTERN = re.compile(
    rf"^(?P<identifier>{_IDENTIFIER_PATTERN})"
    rf"\.(?P<category>{'|'.join(CATEGORIES)})"
    r"(?P<counter>\.[1-9]\d*)?\.md$"
)
_SKIP_PATTERN = re.compile(rf"^(?P<identifier>{_IDENTIFIER_PATTERN})\.skip$")


@dataclass(frozen=True)
class Fragment:
    """Describe a parsed changelog fragment."""

    path: Path
    identifier: str
    category: str | None
    counter: int | None
    is_skip: bool


@dataclass(frozen=True)
class Change:
    """Describe one path change from ``git diff --name-status``."""

    status: str
    path: PurePosixPath
    old_path: PurePosixPath | None = None


def parse_fragment(path: Path) -> Fragment | None:
    """Parse a fragment filename, returning ``None`` for an invalid name."""
    typed_match = _TYPED_PATTERN.fullmatch(path.name)
    if typed_match:
        counter_text = typed_match.group("counter")
        return Fragment(
            path=path,
            identifier=typed_match.group("identifier"),
            category=typed_match.group("category"),
            counter=int(counter_text[1:]) if counter_text else None,
            is_skip=False,
        )

    skip_match = _SKIP_PATTERN.fullmatch(path.name)
    if skip_match:
        return Fragment(
            path=path,
            identifier=skip_match.group("identifier"),
            category=None,
            counter=None,
            is_skip=True,
        )

    return None


def validate_fragment(fragment: Fragment) -> list[str]:
    """Validate one fragment's content."""
    try:
        content = fragment.path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return [f"{fragment.path}: cannot read UTF-8 content: {error}"]

    text = content.strip()
    if not text:
        return [f"{fragment.path}: fragment must not be empty"]

    if fragment.is_skip:
        if len(text.splitlines()) != 1:
            return [f"{fragment.path}: .skip fragments must contain a one-line reason"]
        return []

    errors = []
    if text.lstrip().startswith(("- ", "* ")):
        errors.append(f"{fragment.path}: omit the leading bullet; Towncrier adds it")
    if not text.endswith("."):
        errors.append(f"{fragment.path}: entry must end with a period")
    return errors


def validate_directory(directory: Path) -> list[str]:
    """Validate every pending fragment in a directory."""
    if not directory.is_dir():
        return [f"{directory}: fragment directory does not exist"]

    errors: list[str] = []
    fragments: list[Fragment] = []
    for path in sorted(directory.iterdir()):
        relative_path = PurePosixPath(FRAGMENT_DIRECTORY, path.name)
        if relative_path in METADATA_FILES:
            continue
        if not path.is_file():
            errors.append(f"{path}: only fragment files are allowed")
            continue

        fragment = parse_fragment(path)
        if fragment is None:
            errors.append(f"{path}: use ISSUE.CATEGORY.md, +SLUG-RANDOM.CATEGORY.md, or the corresponding .skip name")
            continue
        fragments.append(fragment)
        errors.extend(validate_fragment(fragment))

    identifiers_with_skip = {fragment.identifier for fragment in fragments if fragment.is_skip}
    for identifier in sorted(identifiers_with_skip):
        if any(fragment.identifier == identifier and not fragment.is_skip for fragment in fragments):
            errors.append(f"{directory}: {identifier} cannot mix .skip and user-facing fragments")

    return errors


def _parse_git_changes(output: str) -> list[Change]:
    """Parse the relevant fields from ``git diff --name-status``."""
    changes = []
    for line in output.splitlines():
        fields = line.split("\t")
        status = fields[0][0]
        if status in {"R", "C"}:
            changes.append(
                Change(
                    status=status,
                    old_path=PurePosixPath(fields[1]),
                    path=PurePosixPath(fields[2]),
                )
            )
        else:
            changes.append(Change(status=status, path=PurePosixPath(fields[1])))
    return changes


def get_git_changes(base_ref: str) -> list[Change]:
    """Return changes between a merge base and ``HEAD``."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", f"{base_ref}...HEAD", "--"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return _parse_git_changes(result.stdout)


def _is_fragment_path(path: PurePosixPath) -> bool:
    """Return whether a path is a fragment rather than fragment metadata."""
    return path.parent == FRAGMENT_DIRECTORY and path not in METADATA_FILES


def _fragment_from_change(change: Change) -> Fragment | None:
    """Parse the current fragment path from a Git change."""
    if not _is_fragment_path(change.path):
        return None
    return parse_fragment(Path(change.path))


def validate_pr_changes(
    changes: list[Change],
    *,
    target_branch: str,
    labels: set[str],
) -> list[str]:
    """Validate fragment changes for a pull request."""
    if RELEASE_MANAGEMENT_LABEL in labels:
        return _validate_release_management_changes(changes)

    errors: list[str] = []
    is_bootstrap = any(change.status == "A" and change.path == BOOTSTRAP_POLICY_PATH for change in changes)
    if not is_bootstrap and any(
        change.path == CHANGELOG_PATH or change.old_path == CHANGELOG_PATH for change in changes
    ):
        errors.append("Normal pull requests must not edit CHANGELOG.md; add a fragment under changelog.d/")

    fragment_changes = [
        change
        for change in changes
        if _is_fragment_path(change.path) or (change.old_path is not None and _is_fragment_path(change.old_path))
    ]
    if not fragment_changes:
        errors.append("Add a changelog.d fragment or a one-line .skip reason")
        return errors

    fragments: list[Fragment] = []
    for change in fragment_changes:
        if change.status != "A":
            errors.append(f"{change.path}: normal pull requests may only add fragments")
            continue
        fragment = _fragment_from_change(change)
        if fragment is None:
            errors.append(f"{change.path}: invalid fragment filename")
            continue
        fragments.append(fragment)

    identifiers = {fragment.identifier for fragment in fragments}
    target_name = target_branch.removeprefix("refs/heads/")
    skip_fragments = [fragment for fragment in fragments if fragment.is_skip]
    if target_name == "main":
        if len(identifiers) != 1:
            errors.append("Pull requests to main must add exactly one logical fragment identifier")
        if skip_fragments and len(fragments) != 1:
            errors.append(".skip must be the pull request's only fragment")

    return errors


def _validate_release_management_changes(changes: list[Change]) -> list[str]:
    """Validate a Towncrier build synchronization pull request."""
    errors: list[str] = []
    changelog_changes = [change for change in changes if change.path == CHANGELOG_PATH]
    fragment_changes = [change for change in changes if _is_fragment_path(change.path)]

    allowed_paths = {CHANGELOG_PATH, *(change.path for change in fragment_changes)}
    unexpected_paths = sorted(str(change.path) for change in changes if change.path not in allowed_paths)
    if unexpected_paths:
        errors.append(
            "release-management pull requests must be changelog-only; unexpected paths: " + ", ".join(unexpected_paths)
        )
    if len(changelog_changes) != 1 or changelog_changes[0].status != "M":
        errors.append("release-management pull requests must modify CHANGELOG.md exactly once")
    if not fragment_changes:
        errors.append("release-management pull requests must delete shipped fragments")
    for change in fragment_changes:
        if change.status != "D":
            errors.append(f"{change.path}: release-management pull requests may only delete fragments")

    return errors


def _print_errors(errors: list[str]) -> int:
    """Print policy errors and return a process exit code."""
    if not errors:
        print("Changelog policy checks passed.")
        return 0
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the changelog policy command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate pending fragments")
    validate_parser.add_argument("--directory", type=Path, default=Path(FRAGMENT_DIRECTORY))

    check_parser = subparsers.add_parser("check-pr", help="validate a pull request diff")
    check_parser.add_argument("--base-ref", required=True)
    check_parser.add_argument("--target-branch", required=True)
    check_parser.add_argument("--labels", default="")

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _print_errors(validate_directory(args.directory))

    labels = {label for label in args.labels.split(",") if label}
    changes = get_git_changes(args.base_ref)
    errors = validate_directory(Path(FRAGMENT_DIRECTORY))
    errors.extend(
        validate_pr_changes(
            changes,
            target_branch=args.target_branch,
            labels=labels,
        )
    )
    return _print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
