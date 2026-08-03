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

import tomllib

CHANGELOG_PATH = PurePosixPath("CHANGELOG.md")
BOOTSTRAP_POLICY_PATH = PurePosixPath("scripts/changelog_policy.py")
FRAGMENT_DIRECTORY = PurePosixPath("changelog")
METADATA_FILES = {PurePosixPath("changelog/README.md")}
RELEASE_MANAGEMENT_LABEL = "release-management"

_IDENTIFIER_PATTERN = r"(?:\d+|\+[a-z0-9]+(?:-[a-z0-9]+)*)"
_SKIP_PATTERN = re.compile(rf"^(?P<identifier>{_IDENTIFIER_PATTERN})\.skip$")


@dataclass(frozen=True)
class Fragment:
    """Describe the Newton-specific properties of a changelog fragment."""

    path: Path
    identifier: str | None
    is_skip: bool


@dataclass(frozen=True)
class Change:
    """Describe one path change from ``git diff --name-status``."""

    status: str
    path: PurePosixPath
    old_path: PurePosixPath | None = None


def load_towncrier_fragment_types(config_path: Path = Path("pyproject.toml")) -> tuple[str, ...]:
    """Load the configured Towncrier fragment types in rendering order."""
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)
    return tuple(fragment_type["directory"] for fragment_type in config["tool"]["towncrier"]["type"])


def describe_fragment(path: Path, fragment_types: tuple[str, ...]) -> Fragment | None:
    """Extract properties needed by Newton policy without validating Towncrier grammar."""
    skip_match = _SKIP_PATTERN.fullmatch(path.name)
    if skip_match:
        return Fragment(
            path=path,
            identifier=skip_match.group("identifier"),
            is_skip=True,
        )

    if path.suffix == ".skip":
        return None

    parts = path.name.split(".")
    for index in reversed(range(1, len(parts))):
        if parts[index] in fragment_types:
            identifier = ".".join(parts[:index]).strip()
            if identifier.isdigit():
                identifier = str(int(identifier))
            return Fragment(path=path, identifier=identifier, is_skip=False)

    # Towncrier reports invalid renderable filenames when ``ignore`` is configured.
    return Fragment(path=path, identifier=None, is_skip=False)


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


def validate_directory(directory: Path, *, fragment_types: tuple[str, ...] | None = None) -> list[str]:
    """Validate every pending fragment in a directory."""
    if not directory.is_dir():
        return [f"{directory}: fragment directory does not exist"]

    if fragment_types is None:
        try:
            fragment_types = load_towncrier_fragment_types()
        except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
            return [f"pyproject.toml: cannot load Towncrier fragment types: {error}"]

    errors: list[str] = []
    fragments: list[Fragment] = []
    for path in sorted(directory.iterdir()):
        relative_path = PurePosixPath(FRAGMENT_DIRECTORY, path.name)
        if relative_path in METADATA_FILES:
            continue
        if not path.is_file():
            errors.append(f"{path}: only fragment files are allowed")
            continue

        fragment = describe_fragment(path, fragment_types)
        if fragment is None:
            errors.append(f"{path}: use ISSUE.skip or +SLUG-RANDOM.skip")
            continue
        fragments.append(fragment)
        errors.extend(validate_fragment(fragment))

    identifiers_with_skip = {
        fragment.identifier for fragment in fragments if fragment.is_skip and fragment.identifier is not None
    }
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


def _get_git_changes(arguments: list[str]) -> list[Change]:
    """Return parsed changes from a ``git diff`` invocation."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", *arguments, "--"],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    return _parse_git_changes(result.stdout)


def get_git_changes(base_ref: str) -> list[Change]:
    """Return changes between a merge base and ``HEAD``."""
    return _get_git_changes([f"{base_ref}...HEAD"])


def get_staged_changes() -> list[Change]:
    """Return changes staged in the Git index."""
    return _get_git_changes(["--cached"])


def _is_fragment_path(path: PurePosixPath) -> bool:
    """Return whether a path is a fragment rather than fragment metadata."""
    return path.parent == FRAGMENT_DIRECTORY and path not in METADATA_FILES


def _fragment_from_change(change: Change, fragment_types: tuple[str, ...]) -> Fragment | None:
    """Describe the current fragment path from a Git change."""
    if not _is_fragment_path(change.path):
        return None
    return describe_fragment(Path(change.path), fragment_types)


def pending_fragment_paths(directory: Path) -> set[PurePosixPath]:
    """Return pending fragment paths, excluding repository metadata."""
    if not directory.is_dir():
        return set()
    return {
        PurePosixPath(FRAGMENT_DIRECTORY, path.name)
        for path in directory.iterdir()
        if path.is_file() and PurePosixPath(FRAGMENT_DIRECTORY, path.name) not in METADATA_FILES
    }


def validate_pr_changes(
    changes: list[Change],
    *,
    target_branch: str,
    labels: set[str],
    fragment_types: tuple[str, ...] | None = None,
    pending_fragments: set[PurePosixPath] | None = None,
) -> list[str]:
    """Validate fragment changes for a pull request."""
    if fragment_types is None:
        fragment_types = load_towncrier_fragment_types()
    if pending_fragments is None:
        pending_fragments = set()
    if RELEASE_MANAGEMENT_LABEL in labels:
        return _validate_release_management_changes(
            changes,
            target_branch=target_branch,
            pending_fragments=pending_fragments,
        )

    errors: list[str] = []
    is_bootstrap = any(change.status == "A" and change.path == BOOTSTRAP_POLICY_PATH for change in changes)
    if not is_bootstrap and any(
        change.path == CHANGELOG_PATH or change.old_path == CHANGELOG_PATH for change in changes
    ):
        errors.append("Normal pull requests must not edit CHANGELOG.md; add a fragment under changelog/")

    fragment_changes = [
        change
        for change in changes
        if _is_fragment_path(change.path) or (change.old_path is not None and _is_fragment_path(change.old_path))
    ]
    if not fragment_changes:
        errors.append("Add a changelog fragment or a one-line .skip reason")
        return errors

    fragments: list[Fragment] = []
    for change in fragment_changes:
        if change.status != "A":
            errors.append(f"{change.path}: normal pull requests may only add fragments")
            continue
        fragment = _fragment_from_change(change, fragment_types)
        if fragment is None:
            errors.append(f"{change.path}: invalid .skip fragment filename")
            continue
        fragments.append(fragment)

    identifiers = {fragment.identifier for fragment in fragments if fragment.identifier is not None}
    target_name = target_branch.removeprefix("refs/heads/")
    skip_fragments = [fragment for fragment in fragments if fragment.is_skip]
    if target_name == "main":
        if len(identifiers) != 1:
            errors.append("Pull requests to main must add exactly one logical fragment identifier")
        if skip_fragments and len(fragments) != 1:
            errors.append(".skip must be the pull request's only fragment")

    return errors


def _validate_release_management_changes(
    changes: list[Change],
    *,
    target_branch: str,
    pending_fragments: set[PurePosixPath],
) -> list[str]:
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

    target_name = target_branch.removeprefix("refs/heads/")
    if target_name != "main" and pending_fragments:
        errors.append(
            "release-management pull requests to a release branch must consume all fragments; remaining paths: "
            + ", ".join(str(path) for path in sorted(pending_fragments))
        )

    return errors


def validate_staged_changes(
    changes: list[Change],
    *,
    fragment_types: tuple[str, ...],
    pending_fragments: set[PurePosixPath],
) -> list[str]:
    """Validate staged changes without requiring pull-request metadata."""
    if not changes:
        return []

    has_changelog_change = any(change.path == CHANGELOG_PATH for change in changes)
    has_fragment_change = any(
        _is_fragment_path(change.path) or (change.old_path is not None and _is_fragment_path(change.old_path))
        for change in changes
    )
    if not has_changelog_change and not has_fragment_change:
        return []

    other_changes = [change for change in changes if change.path != CHANGELOG_PATH]
    is_release_management_shape = has_changelog_change and all(
        _is_fragment_path(change.path) and change.status == "D" for change in other_changes
    )
    if is_release_management_shape:
        return _validate_release_management_changes(
            changes,
            target_branch="main",
            pending_fragments=pending_fragments,
        )

    return validate_pr_changes(
        changes,
        target_branch="main",
        labels=set(),
        fragment_types=fragment_types,
        pending_fragments=pending_fragments,
    )


def validate_merge_group_changes(
    changes: list[Change],
    *,
    target_branch: str,
    fragment_types: tuple[str, ...],
    pending_fragments: set[PurePosixPath],
) -> list[str]:
    """Validate the composable changelog rules for a merge-group diff."""
    if any(change.path == CHANGELOG_PATH for change in changes):
        return _validate_release_management_changes(
            changes,
            target_branch=target_branch,
            pending_fragments=pending_fragments,
        )

    # A merge group can contain several PRs, so multiple logical identifiers are valid.
    return validate_pr_changes(
        changes,
        target_branch="merge-group",
        labels=set(),
        fragment_types=fragment_types,
        pending_fragments=pending_fragments,
    )


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

    merge_group_parser = subparsers.add_parser("check-merge-group", help="validate a merge-group diff")
    merge_group_parser.add_argument("--base-ref", required=True)
    merge_group_parser.add_argument("--target-branch", required=True)

    subparsers.add_parser("check-staged", help="validate staged changelog changes")

    args = parser.parse_args(argv)
    try:
        fragment_types = load_towncrier_fragment_types()
    except (KeyError, OSError, TypeError, tomllib.TOMLDecodeError) as error:
        return _print_errors([f"pyproject.toml: cannot load Towncrier fragment types: {error}"])

    if args.command == "validate":
        return _print_errors(validate_directory(args.directory, fragment_types=fragment_types))

    fragment_directory = Path(FRAGMENT_DIRECTORY)
    pending_fragments = pending_fragment_paths(fragment_directory)
    errors = validate_directory(fragment_directory, fragment_types=fragment_types)
    if args.command == "check-staged":
        errors.extend(
            validate_staged_changes(
                get_staged_changes(),
                fragment_types=fragment_types,
                pending_fragments=pending_fragments,
            )
        )
    elif args.command == "check-merge-group":
        errors.extend(
            validate_merge_group_changes(
                get_git_changes(args.base_ref),
                target_branch=args.target_branch,
                fragment_types=fragment_types,
                pending_fragments=pending_fragments,
            )
        )
    else:
        labels = {label for label in args.labels.split(",") if label}
        errors.extend(
            validate_pr_changes(
                get_git_changes(args.base_ref),
                target_branch=args.target_branch,
                labels=labels,
                fragment_types=fragment_types,
                pending_fragments=pending_fragments,
            )
        )
    return _print_errors(errors)


if __name__ == "__main__":
    raise SystemExit(main())
