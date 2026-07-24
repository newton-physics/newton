# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Create, validate, consolidate, and release Newton changelog fragments."""

from __future__ import annotations

import argparse
import re
import secrets
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAGMENTS_DIRECTORY = "changelog.d"
CHANGELOG_FILENAME = "CHANGELOG.md"
UNRELEASED_HEADING = "## [Unreleased]"
RELEASES_MARKER = "<!-- changelog releases start -->"
CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed")
IGNORED_FRAGMENT_FILES = {".gitkeep", "README.md"}
FRAGMENT_NAME_RE = re.compile(r"^(?P<slug>[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)-(?P<token>[0-9a-f]{8})\.(?P<kind>md|skip)$")
FRAGMENT_ID_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?-[0-9a-f]{8}$")
FRAGMENT_MARKER_RE = re.compile(r"^<!-- changelog-fragment: (?P<fragment_id>[^ ]+) -->$")
RELEASE_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
HEADING_RE = re.compile(r"^### (?P<category>[^\r\n]+)$")


class ChangelogError(ValueError):
    """Report an invalid changelog operation."""


@dataclass(frozen=True)
class ChangelogEntry:
    """Store one changelog bullet and its hidden fragment provenance."""

    text: str
    fragment_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fragment:
    """Store one validated changelog fragment."""

    path: Path
    sections: dict[str, tuple[ChangelogEntry, ...]]
    skip_reason: str | None = None

    @property
    def is_skip(self) -> bool:
        """Return whether the fragment explicitly skips release notes."""
        return self.skip_reason is not None


@dataclass(frozen=True)
class GitChange:
    """Store one name-status record from a Git diff."""

    status: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class ChangelogLayout:
    """Store the stable and pending portions of the changelog."""

    prefix: str
    unreleased: dict[str, tuple[ChangelogEntry, ...]]
    suffix: str


@dataclass(frozen=True)
class ReleaseSection:
    """Store one dated release section parsed from a changelog."""

    text: str
    sections: dict[str, tuple[ChangelogEntry, ...]]


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with repository-standard LF newlines."""
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _read_text(path: Path) -> str:
    """Read UTF-8 text and normalize newlines."""
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ChangelogError("The fragment name must contain at least one letter or digit.")
    return slug[:64].rstrip("-")


def _run_git_process(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise ChangelogError(f"git {' '.join(args)} failed: {detail}") from exc
    return result.stdout.replace("\r\n", "\n")


def _run_git(repo_root: Path, *args: str) -> str:
    return _run_git_process(repo_root, *args).strip()


def _git_show_file(repo_root: Path, ref: str, path: str) -> str:
    return _run_git_process(repo_root, "show", f"{ref}:{path}")


def _default_fragment_name(repo_root: Path) -> str:
    branch = _run_git(repo_root, "branch", "--show-current")
    if not branch:
        raise ChangelogError("Detached HEAD has no usable fragment name; pass an explicit name to 'create'.")
    return branch.rsplit("/", maxsplit=1)[-1]


def create_fragment(
    repo_root: Path,
    *,
    name: str | None,
    category: str | None,
    content: str | None,
    skip_reason: str | None,
) -> Path:
    """Create a uniquely named changelog fragment."""
    if skip_reason is None and (category is None or content is None):
        raise ChangelogError("A release-note fragment requires both --category and --content.")
    if skip_reason is not None and (category is not None or content is not None):
        raise ChangelogError("Use either --skip or --category/--content, not both.")

    source_name = name or _default_fragment_name(repo_root)
    slug = _slugify(source_name)
    fragments_dir = repo_root / FRAGMENTS_DIRECTORY
    fragments_dir.mkdir(exist_ok=True)

    for _ in range(100):
        token = secrets.token_hex(4)
        suffix = "skip" if skip_reason is not None else "md"
        path = fragments_dir / f"{slug}-{token}.{suffix}"
        if path.exists():
            continue

        if skip_reason is not None:
            reason = skip_reason.strip()
            if not reason or "\n" in reason or "\r" in reason:
                raise ChangelogError("A skip fragment requires a non-empty, single-line reason.")
            text = f"{reason}\n"
        else:
            assert category is not None
            assert content is not None
            normalized_category = category.capitalize()
            if normalized_category not in CATEGORIES:
                raise ChangelogError(f"Unknown category {category!r}; choose from {', '.join(CATEGORIES)}.")
            entry = content.strip()
            if not entry:
                raise ChangelogError("Fragment content must not be empty.")
            entry_lines = entry.splitlines()
            text = f"### {normalized_category}\n\n- {entry_lines[0]}\n"
            for line in entry_lines[1:]:
                text += f"  {line}\n"

        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        return path

    raise ChangelogError("Could not generate a unique fragment filename after 100 attempts.")


def parse_fragment_text(
    text: str,
    *,
    source: str = "fragment",
    allow_fragment_markers: bool = False,
) -> dict[str, tuple[ChangelogEntry, ...]]:
    """Parse strict categorized Markdown into changelog entries."""
    if not text.strip():
        raise ChangelogError(f"{source}: fragment is empty.")

    sections: dict[str, tuple[ChangelogEntry, ...]] = {}
    current_category: str | None = None
    current_entries: list[ChangelogEntry] = []
    current_entry_lines: list[str] = []
    current_entry_ids: list[str] = []
    pending_fragment_ids: list[str] = []
    last_category_index = -1

    def flush_entry() -> None:
        nonlocal current_entry_lines, current_entry_ids
        while current_entry_lines and not current_entry_lines[-1]:
            current_entry_lines.pop()
        if current_entry_lines:
            current_entries.append(
                ChangelogEntry(text="\n".join(current_entry_lines), fragment_ids=tuple(current_entry_ids))
            )
        current_entry_lines = []
        current_entry_ids = []

    def flush_section() -> None:
        nonlocal current_entries
        flush_entry()
        if pending_fragment_ids:
            raise ChangelogError(f"{source}: fragment provenance marker has no following bullet entry.")
        if current_category is not None:
            if not current_entries:
                raise ChangelogError(f"{source}: heading {current_category!r} has no bullet entries.")
            sections[current_category] = tuple(current_entries)
        current_entries = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        heading = HEADING_RE.fullmatch(line)
        if heading:
            flush_section()
            category = heading.group("category")
            if category not in CATEGORIES:
                raise ChangelogError(
                    f"{source}:{line_number}: unknown heading {category!r}; choose from {', '.join(CATEGORIES)}."
                )
            if category in sections:
                raise ChangelogError(f"{source}:{line_number}: duplicate heading {category!r}.")
            category_index = CATEGORIES.index(category)
            if category_index <= last_category_index:
                raise ChangelogError(f"{source}:{line_number}: headings must follow {', '.join(CATEGORIES)} order.")
            current_category = category
            last_category_index = category_index
            continue

        fragment_marker = FRAGMENT_MARKER_RE.fullmatch(line)
        if fragment_marker:
            if not allow_fragment_markers:
                raise ChangelogError(f"{source}:{line_number}: fragment provenance markers are generated by the tool.")
            if current_category is None:
                raise ChangelogError(f"{source}:{line_number}: fragment marker must follow an allowed heading.")
            flush_entry()
            fragment_id = fragment_marker.group("fragment_id")
            if FRAGMENT_ID_RE.fullmatch(fragment_id) is None:
                raise ChangelogError(f"{source}:{line_number}: invalid fragment provenance ID {fragment_id!r}.")
            if fragment_id in pending_fragment_ids:
                raise ChangelogError(f"{source}:{line_number}: duplicate fragment provenance ID {fragment_id!r}.")
            pending_fragment_ids.append(fragment_id)
            continue

        if not line:
            if current_entry_lines:
                current_entry_lines.append("")
            continue
        if current_category is None:
            raise ChangelogError(f"{source}:{line_number}: content must follow an allowed '###' heading.")
        if line.startswith("- "):
            flush_entry()
            if not line[2:].strip():
                raise ChangelogError(f"{source}:{line_number}: bullet entry is empty.")
            current_entry_lines = [line]
            current_entry_ids = pending_fragment_ids
            pending_fragment_ids = []
            continue
        if line.startswith(("  ", "\t")):
            if not current_entry_lines:
                raise ChangelogError(f"{source}:{line_number}: continuation text has no preceding bullet.")
            current_entry_lines.append(line)
            continue
        raise ChangelogError(
            f"{source}:{line_number}: entries must start with '- ' and continuation lines must be indented."
        )

    flush_section()
    if not sections:
        raise ChangelogError(f"{source}: fragment has no allowed headings.")
    return sections


def _validate_fragment_name(path: Path) -> re.Match[str]:
    match = FRAGMENT_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ChangelogError(f"{path}: invalid filename; expected <descriptive-slug>-<8 lowercase hex>.(md|skip).")
    return match


def load_fragments(repo_root: Path) -> list[Fragment]:
    """Load and validate every pending fragment in a repository."""
    fragments_dir = repo_root / FRAGMENTS_DIRECTORY
    if not fragments_dir.is_dir():
        raise ChangelogError(f"Missing fragment directory: {fragments_dir}")

    fragments: list[Fragment] = []
    kinds_by_stem: dict[str, str] = {}
    for path in sorted(fragments_dir.iterdir(), key=lambda candidate: candidate.name):
        if path.name in IGNORED_FRAGMENT_FILES:
            continue
        if not path.is_file():
            raise ChangelogError(f"{path}: fragment directory must not contain subdirectories.")
        match = _validate_fragment_name(path)
        fragment_id = path.name.rsplit(".", maxsplit=1)[0]
        kind = match.group("kind")
        previous_kind = kinds_by_stem.get(fragment_id)
        if previous_kind is not None:
            raise ChangelogError(f"{path}: cannot combine .{previous_kind} and .{kind} for the same fragment ID.")
        kinds_by_stem[fragment_id] = kind

        text = _read_text(path)
        if kind == "skip":
            lines = text.splitlines()
            if len(lines) != 1 or not lines[0].strip():
                raise ChangelogError(f"{path}: skip fragments require a non-empty, single-line reason.")
            fragments.append(Fragment(path=path, sections={}, skip_reason=lines[0].strip()))
        else:
            parsed = parse_fragment_text(text, source=str(path))
            sections = {
                category: tuple(ChangelogEntry(text=entry.text, fragment_ids=(fragment_id,)) for entry in entries)
                for category, entries in parsed.items()
            }
            fragments.append(Fragment(path=path, sections=sections))
    return fragments


def _standalone_match(text: str, value: str) -> re.Match[str]:
    matches = list(re.finditer(rf"(?m)^{re.escape(value)}$", text))
    if text.count(value) != 1 or len(matches) != 1:
        raise ChangelogError(f"{CHANGELOG_FILENAME}: expected exactly one standalone {value!r} line.")
    return matches[0]


def parse_changelog_layout(text: str, *, source: str = CHANGELOG_FILENAME) -> ChangelogLayout:
    """Parse the Unreleased accumulator and stable release boundary."""
    text = text.replace("\r\n", "\n")
    try:
        unreleased_match = _standalone_match(text, UNRELEASED_HEADING)
        marker_match = _standalone_match(text, RELEASES_MARKER)
    except ChangelogError as exc:
        raise ChangelogError(str(exc).replace(CHANGELOG_FILENAME, source)) from exc
    if unreleased_match.end() >= marker_match.start():
        raise ChangelogError(f"{source}: {UNRELEASED_HEADING!r} must appear before the release marker.")

    body = text[unreleased_match.end() : marker_match.start()].strip()
    unreleased = parse_fragment_text(body, source=f"{source} [Unreleased]", allow_fragment_markers=True) if body else {}
    return ChangelogLayout(
        prefix=text[: unreleased_match.end()],
        unreleased=unreleased,
        suffix=text[marker_match.start() :],
    )


def validate_changelog(repo_root: Path) -> ChangelogLayout:
    """Validate and parse the changelog accumulator."""
    path = repo_root / CHANGELOG_FILENAME
    if not path.is_file():
        raise ChangelogError(f"Missing changelog: {path}")
    return parse_changelog_layout(_read_text(path), source=str(path))


def validate_repository(repo_root: Path) -> tuple[ChangelogLayout, list[Fragment]]:
    """Validate the changelog and all pending fragments."""
    return validate_changelog(repo_root), load_fragments(repo_root)


def _merge_sections(
    *section_groups: dict[str, tuple[ChangelogEntry, ...]],
) -> dict[str, tuple[ChangelogEntry, ...]]:
    merged: dict[str, list[ChangelogEntry]] = {category: [] for category in CATEGORIES}
    positions: dict[tuple[str, str], int] = {}
    for sections in section_groups:
        for category in CATEGORIES:
            for entry in sections.get(category, ()):
                key = (category, entry.text)
                position = positions.get(key)
                if position is None:
                    positions[key] = len(merged[category])
                    merged[category].append(entry)
                    continue
                existing = merged[category][position]
                fragment_ids = tuple(dict.fromkeys((*existing.fragment_ids, *entry.fragment_ids)))
                merged[category][position] = ChangelogEntry(text=existing.text, fragment_ids=fragment_ids)
    return {category: tuple(entries) for category, entries in merged.items() if entries}


def _fragment_sections(fragments: list[Fragment]) -> dict[str, tuple[ChangelogEntry, ...]]:
    return _merge_sections(*(fragment.sections for fragment in fragments if not fragment.is_skip))


def render_sections(
    sections: dict[str, tuple[ChangelogEntry, ...]],
    *,
    include_provenance: bool = True,
) -> str:
    """Render categorized entries, optionally with invisible provenance."""
    lines: list[str] = []
    for category in CATEGORIES:
        entries = sections.get(category, ())
        if not entries:
            continue
        lines.extend((f"### {category}", ""))
        for entry in entries:
            if include_provenance:
                for fragment_id in entry.fragment_ids:
                    lines.append(f"<!-- changelog-fragment: {fragment_id} -->")
            lines.extend(entry.text.splitlines())
        lines.append("")
    return "\n".join(lines).rstrip()


def render_unreleased(sections: dict[str, tuple[ChangelogEntry, ...]]) -> str:
    """Render the complete Unreleased accumulator."""
    body = render_sections(sections)
    return f"{UNRELEASED_HEADING}\n\n{body}\n" if body else f"{UNRELEASED_HEADING}\n"


def render_release(sections: dict[str, tuple[ChangelogEntry, ...]], *, version: str, release_date: str) -> str:
    """Render one public dated release section without provenance metadata."""
    body = render_sections(sections, include_provenance=False)
    if not body:
        raise ChangelogError("No user-facing changelog entries are available to release.")
    return f"## [{version}] - {release_date}\n\n{body}\n"


def _replace_unreleased(text: str, sections: dict[str, tuple[ChangelogEntry, ...]]) -> str:
    layout = parse_changelog_layout(text)
    body = render_sections(sections)
    middle = f"\n\n{body}\n\n" if body else "\n\n"
    return f"{layout.prefix}{middle}{layout.suffix}"


def _version_exists(text: str, version: str) -> bool:
    heading = f"## [{version}]"
    return re.search(rf"(?m)^{re.escape(heading)}(?:\s|$)", text) is not None


def _insert_release(text: str, release: str) -> str:
    insertion = f"{RELEASES_MARKER}\n\n{release.rstrip()}"
    return text.replace(RELEASES_MARKER, insertion, 1)


def _validate_release_metadata(version: str, release_date: str) -> None:
    if RELEASE_VERSION_RE.fullmatch(version) is None:
        raise ChangelogError(f"Invalid release version {version!r}; expected X.Y.Z.")
    try:
        date.fromisoformat(release_date)
    except ValueError as exc:
        raise ChangelogError(f"Invalid release date {release_date!r}; expected YYYY-MM-DD.") from exc


def _consume_fragments(fragments: list[Fragment]) -> None:
    for fragment in fragments:
        fragment.path.unlink()


def consolidate_changelog(repo_root: Path, *, dry_run: bool) -> str:
    """Merge pending fragments into Unreleased and optionally consume them."""
    layout, fragments = validate_repository(repo_root)
    if not fragments:
        if dry_run:
            return render_unreleased(layout.unreleased)
        raise ChangelogError("No pending changelog fragments are available to consolidate.")
    merged = _merge_sections(layout.unreleased, _fragment_sections(fragments))
    changelog_path = repo_root / CHANGELOG_FILENAME
    updated = _replace_unreleased(_read_text(changelog_path), merged)
    preview = render_unreleased(merged)
    if dry_run:
        return preview
    if updated != _read_text(changelog_path):
        _write_text(changelog_path, updated)
    _consume_fragments(fragments)
    return preview


def release_changelog(
    repo_root: Path,
    *,
    version: str,
    release_date: str,
    dry_run: bool,
) -> str:
    """Promote Unreleased and pending fragments into one dated release."""
    _validate_release_metadata(version, release_date)
    layout, fragments = validate_repository(repo_root)
    changelog_path = repo_root / CHANGELOG_FILENAME
    changelog = _read_text(changelog_path)
    if _version_exists(changelog, version):
        raise ChangelogError(f"{CHANGELOG_FILENAME} already contains a {version} release section.")

    merged = _merge_sections(layout.unreleased, _fragment_sections(fragments))
    release = render_release(merged, version=version, release_date=release_date)
    updated = _insert_release(_replace_unreleased(changelog, {}), release)
    if dry_run:
        return release
    _write_text(changelog_path, updated)
    _consume_fragments(fragments)
    return release


def _extract_release(text: str, version: str, *, source: str) -> ReleaseSection:
    heading_re = re.compile(rf"(?m)^## \[{re.escape(version)}\](?: - [^\n]+)?$")
    match = heading_re.search(text)
    if match is None:
        raise ChangelogError(f"{source}: release section {version!r} was not found.")
    next_heading = re.search(r"(?m)^## \[", text[match.end() :])
    end = match.end() + next_heading.start() if next_heading else len(text)
    body = text[match.end() : end].strip()
    sections = parse_fragment_text(body, source=f"{source} [{version}]", allow_fragment_markers=True)
    return ReleaseSection(text=text[match.start() : end].strip() + "\n", sections=sections)


def _fragment_sections_at_ref(repo_root: Path, ref: str) -> dict[str, tuple[ChangelogEntry, ...]]:
    """Load committed Markdown fragments from a Git ref."""
    output = _run_git_process(
        repo_root,
        "ls-tree",
        "-r",
        "--name-only",
        ref,
        "--",
        FRAGMENTS_DIRECTORY,
    )
    groups: list[dict[str, tuple[ChangelogEntry, ...]]] = []
    for relative_path in output.splitlines():
        path = Path(relative_path)
        if path.suffix != ".md" or path.name in IGNORED_FRAGMENT_FILES:
            continue
        _validate_fragment_name(path)
        fragment_id = path.stem
        parsed = parse_fragment_text(
            _git_show_file(repo_root, ref, relative_path),
            source=f"{ref}:{relative_path}",
        )
        groups.append(
            {
                category: tuple(ChangelogEntry(text=entry.text, fragment_ids=(fragment_id,)) for entry in entries)
                for category, entries in parsed.items()
            }
        )
    return _merge_sections(*groups)


def _recover_release_provenance(
    repo_root: Path,
    *,
    source_ref: str,
    version: str,
    released: dict[str, tuple[ChangelogEntry, ...]],
) -> dict[str, tuple[ChangelogEntry, ...]]:
    """Recover release entry IDs from the state before public promotion."""
    try:
        history = _run_git_process(
            repo_root,
            "log",
            "--format=%H",
            "-S",
            f"## [{version}]",
            source_ref,
            "--",
            CHANGELOG_FILENAME,
        )
        introduction_commit = next((line for line in history.splitlines() if line), None)
        if introduction_commit is None:
            return released
        parent = _run_git(repo_root, "rev-parse", f"{introduction_commit}^1")
        previous_layout = parse_changelog_layout(
            _git_show_file(repo_root, parent, CHANGELOG_FILENAME),
            source=f"{parent}:{CHANGELOG_FILENAME}",
        )
        previous = _merge_sections(previous_layout.unreleased, _fragment_sections_at_ref(repo_root, parent))
    except ChangelogError:
        return released

    available: dict[tuple[str, str], list[ChangelogEntry]] = {}
    for category, entries in previous.items():
        for entry in entries:
            available.setdefault((category, entry.text), []).append(entry)

    recovered: dict[str, tuple[ChangelogEntry, ...]] = {}
    for category, entries in released.items():
        recovered_entries: list[ChangelogEntry] = []
        for entry in entries:
            candidates = available.get((category, entry.text), [])
            source_entry = candidates.pop(0) if candidates else entry
            recovered_entries.append(ChangelogEntry(text=entry.text, fragment_ids=source_entry.fragment_ids))
        recovered[category] = tuple(recovered_entries)
    return recovered


def _entry_counts(sections: dict[str, tuple[ChangelogEntry, ...]]) -> Counter[tuple[str, str]]:
    return Counter((category, entry.text) for category, entries in sections.items() for entry in entries)


def _filter_released_entries(
    current: dict[str, tuple[ChangelogEntry, ...]],
    released: dict[str, tuple[ChangelogEntry, ...]],
) -> dict[str, tuple[ChangelogEntry, ...]]:
    released_ids = {
        fragment_id for entries in released.values() for entry in entries for fragment_id in entry.fragment_ids
    }
    released_counts = Counter(
        (category, entry.text) for category, entries in released.items() for entry in entries if not entry.fragment_ids
    )
    remaining: dict[str, tuple[ChangelogEntry, ...]] = {}

    for category in CATEGORIES:
        kept: list[ChangelogEntry] = []
        for entry in current.get(category, ()):
            key = (category, entry.text)
            remove = bool(released_ids.intersection(entry.fragment_ids))
            if not remove and released_counts[key] > 0:
                remove = True
            if remove:
                if released_counts[key] > 0:
                    released_counts[key] -= 1
            else:
                kept.append(entry)
        if kept:
            remaining[category] = tuple(kept)
    return remaining


def reconcile_changelog(
    repo_root: Path,
    *,
    source_ref: str,
    version: str,
    dry_run: bool,
) -> str:
    """Import a release while preserving main-only Unreleased entries."""
    if RELEASE_VERSION_RE.fullmatch(version) is None:
        raise ChangelogError(f"Invalid release version {version!r}; expected X.Y.Z.")
    layout, fragments = validate_repository(repo_root)
    changelog_path = repo_root / CHANGELOG_FILENAME
    changelog = _read_text(changelog_path)
    if _version_exists(changelog, version):
        raise ChangelogError(f"{CHANGELOG_FILENAME} already contains a {version} release section.")

    source_changelog = _git_show_file(repo_root, source_ref, CHANGELOG_FILENAME)
    source_release = _extract_release(source_changelog, version, source=f"{source_ref}:{CHANGELOG_FILENAME}")
    released = _recover_release_provenance(
        repo_root,
        source_ref=source_ref,
        version=version,
        released=source_release.sections,
    )
    pending = _merge_sections(layout.unreleased, _fragment_sections(fragments))
    remaining = _filter_released_entries(pending, released)
    updated = _insert_release(_replace_unreleased(changelog, remaining), source_release.text)
    preview = f"{render_unreleased(remaining).rstrip()}\n\n{source_release.text}"
    if dry_run:
        return preview
    _write_text(changelog_path, updated)
    _consume_fragments(fragments)
    return preview


def _parse_git_changes(output: str) -> list[GitChange]:
    changes: list[GitChange] = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            changes.append(GitChange(status=parts[0], paths=tuple(parts[1:])))
    return changes


def _is_fragment_path(path: str) -> bool:
    prefix = f"{FRAGMENTS_DIRECTORY}/"
    return path.startswith(prefix) and Path(path).name not in IGNORED_FRAGMENT_FILES


def _is_workflow_transition(repo_root: Path, base_ref: str) -> bool:
    current = _read_text(repo_root / CHANGELOG_FILENAME)
    try:
        previous = _git_show_file(repo_root, base_ref, CHANGELOG_FILENAME)
    except ChangelogError:
        return False
    if RELEASES_MARKER not in current or RELEASES_MARKER in previous:
        return False
    return current.replace(f"{RELEASES_MARKER}\n\n", "", 1) == previous.replace("\r\n", "\n")


def check_pull_request(
    repo_root: Path,
    *,
    base_ref: str,
    target_branch: str,
    allow_changelog_update: bool = False,
) -> list[Path]:
    """Validate the fragment contribution made by a pull request."""
    _, fragments = validate_repository(repo_root)
    output = _run_git_process(repo_root, "diff", "--name-status", "--find-renames", f"{base_ref}...HEAD", "--")
    changes = _parse_git_changes(output)
    changelog_changed = any(CHANGELOG_FILENAME in change.paths for change in changes)
    workflow_transition = changelog_changed and _is_workflow_transition(repo_root, base_ref)
    fragment_changes = [
        change for change in changes if any(_is_fragment_path(path.replace("\\", "/")) for path in change.paths)
    ]

    if changelog_changed and not (allow_changelog_update or workflow_transition):
        raise ChangelogError(
            f"Do not edit {CHANGELOG_FILENAME} directly; add a fragment or use an authorized changelog maintenance PR."
        )

    if allow_changelog_update:
        invalid = [change for change in fragment_changes if change.status != "D"]
        if invalid:
            raise ChangelogError(
                "Changelog maintenance PRs may delete consumed fragments but must not add or edit them."
            )
        if not changelog_changed and not fragment_changes:
            raise ChangelogError("Changelog maintenance PRs must update the changelog or consume a fragment.")
        deleted_markdown = any(
            path.replace("\\", "/").endswith(".md")
            for change in fragment_changes
            for path in change.paths
            if _is_fragment_path(path.replace("\\", "/"))
        )
        if deleted_markdown and not changelog_changed:
            raise ChangelogError(
                f"Changelog maintenance that consumes Markdown fragments must update {CHANGELOG_FILENAME}."
            )
        unexpected_paths = sorted(
            {
                path.replace("\\", "/")
                for change in changes
                for path in change.paths
                if path.replace("\\", "/") != CHANGELOG_FILENAME and not _is_fragment_path(path.replace("\\", "/"))
            }
        )
        if unexpected_paths:
            raise ChangelogError(
                "Changelog maintenance PRs must contain only the changelog and consumed fragment deletions: "
                + ", ".join(unexpected_paths)
            )
        return []

    invalid = [change for change in fragment_changes if change.status != "A"]
    if invalid:
        paths = sorted(
            {
                path.replace("\\", "/")
                for change in invalid
                for path in change.paths
                if _is_fragment_path(path.replace("\\", "/"))
            }
        )
        raise ChangelogError(
            "Pending fragments are immutable once merged; add a new fragment instead of modifying, renaming, or deleting: "
            + ", ".join(paths)
        )

    added_paths = {
        path.replace("\\", "/")
        for change in fragment_changes
        if change.status == "A"
        for path in change.paths
        if _is_fragment_path(path.replace("\\", "/"))
    }
    added = [fragment.path for fragment in fragments if fragment.path.relative_to(repo_root).as_posix() in added_paths]

    if target_branch.startswith("release-"):
        if not added:
            raise ChangelogError("A release-branch PR must carry at least one fragment from its backported changes.")
    elif len(added) != 1:
        raise ChangelogError(
            f"A PR targeting {target_branch!r} must add exactly one .md or .skip fragment; found {len(added)}."
        )
    return added


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Create a uniquely named fragment.")
    create.add_argument("name", nargs="?", help="Descriptive name; defaults to the current branch basename.")
    create.add_argument("--category", choices=[category.lower() for category in CATEGORIES])
    create.add_argument("--content")
    create.add_argument("--skip", dest="skip_reason", metavar="REASON")

    subparsers.add_parser("validate", help="Validate the changelog and pending fragments.")

    check = subparsers.add_parser("check", help="Validate the fragment contribution in a pull request.")
    check.add_argument("--base-ref", required=True)
    check.add_argument("--target-branch", required=True)
    check.add_argument("--allow-changelog-update", action="store_true")

    build = subparsers.add_parser("build", help="Consolidate pending fragments into Unreleased.")
    build.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed Unreleased section without changing files.",
    )

    release = subparsers.add_parser("release", help="Promote Unreleased into a dated release.")
    release.add_argument("--version", required=True)
    release.add_argument("--date", dest="release_date", default=date.today().isoformat())
    release.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed dated section without changing files.",
    )

    reconcile = subparsers.add_parser("reconcile", help="Import a release while preserving main-only entries.")
    reconcile.add_argument("--source-ref", required=True)
    reconcile.add_argument("--version", required=True)
    reconcile.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed reconciliation without changing files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the changelog command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    try:
        if args.command == "create":
            path = create_fragment(
                repo_root,
                name=args.name,
                category=args.category,
                content=args.content,
                skip_reason=args.skip_reason,
            )
            print(path.relative_to(repo_root).as_posix())
        elif args.command == "validate":
            layout, fragments = validate_repository(repo_root)
            entry_count = sum(len(entries) for entries in layout.unreleased.values())
            print(f"Validated {entry_count} Unreleased entries and {len(fragments)} pending fragment(s).")
        elif args.command == "check":
            added = check_pull_request(
                repo_root,
                base_ref=args.base_ref,
                target_branch=args.target_branch,
                allow_changelog_update=args.allow_changelog_update,
            )
            print(f"Validated {len(added)} fragment contribution(s).")
        elif args.command == "build":
            preview = consolidate_changelog(repo_root, dry_run=args.dry_run)
            if args.dry_run:
                print(preview, end="")
            else:
                print("Consolidated pending fragments into CHANGELOG.md [Unreleased].")
        elif args.command == "release":
            release = release_changelog(
                repo_root,
                version=args.version,
                release_date=args.release_date,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(release, end="")
            else:
                print(f"Promoted CHANGELOG.md [Unreleased] to release {args.version}.")
        elif args.command == "reconcile":
            preview = reconcile_changelog(
                repo_root,
                source_ref=args.source_ref,
                version=args.version,
                dry_run=args.dry_run,
            )
            if args.dry_run:
                print(preview, end="")
            else:
                print(f"Reconciled release {args.version} from {args.source_ref}.")
    except ChangelogError as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
