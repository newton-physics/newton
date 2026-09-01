#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Emit a dependency and license delta section for Newton release reports.

Compares direct requirements, uv.lock package variants, project license
metadata, and in-tree license notice files between two git refs. Stdlib-only.
Network use is limited to optional PyPI JSON metadata lookups for newly added
packages and changed locked package versions.

The stdlib-only implementation is intentional: this helper runs before releases
across arbitrary git refs without bootstrapping the target environment or adding
release-only dependencies. Existing tools such as pip-licenses, cyclonedx-py,
pip-audit, and syft remain useful for broader inventory work, but they do not
replace this script's deterministic git-ref diff over pyproject metadata,
uv.lock, declared license-file pathspecs, and version-specific PyPI JSON.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on Python < 3.11
    tomllib = None


_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_PYPI_REGISTRY = "https://pypi.org/simple"
_SKIP_PYPI_LICENSE = "not checked (--skip-pypi)"
_LICENSE_REVIEW_RE = re.compile(
    r"(^|[^a-z0-9])(proprietary|agpl|lgpl|gpl|commercial|unknown)([^a-z0-9]|$)",
    re.IGNORECASE,
)
_LICENSE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
_LICENSE_EXPRESSION_OPERATORS = {"AND", "OR", "WITH"}


@dataclass(frozen=True)
class Requirement:
    group: str
    name: str
    normalized_name: str
    spec: str
    self_reference: bool


@dataclass(frozen=True)
class ProjectMetadata:
    requirements: list[Requirement]
    extras: set[str]
    license: str
    license_files: tuple[str, ...]


@dataclass(frozen=True)
class LockedPackage:
    name: str
    normalized_name: str
    version: str | None
    registry: str | None
    source: str | None
    markers: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


def _git(repo: Path, *args: str, required: bool = True) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("license_audit: 'git' executable not found on PATH")
    if result.returncode != 0:
        if required:
            sys.exit(
                f"license_audit: git {' '.join(args)} failed in {repo} "
                f"(exit {result.returncode}): {result.stderr.strip()}"
            )
        if result.stderr.strip():
            print(
                f"license_audit: optional git {' '.join(args)} unavailable in {repo} "
                f"(exit {result.returncode}): {result.stderr.strip()}",
                file=sys.stderr,
            )
        return ""
    return result.stdout


def _load_toml(text: str, label: str) -> dict:
    if tomllib is None:
        sys.exit(
            "license_audit: Python 3.11+ is required because this helper uses "
            f"stdlib tomllib to parse {label}. Run the release audit with "
            "Python 3.11 or newer."
        )
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        sys.exit(f"license_audit: failed to parse {label}: {exc}")


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(spec: str) -> str | None:
    match = _REQ_NAME_RE.match(spec)
    return match.group(1) if match else None


def _format_toml_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _parse_pyproject(text: str) -> ProjectMetadata:
    data = _load_toml(text, "pyproject.toml")
    project = data.get("project", {})

    groups: dict[str, list[str]] = {
        "runtime": list(project.get("dependencies", [])),
    }
    optional = project.get("optional-dependencies", {})
    for extra, specs in optional.items():
        groups[f"extra:{extra}"] = list(specs)

    requirements = []
    for group, specs in groups.items():
        for spec in specs:
            name = _requirement_name(spec)
            if name is None:
                continue
            normalized_name = _normalize_name(name)
            requirements.append(
                Requirement(
                    group=group,
                    name=name,
                    normalized_name=normalized_name,
                    spec=spec,
                    self_reference=normalized_name == "newton",
                )
            )

    extras = {group.removeprefix("extra:") for group in groups if group.startswith("extra:")}
    license_files = tuple(str(item) for item in project.get("license-files") or [])
    return ProjectMetadata(
        requirements=requirements,
        extras=extras,
        license=_format_toml_value(project.get("license")),
        license_files=license_files,
    )


def _requirements_by_key(requirements: list[Requirement]) -> dict[tuple[str, str], Requirement]:
    return {(req.group, req.normalized_name): req for req in requirements}


def _source_description(source: object) -> str | None:
    if not isinstance(source, dict) or not source:
        return None
    if registry := source.get("registry"):
        return f"registry: {registry}"
    for key in ("git", "path", "directory", "url"):
        if key in source:
            return f"{key}: {source[key]}"
    return ", ".join(sorted(str(key) for key in source))


def _parse_lock(text: str) -> list[LockedPackage]:
    if not text.strip():
        return []
    data = _load_toml(text, "uv.lock")
    packages = []
    for package in data.get("package", []):
        name = str(package.get("name", ""))
        if not name:
            continue
        source = package.get("source", {})
        registry = source.get("registry") if isinstance(source, dict) else None
        markers = []
        if marker := package.get("marker"):
            markers.append(str(marker))
        markers.extend(str(marker) for marker in package.get("resolution-markers", []))
        dependencies = []
        dependency_groups = [package.get("dependencies", [])]
        dependency_groups.extend((package.get("optional-dependencies") or {}).values())
        for group in dependency_groups:
            for dependency in group:
                if isinstance(dependency, dict) and dependency.get("name"):
                    dependencies.append(_normalize_name(str(dependency["name"])))
        packages.append(
            LockedPackage(
                name=name,
                normalized_name=_normalize_name(name),
                version=str(package["version"]) if package.get("version") is not None else None,
                registry=registry,
                source=_source_description(source),
                markers=tuple(dict.fromkeys(markers)),
                dependencies=tuple(dict.fromkeys(dependencies)),
            )
        )
    return packages


def _lock_by_name(packages: list[LockedPackage]) -> dict[str, list[LockedPackage]]:
    by_name: dict[str, list[LockedPackage]] = {}
    for package in packages:
        if package.normalized_name == "newton":
            continue
        by_name.setdefault(package.normalized_name, []).append(package)
    return by_name


def _dependency_roots(packages: list[LockedPackage], roots: set[str]) -> dict[str, set[str]]:
    """Map each reachable locked package name to its direct dependency roots."""
    graph: dict[str, set[str]] = {}
    for package in packages:
        graph.setdefault(package.normalized_name, set()).update(package.dependencies)

    reachable_from: dict[str, set[str]] = {}
    for root in roots:
        pending = [root]
        visited = set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            reachable_from.setdefault(name, set()).add(root)
            pending.extend(graph.get(name, ()))
    return reachable_from


def _fetch_pypi_license(package: str, version: str | None, timeout: float) -> dict[str, str]:
    quoted_package = urllib.parse.quote(package)
    if version:
        quoted_version = urllib.parse.quote(version, safe="")
        url = f"https://pypi.org/pypi/{quoted_package}/{quoted_version}/json"
        evidence_url = f"https://pypi.org/project/{package}/{version}/"
    else:
        url = f"https://pypi.org/pypi/{quoted_package}/json"
        evidence_url = f"https://pypi.org/project/{package}/"

    request = urllib.request.Request(url, headers={"User-Agent": "newton-release-license-audit/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"license": f"not checked ({type(exc).__name__})", "url": url}

    info = payload.get("info", {})
    license_expression = (info.get("license_expression") or "").strip()
    license_text = (info.get("license") or "").strip()
    license_files = [str(path) for path in info.get("license_files") or []]
    classifiers = [
        classifier.removeprefix("License :: ").strip()
        for classifier in info.get("classifiers", [])
        if classifier.startswith("License :: ")
    ]
    if license_expression:
        license_value = license_expression
    elif license_text:
        license_value = license_text
    elif classifiers:
        license_value = "; ".join(classifiers)
    elif license_files:
        license_value = f"not declared (license files: {', '.join(license_files)})"
    else:
        license_value = "not declared"
    return {"license": license_value, "url": evidence_url}


def _locked_license(
    package: LockedPackage,
    skip_pypi: bool,
    timeout: float,
    cache: dict[tuple[str, str | None, str | None], dict[str, str]],
) -> dict[str, str]:
    cache_key = (package.normalized_name, package.version, package.registry or package.source)
    if cache_key in cache:
        return cache[cache_key]
    if skip_pypi:
        metadata = {"license": _SKIP_PYPI_LICENSE, "url": ""}
    elif package.registry is None:
        source = package.source or "source unavailable"
        metadata = {"license": f"not checked (non-PyPI source: {source})", "url": ""}
    elif package.registry.rstrip("/") != _PYPI_REGISTRY:
        metadata = {"license": f"not checked (non-PyPI registry: {package.registry})", "url": package.registry}
    else:
        metadata = _fetch_pypi_license(package.name, package.version, timeout)
    cache[cache_key] = metadata
    return metadata


def _name_license(
    package_name: str,
    skip_pypi: bool,
    timeout: float,
    cache: dict[tuple[str, str | None, str | None], dict[str, str]],
) -> dict[str, str]:
    normalized_name = _normalize_name(package_name)
    cache_key = (normalized_name, None, _PYPI_REGISTRY)
    if cache_key in cache:
        return cache[cache_key]
    if skip_pypi:
        metadata = {"license": _SKIP_PYPI_LICENSE, "url": ""}
    else:
        metadata = _fetch_pypi_license(package_name, None, timeout)
    cache[cache_key] = metadata
    return metadata


def _license_needs_review(license_value: str) -> bool:
    lowered = license_value.lower()
    if "not checked" in lowered or "not declared" in lowered:
        return True
    return _LICENSE_REVIEW_RE.search(lowered) is not None


def _metadata_needs_review(metadata: dict[str, str]) -> bool:
    return metadata["license"] != _SKIP_PYPI_LICENSE and _license_needs_review(metadata["license"])


def _is_standard_license_expression(value: str) -> bool:
    """Return whether *value* has the shape of an SPDX license expression."""
    tokens = value.replace("(", " ( ").replace(")", " ) ").split()
    if not tokens:
        return False

    expect_identifier = True
    parenthesis_depth = 0
    for token in tokens:
        if token == "(":
            if not expect_identifier:
                return False
            parenthesis_depth += 1
        elif token == ")":
            if expect_identifier or parenthesis_depth == 0:
                return False
            parenthesis_depth -= 1
        elif expect_identifier:
            if token in _LICENSE_EXPRESSION_OPERATORS or _LICENSE_IDENTIFIER_RE.fullmatch(token) is None:
                return False
            expect_identifier = False
        elif token in _LICENSE_EXPRESSION_OPERATORS:
            expect_identifier = True
        else:
            return False
    return parenthesis_depth == 0 and not expect_identifier


def _concise_license(metadata: dict[str, str]) -> str:
    """Return concise Markdown for a license value while retaining evidence."""
    value = metadata["license"]
    if value.startswith(("not checked", "not declared")) or _is_standard_license_expression(value):
        return value
    if metadata["url"]:
        return f"[package metadata]({metadata['url']})"
    return "package metadata unavailable"


def _md(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", r"\|").replace("\n", " ")


def _render_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return ""
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(_md(cell) for cell in row) + " |")
    return "\n".join(out)


def _format_versions(packages: list[LockedPackage]) -> str:
    versions = sorted({package.version or "(no version)" for package in packages})
    return ", ".join(versions)


def _format_markers(package: LockedPackage) -> str:
    if not package.markers:
        return ""
    return "; ".join(package.markers)


def _format_source(package: LockedPackage) -> str:
    return package.registry or package.source or ""


def _license_file_pathspecs(base_project: ProjectMetadata, head_project: ProjectMetadata) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*base_project.license_files, *head_project.license_files]))


def _git_glob_pathspec(pathspec: str) -> str:
    if pathspec.startswith(":("):
        return pathspec
    return f":(glob){pathspec}"


def _license_summary(
    packages: list[LockedPackage],
    skip_pypi: bool,
    timeout: float,
    cache: dict[tuple[str, str | None, str | None], dict[str, str]],
    *,
    concise: bool = False,
) -> tuple[str, str]:
    entries = []
    urls = []
    for package in sorted(packages, key=lambda item: (item.version or "", item.registry or "", item.markers)):
        metadata = _locked_license(package, skip_pypi, timeout, cache)
        version = package.version or "(no version)"
        license_value = _concise_license(metadata) if concise else metadata["license"]
        entries.append(f"{version}: {license_value}")
        if metadata["url"]:
            urls.append(metadata["url"])
    return "; ".join(entries), "; ".join(dict.fromkeys(urls))


def _license_delta(base_license: str, head_license: str) -> str:
    if base_license == head_license:
        return f"unchanged: {head_license}"
    return f"base: {base_license}; head: {head_license}"


def _is_external(req: Requirement) -> bool:
    return not req.self_reference


def build_audit(repo: Path, base: str, head: str, skip_pypi: bool, pypi_timeout: float) -> str:
    base_project = _parse_pyproject(_git(repo, "show", f"{base}:pyproject.toml"))
    head_project = _parse_pyproject(_git(repo, "show", f"{head}:pyproject.toml"))

    base_req_by_key = _requirements_by_key(base_project.requirements)
    head_req_by_key = _requirements_by_key(head_project.requirements)

    added_req_keys = sorted(set(head_req_by_key) - set(base_req_by_key))
    removed_req_keys = sorted(set(base_req_by_key) - set(head_req_by_key))
    changed_req_keys = sorted(
        key
        for key in set(head_req_by_key) & set(base_req_by_key)
        if head_req_by_key[key].spec != base_req_by_key[key].spec
    )

    base_external_names = {req.normalized_name for req in base_project.requirements if _is_external(req)}
    added_external_reqs = [head_req_by_key[key] for key in added_req_keys if _is_external(head_req_by_key[key])]
    new_direct_external = [req for req in added_external_reqs if req.normalized_name not in base_external_names]

    base_lock = _parse_lock(_git(repo, "show", f"{base}:uv.lock", required=False))
    head_lock = _parse_lock(_git(repo, "show", f"{head}:uv.lock", required=False))
    base_lock_by_name = _lock_by_name(base_lock)
    head_lock_by_name = _lock_by_name(head_lock)

    base_lock_names = set(base_lock_by_name)
    head_lock_names = set(head_lock_by_name)
    added_locked_names = sorted(head_lock_names - base_lock_names)
    removed_locked_names = sorted(base_lock_names - head_lock_names)
    changed_version_names = sorted(
        name
        for name in base_lock_names & head_lock_names
        if {pkg.version for pkg in base_lock_by_name[name]} != {pkg.version for pkg in head_lock_by_name[name]}
    )

    head_external_names = {req.normalized_name for req in head_project.requirements if _is_external(req)}
    existing_direct_roots = base_external_names & head_external_names
    new_direct_roots = head_external_names - base_external_names
    dependency_roots = _dependency_roots(head_lock, head_external_names)
    existing_dependency_added_names = []
    new_resolved_names = []
    for name in added_locked_names:
        package_roots = dependency_roots.get(name, set())
        if package_roots & existing_direct_roots and not package_roots & new_direct_roots:
            existing_dependency_added_names.append(name)
        else:
            new_resolved_names.append(name)
    resolved_change_names = sorted([*changed_version_names, *existing_dependency_added_names])

    project_license_rows = []
    if base_project.license != head_project.license:
        project_license_rows.append(["project.license", base_project.license, head_project.license])
    if base_project.license_files != head_project.license_files:
        project_license_rows.append(
            [
                "project.license-files",
                "; ".join(base_project.license_files),
                "; ".join(head_project.license_files),
            ]
        )

    license_pathspecs = _license_file_pathspecs(base_project, head_project)
    git_license_pathspecs = tuple(_git_glob_pathspec(pathspec) for pathspec in license_pathspecs)
    license_diff = (
        _git(repo, "diff", "--name-status", f"{base}..{head}", "--", *git_license_pathspecs, required=False)
        if license_pathspecs
        else ""
    )
    license_rows = [line.split("\t", 1) for line in license_diff.splitlines() if line.strip()]

    license_cache: dict[tuple[str, str | None, str | None], dict[str, str]] = {}
    review_packages = set()
    for req in new_direct_external:
        if req.normalized_name in head_lock_by_name:
            for package in head_lock_by_name[req.normalized_name]:
                if _metadata_needs_review(_locked_license(package, skip_pypi, pypi_timeout, license_cache)):
                    review_packages.add(req.normalized_name)
        elif _metadata_needs_review(_name_license(req.name, skip_pypi, pypi_timeout, license_cache)):
            review_packages.add(req.normalized_name)
    for name in added_locked_names:
        for package in head_lock_by_name[name]:
            if _metadata_needs_review(_locked_license(package, skip_pypi, pypi_timeout, license_cache)):
                review_packages.add(name)
    for name in changed_version_names:
        base_license, _base_urls = _license_summary(
            base_lock_by_name[name],
            skip_pypi,
            pypi_timeout,
            license_cache,
        )
        head_license, _head_urls = _license_summary(
            head_lock_by_name[name],
            skip_pypi,
            pypi_timeout,
            license_cache,
        )
        if not skip_pypi and base_license != head_license and _license_needs_review(head_license):
            review_packages.add(name)

    direct_external_names = {
        req.normalized_name for req in [*base_project.requirements, *head_project.requirements] if _is_external(req)
    }
    direct_version_changes = sum(name in direct_external_names for name in resolved_change_names)
    transitive_version_changes = len(resolved_change_names) - direct_version_changes
    license_scope = ", ".join(f"`{path}`" for path in license_pathspecs) or "none declared"

    lines = [
        f"Compared `pyproject.toml`, `uv.lock`, and declared license-file pathspecs between `{base}` and `{head}`.",
        "",
        "### Summary",
        "",
        f"- New external direct dependency names: {len(new_direct_external)}",
        f"- Added external direct requirement scopes: {len(added_external_reqs)}",
        f"- New resolved package names: {len(new_resolved_names)}",
        f"- Removed resolved package names: {len(removed_locked_names)}",
        f"- Existing resolved package version-set changes: {len(resolved_change_names)} "
        f"({direct_version_changes} direct, {transitive_version_changes} transitive)",
        f"- Project license metadata changes: {len(project_license_rows)}",
        f"- In-tree license notice file changes: {len(license_rows)}",
        f"- License notice pathspecs compared: {license_scope}",
    ]
    if skip_pypi:
        lines.append("- License metadata needing review: not evaluated (--skip-pypi)")
    elif review_packages:
        lines.append(f"- License metadata needing review: {', '.join(sorted(review_packages))}")
    else:
        lines.append("- License metadata needing review: none detected")

    new_extras = sorted(head_project.extras - base_project.extras)
    if new_extras:
        rows = []
        for extra in new_extras:
            reqs = [req.spec for req in head_project.requirements if req.group == f"extra:{extra}"]
            rows.append([extra, "; ".join(reqs)])
        lines.extend(["", "### New Optional Extras", "", _render_table(["Extra", "Requirements"], rows)])

    if added_external_reqs:
        rows = []
        for req in added_external_reqs:
            status = (
                "new package name" if req.normalized_name not in base_external_names else "existing package, new scope"
            )
            if req.normalized_name in head_lock_by_name:
                license_value, evidence = _license_summary(
                    head_lock_by_name[req.normalized_name],
                    skip_pypi,
                    pypi_timeout,
                    license_cache,
                )
            else:
                metadata = _name_license(req.name, skip_pypi, pypi_timeout, license_cache)
                license_value = metadata["license"]
                evidence = metadata["url"]
            rows.append([req.name, req.group, status, req.spec, license_value, evidence])
        lines.extend(
            [
                "",
                "### Added Direct Requirements / Scope Changes",
                "",
                _render_table(
                    ["Package", "Scope", "Status", "Requirement", "License metadata", "Evidence"],
                    rows,
                ),
            ]
        )

    if new_resolved_names:
        rows = []
        for name in new_resolved_names:
            for package in head_lock_by_name[name]:
                metadata = _locked_license(package, skip_pypi, pypi_timeout, license_cache)
                rows.append(
                    [
                        package.name,
                        package.version or "",
                        _format_source(package),
                        _format_markers(package),
                        metadata["license"],
                        metadata["url"],
                    ]
                )
        lines.extend(
            [
                "",
                "### New Resolved Packages",
                "",
                _render_table(
                    ["Package", "Version", "Source", "Markers", "License metadata", "Evidence"],
                    rows,
                ),
            ]
        )

    if changed_req_keys:
        rows = []
        for key in changed_req_keys:
            old_req = base_req_by_key[key]
            new_req = head_req_by_key[key]
            rows.append([new_req.name, new_req.group, old_req.spec, new_req.spec])
        lines.extend(
            [
                "",
                "### Direct Requirement Changes",
                "",
                _render_table(["Package", "Scope", "Base", "Head"], rows),
            ]
        )

    if resolved_change_names:
        rows = []
        for name in resolved_change_names:
            if name in base_lock_by_name:
                base_license, base_evidence = _license_summary(
                    base_lock_by_name[name],
                    skip_pypi,
                    pypi_timeout,
                    license_cache,
                    concise=True,
                )
                base_versions = _format_versions(base_lock_by_name[name])
            else:
                base_license, base_evidence = "not resolved", ""
                base_versions = "(not resolved)"
            head_license, head_evidence = _license_summary(
                head_lock_by_name[name],
                skip_pypi,
                pypi_timeout,
                license_cache,
                concise=True,
            )
            evidence = "; ".join(url for url in (base_evidence, head_evidence) if url)
            if name in direct_external_names:
                scope = "direct"
            elif name in existing_dependency_added_names:
                roots = sorted(dependency_roots.get(name, set()) & existing_direct_roots)
                scope = f"transitive via {', '.join(roots)}"
            else:
                scope = "transitive"
            rows.append(
                [
                    head_lock_by_name[name][0].name,
                    scope,
                    base_versions,
                    _format_versions(head_lock_by_name[name]),
                    _license_delta(base_license, head_license),
                    evidence,
                ]
            )
        change_label = "change" if len(resolved_change_names) == 1 else "changes"
        lines.extend(
            [
                "",
                "### Existing Resolved Package Version-Set Changes",
                "",
                "<details>",
                f"<summary>{len(resolved_change_names)} package version-set {change_label} (click to expand)</summary>",
                "",
                _render_table(
                    ["Package", "Scope", "Base versions", "Head versions", "License metadata", "Evidence"], rows
                ),
                "",
                "</details>",
            ]
        )

    if project_license_rows:
        lines.extend(
            [
                "",
                "### Project License Metadata Changes",
                "",
                _render_table(["Field", "Base", "Head"], project_license_rows),
            ]
        )

    if removed_req_keys:
        rows = []
        for key in removed_req_keys:
            req = base_req_by_key[key]
            if _is_external(req):
                rows.append([req.name, req.group, req.spec])
        if rows:
            lines.extend(
                [
                    "",
                    "### Removed Direct Requirements",
                    "",
                    _render_table(["Package", "Scope", "Requirement"], rows),
                ]
            )

    if removed_locked_names:
        rows = []
        for name in removed_locked_names:
            for package in base_lock_by_name[name]:
                rows.append([package.name, package.version or "", _format_source(package), _format_markers(package)])
        lines.extend(
            [
                "",
                "### Removed Resolved Packages",
                "",
                _render_table(["Package", "Version", "Source", "Markers"], rows),
            ]
        )

    if license_rows:
        rows = [[status, path] for status, path in license_rows]
        lines.extend(
            [
                "",
                "### In-Tree License Notice File Changes",
                "",
                _render_table(["Status", "Path"], rows),
            ]
        )

    if not any(
        [
            added_external_reqs,
            added_locked_names,
            removed_locked_names,
            resolved_change_names,
            project_license_rows,
            license_rows,
        ]
    ):
        lines.extend(["", "No dependency package-name changes or in-tree license notice changes were detected."])

    return "\n".join(line for line in lines if line is not None)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit dependency and license audit markdown for a release range.")
    parser.add_argument("--base", required=True, help="Base git ref, e.g. v1.2.1")
    parser.add_argument("--head", required=True, help="Head git ref, e.g. v1.3.0rc1")
    parser.add_argument("--skip-pypi", action="store_true", help="Do not query PyPI JSON metadata for packages")
    parser.add_argument("--pypi-timeout", type=float, default=3.0, help="Per-package PyPI metadata timeout in seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    repo = Path.cwd()
    print(build_audit(repo, args.base, args.head, args.skip_pypi, args.pypi_timeout))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
