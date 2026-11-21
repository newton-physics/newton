#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Emit a JSON commit list for Newton release-candidate reporting.

Deterministic, stdlib-only. Enumerates commits in <base>..<head>, extracts
GH refs, identifies main equivalents via subject matching, and computes
soak-time metrics. No analysis, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType

# ASCII Unit Separator (U+001F). Not valid inside a git commit subject or
# author name, so safe to use as a field delimiter in log output.
_LOG_DELIM = "\x1f"
_LOG_FMT = _LOG_DELIM.join(["%H", "%s", "%an", "%cs"])


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C repo args...`` and return stdout, exiting clearly on failure.

    On non-zero exit, surfaces git's stderr (which the raw
    ``subprocess.CalledProcessError`` hides). On missing git binary, reports
    that explicitly rather than the cryptic ``FileNotFoundError``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("list_commits: 'git' executable not found on PATH")
    if result.returncode != 0:
        sys.exit(
            f"list_commits: git {' '.join(args)} failed in {repo} (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def get_commits_in_range(repo: Path, base: str, head: str) -> list[dict]:
    """Enumerate commits in base..head, oldest first, skipping merges.

    Returns one dict per commit with keys: sha, subject, author, committer_date.
    """
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        f"--format={_LOG_FMT}",
        f"{base}..{head}",
    )
    commits = []
    for line in stdout.splitlines():
        if not line:
            continue
        sha, subject, author, committer_date = line.split(_LOG_DELIM, 3)
        commits.append(
            {
                "sha": sha,
                "subject": subject,
                "author": author,
                "committer_date": committer_date,
            }
        )
    return commits


# Word boundary before GH prevents matching "Regraph-42" etc.
_GH_REF_RE = re.compile(r"\bGH-(\d+)")
# Bare PR refs as written in commit trailers like "Closes #1287". The
# lookbehind rejects `identifier#N` / URL fragments, leaving genuine
# standalone PR references. Matches the appendix regex in SKILL.md.
_PR_REF_RE = re.compile(r"(?<![\w/])#(\d+)\b")


def extract_gh_refs(subject: str, body: str) -> list[int]:
    """Return sorted, deduplicated GH / bare-PR issue numbers from subject+body."""
    nums = set()
    for text in (subject, body):
        for pattern in (_GH_REF_RE, _PR_REF_RE):
            for match in pattern.finditer(text):
                nums.add(int(match.group(1)))
    return sorted(nums)


def get_commit_body(repo: Path, sha: str) -> str:
    """Fetch the full commit message body (everything after the subject line)."""
    return _git(repo, "log", "-1", "--format=%b", sha).rstrip("\n")


def get_commit_files(repo: Path, sha: str) -> list[str]:
    """Return the list of file paths changed by a single commit.

    Uses --root so the initial commit (with no parent) also enumerates files.
    """
    stdout = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha)
    return [line for line in stdout.splitlines() if line]


def days_between(earlier_iso: str, later_iso: str) -> int:
    """Days from earlier_iso to later_iso (YYYY-MM-DD). May be negative."""
    d1 = date.fromisoformat(earlier_iso)
    d2 = date.fromisoformat(later_iso)
    return (d2 - d1).days


# Sentinel used to mark a subject whose committer SHA could not be
# unambiguously resolved (i.e., it appears more than once on main_ref).
# Stored alongside real entries so that find_main_equivalent can distinguish
# "not present" from "present but ambiguous". Wrapped in MappingProxyType so
# the shared instance cannot be mutated through any one index slot.
_AMBIGUOUS = MappingProxyType({"sha": None, "committer_date": None, "_ambiguous": True})


def build_main_subject_index(repo: Path, base: str, main_ref: str) -> dict:
    """Build subject → {sha, committer_date} map from <base>..<main_ref> commits.

    Walks the range oldest-first (``git log --reverse``) so the first insertion
    per subject is the original occurrence on main. A subject that appears more
    than once is treated as ambiguous and will resolve to None via
    ``find_main_equivalent`` (safer than silently picking one candidate).
    """
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        f"--format={_LOG_FMT}",
        f"{base}..{main_ref}",
    )
    # Values are either a plain dict for a unique subject, or the shared
    # MappingProxyType sentinel ``_AMBIGUOUS`` for a duplicated one.
    index: dict[str, Mapping] = {}
    for line in stdout.splitlines():
        if not line:
            continue
        sha, subject, _author, committer_date = line.split(_LOG_DELIM, 3)
        if subject in index:
            # Second occurrence: mark ambiguous so downstream doesn't pick
            # whichever landed first as the canonical main-side commit.
            index[subject] = _AMBIGUOUS
        else:
            index[subject] = {"sha": sha, "committer_date": committer_date}
    return index


def find_main_equivalent(index: dict, subject: str) -> dict | None:
    """Look up a head commit's main equivalent by exact subject.

    Returns ``{sha, committer_date}`` on an unambiguous match, ``None``
    otherwise (no match, or subject appears multiple times on main_ref).
    """
    entry = index.get(subject)
    if entry is None or entry.get("_ambiguous"):
        return None
    return entry


def _resolve_sha(repo: Path, ref: str) -> str:
    """Resolve a git ref to its full commit SHA.

    Uses ``<ref>^{commit}`` to peel annotated tags (which resolve to the
    tag object's SHA) down to the commit they point at. A no-op for
    lightweight tags, branches, and raw commit SHAs.
    """
    return _git(repo, "rev-parse", f"{ref}^{{commit}}").strip()


def build_commit_entry(
    repo: Path,
    commit: dict,
    subject_index: dict,
    report_date: str,
) -> dict:
    """Enrich a commit dict with files, gh_refs, days, main_equivalent."""
    body = get_commit_body(repo, commit["sha"])
    gh_refs = extract_gh_refs(commit["subject"], body)
    files = get_commit_files(repo, commit["sha"])
    days_since_merge = days_between(commit["committer_date"], report_date)
    main_match = find_main_equivalent(subject_index, commit["subject"])
    if main_match is not None:
        main_sha = main_match["sha"]
        days_in_main = days_between(main_match["committer_date"], report_date)
    else:
        main_sha = None
        days_in_main = None
    return {
        "sha": commit["sha"],
        "subject": commit["subject"],
        "author": commit["author"],
        "committer_date": commit["committer_date"],
        "days_since_merge": days_since_merge,
        "main_equivalent_sha": main_sha,
        "days_in_main": days_in_main,
        "files": files,
        "gh_refs": gh_refs,
    }


def parse_args(argv):
    """Parse CLI args; see ``--help``."""
    p = argparse.ArgumentParser(
        description="Emit commit metadata as JSON for RC report generation.",
    )
    p.add_argument("--base", required=True, help="Base git ref (e.g. v1.1.0)")
    p.add_argument("--head", required=True, help="Head git ref (e.g. upstream/release-1.2)")
    p.add_argument("--report-date", required=True, help="Report date YYYY-MM-DD")
    p.add_argument(
        "--main-ref",
        default="upstream/main",
        help="Main-branch ref for cherry-pick detection (default: upstream/main)",
    )
    return p.parse_args(argv)


def main(argv=None):
    """Build the commit-list JSON for <base>..<head> and print it to stdout."""
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Validate --report-date upfront rather than discovering the problem
    # deep inside the per-commit loop.
    try:
        date.fromisoformat(args.report_date)
    except ValueError:
        sys.exit(f"list_commits: --report-date must be YYYY-MM-DD, got {args.report_date!r}")

    repo = Path.cwd()
    base_sha = _resolve_sha(repo, args.base)
    head_sha = _resolve_sha(repo, args.head)

    commits = get_commits_in_range(repo, args.base, args.head)
    if not commits:
        sys.stderr.write(
            f"list_commits: warning: {args.base}..{args.head} contains no commits. "
            f"Resolved: base={base_sha[:12]}, head={head_sha[:12]}. "
            f"Check that --base is an ancestor of --head.\n"
        )

    subject_index = build_main_subject_index(repo, args.base, args.main_ref)
    entries = [build_commit_entry(repo, c, subject_index, args.report_date) for c in commits]

    out = {
        "resolved": {
            "target_version": None,
            "report_date": args.report_date,
            "base": {"ref": args.base, "sha": base_sha},
            "head": {"ref": args.head, "sha": head_sha},
            "main_ref": args.main_ref,
        },
        "commits": entries,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
