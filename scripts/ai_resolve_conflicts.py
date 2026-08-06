#!/usr/bin/env python3
"""AI-assisted git merge conflict resolver.

Resolves conflicts left by an in-progress ``git merge`` by asking an Anthropic
model to produce a merged version of each conflicted text file, using full
three-way context (base / ours / theirs). Resolved files are written back and
staged. Binary conflicts are never touched. Intended for CI use by the
``Sync Upstream`` workflow; run it while the merge is still in progress.

Environment:
    ANTHROPIC_API_KEY   Required. Anthropic API key.
    ANTHROPIC_MODEL     Optional. Model id (default: claude-sonnet-4-5).
    AI_RESOLVE_MAX_TOKENS  Optional. Max output tokens per file (default 32000).

Exit codes:
    0  all conflicts resolved and staged (or nothing to resolve)
    1  one or more files could not be resolved (left for a human)
    2  misconfiguration (missing API key)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL") or "claude-sonnet-4-5"
API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MAX_TOKENS = int(os.environ.get("AI_RESOLVE_MAX_TOKENS", "32000"))
MARKERS = ("<<<<<<<", ">>>>>>>")


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def conflicted_files() -> list[str]:
    out = sh("git", "diff", "--name-only", "--diff-filter=U").stdout
    return [line for line in out.splitlines() if line.strip()]


def stage_blob(stage: int, path: str) -> str | None:
    """Return the text of a merge stage (1=base, 2=ours, 3=theirs), or None."""
    r = subprocess.run(["git", "show", f":{stage}:{path}"], capture_output=True)
    if r.returncode != 0:
        return ""  # side added/removed the file; empty is the right context
    if b"\x00" in r.stdout:
        return None  # binary
    return r.stdout.decode("utf-8", errors="replace")


def strip_fences(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines)


def call_anthropic(prompt: str) -> str:
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


def build_prompt(path: str, base: str, ours: str, theirs: str, merged: str) -> str:
    return f"""You are resolving a git merge conflict in the file `{path}`.

Context: this is the "Digital Instron" fork of the Newton GPU physics engine.
"ours" is the fork; "theirs" is upstream newton-physics/newton `main`.

Produce the correct fully-merged file that keeps BOTH sides' intent:
- preserve the fork's customizations present in OURS,
- incorporate upstream's changes from THEIRS,
- when they genuinely conflict, combine them so the fork keeps working while
  adopting upstream's improvements; do not silently drop functionality from
  either side unless one clearly supersedes the other,
- keep imports, syntax, and indentation valid; follow the file's existing style.

Return ONLY the complete resolved file contents. No explanation, no markdown
code fences, and no conflict markers (<<<<<<<, =======, >>>>>>>).

===== COMMON ANCESTOR (base) =====
{base}
===== OURS (Digital Instron fork) =====
{ours}
===== THEIRS (upstream newton-physics/newton) =====
{theirs}
===== CURRENT FILE WITH CONFLICT MARKERS =====
{merged}
"""


def main() -> int:
    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    files = conflicted_files()
    if not files:
        print("No conflicted files to resolve.")
        return 0

    print(f"Resolving {len(files)} conflicted file(s) with model '{MODEL}'.")
    failed: list[str] = []

    for path in files:
        base = stage_blob(1, path)
        ours = stage_blob(2, path)
        theirs = stage_blob(3, path)
        if None in (base, ours, theirs):
            print(f"  ! {path}: binary conflict, leaving for manual resolution")
            failed.append(path)
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                merged = fh.read()
        except OSError as exc:
            print(f"  ! {path}: cannot read ({exc})")
            failed.append(path)
            continue

        try:
            resolved = strip_fences(call_anthropic(build_prompt(path, base, ours, theirs, merged)))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
            print(f"  ! {path}: API error ({exc})", file=sys.stderr)
            failed.append(path)
            continue

        if not resolved.strip() or any(m in resolved for m in MARKERS):
            print(f"  ! {path}: model left conflict markers or empty output")
            failed.append(path)
            continue

        if not resolved.endswith("\n"):
            resolved += "\n"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(resolved)
        sh("git", "add", "--", path)
        print(f"  resolved {path}")

    if failed:
        print(f"Unresolved file(s): {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All conflicts resolved and staged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
