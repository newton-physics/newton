#!/usr/bin/env python3
"""AI-assisted git merge conflict resolver.

Resolves conflicts left by an in-progress ``git merge`` by asking an LLM to
produce a merged version of each conflicted text file, using full three-way
context (base / ours / theirs). Resolved files are written back and staged.
Binary conflicts are never touched. Intended for CI use by the ``Sync
Upstream`` workflow; run it while the merge is still in progress.

Providers (env ``AI_RESOLVE_PROVIDER``):
    anthropic       Anthropic Messages API (default). Key: ANTHROPIC_API_KEY.
    github-models   GitHub Models, OpenAI-compatible. Key: GITHUB_TOKEN
                    (needs ``models: read``) or AI_RESOLVE_API_KEY.
    openai          Any OpenAI-compatible endpoint. Key: OPENAI_API_KEY
                    or AI_RESOLVE_API_KEY.

Other env:
    AI_RESOLVE_MODEL       Model id (falls back to a per-provider default).
    AI_RESOLVE_BASE_URL    Override the OpenAI-compatible base URL.
    AI_RESOLVE_MAX_TOKENS  Max output tokens per file (default 32000).

Exit codes:
    0  all conflicts resolved and staged (or nothing to resolve)
    1  one or more files could not be resolved (left for a human)
    2  misconfiguration (missing key / unknown provider)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

PROVIDER = (os.environ.get("AI_RESOLVE_PROVIDER") or "anthropic").lower()
MAX_TOKENS = int(os.environ.get("AI_RESOLVE_MAX_TOKENS", "32000"))
MARKERS = ("<<<<<<<", ">>>>>>>")

_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-4-5",
    "github-models": "openai/gpt-4.1",
    "openai": "gpt-4.1",
}
_DEFAULT_BASE_URL = {
    "github-models": "https://models.github.ai/inference",
    "openai": "https://api.openai.com/v1",
}
MODEL = (
    os.environ.get("AI_RESOLVE_MODEL")
    or os.environ.get("ANTHROPIC_MODEL")
    or _DEFAULT_MODEL.get(PROVIDER, "claude-sonnet-4-5")
)


def sh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=False, capture_output=True, text=True)


def conflicted_files() -> list[str]:
    out = sh("git", "diff", "--name-only", "--diff-filter=U").stdout
    return [line for line in out.splitlines() if line.strip()]


def stage_blob(stage: int, path: str) -> str | None:
    """Return a merge stage (1=base, 2=ours, 3=theirs) as text, or None if binary."""
    r = subprocess.run(["git", "show", f":{stage}:{path}"], capture_output=True)
    if r.returncode != 0:
        return ""  # side added/removed the file; empty is the right context
    if b"\x00" in r.stdout:
        return None
    return r.stdout.decode("utf-8", errors="replace")


def strip_fences(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(lines)


def _post(url: str, headers: dict, body: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _resolve_key() -> str | None:
    if PROVIDER == "anthropic":
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AI_RESOLVE_API_KEY")
    if PROVIDER == "github-models":
        return os.environ.get("AI_RESOLVE_API_KEY") or os.environ.get("GITHUB_TOKEN")
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("AI_RESOLVE_API_KEY")


def call_llm(prompt: str) -> str:
    key = _resolve_key()
    if PROVIDER == "anthropic":
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": [{"role": "user", "content": prompt}]},
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    base = (os.environ.get("AI_RESOLVE_BASE_URL") or _DEFAULT_BASE_URL[PROVIDER]).rstrip("/")
    data = _post(
        f"{base}/chat/completions",
        {"authorization": f"Bearer {key}"},
        {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    return data["choices"][0]["message"]["content"]


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
    if PROVIDER not in ("anthropic", "github-models", "openai"):
        print(f"ERROR: unknown AI_RESOLVE_PROVIDER '{PROVIDER}'.", file=sys.stderr)
        return 2
    if not _resolve_key():
        print(f"ERROR: no API key/token for provider '{PROVIDER}'.", file=sys.stderr)
        return 2

    files = conflicted_files()
    if not files:
        print("No conflicted files to resolve.")
        return 0

    print(f"Resolving {len(files)} conflicted file(s) via '{PROVIDER}' model '{MODEL}'.")
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
            resolved = strip_fences(call_llm(build_prompt(path, base, ours, theirs, merged)))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError) as exc:
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
