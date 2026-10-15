# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Publish the automated pull request review from its structured output.

Security invariant - DO NOT BREAK:
    This script runs under ``pull_request_target`` from the trusted base
    checkout. The review output and the diff are untrusted data: never
    execute or evaluate them, and only pass them to ``gh`` as JSON or files,
    never as shell text.

The reviewer returns ``{"summary", "complete", "findings"}``. Findings anchored
to lines in the diff are posted as one pull request review with inline
comments; the rest are listed in a summary comment that is updated on each
run. Findings already posted on earlier runs are skipped.

Environment: ``REVIEW_OUTPUT``, ``GITHUB_REPOSITORY``, ``PR_NUMBER``,
``HEAD_SHA``, and ``RUNNER_TEMP``. The diff is read from
``.pr-review/pr.diff``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SUMMARY_MARKER = "<!-- newton-automated-review -->"
FINDING_MARKER = "<!-- newton-automated-review-finding:{} -->"
FINDING_MARKER_RE = re.compile(r"newton-automated-review-finding:([0-9a-f]{16})")
BOT_LOGIN = "github-actions[bot]"
MAX_INLINE_COMMENTS = 15
MAX_BODY_CHARS = 6000
SEVERITY_ICONS = {"high": "🔴", "medium": "🟠", "low": "🔵", "nit": "⚪"}
HUNK_RE = re.compile(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def commentable_lines(diff_text: str) -> dict[str, set[int]]:
    """Return the head-side line numbers of each file that appear in the diff hunks."""
    lines: dict[str, set[int]] = {}
    path = None
    in_header = False
    line = 0
    for raw in diff_text.splitlines():
        if raw.startswith("diff --git "):
            path, in_header = None, True
        elif in_header and raw.startswith("+++ "):
            target = raw[4:]
            path = target[2:] if target.startswith("b/") else None
            if path is not None:
                lines.setdefault(path, set())
        elif raw.startswith("@@"):
            in_header = False
            match = HUNK_RE.match(raw)
            line = int(match.group(1)) if match else 0
        elif in_header:
            continue
        elif path is not None and raw[:1] in ("+", " "):
            lines[path].add(line)
            line += 1
    return lines


def fingerprint(finding: dict) -> str:
    """Identify a finding across runs by its file and normalized title."""
    title = " ".join(str(finding.get("title", "")).lower().split())
    return hashlib.sha256(f"{finding.get('path', '')}|{title}".encode()).hexdigest()[:16]


def clip(text: object, limit: int = MAX_BODY_CHARS) -> str:
    """Strip and truncate text to ``limit`` characters."""
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 20].rstrip() + "\n\n*(truncated)*"


def icon(finding: dict) -> str:
    return SEVERITY_ICONS.get(str(finding.get("severity")), SEVERITY_ICONS["nit"])


def comment_body(finding: dict) -> str:
    """Format a finding as an inline comment body."""
    parts = [f"{icon(finding)} **{clip(finding.get('title'), 300)}**", "", clip(finding.get("body"))]
    suggestion = finding.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip() and "```" not in suggestion:
        parts += ["", "```suggestion", suggestion.rstrip("\n"), "```"]
    parts += ["", FINDING_MARKER.format(fingerprint(finding))]
    return "\n".join(parts)


def plan_comments(
    findings: list[dict], valid_lines: dict[str, set[int]], posted: set[str]
) -> tuple[list[dict], list[dict], int]:
    """Split findings into inline comments, findings for the summary, and the number already posted."""
    inline: list[dict] = []
    elsewhere: list[dict] = []
    skipped = 0
    for finding in findings:
        if fingerprint(finding) in posted:
            skipped += 1
            continue
        path, line, start = finding.get("path"), finding.get("line"), finding.get("start_line")
        file_lines = valid_lines.get(path, set())
        if len(inline) >= MAX_INLINE_COMMENTS or not isinstance(line, int) or line not in file_lines:
            elsewhere.append(finding)
            continue
        comment = {"path": path, "line": line, "side": "RIGHT", "body": comment_body(finding)}
        if isinstance(start, int) and start < line and all(n in file_lines for n in range(start, line + 1)):
            comment.update(start_line=start, start_side="RIGHT")
        inline.append(comment)
    return inline, elsewhere, skipped


def summary_body(output: dict, elsewhere: list[dict], head_sha: str) -> str:
    """Format the summary comment."""
    summary = re.sub(r"^\s*#+\s*automated review\s*\n", "", str(output.get("summary", "")).strip(), flags=re.IGNORECASE)
    lines = [SUMMARY_MARKER, "### Automated review", "", clip(summary, 20000)]
    if elsewhere:
        lines += ["", "**Findings without an inline comment**", ""]
        for finding in elsewhere:
            path, line = finding.get("path"), finding.get("line")
            location = f"`{path}:{line}`" if path and line else f"`{path}`" if path else "General"
            lines.append(
                f"- {icon(finding)} {location}: **{clip(finding.get('title'), 300)}** {clip(finding.get('body'), 1500)}"
            )
    if not output.get("complete", True):
        lines += [
            "",
            "> [!WARNING]",
            "> This automated review is incomplete; see above for the parts it did not cover.",
        ]
    lines += ["", f"<sub>Reviewed commit {head_sha}.</sub>"]
    return "\n".join(lines) + "\n"


def gh_api(*args: str, payload: dict | None = None) -> subprocess.CompletedProcess:
    command = ["gh", "api", *args]
    if payload is not None:
        command += ["--input", "-"]
    data = json.dumps(payload) if payload is not None else None
    return subprocess.run(command, input=data, capture_output=True, text=True, check=False)


def main() -> int:
    output = json.loads(os.environ["REVIEW_OUTPUT"])
    repo = os.environ["GITHUB_REPOSITORY"]
    pr = os.environ["PR_NUMBER"]
    head_sha = os.environ["HEAD_SHA"]
    findings = [f for f in output.get("findings", []) if isinstance(f, dict)]

    result = gh_api(
        "--paginate", f"repos/{repo}/pulls/{pr}/comments", "--jq", f'.[] | select(.user.login == "{BOT_LOGIN}") | .body'
    )
    posted = set(FINDING_MARKER_RE.findall(result.stdout))
    valid_lines = commentable_lines(Path(".pr-review/pr.diff").read_text(encoding="utf-8", errors="replace"))
    inline, elsewhere, skipped = plan_comments(findings, valid_lines, posted)

    if inline:
        review = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": f"Automated review: {len(inline)} inline comment(s). The summary comment has the overview.",
            "comments": inline,
        }
        result = gh_api("--method", "POST", f"repos/{repo}/pulls/{pr}/reviews", payload=review)
        if result.returncode != 0:
            print(f"::warning::Posting inline comments failed; listing them in the summary: {result.stderr[:500]}")
            elsewhere = [f for f in findings if fingerprint(f) not in posted]
            inline = []
    print(f"Inline comments: {len(inline)}; summary-only findings: {len(elsewhere)}; already posted: {skipped}")

    summary_path = Path(os.environ["RUNNER_TEMP"]) / "review-summary.md"
    summary_path.write_text(summary_body(output, elsewhere, head_sha), encoding="utf-8")
    result = gh_api(
        "--paginate",
        f"repos/{repo}/issues/{pr}/comments",
        "--jq",
        f'.[] | select(.user.login == "{BOT_LOGIN}" and (.body | startswith("{SUMMARY_MARKER}"))) | .id',
    )
    comment_ids = result.stdout.split()
    if comment_ids:
        result = gh_api(
            "--method", "PATCH", f"repos/{repo}/issues/comments/{comment_ids[-1]}", "-F", f"body=@{summary_path}"
        )
    else:
        result = gh_api("--method", "POST", f"repos/{repo}/issues/{pr}/comments", "-F", f"body=@{summary_path}")
    if result.returncode != 0:
        print(f"Publishing the summary failed: {result.stderr[:500]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
