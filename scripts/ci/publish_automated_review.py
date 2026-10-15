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

Both name the reviewer and link the workflow run, and the summary states the
model and effort level, including any fallback model recorded in the run's
execution file. Only display names are published, never model IDs.

Environment: ``REVIEW_OUTPUT``, ``GITHUB_REPOSITORY``, ``PR_NUMBER``,
``HEAD_SHA``, ``RUNNER_TEMP``, ``REVIEW_MODEL_NAME``, ``REVIEW_EFFORT``,
``EXECUTION_FILE`` (optional), and the default ``GITHUB_SERVER_URL``,
``GITHUB_RUN_ID``, and ``GITHUB_RUN_ATTEMPT``. The diff is read from
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
REVIEWER_NAME = "Claude Code Action review"
WORKFLOW_PATH = ".github/workflows/claude_code_review.yml"
SEVERITY_LABELS = {"P0": "🔴 P0", "P1": "🟠 P1", "P2": "🔵 P2", "nit": "⚪ Nit"}
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


def severity_label(finding: dict) -> str:
    return SEVERITY_LABELS.get(str(finding.get("severity")), SEVERITY_LABELS["nit"])


def comment_body(finding: dict) -> str:
    """Format a finding as an inline comment body."""
    label = severity_label(finding)
    parts = [f"{label[0]} **{label[2:]}: {clip(finding.get('title'), 300)}**", "", clip(finding.get("body"))]
    suggestion = finding.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip() and "```" not in suggestion:
        parts += ["", "```suggestion", suggestion.rstrip("\n"), "```"]
    parts += ["", FINDING_MARKER.format(fingerprint(finding))]
    return "\n".join(parts)


def plan_comments(
    findings: list[dict], valid_lines: dict[str, set[int]], posted: set[str]
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Split findings into inline comments, the findings they come from, findings for the summary, and the number already posted."""
    inline: list[dict] = []
    inline_findings: list[dict] = []
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
        inline_findings.append(finding)
    return inline, inline_findings, elsewhere, skipped


def count_line(inline_findings: list[dict]) -> str:
    """Summarize the inline comments by priority, for example "3 inline comments: 🔵 1 P2 · ⚪ 2 Nit"."""
    if not inline_findings:
        return ""
    counts = dict.fromkeys(SEVERITY_LABELS.values(), 0)
    for finding in inline_findings:
        counts[severity_label(finding)] += 1
    parts = [f"{label[0]} {n} {label[2:]}" for label, n in counts.items() if n]
    noun = "comment" if len(inline_findings) == 1 else "comments"
    return f"{len(inline_findings)} inline {noun}: " + " · ".join(parts)


def display_name(model_id: object) -> str:
    """Turn a model ID such as ``claude-opus-4-8`` into a display name, without any provider prefix."""
    match = re.fullmatch(r"claude-([a-z]+)-(\d+)(?:-(\d+))?", str(model_id).rsplit("/", 1)[-1])
    if not match:
        return "another model"
    family, major, minor = match.groups()
    return f"Claude {family.capitalize()} {major}{'.' + minor if minor else ''}"


def fallback_notes(execution_file: str | None, model_name: str) -> list[str]:
    """Describe any model fallback recorded in Claude Code's execution file."""
    try:
        messages = json.loads(Path(execution_file).read_text(encoding="utf-8")) if execution_file else []
    except (OSError, ValueError):
        return []
    notes = []
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict) and message.get("type") == "system":
            fallback = display_name(message.get("fallback_model"))
            if message.get("subtype") == "model_refusal_fallback":
                notes.append(f"{model_name} declined a response, so Claude Code continued with {fallback}.")
            elif message.get("subtype") == "model_fallback":
                notes.append(f"{model_name} was unavailable, so Claude Code continued with {fallback}.")
    return notes


def run_details(execution_file: str | None) -> list[str]:
    """Describe the reviewer, workflow run, model, and effort level without revealing model IDs."""
    repo = os.environ["GITHUB_REPOSITORY"]
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    run_url = f"{server}/{repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    if attempt not in ("", "1"):
        run_url += f"/attempts/{attempt}"
    model_name = os.environ.get("REVIEW_MODEL_NAME", "Claude")
    lines = [
        f"Produced by [`{Path(WORKFLOW_PATH).name}`]({server}/{repo}/blob/main/{WORKFLOW_PATH}) in "
        f"[this workflow run]({run_url}) with {model_name} at effort `{os.environ.get('REVIEW_EFFORT', 'default')}`."
    ]
    return lines + fallback_notes(execution_file, model_name)


def summary_body(
    output: dict, inline_findings: list[dict], elsewhere: list[dict], head_sha: str, details: list[str]
) -> str:
    """Format the summary comment: the assessment, a count of inline comments, and findings without a line."""
    summary = re.sub(r"\A(\s*#{1,6}[^\n]*\n)+", "", str(output.get("summary", "")).strip() + "\n").strip()
    lines = [SUMMARY_MARKER, f"### {REVIEWER_NAME}", "", clip(summary, 3000)]
    if inline_findings:
        lines += ["", count_line(inline_findings)]
    if elsewhere:
        lines += ["", "**Not attached to a line:**", ""]
        for finding in elsewhere:
            path, line = finding.get("path"), finding.get("line")
            location = f"`{path}:{line}`" if path and line else f"`{path}`" if path else "General"
            lines.append(
                f"- {severity_label(finding)} {location}: **{clip(finding.get('title'), 300)}** "
                f"{clip(finding.get('body'), 800)}"
            )
    if not output.get("complete", True):
        lines += [
            "",
            "> [!WARNING]",
            "> This review did not cover the whole change.",
        ]
    lines += ["", f"<sub>Reviewed commit {head_sha}. {' '.join(details)}</sub>"]
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
    details = run_details(os.environ.get("EXECUTION_FILE"))

    result = gh_api(
        "--paginate", f"repos/{repo}/pulls/{pr}/comments", "--jq", f'.[] | select(.user.login == "{BOT_LOGIN}") | .body'
    )
    posted = set(FINDING_MARKER_RE.findall(result.stdout))
    valid_lines = commentable_lines(Path(".pr-review/pr.diff").read_text(encoding="utf-8", errors="replace"))
    inline, inline_findings, elsewhere, skipped = plan_comments(findings, valid_lines, posted)

    if inline:
        review = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": f"**{REVIEWER_NAME}**: {count_line(inline_findings)}. The summary comment has the overview.",
            "comments": inline,
        }
        result = gh_api("--method", "POST", f"repos/{repo}/pulls/{pr}/reviews", payload=review)
        if result.returncode != 0:
            print(f"::warning::Posting inline comments failed; listing them in the summary: {result.stderr[:500]}")
            elsewhere = [f for f in findings if fingerprint(f) not in posted]
            inline, inline_findings = [], []
    print(f"Inline comments: {len(inline)}; summary-only findings: {len(elsewhere)}; already posted: {skipped}")

    summary_path = Path(os.environ["RUNNER_TEMP"]) / "review-summary.md"
    summary_path.write_text(summary_body(output, inline_findings, elsewhere, head_sha, details), encoding="utf-8")
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
