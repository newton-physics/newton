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

With ``REVIEW_MODE=follow-up`` (the workflow has reviewed the pull request
before), only P0 and P1 findings are published, and the summary comment keeps
the first review and replaces only its follow-up section. Because the next
follow-up replaces that section, a follow-up's findings without a line are
also posted in the body of its pull request review, which stays. The workflow's
prepare step detects follow-ups from ``SUMMARY_MARKER``, ``FOLLOW_UP_MARKER``,
``INCOMPLETE_MARKER``, and the last ``Reviewed commit <sha>`` line, so keep
these formats in sync with it. Reviewer text is escaped so it cannot forge any
of these markers. After a complete follow-up of the full diff (``FULL_DIFF``),
the first review's incomplete warning is marked as covered, including warnings
posted before ``INCOMPLETE_MARKER`` existed.

Environment: ``REVIEW_OUTPUT``, ``GITHUB_REPOSITORY``, ``PR_NUMBER``,
``HEAD_SHA``, ``RUNNER_TEMP``, ``REVIEW_MODEL_NAME``, ``REVIEW_EFFORT``,
``EXECUTION_FILE``, ``REVIEW_MODE``, and ``FULL_DIFF`` (optional), and the
default ``GITHUB_SERVER_URL``, ``GITHUB_RUN_ID``, and ``GITHUB_RUN_ATTEMPT``.
The diff is read from ``.pr-review/pr.diff``.
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
FOLLOW_UP_MARKER = "<!-- newton-automated-review-follow-up -->"
INCOMPLETE_MARKER = "<!-- newton-automated-review-incomplete -->"
INCOMPLETE_TEXT = "This review did not cover the whole change."
INCOMPLETE_LINE_RE = re.compile(rf"^> {re.escape(INCOMPLETE_TEXT)}(?: {re.escape(INCOMPLETE_MARKER)})?$", re.MULTILINE)
FINDING_MARKER = "<!-- newton-automated-review-finding:{} -->"
FINDING_MARKER_RE = re.compile(r"<!-- newton-automated-review-finding:([0-9a-f]{16}) -->")
BOT_LOGIN = "github-actions[bot]"
MAX_INLINE_COMMENTS = 15
MAX_BODY_CHARS = 6000
MAX_FIRST_REVIEW_CHARS = 30000
REVIEWER_NAME = "Claude Code Action review"
WORKFLOW_PATH = ".github/workflows/claude_code_review.yml"
SEVERITY_LABELS = {"P0": "🔴 P0", "P1": "🟠 P1", "P2": "🔵 P2", "nit": "⚪ Nit"}
FOLLOW_UP_SEVERITIES = ("P0", "P1")
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


def neutralize(text: str) -> str:
    """Break HTML comments and the incomplete warning in reviewer text, so it cannot forge what the workflow parses.

    A zero-width space keeps the text readable, also inside code spans, where an HTML entity would show literally.
    """
    return text.replace("<!--", "<!\u200b--").replace(INCOMPLETE_TEXT, INCOMPLETE_TEXT.replace(" ", " \u200b", 1))


def keep_tail_clip(text: str, limit: int) -> str:
    """Clip a previous review section from the middle, keeping its incomplete-review warning and footer."""
    if len(text) <= limit:
        return text
    starts = [i for i in (text.rfind("\n> [!WARNING]"), text.rfind("\n<sub>Reviewed commit")) if i >= 0]
    split = min(starts) if starts else len(text)
    return clip(text[:split], max(limit - (len(text) - split), 100)) + text[split:]


def severity_label(finding: dict) -> str:
    return SEVERITY_LABELS.get(str(finding.get("severity")), SEVERITY_LABELS["nit"])


def comment_body(finding: dict, with_suggestion: bool = True) -> str:
    """Format a finding as an inline comment body, optionally without its suggestion block."""
    label = severity_label(finding)
    title, body = neutralize(clip(finding.get("title"), 300)), neutralize(clip(finding.get("body")))
    parts = [f"{label[0]} **{label[2:]}: {title}**", "", body]
    suggestion = finding.get("suggestion")
    if with_suggestion and isinstance(suggestion, str) and suggestion.strip() and "```" not in suggestion:
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
        has_range = isinstance(start, int) and start != line
        keep_range = has_range and start < line and all(n in file_lines for n in range(start, line + 1))
        # A suggestion replaces exactly the anchored lines, so drop it if its range cannot be anchored.
        comment = {
            "path": path,
            "line": line,
            "side": "RIGHT",
            "body": comment_body(finding, with_suggestion=keep_range or not has_range),
        }
        if keep_range:
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
    """Turn a model ID such as ``claude-opus-4-8`` into a display name, without any provider prefix or date."""
    match = re.fullmatch(r"claude-([a-z]+)-(\d+)(?:-(\d{1,2}))?(?:-\d{8})?", str(model_id).rsplit("/", 1)[-1])
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


def finding_lines(findings: list[dict], with_marker: bool = False) -> list[str]:
    """Format findings that have no inline comment as list items, optionally with their fingerprint markers."""
    lines = []
    for finding in findings:
        path, line = finding.get("path"), finding.get("line")
        location = neutralize(f"`{path}:{line}`" if path and line else f"`{path}`" if path else "General")
        marker = f" {FINDING_MARKER.format(fingerprint(finding))}" if with_marker else ""
        title, body = neutralize(clip(finding.get("title"), 300)), neutralize(clip(finding.get("body"), 800))
        lines.append(f"- {severity_label(finding)} {location}: **{title}** {body}{marker}")
    return lines


def review_payload(head_sha: str, inline: list[dict], inline_findings: list[dict], listed: list[dict]) -> dict:
    """Build a pull request review with the inline comments and, in its body, the findings in ``listed``."""
    count = count_line(inline_findings)
    body = [f"**{REVIEWER_NAME}**: {count + '. ' if count else ''}The summary comment has the overview."]
    if listed:
        body += ["", "**Not attached to a line:**", "", *finding_lines(listed, with_marker=True)]
    payload = {"commit_id": head_sha, "event": "COMMENT", "body": "\n".join(body)}
    if inline:
        payload["comments"] = inline
    return payload


def summary_body(
    output: dict,
    inline_findings: list[dict],
    elsewhere: list[dict],
    head_sha: str,
    details: list[str],
    previous: str | None = None,
    covered_rest: bool = False,
) -> str:
    """Format the summary comment: the assessment, a count of inline comments, and findings without a line.

    With ``previous``, the body of the existing summary comment, keep its first review and replace its follow-up section.
    With ``covered_rest``, a complete follow-up checked the full diff, so clear the first review's incomplete state.
    """
    summary = re.sub(r"\A(\s*#{1,6}[^\n]*\n)+", "", str(output.get("summary", "")).strip() + "\n").strip()
    summary = neutralize(clip(summary, 3000))
    if previous is None:
        lines = [SUMMARY_MARKER, f"### {REVIEWER_NAME}", "", summary]
    else:
        # Reviewer text is neutralized, so the first marker is the generated section boundary.
        first_review = keep_tail_clip(previous.split(FOLLOW_UP_MARKER, 1)[0].strip(), MAX_FIRST_REVIEW_CHARS)
        if covered_rest:
            first_review = INCOMPLETE_LINE_RE.sub(
                f"> {INCOMPLETE_TEXT} A later follow-up checked the full diff for blocking issues.", first_review
            )
        lines = [first_review, "", FOLLOW_UP_MARKER, "#### Latest follow-up review", "", summary]
    if inline_findings:
        lines += ["", count_line(inline_findings)]
    if elsewhere:
        lines += ["", "**Not attached to a line:**", "", *finding_lines(elsewhere)]
    if not output.get("complete", True):
        lines += [
            "",
            "> [!WARNING]",
            f"> {INCOMPLETE_TEXT} {INCOMPLETE_MARKER}",
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
    follow_up = os.environ.get("REVIEW_MODE") == "follow-up"
    if follow_up:
        # Follow-up reviews only gate merges, so drop anything below P1 even if the reviewer reported it.
        blocking = [f for f in findings if f.get("severity") in FOLLOW_UP_SEVERITIES]
        print(f"Follow-up review: dropped {len(findings) - len(blocking)} finding(s) below P1")
        findings = blocking
        details.append("Follow-up reviews report only blocking (P0/P1) issues.")

    posted: set[str] = set()
    for endpoint in ("comments", "reviews"):
        result = gh_api(
            "--paginate",
            f"repos/{repo}/pulls/{pr}/{endpoint}",
            "--jq",
            f'.[] | select(.user.login == "{BOT_LOGIN}") | .body',
        )
        posted.update(FINDING_MARKER_RE.findall(result.stdout))
    valid_lines = commentable_lines(Path(".pr-review/pr.diff").read_text(encoding="utf-8", errors="replace"))
    inline, inline_findings, elsewhere, skipped = plan_comments(findings, valid_lines, posted)

    # The next follow-up replaces this follow-up's summary section, so also keep its findings without a line
    # in the review body, which stays.
    if inline or (follow_up and elsewhere):
        review = review_payload(head_sha, inline, inline_findings, elsewhere if follow_up else [])
        result = gh_api("--method", "POST", f"repos/{repo}/pulls/{pr}/reviews", payload=review)
        if result.returncode != 0 and inline:
            print(f"::warning::Posting inline comments failed; listing them in the summary: {result.stderr[:500]}")
            elsewhere = [f for f in findings if fingerprint(f) not in posted]
            inline, inline_findings = [], []
            if follow_up:
                result = gh_api(
                    "--method",
                    "POST",
                    f"repos/{repo}/pulls/{pr}/reviews",
                    payload=review_payload(head_sha, [], [], elsewhere),
                )
        if result.returncode != 0:
            print(f"::warning::Posting the review failed: {result.stderr[:500]}")
    print(f"Inline comments: {len(inline)}; summary-only findings: {len(elsewhere)}; already posted: {skipped}")

    result = gh_api(
        "--paginate",
        f"repos/{repo}/issues/{pr}/comments",
        "--jq",
        f'.[] | select(.user.login == "{BOT_LOGIN}" and (.body | startswith("{SUMMARY_MARKER}"))) | {{id, body}}',
    )
    comments = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    previous = comments[-1]["body"] if comments and follow_up else None
    covered_rest = os.environ.get("FULL_DIFF") == "true" and output.get("complete", True) is True
    summary_path = Path(os.environ["RUNNER_TEMP"]) / "review-summary.md"
    summary_path.write_text(
        summary_body(output, inline_findings, elsewhere, head_sha, details, previous, covered_rest), encoding="utf-8"
    )
    if comments:
        result = gh_api(
            "--method", "PATCH", f"repos/{repo}/issues/comments/{comments[-1]['id']}", "-F", f"body=@{summary_path}"
        )
    else:
        result = gh_api("--method", "POST", f"repos/{repo}/issues/{pr}/comments", "-F", f"body=@{summary_path}")
    if result.returncode != 0:
        print(f"Publishing the summary failed: {result.stderr[:500]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
