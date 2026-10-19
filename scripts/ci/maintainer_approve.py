# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Submit an explicitly delegated approval without merging or bypassing rules.

Security invariant - DO NOT BREAK: run only trusted default-branch code. Treat
the event and comment as data; never check out or execute the pull request head.
GH_TOKEN must identify the dedicated approval App, not the requesting human.
"""

import json
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

REPOSITORY = "newton-physics/newton"
TEAM = "maintainers"
COMMAND = re.compile(r"/maintainer-approve ([0-9a-f]{40}) ([^\r\n]{1,500})")
PROTECTED_FILES = {
    "AGENTS.md",
    "CODING_GUIDELINES.rst",
    "CONTRIBUTING.md",
    "REVIEW_GUIDELINES.rst",
}
PROTECTED_PREFIXES = (".github/", ".agents/", ".claude/", ".codex/", "scripts/")


class Rejected(ValueError):
    """Indicate that an approval request failed a policy check."""


class GitHub:
    """Use the runner's gh CLI without shell evaluation or token arguments."""

    def request(self, method: str, endpoint: str, body: dict | None = None):
        command = [
            "gh",
            "api",
            "--hostname",
            "github.com",
            "--method",
            method,
            "-H",
            "Accept: application/vnd.github+json",
            endpoint,
        ]
        if body is not None:
            command.extend(["--input", "-"])
        result = subprocess.run(
            command,
            input=json.dumps(body) if body is not None else None,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            # Do not print response bodies or credentials on failed API calls.
            raise Rejected(f"GitHub {method} failed; no approval retry was attempted.")
        return json.loads(result.stdout)

    def list(self, endpoint: str) -> list:
        items = []
        for page in range(1, 101):
            batch = self.request("GET", f"{endpoint}?per_page=100&page={page}")
            items.extend(batch)
            if len(batch) < 100:
                return items
        raise Rejected("GitHub pagination limit reached.")


def validate_pull(pull: dict, sha: str, actor_id: int) -> None:
    if pull["state"] != "open" or pull["draft"]:
        raise Rejected("The pull request must be open and ready for review.")
    if pull["user"]["id"] == actor_id:
        raise Rejected("The pull request author cannot request delegated approval.")
    if pull["head"]["sha"] != sha:
        raise Rejected("The requested SHA is stale; post a fresh command for the current head.")
    if pull["base"]["repo"]["full_name"] != REPOSITORY or pull["base"]["ref"] != "main":
        raise Rejected("Only pull requests targeting newton-physics/newton main are supported.")


def approve(event: dict, api: GitHub, *, bot_login: str) -> str:
    """Validate a fresh human command and submit at most one App review."""
    if event.get("action") != "created" or "pull_request" not in event.get("issue", {}):
        raise Rejected("Only newly created pull request comments are supported.")
    if event["repository"]["full_name"] != REPOSITORY:
        raise Rejected("Unexpected repository.")
    comment = event["comment"]
    actor = comment["user"]
    if actor["type"] != "User" or event["sender"]["id"] != actor["id"]:
        raise Rejected("The command must be posted directly by a human.")
    match = COMMAND.fullmatch(comment["body"].strip())
    if match is None or not match[2].strip():
        raise Rejected("Use /maintainer-approve <full lowercase head SHA> <single-line reason, max 500 characters>.")
    sha, reason = match.groups()
    if not bot_login.endswith("[bot]"):
        raise Rejected("The approval identity must be a GitHub App bot.")

    prefix = f"repos/{REPOSITORY}"
    pull_path = f"{prefix}/pulls/{int(event['issue']['number'])}"
    comment_path = f"{prefix}/issues/comments/{int(comment['id'])}"

    def validate_request() -> dict:
        # Re-fetch the comment so deleted/edited comments cannot authorize a run.
        live_comment = api.request("GET", comment_path)
        for field in ("id", "body", "created_at", "updated_at"):
            if live_comment[field] != comment[field]:
                raise Rejected("The command was edited; post a new comment.")
        if live_comment["updated_at"] != live_comment["created_at"]:
            raise Rejected("Edited commands are not accepted.")
        if live_comment["user"]["id"] != actor["id"]:
            raise Rejected("The command author changed.")
        membership = api.request(
            "GET", f"orgs/newton-physics/teams/{TEAM}/memberships/{quote(actor['login'], safe='')}"
        )
        if membership["state"] != "active":
            raise Rejected("Active membership in newton-physics/maintainers is required.")
        pull = api.request("GET", pull_path)
        validate_pull(pull, sha, actor["id"])
        return pull

    pull = validate_request()
    files = api.list(f"{pull_path}/files")
    # GitHub limits this endpoint to 3,000 files. Never silently inspect a subset.
    if len(files) != pull["changed_files"]:
        raise Rejected("Could not inspect the complete changed-file list.")
    for file in files:
        for name in (file["filename"], file.get("previous_filename", "")):
            if name in PROTECTED_FILES or name.startswith(PROTECTED_PREFIXES):
                raise Rejected("Automation and review-policy changes require independent review.")

    marker = f"<!-- maintainer-approval:{comment['id']}:{sha} -->"
    for review in api.list(f"{pull_path}/reviews"):
        if review["user"]["login"] == bot_login and review["user"]["type"] == "Bot":
            if marker in (review.get("body") or ""):
                return "This command was already processed; no review was submitted."
            if review["state"] == "APPROVED" and review["commit_id"] == sha:
                return "The App has already approved this head; no additional review was submitted."

    validate_request()
    # The API has no compare-and-swap head parameter. Bind the review to the SHA,
    # recheck after submission, and rely on native stale/latest-push protections.
    review = api.request(
        "POST",
        f"{pull_path}/reviews",
        {
            "commit_id": sha,
            "event": "APPROVE",
            "body": (
                f"Maintainer exception requested by @{actor['login']} for `{sha}`.\n\n"
                f"Reason (quoted from the command):\n> {reason}\n\n"
                f"Command: https://github.com/{REPOSITORY}/pull/{event['issue']['number']}"
                f"#issuecomment-{comment['id']}\n\n"
                "This is a delegated approval, not an independent review. "
                "Existing review requirements and merge queue checks still apply.\n\n"
                f"{marker}"
            ),
        },
    )
    try:
        validate_request()
    except Exception:
        # Best-effort rollback also covers a membership/comment change or an API
        # failure after submission. Configure dismissal permissions before use.
        api.request(
            "PUT",
            f"{pull_path}/reviews/{int(review['id'])}/dismissals",
            {"message": "Delegated approval invalidated during submission; a fresh command is required."},
        )
        raise
    return f"Delegated approval submitted for {sha}. Add the PR to the normal merge queue."


def main() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") != "issue_comment" or os.environ.get("GITHUB_RUN_ATTEMPT") != "1":
        raise Rejected("Only initial issue_comment runs are accepted; reruns require a new command.")
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise Rejected("Unexpected workflow repository.")
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    result = approve(event, GitHub(), bot_login=os.environ["APP_BOT_LOGIN"])
    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as summary:
        summary.write(result + "\n")


if __name__ == "__main__":
    main()
