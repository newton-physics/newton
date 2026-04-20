#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nightly upstream merge helper for personal Newton research branches.

This script keeps a fork aligned with ``newton-physics/newton`` without
touching the currently checked-out branch in the user's main worktree.
It operates in disposable git worktrees created from ``origin/<branch>``,
merges ``upstream/main`` into each active branch, optionally runs validation
commands, and only pushes after those steps succeed.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Sequence


DEFAULT_VALIDATION_BRANCH = "research/pressure-field"
DEFAULT_TEST_COMMANDS = (
    "uv run --extra dev -m newton.tests -k pressure",
    "uv run --extra examples -m newton.examples.contacts.example_hydro_pressure_slice --test --viewer null --num-frames 1 --shape box",
)
BRANCH_VALIDATION_COMMANDS = {
    "research/pressure-field": DEFAULT_TEST_COMMANDS,
    "protomotions": DEFAULT_TEST_COMMANDS,
}
IGNORED_REMOTE_BRANCH_PATTERNS = (
    "-pre-upstream-",
    "-pre-upstream-rebase-",
)


class CommandError(RuntimeError):
    """Raised when a subprocess fails."""

    def __init__(self, message: str, *, output: str = "", returncode: int | None = None):
        super().__init__(message)
        self.output = output
        self.returncode = returncode


@dataclasses.dataclass
class MainSyncResult:
    status: str
    ahead_of_upstream: int
    behind_upstream: int
    message: str


@dataclasses.dataclass
class BranchResult:
    branch: str
    status: str
    start_sha: str
    end_sha: str
    merged: bool
    pushed: bool
    tests_ran: bool
    message: str
    log_path: pathlib.Path


def summarize_message(message: str, *, limit: int = 180) -> str:
    """Collapse multiline command output into a short report-friendly sentence."""
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def automation_dir(root: pathlib.Path) -> pathlib.Path:
    return root / ".codex" / "automations" / "nightly-upstream-merge"


def failure_queue_path(root: pathlib.Path) -> pathlib.Path:
    return automation_dir(root) / "failures.json"


def load_failure_queue(root: pathlib.Path) -> dict:
    path = failure_queue_path(root)
    if not path.exists():
        return {"version": 1, "updated_at": None, "items": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_failure_queue(root: pathlib.Path, queue: dict) -> None:
    path = failure_queue_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    queue["version"] = 1
    queue["updated_at"] = now_iso()
    path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def result_needs_repair(result: BranchResult) -> bool:
    return result.status in {"missing", "merge_conflict", "tests_failed", "push_failed"}


def branch_validation_commands(branch: str, validation_branch: str | None, test_commands: Sequence[str]) -> list[str]:
    if branch in BRANCH_VALIDATION_COMMANDS:
        return list(BRANCH_VALIDATION_COMMANDS[branch])
    if validation_branch and branch == validation_branch:
        return list(test_commands)
    return []


def update_failure_queue(
    root: pathlib.Path,
    *,
    selected_branches: Sequence[str],
    validation_branch: str | None,
    test_commands: Sequence[str],
    branch_results: Sequence[BranchResult],
    report_path: pathlib.Path,
) -> None:
    queue = load_failure_queue(root)
    existing_by_branch = {item["branch"]: item for item in queue.get("items", []) if "branch" in item}
    next_items: list[dict] = []
    touched = set(selected_branches)

    for result in branch_results:
        if not result_needs_repair(result):
            continue
        previous = existing_by_branch.get(result.branch, {})
        first_seen_at = previous.get("first_seen_at", now_iso())
        attempt_count = int(previous.get("attempt_count", 0))
        next_items.append(
            {
                "attempt_count": attempt_count,
                "branch": result.branch,
                "end_sha": result.end_sha,
                "failure_type": result.status,
                "first_seen_at": first_seen_at,
                "last_attempted_at": previous.get("last_attempted_at"),
                "last_seen_at": now_iso(),
                "last_summary": summarize_message(result.message, limit=400),
                "log_path": str(result.log_path.relative_to(root)),
                "repair_status": "queued",
                "report_path": str(report_path.relative_to(root)),
                "start_sha": result.start_sha,
                "validation_commands": branch_validation_commands(
                    result.branch,
                    validation_branch,
                    test_commands,
                ),
            }
        )

    for branch, item in existing_by_branch.items():
        if branch in touched:
            continue
        next_items.append(item)

    queue["items"] = sorted(next_items, key=lambda item: item["branch"])
    save_failure_queue(root, queue)


def repo_root() -> pathlib.Path:
    output = git_capture(("git", "rev-parse", "--show-toplevel"))
    return pathlib.Path(output.strip())


def git_capture(
    command: Sequence[str],
    *,
    cwd: pathlib.Path | None = None,
    check: bool = True,
) -> str:
    """Run a command and return combined stdout/stderr."""
    process = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (process.stdout + process.stderr).strip()
    if check and process.returncode != 0:
        raise CommandError(
            f"{shlex.join(command)} failed with exit code {process.returncode}.",
            output=output,
            returncode=process.returncode,
        )
    return output


def run_logged_command(
    command: Sequence[str],
    *,
    cwd: pathlib.Path,
    log_file: pathlib.Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, mirror the command line into the log, and append output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        process = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
        if process.stdout:
            handle.write(process.stdout)
        if process.stderr:
            handle.write(process.stderr)
        if process.stdout and not process.stdout.endswith("\n"):
            handle.write("\n")
        if process.stderr and not process.stderr.endswith("\n"):
            handle.write("\n")
        handle.write(f"[exit {process.returncode}]\n\n")
    if check and process.returncode != 0:
        output = ((process.stdout or "") + (process.stderr or "")).strip()
        raise CommandError(
            f"{shlex.join(command)} failed with exit code {process.returncode}.",
            output=output,
            returncode=process.returncode,
        )
    return process


def git_ref_exists(root: pathlib.Path, ref: str) -> bool:
    process = subprocess.run(
        ("git", "show-ref", "--verify", "--quiet", ref),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode == 0


def list_origin_branches(root: pathlib.Path) -> list[str]:
    output = git_capture(
        ("git", "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin"),
        cwd=root,
    )
    branches: list[str] = []
    for ref in output.splitlines():
        if ref == "origin/HEAD":
            continue
        if not ref.startswith("origin/"):
            continue
        branch = ref.removeprefix("origin/")
        if branch in {"main", "gh-pages"}:
            continue
        if any(token in branch for token in IGNORED_REMOTE_BRANCH_PATTERNS):
            continue
        branches.append(branch)
    return sorted(branches)


def reorder_branches(branches: Sequence[str], validation_branch: str | None) -> list[str]:
    ordered = list(dict.fromkeys(branches))
    if validation_branch and validation_branch in ordered:
        ordered.remove(validation_branch)
        ordered.append(validation_branch)
    return ordered


def branch_contains_upstream(root: pathlib.Path, branch: str) -> bool:
    process = subprocess.run(
        ("git", "merge-base", "--is-ancestor", "refs/remotes/upstream/main", f"refs/remotes/origin/{branch}"),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode == 0


def sync_main(root: pathlib.Path, push: bool) -> MainSyncResult:
    output = git_capture(
        ("git", "rev-list", "--left-right", "--count", "refs/remotes/origin/main...refs/remotes/upstream/main"),
        cwd=root,
    )
    ahead_text, behind_text = output.split()
    ahead = int(ahead_text)
    behind = int(behind_text)

    if ahead > 0:
        return MainSyncResult(
            status="blocked",
            ahead_of_upstream=ahead,
            behind_upstream=behind,
            message=(
                "origin/main has fork-only commits and cannot be fast-forwarded "
                "to upstream/main safely."
            ),
        )

    if behind == 0:
        return MainSyncResult(
            status="up_to_date",
            ahead_of_upstream=0,
            behind_upstream=0,
            message="origin/main already matches upstream/main.",
        )

    if not push:
        return MainSyncResult(
            status="would_update",
            ahead_of_upstream=0,
            behind_upstream=behind,
            message=f"origin/main is behind upstream/main by {behind} commit(s).",
        )

    git_capture(("git", "push", "origin", "refs/remotes/upstream/main:refs/heads/main"), cwd=root)
    return MainSyncResult(
        status="updated",
        ahead_of_upstream=0,
        behind_upstream=behind,
        message=f"Fast-forwarded origin/main by {behind} commit(s).",
    )


def create_worktree(root: pathlib.Path, worktree_dir: pathlib.Path, branch: str) -> None:
    git_capture(("git", "worktree", "add", "--detach", str(worktree_dir), f"refs/remotes/origin/{branch}"), cwd=root)


def remove_worktree(root: pathlib.Path, worktree_dir: pathlib.Path) -> None:
    subprocess.run(
        ("git", "worktree", "remove", "--force", str(worktree_dir)),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    subprocess.run(
        ("git", "worktree", "prune"),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )


def run_test_commands(
    branch: str,
    *,
    cwd: pathlib.Path,
    log_path: pathlib.Path,
    validation_branch: str | None,
    test_commands: Sequence[str],
) -> bool:
    if branch != validation_branch or not test_commands:
        return False
    for command_text in test_commands:
        command = tuple(shlex.split(command_text))
        run_logged_command(command, cwd=cwd, log_file=log_path)
    return True


def process_branch(
    root: pathlib.Path,
    branch: str,
    *,
    push: bool,
    validation_branch: str | None,
    test_commands: Sequence[str],
    worktree_parent: pathlib.Path,
    log_dir: pathlib.Path,
    stamp: str,
) -> BranchResult:
    safe_branch = branch.replace("/", "__")
    worktree_dir = worktree_parent / f"{stamp}-{safe_branch}"
    log_path = log_dir / f"{stamp}-{safe_branch}.log"
    start_sha = git_capture(("git", "rev-parse", f"refs/remotes/origin/{branch}"), cwd=root).strip()
    tests_ran = False

    create_worktree(root, worktree_dir, branch)
    try:
        temp_branch = f"nightly-sync/{safe_branch}/{stamp}"
        run_logged_command(("git", "switch", "-c", temp_branch), cwd=worktree_dir, log_file=log_path)

        merged = False
        if not branch_contains_upstream(root, branch):
            try:
                run_logged_command(
                    ("git", "merge", "--no-edit", "--no-ff", "refs/remotes/upstream/main"),
                    cwd=worktree_dir,
                    log_file=log_path,
                )
                merged = True
            except CommandError as error:
                subprocess.run(
                    ("git", "merge", "--abort"),
                    cwd=str(worktree_dir),
                    text=True,
                    capture_output=True,
                    check=False,
                )
                return BranchResult(
                    branch=branch,
                    status="merge_conflict",
                    start_sha=start_sha,
                    end_sha=start_sha,
                    merged=False,
                    pushed=False,
                    tests_ran=False,
                    message=error.output or "Merge conflict while applying upstream/main.",
                    log_path=log_path,
                )

        try:
            tests_ran = run_test_commands(
                branch,
                cwd=worktree_dir,
                log_path=log_path,
                validation_branch=validation_branch,
                test_commands=test_commands,
            )
        except CommandError as error:
            end_sha = git_capture(("git", "rev-parse", "HEAD"), cwd=worktree_dir).strip()
            return BranchResult(
                branch=branch,
                status="tests_failed",
                start_sha=start_sha,
                end_sha=end_sha,
                merged=merged,
                pushed=False,
                tests_ran=True,
                message=error.output or "Validation command failed.",
                log_path=log_path,
            )

        pushed = False
        if merged and push:
            try:
                run_logged_command(
                    ("git", "push", "origin", f"HEAD:refs/heads/{branch}"),
                    cwd=worktree_dir,
                    log_file=log_path,
                )
                pushed = True
            except CommandError as error:
                end_sha = git_capture(("git", "rev-parse", "HEAD"), cwd=worktree_dir).strip()
                return BranchResult(
                    branch=branch,
                    status="push_failed",
                    start_sha=start_sha,
                    end_sha=end_sha,
                    merged=merged,
                    pushed=False,
                    tests_ran=tests_ran,
                    message=error.output or "Push to origin failed.",
                    log_path=log_path,
                )

        end_sha = git_capture(("git", "rev-parse", "HEAD"), cwd=worktree_dir).strip()
        if merged and pushed:
            status = "merged_and_pushed"
            message = "Merged upstream/main and pushed the updated branch."
        elif merged:
            status = "merged_not_pushed"
            message = "Merged upstream/main in the temp worktree; rerun with --push to publish it."
        elif tests_ran:
            status = "validated"
            message = "Branch already contained upstream/main; validation still passed."
        else:
            status = "up_to_date"
            message = "Branch already contained upstream/main."
        return BranchResult(
            branch=branch,
            status=status,
            start_sha=start_sha,
            end_sha=end_sha,
            merged=merged,
            pushed=pushed,
            tests_ran=tests_ran,
            message=message,
            log_path=log_path,
        )
    finally:
        remove_worktree(root, worktree_dir)


def write_report(
    report_path: pathlib.Path,
    *,
    root: pathlib.Path,
    push: bool,
    selected_branches: Sequence[str],
    validation_branch: str | None,
    test_commands: Sequence[str],
    main_result: MainSyncResult,
    branch_results: Sequence[BranchResult],
) -> None:
    timestamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Nightly Upstream Merge Report",
        "",
        f"- Timestamp: {timestamp}",
        f"- Repository: {root}",
        f"- Push enabled: {push}",
        f"- Branches: {', '.join(selected_branches) if selected_branches else '(none)'}",
        f"- Validation branch: {validation_branch or '(none)'}",
        "",
        "## Main Sync",
        "",
        f"- Status: {main_result.status}",
        f"- Ahead of upstream: {main_result.ahead_of_upstream}",
        f"- Behind upstream: {main_result.behind_upstream}",
        f"- Summary: {main_result.message}",
        "",
        "## Validation Commands",
        "",
    ]
    if test_commands:
        lines.extend(f"- `{command}`" for command in test_commands)
    else:
        lines.append("- (none)")
    lines.extend(["", "## Branch Results", ""])
    if branch_results:
        lines.append("| Branch | Status | Merged | Pushed | Tests | Start | End | Log |")
        lines.append("|---|---|---:|---:|---:|---|---|---|")
        for result in branch_results:
            rel_log = result.log_path.relative_to(root)
            lines.append(
                "| "
                f"`{result.branch}` | {result.status} | "
                f"{'yes' if result.merged else 'no'} | "
                f"{'yes' if result.pushed else 'no'} | "
                f"{'yes' if result.tests_ran else 'no'} | "
                f"`{result.start_sha[:12]}` | `{result.end_sha[:12]}` | `{rel_log}` |"
            )
            lines.append(f"| Summary | {summarize_message(result.message).replace('|', '/')} |  |  |  |  |  |  |")
    else:
        lines.append("- No active origin branches matched the selection.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge upstream/main into active origin branches in disposable worktrees.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""\
            Examples:
              uv run --script scripts/nightly_upstream_merge.py
              uv run --script scripts/nightly_upstream_merge.py --push
              uv run --script scripts/nightly_upstream_merge.py --branch protomotions --branch {DEFAULT_VALIDATION_BRANCH}
            """
        ),
    )
    parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        default=[],
        help="Origin branch to process. Repeat to sync an explicit branch set.",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push successful branch merges back to origin and fast-forward origin/main when safe.",
    )
    parser.add_argument(
        "--validation-branch",
        default=DEFAULT_VALIDATION_BRANCH,
        help="Branch that must run validation commands, even when already up to date.",
    )
    parser.add_argument(
        "--test-command",
        action="append",
        default=[],
        help="Validation command for the validation branch. Repeat to run multiple commands.",
    )
    parser.add_argument(
        "--skip-main-sync",
        action="store_true",
        help="Skip the origin/main fast-forward check and optional push.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    root = repo_root()

    explicit_branches = list(dict.fromkeys(args.branches))
    selected_branches = explicit_branches or list_origin_branches(root)
    selected_branches = reorder_branches(selected_branches, args.validation_branch)
    test_commands = tuple(args.test_command) if args.test_command else DEFAULT_TEST_COMMANDS

    git_capture(("git", "fetch", "origin", "--prune"), cwd=root)
    git_capture(("git", "fetch", "upstream", "--prune"), cwd=root)

    automation_dir = root / ".codex" / "automations" / "nightly-upstream-merge"
    report_dir = automation_dir / "reports"
    log_dir = report_dir / "logs"
    worktree_dir = automation_dir / "worktrees"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    worktree_dir.mkdir(parents=True, exist_ok=True)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    report_path = report_dir / f"{stamp}.md"

    if not git_ref_exists(root, "refs/remotes/upstream/main"):
        raise CommandError("Missing refs/remotes/upstream/main. Add the upstream remote first.")

    if not git_ref_exists(root, "refs/remotes/origin/main"):
        raise CommandError("Missing refs/remotes/origin/main.")

    main_result = (
        MainSyncResult(status="skipped", ahead_of_upstream=0, behind_upstream=0, message="Main sync skipped.")
        if args.skip_main_sync
        else sync_main(root, args.push)
    )

    branch_results: list[BranchResult] = []
    for branch in selected_branches:
        if not git_ref_exists(root, f"refs/remotes/origin/{branch}"):
            branch_results.append(
                BranchResult(
                    branch=branch,
                    status="missing",
                    start_sha="",
                    end_sha="",
                    merged=False,
                    pushed=False,
                    tests_ran=False,
                    message="Origin branch does not exist.",
                    log_path=log_dir / f"{stamp}-{branch.replace('/', '__')}.log",
                )
            )
            continue
        branch_results.append(
            process_branch(
                root,
                branch,
                push=args.push,
                validation_branch=args.validation_branch,
                test_commands=test_commands,
                worktree_parent=worktree_dir,
                log_dir=log_dir,
                stamp=stamp,
            )
        )

    write_report(
        report_path,
        root=root,
        push=args.push,
        selected_branches=selected_branches,
        validation_branch=args.validation_branch,
        test_commands=test_commands,
        main_result=main_result,
        branch_results=branch_results,
    )
    update_failure_queue(
        root,
        selected_branches=selected_branches,
        validation_branch=args.validation_branch,
        test_commands=test_commands,
        branch_results=branch_results,
        report_path=report_path,
    )

    print(f"Nightly upstream merge report: {report_path}")
    print(f"Nightly upstream merge queue: {failure_queue_path(root)}")

    failures = [result for result in branch_results if result.status in {"missing", "merge_conflict", "tests_failed", "push_failed"}]
    if main_result.status == "blocked":
        return 1
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CommandError as error:
        print(error, file=sys.stderr)
        if error.output:
            print(error.output, file=sys.stderr)
        raise SystemExit(1) from error
