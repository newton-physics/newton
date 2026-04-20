#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded repair loop for nightly upstream merge failures."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import subprocess
import sys
from collections.abc import Sequence


FAILURE_PRIORITY = {
    "tests_failed": 0,
    "push_failed": 1,
    "merge_conflict": 2,
    "missing": 3,
}


class CommandError(RuntimeError):
    """Raised when a subprocess call fails."""

    def __init__(self, message: str, *, output: str = "", returncode: int | None = None):
        super().__init__(message)
        self.output = output
        self.returncode = returncode


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def summarize_message(message: str, *, limit: int = 240) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def repo_root() -> pathlib.Path:
    return pathlib.Path(git_capture(("git", "rev-parse", "--show-toplevel")).strip())


def git_capture(command: Sequence[str], *, cwd: pathlib.Path | None = None, check: bool = True) -> str:
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


def run_logged_command(command: Sequence[str], *, cwd: pathlib.Path, log_file: pathlib.Path, check: bool = True) -> None:
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


def sanitize_branch(branch: str) -> str:
    return branch.replace("/", "__")


def merge_automation_dir(root: pathlib.Path) -> pathlib.Path:
    return root / ".codex" / "automations" / "nightly-upstream-merge"


def repair_automation_dir(root: pathlib.Path) -> pathlib.Path:
    return root / ".codex" / "automations" / "nightly-upstream-repair"


def failure_queue_path(root: pathlib.Path) -> pathlib.Path:
    return merge_automation_dir(root) / "failures.json"


def current_repair_path(root: pathlib.Path) -> pathlib.Path:
    return repair_automation_dir(root) / "current_repair.json"


def repair_log_dir(root: pathlib.Path) -> pathlib.Path:
    return repair_automation_dir(root) / "logs"


def repair_worktree_dir(root: pathlib.Path) -> pathlib.Path:
    return repair_automation_dir(root) / "worktrees"


def load_json(path: pathlib.Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_queue(root: pathlib.Path) -> dict:
    return load_json(failure_queue_path(root), {"version": 1, "updated_at": None, "items": []})


def save_queue(root: pathlib.Path, queue: dict) -> None:
    queue["version"] = 1
    queue["updated_at"] = now_iso()
    save_json(failure_queue_path(root), queue)


def find_queue_item(queue: dict, branch: str) -> dict | None:
    for item in queue.get("items", []):
        if item.get("branch") == branch:
            return item
    return None


def git_ref_exists(root: pathlib.Path, ref: str) -> bool:
    process = subprocess.run(
        ("git", "show-ref", "--verify", "--quiet", ref),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode == 0


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


def branch_contains_upstream(root: pathlib.Path, branch: str) -> bool:
    process = subprocess.run(
        ("git", "merge-base", "--is-ancestor", "refs/remotes/upstream/main", f"refs/remotes/origin/{branch}"),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return process.returncode == 0


def unresolved_files(worktree: pathlib.Path) -> list[str]:
    output = git_capture(("git", "diff", "--name-only", "--diff-filter=U"), cwd=worktree, check=False)
    return [line for line in output.splitlines() if line.strip()]


def choose_next_item(queue: dict, *, max_attempts: int) -> dict | None:
    for item in queue.get("items", []):
        if int(item.get("attempt_count", 0)) >= max_attempts:
            item["repair_status"] = "exhausted"
        elif item.get("repair_status") == "repairing":
            item["repair_status"] = "queued"

    candidates = [
        item
        for item in queue.get("items", [])
        if item.get("repair_status", "queued") in {"queued", "failed", "retryable"}
        and int(item.get("attempt_count", 0)) < max_attempts
    ]
    if not candidates:
        return None

    def sort_key(item: dict) -> tuple:
        has_validation = 0 if item.get("validation_commands") else 1
        priority = FAILURE_PRIORITY.get(item.get("failure_type", ""), 99)
        attempts = int(item.get("attempt_count", 0))
        first_seen = item.get("first_seen_at", "")
        return (has_validation, priority, attempts, first_seen, item.get("branch", ""))

    candidates.sort(key=sort_key)
    return candidates[0]


def prepare_repair(root: pathlib.Path, *, max_attempts: int) -> int:
    queue = load_queue(root)
    active_path = current_repair_path(root)
    active = load_json(active_path, {})
    if active:
        worktree = pathlib.Path(active["worktree_path"])
        if worktree.exists():
            print(f"Reusing prepared repair: {active_path}")
            print(f"Worktree: {worktree}")
            return 0
        active_path.unlink(missing_ok=True)

    git_capture(("git", "fetch", "origin", "--prune"), cwd=root)
    git_capture(("git", "fetch", "upstream", "--prune"), cwd=root)

    item = choose_next_item(queue, max_attempts=max_attempts)
    if item is None:
        save_queue(root, queue)
        print("No queued failures remain.")
        return 0

    branch = item["branch"]
    if not git_ref_exists(root, f"refs/remotes/origin/{branch}"):
        item["repair_status"] = "exhausted"
        item["last_summary"] = "Origin branch no longer exists."
        save_queue(root, queue)
        raise CommandError(f"Cannot prepare repair for missing origin branch: {branch}")

    safe_branch = sanitize_branch(branch)
    worktree = repair_worktree_dir(root) / safe_branch
    remove_worktree(root, worktree)

    git_capture(("git", "worktree", "add", "--detach", str(worktree), f"refs/remotes/origin/{branch}"), cwd=root)

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    log_path = repair_log_dir(root) / f"{stamp}-{safe_branch}.log"
    temp_branch = f"nightly-repair/{safe_branch}/{stamp}"
    run_logged_command(("git", "switch", "-c", temp_branch), cwd=worktree, log_file=log_path)

    merge_status = "already_contains_upstream" if branch_contains_upstream(root, branch) else "merge_required"
    if merge_status == "merge_required":
        try:
            run_logged_command(
                ("git", "merge", "--no-edit", "--no-ff", "refs/remotes/upstream/main"),
                cwd=worktree,
                log_file=log_path,
            )
            merge_status = "merged_cleanly"
        except CommandError as error:
            conflict_files = unresolved_files(worktree)
            merge_status = "merge_conflict"
            item["last_summary"] = summarize_message(error.output or "Merge conflict while preparing repair.")
        else:
            conflict_files = []
    else:
        conflict_files = unresolved_files(worktree)

    item["attempt_count"] = int(item.get("attempt_count", 0)) + 1
    item["last_attempted_at"] = now_iso()
    item["repair_status"] = "repairing"

    current = {
        "attempt_count": item["attempt_count"],
        "branch": branch,
        "conflict_files": conflict_files,
        "failure_queue_path": str(failure_queue_path(root)),
        "failure_type": item.get("failure_type"),
        "log_path": str(log_path),
        "merge_status": merge_status,
        "prepared_at": now_iso(),
        "report_path": item.get("report_path"),
        "validation_commands": item.get("validation_commands", []),
        "worktree_path": str(worktree),
    }
    save_json(active_path, current)
    save_queue(root, queue)

    print(f"Prepared repair file: {active_path}")
    print(f"Prepared repair worktree: {worktree}")
    return 0


def cleanup_active_repair(root: pathlib.Path) -> None:
    active_path = current_repair_path(root)
    active = load_json(active_path, {})
    worktree_path = pathlib.Path(active["worktree_path"]) if active else None
    if worktree_path is not None:
        remove_worktree(root, worktree_path)
    active_path.unlink(missing_ok=True)


def run_validation_commands(worktree: pathlib.Path, log_path: pathlib.Path, commands: Sequence[str]) -> None:
    for command_text in commands:
        run_logged_command(tuple(shlex.split(command_text)), cwd=worktree, log_file=log_path)


def finalize_repair(root: pathlib.Path, *, success: bool, push: bool, message: str | None, max_attempts: int) -> int:
    active = load_json(current_repair_path(root), {})
    if not active:
        print("No active repair is prepared.")
        return 0

    branch = active["branch"]
    worktree = pathlib.Path(active["worktree_path"])
    log_path = pathlib.Path(active["log_path"])
    queue = load_queue(root)
    item = find_queue_item(queue, branch)
    if item is None:
        cleanup_active_repair(root)
        raise CommandError(f"Queue item for {branch} no longer exists.")

    if success:
        conflict_files = unresolved_files(worktree)
        if conflict_files:
            raise CommandError(
                f"Cannot finalize successful repair for {branch}; unresolved conflicts remain: {', '.join(conflict_files)}"
            )
        run_validation_commands(worktree, log_path, active.get("validation_commands", []))
        if push:
            run_logged_command(("git", "push", "origin", f"HEAD:refs/heads/{branch}"), cwd=worktree, log_file=log_path)
        queue["items"] = [entry for entry in queue.get("items", []) if entry.get("branch") != branch]
        save_queue(root, queue)
        cleanup_active_repair(root)
        print(f"Finalized repair for {branch}.")
        return 0

    item["repair_status"] = "exhausted" if int(item.get("attempt_count", 0)) >= max_attempts else "queued"
    item["last_summary"] = summarize_message(message or "Repair attempt ended without a passing validation result.")
    item["last_seen_at"] = now_iso()
    save_queue(root, queue)
    cleanup_active_repair(root)
    print(f"Requeued repair for {branch}.")
    return 0


def show_status(root: pathlib.Path) -> int:
    queue = load_queue(root)
    active = load_json(current_repair_path(root), {})
    print(f"Failure queue: {failure_queue_path(root)}")
    print(f"Queued items: {len(queue.get('items', []))}")
    for item in queue.get("items", []):
        print(
            f"- {item.get('branch')}: {item.get('failure_type')} "
            f"(attempts={item.get('attempt_count', 0)}, status={item.get('repair_status', 'queued')})"
        )
    if active:
        print(f"Active repair: {current_repair_path(root)}")
        print(f"Worktree: {active.get('worktree_path')}")
    else:
        print("Active repair: none")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and finalize bounded repairs for nightly upstream merge failures.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare the next queued failure in a disposable worktree.")
    prepare.add_argument("--max-attempts", type=int, default=3, help="Maximum repair attempts before a queue item is exhausted.")

    finalize = subparsers.add_parser("finalize", help="Finalize the active repair as success or failure.")
    outcome = finalize.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--success", action="store_true", help="Validate and optionally push the repaired branch.")
    outcome.add_argument("--failure", action="store_true", help="Requeue the repaired branch without pushing.")
    finalize.add_argument("--push", action="store_true", help="Push the repaired branch back to origin on success.")
    finalize.add_argument("--message", default=None, help="Short failure summary to record when finalizing with --failure.")
    finalize.add_argument("--max-attempts", type=int, default=3, help="Maximum repair attempts before a queue item is exhausted.")

    subparsers.add_parser("status", help="Show queue and active repair state.")
    return parser


def main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root()
    if args.command == "prepare":
        return prepare_repair(root, max_attempts=args.max_attempts)
    if args.command == "finalize":
        return finalize_repair(
            root,
            success=args.success,
            push=args.push,
            message=args.message,
            max_attempts=args.max_attempts,
        )
    if args.command == "status":
        return show_status(root)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except CommandError as error:
        print(error, file=sys.stderr)
        if error.output:
            print(error.output, file=sys.stderr)
        raise SystemExit(1) from error
