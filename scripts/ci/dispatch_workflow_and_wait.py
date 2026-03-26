# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a GitHub Actions workflow and poll until it completes.

Designed to be called from a GitHub Actions workflow step. Uses the
``gh`` CLI, which must be available on ``PATH`` and authenticated via
the ``GH_TOKEN`` environment variable.

Required environment variables:
    REPO -- owner/repo slug (e.g. ``newton-physics/newton``).
    REF -- git ref to dispatch on (e.g. ``refs/heads/main``).
    GITHUB_OUTPUT -- path to the GitHub Actions step-output file.

Usage::

    python scripts/ci/dispatch_workflow_and_wait.py <workflow-file> [extra-gh-api-args...]

Example::

    python scripts/ci/dispatch_workflow_and_wait.py aws_gpu_tests.yml \\
        -f "inputs[instance-type]=g7e.12xlarge"

Step outputs (written to ``$GITHUB_OUTPUT``):
    conclusion
        Workflow run conclusion: ``success``, ``failure``, ``cancelled``,
        ``timed_out``, or ``dispatch_error``.
    run-url
        HTML URL of the dispatched workflow run on GitHub.

The script uses the ``return_run_details`` parameter (available since
February 2026) to obtain the run ID directly from the dispatch
response, avoiding the need to poll the runs list.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

POLL_INTERVAL: int = 30
"""Seconds between status polls."""

MAX_POLL_DURATION: int = 2 * 60 * 60
"""Maximum total seconds to wait for the dispatched run to complete (2 hours)."""


def gh(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command and return the completed process.

    Args:
        args: Arguments forwarded to the ``gh`` CLI.

    Returns:
        The :class:`~subprocess.CompletedProcess` result.  The caller is
        responsible for checking ``returncode``.
    """
    return subprocess.run(
        ["gh", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def set_output(name: str, value: str) -> None:
    """Write a key-value pair to the GitHub Actions step-output file.

    Args:
        name: Output name (e.g. ``conclusion``).
        value: Output value.
    """
    path = os.environ.get("GITHUB_OUTPUT", "")
    if path:
        with open(path, "a") as f:
            f.write(f"{name}={value}\n")


def log_group(title: str) -> None:
    """Emit a ``::group::`` workflow command to start a collapsible log section."""
    print(f"::group::{title}", flush=True)


def log_endgroup() -> None:
    """Emit ``::endgroup::`` to close the current collapsible log section."""
    print("::endgroup::", flush=True)


def log_error(msg: str) -> None:
    """Emit an ``::error::`` workflow command that surfaces as an annotation."""
    print(f"::error::{msg}", flush=True)


def log_warning(msg: str) -> None:
    """Emit a ``::warning::`` workflow command that surfaces as an annotation."""
    print(f"::warning::{msg}", flush=True)


def dispatch(repo: str, ref: str, workflow_file: str, extra_args: list[str]) -> tuple[int, str]:
    """Dispatch a workflow via the GitHub REST API.

    Calls ``POST /repos/{owner}/{repo}/actions/workflows/{id}/dispatches``
    with ``return_run_details=true`` to obtain the run ID in the response.

    Args:
        repo: Repository slug (``owner/repo``).
        ref: Git ref to dispatch on.
        workflow_file: Workflow filename (e.g. ``aws_gpu_tests.yml``).
        extra_args: Additional arguments forwarded to ``gh api``
            (e.g. ``["-f", "inputs[instance-type]=g7e.12xlarge"]``).

    Returns:
        A ``(run_id, html_url)`` tuple for the dispatched workflow run.

    Raises:
        RuntimeError: If the dispatch API call fails or the response does
            not contain a ``workflow_run_id``.
    """
    result = gh(
        "api",
        f"repos/{repo}/actions/workflows/{workflow_file}/dispatches",
        "-f",
        f"ref={ref}",
        *extra_args,
        "-F",
        "return_run_details=true",
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh api failed:\n{result.stderr.strip()}")

    data = json.loads(result.stdout)
    run_id = data.get("workflow_run_id")
    html_url = data.get("html_url", "")
    if not run_id:
        raise RuntimeError(f"Missing workflow_run_id in response:\n{result.stdout.strip()}")

    return int(run_id), html_url


def wait_for_completion(repo: str, run_id: int) -> str:
    """Poll a workflow run until it reaches ``completed`` status.

    Polls every :data:`POLL_INTERVAL` seconds up to
    :data:`MAX_POLL_DURATION`. Transient API errors (network issues,
    rate limiting) are logged as warnings and retried automatically.

    Args:
        repo: Repository slug (``owner/repo``).
        run_id: The workflow run ID to monitor.

    Returns:
        The run conclusion (e.g. ``success``, ``failure``, ``cancelled``)
        or ``timed_out`` if the maximum poll duration is exceeded.
    """
    elapsed = 0
    while elapsed < MAX_POLL_DURATION:
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

        result = gh("run", "view", str(run_id), "--repo", repo, "--json", "status,conclusion")
        if result.returncode != 0:
            log_warning(f"gh run view failed ({elapsed}s elapsed): {result.stderr.strip()}")
            continue

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            log_warning(f"Failed to parse gh run view output ({elapsed}s elapsed)")
            continue

        status = data.get("status")
        if status == "completed":
            conclusion = data.get("conclusion", "unknown")
            print(f"Run {run_id} completed with conclusion: {conclusion}", flush=True)
            return conclusion

        print(f"Status: {status} ({elapsed}s elapsed)", flush=True)

    log_error(f"Timed out waiting for run {run_id} after {elapsed // 60} minutes")
    return "timed_out"


def main() -> int:
    """Entry point: parse arguments, dispatch, poll, and write outputs."""
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} <workflow-file> [extra-gh-api-args...]",
            file=sys.stderr,
        )
        return 1

    workflow_file = sys.argv[1]
    extra_args = sys.argv[2:]

    repo = os.environ["REPO"]
    ref = os.environ["REF"]

    # --- Dispatch ---
    log_group(f"Dispatching {workflow_file}")
    try:
        run_id, html_url = dispatch(repo, ref, workflow_file, extra_args)
    except RuntimeError as e:
        log_error(f"Failed to dispatch {workflow_file}: {e}")
        log_endgroup()
        set_output("run-url", "")
        set_output("conclusion", "dispatch_error")
        return 0

    print(f"Triggered run {run_id}: {html_url}", flush=True)
    set_output("run-url", html_url)
    log_endgroup()

    # --- Poll for completion ---
    log_group("Waiting for completion")
    conclusion = wait_for_completion(repo, run_id)
    set_output("conclusion", conclusion)
    log_endgroup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
