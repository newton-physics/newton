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
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

POLL_INTERVAL: int = 30
"""Seconds between status polls."""

MAX_POLL_DURATION: int = 60 * 60
"""Maximum total seconds to wait for the dispatched run to complete (1 hour)."""

GH_TIMEOUT: int = 120
"""Maximum seconds to wait for a single ``gh`` CLI invocation."""


def gh(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command and return the completed process.

    If the command does not finish within :data:`GH_TIMEOUT` seconds, a
    synthetic failed :class:`~subprocess.CompletedProcess` is returned
    instead of raising :exc:`~subprocess.TimeoutExpired`.

    Args:
        args: Arguments forwarded to the ``gh`` CLI.

    Returns:
        The :class:`~subprocess.CompletedProcess` result.  The caller is
        responsible for checking ``returncode``.
    """
    try:
        return subprocess.run(
            ["gh", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=GH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            args=["gh", *args],
            returncode=1,
            stdout="",
            stderr=f"gh command timed out after {GH_TIMEOUT}s",
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


def dispatch(repo: str, ref: str, workflow_file: str, extra_args: list[str]) -> tuple[int, str]:
    """Dispatch a workflow via the GitHub REST API.

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
    start_time = time.monotonic()
    while time.monotonic() - start_time < MAX_POLL_DURATION:
        time.sleep(POLL_INTERVAL)
        elapsed = int(time.monotonic() - start_time)

        result = gh("run", "view", str(run_id), "--repo", repo, "--json", "status,conclusion")
        if result.returncode != 0:
            print(f"::warning::gh run view failed ({elapsed}s elapsed): {result.stderr.strip()}", flush=True)
            continue

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"::warning::Failed to parse gh run view output ({elapsed}s elapsed)", flush=True)
            continue

        status = data.get("status")
        if status == "completed":
            conclusion = data.get("conclusion", "unknown")
            print(f"Run {run_id} completed with conclusion: {conclusion}", flush=True)
            return conclusion

        print(f"Status: {status} ({elapsed}s elapsed)", flush=True)

    elapsed = int(time.monotonic() - start_time)
    print(f"::error::Timed out waiting for run {run_id} after {elapsed // 60} minutes", flush=True)
    return "timed_out"


def main() -> int:
    """Entry point: parse arguments, dispatch, poll, and write outputs."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workflow-file> [extra-gh-api-args...]", file=sys.stderr)
        return 1

    workflow_file = sys.argv[1]
    extra_args = sys.argv[2:]

    repo = os.environ["REPO"]
    ref = os.environ["REF"]

    # --- Dispatch ---
    print(f"::group::Dispatching {workflow_file}", flush=True)
    try:
        run_id, html_url = dispatch(repo, ref, workflow_file, extra_args)
    except RuntimeError as e:
        print(f"::error::Failed to dispatch {workflow_file}: {e}", flush=True)
        print("::endgroup::", flush=True)
        set_output("run-url", "")
        set_output("conclusion", "dispatch_error")
        # Exit 0 so the orchestrator step is not marked as failed — the
        # "dispatch_error" conclusion output lets downstream jobs decide
        # how to handle it without aborting the entire nightly run.
        return 0

    print(f"Triggered run {run_id}: {html_url}", flush=True)
    set_output("run-url", html_url)
    print("::endgroup::", flush=True)

    # --- Poll for completion ---
    print("::group::Waiting for completion", flush=True)
    conclusion = wait_for_completion(repo, run_id)
    set_output("conclusion", conclusion)
    print("::endgroup::", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
