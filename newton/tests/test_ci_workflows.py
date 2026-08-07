# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from os import environ, pathsep
from pathlib import Path


def find_bash() -> str | None:
    """Find a Bash executable without selecting WSL on Windows."""
    if sys.platform != "win32":
        return shutil.which("bash")

    git = shutil.which("git")
    if git is None:
        return None
    git_dir = Path(git).resolve().parent
    for candidate in (git_dir / "bash.exe", git_dir.parent / "bin" / "bash.exe"):
        if candidate.is_file():
            return str(candidate)
    return None


def extract_run_script(workflow: str) -> str:
    """Extract the sole step script from a workflow."""
    marker = "        run: |"
    lines = workflow.splitlines()
    marker_indexes = [index for index, line in enumerate(lines) if line == marker]
    if len(marker_indexes) != 1:
        raise ValueError(f"expected one workflow run block, found {len(marker_indexes)}")

    script_lines = []
    for line in lines[marker_indexes[0] + 1 :]:
        if line.startswith("          "):
            script_lines.append(line[10:])
        elif not line:
            script_lines.append(line)
        else:
            break
    return "\n".join(script_lines) + "\n"


class TestPullRequestWorkflows(unittest.TestCase):
    def test_closed_pr_cancels_only_exact_head_runs(self):
        """Cancel only exact-head runs for the closed pull request."""
        bash = find_bash()
        if bash is None:
            self.skipTest("Bash is required to exercise the workflow filter")
        if shutil.which("jq") is None:
            self.skipTest("jq is required to exercise the workflow filter")

        repo_root = Path(__file__).parents[2]
        workflow = (repo_root / ".github/workflows/pr_closed.yml").read_text()

        shell = extract_run_script(workflow)

        head_repo = {"id": 979096143, "full_name": "contributor/newton"}
        base_repo = {"id": 970467647, "full_name": "newton-physics/newton"}
        exact_head = {
            "head_repository": head_repo,
            "head_branch": "main",
            "head_sha": "pull-request-head-sha",
            "pull_requests": [],
        }
        in_progress = {
            "workflow_runs": [
                {
                    "id": 108,
                    "event": "pull_request_target",
                    "path": ".github/workflows/pr_target_aws_gpu_tests.yml",
                    **exact_head,
                },
                {
                    "id": 109,
                    "event": "push",
                    "path": ".github/workflows/pr.yml",
                    **exact_head,
                },
                {
                    "id": 101,
                    "event": "push",
                    "path": ".github/workflows/push_aws_gpu.yml",
                    "head_repository": base_repo,
                    "head_branch": "main",
                    "head_sha": "base-branch-sha",
                    "pull_requests": [],
                },
                {
                    "id": 102,
                    "event": "pull_request",
                    "path": ".github/workflows/pr.yml",
                    **exact_head,
                },
                {
                    "id": 105,
                    "event": "pull_request",
                    "path": ".github/workflows/unrelated.yml",
                    **exact_head,
                },
            ]
        }
        queued = {
            "workflow_runs": [
                {
                    "id": 110,
                    "event": "pull_request",
                    "path": ".github/workflows/pr.yml",
                    **exact_head,
                    "head_branch": "other-branch",
                },
                {
                    "id": 103,
                    "event": "pull_request_target",
                    "path": ".github/workflows/pr_target_aws_gpu_tests.yml",
                    **exact_head,
                },
                {
                    "id": 104,
                    "event": "pull_request_target",
                    "path": ".github/workflows/pr_target_aws_gpu_benchmarks.yml",
                    **exact_head,
                    "head_sha": "stale-head-sha",
                },
                {
                    "id": 106,
                    "event": "pull_request",
                    "path": ".github/workflows/pr.yml",
                    **exact_head,
                    "head_repository": base_repo,
                },
                {
                    "id": 107,
                    "event": "pull_request_target",
                    "path": ".github/workflows/pr_target_aws_gpu_benchmarks.yml",
                    **exact_head,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "in_progress.json").write_text(json.dumps(in_progress))
            (temp_path / "queued.json").write_text(json.dumps(queued))
            fake_gh = temp_path / "gh"
            fake_gh.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    if [ "$1" = "api" ] && [ "$2" = "--paginate" ]; then
                      case "$3" in
                        *status=in_progress*) cat "$IN_PROGRESS_FIXTURE" ;;
                        *status=queued*) cat "$QUEUED_FIXTURE" ;;
                        *) exit 64 ;;
                      esac
                    elif [ "$1" = "run" ] && [ "$2" = "cancel" ]; then
                      echo "$3" >> "$CANCEL_LOG"
                    else
                      exit 64
                    fi
                    """
                )
            )
            fake_gh.chmod(0o755)

            cancel_log = temp_path / "cancelled"
            env = environ.copy()
            env.update(
                {
                    "PATH": f"{temp_path}{pathsep}{env['PATH']}",
                    "GH_TOKEN": "test-token",
                    "GH_REPO": "newton-physics/newton",
                    "PR_NUMBER": "42",
                    "HEAD_REPO_ID": str(head_repo["id"]),
                    "HEAD_REPO": head_repo["full_name"],
                    "HEAD_BRANCH": "main",
                    "HEAD_SHA": "pull-request-head-sha",
                    "CURRENT_RUN": "103",
                    "IN_PROGRESS_FIXTURE": str(temp_path / "in_progress.json"),
                    "QUEUED_FIXTURE": str(temp_path / "queued.json"),
                    "CANCEL_LOG": str(cancel_log),
                }
            )
            result = subprocess.run(
                [bash, "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", shell],
                cwd=repo_root,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(sorted(cancel_log.read_text().splitlines()), ["102", "107", "108"])


if __name__ == "__main__":
    unittest.main()
