# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the PR ASV gate from the full configuration and selection manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).parents[1]
FULL_CONFIG_PATH = ROOT / "asv.conf.json"
SELECTION_PATH = Path(__file__).with_name("pr_benchmarks.txt")


def load_benchmark_patterns(path: Path = SELECTION_PATH) -> tuple[str, ...]:
    """Load non-empty benchmark selection expressions from *path*."""
    patterns = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not patterns:
        raise ValueError(f"No PR benchmark patterns found in {path}")
    return patterns


def build_pr_config(path: Path = FULL_CONFIG_PATH) -> dict:
    """Derive the PR environment from the full ASV configuration."""
    config = json.loads(path.read_text(encoding="utf-8"))
    config["env_dir"] = "asv/pr-env"

    install_commands = config["install_command"]
    torch_commands = [command for command in install_commands if "torch==" in command]
    if len(torch_commands) != 1:
        raise ValueError(f"Expected one full-ASV Torch install command, found {len(torch_commands)}")
    install_commands.remove(torch_commands[0])
    return config


def build_asv_command(config_path: Path, patterns: tuple[str, ...], revisions: list[str], quick: bool) -> list[str]:
    """Build the ASV command comparing two revisions, or running one once with *quick*."""
    if quick:
        (revision,) = revisions
        subcommand, options, targets = "run", ["--quick"], [f"{revision}^!"]
    else:
        subcommand = "continuous"
        options = ["--interleave-rounds", "--append-samples", "--no-only-changed"]
        targets = list(revisions)

    command = ["uvx", "--with", "virtualenv", "asv", subcommand, "--config", str(config_path)]
    command += ["--launch-method", "spawn", *options, "--show-stderr"]
    for pattern in patterns:
        command.extend(("--bench", pattern))
    return command + targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "revisions", nargs="+", metavar="REV", help="Base and branch revisions, or one revision with --quick"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run each benchmark once on a single revision without timing, e.g. to compile its Warp kernels",
    )
    args = parser.parse_args()
    if len(args.revisions) != (1 if args.quick else 2):
        parser.error("expected one revision with --quick, otherwise a base and a branch revision")
    return args


def main() -> int:
    args = _parse_args()
    config = build_pr_config()
    patterns = load_benchmark_patterns()

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".asv-pr-",
        suffix=".json",
        dir=ROOT,
        delete=False,
    ) as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
        config_path = Path(config_file.name)

    command = build_asv_command(config_path, patterns, args.revisions, args.quick)

    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
