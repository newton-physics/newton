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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", help="Base Git revision")
    parser.add_argument("branch", help="Git revision under test")
    return parser.parse_args()


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

    command = [
        "uvx",
        "--with",
        "virtualenv",
        "asv",
        "continuous",
        "--config",
        str(config_path),
        "--launch-method",
        "spawn",
        "--interleave-rounds",
        "--append-samples",
        "--no-only-changed",
        "--show-stderr",
    ]
    for pattern in patterns:
        command.extend(("--bench", pattern))
    command.extend((args.base, args.branch))

    try:
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    finally:
        config_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
