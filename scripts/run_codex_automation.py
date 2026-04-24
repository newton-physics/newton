#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a repo-local Codex automation prompt non-interactively."""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tomllib


def repo_root() -> pathlib.Path:
    process = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"),
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError("Run this script from inside the Newton repository.")
    return pathlib.Path(process.stdout.strip())


def find_codex() -> str:
    codex = shutil.which("codex")
    if codex:
        return codex

    candidates = sorted(
        (path for path in pathlib.Path.home().glob(".local/share/mise/installs/node/*/bin/codex") if path.is_file()),
        reverse=True,
    )
    if candidates:
        return str(candidates[0])

    raise RuntimeError("Could not find the Codex CLI. Install it or add it to PATH before scheduling automation.")


def load_automation(path: pathlib.Path) -> tuple[pathlib.Path, str]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)

    automation = payload.get("automation", {})
    prompt = payload.get("prompt", {})
    cwd_text = automation.get("cwd")
    prompt_text = prompt.get("content")

    if not isinstance(cwd_text, str) or not cwd_text:
        raise RuntimeError(f"{path} is missing automation.cwd.")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise RuntimeError(f"{path} is missing prompt.content.")

    return pathlib.Path(cwd_text), prompt_text


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("automation_toml", type=pathlib.Path, help="Path to an automation.toml file.")
    parser.add_argument(
        "--codex",
        default=None,
        help="Codex executable path. Defaults to PATH, then the local mise node install.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command that would be run without invoking Codex.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    automation_path = args.automation_toml
    if not automation_path.is_absolute():
        automation_path = root / automation_path

    cwd, prompt = load_automation(automation_path)
    codex = args.codex or find_codex()
    command = (
        codex,
        "exec",
        "-C",
        str(cwd),
        "--dangerously-bypass-approvals-and-sandbox",
        prompt,
    )

    if args.dry_run:
        print(" ".join(command[:-1]) + " <prompt>")
        return 0

    env = os.environ.copy()
    codex_dir = str(pathlib.Path(codex).parent)
    env["PATH"] = codex_dir + os.pathsep + env.get("PATH", "")
    process = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    return process.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
