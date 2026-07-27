# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Safely remove a temporary Newton release-audit report."""

from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

_REPORT_NAME = re.compile(r"newton-[A-Za-z0-9][A-Za-z0-9.+-]*-(?:prerelease|rc|retrospective)-report\.md")


def cleanup_report(path: Path, *, temporary_directory: Path | None = None) -> None:
    """Remove an allowed report directly beneath the temporary directory."""
    temporary_root = (temporary_directory or Path(tempfile.gettempdir())).resolve()
    if path.is_symlink():
        raise ValueError("report path must not be a symlink")
    candidate = path.resolve()
    if candidate.parent != temporary_root:
        raise ValueError(f"report must be directly beneath {temporary_root}")
    if _REPORT_NAME.fullmatch(candidate.name) is None:
        raise ValueError(f"not an allowed Newton report filename: {candidate.name}")
    path.unlink(missing_ok=True)


def main() -> None:
    """Run the report cleanup command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        cleanup_report(args.path)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
