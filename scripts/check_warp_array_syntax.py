# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that enforces bracket syntax for Warp array type annotations.

Detects and autofixes:
  wp.array(dtype=X)           -> wp.array[X]
  wp.array(dtype=X, ndim=2)   -> wp.array2d[X]
  wp.array2d(dtype=X)         -> wp.array2d[X]
  wp.array1d[X]               -> wp.array[X]

Runtime constructor calls (e.g. ``wp.array(dtype=X, shape=...)``) are not
affected because the regexes only match when ``dtype=`` (and optionally
``ndim=``) is the complete argument list.
"""

import re
import sys
from pathlib import Path

# Order matters: most specific patterns first.
_TRANSFORMS: list[tuple[re.Pattern[str], str]] = [
    # wp.array(dtype=X, ndim=2) -> wp.array2d[X]  (handles ndim 1..4)
    (re.compile(r"wp\.array\(dtype=([\w.]+),\s*ndim=([1-4])\)"), r"wp.array\2d[\1]"),
    # wp.array2d(dtype=X) -> wp.array2d[X]  (handles 1d..4d)
    (re.compile(r"wp\.array([1-4])d\(dtype=([\w.]+)\)"), r"wp.array\1d[\2]"),
    # wp.array(dtype=X) -> wp.array[X]
    (re.compile(r"wp\.array\(dtype=([\w.]+)\)"), r"wp.array[\1]"),
    # wp.array1d[X] -> wp.array[X]
    (re.compile(r"wp\.array1d\["), "wp.array["),
]


def fix_content(content: str) -> str:
    for pattern, replacement in _TRANSFORMS:
        content = pattern.sub(replacement, content)
    return content


def main() -> int:
    changed: list[str] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.suffix == ".py":
            continue
        content = path.read_text(encoding="utf-8")
        fixed = fix_content(content)
        if content != fixed:
            path.write_text(fixed, encoding="utf-8")
            changed.append(arg)

    if changed:
        for f in changed:
            print(f"Fixed warp array syntax: {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
