# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pre-commit hook that enforces bracket syntax for Warp array type annotations.

Detects and autofixes:
  wp.array(dtype=X)           -> wp.array[X]
  wp.array(dtype=X, ndim=2)   -> wp.array2d[X]
  wp.array2d(dtype=X)         -> wp.array2d[X]
  wp.array1d[X]               -> wp.array[X]

Handles complex dtype expressions (e.g. ``wp.types.matrix((2, 3), wp.float32)``)
and multi-line variants.

Runtime constructor calls (e.g. ``wp.array(dtype=X, shape=...)``) are not
affected because the scanner only matches when ``dtype=`` (and optionally
``ndim=``) is the complete argument list.
"""

import re
import sys
from pathlib import Path

# Matches the start of a parenthesized wp.array type annotation.
_PAREN_ARRAY_RE = re.compile(r"wp\.array([1-4]?)d?\(\s*dtype=")

# wp.array1d[X] -> wp.array[X]
_ARRAY1D_BRACKET_RE = re.compile(r"wp\.array1d\[")


def _find_closing_paren(content: str, open_pos: int) -> int:
    """Return the index of the paren that closes the one at *open_pos*."""
    depth = 1
    i = open_pos + 1
    while i < len(content) and depth > 0:
        if content[i] == "(":
            depth += 1
        elif content[i] == ")":
            depth -= 1
        i += 1
    return i - 1 if depth == 0 else -1


def _parse_dtype_ndim(interior: str) -> tuple[str, int | None] | None:
    """Parse the interior of ``wp.array(...)``, returning ``(dtype_expr, ndim)``.

    Returns ``None`` if the interior contains arguments other than ``dtype``
    and ``ndim`` (i.e. it is a runtime constructor call, not a type annotation).
    """
    # Split on top-level commas (respecting nested parens/brackets).
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(interior):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(interior[start:i].strip())
            start = i + 1
    parts.append(interior[start:].strip())

    dtype_expr: str | None = None
    ndim: int | None = None
    for part in parts:
        if part.startswith("dtype="):
            dtype_expr = part[len("dtype=") :]
        elif part.startswith("ndim="):
            try:
                ndim = int(part[len("ndim=") :])
            except ValueError:
                return None
        else:
            # Unknown argument — this is a constructor call, not an annotation.
            return None

    if dtype_expr is None:
        return None
    return dtype_expr, ndim


def fix_content(content: str) -> str:
    # Pass 1: replace parenthesized forms using a balanced-paren scanner.
    result: list[str] = []
    last = 0
    for m in _PAREN_ARRAY_RE.finditer(content):
        # Find the opening '(' — it's just before 'dtype='.
        open_pos = m.start() + content[m.start() :].index("(")
        close_pos = _find_closing_paren(content, open_pos)
        if close_pos < 0:
            continue

        interior = content[open_pos + 1 : close_pos]
        parsed = _parse_dtype_ndim(interior)
        if parsed is None:
            continue

        dtype_expr, ndim = parsed
        # Determine dimensionality: explicit ndim= wins, then the digit in array2d/array3d/etc.
        dim_suffix = m.group(1)  # '' or '1'..'4' from wp.array<N>d(
        if ndim is not None:
            effective_ndim = ndim
        elif dim_suffix:
            effective_ndim = int(dim_suffix)
        else:
            effective_ndim = 1

        if effective_ndim == 1:
            replacement = f"wp.array[{dtype_expr}]"
        else:
            replacement = f"wp.array{effective_ndim}d[{dtype_expr}]"

        result.append(content[last : m.start()])
        result.append(replacement)
        last = close_pos + 1

    result.append(content[last:])
    content = "".join(result)

    # Pass 2: wp.array1d[X] -> wp.array[X]
    content = _ARRAY1D_BRACKET_RE.sub("wp.array[", content)
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
