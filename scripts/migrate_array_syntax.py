#!/usr/bin/env python3
"""Migrate wp.array(dtype=X) annotations to wp.array[X] bracket syntax.

Transforms:
  wp.array(dtype=X)           -> wp.array[X]
  wp.array(dtype=X, ndim=N)   -> wp.arrayNd[X]   (N in 1..4)
  wp.arrayNd(dtype=X)         -> wp.arrayNd[X]    (N in 2..4)
"""

import argparse
import re
import sys
from pathlib import Path

# Pattern 1: wp.array(dtype=X, ndim=N) -> wp.arrayNd[X]
PATTERN_ARRAY_NDIM = re.compile(
    r"wp\.array\(dtype=([\w.]+),\s*ndim=(\d)\)"
)

# Pattern 2: wp.arrayNd(dtype=X) -> wp.arrayNd[X]  (N in 1..4)
PATTERN_ARRAYND = re.compile(
    r"wp\.array([1-4])d\(dtype=([\w.]+)\)"
)

# Pattern 3: wp.array(dtype=X) -> wp.array[X]  (plain 1d, no other args)
PATTERN_ARRAY_PLAIN = re.compile(
    r"wp\.array\(dtype=([\w.]+)\)"
)


def migrate_content(content: str) -> str:
    """Apply all three transformations to file content."""
    content = PATTERN_ARRAY_NDIM.sub(r"wp.array\2d[\1]", content)
    content = PATTERN_ARRAYND.sub(r"wp.array\1d[\2]", content)
    content = PATTERN_ARRAY_PLAIN.sub(r"wp.array[\1]", content)
    return content


def migrate_file(path: Path, *, dry_run: bool = False) -> bool:
    """Migrate a single file. Returns True if changes were made."""
    content = path.read_text()
    new_content = migrate_content(content)
    if content == new_content:
        return False
    if not dry_run:
        path.write_text(new_content)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        default=["newton", "asv"],
        help="Directories or files to process (default: newton asv)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing files",
    )
    args = parser.parse_args()

    changed = []
    for root in args.paths:
        root_path = Path(root)
        if root_path.is_file():
            files = [root_path]
        else:
            files = sorted(root_path.rglob("*.py"))
        for path in files:
            if migrate_file(path, dry_run=args.dry_run):
                changed.append(path)

    for path in changed:
        print(f"{'Would change' if args.dry_run else 'Changed'}: {path}")
    print(f"\n{'Would change' if args.dry_run else 'Changed'} {len(changed)} files.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
