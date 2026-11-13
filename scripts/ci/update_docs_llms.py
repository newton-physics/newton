# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate the root documentation router from published version metadata."""

import argparse
import json
import re
from pathlib import Path


def update(root: Path) -> None:
    """Link only to entrypoints that have actually been published."""
    lines = ["# Newton documentation", "", "> Select documentation matching your Newton version.", ""]
    versions = [("Stable release", "stable"), ("Development version", "latest")]
    switcher = root / "switcher.json"
    if switcher.is_file():
        versions.extend(
            (entry.get("name", entry["version"]), entry["version"])
            for entry in json.loads(switcher.read_text(encoding="utf-8"))
            if re.fullmatch(r"\d+\.\d+\.\d+", entry["version"])
        )
    for label, directory in versions:
        for entrypoint in ("llms.txt", "index.html"):
            if (root / directory / entrypoint).is_file():
                lines.append(f"- [{label}]({directory}/{entrypoint})")
                break
    (root / "llms.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    update(parser.parse_args().root)
