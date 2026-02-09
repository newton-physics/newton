# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Update switcher.json with a new documentation version.

This script is called by the docs-release workflow to add new versions
to the documentation version switcher dropdown.

Usage:
    python update_docs_switcher.py <version>

Example:
    python update_docs_switcher.py 1.0.0
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

REPO_URL = "https://newton-physics.github.io/newton"
SWITCHER_PATH = Path("switcher.json")
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def validate_version(version: str) -> bool:
    """Validate that version is strict semver (X.Y.Z)."""
    return bool(SEMVER_PATTERN.match(version))


def validate_switcher_json(data: list) -> bool:
    """Validate switcher.json structure."""
    if not isinstance(data, list):
        return False
    for entry in data:
        if not isinstance(entry, dict):
            return False
        if "name" not in entry or "version" not in entry or "url" not in entry:
            return False
    return True


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <version>", file=sys.stderr)
        return 1

    version = sys.argv[1]

    # Validate version format
    if not validate_version(version):
        print(f"Error: Invalid version format '{version}'. Must be strict semver (X.Y.Z).", file=sys.stderr)
        return 1

    # Bootstrap switcher.json if missing
    if SWITCHER_PATH.exists():
        try:
            versions = json.loads(SWITCHER_PATH.read_text())
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {SWITCHER_PATH}: {e}", file=sys.stderr)
            return 1

        if not validate_switcher_json(versions):
            print(f"Error: {SWITCHER_PATH} has invalid structure", file=sys.stderr)
            return 1

        # Create backup before modifying
        backup_path = SWITCHER_PATH.with_suffix(".json.bak")
        shutil.copy2(SWITCHER_PATH, backup_path)
        print(f"Created backup: {backup_path}")
    else:
        versions = [{"name": "dev", "version": "dev", "url": f"{REPO_URL}/latest/"}]
        print(f"Bootstrapping new {SWITCHER_PATH}")

    # Remove preferred flag and "(stable)" suffix from all versions
    for v in versions:
        v.pop("preferred", None)
        if "(stable)" in v.get("name", ""):
            v["name"] = v["version"]

    # Check if version already exists
    existing = next((v for v in versions if v["version"] == version), None)
    if existing:
        existing["name"] = f"{version} (stable)"
        existing["preferred"] = True
    else:
        # Insert after dev entry (index 1)
        new_entry = {
            "name": f"{version} (stable)",
            "version": version,
            "preferred": True,
            "url": f"{REPO_URL}/{version}/",
        }
        versions.insert(1, new_entry)

    # Validate structure before writing
    if not validate_switcher_json(versions):
        print("Error: Generated switcher data failed validation", file=sys.stderr)
        return 1

    SWITCHER_PATH.write_text(json.dumps(versions, indent=2) + "\n")
    print(f"Updated {SWITCHER_PATH} with version {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
