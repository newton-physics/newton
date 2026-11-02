# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build release-matched documentation for distributable Newton wheels."""

import hashlib
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class DocumentationBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        if version == "editable":
            return
        root = Path(self.root)
        self.temporary = tempfile.TemporaryDirectory(prefix="newton-wheel-docs-")
        output = Path(self.temporary.name) / "docs"
        try:
            build = runpy.run_path(str(root / "docs/build_markdown.py"))["build"]
            build(root, output)
            digest = hashlib.sha256()
            inputs = [root / "pyproject.toml", *root.glob("*.py"), *root.glob("*.rst"), *root.glob("*.md")]
            for directory in ("newton", "docs"):
                inputs.extend(
                    path
                    for path in (root / directory).rglob("*")
                    if path.is_file()
                    and not any(part in ("_build", "__pycache__", ".cache", "sphinx-env") for part in path.parts)
                    and not (directory == "docs" and "_generated" in path.parts)
                )
            for path in sorted(set(inputs)):
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes())
                digest.update(b"\0")
            try:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
                )
                source_revision = revision.stdout.strip() if revision.returncode == 0 else None
            except FileNotFoundError:
                source_revision = None
            manifest = {
                "version": self.metadata.version,
                "source_digest": digest.hexdigest(),
                "source_revision": source_revision,
            }
            (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            build_data["force_include"][str(output)] = "newton/docs"
        except BaseException:
            self.temporary.cleanup()
            raise

    def finalize(self, version, build_data, artifact_path):
        if version != "editable":
            self.temporary.cleanup()
