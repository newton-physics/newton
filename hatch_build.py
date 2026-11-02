# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build release-matched documentation for distributable Newton wheels."""

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
            try:
                revision = subprocess.run(
                    ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
                )
                source_revision = revision.stdout.strip() if revision.returncode == 0 else None
            except FileNotFoundError:
                source_revision = None
            build = runpy.run_path(str(root / "docs/build_markdown.py"))["build"]
            source_digest = build(root, output)
            manifest = {
                "version": self.metadata.version,
                "source_digest": source_digest,
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
