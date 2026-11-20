# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate wheel documentation, delegating packaging and editables to uv_build."""

import json
import os
import runpy
import subprocess
import tarfile
import tempfile
from pathlib import Path

import uv_build as _backend


def __getattr__(name):
    return getattr(_backend, name)


def _configuration():
    try:
        import tomllib  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise RuntimeError("Build Newton wheels with Python 3.11 or newer.") from exc

    with Path("pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def get_requires_for_build_wheel(config_settings=None):
    """Request rendering tools only for distributable-wheel builds."""
    config = _configuration()
    return [
        *_backend.get_requires_for_build_wheel(config_settings),
        *config["project"]["dependencies"],
        *config["tool"]["newton"]["docs"]["requires"],
    ]


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Render into an isolated source snapshot and let uv_build assemble it."""
    root = Path.cwd()
    output = str(Path(wheel_directory).resolve())
    metadata = str(Path(metadata_directory).resolve()) if metadata_directory else None
    config = _configuration()
    revision = None
    if (root / ".git").exists():
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            revision = result.stdout.strip() if result.returncode == 0 else None
        except FileNotFoundError:
            pass
    with tempfile.TemporaryDirectory(prefix="newton-uv-wheel-") as temporary:
        staged = Path(temporary)
        # Reuse uv_build's source selection, including licenses and exclusions.
        archive_name = _backend.build_sdist(temporary, config_settings)
        with tarfile.open(staged / archive_name) as archive:
            archive.extractall(staged, filter="data")
        source = staged / archive_name.removesuffix(".tar.gz")
        docs = source / "newton/docs"
        build = runpy.run_path(str(source / "docs/build_markdown.py"))["build"]
        digest = build(source, docs)
        manifest = {"version": config["project"]["version"], "source_digest": digest, "source_revision": revision}
        (docs / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        try:
            os.chdir(source)
            return _backend.build_wheel(output, config_settings, metadata)
        finally:
            os.chdir(root)
