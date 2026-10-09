# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Best-effort disk cache for ordered convex decomposition parts.

Entries are content-addressed ``*.convex.npz`` files containing only arrays.
Bump ``CACHE_FORMAT_VERSION`` when the format or Newton's decomposition
pipeline changes in a way that affects the generated geometry. Backend
versions independently invalidate entries when backend defaults change.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from .types import Mesh

CACHE_FORMAT_VERSION = 1
logger = logging.getLogger(__name__)


def _json_scalar(value: Any) -> Any:
    """Accept NumPy scalars in otherwise JSON-serializable backend settings."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported cache setting type: {type(value).__name__}")


def hash_inputs(mesh: Mesh, method: str, settings: dict[str, Any]) -> str | None:
    """Hash geometry and effective settings, or disable caching if they cannot be keyed."""
    packages = ("coacd",) if method == "coacd" else ("trimesh", "vhacdx")
    try:
        metadata = json.dumps(
            {
                "format": CACHE_FORMAT_VERSION,
                "method": method,
                "versions": {package: importlib.metadata.version(package) for package in packages},
                "settings": settings,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_scalar,
        )
    except (importlib.metadata.PackageNotFoundError, TypeError, ValueError) as exc:
        logger.warning("Convex cache: cannot key decomposition; caching disabled: %s", exc)
        return None

    digest = hashlib.sha256(metadata.encode())
    for data in (mesh.vertices, mesh.indices):
        digest.update(data.dtype.str.encode())
        digest.update(np.asarray(data.shape, dtype="<i8").tobytes())
        digest.update(data.tobytes())
    return digest.hexdigest()


def try_load(cache_dir: str | os.PathLike[str], key: str) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """Return ordered parts, treating missing, incompatible, or corrupt entries as misses."""
    path = Path(cache_dir) / f"{key}.convex.npz"
    try:
        with path.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
            version = data["version"]
            count = data["part_count"]
            if version.shape != () or version.dtype != np.int64 or version.item() != CACHE_FORMAT_VERSION:
                return None
            if count.shape != () or count.dtype != np.int64 or count.item() <= 0:
                raise ValueError("invalid part count")
            if len(data.files) != 2 + 2 * count.item():
                raise ValueError("incomplete parts")
            parts = []
            for i in range(count.item()):
                vertices, faces = data[f"vertices_{i}"], data[f"faces_{i}"]
                if (
                    vertices.ndim != 2
                    or vertices.shape[1] != 3
                    or len(vertices) == 0
                    or vertices.dtype.kind != "f"
                    or not np.isfinite(vertices).all()
                    or faces.ndim != 2
                    or faces.shape[1] != 3
                    or len(faces) == 0
                    or faces.dtype.kind not in "iu"
                    or np.any(faces < 0)
                    or np.any(faces >= len(vertices))
                ):
                    raise ValueError(f"invalid geometry in part {i}")
                parts.append((vertices, faces))
            return parts
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as exc:
        logger.warning("Convex cache: failed to load %s: %s", path, exc)
        return None


def write(cache_dir: str | os.PathLike[str], key: str, parts: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Publish parts atomically; cache I/O failures must not abort mesh approximation."""
    if not parts:
        return
    path = Path(cache_dir) / f"{key}.convex.npz"
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "version": np.asarray(CACHE_FORMAT_VERSION, dtype=np.int64),
            "part_count": np.asarray(len(parts), dtype=np.int64),
        }
        for i, (vertices, faces) in enumerate(parts):
            arrays[f"vertices_{i}"] = np.asarray(vertices)
            arrays[f"faces_{i}"] = np.asarray(faces)
        # A unique file per writer prevents readers from seeing partial results.
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        os.replace(temporary, path)
    except OSError as exc:
        logger.warning("Convex cache: failed to write %s: %s", path, exc)
    finally:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink()
