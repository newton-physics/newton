# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load the licensed native jacket without changing its surface geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .assets import AssetMesh


def load_native_jacket():
    """Return the complete native mesh and its pinned source/license record."""
    folder = Path(__file__).with_name("assets")
    metadata = json.loads((folder / "cached_jacket_native.metadata.json").read_text())
    with np.load(folder / "cached_jacket_native.npz", allow_pickle=False) as arrays:
        vertices, triangles = arrays["vertices"], arrays["triangles"]
    digest = hashlib.sha256(vertices.tobytes() + triangles.tobytes()).hexdigest()
    if digest != metadata["geometry_sha256"]:
        raise ValueError("Bundled native jacket geometry does not match its provenance")
    return AssetMesh("cached_jacket_native", vertices, triangles, metadata=metadata)
