# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Versioned public initial states and commands for the four H1 recipes."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_input(case_id):
    """Verify and load an input fixture; never restore solver-owned history."""
    if case_id not in ("j3_h1_jacket", "r7_h1_towel", "r7_h1_sheet", "r7_h1_gusset_bag"):
        raise ValueError(f"No H1 input fixture for {case_id}")
    folder = Path(__file__).with_name("assets") / "h1"
    metadata = json.loads((folder / f"{case_id}.json").read_text())
    path = folder / f"{case_id}.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["npz_sha256"]:
        raise ValueError(f"H1 input fixture checksum mismatch: {case_id}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    if arrays.keys() != metadata["arrays"].keys():
        raise ValueError("H1 input array set differs from the versioned manifest")
    for key, array in arrays.items():
        expected = metadata["arrays"][key]
        if (
            list(array.shape) != expected["shape"]
            or str(array.dtype) != expected["dtype"]
            or hashlib.sha256(array.tobytes()).hexdigest() != expected["sha256"]
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"Invalid H1 input array: {key}")
    return metadata, arrays
