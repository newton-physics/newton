# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate and load the frozen measured Cartesian reference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_SCHEMA = "cartesian_single_leg_1"
_SIDES = ("left", "right")
_REQUIRED = (
    "time_s",
    "state",
    "velocity",
    "hip_target_m",
    "joint_target_rad",
    "lengths_m",
    "endpoint_local_m",
    "static_pitch_rad",
    "static_ankle_m",
    "static_heel_m",
    "subject_mass_kg",
    "grf_time_s",
    "grf_target_n",
    "metadata_json",
)


def _json_scalar(value: np.ndarray | str, name: str = "metadata_json") -> dict:
    """Decode a scalar JSON string and require an object."""
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "US":
        raise ValueError(f"{name} must be a scalar string")
    try:
        metadata = json.loads(str(array))
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must contain valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return metadata


def validate(reference: dict[str, np.ndarray]) -> None:
    """Validate single-leg reference clocks, shapes, values, and metadata.

    Args:
        reference: NumPy arrays in schema ``cartesian_single_leg_1``. Subject
            mass [kg] is context only; it is not the simulated mass total.
    """
    for name in _REQUIRED:
        if name not in reference:
            raise ValueError(f"Missing reference field: {name}")
    metadata = _json_scalar(reference["metadata_json"])
    if metadata.get("schema") != _SCHEMA:
        raise ValueError("Unsupported Cartesian reference schema")
    if metadata.get("side") not in _SIDES:
        raise ValueError("Reference metadata must identify the selected side")
    for name, value in reference.items():
        allowed = "US" if name == "metadata_json" else "iuf"
        if not isinstance(value, np.ndarray) or value.dtype.kind not in allowed:
            raise ValueError(f"Invalid reference array type: {name}")
        if name != "metadata_json" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite reference array: {name}")
    time = reference["time_s"]
    if time.ndim != 1 or len(time) < 3 or time[0] != 0 or np.any(np.diff(time) <= 0):
        raise ValueError("time_s must start at zero and increase strictly for at least three frames")
    n = len(time)
    force_time = reference["grf_time_s"]
    if force_time.ndim != 1 or len(force_time) < 2 or np.any(np.diff(force_time) <= 0):
        raise ValueError("grf_time_s must increase strictly for at least two samples")
    if force_time[0] > time[0] or force_time[-1] < time[-1]:
        raise ValueError("Native GRF must cover the entire reference window")
    shapes = {
        "state": (n, 5),
        "velocity": (n, 5),
        "hip_target_m": (n, 2),
        "joint_target_rad": (n, 2),
        "lengths_m": (2,),
        "endpoint_local_m": (2,),
        "static_pitch_rad": (),
        "static_ankle_m": (2,),
        "static_heel_m": (2,),
        "subject_mass_kg": (),
        "grf_target_n": (len(force_time), 2),
        "metadata_json": (),
    }
    for name, shape in shapes.items():
        if reference[name].shape != shape:
            raise ValueError(f"Invalid shape for {name}: expected {shape}")
    if np.any(reference["lengths_m"] <= 0) or float(reference["subject_mass_kg"]) <= 0:
        raise ValueError("Segment lengths and context subject mass must be positive")
    if np.any(reference["grf_target_n"][:, 1] < 0):
        raise ValueError("Measured upward GRF must be nonnegative")
    ground_keys = ("foot_ground_target_rad", "shoe_static_pitch_rad")
    if any(key in reference for key in ground_keys):
        if not all(key in reference for key in ground_keys):
            raise ValueError("Ground foot angles require the fixed shoe pitch")
        if reference[ground_keys[0]].shape != (n,) or reference[ground_keys[1]].shape != ():
            raise ValueError("Ground foot angles must match the motion clock and shoe pitch must be scalar")
        state = reference["state"]
        reconstructed = state[:, 2] + state[:, 3] + state[:, 4] + np.pi / 2 - reference[ground_keys[1]]
        if not np.allclose(reconstructed, reference[ground_keys[0]], rtol=0, atol=1e-10):
            raise ValueError("Reference state does not reconstruct the declared ground foot angle")
    if "cop_target_m" in reference and reference["cop_target_m"].shape != (len(force_time),):
        raise ValueError("cop_target_m must match the GRF clock")
    if "pelvis_target_rad" in reference and reference["pelvis_target_rad"].shape != (n,):
        raise ValueError("pelvis_target_rad must match the motion clock")
    if "ankle_target_m" in reference and reference["ankle_target_m"].shape != (n, 2):
        raise ValueError("ankle_target_m must have shape [frames, 2]")
    marker_keys = ("foot_marker_target_m", "foot_marker_local_m")
    if any(name in reference for name in marker_keys):
        if not all(name in reference for name in marker_keys):
            raise ValueError("Foot marker targets and local points must be provided together")
        local = reference["foot_marker_local_m"]
        if local.ndim != 2 or local.shape[1] != 2 or len(local) == 0:
            raise ValueError("foot_marker_local_m must have shape [marker_count, 2]")
        if reference["foot_marker_target_m"].shape != (n, len(local), 2):
            raise ValueError("foot_marker_target_m must have shape [frame_count, marker_count, 2]")
        names = metadata.get("foot_marker_names")
        if not isinstance(names, list) or len(names) != len(local) or any(not isinstance(x, str) for x in names):
            raise ValueError("metadata.foot_marker_names must identify every foot marker")


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a Cartesian reference NPZ without pickle."""
    with np.load(Path(path), allow_pickle=False) as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    validate(reference)
    return reference
