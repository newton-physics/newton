# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare a Cartesian reference bundle from complete Visual3D exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from projects.gait_c3d.c3d_adapter import lab_to_newton_rotation

from .data import validate as validate_reference
from .prepare_subject import _kabsch_rigid_transforms
from .profile import validate as validate_profile
from .visual3d import _ascii, load_visual3d_export


def _read_triplets(path: Path, manifest: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    labels, values = _ascii(path)
    if values.shape[1] != 3 * len(labels) or len(set(labels)) != len(labels):
        raise ValueError(f"{path} must contain unique XYZ triplets")
    scale = {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3}.get(str(manifest["marker_units"]).lower())
    if scale is None:
        raise ValueError("marker_units must be m, cm, or mm")
    points = values.reshape(len(values), len(labels), 3) * scale
    rotation = lab_to_newton_rotation(manifest.get("up_axis", "+Z"), manifest.get("forward_axis", "-Y"))
    valid = np.all(np.isfinite(points), axis=-1)
    points = points @ rotation.T
    return labels, points, valid


def _signal(path: Path, count: int) -> np.ndarray:
    _, values = _ascii(path)
    if values.shape != (count, 1):
        raise ValueError(f"{path} must contain one column and {count} samples")
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains nonfinite values")
    return values[:, 0]


def _joint_angles(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    """Read XYZ angles while skipping Visual3D's empty scalar placeholders."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 6:
        raise ValueError(f"{path} has no joint-angle samples")

    def fields(line: str) -> list[str]:
        return line.split("\t") if "\t" in line else line.split()

    labels, kinds, folders = (parts[1:] if parts[0] == "" else parts for parts in (fields(line) for line in lines[1:4]))
    components = fields(lines[4])
    if components[0] != "ITEM" or any(len(row) != len(components) - 1 for row in (labels, kinds, folders)):
        raise ValueError(f"{path} has inconsistent joint-angle headers")
    axes = components[1:]
    columns: list[int] = []
    names: list[str] = []
    index = 0
    while index < len(axes):
        if axes[index] == "0":
            index += 1
            continue
        group = range(index, index + 3)
        if axes[index : index + 3] != ["X", "Y", "Z"] or any(
            labels[j] != labels[index] or kinds[j] != "LINK_MODEL_BASED" or folders[j] != "ORIGINAL" for j in group
        ):
            raise ValueError(f"{path} must contain complete XYZ joint-angle groups")
        names.append(labels[index])
        columns.extend(group)
        index += 3
    if len(set(names)) != len(names):
        raise ValueError(f"{path} contains duplicate joint angles")
    values = []
    for line_number, line in enumerate(lines[5:], 6):
        row = fields(line)
        if len(row) != len(axes) + 1 or row[0] != str(len(values) + 1):
            raise ValueError(f"{path}:{line_number} has an invalid joint-angle row")
        try:
            values.append([float(row[j + 1]) for j in columns])
        except ValueError as error:
            raise ValueError(f"{path}:{line_number} has a nonnumeric joint angle") from error
    return tuple(names), np.asarray(values, dtype=np.float64)


def _lookup(labels: tuple[str, ...], names: tuple[str, ...], what: str) -> list[int]:
    indices = {name: i for i, name in enumerate(labels)}
    missing = [name for name in names if name not in indices]
    if missing:
        raise ValueError(f"{what} is missing required signals: {', '.join(missing)}")
    return [indices[name] for name in names]


def _joint_centers(root: Path, manifest: dict[str, Any], count: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    path = root / "motion_joint_centers.txt"
    if not path.exists():
        raise ValueError(f"missing required model-based export: {path}")
    labels, points, valid = _read_triplets(path, manifest)
    if len(points) != count:
        raise ValueError("motion_joint_centers.txt and motion targets have different sample counts")
    return labels, points, valid


def _static_points(root: Path, manifest: dict[str, Any]) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    path = root / "static_all_targets.txt"
    if not path.exists():
        raise ValueError(f"missing required static export: {path}")
    return _read_triplets(path, manifest)


def _static_centers(root: Path, manifest: dict[str, Any], count: int) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    path = root / "static_joint_centers.txt"
    if not path.exists():
        raise ValueError(f"missing required static model-based export: {path}")
    labels, points, valid = _read_triplets(path, manifest)
    if len(points) != count:
        raise ValueError("static joint-center export has an invalid sample count")
    return labels, points, valid


def _aliases(labels: tuple[str, ...], side: str) -> tuple[list[int], int, tuple[int, int]]:
    prefix = "L" if side == "left" else "R"
    cluster = [tuple(f"{prefix}{name}{i}" for i in range(1, 4)) for name in ("HEE", "CAL")]
    heel_names = next((names for names in cluster if all(name in labels for name in names)), None)
    if heel_names is None:
        raise ValueError(f"dynamic/static targets require {prefix}HEE1..3 or {prefix}CAL1..3")
    toe = next((name for name in (f"{prefix}TOE", f"{prefix}Toe") if name in labels), None)
    if toe is None:
        raise ValueError(f"missing {prefix}TOE target")
    mth_names = (f"{prefix}MTH1", f"{prefix}MTH5")
    if not all(name in labels for name in mth_names):
        mth_names = (f"{prefix}MT1H", f"{prefix}MT5H")
    if not all(name in labels for name in mth_names):
        raise ValueError(f"missing {prefix}MTH1/{prefix}MTH5 or {prefix}MT1H/{prefix}MT5H targets")
    mth_ids = _lookup(labels, mth_names, "metatarsal")
    return (
        [_lookup(labels, heel_names, "heel cluster")[i] for i in range(3)],
        labels.index(toe),
        (mth_ids[0], mth_ids[1]),
    )


def reconstruct_ground_pitch_from_cardan(
    q_knee: np.ndarray,
    q_ankle: np.ndarray,
    v_exp_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct foot ground pitch and shank inclination from 3D Visual3D Cardan angles and joint centers.

    Follows the convention R_relative = R_shank.T @ R_virtual_foot where R_relative is decomposed
    as an intrinsic Cardan sequence (flexion about lateral, abduction about anterior, axial about proximal).
    The calibrated virtual foot forward axis [1, 0, 0] is transported by R_foot = R_shank @ R_rel,
    and projected onto the sagittal simulation plane.

    Args:
        q_knee: Knee joint center position [m], shape (N, 2) or (N, 3).
        q_ankle: Ankle joint center position [m], shape (N, 2) or (N, 3).
        v_exp_deg: Exported 3D VirtualFoot angle [deg] (alpha, beta, gamma), shape (N, 3).

    Returns:
        ground_pitch_rad: Reconstructed foot ground pitch [rad] (+ toe-up, unwrapped), shape (N,).
        shank_inclination_rad: Shank tilt from vertical in simulation plane [rad] (+ forward tilt), shape (N,).
    """
    q_knee = np.asarray(q_knee, dtype=np.float64)
    q_ankle = np.asarray(q_ankle, dtype=np.float64)
    v_exp_deg = np.asarray(v_exp_deg, dtype=np.float64)
    n = len(q_knee)
    if len(q_ankle) != n or len(v_exp_deg) != n:
        raise ValueError("Sample counts for knee, ankle, and virtual foot angles must match")
    if v_exp_deg.ndim != 2 or v_exp_deg.shape[1] != 3:
        raise ValueError("v_exp_deg must have shape (N, 3)")

    pitches = np.empty(n, dtype=np.float64)
    shank_tilts = np.empty(n, dtype=np.float64)
    is_3d = q_knee.shape[1] == 3 and q_ankle.shape[1] == 3

    for i in range(n):
        if is_3d:
            delta = q_knee[i] - q_ankle[i]
            z_s = delta / np.linalg.norm(delta)
            y_ref = np.array([0.0, 1.0, 0.0])
            x_s = np.cross(y_ref, z_s)
            norm_x = np.linalg.norm(x_s)
            if norm_x < 1e-8:
                x_s = np.array([1.0, 0.0, 0.0])
            else:
                x_s /= norm_x
            y_s = np.cross(z_s, x_s)
            y_s /= np.linalg.norm(y_s)
            shank_tilts[i] = np.arctan2(delta[0], delta[2])
        else:
            dx = q_knee[i, 0] - q_ankle[i, 0]
            dz = q_knee[i, 1] - q_ankle[i, 1]
            shank_tilts[i] = np.arctan2(dx, dz)
            z_s = np.array([np.sin(shank_tilts[i]), 0.0, np.cos(shank_tilts[i])])
            y_s = np.array([0.0, 1.0, 0.0])
            x_s = np.array([np.cos(shank_tilts[i]), 0.0, -np.sin(shank_tilts[i])])

        R_shank = np.column_stack([x_s, y_s, z_s])

        alpha = np.deg2rad(v_exp_deg[i, 0])
        beta = np.deg2rad(v_exp_deg[i, 1])
        gamma = np.deg2rad(v_exp_deg[i, 2])

        R_alpha = np.array([
            [np.cos(alpha), 0.0, -np.sin(alpha)],
            [0.0, 1.0, 0.0],
            [np.sin(alpha), 0.0, np.cos(alpha)],
        ])
        R_beta = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(beta), -np.sin(beta)],
            [0.0, np.sin(beta), np.cos(beta)],
        ])
        R_gamma = np.array([
            [np.cos(gamma), -np.sin(gamma), 0.0],
            [np.sin(gamma), np.cos(gamma), 0.0],
            [0.0, 0.0, 1.0],
        ])

        R_rel = R_alpha @ R_beta @ R_gamma
        R_foot = R_shank @ R_rel
        v_fwd = R_foot @ np.array([1.0, 0.0, 0.0])
        pitches[i] = np.arctan2(v_fwd[2], v_fwd[0])

    return np.unwrap(pitches), shank_tilts


def prepare(
    dynamic_root: str | Path,
    static_root: str | Path,
    output: str | Path,
    profile_path: str | Path,
    shoe_path: str | Path,
    *,
    side: str,
    start_s: float,
    end_s: float,
    subject_mass_kg: float,
    belt_speed_m_s: float | None = None,
    virtual_foot_reference: str = "shank",
    shoe_static_pitch_rad: float | None = None,
) -> Path:
    """Prepare a validated reference bundle from a selected Visual3D window."""
    if (
        side not in {"left", "right"}
        or belt_speed_m_s is None
        or not np.isfinite([start_s, end_s, subject_mass_kg, belt_speed_m_s]).all()
        or end_s <= start_s
    ):
        raise ValueError("side, selected window, and subject mass are invalid")
    dynamic_root, static_root, output = Path(dynamic_root), Path(static_root), Path(output)
    valid_refs = {"shank", "ground", "reconstructed_ground", "raw_ground_deprecated"}
    if virtual_foot_reference not in valid_refs:
        raise ValueError(f"virtual_foot_reference must be one of {sorted(valid_refs)}")
    if virtual_foot_reference in {"ground", "reconstructed_ground", "raw_ground_deprecated"} and (
        shoe_static_pitch_rad is None or not np.isfinite(shoe_static_pitch_rad)
    ):
        raise ValueError("Ground-referenced input requires the fixed shoe static pitch")
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    validate_profile(profile)
    shoe_path = Path(shoe_path)
    if not shoe_path.is_file():
        raise ValueError(f"missing shoe artifact: {shoe_path}")
    trial = load_visual3d_export(dynamic_root)
    static_manifest = json.loads((static_root / "visual3d_manifest.json").read_text(encoding="utf-8"))
    labels, static, static_valid = _static_points(static_root, static_manifest)
    static_center_labels, static_centers, static_center_valid = _static_centers(
        static_root, static_manifest, len(static)
    )
    centers_labels, centers, centers_valid = _joint_centers(dynamic_root, trial.manifest, len(trial.marker_positions_m))
    if not np.array_equal(trial.marker_time_s.shape, centers[:, 0, 0].shape):
        raise ValueError("joint-center export length does not match marker clock")
    processed_path = dynamic_root / "motion_processed_targets.txt"
    if not processed_path.exists():
        raise ValueError("missing motion_processed_targets.txt; export PROCESSED targets explicitly")
    proc_labels, proc, proc_valid = _read_triplets(processed_path, trial.manifest)
    if proc.shape != trial.marker_positions_m.shape or proc_labels != trial.marker_names:
        raise ValueError("processed targets must have the same marker order and shape as motion_all_targets")
    side_prefix = "L" if side == "left" else "R"
    center_ids = _lookup(
        centers_labels, (f"{side_prefix}HIP", f"{side_prefix}KNEE", f"{side_prefix}ANKLE"), "joint centers"
    )
    heel_ids, toe_id, mth_ids = _aliases(proc_labels, side)
    static_center_ids = _lookup(
        static_center_labels,
        (f"{side_prefix}HIP", f"{side_prefix}KNEE", f"{side_prefix}ANKLE"),
        "static joint centers",
    )
    static_heel_ids, _static_toe_id, static_mth_ids = _aliases(labels, side)
    static_ankle = static_centers[:, static_center_ids[2]]
    if not np.all(static_center_valid[:, static_center_ids[2]]):
        raise ValueError("static_joint_centers.txt must contain finite anatomical ankle centers")
    static_heel = np.mean(static[:, static_heel_ids], axis=1)
    static_mth = 0.5 * (static[:, static_mth_ids[0]] + static[:, static_mth_ids[1]])
    if not np.all(static_valid[:, static_heel_ids]) or not np.all(static_valid[:, list(static_mth_ids)]):
        raise ValueError("static shoe registration targets contain missing values")
    pitch = float(
        np.arctan2((static_mth.mean(0) - static_heel.mean(0))[2], (static_mth.mean(0) - static_heel.mean(0))[0])
    )
    rotation = np.array([[np.cos(pitch), -np.sin(pitch)], [np.sin(pitch), np.cos(pitch)]])
    endpoint_local = rotation.T @ (static_mth.mean(0)[[0, 2]] - static_ankle.mean(0)[[0, 2]])
    lengths = np.array(
        [
            np.linalg.norm(
                (static_centers[:, static_center_ids[1]] - static_centers[:, static_center_ids[0]])[:, [0, 2]], axis=1
            ).mean(),
            np.linalg.norm(
                (static_centers[:, static_center_ids[2]] - static_centers[:, static_center_ids[1]])[:, [0, 2]], axis=1
            ).mean(),
        ]
    )
    mask = (trial.marker_time_s >= start_s) & (trial.marker_time_s <= end_s)
    if mask.sum() < 3:
        raise ValueError("selected window must contain at least three point samples")
    if not np.all(proc_valid[mask][:, [*heel_ids, toe_id, *mth_ids]]) or not np.all(centers_valid[mask][:, center_ids]):
        raise ValueError("selected window contains missing required processed targets or joint centers")
    # Anchor treadmill translation at the selected window to avoid a position
    # offset that depends on how long the recording ran before this stance.
    belt_shift = (trial.marker_time_s - trial.marker_time_s[mask][0])[:, None, None] * float(belt_speed_m_s)
    proc[:, :, 0] += belt_shift[:, 0, 0, None]
    centers[:, :, 0] += belt_shift[:, 0, 0, None]
    q = centers[mask][:, center_ids][:, :, [0, 2]]
    a0 = np.arctan2(q[:, 1, 1] - q[:, 0, 1], q[:, 1, 0] - q[:, 0, 0])
    a1 = np.arctan2(q[:, 2, 1] - q[:, 1, 1], q[:, 2, 0] - q[:, 1, 0])
    geom_knee = a1 - a0

    cluster_rotations, _, frame_rms, point_max = _kabsch_rigid_transforms(
        static[:, static_heel_ids].mean(0), proc[mask][:, heel_ids]
    )

    angles_path = dynamic_root / "motion_joint_angles.txt"
    joint_angles_source = "markers"
    virt_3d_deg = None
    knee_name = None
    knee_idx = None
    angle_values = None
    if angles_path.exists():
        angle_labels, angle_values = _joint_angles(angles_path)
        if len(angle_values) != len(trial.marker_positions_m):
            raise ValueError("motion_joint_angles.txt length does not match marker clock")

        # Resolve knee angle from Visual3D
        knee_names = (f"{side_prefix}KneeAngle", f"{side_prefix}Knee_Angle")
        knee_name = next((name for name in knee_names if name in angle_labels), None)
        if knee_name is not None:
            knee_idx = angle_labels.index(knee_name)
            knee_deg = angle_values[mask, knee_idx * 3]
            # In Newton mechanics, knee flexion is negative. If Visual3D uses clinical positive
            # flexion, negate so knee flexion is negative.
            if np.nanmedian(knee_deg) > 0:
                knee_angle = -np.deg2rad(knee_deg)
            else:
                knee_angle = np.deg2rad(knee_deg)
        else:
            knee_angle = geom_knee

        # Resolve foot / ankle angle from Visual3D Virtual Foot (neutral standing = 0)
        # or anatomical ankle angle (with static standing offset correction)
        virt_names = (
            f"{side_prefix}VirtualFootAngle",
            f"{side_prefix}VirualFootAngle",
        )
        virt_name = next((name for name in virt_names if name in angle_labels), None)
        static_angles_path = static_root / "static_joint_angles.txt"
        virt_3d_deg = None

        if virt_name is not None:
            ankle_idx = angle_labels.index(virt_name)
            virt_3d_deg = angle_values[mask, ankle_idx * 3 : ankle_idx * 3 + 3]
            ankle_deg = virt_3d_deg[:, 0]
            ankle_angle = np.deg2rad(ankle_deg)
            joint_angles_source = "visual3d_virtual_foot"
        else:
            anat_names = (f"{side_prefix}AnkleAngle", f"{side_prefix}CalAngle")
            anat_name = next((name for name in anat_names if name in angle_labels), None)
            if anat_name is not None:
                ankle_idx = angle_labels.index(anat_name)
                ankle_deg = angle_values[mask, ankle_idx * 3]
                static_offset = 0.0
                if static_angles_path.exists():
                    st_labels, st_values = _joint_angles(static_angles_path)
                    if anat_name in st_labels:
                        st_idx = st_labels.index(anat_name)
                        static_offset = float(np.nanmedian(st_values[:, st_idx * 3]))
                ankle_angle = np.deg2rad(ankle_deg - static_offset)
                joint_angles_source = "visual3d_ankle_offset_corrected"

    if joint_angles_source == "markers":
        if np.any(frame_rms > 0.002) or np.any(point_max > 0.003):
            raise ValueError("selected heel cluster fails rigid-registration QC (RMS 2 mm / point 3 mm)")
        static_heading = static_mth.mean(0) - static_heel.mean(0)
        transformed_heading = np.einsum("nij,j->ni", cluster_rotations, static_heading)
        foot_pitch = np.unwrap(np.arctan2(transformed_heading[:, 2], transformed_heading[:, 0]))
        knee_angle = a1 - a0
        ankle_angle = foot_pitch - a1 - np.pi / 2.0

    state = np.column_stack((q[:, 0], a0, knee_angle, ankle_angle))
    ground_target = None
    shank_inclination = None
    if virtual_foot_reference in {"ground", "reconstructed_ground"}:
        if joint_angles_source != "visual3d_virtual_foot" or virt_3d_deg is None:
            raise ValueError("Ground interpretation requires an exported virtual-foot angle")
        ground_target, shank_inclination = reconstruct_ground_pitch_from_cardan(
            centers[mask, center_ids[1]], centers[mask, center_ids[2]], virt_3d_deg
        )
        # Invert the existing carrier transform, preserving the fixed shoe mount.
        state[:, 4] = np.unwrap(ground_target + shoe_static_pitch_rad - (state[:, 2] + state[:, 3]) - np.pi / 2)
    elif virtual_foot_reference == "raw_ground_deprecated":
        if joint_angles_source != "visual3d_virtual_foot":
            raise ValueError("Ground interpretation requires an exported virtual-foot angle")
        ground_target = np.unwrap(ankle_angle.copy())
        state[:, 4] = np.unwrap(ground_target + shoe_static_pitch_rad - (state[:, 2] + state[:, 3]) - np.pi / 2)
    else:
        state[:, 4] = np.unwrap(state[:, 4])

    state[:, 2] = np.unwrap(state[:, 2])
    state[:, 3] = np.unwrap(state[:, 3])
    time = trial.marker_time_s[mask] - trial.marker_time_s[mask][0]
    velocity = np.gradient(state, time, axis=0, edge_order=2)
    analog_start = max(0, int(np.searchsorted(trial.analog_time_s, trial.marker_time_s[mask][0], side="left")) - 1)
    analog_end = min(
        len(trial.analog_time_s) - 1,
        int(np.searchsorted(trial.analog_time_s, trial.marker_time_s[mask][-1], side="right")),
    )
    force_mask = np.zeros(len(trial.analog_time_s), dtype=bool)
    force_mask[analog_start : analog_end + 1] = True
    force_time = trial.analog_time_s[force_mask] - trial.marker_time_s[mask][0]
    force = trial.force_n[force_mask][:, [0, 2]]
    if np.any(force[:, 1] < 0.0) or not np.any(force[:, 1] > 0.0):
        raise ValueError("selected force window contains negative upward force; verify Visual3D axis/sign metadata")
    if trial.manifest.get("force_side") != side:
        raise ValueError("manifest.force_side must explicitly identify the selected stance side")
    loaded = force_mask & (trial.force_n[:, 2] > 50.0)
    if not np.any(loaded) or not np.all(np.isfinite(trial.cop_m[loaded])):
        raise ValueError("selected stance contains nonfinite COP/support data")
    static_cluster = static[:, static_heel_ids].mean(axis=0)
    rigid_lengths = np.linalg.norm(static_cluster - static_cluster.mean(0), axis=-1)
    qc = {
        "static_marker_count": int(len(static_cluster)),
        "selected_frames": int(mask.sum()),
        "dynamic_cluster_residual_max_m": float(np.max(point_max)),
        "dynamic_cluster_rms_max_m": float(np.max(frame_rms)),
        "static_cluster_spread_m": float(np.max(rigid_lengths)),
    }
    sh_names = (f"{side_prefix}SH1", f"{side_prefix}SH2", f"{side_prefix}SH3", f"{side_prefix}SH4")
    if all(name in proc_labels for name in sh_names) and all(name in labels for name in sh_names):
        sh_proc_ids = [proc_labels.index(name) for name in sh_names]
        sh_static_ids = [labels.index(name) for name in sh_names]
        _, _, sh_rms, sh_pt_max = _kabsch_rigid_transforms(
            static[:, sh_static_ids].mean(0), proc[mask][:, sh_proc_ids]
        )
        qc["shank_tracking_rms_max_m"] = float(np.max(sh_rms))
        qc["shank_tracking_point_max_m"] = float(np.max(sh_pt_max))

    foot_model_names = (
        f"{side_prefix}CAL1",
        f"{side_prefix}CAL2",
        f"{side_prefix}CAL3",
        f"{side_prefix}MT1H",
        f"{side_prefix}MT5H",
        f"{side_prefix}TOE",
    )
    if all(name in proc_labels for name in foot_model_names) and all(name in labels for name in foot_model_names):
        ft_proc_ids = [proc_labels.index(name) for name in foot_model_names]
        ft_static_ids = [labels.index(name) for name in foot_model_names]
        _, _, ft_rms, ft_pt_max = _kabsch_rigid_transforms(
            static[:, ft_static_ids].mean(0), proc[mask][:, ft_proc_ids]
        )
        qc["foot_model_tracking_rms_max_m"] = float(np.max(ft_rms))
        qc["foot_model_tracking_point_max_m"] = float(np.max(ft_pt_max))

    profile = dict(profile)
    reference = {
        "time_s": time,
        "state": state,
        "velocity": velocity,
        "hip_target_m": state[:, :2],
        "joint_target_rad": state[:, 3:5].copy(),
        "lengths_m": lengths,
        "endpoint_local_m": endpoint_local,
        "static_pitch_rad": np.asarray(pitch),
        "static_ankle_m": static_ankle.mean(0)[[0, 2]],
        "static_heel_m": static_heel.mean(0)[[0, 2]],
        "subject_mass_kg": np.asarray(subject_mass_kg),
        "grf_time_s": force_time,
        "grf_target_n": force,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema": "cartesian_single_leg_1",
                    "side": side,
                    "foot_marker_names": ["heel_cluster", "toe", "mth"],
                    "source": "visual3d",
                    "joint_angles_source": joint_angles_source,
                }
            )
        ),
    }
    if ground_target is not None:
        reference["foot_ground_target_rad"] = ground_target
        reference["shoe_static_pitch_rad"] = np.asarray(shoe_static_pitch_rad)
    if shank_inclination is not None:
        reference["shank_inclination_rad"] = shank_inclination
    if virt_3d_deg is not None:
        reference["raw_virtual_foot_angle_deg"] = virt_3d_deg
        reference["raw_virtual_foot_angle_rad"] = np.deg2rad(virt_3d_deg)
    if angles_path.exists() and knee_name is not None:
        reference["raw_knee_angle_deg"] = angle_values[mask, knee_idx * 3 : knee_idx * 3 + 3]
    reference["geom_knee_angle_rad"] = geom_knee

    angle_convention = {
        "virtual_foot_reference": virtual_foot_reference,
        "ground_sign": "positive toe-up, negative toe-down",
        "selection": "explicit preparation argument; not inferred from signal name",
        "shoe_static_pitch_rad": shoe_static_pitch_rad,
        "reconstruction_method": "3D Cardan rotation matrix transport on calibrated forward axis"
        if virtual_foot_reference in {"ground", "reconstructed_ground"}
        else "none",
    }
    metadata = json.loads(str(reference["metadata_json"]))
    metadata["angle_convention"] = angle_convention
    metadata["selected_source_time_s"] = trial.marker_time_s[mask][[0, -1]].tolist()
    reference["metadata_json"] = np.asarray(json.dumps(metadata))
    if ground_target is not None:
        reference["foot_ground_target_rad"] = ground_target
        reference["shoe_static_pitch_rad"] = np.asarray(shoe_static_pitch_rad)
    validate_reference(reference)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "reference.npz", **reference)
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(shoe_path, output / "digital_shoe.json")
    consumed = [
        dynamic_root / "visual3d_manifest.json",
        dynamic_root / "motion_all_targets.txt",
        processed_path,
        dynamic_root / "motion_joint_centers.txt",
        static_root / "visual3d_manifest.json",
        static_root / "static_all_targets.txt",
        static_root / "static_joint_centers.txt",
        Path(profile_path),
        shoe_path,
    ]
    if (dynamic_root / "motion_joint_angles.txt").exists():
        consumed.append(dynamic_root / "motion_joint_angles.txt")
    if (static_root / "static_joint_angles.txt").exists():
        consumed.append(static_root / "static_joint_angles.txt")
    summary = {
        "schema": "visual3d_cartesian_preparation_1",
        "status": "input_prepared_uncertified",
        "accepted": False,
        "side": side,
        "selected_window_s": [float(start_s), float(end_s)],
        "subject_mass_kg": float(subject_mass_kg),
        "belt_speed_m_s": belt_speed_m_s,
        "joint_angles_source": joint_angles_source,
        "angle_convention": angle_convention,
        "quality": qc,
        "source": {"dynamic_root": str(dynamic_root.resolve()), "static_root": str(static_root.resolve())},
        "source_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in consumed},
        "runtime": {"fit_ready": False, "fit_config": None, "simulation_config": None, "shoe_mount_m": None},
        "reference": {
            "file": str((output / "reference.npz").resolve()),
            "sha256": hashlib.sha256((output / "reference.npz").read_bytes()).hexdigest(),
        },
        "profile": {"file": str((output / "profile.json").resolve())},
        "digital_shoe": {"file": str((output / "digital_shoe.json").resolve())},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dynamic-root", "static-root", "output", "profile", "shoe"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float, required=True)
    parser.add_argument("--subject-mass-kg", type=float, required=True)
    parser.add_argument("--belt-speed-m-s", type=float)
    parser.add_argument(
        "--virtual-foot-reference",
        choices=("shank", "ground", "reconstructed_ground", "raw_ground_deprecated"),
        default="shank",
    )
    parser.add_argument("--shoe-static-pitch-rad", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    print(
        prepare(
            args.dynamic_root,
            args.static_root,
            args.output,
            args.profile,
            args.shoe,
            side=args.side,
            start_s=args.start_s,
            end_s=args.end_s,
            subject_mass_kg=args.subject_mass_kg,
            belt_speed_m_s=args.belt_speed_m_s,
            virtual_foot_reference=args.virtual_foot_reference,
            shoe_static_pitch_rad=args.shoe_static_pitch_rad,
        )
    )


if __name__ == "__main__":
    main()
