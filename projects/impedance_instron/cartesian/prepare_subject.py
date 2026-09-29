# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare a reusable Cartesian single-leg input bundle from one gait subject."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from projects.gait_c3d.c3d_adapter import lab_to_newton_rotation, load_marker_artifact, read_c3d_markers
from projects.gait_c3d.native_motion_fit import load_native_motion_artifact
from projects.gait_c3d.treadmill import belt_motion, load_treadmill_log

ROOT = Path(__file__).resolve().parents[3]
BASELINE_BUNDLE = ROOT / "outputs/impedance_instron/baseline12_maxwell"
STATIC_MARKER_NAMES = (
    "LHEE",
    "LHEE2",
    "LHEE3",
    "LMTH1",
    "LMTH5",
    "LHLX",
    "LTOE",
    "LANK",
    "LMANK",
)
DYNAMIC_MARKER_NAMES = (
    *STATIC_MARKER_NAMES,
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "LKNE",
    "LMKNE",
    "RHEE",
    "RHEE2",
    "RHEE3",
    "RMTH1",
    "RMTH5",
    "RHLX",
    "RTOE",
)
MOTION_MARKER_MAP = {
    "LASI": "L.ASIS",
    "RASI": "R.ASIS",
    "LPSI": "L.PSIS",
    "RPSI": "R.PSIS",
    "LKNE": "L.Knee.Lat",
    "LMKNE": "L.Knee.Med",
    "LANK": "L.Ankle.Lat",
    "LMANK": "L.Ankle.Med",
    "LHEE": "L.Heel",
    "LTOE": "L.Toe.Lat",
    "RHEE": "R.Heel",
    "RTOE": "R.Toe.Lat",
}
CODA_LEFT_HIP_OFFSET = np.array([-0.36, -0.19, -0.30], dtype=np.float64)
FORCE_THRESHOLD_N = 50.0
CONTACT_GATE_N = 20.0
EPISODE_MIN_DURATION_S = 0.08
EPISODE_MIN_PEAK_N = 200.0
BRIDGE_S = 0.01
FORCE_FILTER_SAMPLES = 21
FILTER_ORDER = 4
FILTER_CUTOFF_HZ = 20.0
FILTER_PAD_MULTIPLE = 3.0
FILTER_MIN_PAD = 15
WINDOW_PADDING_FRAMES = 3
TARE_TOTAL_UP_LIMIT_N = 30.0
CONSTANT_SPEED_MARGIN_S = 0.13
CONSTANT_SPEED_TOLERANCE_M_S = 1.0e-6
SYNC_GUARD_S = 0.05
RIGID_FRAME_RMS_LIMIT_M = 0.002
RIGID_POINT_MAX_LIMIT_M = 0.003
SIDE_DISTANCE_MARGIN_M = 0.30
SIDE_NEAREST_FRACTION = 0.95
OTHER_FOOT_BULK_MIN_HEIGHT_M = 0.05
OTHER_FOOT_HEEL_MIN_HEIGHT_M = 0.10
SUMMARY_SCHEMA = "cartesian_subject_preparation_1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation(angle_rad: np.ndarray) -> np.ndarray:
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.stack(
        (
            np.stack((cosine, -sine), axis=-1),
            np.stack((sine, cosine), axis=-1),
        ),
        axis=-2,
    )


def _require_finite(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _padlen(rate_hz: float) -> int:
    return max(FILTER_MIN_PAD, int(math.ceil(FILTER_PAD_MULTIPLE * rate_hz / FILTER_CUTOFF_HZ)))


def _butterworth_filter(
    values: np.ndarray, rate_hz: float, *, unwrap: bool = False
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.signal import butter, sosfiltfilt

    array = _require_finite("filter input", values)
    if array.ndim == 0 or len(array) <= _padlen(rate_hz):
        raise ValueError("input is too short for the frozen zero-phase low-pass filter")
    source = np.unwrap(array, axis=0) if unwrap else array
    sos = butter(FILTER_ORDER, FILTER_CUTOFF_HZ, btype="low", fs=rate_hz, output="sos")
    padlen = _padlen(rate_hz)
    filtered = sosfiltfilt(sos, source, axis=0, padtype="odd", padlen=padlen)
    return filtered, {
        "family": "Butterworth",
        "order": FILTER_ORDER,
        "effective_order": 2 * FILTER_ORDER,
        "cutoff_hz": FILTER_CUTOFF_HZ,
        "cutoff_definition": "single-pass -3 dB; combined -6 dB; no compensation",
        "passes": "forward/backward",
        "phase": "zero",
        "implementation": "scipy.signal.butter(output=sos) + sosfiltfilt(axis=0)",
        "sos": sos.tolist(),
        "sampling_rate_hz": float(rate_hz),
        "padlen_samples": padlen,
        "padding_rule": "max(15 samples, ceil(3 * native_rate / cutoff)); reject shorter inputs",
        "padtype": "odd",
        "angular_handling": "unwrap along time before filtering" if unwrap else "no angular unwrap",
    }


def _finite_difference(values: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    return np.gradient(_require_finite("values", values), _require_finite("time_s", time_s), axis=0, edge_order=2)


def _marker_indices(names: tuple[str, ...], required: tuple[str, ...]) -> dict[str, int]:
    lookup = {name: i for i, name in enumerate(names)}
    missing = [name for name in required if name not in lookup]
    if missing:
        raise ValueError(f"missing required markers: {missing}")
    return {name: lookup[name] for name in required}


def _static_means(
    positions: np.ndarray, valid: np.ndarray, indices: dict[str, int], names: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    complete = np.all(valid[:, [indices[name] for name in names]], axis=1)
    if not np.any(complete):
        raise ValueError("no complete static frames contain the required markers")
    means = {name: positions[complete, indices[name]].mean(axis=0).astype(np.float64) for name in names}
    return means, np.flatnonzero(complete)


def _hip_center_from_pelvis(markers: dict[str, np.ndarray], asis_width_m: float) -> np.ndarray:
    asis_mid = 0.5 * (markers["LASI"] + markers["RASI"])
    psis_mid = 0.5 * (markers["LPSI"] + markers["RPSI"])
    right = markers["RASI"] - markers["LASI"]
    right /= np.linalg.norm(right)
    anterior = asis_mid - psis_mid
    anterior -= right * float(np.dot(anterior, right))
    anterior /= np.linalg.norm(anterior)
    up = np.cross(right, anterior)
    basis = np.column_stack((right, anterior, up))
    return asis_mid + basis @ (asis_width_m * CODA_LEFT_HIP_OFFSET)


def _planar_xz(points: np.ndarray) -> np.ndarray:
    return _require_finite("planar points", points)[..., (0, 2)]


def _fit_state(
    hip_target_m: np.ndarray,
    knee_target_m: np.ndarray,
    ankle_target_m: np.ndarray,
    foot_pitch_rad: np.ndarray,
    lengths_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(hip_target_m)
    state = np.empty((count, 5), dtype=np.float64)
    reference = np.empty((count, 3, 2), dtype=np.float64)
    residual = np.empty((count, 3, 2), dtype=np.float64)
    iterations = np.empty(count, dtype=np.int32)
    guess = None
    for i in range(count):
        hip = hip_target_m[i]
        target = np.concatenate((knee_target_m[i] - hip, ankle_target_m[i] - hip))
        if guess is None:
            first = math.atan2(knee_target_m[i, 1] - hip[1], knee_target_m[i, 0] - hip[0])
            second = math.atan2(ankle_target_m[i, 1] - knee_target_m[i, 1], ankle_target_m[i, 0] - knee_target_m[i, 0])
            guess = np.array([first, second], dtype=np.float64)
        solved = guess.copy()
        for _iteration in range(32):
            a0, a1 = solved
            model = np.array(
                [
                    lengths_m[0] * math.cos(a0),
                    lengths_m[0] * math.sin(a0),
                    lengths_m[0] * math.cos(a0) + lengths_m[1] * math.cos(a1),
                    lengths_m[0] * math.sin(a0) + lengths_m[1] * math.sin(a1),
                ],
                dtype=np.float64,
            )
            error = model - target
            jacobian = np.array(
                [
                    [-lengths_m[0] * math.sin(a0), 0.0],
                    [lengths_m[0] * math.cos(a0), 0.0],
                    [-lengths_m[0] * math.sin(a0), -lengths_m[1] * math.sin(a1)],
                    [lengths_m[0] * math.cos(a0), lengths_m[1] * math.cos(a1)],
                ],
                dtype=np.float64,
            )
            lhs = jacobian.T @ jacobian + 1.0e-8 * np.eye(2)
            step = np.linalg.solve(lhs, jacobian.T @ error)
            solved -= step
            if np.linalg.norm(step) <= 1.0e-12:
                break
        guess = solved
        a0, a1 = solved
        knee = hip + lengths_m[0] * np.array([math.cos(a0), math.sin(a0)])
        ankle = knee + lengths_m[1] * np.array([math.cos(a1), math.sin(a1)])
        foot = foot_pitch_rad[i]
        state[i] = (hip[0], hip[1], a0, a1 - a0, foot - a1 - math.pi / 2.0)
        reference[i] = np.stack((hip, knee, ankle))
        residual[i] = np.stack((hip - hip_target_m[i], knee - knee_target_m[i], ankle - ankle_target_m[i]))
        iterations[i] = iteration + 1
    return state, reference, residual, iterations


def _extract_force_platforms(c3d_path: Path) -> dict[str, Any]:
    try:
        import ezc3d  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - optional offline dependency
        raise ImportError("Run with `uv run --with ezc3d==1.7.2 ...` to decode force plates") from error

    c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    platforms = c3d["data"]["platform"]
    if len(platforms) == 0:
        raise ValueError(f"no force platforms found in {c3d_path}")
    point_rate_hz = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    analog_rate_hz = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    rotation = lab_to_newton_rotation("+Z", "-Y")

    def _position_scale(unit: str) -> float:
        text = unit.strip().lower()
        if text == "m":
            return 1.0
        if text == "mm":
            return 1.0e-3
        raise ValueError(f"unsupported force-platform position unit {unit!r}")

    def _moment_scale(unit: str) -> float:
        text = unit.strip().lower().replace(" ", "")
        if text in {"nm", "n*m"}:
            return 1.0
        if text == "nmm":
            return 1.0e-3
        raise ValueError(f"unsupported force-platform moment unit {unit!r}")

    plate_force = []
    plate_moment = []
    plate_cop = []
    for platform in platforms:
        if str(platform["unit_force"]).strip() != "N":
            raise ValueError("force-platform extraction must already be in newtons")
        pos_scale = _position_scale(str(platform["unit_position"]))
        moment_scale = _moment_scale(str(platform["unit_moment"]))
        force = np.asarray(platform["force"], dtype=np.float64).T @ rotation.T
        moment = (np.asarray(platform["moment"], dtype=np.float64).T * moment_scale) @ rotation.T
        cop = (np.asarray(platform["center_of_pressure"], dtype=np.float64).T * pos_scale) @ rotation.T
        plate_force.append(force)
        plate_moment.append(moment)
        plate_cop.append(cop)
    force = np.stack(plate_force, axis=1)
    moment = np.stack(plate_moment, axis=1)
    cop = np.stack(plate_cop, axis=1)

    channel_indices = np.asarray(c3d["parameters"]["FORCE_PLATFORM"]["CHANNEL"]["value"], dtype=np.int32)
    analog_labels = tuple(str(item) for item in c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    if channel_indices.ndim != 2:
        raise ValueError("FORCE_PLATFORM:CHANNEL must be a 2-D channel index table")
    channel_labels = tuple(
        tuple(analog_labels[int(channel_indices[row, column]) - 1] for row in range(channel_indices.shape[0]))
        for column in range(channel_indices.shape[1])
    )
    corners_lab_m = np.moveaxis(
        np.asarray(c3d["parameters"]["FORCE_PLATFORM"]["CORNERS"]["value"], dtype=np.float64), 0, -1
    ) * _position_scale("mm")
    origin_lab_m = np.asarray(
        c3d["parameters"]["FORCE_PLATFORM"]["ORIGIN"]["value"], dtype=np.float64
    ).T * _position_scale("mm")
    corners_newton_m = corners_lab_m @ rotation.T
    mean_corners_newton_m = corners_newton_m.mean(axis=0)
    origin_newton_m = origin_lab_m @ rotation.T
    return {
        "point_rate_hz": point_rate_hz,
        "analog_rate_hz": analog_rate_hz,
        "rotation": rotation,
        "force": force,
        "moment": moment,
        "cop": cop,
        "channel_indices": channel_indices,
        "channel_labels": channel_labels,
        "corners_lab_m": corners_lab_m,
        "corners_newton_m": corners_newton_m,
        "mean_corners_newton_m": mean_corners_newton_m,
        "origin_lab_m": origin_lab_m,
        "origin_newton_m": origin_newton_m,
        "type": [int(value) for value in c3d["parameters"]["FORCE_PLATFORM"]["TYPE"]["value"]],
    }


def _hann_filter(values: np.ndarray) -> np.ndarray:
    array = _require_finite("Hann filter input", values)
    if len(array) < FORCE_FILTER_SAMPLES:
        raise ValueError("force array is too short for the frozen 21-sample Hann filter")
    kernel = np.hanning(FORCE_FILTER_SAMPLES)
    kernel /= kernel.sum()
    flat = array.reshape(len(array), -1)
    filtered = np.empty_like(flat)
    for i in range(flat.shape[1]):
        filtered[:, i] = np.convolve(flat[:, i], kernel, mode="same")
    return filtered.reshape(array.shape)


def _bridge_contact(mask: np.ndarray, bridge_samples: int) -> np.ndarray:
    active = np.asarray(mask, dtype=bool).copy()
    start = 0
    while start < len(active):
        if active[start]:
            start += 1
            continue
        stop = start
        while stop < len(active) and not active[stop]:
            stop += 1
        if start > 0 and stop < len(active) and active[start - 1] and active[stop] and stop - start <= bridge_samples:
            active[start:stop] = True
        start = stop
    return active


def _shift_moment_to_common_origin(moment: np.ndarray, force: np.ndarray, origin_m: np.ndarray) -> np.ndarray:
    moment_array = _require_finite("moment", moment)
    force_array = _require_finite("force", force)
    origin_array = _require_finite("origin_m", origin_m)
    if moment_array.shape != force_array.shape or moment_array.ndim != 3 or moment_array.shape[-1] != 3:
        raise ValueError("moment and force must both have shape [sample_count, plate_count, 3]")
    if origin_array.shape != (moment_array.shape[1], 3):
        raise ValueError("origin_m must have shape [plate_count, 3]")
    return moment_array + np.cross(origin_array[None, :, :], force_array, axis=-1)


def _combine_cop(force: np.ndarray, moment: np.ndarray) -> np.ndarray:
    total_force = _require_finite("total force", force)
    total_moment = _require_finite("total moment", moment)
    if total_force.shape != total_moment.shape or total_force.ndim != 2 or total_force.shape[1] != 3:
        raise ValueError("force and moment must both have shape [sample_count, 3]")
    combined = np.full((len(total_force), 3), np.nan, dtype=np.float64)
    positive = total_force[:, 2] > 0.0
    combined[positive, 0] = -total_moment[positive, 1] / total_force[positive, 2]
    combined[positive, 1] = total_moment[positive, 0] / total_force[positive, 2]
    combined[positive, 2] = 0.0
    return combined


def _distance_to_interval(x: np.ndarray, interval: np.ndarray) -> np.ndarray:
    return np.maximum(interval[:, 0] - x, 0.0) + np.maximum(x - interval[:, 1], 0.0)


def _dynamic_support_metrics(
    markers: np.ndarray,
    valid: np.ndarray,
    index: dict[str, int],
    analog_time_s: np.ndarray,
    marker_time_s: np.ndarray,
) -> dict[str, np.ndarray]:
    point_index = np.clip(np.searchsorted(marker_time_s, analog_time_s, side="left"), 0, len(marker_time_s) - 1)

    def foot(names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        marker_ids = [index[name] for name in names]
        return (
            markers[point_index[:, None], marker_ids].astype(np.float64),
            valid[point_index[:, None], marker_ids].astype(bool),
        )

    left, left_valid = foot(("LHEE", "LHEE2", "LHEE3", "LMTH1", "LMTH5", "LHLX", "LTOE"))
    right, right_valid = foot(("RHEE", "RHEE2", "RHEE3", "RMTH1", "RMTH5", "RHLX", "RTOE"))
    left_complete = np.all(left_valid, axis=1)
    right_complete = np.all(right_valid, axis=1)

    def interval(points: np.ndarray, complete: np.ndarray) -> np.ndarray:
        bounds = np.full((len(points), 2), np.nan, dtype=np.float64)
        bounds[complete, 0] = points[complete, :, 0].min(axis=1)
        bounds[complete, 1] = points[complete, :, 0].max(axis=1)
        return bounds

    def minimum_height(points: np.ndarray, complete: np.ndarray) -> np.ndarray:
        height = np.full(len(points), np.nan, dtype=np.float64)
        height[complete] = points[complete, :, 2].min(axis=1)
        return height

    def minimum_heel_height(points: np.ndarray, complete: np.ndarray) -> np.ndarray:
        height = np.full(len(points), np.nan, dtype=np.float64)
        height[complete] = points[complete, :3, 2].min(axis=1)
        return height

    return {
        "left_xy": left[..., :2],
        "right_xy": right[..., :2],
        "left_interval": interval(left, left_complete),
        "right_interval": interval(right, right_complete),
        "left_complete": left_complete,
        "right_complete": right_complete,
        "left_bulk_min_height": minimum_height(left, left_complete),
        "right_bulk_min_height": minimum_height(right, right_complete),
        "left_heel_min_height": minimum_heel_height(left, left_complete),
        "right_heel_min_height": minimum_heel_height(right, right_complete),
    }


def _kabsch_rigid_transforms(
    source_points_m: np.ndarray, target_points_m: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = _require_finite("source_points_m", source_points_m)
    target = _require_finite("target_points_m", target_points_m)
    if source.shape[0] < 3 or source.shape[-1] != 3 or target.ndim != 3 or target.shape[1:] != source.shape:
        raise ValueError("Kabsch rigid fit expects source [point_count, 3] and target [frame_count, point_count, 3]")
    source_centroid = source.mean(axis=0)
    source_centered = source - source_centroid
    rotations = np.empty((len(target), 3, 3), dtype=np.float64)
    translations = np.empty((len(target), 3), dtype=np.float64)
    frame_rms = np.empty(len(target), dtype=np.float64)
    point_max = np.empty(len(target), dtype=np.float64)
    for i, points in enumerate(target):
        target_centroid = points.mean(axis=0)
        target_centered = points - target_centroid
        covariance = source_centered.T @ target_centered
        left, _, right_t = np.linalg.svd(covariance)
        rotation = right_t.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right_t[-1] *= -1.0
            rotation = right_t.T @ left.T
        translation = target_centroid - rotation @ source_centroid
        fitted = (rotation @ source.T).T + translation
        residual = points - fitted
        rotations[i] = rotation
        translations[i] = translation
        point_norm = np.linalg.norm(residual, axis=1)
        frame_rms[i] = math.sqrt(float(np.mean(np.sum(np.square(residual), axis=1))))
        point_max[i] = float(np.max(point_norm))
    return rotations, translations, frame_rms, point_max


def _window_treadmill_metrics(
    treadmill: Any,
    treadmill_motion: Any,
    start_time_s: float,
    end_time_s: float,
) -> dict[str, Any]:
    guard_total_s = CONSTANT_SPEED_MARGIN_S + SYNC_GUARD_S
    start_log_s = start_time_s + float(treadmill_motion.offset) - guard_total_s
    end_log_s = end_time_s + float(treadmill_motion.offset) + guard_total_s
    metrics = {
        "guard_total_s": guard_total_s,
        "constant_speed_check_margin_s": CONSTANT_SPEED_MARGIN_S,
        "sync_guard_s": SYNC_GUARD_S,
        "constant_speed_tolerance_m_s": CONSTANT_SPEED_TOLERANCE_M_S,
        "log_interval_s": [start_log_s, end_log_s],
        "covered": False,
        "positive": False,
        "constant": False,
        "tied": False,
        "accepted": False,
    }
    if start_log_s < float(treadmill.t[0]) or end_log_s > float(treadmill.t[-1]):
        return metrics
    mask = (treadmill.t >= start_log_s) & (treadmill.t <= end_log_s)
    if np.count_nonzero(mask) < 2:
        return metrics
    left_speed = treadmill.left_speed[mask]
    right_speed = treadmill.right_speed[mask]
    left_range = float(np.max(left_speed) - np.min(left_speed))
    right_range = float(np.max(right_speed) - np.min(right_speed))
    tied_range = float(np.max(np.abs(left_speed - right_speed)))
    metrics.update(
        {
            "covered": True,
            "positive": bool(np.all(left_speed > 0.0) and np.all(right_speed > 0.0)),
            "constant": bool(
                left_range <= CONSTANT_SPEED_TOLERANCE_M_S and right_range <= CONSTANT_SPEED_TOLERANCE_M_S
            ),
            "tied": bool(tied_range <= CONSTANT_SPEED_TOLERANCE_M_S),
            "left_speed_range_m_s": [float(np.min(left_speed)), float(np.max(left_speed))],
            "right_speed_range_m_s": [float(np.min(right_speed)), float(np.max(right_speed))],
            "tied_speed_difference_range_m_s": tied_range,
            "log_sample_count": int(np.count_nonzero(mask)),
        }
    )
    metrics["accepted"] = bool(metrics["covered"] and metrics["positive"] and metrics["constant"] and metrics["tied"])
    return metrics


def _trial_force_reference(
    trial_path: Path,
    static_path: Path,
    subject_mass_kg: float,
    marker_time_s: np.ndarray,
    marker_positions: np.ndarray,
    marker_valid: np.ndarray,
    marker_index: dict[str, int],
) -> dict[str, Any]:
    trial = _extract_force_platforms(trial_path)
    static = _extract_force_platforms(static_path)
    static_total = static["force"].sum(axis=1)
    mean_weight = float(static_total[:, 2].mean())
    if mean_weight <= 0.0:
        raise ValueError("static force-platform sign check failed: mean extracted +Z load is not positive")
    subject_weight = subject_mass_kg * 9.81
    if abs(mean_weight - subject_weight) > max(0.25 * subject_weight, 150.0):
        raise ValueError("static force-platform sign check failed: extracted +Z load is not near subject weight")

    raw_total = trial["force"].sum(axis=1)
    tare_mask = np.abs(raw_total[:, 2]) <= TARE_TOTAL_UP_LIMIT_N
    if np.count_nonzero(tare_mask) < FORCE_FILTER_SAMPLES:
        raise ValueError("not enough low-total-load samples for force-platform tare estimation")
    tare_force = np.median(trial["force"][tare_mask], axis=0)
    tare_moment = np.median(trial["moment"][tare_mask], axis=0)
    corrected_force = trial["force"] - tare_force[None, :, :]
    corrected_moment = trial["moment"] - tare_moment[None, :, :]
    corrected_common_moment = _shift_moment_to_common_origin(
        corrected_moment,
        corrected_force,
        trial["mean_corners_newton_m"],
    )
    corrected_total_force = corrected_force.sum(axis=1)
    corrected_total_moment = corrected_common_moment.sum(axis=1)
    filtered_total = _hann_filter(corrected_total_force)
    filtered_total_moment = _hann_filter(corrected_total_moment)
    filtered_cop = _combine_cop(filtered_total, filtered_total_moment)

    bridge_samples = int(round(BRIDGE_S * trial["analog_rate_hz"]))
    active = _bridge_contact(filtered_total[:, 2] > CONTACT_GATE_N, bridge_samples)
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    support = _dynamic_support_metrics(
        marker_positions,
        marker_valid,
        marker_index,
        np.arange(len(filtered_total), dtype=np.float64) / trial["analog_rate_hz"],
        marker_time_s,
    )
    episodes = []
    rejected = []
    for start, end in zip(starts, ends, strict=True):
        duration_s = (end - start + 1) / trial["analog_rate_hz"]
        peak_n = float(filtered_total[start : end + 1, 2].max())
        base = {
            "duration_s": duration_s,
            "peak_total_up_n": peak_n,
            "sample_indices": [int(start), int(end)],
            "time_s": [float(start / trial["analog_rate_hz"]), float(end / trial["analog_rate_hz"])],
            "partial_source_window": False,
        }
        if duration_s < EPISODE_MIN_DURATION_S or peak_n < EPISODE_MIN_PEAK_N:
            rejected.append({**base, "reason": "duration_or_peak_gate"})
            continue
        episode_slice = slice(start, end + 1)
        support_complete = support["left_complete"][episode_slice] & support["right_complete"][episode_slice]
        finite_cop = np.all(np.isfinite(filtered_cop[episode_slice]), axis=1)
        if not np.any(support_complete & finite_cop):
            rejected.append(
                {
                    **base,
                    "reason": "nonfinite_cop_or_support_markers",
                    "finite_cop_fraction": float(np.mean(finite_cop)),
                    "support_complete_fraction": float(np.mean(support_complete)),
                }
            )
            continue
        if not np.all(support_complete & finite_cop):
            rejected.append(
                {
                    **base,
                    "reason": "incomplete_support_window",
                    "finite_cop_fraction": float(np.mean(finite_cop)),
                    "support_complete_fraction": float(np.mean(support_complete)),
                }
            )
            continue
        cop_xy = filtered_cop[episode_slice, :2]
        left_distance = np.min(np.linalg.norm(support["left_xy"][episode_slice] - cop_xy[:, None, :], axis=2), axis=1)
        right_distance = np.min(np.linalg.norm(support["right_xy"][episode_slice] - cop_xy[:, None, :], axis=2), axis=1)
        left_x_distance = _distance_to_interval(filtered_cop[episode_slice, 0], support["left_interval"][episode_slice])
        right_x_distance = _distance_to_interval(
            filtered_cop[episode_slice, 0], support["right_interval"][episode_slice]
        )
        median_left = float(np.median(left_distance))
        median_right = float(np.median(right_distance))
        if (
            not math.isfinite(median_left)
            or not math.isfinite(median_right)
            or math.isclose(median_left, median_right, abs_tol=1.0e-12)
        ):
            rejected.append(
                {
                    **base,
                    "reason": "ambiguous_side_distance",
                    "median_left_distance_m": median_left,
                    "median_right_distance_m": median_right,
                }
            )
            continue
        side = "left" if median_left < median_right else "right"
        other = "right" if side == "left" else "left"
        selected_distance = left_distance if side == "left" else right_distance
        other_distance = right_distance if side == "left" else left_distance
        selected_x_distance = left_x_distance if side == "left" else right_x_distance
        other_x_distance = right_x_distance if side == "left" else left_x_distance
        episodes.append(
            {
                **base,
                "assigned_side": side,
                "finite_cop_fraction": 1.0,
                "support_complete_fraction": 1.0,
                "median_left_distance_m": median_left,
                "median_right_distance_m": median_right,
                "median_distance_m": float(np.median(selected_distance)),
                "median_other_distance_margin_m": float(np.median(other_distance - selected_distance)),
                "median_x_distance_m": float(np.median(selected_x_distance)),
                "median_other_x_distance_margin_m": float(np.median(other_x_distance - selected_x_distance)),
                "nearest_fraction": float(np.mean(selected_distance < other_distance)),
                "x_nearest_fraction": float(np.mean(selected_x_distance <= other_x_distance)),
                "negative_normal_corrections": 0,
                "other_foot_bulk_heel_min_height_m": float(np.min(support[f"{other}_heel_min_height"][episode_slice])),
                "other_foot_bulk_min_height_m": float(np.min(support[f"{other}_bulk_min_height"][episode_slice])),
            }
        )
    static_sign_check = {
        "sign_check": "positive extracted laboratory +Z; no sign flip or hidden guessing",
        "mean_weight_n": mean_weight,
        "mass_equivalent_kg": mean_weight / 9.81,
        "normal_force_range_n": [float(np.min(static_total[:, 2])), float(np.max(static_total[:, 2]))],
    }
    if not episodes:
        raise ValueError("no retained contact episodes passed the force and geometry gates")
    return {
        "analog_rate_hz": trial["analog_rate_hz"],
        "point_rate_hz": trial["point_rate_hz"],
        "channel_indices": trial["channel_indices"],
        "channel_labels": trial["channel_labels"],
        "type": trial["type"],
        "corrected_force": corrected_force,
        "corrected_moment": corrected_moment,
        "corrected_total_force": corrected_total_force,
        "corrected_total_moment": corrected_total_moment,
        "filtered_total": filtered_total,
        "filtered_total_moment": filtered_total_moment,
        "filtered_cop": filtered_cop,
        "tare_force": tare_force,
        "tare_moment": tare_moment,
        "tare_mask": tare_mask,
        "mean_corners_newton_m": trial["mean_corners_newton_m"],
        "origin_newton_m": trial["origin_newton_m"],
        "episodes": episodes,
        "rejected_episodes": rejected,
        "diagnostics": {
            "contact_gate": {"threshold_n": CONTACT_GATE_N, "bridge_s": BRIDGE_S},
            "episode_minimums": {"duration_s": EPISODE_MIN_DURATION_S, "peak_n": EPISODE_MIN_PEAK_N},
            "episodes_retained": episodes,
            "episodes_rejected": rejected,
            "cop_frame": "raw Newton/lab frame; pooled from common-origin moments before any treadmill overground translation",
            "cop_origin": "common Newton/lab origin on the force-platform ground plane",
            "filter": {
                "method": "symmetric Hann",
                "samples": FORCE_FILTER_SAMPLES,
                "duration_s": (FORCE_FILTER_SAMPLES - 1) / trial["analog_rate_hz"],
            },
            "moment_common_origin": {
                "plate_surface_centers_newton_m": trial["mean_corners_newton_m"].tolist(),
                "policy": "shift each tare-corrected plate moment by plate_surface_center x force before pooling",
                "ezc3d_reference": "ForcePlatforms.cpp Type-2 rotates plate-frame moments, then adds meanCorners only to CoP; pooled COP therefore rebuilds from common-origin moments",
            },
            "moment_filtering": True,
            "negative_normal_corrections": 0,
            "negative_normal_policy": "sum signed plate forces first; zero negative TOTAL normal only in bridged contact holes; never rectify each unloaded plate",
            "tare_assumption": "Low-total-load samples are a force-based unloaded candidate; marker height is not used as a strict flight certificate.",
            "tare_bias_lab_n": tare_force.tolist(),
            "tare_moment_bias_lab_nm": tare_moment.tolist(),
            "tare_raw_total_up_abs_max_n": float(np.max(np.abs(raw_total[tare_mask, 2]))),
            "tare_sample_count": int(np.count_nonzero(tare_mask)),
            "filter_residual_rms_n": float(np.sqrt(np.mean(np.square(filtered_total - corrected_total_force)))),
            "filter_residual_max_n": float(np.max(np.abs(filtered_total - corrected_total_force))),
            "cop_positive_load_fraction": float(np.mean(filtered_total[:, 2] > 0.0)),
            "cop_finite_fraction_when_positive": float(
                np.mean(np.all(np.isfinite(filtered_cop[filtered_total[:, 2] > 0.0]), axis=1))
            ),
        },
        "static_sign_check": static_sign_check,
    }


def _subject_profile(
    subject_xml: Path,
    baseline_bundle: Path,
    lengths_m: np.ndarray,
    endpoint_local_m: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = ET.parse(subject_xml).getroot()

    def body(name: str) -> ET.Element:
        element = root.find(f".//body[@name='{name}']")
        if element is None:
            raise ValueError(f"missing body {name!r} in {subject_xml}")
        return element

    femur = body("femur_left")
    tibia = body("tibia_left")
    foot = body("foot_left")
    toes = body("toes_left")

    def vec(text: str) -> np.ndarray:
        return np.fromstring(text, sep=" ", dtype=np.float64)

    def inertial(element: ET.Element) -> tuple[float, np.ndarray, np.ndarray, dict[str, str]]:
        node = element.find("inertial")
        if node is None:
            raise ValueError(f"missing inertial data for {element.attrib.get('name')}")
        mass = float(node.attrib["mass"])
        com = vec(node.attrib["pos"])
        inertia = vec(node.attrib["fullinertia"])
        orientation_keys = ("quat", "axisangle", "euler", "xyaxes", "zaxis")
        orientation = {key: node.attrib[key] for key in orientation_keys if key in node.attrib}
        return mass, com, inertia, orientation

    def segment_local(
        element: ET.Element, proximal: np.ndarray, distal: np.ndarray, static_length_m: float
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        mass, com, inertia, inertial_orientation = inertial(element)
        axis = (distal - proximal)[[0, 2]]
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 0.0:
            raise ValueError(f"degenerate local segment axis for {element.attrib.get('name')}")
        x = axis / axis_length
        z = np.array([-x[1], x[0]], dtype=np.float64)
        rel = (com - proximal)[[0, 2]]
        local = np.array([float(np.dot(x, rel)), float(np.dot(z, rel))])
        scale = static_length_m / axis_length
        scaled_inertia = float(inertia[1] * scale**2)
        return (
            local * scale,
            scaled_inertia,
            {
                "mass_kg": mass,
                "model_axis_length_m": axis_length,
                "scale_to_static_length": scale,
                "proximal_local_xz": proximal[[0, 2]].tolist(),
                "distal_local_xz": distal[[0, 2]].tolist(),
                "scaled_com_local_m": (local * scale).tolist(),
                "source_fullinertia_i_yy_component_kg_m2": float(inertia[1]),
                "scaled_fullinertia_i_yy_component_kg_m2": scaled_inertia,
                "inertia_scale_factor": scale**2,
                "body_rotation_attributes": {
                    key: element.attrib[key]
                    for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
                    if key in element.attrib
                },
                "inertial_rotation_attributes": inertial_orientation,
            },
        )

    hip = vec(femur.find("joint").attrib["pos"])
    knee_mid = np.mean(
        [
            vec(site.attrib["pos"])
            for site in femur.findall("site")
            if site.attrib.get("name") in {"marker_L.Knee.Lat", "marker_L.Knee.Med"}
        ],
        axis=0,
    )
    knee = vec(tibia.find("joint").attrib["pos"])
    ankle_mid = np.mean(
        [
            vec(site.attrib["pos"])
            for site in tibia.findall("site")
            if site.attrib.get("name") in {"marker_L.Ankle.Lat", "marker_L.Ankle.Med"}
        ],
        axis=0,
    )
    ankle = vec(foot.find("joint").attrib["pos"])
    heel = next(vec(site.attrib["pos"]) for site in foot.findall("site") if site.attrib.get("name") == "marker_L.Heel")
    toe_lat = next(
        vec(site.attrib["pos"]) for site in foot.findall("site") if site.attrib.get("name") == "marker_L.Toe.Lat"
    )
    toe_med = next(
        vec(site.attrib["pos"]) for site in foot.findall("site") if site.attrib.get("name") == "marker_L.Toe.Med"
    )
    mth = 0.5 * (toe_lat + toe_med)

    thigh_com, thigh_inertia, thigh_meta = segment_local(femur, hip, knee_mid, float(lengths_m[0]))
    shank_com, shank_inertia, shank_meta = segment_local(tibia, knee, ankle_mid, float(lengths_m[1]))

    foot_mass, foot_com_raw, foot_inertia, foot_inertial_orientation = inertial(foot)
    toes_mass, toes_com_raw, toes_inertia, toes_inertial_orientation = inertial(toes)
    toes_com_in_foot = vec(toes.attrib["pos"]) + toes_com_raw
    combined_mass = foot_mass + toes_mass
    combined_com = (foot_mass * foot_com_raw + toes_mass * toes_com_in_foot) / combined_mass
    pitch = math.atan2((mth - heel)[2], (mth - heel)[0])
    rotation = np.array([[math.cos(pitch), -math.sin(pitch)], [math.sin(pitch), math.cos(pitch)]], dtype=np.float64)
    foot_local = rotation.T @ (combined_com[[0, 2]] - ankle[[0, 2]])
    model_endpoint_local = rotation.T @ (mth[[0, 2]] - ankle[[0, 2]])
    endpoint_scale = endpoint_local_m[0] / model_endpoint_local[0]
    foot_com = foot_local * endpoint_scale
    foot_offsets = [foot_com_raw[[0, 2]] - combined_com[[0, 2]], toes_com_in_foot[[0, 2]] - combined_com[[0, 2]]]
    foot_combined_inertia_model = float(
        foot_inertia[1]
        + foot_mass * float(np.dot(foot_offsets[0], foot_offsets[0]))
        + toes_inertia[1]
        + toes_mass * float(np.dot(foot_offsets[1], foot_offsets[1]))
    )
    foot_combined_inertia = foot_combined_inertia_model * endpoint_scale**2

    segment_masses = [float(inertial(femur)[0]), float(inertial(tibia)[0])]
    de_leva_i_yy_radius_fractions = {
        "female": (0.364, 0.263),
        "male": (0.329, 0.246),
    }
    de_leva_comparison = {
        sex: {
            name: {
                "population_i_yy_kg_m2": float(mass * (radius_fraction * length) ** 2),
                "scaled_subject_model_i_yy_component_kg_m2": float(inertia),
                "component_to_population_i_yy_ratio_assuming_aligned_frames": float(
                    inertia / (mass * (radius_fraction * length) ** 2)
                ),
            }
            for name, mass, length, inertia, radius_fraction in zip(
                ("thigh", "shank"),
                segment_masses,
                lengths_m,
                (thigh_inertia, shank_inertia),
                radius_fractions,
                strict=True,
            )
        }
        for sex, radius_fractions in de_leva_i_yy_radius_fractions.items()
    }

    baseline = json.loads((baseline_bundle / "profile.json").read_text())
    profile = {
        "schema": "cartesian_single_leg_1",
        "masses_kg": [*segment_masses, combined_mass],
        "com_local_m": [thigh_com.tolist(), shank_com.tolist(), foot_com.tolist()],
        "inertias_kg_m2": [thigh_inertia, shank_inertia, foot_combined_inertia],
        "hip_stiffness_n_m": baseline["hip_stiffness_n_m"],
        "hip_damping_ns_m": baseline["hip_damping_ns_m"],
        "joint_stiffness_nm_rad": baseline["joint_stiffness_nm_rad"],
        "joint_damping_nms_rad": baseline["joint_damping_nms_rad"],
        "joint_lower_rad": baseline["joint_lower_rad"],
        "joint_upper_rad": baseline["joint_upper_rad"],
        "equilibrium_lower": None,
        "equilibrium_upper": None,
        "equilibrium_rate_limit": baseline["equilibrium_rate_limit"],
        "equilibrium_acceleration_limit": baseline["equilibrium_acceleration_limit"],
        "provenance": {
            "inertial": "S014 left thigh/shank fullinertia[1] components from the subject model are scaled by the square of the planar length scale; foot and toes are combined about their shared COM, then scaled by the square of the endpoint scale. Segment masses stay subject-model values. de Leva (1996) adjusted Zatsiorsky-Seluyanov parameters are recorded as population comparison values, not substituted for this subject model.",
            "impedance": baseline["provenance"]["impedance"],
            "limits": "Equilibrium bounds are selected raw target min/max plus/minus [0.5 m, 0.5 m, 1.5 rad, 1.5 rad]. Joint ranges and equilibrium rate/acceleration limits remain the frozen baseline engineering limits; no acceptance or screen limit is relaxed.",
            "source_subject_xml": {"file": str(subject_xml.resolve()), "sha256": _sha256(subject_xml)},
            "baseline_profile": {
                "file": str((baseline_bundle / "profile.json").resolve()),
                "sha256": _sha256(baseline_bundle / "profile.json"),
            },
            "segment_scaling": {
                "thigh": thigh_meta,
                "shank": shank_meta,
                "foot": {
                    "combined_mass_kg": combined_mass,
                    "combined_com_local_model_xz_m": foot_local.tolist(),
                    "model_endpoint_local_m": model_endpoint_local.tolist(),
                    "endpoint_scale": endpoint_scale,
                    "scaled_com_local_m": foot_com.tolist(),
                    "combined_sagittal_inertia_kg_m2": foot_combined_inertia,
                    "combined_sagittal_inertia_before_endpoint_scaling_kg_m2": foot_combined_inertia_model,
                    "inertia_scale_factor": endpoint_scale**2,
                    "parallel_axis_offsets_xz_m": [offset.tolist() for offset in foot_offsets],
                    "body_rotation_attributes": {
                        key: foot.attrib[key]
                        for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
                        if key in foot.attrib
                    },
                    "body_rotation_attributes_toes": {
                        key: toes.attrib[key]
                        for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
                        if key in toes.attrib
                    },
                    "inertial_rotation_attributes": foot_inertial_orientation,
                    "inertial_rotation_attributes_toes": toes_inertial_orientation,
                },
            },
            "de_leva_1996_population_comparison": {
                "reference": "https://doi.org/10.1016/0021-9290(95)00178-6",
                "parameter_convention": "I = m * (k * segment_length)^2; planar sagittal motion uses out-of-plane IYY",
                "source_axis_assumption": "The profile currently selects fullinertia[1] as IYY without rotating the tensor into the planar frame. The recorded inertial-frame orientation must be verified against the source XML before interpreting these ratios.",
                "thigh_i_yy_radius_of_gyration_fraction": {"female": 0.364, "male": 0.329},
                "shank_i_yy_radius_of_gyration_fraction": {"female": 0.263, "male": 0.246},
                "foot_i_yy_radius_of_gyration_fraction": {"female": 0.279, "male": 0.245},
                "subject_model_thigh_shank_comparison": de_leva_comparison,
                "note": "Population comparison only. Do not compare foot values directly because the reduced model uses ankle-to-MTH endpoint geometry.",
            },
            "scope": "three actual leg masses; subject mass context only; same engineering gains and bounds as the fixed-gain baseline",
        },
    }
    return profile, {
        "static_model_pitch_rad": pitch,
        "model_endpoint_local_m": model_endpoint_local.tolist(),
        "body_rotation_check": {
            "femur_left": {
                key: femur.attrib[key]
                for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
                if key in femur.attrib
            },
            "tibia_left": {
                key: tibia.attrib[key]
                for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis")
                if key in tibia.attrib
            },
            "foot_left": {
                key: foot.attrib[key] for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis") if key in foot.attrib
            },
            "toes_left": {
                key: toes.attrib[key] for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis") if key in toes.attrib
            },
        },
    }


def _shoe_metadata(
    baseline_bundle: Path, static_pitch_rad: float, static_ankle_m: np.ndarray, static_heel_m: np.ndarray
) -> dict[str, Any]:
    """Register the unchanged shoe using measured static ankle/heel geometry."""
    baseline = json.loads((baseline_bundle / "summary.json").read_text())["shoe"]
    with np.load(baseline_bundle / "reference.npz", allow_pickle=False) as source:
        original_ankle = source["static_ankle_m"]
        original_heel = source["static_heel_m"]
    mount = np.asarray(baseline["mount_m"], dtype=np.float64).copy()
    heel_intrinsic_x = mount[0] + original_heel[0] - original_ankle[0]
    mount[0] = heel_intrinsic_x + static_ankle_m[0] - static_heel_m[0]
    mount[2] = static_ankle_m[2]
    data = copy.deepcopy(baseline)
    data.update(
        path=str((baseline_bundle / "digital_shoe.json").resolve()),
        sha256=_sha256(baseline_bundle / "digital_shoe.json"),
        mount_m=mount.tolist(),
        static_pitch_rad=static_pitch_rad,
        registration="static ground-plane height and transferred longitudinal heel-marker registration; no geometry scaling or material refit",
        registration_inputs={
            "anatomical_static_ankle_m": static_ankle_m.tolist(),
            "static_heel_marker_m": static_heel_m.tolist(),
            "baseline_heel_marker_intrinsic_x_m": float(heel_intrinsic_x),
            "height_policy": "intrinsic shoe level at static ground plane z=0; mount height equals measured static ankle height",
            "longitudinal_policy": "preserve baseline heel-marker x registration; use this subject's ankle-to-heel displacement",
            "lateral_policy": "retain intrinsic baseline lateral mount for the sagittal projection",
        },
        limitations="shared shoe and transferred heel-marker registration are engineering assumptions, not a subject-specific shoe/last identification",
    )
    return data


def _build_reference(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    subject_root = args.subject_root.resolve()
    baseline_bundle = args.baseline_bundle.resolve()
    baseline_summary = json.loads((baseline_bundle / "summary.json").read_text())
    static_markers = load_marker_artifact(subject_root / "markers")
    motion = load_native_motion_artifact(subject_root / "motions" / args.motion_name)
    trial_markers = read_c3d_markers(subject_root / args.trial_c3d)
    treadmill = load_treadmill_log(subject_root / args.treadmill_log)
    treadmill_motion = belt_motion(
        treadmill,
        trial_markers.times,
        side="left",
        offset=float(motion.treadmill.get("offset_s", 0.0) if isinstance(motion.treadmill, dict) else 0.0),
        up_axis="+Z",
        forward_axis="-Y",
    )
    trial_positions = trial_markers.positions.astype(np.float64)
    shifted_positions = trial_positions + treadmill_motion.distance[:, None, None] * treadmill_motion.axis
    trial_index = _marker_indices(trial_markers.marker_names, DYNAMIC_MARKER_NAMES)
    static_index = _marker_indices(
        static_markers.marker_names, (*STATIC_MARKER_NAMES, "LASI", "RASI", "LPSI", "RPSI", "LKNE", "LMKNE")
    )
    static_means, static_frames = _static_means(
        static_markers.positions.astype(np.float64),
        static_markers.valid,
        static_index,
        (*STATIC_MARKER_NAMES, "LASI", "RASI", "LPSI", "RPSI", "LKNE", "LMKNE"),
    )
    static_pitch_rad = float(
        math.atan2(
            (0.5 * (static_means["LMTH1"] + static_means["LMTH5"]) - static_means["LHEE"])[2],
            (0.5 * (static_means["LMTH1"] + static_means["LMTH5"]) - static_means["LHEE"])[0],
        )
    )
    foot_rotation = np.array(
        [
            [math.cos(static_pitch_rad), -math.sin(static_pitch_rad)],
            [math.sin(static_pitch_rad), math.cos(static_pitch_rad)],
        ],
        dtype=np.float64,
    )
    static_ankle_m = 0.5 * (static_means["LANK"] + static_means["LMANK"])
    static_heel_m = static_means["LHEE"]
    static_cluster = np.stack([static_means[name] for name in ("LHEE", "LHEE2", "LHEE3")])
    static_cluster_centroid = static_cluster.mean(axis=0)
    static_ground_heading = static_means["LTOE"] - static_cluster_centroid
    static_ground_heading[2] = 0.0
    static_ground_heading_norm = float(np.linalg.norm(static_ground_heading))
    if static_ground_heading_norm <= 0.0:
        raise ValueError("static heel-centroid-to-TOE ground heading is degenerate")
    static_ground_heading /= static_ground_heading_norm
    foot_marker_local_m = np.stack(
        [foot_rotation.T @ (static_means[name][[0, 2]] - static_ankle_m[[0, 2]]) for name in STATIC_MARKER_NAMES]
    )
    endpoint_local_m = foot_rotation.T @ (
        0.5 * (static_means["LMTH1"] + static_means["LMTH5"])[[0, 2]] - static_ankle_m[[0, 2]]
    )
    static_hip = _hip_center_from_pelvis(
        static_means, float(np.linalg.norm(static_means["LASI"] - static_means["RASI"]))
    )
    static_knee = 0.5 * (static_means["LKNE"] + static_means["LMKNE"])
    lengths_m = np.array(
        [
            float(np.linalg.norm((static_knee - static_hip)[[0, 2]])),
            float(np.linalg.norm((static_ankle_m - static_knee)[[0, 2]])),
        ],
        dtype=np.float64,
    )
    static_cluster_triangle_area = 0.5 * float(
        np.linalg.norm(np.cross(static_cluster[1] - static_cluster[0], static_cluster[2] - static_cluster[0]))
    )

    subject_json = json.loads((subject_root / "subject.json").read_text())
    force = _trial_force_reference(
        subject_root / args.trial_c3d,
        subject_root / args.static_c3d,
        float(subject_json["subject"]["mass_kg"]),
        trial_markers.times,
        trial_positions,
        trial_markers.valid,
        trial_index,
    )
    candidate_episodes = [episode for episode in force["episodes"] if episode["assigned_side"] == args.side]
    if not candidate_episodes:
        raise ValueError(f"no retained {args.side} contact episode was found")

    required_window_markers = [trial_index[name] for name in DYNAMIC_MARKER_NAMES]
    cluster_ids = [trial_index[name] for name in ("LHEE", "LHEE2", "LHEE3")]
    candidate_rejections = []
    accepted_candidates = []
    for episode in candidate_episodes:
        onset = (
            int(
                np.argmax(
                    force["filtered_total"][episode["sample_indices"][0] : episode["sample_indices"][1] + 1, 2]
                    > FORCE_THRESHOLD_N
                )
            )
            + episode["sample_indices"][0]
        )
        offset = int(
            episode["sample_indices"][1]
            - np.argmax(
                force["filtered_total"][episode["sample_indices"][0] : episode["sample_indices"][1] + 1, 2][::-1]
                > FORCE_THRESHOLD_N
            )
        )
        start = max(
            0,
            int(np.searchsorted(trial_markers.times, onset / force["analog_rate_hz"], side="left"))
            - WINDOW_PADDING_FRAMES,
        )
        end = min(
            len(trial_markers.times) - 1,
            int(np.searchsorted(trial_markers.times, offset / force["analog_rate_hz"], side="right"))
            - 1
            + WINDOW_PADDING_FRAMES,
        )
        rejection = []
        if not np.all(trial_markers.valid[start : end + 1, required_window_markers]):
            rejection.append("missing_required_markers")
        speed_metrics = _window_treadmill_metrics(
            treadmill, treadmill_motion, float(trial_markers.times[start]), float(trial_markers.times[end])
        )
        if not speed_metrics["accepted"]:
            rejection.append("treadmill_speed_guard")
        if episode["finite_cop_fraction"] < 1.0 or episode["support_complete_fraction"] < 1.0:
            rejection.append("cop_or_support_completeness")
        if episode["median_other_distance_margin_m"] <= SIDE_DISTANCE_MARGIN_M:
            rejection.append("side_distance_margin")
        if episode["nearest_fraction"] <= SIDE_NEAREST_FRACTION:
            rejection.append("side_nearest_fraction")
        if episode["other_foot_bulk_min_height_m"] <= OTHER_FOOT_BULK_MIN_HEIGHT_M:
            rejection.append("other_foot_bulk_height")
        if episode["other_foot_bulk_heel_min_height_m"] <= OTHER_FOOT_HEEL_MIN_HEIGHT_M:
            rejection.append("other_foot_heel_height")
        frame_slice = slice(start, end + 1)
        source_indices = np.arange(start, end + 1, dtype=np.int32)
        selected_treadmill_motion = treadmill_motion.select(source_indices)
        selected_positions = (
            trial_positions[frame_slice]
            + selected_treadmill_motion.distance[:, None, None] * selected_treadmill_motion.axis
        )
        _, _, frame_rms, point_max = _kabsch_rigid_transforms(static_cluster, selected_positions[:, cluster_ids])
        rigid_fit = {
            "frame_rms_max_m": float(np.max(frame_rms)),
            "frame_rms_mean_m": float(np.mean(frame_rms)),
            "point_max_max_m": float(np.max(point_max)),
            "point_max_mean_m": float(np.mean(point_max)),
            "limits_m": {"frame_rms": RIGID_FRAME_RMS_LIMIT_M, "point_max": RIGID_POINT_MAX_LIMIT_M},
            "evaluation": "raw selected-window heel-cluster rigid fit before reference filtering",
        }
        if (
            rigid_fit["frame_rms_max_m"] > RIGID_FRAME_RMS_LIMIT_M
            or rigid_fit["point_max_max_m"] > RIGID_POINT_MAX_LIMIT_M
        ):
            rejection.append("heel_cluster_rigid_fit")
        if rejection:
            candidate_rejections.append(
                {
                    "episode": copy.deepcopy(episode),
                    "touchdown_index": onset,
                    "toeoff_index": offset,
                    "source_frame_indices": [int(start), int(end)],
                    "reasons": rejection,
                    "treadmill_guard": speed_metrics,
                    "heel_cluster_rigid_fit": rigid_fit,
                }
            )
            continue
        accepted_candidates.append(
            {
                "episode": copy.deepcopy(episode),
                "touchdown_index": onset,
                "toeoff_index": offset,
                "start_frame": int(start),
                "end_frame": int(end),
                "treadmill_guard": speed_metrics,
                "heel_cluster_rigid_fit": rigid_fit,
            }
        )
    selection_diagnostics = {
        "schema": "cartesian_stance_selection_1",
        "subject_root": str(subject_root),
        "generator_sha256": _sha256(Path(__file__)),
        "heel_rigidity_evaluation": "raw markers before reference filtering",
        "accepted_candidates": accepted_candidates,
        "rejected_candidates": candidate_rejections,
    }
    (args.output / "selection_diagnostics.json").write_text(
        json.dumps(selection_diagnostics, indent=2, allow_nan=False) + "\n"
    )
    if not accepted_candidates:
        raise ValueError(
            f"no retained {args.side} contact episode passed side, treadmill, marker, and raw heel-cluster rigid-fit gates; "
            f"inspect {args.output / 'selection_diagnostics.json'}"
        )

    selected_candidate = min(
        accepted_candidates,
        key=lambda item: (
            item["heel_cluster_rigid_fit"]["frame_rms_max_m"],
            item["heel_cluster_rigid_fit"]["point_max_max_m"],
            -item["episode"]["nearest_fraction"],
            -item["episode"]["median_other_distance_margin_m"],
            item["episode"]["sample_indices"][0],
        ),
    )
    selected_episode = copy.deepcopy(selected_candidate["episode"])
    touchdown_index = int(selected_candidate["touchdown_index"])
    toeoff_index = int(selected_candidate["toeoff_index"])
    start_frame = int(selected_candidate["start_frame"])
    end_frame = int(selected_candidate["end_frame"])
    frame_slice = slice(start_frame, end_frame + 1)
    source_indices = np.arange(start_frame, end_frame + 1, dtype=np.int32)
    selected_treadmill_motion = treadmill_motion.select(source_indices)
    rebase_translation_m = treadmill_motion.distance[start_frame] * treadmill_motion.axis
    selected_positions = (
        trial_positions[frame_slice]
        + selected_treadmill_motion.distance[:, None, None] * selected_treadmill_motion.axis
    )
    time_s = trial_markers.times[frame_slice] - trial_markers.times[start_frame]

    grf_start = int(round(trial_markers.times[start_frame] * force["analog_rate_hz"]))
    grf_end = int(round(trial_markers.times[end_frame] * force["analog_rate_hz"]))
    grf_slice = slice(grf_start, grf_end + 1)
    grf_global_indices = np.arange(grf_start, grf_end + 1, dtype=np.int32)
    grf_time_s = grf_global_indices.astype(np.float64) / force["analog_rate_hz"] - trial_markers.times[start_frame]
    accepted_episode_mask = (grf_global_indices >= selected_episode["sample_indices"][0]) & (
        grf_global_indices <= selected_episode["sample_indices"][1]
    )
    unfiltered_grf = force["filtered_total"][grf_slice][:, (0, 2)].copy()
    unfiltered_grf[~accepted_episode_mask] = 0.0

    pelvis_markers = {name: selected_positions[:, trial_index[name]] for name in ("LASI", "RASI", "LPSI", "RPSI")}
    asis_width_m = float(np.linalg.norm(static_means["LASI"] - static_means["RASI"]))
    hip_target_full = np.stack(
        [
            _hip_center_from_pelvis({name: pelvis_markers[name][i] for name in pelvis_markers}, asis_width_m)[[0, 2]]
            for i in range(len(selected_positions))
        ]
    )
    knee_target_full = 0.5 * (
        selected_positions[:, trial_index["LKNE"], (0, 2)] + selected_positions[:, trial_index["LMKNE"], (0, 2)]
    )
    ankle_marker_full = 0.5 * (
        selected_positions[:, trial_index["LANK"], (0, 2)] + selected_positions[:, trial_index["LMANK"], (0, 2)]
    )
    foot_marker_target_full = np.stack(
        [selected_positions[:, trial_index[name], (0, 2)] for name in STATIC_MARKER_NAMES],
        axis=1,
    )

    cluster_dynamic = selected_positions[:, cluster_ids]
    cluster_rotations, cluster_translations, raw_cluster_frame_rms, raw_cluster_point_max = _kabsch_rigid_transforms(
        static_cluster, cluster_dynamic
    )
    transported_ankle = np.einsum("nij,j->ni", cluster_rotations, static_ankle_m) + cluster_translations
    cluster_heading = np.einsum("nij,j->ni", cluster_rotations, static_ground_heading)
    raw_foot_pitch = static_pitch_rad + np.unwrap(np.arctan2(cluster_heading[:, 2], cluster_heading[:, 0]))
    cluster_triangle_area = 0.5 * np.linalg.norm(
        np.cross(cluster_dynamic[:, 1] - cluster_dynamic[:, 0], cluster_dynamic[:, 2] - cluster_dynamic[:, 0]),
        axis=1,
    )
    filtered_cluster, _ = _butterworth_filter(cluster_dynamic, float(trial_markers.rate))
    _, _, filtered_cluster_frame_rms, filtered_cluster_point_max = _kabsch_rigid_transforms(
        static_cluster, filtered_cluster
    )

    unfiltered_joint_center_target = np.stack(
        (hip_target_full, knee_target_full, transported_ankle[:, (0, 2)]),
        axis=1,
    )
    unfiltered_ankle_marker_target = ankle_marker_full
    unfiltered_foot_marker_target = foot_marker_target_full
    unfiltered_foot_pitch = raw_foot_pitch
    filtered_joint_center_target, marker_filter = _butterworth_filter(
        unfiltered_joint_center_target,
        float(1.0 / np.median(np.diff(time_s))),
    )
    filtered_ankle_marker_target, _ = _butterworth_filter(
        unfiltered_ankle_marker_target, float(1.0 / np.median(np.diff(time_s)))
    )
    filtered_foot_marker_target, _ = _butterworth_filter(
        unfiltered_foot_marker_target, float(1.0 / np.median(np.diff(time_s)))
    )
    filtered_foot_pitch, _ = _butterworth_filter(
        unfiltered_foot_pitch[:, None], float(1.0 / np.median(np.diff(time_s))), unwrap=True
    )
    filtered_grf_signed, grf_filter = _butterworth_filter(unfiltered_grf, force["analog_rate_hz"])
    filtered_grf = filtered_grf_signed.copy()
    clamped = filtered_grf[:, 1] < 0.0
    filtered_grf[clamped, 1] = 0.0

    filtered_state, joint_center_reference, joint_center_residual, fit_iterations = _fit_state(
        filtered_joint_center_target[:, 0],
        filtered_joint_center_target[:, 1],
        filtered_joint_center_target[:, 2],
        filtered_foot_pitch[:, 0],
        lengths_m,
    )
    unfiltered_state, unfiltered_joint_center_reference, unfiltered_joint_center_residual, unfiltered_fit_iterations = (
        _fit_state(
            unfiltered_joint_center_target[:, 0],
            unfiltered_joint_center_target[:, 1],
            unfiltered_joint_center_target[:, 2],
            unfiltered_foot_pitch,
            lengths_m,
        )
    )
    hip_target = filtered_joint_center_target[:, 0]
    unfiltered_hip_target = unfiltered_joint_center_target[:, 0]
    shank_abs = np.arctan2(
        filtered_joint_center_target[:, 2, 1] - filtered_joint_center_target[:, 1, 1],
        filtered_joint_center_target[:, 2, 0] - filtered_joint_center_target[:, 1, 0],
    )
    thigh_abs = np.arctan2(
        filtered_joint_center_target[:, 1, 1] - filtered_joint_center_target[:, 0, 1],
        filtered_joint_center_target[:, 1, 0] - filtered_joint_center_target[:, 0, 0],
    )
    joint_target = np.column_stack((shank_abs - thigh_abs, filtered_foot_pitch[:, 0] - shank_abs - math.pi / 2.0))
    raw_shank_abs = np.arctan2(
        unfiltered_joint_center_target[:, 2, 1] - unfiltered_joint_center_target[:, 1, 1],
        unfiltered_joint_center_target[:, 2, 0] - unfiltered_joint_center_target[:, 1, 0],
    )
    raw_thigh_abs = np.arctan2(
        unfiltered_joint_center_target[:, 1, 1] - unfiltered_joint_center_target[:, 0, 1],
        unfiltered_joint_center_target[:, 1, 0] - unfiltered_joint_center_target[:, 0, 0],
    )
    unfiltered_joint_target = np.column_stack(
        (raw_shank_abs - raw_thigh_abs, unfiltered_foot_pitch - raw_shank_abs - math.pi / 2.0)
    )
    velocity = _finite_difference(filtered_state, time_s)
    unfiltered_velocity = _finite_difference(unfiltered_state, time_s)
    foot_marker_reference = (
        np.einsum(
            "nij,mj->nmi",
            _rotation(filtered_foot_pitch[:, 0]),
            foot_marker_local_m,
        )
        + filtered_joint_center_target[:, 2, None, :]
    )
    foot_marker_residual = filtered_foot_marker_target - foot_marker_reference
    unfiltered_foot_marker_reference = (
        np.einsum(
            "nij,mj->nmi",
            _rotation(unfiltered_foot_pitch),
            foot_marker_local_m,
        )
        + unfiltered_joint_center_target[:, 2, None, :]
    )
    unfiltered_foot_marker_residual = unfiltered_foot_marker_target - unfiltered_foot_marker_reference

    foot_marker_static = np.stack([static_means[name][[0, 2]] for name in STATIC_MARKER_NAMES])
    profile, model_meta = _subject_profile(
        subject_root / "model" / "subject.xml", baseline_bundle, lengths_m, endpoint_local_m
    )
    channels = filtered_state[:, (0, 1, 3, 4)]
    margin = np.array([0.5, 0.5, 1.5, 1.5], dtype=np.float64)
    profile["equilibrium_lower"] = (np.min(channels, axis=0) - margin).tolist()
    profile["equilibrium_upper"] = (np.max(channels, axis=0) + margin).tolist()

    reference = {
        "time_s": time_s.astype(np.float64),
        "state": filtered_state.astype(np.float64),
        "velocity": velocity.astype(np.float64),
        "hip_target_m": hip_target.astype(np.float64),
        "joint_target_rad": joint_target.astype(np.float64),
        "lengths_m": lengths_m.astype(np.float64),
        "endpoint_local_m": endpoint_local_m.astype(np.float64),
        "static_pitch_rad": np.asarray(static_pitch_rad, dtype=np.float64),
        "static_ankle_m": static_ankle_m[[0, 2]].astype(np.float64),
        "static_heel_m": static_heel_m[[0, 2]].astype(np.float64),
        "subject_mass_kg": np.asarray(float(subject_json["subject"]["mass_kg"]), dtype=np.float64),
        "grf_time_s": grf_time_s.astype(np.float64),
        "grf_target_n": filtered_grf.astype(np.float64),
        "joint_center_target_m": filtered_joint_center_target.astype(np.float64),
        "joint_center_reference_m": joint_center_reference.astype(np.float64),
        "joint_center_residual_m": joint_center_residual.astype(np.float64),
        "single_leg_fit_iterations": fit_iterations.astype(np.int32),
        "foot_pitch_target_rad": filtered_foot_pitch[:, 0].astype(np.float64),
        "foot_marker_target_m": filtered_foot_marker_target.astype(np.float64),
        "foot_marker_local_m": foot_marker_local_m.astype(np.float64),
        "foot_marker_reference_m": foot_marker_reference.astype(np.float64),
        "foot_marker_residual_m": foot_marker_residual.astype(np.float64),
        "foot_marker_static_m": foot_marker_static.astype(np.float64),
        "ankle_marker_target_m": filtered_ankle_marker_target.astype(np.float64),
        "selected_source_indices": source_indices.astype(np.int32),
        "unfiltered_foot_pitch_target_rad": unfiltered_foot_pitch.astype(np.float64),
        "unfiltered_joint_center_target_m": unfiltered_joint_center_target.astype(np.float64),
        "unfiltered_foot_marker_target_m": unfiltered_foot_marker_target.astype(np.float64),
        "unfiltered_ankle_marker_target_m": unfiltered_ankle_marker_target.astype(np.float64),
        "unfiltered_hip_target_m": unfiltered_hip_target.astype(np.float64),
        "unfiltered_joint_target_rad": unfiltered_joint_target.astype(np.float64),
        "unfiltered_state": unfiltered_state.astype(np.float64),
        "unfiltered_single_leg_fit_iterations": unfiltered_fit_iterations.astype(np.int32),
        "grf_butterworth_n": filtered_grf_signed.astype(np.float64),
        "unfiltered_grf_target_n": unfiltered_grf.astype(np.float64),
        "unfiltered_velocity": unfiltered_velocity.astype(np.float64),
        "unfiltered_joint_center_reference_m": unfiltered_joint_center_reference.astype(np.float64),
        "unfiltered_joint_center_residual_m": unfiltered_joint_center_residual.astype(np.float64),
        "unfiltered_foot_marker_reference_m": unfiltered_foot_marker_reference.astype(np.float64),
        "unfiltered_foot_marker_residual_m": unfiltered_foot_marker_residual.astype(np.float64),
    }

    motion_check = {
        "common_marker_rms_m": {},
        "common_marker_max_m": {},
        "treadmill_manifest": motion.treadmill,
    }
    motion_lookup = {name: i for i, name in enumerate(motion.marker_names)}
    for raw_name, motion_name in MOTION_MARKER_MAP.items():
        if motion_name not in motion_lookup or raw_name not in trial_index:
            continue
        diff = shifted_positions[:, trial_index[raw_name]] - motion.targets[:, motion_lookup[motion_name]]
        motion_check["common_marker_rms_m"][raw_name] = float(np.sqrt(np.mean(np.square(diff))))
        motion_check["common_marker_max_m"][raw_name] = float(np.max(np.abs(diff)))

    selected_speed = selected_candidate["treadmill_guard"]
    selected_log_mask = (treadmill.t >= selected_speed["log_interval_s"][0]) & (
        treadmill.t <= selected_speed["log_interval_s"][1]
    )
    selected_tied_residual_m = 0.0
    if np.any(selected_log_mask):
        left_distance = treadmill.left_distance[selected_log_mask] - treadmill.left_distance[selected_log_mask][0]
        right_distance = treadmill.right_distance[selected_log_mask] - treadmill.right_distance[selected_log_mask][0]
        selected_tied_residual_m = float(np.max(np.abs(left_distance - right_distance)))

    selected_episode.update(
        {
            "touchdown_index": touchdown_index,
            "toeoff_index": toeoff_index,
            "source_frame_indices": [int(start_frame), int(end_frame)],
            "source_window_s": [float(trial_markers.times[start_frame]), float(trial_markers.times[end_frame])],
            "partial_source_window": bool(start_frame == 0 or end_frame == len(trial_markers.times) - 1),
            "treadmill_guard": selected_speed,
            "heel_cluster_rigid_fit": selected_candidate["heel_cluster_rigid_fit"],
        }
    )

    metadata = {
        "schema": "cartesian_single_leg_1",
        "side": args.side,
        "plane": "[forward, up], ground-relative overground; code axes x/z",
        "state_order": ["hip_x", "hip_z", "thigh_absolute_angle", "knee_relative_angle", "ankle_relative_angle"],
        "state_units": ["m", "m", "rad", "rad", "rad"],
        "equilibrium_order": ["hip_x", "hip_z", "knee", "ankle"],
        "angle_convention": {
            "thigh": "absolute proximal-to-distal angle; unactuated rotation",
            "shank": "thigh + knee; knee flexion negative",
            "foot": "thigh + knee + pi/2 + ankle; ankle dorsiflexion positive",
        },
        "qualification": "single-leg marker-compatible reduction with explicit engineering actuation; not validated physiology",
        "excluded_mechanics": "no trunk, pelvis body, opposite leg, hip torque, added upper-body mass, or imposed hip trajectory",
        "subject_mass_use": "context only; never runtime inertia, added weight, or native GRF scaling",
        "velocity_use": "second-order finite differences of filtered state; initial velocity only, never a fit target",
        "hip_target": "20 Hz low-pass selected-side hip position from CODA/Bell pelvis regression with frozen static ASIS width",
        "joint_target": "knee/ankle geometry rebuilt from low-pass centers and unwrapped heel-cluster pitch",
        "foot_marker_names": list(STATIC_MARKER_NAMES),
        "foot_markers": {
            "frame": {
                "ankle_transport": "R @ static anatomical ANK/MANK midpoint + T; no engineering mount substitution",
                "cluster_marker_names": ["LHEE", "LHEE2", "LHEE3"],
                "diagnostic_marker_names": list(STATIC_MARKER_NAMES),
                "heading_marker": "LTOE",
                "heading_origin": "heel cluster centroid",
                "heading_scope": "static ground-projected heading_origin-to-TOE direction only; dynamic TOE/HLX do not set pitch",
                "pitch": "static_pitch_rad + unwrap(atan2((R @ static_ground_heading)_z, (R @ static_ground_heading)_x)); toe-up positive",
                "quality_limits_m": {"frame_rms": RIGID_FRAME_RMS_LIMIT_M, "point_max": RIGID_POINT_MAX_LIMIT_M},
                "quality_selected_window_m": {
                    "filtered_cluster_frame_rms_max": float(np.max(filtered_cluster_frame_rms)),
                    "filtered_cluster_frame_rms_mean": float(np.mean(filtered_cluster_frame_rms)),
                    "filtered_cluster_point_max": float(np.max(filtered_cluster_point_max)),
                    "raw_cluster_frame_rms_max": float(np.max(raw_cluster_frame_rms)),
                    "raw_cluster_point_max": float(np.max(raw_cluster_point_max)),
                },
                "static_frame": "ANK/MANK midpoint origin; unchanged profile static_pitch_rad defines the sagittal body basis",
                "triangle_area_m2": static_cluster_triangle_area,
                "minimum_dynamic_triangle_area_m2": float(np.min(cluster_triangle_area)),
                "limitations": "proper rigid marker fit, not an independently measured sole or last frame; roll and yaw are diagnostic",
            },
            "use": "report only; local points relative to anatomical ankle, not a sole or mount identification",
            "static_geometry": "original selected endpoint/static profile arrays retained unchanged",
        },
        "stance": {
            "index": 0,
            "side": args.side,
            "threshold_n": FORCE_THRESHOLD_N,
            "event_resolution_s": 1.0 / force["analog_rate_hz"],
            "touchdown_s": float(touchdown_index / force["analog_rate_hz"] - trial_markers.times[start_frame]),
            "toeoff_s": float(toeoff_index / force["analog_rate_hz"] - trial_markers.times[start_frame]),
            "padding_frames": WINDOW_PADDING_FRAMES,
            "minimum_duration_s": EPISODE_MIN_DURATION_S,
            "source_frame_indices": [int(start_frame), int(end_frame)],
            "source_window_s": [float(trial_markers.times[start_frame]), float(trial_markers.times[end_frame])],
            "scope": "one isolated single-foot contact and adjacent flight frames; not a periodic full-stride fit",
            "initial_state": "filtered selected hip and fixed-length state; finite-difference tangent",
        },
        "single_leg_fit": {
            "method": "per-frame fixed-length leg fit to filtered centers; filtered hip held exact",
            "center_order": ["hip", "knee", "ankle"],
            "residual_rms_m": np.sqrt(np.mean(np.square(joint_center_residual), axis=0)).tolist(),
            "residual_sign": "filtered target minus reconstructed prediction",
            "joint_target_minus_state_rms_rad": np.sqrt(
                np.mean(np.square(joint_target - filtered_state[:, 3:]), axis=0)
            ).tolist(),
        },
        "reference_filter": {
            **marker_filter,
            "schema": "cartesian_reference_filter_1",
            "filtered_arrays": [
                "joint_center_target_m",
                "ankle_marker_target_m",
                "foot_marker_target_m",
                "foot_pitch_target_rad",
                "grf_target_n",
            ],
            "rebuilt_arrays": [
                "hip_target_m",
                "joint_target_rad",
                "state",
                "single_leg_fit_iterations",
                "velocity",
                "joint_center_reference_m",
                "joint_center_residual_m",
                "foot_marker_reference_m",
                "foot_marker_residual_m",
            ],
            "raw_arrays": "unfiltered_ prefix preserves every pre-20 Hz array",
            "window": "selected stance and padded flight frames only; no full-trial context; endpoint transients can remain",
            "force": {
                **grf_filter,
                "time_array": "grf_time_s",
                "sample_count": int(len(grf_time_s)),
                "minimum_before_clamp_n": float(np.min(filtered_grf_signed[:, 1])),
                "clamped_sample_count": int(np.count_nonzero(clamped)),
                "postprocessing": "clamp only upward GRF below zero; retain signed output in grf_butterworth_n",
                "resampled": False,
                "pre_lowpass_reference_array": "unfiltered_grf_target_n",
                "pre_lowpass_reference_note": "historical unfiltered_grf_target_n name is retained, but the array is already tare-corrected, pooled, Hann-filtered, episode-masked, and reduced to [forward, up]",
            },
            "preserved": "native clocks, raw stance events, lengths, static geometry, and heel-cluster rigid-fit diagnostics",
        },
        "native_grf": {
            "operation": "fourth-order low-pass forward/backward at native rate, then clamp upward GRF >= 0",
            "original": {
                "operation": "tare-correct each native plate, shift moments to a common Newton/lab origin, pool signed forces, apply a 21-sample symmetric Hann filter at native 2000 Hz, keep only the accepted single-foot episode inside the padded native window, and preserve the native clock; no marker-rate resampling",
                "source_array": "unfiltered_grf_target_n",
                "provenance": {
                    "file": str(subject_root / args.trial_c3d),
                    "sha256": _sha256(subject_root / args.trial_c3d),
                    "metadata": {
                        "schema": "impedance_joint_grf_1",
                        "time_origin": "reference_window_start",
                        "units": "N",
                        "frame": "selected local overground world for positions; forces are invariant under the constant horizontal rebase",
                        "force_on": "body",
                        "components": ["forward", "up"],
                        "ezc3d_version": "1.7.2",
                        "provenance": "ezc3d extract_forceplat_data=True; ForcePlatforms.cpp Type-2 corner-defined frame rotation to lab/Newton force and moment, then common-origin moment pooling from mean plate corners before pooled COP; pooled low-total-load per-plate median tare; symmetric 21-sample Hann filter; explicit single-foot contact and foot-geometry gates. Type-2 engineering calibration is trusted as a source export, not independently calibrated in the lab.",
                        "reference_window_start_s": float(trial_markers.times[start_frame]),
                        "reference_window_duration_s": float(grf_time_s[-1] - grf_time_s[0]),
                        "sides": [args.side],
                        "source_sha256": {
                            "static": _sha256(subject_root / args.static_c3d),
                            "trial": _sha256(subject_root / args.trial_c3d),
                        },
                        "static_sign_check": force["static_sign_check"],
                        "calibration_limitation": "Type-2 engineering calibration trusted export; not an independent laboratory calibration certificate.",
                        "diagnostics": force["diagnostics"],
                        "episode_mask": {
                            "accepted_episode_source_indices": selected_episode["sample_indices"],
                            "selected_window_source_indices": [int(grf_start), int(grf_end)],
                            "masked_outside_episode_sample_count": int(np.count_nonzero(~accepted_episode_mask)),
                        },
                    },
                    "use": "fitting target only; native sampling retained; never applied to forward dynamics",
                },
            },
            "raw_array": "unfiltered_grf_target_n",
            "signed_filtered_array": "grf_butterworth_n",
        },
        "source_preparation": {
            "schema": "impedance_paper_data_1",
            "side": args.side,
            "requested_window_s": [float(trial_markers.times[start_frame]), float(trial_markers.times[end_frame])],
            "actual_window_s": [float(trial_markers.times[start_frame]), float(trial_markers.times[end_frame])],
            "frame": "selected local overground world; x forward, z up; one constant horizontal treadmill translation rebases the selected start frame to zero",
            "forward_axis": "-Y",
            "up_axis": "+Z",
            "time_origin": "first selected point frame; C3D header frame number is provenance only",
            "height_origin": "force-platform plane; the selected-window rebase is horizontal only",
            "geometry": "constant lengths of projected static landmarks; foot orientation comes from 3D heel-cluster transport plus the static ground-projected heel-centroid-to-TOE heading",
            "marker_gaps": "reject dynamic nonfinite coordinates or incomplete required markers in the selected padded window; static averages use complete frames only; no repair",
            "mass": "explicit subject.json value; not estimated from force plates",
            "reference": "q_rad uses the transported anatomical ankle and heel-cluster pitch; endpoint_local_m remains the static MTH midpoint relative to the ankle",
            "hip": {
                "method": "approximate CODA/Bell landmark regression; not measured joint centers",
                "basis": "ASIS right, orthogonalized PSIS-midpoint to ASIS-midpoint anterior, cross-product up",
                "offset_fractions_right_anterior_up": CODA_LEFT_HIP_OFFSET.tolist(),
                "static_asis_width_m": asis_width_m,
                "dynamic": "frozen static width transported by each measured pelvis frame",
                "marker_radius_correction_m": 0.0,
                "source": "https://wiki.has-motion.com/doku.php?id=visual3d:documentation:modeling:segments:coda_pelvis",
            },
            "foot_orientation": {
                "marker_names": ["LHEE", "LHEE2", "LHEE3"],
                "heading_marker": "LTOE",
                "heading_scope": "static ground-projected heel-centroid-to-TOE only; dynamic TOE/HLX never set pitch",
                "method": "proper 3D Kabsch rigid fit on the heel cluster; transport the static anatomical ankle midpoint with R,T",
                "smoothing": "none before the frozen 20 Hz reference filter",
                "static_frame": "static ankle origin and static LHEE-to-MTH midpoint pitch",
                "quality_selected_window_m": {
                    "filtered_cluster_frame_rms_max": float(np.max(filtered_cluster_frame_rms)),
                    "filtered_cluster_point_max": float(np.max(filtered_cluster_point_max)),
                    "raw_cluster_frame_rms_max": float(np.max(raw_cluster_frame_rms)),
                    "raw_cluster_point_max": float(np.max(raw_cluster_point_max)),
                },
                "pitch_joint_distinction": "foot_pitch_target_rad is absolute foot pitch; state[:,4] is foot pitch relative to the absolute shank angle",
            },
            "trial": {
                "file": str(subject_root / args.trial_c3d),
                "sha256": _sha256(subject_root / args.trial_c3d),
                "point_rate_hz": float(trial_markers.rate),
                "point_units_raw": "mm",
                "selected_frame_indices": [int(start_frame), int(end_frame)],
                "selected_source_frame_numbers": [
                    int(trial_markers.first_frame + start_frame),
                    int(trial_markers.first_frame + end_frame),
                ],
                "first_frame_header": int(trial_markers.first_frame),
                "forces": {
                    "analog_rate_hz": force["analog_rate_hz"],
                    "components": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                    "corners": "ezc3d Type-2 corner-defined plate frame extraction; mean plate corners are used as the common-origin shift for pooled moments",
                    "origin": "ezc3d FORCE_PLATFORM origin stays plate-local; pooled COP is rebuilt from common-origin moments instead of averaging per-plate COP",
                    "raw": "ezc3d-decoded engineering channels including C3D analog scaling, not ADC counts",
                    "runtime_applied": False,
                    "raw_units": ["N", "N", "N", "Nmm or Nm", "Nmm or Nm", "Nmm or Nm"],
                    "si_units": ["N", "N", "N", "N m", "N m", "N m"],
                    "type": force["type"],
                    "side_assignment": "pooled total retained only when isolated single-foot support is established by COP and opposite-foot-airborne gates; no per-plate side label is invented",
                    "assignment_frame": "raw Newton/lab marker frame before treadmill overground translation",
                    "sign": "positive +Z verified against the static standing weight check; no manual sign flip applied",
                    "channel_indices": force["channel_indices"].T.tolist(),
                    "channel_labels": [list(labels) for labels in force["channel_labels"]],
                },
            },
            "static": {
                "file": str(subject_root / args.static_c3d),
                "sha256": _sha256(subject_root / args.static_c3d),
                "point_rate_hz": float(static_markers.rate),
                "point_units_raw": "mm",
                "selected_frame_indices": [int(static_frames[0]), int(static_frames[-1])],
                "selected_source_frame_numbers": [
                    int(static_markers.first_frame + static_frames[0]),
                    int(static_markers.first_frame + static_frames[-1]),
                ],
                "first_frame_header": int(static_markers.first_frame),
                "complete_frame_indices": static_frames.tolist(),
                "discarded_incomplete_frames": int(len(static_markers.times) - len(static_frames)),
            },
            "treadmill": {
                "file": str(subject_root / args.treadmill_log),
                "source": {"file": args.treadmill_log, "sha256": _sha256(subject_root / args.treadmill_log)},
                "offset_s": float(treadmill_motion.offset),
                "side": treadmill_motion.side,
                "axis": treadmill_motion.axis.tolist(),
                "distance_m": float(selected_treadmill_motion.travel),
                "speed_max_m_s": float(np.max(np.abs(selected_treadmill_motion.speed))),
                "speed_range_m_s": [
                    float(np.min(selected_treadmill_motion.speed)),
                    float(np.max(selected_treadmill_motion.speed)),
                ],
                "tied_belt_residual_m": float(treadmill.tied_belt_residual),
                "selected_window_tied_residual_m": selected_tied_residual_m,
                "global_tied_residual_m": float(treadmill.tied_belt_residual),
                "covered_frames": int(np.count_nonzero(selected_treadmill_motion.covered)),
                "raw_speed_columns": ["left", "right"],
                "raw_speed_units": "m/s",
                "raw_time": "controller clock [s]; C3D time = log-relative time - offset_s",
                "source_rate_hz": float(treadmill.rate),
                "constant_speed_check_margin_s": CONSTANT_SPEED_MARGIN_S,
                "constant_speed_tolerance_m_s": CONSTANT_SPEED_TOLERANCE_M_S,
                "sync_guard_s": SYNC_GUARD_S,
                "sync_guard_limitation": "assumed uncertainty guard for the treadmill log offset; force and point data still share the same C3D clock",
                "positive_constant_guard_log_interval_s": selected_speed["log_interval_s"],
                "window_guard": selected_speed,
                "synchronization": "marker and force channels share the C3D clock; treadmill-to-overground shift reuses the saved motion-artifact offset and remains unverified beyond the guard note",
                "axis_assumption": "belt surface travels opposite declared laboratory forward axis",
                "applied_stage": "selected-window local overground rebase after native overground translation",
                "local_rebase_translation_m": rebase_translation_m.tolist(),
                "log_parser": "existing loader drops nonfinite/non-increasing rows; no automatic synchronization",
            },
            "motion_artifact_cross_check": motion_check,
            "model_inertial_projection": model_meta,
        },
        "sources": {
            "subject_root": str(subject_root),
            "subject_json": {
                "file": str(subject_root / "subject.json"),
                "sha256": _sha256(subject_root / "subject.json"),
            },
            "marker_artifact": {
                "file": str(subject_root / "markers"),
                "sha256": _sha256(subject_root / "markers" / "markers.npz"),
            },
            "motion_artifact": {
                "file": str(subject_root / "motions" / args.motion_name),
                "sha256": _sha256(subject_root / "motions" / args.motion_name / "motion.npz"),
            },
            "baseline_bundle": {"file": str(baseline_bundle), "sha256": _sha256(baseline_bundle / "summary.json")},
        },
    }
    reference["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=np.str_)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "complete": True,
        "accepted": False,
        "input_quality": {
            "schema": "cartesian_subject_input_quality_1",
            "passed": True,
            "combined_selection_passed": True,
            "rigidity_evaluation": "raw heel cluster before reference filtering",
            "raw_frame_rms_max_m": float(np.max(raw_cluster_frame_rms)),
            "raw_point_max_m": float(np.max(raw_cluster_point_max)),
            "rigidity_limits_m": {"frame_rms": RIGID_FRAME_RMS_LIMIT_M, "point_max": RIGID_POINT_MAX_LIMIT_M},
            "scope": "Input screens only; not controller acceptance, independent synchronization, or shoe validation.",
        },
        "subject": subject_json["subject"]["name"],
        "side": args.side,
        "reference_schema": metadata["schema"],
        "fit_config": baseline_summary["fit_config"],
        "simulation_config": baseline_summary["simulation_config"],
        "shoe": _shoe_metadata(baseline_bundle, static_pitch_rad, static_ankle_m, static_heel_m),
        "qualification": "Prepared measured inputs only. Screen and acceptance limits remain frozen. The shared shoe artifact is still an engineering assumption.",
        "selection": {
            "candidate_episode_count": len(candidate_episodes),
            "accepted_candidate_count": len(accepted_candidates),
            "rejected_candidates": candidate_rejections,
            "contact_episode": selected_episode,
            "touchdown_index": touchdown_index,
            "toeoff_index": toeoff_index,
            "source_frame_indices": [int(start_frame), int(end_frame)],
            "source_window_s": [float(trial_markers.times[start_frame]), float(trial_markers.times[end_frame])],
            "rebase_translation_m": rebase_translation_m.tolist(),
        },
        "provenance": {
            "generator": f"{Path(__file__).relative_to(ROOT)}",
            "generator_sha256": _sha256(Path(__file__)),
            "baseline_summary": {
                "file": str((baseline_bundle / "summary.json").resolve()),
                "sha256": _sha256(baseline_bundle / "summary.json"),
            },
            "baseline_profile": {
                "file": str((baseline_bundle / "profile.json").resolve()),
                "sha256": _sha256(baseline_bundle / "profile.json"),
            },
            "digital_shoe_source": {
                "file": str((baseline_bundle / "digital_shoe.json").resolve()),
                "sha256": _sha256(baseline_bundle / "digital_shoe.json"),
            },
            "command": [sys.executable, *sys.argv],
        },
    }
    return reference, profile, summary, metadata, {"source_indices": source_indices.tolist()}


def create_parser() -> argparse.ArgumentParser:
    """Build the fail-closed subject preparation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    parser.add_argument("--side", choices=("left",), default="left")
    parser.add_argument("--motion-name", default="trial_101_native_motion")
    parser.add_argument("--trial-c3d", default="Trial 101.v3d.c3d")
    parser.add_argument("--static-c3d", default="Cal 101.v3d.c3d")
    parser.add_argument("--treadmill-log", default="tm0001.txt")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Write screened measured inputs or retain rejected-window diagnostics."""
    args = create_parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose a new or empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference, profile, summary, metadata, extra = _build_reference(args)
    np.savez_compressed(output / "reference.npz", **reference)
    (output / "profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n")
    (output / "digital_shoe.json").write_bytes((args.baseline_bundle / "digital_shoe.json").read_bytes())
    summary["shoe"]["path"] = str(output / "digital_shoe.json")
    summary["reference"] = {"file": str(output / "reference.npz"), "sha256": _sha256(output / "reference.npz")}
    summary["profile"] = {"file": str(output / "profile.json"), "sha256": _sha256(output / "profile.json")}
    summary["digital_shoe"] = {
        "file": str(output / "digital_shoe.json"),
        "sha256": _sha256(output / "digital_shoe.json"),
    }
    summary["input_quality"]["files_sha256"] = {
        name: _sha256(output / name) for name in ("reference.npz", "profile.json", "digital_shoe.json")
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    audit = {
        "metadata_schema": metadata["schema"],
        "selected_source_indices": extra["source_indices"],
        "reference_sha256": summary["reference"]["sha256"],
        "profile_sha256": summary["profile"]["sha256"],
        "digital_shoe_sha256": summary["digital_shoe"]["sha256"],
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
