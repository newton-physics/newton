"""Diagnostics for Digital Instron trial responses."""

from __future__ import annotations

import numpy as np


def force_jump_metrics(force_n: np.ndarray, limit_n: float) -> dict[str, float | int]:
    force = np.asarray(force_n, dtype=np.float64)
    if force.size < 2:
        return {
            "force_jump_limit_n": float(limit_n),
            "max_adjacent_force_jump_n": 0.0,
            "max_adjacent_force_jump_index": 0,
            "max_adjacent_force_jump_from_n": float(force[0]) if force.size else 0.0,
            "max_adjacent_force_jump_to_n": float(force[0]) if force.size else 0.0,
            "force_jump_violation_count": 0,
            "force_jump_violation": 0,
        }

    deltas = np.diff(force)
    abs_deltas = np.abs(deltas)
    max_delta_idx = int(np.argmax(abs_deltas))
    max_delta = float(abs_deltas[max_delta_idx])
    limit = max(float(limit_n), 0.0)
    violations = int(np.count_nonzero(abs_deltas > limit)) if limit > 0.0 else 0
    return {
        "force_jump_limit_n": limit,
        "max_adjacent_force_jump_n": max_delta,
        "max_adjacent_force_jump_index": max_delta_idx + 1,
        "max_adjacent_force_jump_from_n": float(force[max_delta_idx]),
        "max_adjacent_force_jump_to_n": float(force[max_delta_idx + 1]),
        "force_jump_violation_count": violations,
        "force_jump_violation": int(violations > 0),
    }


def residual_motion_metrics(contact_stats: list[dict[str, float | int]]) -> dict[str, float]:
    vertical_velocity = np.asarray(
        [float(stats.get("shoe_vertical_velocity_m_s", 0.0)) for stats in contact_stats],
        dtype=np.float64,
    )
    free_axis_angular_velocity = np.asarray(
        [float(stats.get("shoe_free_axis_angular_velocity_rad_s", 0.0)) for stats in contact_stats],
        dtype=np.float64,
    )
    angular_velocity_norm = np.asarray(
        [float(stats.get("shoe_angular_velocity_norm_rad_s", 0.0)) for stats in contact_stats],
        dtype=np.float64,
    )
    if vertical_velocity.size == 0:
        return {
            "max_abs_shoe_vertical_velocity_m_s": 0.0,
            "rms_shoe_vertical_velocity_m_s": 0.0,
            "max_abs_shoe_free_axis_angular_velocity_rad_s": 0.0,
            "rms_shoe_free_axis_angular_velocity_rad_s": 0.0,
            "max_shoe_angular_velocity_norm_rad_s": 0.0,
            "rms_shoe_angular_velocity_norm_rad_s": 0.0,
        }
    return {
        "max_abs_shoe_vertical_velocity_m_s": float(np.max(np.abs(vertical_velocity))),
        "rms_shoe_vertical_velocity_m_s": float(np.sqrt(np.mean(vertical_velocity * vertical_velocity))),
        "max_abs_shoe_free_axis_angular_velocity_rad_s": float(np.max(np.abs(free_axis_angular_velocity))),
        "rms_shoe_free_axis_angular_velocity_rad_s": float(
            np.sqrt(np.mean(free_axis_angular_velocity * free_axis_angular_velocity))
        ),
        "max_shoe_angular_velocity_norm_rad_s": float(np.max(angular_velocity_norm)),
        "rms_shoe_angular_velocity_norm_rad_s": float(np.sqrt(np.mean(angular_velocity_norm * angular_velocity_norm))),
    }

