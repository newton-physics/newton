# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fit four equilibrium channels to recorded motion and native single-foot GRF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitConfig:
    """Declare measured tolerances and a bounded mixed-unit search budget.

    ``parameter_scale`` gives coordinate scales [m, m, rad, rad], not search
    bounds. Each step is the current dimensionless fraction times this scale.
    Profile bounds, masses, gains, initial state, and shoe stay fixed. Force
    tolerances are in newtons, with no subject-bodyweight normalization.
    Refinement tolerances bound maximum differences, not RMS differences.
    These are numerical acceptance criteria, not physiological validation.
    """

    control_count: int = 12
    max_evaluations: int = 60
    initial_step_fraction: float = 0.08
    minimum_step_fraction: float = 0.01
    parameter_scale: tuple[float, float, float, float] = (0.2, 0.2, 0.5, 0.5)
    hip_tolerance_m: float = 0.02
    joint_tolerance_rad: float = 0.05
    force_tolerance_n: float = 100.0
    refinement_position_m: float = 0.002
    refinement_angle_rad: float = 0.01
    refinement_force_n: float = 25.0
    refine: bool = True

    def __post_init__(self):
        """Require finite positive scales, tolerances, and valid search budgets."""
        for name, value in vars(self).items():
            if name in ("control_count", "max_evaluations"):
                minimum = 4 if name == "control_count" else 1
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{name} must be an integer of at least {minimum}")
            elif name == "refine":
                if not isinstance(value, bool):
                    raise ValueError("refine must be a boolean")
            elif name == "parameter_scale":
                scale = np.asarray(value, dtype=float)
                if scale.shape != (4,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("parameter_scale must contain four finite positive values")
                object.__setattr__(self, name, tuple(float(item) for item in scale))
            elif isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_step_fraction > self.initial_step_fraction:
            raise ValueError("minimum_step_fraction must not exceed initial_step_fraction")


def _weights(time):
    """Return normalized trapezoidal weights on the recorded sample grid."""
    if len(time) == 1:
        return np.ones(1)
    intervals = np.diff(time)
    weights = 0.5 * np.r_[intervals[0], intervals[:-1] + intervals[1:], intervals[-1]]
    return weights / weights.sum()


def _mapping(source, query):
    """Prepare interpolation without extrapolating simulated output."""
    if query[0] < source[0] - 1e-12 or query[-1] > source[-1] + 1e-12:
        raise ValueError("Recorded sample times extend outside simulated support")
    query = np.clip(query, source[0], source[-1])
    if len(source) == 1:
        return np.zeros(len(query), dtype=int), np.zeros(len(query), dtype=int), np.zeros((len(query), 1))
    upper = np.clip(np.searchsorted(source, query, side="right"), 1, len(source) - 1)
    lower = upper - 1
    return lower, upper, ((query - source[lower]) / (source[upper] - source[lower]))[:, None]


def _interpolate(values, mapping):
    """Apply a prepared map to a two-dimensional output array."""
    lower, upper, fraction = mapping
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _complete(trace, summary, duration):
    """Require an entire finite rollout before computing measured fit errors."""
    return bool(
        summary["failure"] is None
        and len(trace["time_s"]) > 0
        and summary["integrated_steps"] == len(trace["time_s"])
        and np.isclose(summary["integrated_duration_s"], duration, rtol=0, atol=1e-12)
        and np.isfinite(trace["state"]).all()
        and np.isfinite(summary["terminal_state"]).all()
        and np.isfinite(trace["grf_n"]).all()
    )


class _Objective:
    """Cache sample maps for the six measured channels on their native grids."""

    def __init__(self, reference, settings):
        self.reference, self.settings = reference, settings
        self.duration = float(reference["time_s"][-1])
        self.time = None
        self.description = {
            "schema": "cartesian_measured_six_channels_1",
            "blocks": ["hip_position", "knee_ankle_angle", "native_grf"],
            "channel_counts": [2, 2, 2],
            "channel_order": ["hip_x", "hip_z", "knee", "ankle", "force_x", "force_z"],
            "quadrature": "normalized trapezoidal weights on each recorded grid; equal block weights",
            "scales": {
                "hip_m": settings.hip_tolerance_m,
                "joint_rad": settings.joint_tolerance_rad,
                "force_n": settings.force_tolerance_n,
            },
            "incomplete_candidates": "completion priority only; no truncated measured fit score",
            "force_support": "native measured times within preintegration simulation support; no extrapolation",
        }
        if "foot_ground_target_rad" in reference:
            self.description["blocks"][1] = "knee_ground_foot_angle"
            self.description["channel_order"][3] = "foot_ground"
            self.description["angle_convention"] = "ground foot pitch; positive toe-up; fixed shoe frame"

    def evaluate(self, trace, summary):
        """Return the shared residual, measured metrics, and block costs."""
        if not _complete(trace, summary, self.duration):
            return None, {}, {}
        time = np.asarray(trace["time_s"])
        if self.time is None:
            motion_time = np.asarray(self.reference["time_s"])
            native_time = np.asarray(self.reference["grf_time_s"])
            mask = (native_time >= time[0]) & (native_time <= time[-1])
            if not np.any(mask):
                raise ValueError("No native GRF samples lie within simulated force support; reduce dt_s")
            self.time = time.copy()
            self.motion_map = _mapping(np.r_[time, self.duration], motion_time)
            self.force_map = _mapping(time, native_time[mask])
            self.motion_weights, self.force_weights = _weights(motion_time), _weights(native_time[mask])
            self.target_force = np.asarray(self.reference["grf_target_n"])[mask]
            self.description.update(
                motion_sample_count=len(motion_time),
                force_sample_count=int(mask.sum()),
                force_interval_s=[float(native_time[mask][0]), float(native_time[mask][-1])],
                native_force_samples_outside_simulated_support=int(np.count_nonzero(~mask)),
            )
        elif not np.array_equal(time, self.time):
            raise ValueError("Simulation time grid changed within a fitting search")
        state = np.vstack((trace["state"], summary["terminal_state"]))
        predicted = _interpolate(state, self.motion_map)
        errors = {
            "hip": predicted[:, :2] - self.reference["hip_target_m"],
            "joint": predicted[:, 3:5] - self.reference["joint_target_rad"],
            "force": _interpolate(np.asarray(trace["grf_n"]), self.force_map) - self.target_force,
        }
        if "foot_ground_target_rad" in self.reference:
            pitch = predicted[:, 2] + predicted[:, 3] + predicted[:, 4] + np.pi / 2
            errors["joint"][:, 1] = (
                pitch - self.reference["shoe_static_pitch_rad"] - self.reference["foot_ground_target_rad"]
            )
        metrics, costs, residuals = {}, {}, []
        for name, unit, scale, weights in (
            ("hip", "m", self.settings.hip_tolerance_m, self.motion_weights),
            ("joint", "rad", self.settings.joint_tolerance_rad, self.motion_weights),
            ("force", "n", self.settings.force_tolerance_n, self.force_weights),
        ):
            error = errors[name]
            residual = (error / scale * np.sqrt(weights[:, None] / 2)).ravel()
            residuals.append(residual)
            costs[name] = float(residual @ residual)
            metrics[f"{name}_rmse_{unit}"] = np.sqrt(np.sum(weights[:, None] * error**2, axis=0)).tolist()
            metrics[f"{name}_maximum_error_{unit}"] = np.max(np.abs(error), axis=0).tolist()
        residual = np.concatenate(residuals)
        if not np.isfinite(residual).all() or not np.isfinite(residual @ residual):
            return None, {}, {}
        return residual, metrics, costs


def _refinement(coarse, coarse_summary, fine, fine_summary, duration, settings):
    """Compare a frozen controller using maximum motion and force differences."""
    result = {
        "performed": True,
        "complete": _complete(fine, fine_summary, duration),
        "passed": False,
        "run": fine_summary,
    }
    if not result["complete"]:
        return result
    coarse_time, fine_time = np.asarray(coarse["time_s"]), np.asarray(fine["time_s"])
    coarse_state = np.vstack((coarse["state"], coarse_summary["terminal_state"]))
    fine_state = np.vstack((fine["state"], fine_summary["terminal_state"]))
    motion_time = np.unique(np.r_[coarse_time, fine_time, duration])
    state_error = _interpolate(fine_state, _mapping(np.r_[fine_time, duration], motion_time)) - _interpolate(
        coarse_state, _mapping(np.r_[coarse_time, duration], motion_time)
    )
    force_time = motion_time[
        (motion_time >= max(coarse_time[0], fine_time[0])) & (motion_time <= min(coarse_time[-1], fine_time[-1]))
    ]
    force_error = _interpolate(np.asarray(fine["grf_n"]), _mapping(fine_time, force_time)) - _interpolate(
        np.asarray(coarse["grf_n"]), _mapping(coarse_time, force_time)
    )
    result.update(
        maximum_hip_position_difference_m=float(np.max(np.linalg.norm(state_error[:, :2], axis=1))),
        maximum_joint_angle_difference_rad=float(np.max(np.abs(state_error[:, 3:5]))),
        maximum_thigh_angle_difference_rad=float(np.max(np.abs(state_error[:, 2]))),
        maximum_grf_difference_n=float(np.max(np.linalg.norm(force_error, axis=1))),
        comparison_grid="union of motion grids including terminal; union of force grids within shared support",
    )
    result["passed"] = bool(
        result["maximum_hip_position_difference_m"] <= settings.refinement_position_m
        and result["maximum_joint_angle_difference_rad"] <= settings.refinement_angle_rad
        and result["maximum_thigh_angle_difference_rad"] <= settings.refinement_angle_rad
        and result["maximum_grf_difference_n"] <= settings.refinement_force_n
    )
    return result
