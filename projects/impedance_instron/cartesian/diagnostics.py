# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize rollout velocity, hip-load, and friction-proxy diagnostics."""

from __future__ import annotations

import numpy as np


def summarize(trace: dict, reference: dict, friction_mu: float) -> dict:
    """Return rollout-only diagnostics without changing the controller objective.

    The Coulomb-equivalent ratio compares net horizontal ground force with
    ``mu * normal force``. It is a proxy for proximity to a Coulomb limit, not
    a report of the internal state of viscoelastic friction models.
    """
    time = np.asarray(trace["time_s"], dtype=float)
    velocity = np.asarray(trace["velocity"], dtype=float)[:, :2]
    target_time = np.asarray(reference["time_s"], dtype=float)
    target_velocity = np.asarray(reference["velocity"], dtype=float)[:, :2]
    if len(time):
        target = np.column_stack([np.interp(time, target_time, target_velocity[:, i]) for i in range(2)])
        velocity_rmse = np.sqrt(np.mean(np.square(velocity - target), axis=0))
        maximum_speed = float(np.max(np.linalg.norm(velocity, axis=1)))
    else:
        velocity_rmse = np.array([np.nan, np.nan])
        maximum_speed = float("nan")
    force = np.asarray(trace["grf_n"], dtype=float)
    contact = force[:, 1] > 5.0 if len(force) else np.zeros(0, dtype=bool)
    flight = ~contact
    ratios = np.abs(force[contact, 0]) / (friction_mu * force[contact, 1]) if np.any(contact) else np.zeros(0)
    spring = np.asarray(trace["hip_spring_force_n"], dtype=float)
    damping = np.asarray(trace["hip_damping_force_n"], dtype=float)
    dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    return {
        "hip_velocity_rmse_x_m_s": float(velocity_rmse[0]),
        "hip_velocity_rmse_z_m_s": float(velocity_rmse[1]),
        "maximum_hip_speed_m_s": maximum_speed,
        "maximum_hip_spring_force_n": float(np.max(np.linalg.norm(spring, axis=1))) if len(spring) else None,
        "maximum_hip_damping_force_n": float(np.max(np.linalg.norm(damping, axis=1))) if len(damping) else None,
        "maximum_coulomb_equivalent_ratio": float(np.max(ratios)) if len(ratios) else None,
        "fraction_contact_samples_near_coulomb_limit": float(np.mean(ratios >= 0.95)) if len(ratios) else None,
        "contact_sample_fraction": float(np.mean(contact)) if len(contact) else 0.0,
        "minimum_hip_spring_vertical_force_in_flight_n": float(np.min(spring[flight, 1])) if np.any(flight) else None,
        "hip_spring_work_in_flight_j": float(np.sum(np.sum(spring[flight] * velocity[flight], axis=1)) * dt)
        if np.any(flight)
        else 0.0,
        "hip_damping_work_in_flight_j": float(np.sum(np.sum(damping[flight] * velocity[flight], axis=1)) * dt)
        if np.any(flight)
        else 0.0,
        "contact_normal_threshold_n": 5.0,
        "coulomb_equivalent_ratio_is_internal_friction_state": False,
    }
