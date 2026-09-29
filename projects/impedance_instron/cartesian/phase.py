# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reference-contact phase gates for Cartesian actuator rollouts."""

from __future__ import annotations

import numpy as np


def hip_contact_gate(reference: dict, times_s: np.ndarray, ramp_duration_s: float) -> np.ndarray:
    """Sample a smooth hip gate from reference vertical GRF contact events.

    Contact uses the rollout convention of vertical GRF greater than 5 N.
    Stance starts and ends are linearly interpolated at that threshold. The gate
    fades in after touchdown and fades out before toe-off using quintic
    smoothstep ramps; it is exactly zero throughout reference flight.

    Args:
        reference: Validated reference containing ``grf_time_s`` and
            ``grf_target_n``.
        times_s: Monotonic rollout sample times [s].
        ramp_duration_s: Duration of each stance transition ramp [s].

    Returns:
        Hip controller multipliers in [0, 1], one per requested time.
    """
    times = np.asarray(times_s, dtype=float)
    ramp = float(ramp_duration_s)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("times_s must be a finite, monotonic one-dimensional array")
    if not np.isfinite(ramp) or ramp <= 0:
        raise ValueError("ramp_duration_s must be finite and positive")
    force_time = np.asarray(reference["grf_time_s"], dtype=float)
    vertical = np.asarray(reference["grf_target_n"], dtype=float)[:, 1]
    contact = vertical > 5.0
    starts, ends = [], []
    edges = np.diff(np.r_[False, contact, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if start == 0:
            touchdown = force_time[0]
        else:
            t0, t1 = force_time[start - 1 : start + 1]
            f0, f1 = vertical[start - 1 : start + 1]
            touchdown = t0 + (5.0 - f0) * (t1 - t0) / (f1 - f0)
        if stop == len(contact):
            toeoff = force_time[-1]
        else:
            t0, t1 = force_time[stop - 1 : stop + 1]
            f0, f1 = vertical[stop - 1 : stop + 1]
            toeoff = t0 + (5.0 - f0) * (t1 - t0) / (f1 - f0)
        starts.append(float(touchdown))
        ends.append(float(toeoff))

    gate = np.zeros(times.shape, dtype=float)
    for touchdown, toeoff in zip(starts, ends):
        stance = (times >= touchdown) & (times <= toeoff)
        if not np.any(stance):
            continue
        phase_in = np.clip((times[stance] - touchdown) / ramp, 0.0, 1.0)
        phase_out = np.clip((toeoff - times[stance]) / ramp, 0.0, 1.0)
        fade_in = phase_in**3 * (10.0 - 15.0 * phase_in + 6.0 * phase_in**2)
        fade_out = phase_out**3 * (10.0 - 15.0 * phase_out + 6.0 * phase_out**2)
        gate[stance] = fade_in * fade_out
    return gate
