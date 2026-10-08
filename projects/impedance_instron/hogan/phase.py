# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive the normalized gait phase from simulated contact and hip-over-ankle progression.

The phase ``phi`` runs ``[0, 1)`` before touchdown, ``[1, 2)`` over contact, and
``[2, 3]`` after toe-off. In ``"mechanical"`` mode the simulated state advances it:

* before simulated touchdown, the clock relative to the reference touchdown, held at 1 until contact;
* during contact, the hip-minus-ankle horizontal offset normalized between its reference values
  at touchdown and toe-off;
* after simulated toe-off, the time since toe-off relative to the reference flight.

``phi`` never decreases. The reference, feedforward, and gains are all looked up at
``phi`` through the reference's own phase, so contact timing emerges from the mechanics.
"""

from __future__ import annotations

import numpy as np

from .mechanics import Chain
from .plan import Plan

PHASE_GRID = 6001
TOEOFF_MIN_PHASE = 1.5
"""Toe-off is only accepted after mid-stance, so force dips at impact cannot end contact."""


def reference_timing(plan: Plan, threshold_n: float) -> np.ndarray:
    """Return reference touchdown, toe-off, and duration on the plan clock [s], shape (3,)."""
    contact = np.flatnonzero(plan.grf_n[:, 1] > threshold_n)
    return np.array([plan.touchdown_s, float(plan.time_s[contact[-1]]), plan.duration_s])


def hip_over_ankle(chain: Chain, q) -> np.ndarray:
    """Return the hip-minus-ankle horizontal offset [m] for each row of ``q``, shape [N]."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    thigh = q[:, 2] + q[:, 3] + np.pi
    shank = thigh + q[:, 4]
    return -(chain.lengths_m[0] * np.cos(thigh) + chain.lengths_m[1] * np.cos(shank))


def phase_candidate(time_s, hip_over_ankle_m, touchdown_s, toeoff_s, timing, stance_offset_m) -> float:
    """Return the state-driven phase before the never-decreasing clamp; mirrors the CUDA kernel."""
    if touchdown_s is None:
        return min(time_s / max(timing[0], 1.0e-9), 1.0)
    if toeoff_s is None:
        progress = (hip_over_ankle_m - stance_offset_m[0]) / (stance_offset_m[1] - stance_offset_m[0])
        return 1.0 + min(max(progress, 0.0), 1.0)
    return min(2.0 + (time_s - toeoff_s) / max(timing[2] - timing[1], 1.0e-9), 3.0)


def lookup_position(lookup: np.ndarray, phi: float) -> float:
    """Return the fractional reference sample at ``phi``; mirrors the CUDA lookup."""
    x = min(max(phi, 0.0), 3.0) * (len(lookup) - 1) / 3.0
    i = min(int(x), len(lookup) - 2)
    w = x - i
    return (1.0 - w) * lookup[i] + w * lookup[i + 1]


class MechanicalPhase:
    """Reference phase tables for one stance.

    Args:
        plan: Reference plan.
        chain: Pelvis-leg mechanics of the same stance.
        threshold_n: Vertical force that marks contact [N].

    Raises:
        ValueError: If the reference hip-over-ankle offset does not increase strictly through contact.
    """

    def __init__(self, plan: Plan, chain: Chain, threshold_n: float):
        self.timing = reference_timing(plan, threshold_n)
        touchdown, toeoff, duration = self.timing
        t = plan.time_s
        offset = hip_over_ankle(chain, plan.q)
        self.stance_offset_m = np.array([np.interp(touchdown, t, offset), np.interp(toeoff, t, offset)])
        progress = (offset - self.stance_offset_m[0]) / (self.stance_offset_m[1] - self.stance_offset_m[0])
        phi = np.where(
            t < touchdown,
            t / touchdown,
            np.where(t <= toeoff, 1.0 + progress, 2.0 + (t - toeoff) / (duration - toeoff)),
        )
        if not np.all(np.diff(phi) > 0.0):
            raise ValueError("Reference hip-over-ankle offset must increase strictly through contact")
        self.reference_phase = phi
        self.lookup = np.interp(np.linspace(0.0, 3.0, PHASE_GRID), phi, np.arange(len(t), dtype=float))
