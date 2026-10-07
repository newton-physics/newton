# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Phase-scheduled impedance about a reference plan."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Impedance:
    """Piecewise-linear stiffness and damping schedules for all six coordinates.

    The control law is ``tau = tau_ff(phi) + K(phi) (q_ref(phi) - q) + D(phi) (v_ref(phi) - v)``.
    Gains are held constant beyond the first and last knots. Channel order and
    units follow :data:`.mechanics.COORDINATE_NAMES`: the pelvis translation
    channels use [N/m] and [N s/m]; the angular channels use [N m/rad] and
    [N m s/rad].

    Args:
        knot_s: Strictly increasing knot phases [s], shape [n].
        stiffness: Nonnegative stiffness at each knot, shape [n, 6].
        damping: Nonnegative damping at each knot, shape [n, 6].
    """

    def __init__(self, knot_s, stiffness, damping):
        self.knot_s = np.asarray(knot_s, dtype=float).copy()
        self.stiffness = np.asarray(stiffness, dtype=float).copy()
        self.damping = np.asarray(damping, dtype=float).copy()
        n = len(self.knot_s)
        if self.knot_s.ndim != 1 or n == 0 or np.any(np.diff(self.knot_s) <= 0):
            raise ValueError("knot_s must be a nonempty strictly increasing vector")
        for name in ("knot_s", "stiffness", "damping"):
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")
        for name in ("stiffness", "damping"):
            value = getattr(self, name)
            if value.shape != (n, 6):
                raise ValueError(f"{name} must have shape ({n}, 6)")
            if np.any(value < 0):
                raise ValueError(f"{name} must be nonnegative")

    @classmethod
    def critically_damped(
        cls, stiffness, inertia, duration_s: float, *, damping_ratio: float = 1.0, knot_count: int = 2
    ) -> Impedance:
        """Build constant gains with ``D = 2 zeta sqrt(K M_ii)`` on uniform knots.

        Args:
            stiffness: Per-channel stiffness, shape (6,).
            inertia: Per-channel effective inertia, usually the mass-matrix diagonal, shape (6,).
            duration_s: Phase span covered by the knots [s].
            damping_ratio: Dimensionless damping ratio ``zeta``.
            knot_count: Number of uniform knots; extra knots give later fits room to vary gains.
        """
        stiffness = np.asarray(stiffness, dtype=float)
        inertia = np.asarray(inertia, dtype=float)
        if stiffness.shape != (6,) or inertia.shape != (6,) or np.any(inertia <= 0):
            raise ValueError("stiffness and positive inertia must have shape (6,)")
        if knot_count < 1 or not np.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("knot_count and duration_s must be positive")
        damping = 2.0 * damping_ratio * np.sqrt(stiffness * inertia)
        knots = np.linspace(0.0, duration_s, knot_count) if knot_count > 1 else np.zeros(1)
        return cls(knots, np.tile(stiffness, (knot_count, 1)), np.tile(damping, (knot_count, 1)))

    def gains(self, phase_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return stiffness and damping at a phase, each shape (6,)."""
        if len(self.knot_s) == 1:
            return self.stiffness[0].copy(), self.damping[0].copy()
        i = int(np.clip(np.searchsorted(self.knot_s, phase_s, side="right") - 1, 0, len(self.knot_s) - 2))
        w = float(np.clip((phase_s - self.knot_s[i]) / (self.knot_s[i + 1] - self.knot_s[i]), 0.0, 1.0))
        return (
            (1.0 - w) * self.stiffness[i] + w * self.stiffness[i + 1],
            (1.0 - w) * self.damping[i] + w * self.damping[i + 1],
        )

    def scaled(self, stiffness_factor, damping_factor) -> Impedance:
        """Return a copy with per-channel stiffness and damping multipliers, each shape (6,)."""
        return Impedance(
            self.knot_s,
            self.stiffness * np.asarray(stiffness_factor, dtype=float),
            self.damping * np.asarray(damping_factor, dtype=float),
        )

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Write the schedule and optional JSON metadata to an NPZ file."""
        np.savez(
            Path(path),
            knot_s=self.knot_s,
            stiffness=self.stiffness,
            damping=self.damping,
            metadata_json=np.asarray(json.dumps(metadata or {}, allow_nan=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> Impedance:
        """Read a schedule written by :meth:`save` without pickle."""
        with np.load(Path(path), allow_pickle=False) as archive:
            return cls(archive["knot_s"], archive["stiffness"], archive["damping"])
