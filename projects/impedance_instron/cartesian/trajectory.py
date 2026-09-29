# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""C2 Cartesian hip/joint and optional ankle-position equilibrium splines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import _derivative_control_polygons, basis


def _vector(value, name: str, channels: int) -> np.ndarray:
    """Validate a finite equilibrium vector with the spline's channel count."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (channels,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {channels}-element vector")
    return array


@dataclass(frozen=True)
class Spline:
    """Clamped cubic equilibrium with four or six mixed-unit channels.

    Simple interior knots give continuous position, rate, and acceleration.
    The first four channels are hip XY [m] and knee/ankle angles [rad]. A
    six-channel spline adds ankle XY position [m] as channels five and six.
    """

    duration_s: float
    coefficients: np.ndarray

    def __post_init__(self):
        """Copy finite coefficients and validate the spline dimensions."""
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 2 or coefficients.shape[1] not in (4, 6) or len(coefficients) < 4:
            raise ValueError("coefficients must have shape (control_count >= 4, 4 or 6)")
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite")
        duration = float(self.duration_s)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration_s must be finite and positive")
        coefficients = coefficients.copy()
        coefficients.setflags(write=False)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "coefficients", coefficients)

    def sample(self, time_s):
        """Return equilibrium, rate, and acceleration in [m, m, rad, rad] per time order."""
        return tuple(
            basis(time_s, self.duration_s, len(self.coefficients), derivative=order) @ self.coefficients
            for order in range(3)
        )

    def bounds(self, lower, upper, rate, acceleration) -> bool:
        """Check sufficient global mixed-unit position and derivative limits.

        The convex hulls of the control polygons bound the entire curve.
        These bounds are conservative, not sampled approximations.

        Args:
            lower: Lower equilibrium bounds [m, m, rad, rad].
            upper: Upper equilibrium bounds [m, m, rad, rad].
            rate: Absolute rate limits [m/s, m/s, rad/s, rad/s].
            acceleration: Absolute acceleration limits [m/s^2, m/s^2, rad/s^2, rad/s^2].
        """
        channels = self.coefficients.shape[1]
        lower, upper = _vector(lower, "lower", channels), _vector(upper, "upper", channels)
        rate, acceleration = _vector(rate, "rate", channels), _vector(acceleration, "acceleration", channels)
        if np.any(lower > upper) or np.any(rate < 0) or np.any(acceleration < 0):
            raise ValueError("Bounds must be ordered and derivative limits nonnegative")
        first, second = _derivative_control_polygons(self.coefficients)
        return bool(
            np.all(self.coefficients >= lower)
            and np.all(self.coefficients <= upper)
            and np.all(np.abs(first) <= rate * self.duration_s)
            and np.all(np.abs(second) <= acceleration * self.duration_s**2)
        )
