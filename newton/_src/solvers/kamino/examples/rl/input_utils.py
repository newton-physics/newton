# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Torch-free scalar input processing helpers for RL examples."""

from __future__ import annotations

# Python
import math


def _deadband(value: float, threshold: float) -> float:
    """Remove a dead zone and rescale the remaining range."""
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - threshold) / (1.0 - threshold)


def _scale_asym(value: float, negative_scale: float, positive_scale: float) -> float:
    """Scale a signed value with separate negative and positive limits."""
    return value * negative_scale if value < 0.0 else value * positive_scale


class _LowPassFilter:
    """Scalar backward-Euler low-pass filter."""

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        omega = cutoff_hz * 2.0 * math.pi
        self.alpha = omega * dt / (omega * dt + 1.0)
        self.value: float | None = None

    def update(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * value
        return self.value

    def reset(self) -> None:
        self.value = None


class RateLimitedValue:
    """Clamp the rate of change of a scalar value."""

    def __init__(self, rate_limit: float, dt: float) -> None:
        self.rate_limit = rate_limit
        self.dt = dt
        self.value = 0.0
        self._initialized = False

    def update(self, target: float) -> float:
        if not self._initialized:
            self._initialized = True
            self.value = target
        else:
            max_delta = self.rate_limit * self.dt
            self.value += max(-max_delta, min(target - self.value, max_delta))
        return self.value

    def reset(self) -> None:
        self.value = 0.0
        self._initialized = False
