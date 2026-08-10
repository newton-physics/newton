# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Held-out validation metrics for Digital Instron effective shoe properties.

Implements the official Phase-1/Phase-2 metric definitions: baseline-corrected
force, an active-frame mask at 5% of the robust peak, the top-five robust peak,
active-frame relative RMSE, and positive dissipated hysteresis work from a
loading/unloading branch split at the measured displacement peak. Every metric
is evaluated on held-out cycles only and gated at 10%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TraceValidationMetrics:
    """Pass/fail metrics for one measured-vs-simulated trace."""

    peak_force_error: float
    force_rmse_relative: float
    hysteresis_error: float
    measured_peak_force_n: float
    simulated_peak_force_n: float
    measured_hysteresis_j: float
    simulated_hysteresis_j: float
    active_count: int
    passed: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        """Return the metrics as a plain JSON-serializable mapping."""
        return {
            "peak_force_error": self.peak_force_error,
            "force_rmse_relative": self.force_rmse_relative,
            "hysteresis_error": self.hysteresis_error,
            "measured_peak_force_n": self.measured_peak_force_n,
            "simulated_peak_force_n": self.simulated_peak_force_n,
            "measured_hysteresis_j": self.measured_hysteresis_j,
            "simulated_hysteresis_j": self.simulated_hysteresis_j,
            "active_count": self.active_count,
            "passed": self.passed,
        }


def robust_peak(force_n: np.ndarray, active_mask: np.ndarray | None = None, *, top_count: int = 5) -> float:
    """Return the mean of the top ``top_count`` active force samples."""
    force = np.asarray(force_n, dtype=np.float64)
    if force.ndim != 1:
        raise ValueError("force_n must be a 1D array")
    if top_count <= 0:
        raise ValueError("top_count must be positive")
    if active_mask is None:
        values = force[np.isfinite(force)]
    else:
        mask = np.asarray(active_mask, dtype=bool)
        if mask.shape != force.shape:
            raise ValueError("active_mask must match force_n")
        values = force[mask & np.isfinite(force)]
    if len(values) < top_count:
        raise ValueError(f"Trace has fewer than {top_count} active finite samples")
    return float(np.mean(np.sort(values)[-top_count:]))


def active_force_mask(measured_force_n: np.ndarray, *, active_fraction: float = 0.05, top_count: int = 5) -> np.ndarray:
    """Return the official active-frame mask from measured corrected force."""
    measured = np.asarray(measured_force_n, dtype=np.float64)
    if measured.ndim != 1:
        raise ValueError("measured_force_n must be a 1D array")
    if active_fraction <= 0.0:
        raise ValueError("active_fraction must be positive")
    peak = robust_peak(measured, top_count=top_count)
    active = np.isfinite(measured) & (measured >= active_fraction * peak)
    if int(np.count_nonzero(active)) < top_count:
        raise ValueError(f"Active mask has fewer than {top_count} samples")
    return active


def positive_hysteresis_work(displacement_m: np.ndarray, force_n: np.ndarray, active_mask: np.ndarray) -> float:
    """Return positive dissipated work from the loading and unloading branches."""
    displacement = np.asarray(displacement_m, dtype=np.float64)
    force = np.asarray(force_n, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if displacement.shape != force.shape or displacement.shape != active.shape or displacement.ndim != 1:
        raise ValueError("displacement_m, force_n, and active_mask must be matching 1D arrays")
    active_indices = np.nonzero(active & np.isfinite(displacement) & np.isfinite(force))[0]
    if len(active_indices) < 3:
        raise ValueError("Need at least three active finite samples for hysteresis")
    local_peak = int(active_indices[np.argmax(displacement[active_indices])])
    loading_indices = active_indices[active_indices <= local_peak]
    unloading_indices = active_indices[active_indices >= local_peak]
    if len(loading_indices) < 2 or len(unloading_indices) < 2:
        raise ValueError("Loading and unloading branches must each contain at least two active samples")
    loading_work = float(np.trapezoid(force[loading_indices], displacement[loading_indices]))
    unloading_work = -float(np.trapezoid(force[unloading_indices], displacement[unloading_indices]))
    return loading_work - unloading_work


def validate_trace_metrics(
    measured_force_n: np.ndarray,
    simulated_force_n: np.ndarray,
    displacement_m: np.ndarray,
    *,
    active_fraction: float = 0.05,
    top_count: int = 5,
    pass_threshold: float = 0.10,
) -> TraceValidationMetrics:
    """Compute the official held-out trace metrics from corrected force.

    Args:
        measured_force_n: Baseline-corrected measured force [N].
        simulated_force_n: Baseline-corrected simulated force [N].
        displacement_m: Indenter penetration aligned with the force samples [m].
        active_fraction: Active-frame threshold as a fraction of the robust peak.
        top_count: Number of samples averaged for the robust peak.
        pass_threshold: Gate applied to each relative error.

    Returns:
        The per-trace :class:`TraceValidationMetrics`.
    """
    measured = np.asarray(measured_force_n, dtype=np.float64)
    simulated = np.asarray(simulated_force_n, dtype=np.float64)
    displacement = np.asarray(displacement_m, dtype=np.float64)
    if measured.shape != simulated.shape or measured.shape != displacement.shape or measured.ndim != 1:
        raise ValueError("measured_force_n, simulated_force_n, and displacement_m must be matching 1D arrays")

    active = active_force_mask(measured, active_fraction=active_fraction, top_count=top_count)
    measured_peak = robust_peak(measured, active, top_count=top_count)
    simulated_peak = robust_peak(simulated, active, top_count=top_count)
    peak_error = abs(simulated_peak - measured_peak) / max(abs(measured_peak), 1.0e-9)
    residual = simulated[active] - measured[active]
    rmse_relative = float(np.sqrt(np.mean(residual**2)) / max(abs(measured_peak), 1.0e-9))
    measured_hysteresis = positive_hysteresis_work(displacement, measured, active)
    simulated_hysteresis = positive_hysteresis_work(displacement, simulated, active)
    hysteresis_error = abs(simulated_hysteresis - measured_hysteresis) / max(abs(measured_hysteresis), 1.0e-9)
    passed = peak_error < pass_threshold and rmse_relative < pass_threshold and hysteresis_error < pass_threshold
    return TraceValidationMetrics(
        peak_force_error=float(peak_error),
        force_rmse_relative=rmse_relative,
        hysteresis_error=float(hysteresis_error),
        measured_peak_force_n=measured_peak,
        simulated_peak_force_n=simulated_peak,
        measured_hysteresis_j=float(measured_hysteresis),
        simulated_hysteresis_j=float(simulated_hysteresis),
        active_count=int(np.count_nonzero(active)),
        passed=bool(passed),
    )
