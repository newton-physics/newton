# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark measurement infrastructure.

Provides fresh-solver-per-measurement timing and power-law fitting.
Every measurement builds a new model, solver, and states from scratch
to prevent state contamination between modes or N values.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import warp as wp

import newton


@dataclass
class MeasureResult:
    """Timing result from a single (N, mode) measurement."""

    times: np.ndarray
    ks: np.ndarray
    median: float
    p25: float
    p75: float
    k_mean: float
    k_max: int
    k_p25: float
    k_p75: float
    per_iter_median: float  # median of (time_i / K_i) -- the real per-iteration cost


def measure(
    build_model_fn: Callable[[int], newton.Model],
    step_fn: Callable[
        [newton.Model, newton.State, newton.State, newton.Control],
        tuple[newton.State, newton.State],
    ],
    n: int,
    steps: int,
    warmup: int,
    get_k: Callable[..., int] | None = None,
) -> MeasureResult:
    """Run a benchmark with a fresh model and solver.

    Args:
        build_model_fn: Callable that takes n_worlds and returns a Model.
        step_fn: Callable that takes (model, s0, s1, ctrl) and returns (s0, s1).
            The step_fn is responsible for creating its own solver internally
            on first call (via closure) or receiving it as a bound argument.
        n: Number of worlds.
        steps: Number of timed steps.
        warmup: Number of warmup steps (not timed).
        get_k: Optional callable returning iteration count after each step.
    """
    model = build_model_fn(n)
    s0 = model.state()
    s1 = model.state()
    ctrl = model.control()

    for _ in range(warmup):
        s0, s1 = step_fn(model, s0, s1, ctrl)
    wp.synchronize()

    times = []
    ks = []
    for _ in range(steps):
        wp.synchronize()
        t0 = time.perf_counter()
        s0, s1 = step_fn(model, s0, s1, ctrl)
        wp.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        ks.append(get_k() if get_k is not None else 1)

    times_arr = np.array(times)
    ks_arr = np.array(ks, dtype=np.int32)

    per_iter = times_arr / np.maximum(ks_arr, 1)

    return MeasureResult(
        times=times_arr,
        ks=ks_arr,
        median=float(np.median(times_arr)),
        p25=float(np.percentile(times_arr, 25)),
        p75=float(np.percentile(times_arr, 75)),
        k_mean=float(np.mean(ks_arr)),
        k_max=int(np.max(ks_arr)),
        k_p25=float(np.percentile(ks_arr, 25)),
        k_p75=float(np.percentile(ks_arr, 75)),
        per_iter_median=float(np.median(per_iter)),
    )


def power_law_exponent(ns: list[int], values: list[float]) -> float:
    """Fit log(value) = alpha * log(N) + c; return alpha."""
    valid = [(n, v) for n, v in zip(ns, values) if v > 0]
    if len(valid) < 2:
        return float("nan")
    log_n = np.log([v[0] for v in valid])
    log_v = np.log([v[1] for v in valid])
    alpha, _ = np.polyfit(log_n, log_v, 1)
    return float(alpha)
