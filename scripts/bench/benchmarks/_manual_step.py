# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Manual stepping mode for scaling benchmark.

Measures individual kernel launches at fixed K (no CUDA graph).
Isolated because it imports solver internals.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from scripts.bench.infra import MeasureResult
from scripts.scenes.contact_objects import DT_OUTER, build_model, make_solver
from newton._src.solvers.mujoco.solver_mujoco_cenic import (
    _apply_dt_cap,
    _boundary_advance,
)


def measure_manual(n: int, steps: int, warmup: int) -> MeasureResult:
    """Manual kernel launches (fixed K, no CUDA graph) with fresh solver."""
    model = build_model(n)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # Warmup with step_dt -- run enough steps to reach the contact phase
    # (contact onset ~step 45 for this scene) so K sampling is representative.
    contact_warmup = max(warmup, 50)
    for _ in range(contact_warmup):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()

    # Measure K during the contact phase (not free-fall).
    k_samples = []
    for _ in range(10):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
        k_samples.append(int(solver.iteration_count.numpy()[0]))
    k_fixed = max(round(float(np.mean(k_samples))), 1)
    wp.synchronize()

    dev = model.device
    nw = model.world_count

    times = []
    for _ in range(steps):
        effective_dt_max = min(solver._dt_max, DT_OUTER)
        wp.launch(
            _apply_dt_cap, dim=nw,
            inputs=[solver._ideal_dt, solver._dt_min, effective_dt_max,
                    solver._dt, solver._dt_half],
            device=dev,
        )

        wp.synchronize()
        t0 = time.perf_counter()

        wp.copy(solver._state_cur.joint_q, s0.joint_q)
        wp.copy(solver._state_cur.joint_qd, s0.joint_qd)
        if s0.body_q is not None:
            wp.copy(solver._state_cur.body_q, s0.body_q)
        if s0.body_qd is not None:
            wp.copy(solver._state_cur.body_qd, s0.body_qd)
        solver._apply_mjc_control(model, s0, ctrl, solver.mjw_data)
        solver._enable_rne_postconstraint(solver._state_cur)
        wp.launch(_boundary_advance, dim=nw,
                  inputs=[solver._next_time, DT_OUTER], device=dev)

        for _ in range(k_fixed):
            solver._run_iteration_body(effective_dt_max)

        wp.copy(s0.joint_q, solver._state_cur.joint_q)
        wp.copy(s0.joint_qd, solver._state_cur.joint_qd)
        if s0.body_q is not None:
            wp.copy(s0.body_q, solver._state_cur.body_q)
        if s0.body_qd is not None:
            wp.copy(s0.body_qd, solver._state_cur.body_qd)

        wp.synchronize()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    ks_arr = np.full(steps, k_fixed, dtype=np.int32)

    per_iter = times_arr / max(k_fixed, 1)

    return MeasureResult(
        times=times_arr,
        ks=ks_arr,
        median=float(np.median(times_arr)),
        p25=float(np.percentile(times_arr, 25)),
        p75=float(np.percentile(times_arr, 75)),
        k_mean=float(k_fixed),
        k_max=k_fixed,
        k_p25=float(k_fixed),
        k_p75=float(k_fixed),
        per_iter_median=float(np.median(per_iter)),
    )
