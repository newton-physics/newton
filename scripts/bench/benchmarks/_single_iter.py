# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Single-iteration timing mode for scaling benchmark.

Times one _run_iteration_body() call directly, sync-to-sync,
from a saved contact-phase state. Gives clean per-iteration GPU
cost without K division artifacts.
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


def measure_single_iter(n: int, steps: int, warmup: int) -> MeasureResult:
    """Time a single _run_iteration_body() call from saved contact state."""
    model = build_model(n)
    solver = make_solver(model)
    s0, s1, ctrl = model.state(), model.state(), model.control()

    # Warmup to contact phase.
    contact_warmup = max(warmup, 50)
    for _ in range(contact_warmup):
        s0, s1 = solver.step_dt(DT_OUTER, s0, s1, ctrl)
    wp.synchronize()

    # Set up a step_dt boundary so _run_iteration_body has valid state.
    dev = model.device
    nw = model.world_count
    effective_dt_max = min(solver._dt_max, DT_OUTER)

    wp.launch(
        _apply_dt_cap, dim=nw,
        inputs=[solver._ideal_dt, solver._dt_min, effective_dt_max,
                solver._dt, solver._dt_half],
        device=dev,
    )
    wp.copy(solver._state_cur.joint_q, s0.joint_q)
    wp.copy(solver._state_cur.joint_qd, s0.joint_qd)
    if s0.body_q is not None:
        wp.copy(solver._state_cur.body_q, s0.body_q)
    if s0.body_qd is not None:
        wp.copy(solver._state_cur.body_qd, s0.body_qd)
    solver._apply_mjc_control(model, s0, ctrl, solver.mjw_data)
    solver._enable_rne_postconstraint(solver._state_cur)
    wp.launch(
        _boundary_advance, dim=nw,
        inputs=[solver._next_time, DT_OUTER], device=dev,
    )
    solver._iteration_count_buf.fill_(0)
    solver._boundary_flag.fill_(1)
    wp.synchronize()

    # Save solver internal state for restore between samples.
    saved_cur_q = wp.clone(solver._state_cur.joint_q)
    saved_cur_qd = wp.clone(solver._state_cur.joint_qd)
    saved_cur_bq = wp.clone(solver._state_cur.body_q) if solver._state_cur.body_q is not None else None
    saved_cur_bqd = wp.clone(solver._state_cur.body_qd) if solver._state_cur.body_qd is not None else None
    saved_dt = wp.clone(solver._dt)
    saved_dt_half = wp.clone(solver._dt_half)
    saved_ideal_dt = wp.clone(solver._ideal_dt)
    saved_sim_time = wp.clone(solver._sim_time)
    saved_next_time = wp.clone(solver._next_time)
    wp.synchronize()

    times = []
    for _ in range(steps):
        # Restore to saved contact state.
        wp.copy(solver._state_cur.joint_q, saved_cur_q)
        wp.copy(solver._state_cur.joint_qd, saved_cur_qd)
        if saved_cur_bq is not None:
            wp.copy(solver._state_cur.body_q, saved_cur_bq)
        if saved_cur_bqd is not None:
            wp.copy(solver._state_cur.body_qd, saved_cur_bqd)
        wp.copy(solver._dt, saved_dt)
        wp.copy(solver._dt_half, saved_dt_half)
        wp.copy(solver._ideal_dt, saved_ideal_dt)
        wp.copy(solver._sim_time, saved_sim_time)
        wp.copy(solver._next_time, saved_next_time)
        solver._boundary_flag.fill_(1)
        solver._iteration_count_buf.fill_(0)

        wp.synchronize()
        t0 = time.perf_counter()
        solver._run_iteration_body(effective_dt_max)
        wp.synchronize()
        times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    ks_arr = np.ones(steps, dtype=np.int32)

    return MeasureResult(
        times=times_arr,
        ks=ks_arr,
        median=float(np.median(times_arr)),
        p25=float(np.percentile(times_arr, 25)),
        p75=float(np.percentile(times_arr, 75)),
        k_mean=1.0,
        k_max=1,
        k_p25=1.0,
        k_p75=1.0,
        per_iter_median=float(np.median(times_arr)),
    )
