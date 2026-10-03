# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fast temperature check for FPGS articulated-contact propagation.

Runs a small config matrix that isolates the three scaling axes (contact
count, chain depth, articulation count) across the immediate, propagation,
and propagation-fused response modes. For each point it reports:

- per-step wall time (contacts precomputed, solver-only, restore-state loop),
- analytic row/propagation buffer memory,
- one-step generalized-velocity error vs immediate,
- settled articulated-contact penetration under gravity after consecutive
  steps with live collision, plus generalized-velocity drift vs immediate.

The settle phase exists because the one-step error cannot see slow-coupling
regressions that only show up in resting contact (the P11-class anomaly).

Usage (from repo root, kernels cache after first run):

    python newton/tests/benchmarks/fpgs_propagation_temperature.py \
        --out /tmp/temp.json [--baseline artifacts/.../baseline.json]

    --update-baseline     overwrite the baseline with this run
    --kernels CASE|MODE   print top GPU kernels for one point (wp timing)
    --stages              enable FEATHER_PGS_SYNC_TIMINGS stage prints
    --case SUBSTR         run only cases whose label contains SUBSTR
"""

from __future__ import annotations

import os
import sys

if "--stages" in sys.argv:
    os.environ.setdefault("FEATHER_PGS_SYNC_TIMINGS", "1")
    os.environ.setdefault("FEATHER_PGS_SYNC_TIMINGS_START", "3")
    os.environ.setdefault("FEATHER_PGS_SYNC_TIMINGS_COUNT", "1")

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from fpgs_articulation_row_scaling import (
    PROPAGATION_PATHS,
    BenchCase,
    _abs_linf,
    _build_articulated_free_builder,
    _configure_shape_defaults,
    _planned_row_capacity,
    _propagation_extra_bytes,
    _restore_state,
    _row_solver_bytes,
    _snapshot_state,
    scalar_hinv_jt,
)

import newton
from newton.solvers import SolverFeatherPGS

MODES = ("mf_immediate", "propagation", "propagation-fused", "propagation-colored")
CONTACT_ROW_TYPE = 0


@dataclass(frozen=True)
class TempCase:
    label: str
    axis: str
    articulations: int
    links: int
    contacts_per_articulation: int
    world_count: int
    settle_world_count: int


# One axis moves at a time. world_count sized for stable GPU timing while
# keeping the whole sweep in the low minutes; settle uses fewer envs because
# accuracy does not need scale.
TEMPERATURE_CASES = (
    TempCase("contacts:A1,L8,C2", "contacts", 1, 8, 2, 1024, 64),
    TempCase("contacts:A1,L8,C16", "contacts", 1, 8, 16, 1024, 64),
    TempCase("depth:A1,L4,C2", "depth", 1, 4, 2, 1024, 64),
    TempCase("depth:A1,L32,C2", "depth", 1, 32, 2, 1024, 64),
    TempCase("arts:A4,L8,C1", "arts", 4, 8, 1, 512, 32),
    TempCase("arts:A32,L8,C1", "arts", 32, 8, 1, 512, 32),
)


def _bench_case(case: TempCase, world_count: int, spawn_penetration: float = 0.012) -> BenchCase:
    return BenchCase(
        sweep=case.axis,
        label=case.label,
        case_kind="articulated_free",
        world_count=world_count,
        articulations=case.articulations,
        links=case.links,
        contacts_per_articulation=case.contacts_per_articulation,
        spawn_penetration=spawn_penetration,
    )


def _build_model(case: TempCase, world_count: int, device: str, gravity: float) -> newton.Model:
    # Settle scenes start in gentle contact so gravity loads the boxes onto the
    # links instead of the deep spawn overlap ejecting them.
    spawn_penetration = 0.0005 if gravity != 0.0 else 0.012
    template, spacing = _build_articulated_free_builder(_bench_case(case, world_count, spawn_penetration))
    gravity_vector = (0.0, 0.0, gravity)
    if gravity != 0.0:
        template.gravity = gravity_vector
    if world_count <= 1:
        return template.finalize(device=device)
    builder = newton.ModelBuilder(gravity=gravity_vector)
    _configure_shape_defaults(builder)
    builder.replicate(template, world_count, spacing=spacing)
    return builder.finalize(device=device)


def _make_solver(model: newton.Model, case: TempCase, mode: str, pgs_iterations: int) -> SolverFeatherPGS:
    row_capacity = _planned_row_capacity(_bench_case(case, 1))
    with scalar_hinv_jt():
        return SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="immediate" if mode not in PROPAGATION_PATHS else mode,
            pgs_iterations=pgs_iterations,
            pgs_velocity_iterations=0,
            enable_contact_friction=True,
            contact_friction_position_iterations=-1,
            dense_max_constraints=row_capacity,
            mf_max_constraints=16,
            pgs_warmstart=False,
            mf_warmstart=False,
        )


def _articulated_contact_penetration_m(solver: SolverFeatherPGS) -> float:
    """Max articulated-contact penetration over dense + propagation rows [m]."""
    peak = 0.0
    for count_name, phi_name, type_name in (
        ("constraint_count", "phi", "row_type"),
        ("propagation_constraint_count", "propagation_phi", "propagation_row_type"),
    ):
        counts = getattr(solver, count_name, None)
        phi = getattr(solver, phi_name, None)
        row_type = getattr(solver, type_name, None)
        if counts is None or phi is None or row_type is None:
            continue
        counts_np = counts.numpy()
        phi_np = phi.numpy()
        type_np = row_type.numpy()
        cols = np.arange(phi_np.shape[1])[None, :]
        active = cols < counts_np[:, None]
        normal = type_np == CONTACT_ROW_TYPE
        mask = active & normal & (phi_np < 0.0)
        if np.any(mask):
            peak = max(peak, float(np.max(-phi_np[mask])))
    return peak


def _timed_loop(model, solver, contacts, initial, *, repeats: int, warmups: int, dt: float):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    for _ in range(warmups):
        _restore_state(model, state_in, initial)
        _restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
    wp.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _restore_state(model, state_in, initial)
        _restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
    wp.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / repeats
    return ms, _snapshot_state(state_out)


def _settle_loop(model, solver, *, steps: int, dt: float, tail: int):
    """Consecutive steps with live collision; returns tail-window peak penetration [m] and final state."""
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    peak_tail = 0.0
    for step in range(steps):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        if step >= steps - tail:
            peak_tail = max(peak_tail, _articulated_contact_penetration_m(solver))
        state_in, state_out = state_out, state_in
    return peak_tail, _snapshot_state(state_in)


def run_point(
    case: TempCase,
    mode: str,
    *,
    device: str,
    dt: float,
    repeats: int,
    warmups: int,
    pgs_iterations: int,
    settle_steps: int,
    settle_tail: int,
    timing_model_cache: dict,
    settle_model_cache: dict,
) -> dict:
    if case.label not in timing_model_cache:
        model = _build_model(case, case.world_count, device, gravity=0.0)
        initial_state = model.state()
        newton.eval_fk(model, initial_state.joint_q, initial_state.joint_qd, initial_state)
        initial_state.clear_forces()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(initial_state, contacts)
        wp.synchronize()
        timing_model_cache[case.label] = (model, _snapshot_state(initial_state), contacts)
    model, initial, contacts = timing_model_cache[case.label]

    solver = _make_solver(model, case, mode, pgs_iterations)
    ms, final_state = _timed_loop(model, solver, contacts, initial, repeats=repeats, warmups=warmups, dt=dt)
    row_mib = _row_solver_bytes(solver) / (1024.0 * 1024.0)
    prop_extra_mib = _propagation_extra_bytes(solver) / (1024.0 * 1024.0)
    prop_rows = getattr(solver, "propagation_constraint_count", None)
    prop_rows_total = int(np.sum(prop_rows.numpy())) if prop_rows is not None else 0
    del solver

    if case.label not in settle_model_cache:
        settle_model_cache[case.label] = _build_model(case, case.settle_world_count, device, gravity=-9.81)
    settle_model = settle_model_cache[case.label]
    settle_solver = _make_solver(settle_model, case, mode, pgs_iterations)
    settle_pen_m, settle_state = _settle_loop(settle_model, settle_solver, steps=settle_steps, dt=dt, tail=settle_tail)
    del settle_solver

    return {
        "case": case.label,
        "axis": case.axis,
        "mode": mode,
        "world_count": case.world_count,
        "ms_per_step": ms,
        "row_mib": row_mib,
        "prop_extra_mib": prop_extra_mib,
        "propagation_rows_total": prop_rows_total,
        "settle_pen_mm": settle_pen_m * 1000.0,
        "_final_state": final_state,
        "_settle_state": settle_state,
    }


def _attach_reference_errors(rows: list[dict]) -> None:
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["mode"]] = row
    for case_rows in by_case.values():
        ref = case_rows.get("mf_immediate")
        for row in case_rows.values():
            if ref is None:
                row["qd1_linf"] = None
                row["settle_qd_linf"] = None
            else:
                row["qd1_linf"] = _abs_linf(row["_final_state"]["joint_qd"], ref["_final_state"]["joint_qd"])
                row["settle_qd_linf"] = _abs_linf(row["_settle_state"]["joint_qd"], ref["_settle_state"]["joint_qd"])
    for row in rows:
        row.pop("_final_state", None)
        row.pop("_settle_state", None)


def _print_table(rows: list[dict], baseline: dict[str, dict] | None) -> None:
    header = (
        f"{'case':<22} {'mode':<18} {'ms/step':>9} {'vs MF':>7} {'rowMiB':>8} "
        f"{'+prop':>7} {'qd1max':>9} {'settle_mm':>10} {'settle_qd':>10} {'d_base':>8}"
    )
    print(header)
    print("-" * len(header))
    by_case: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], {})[row["mode"]] = row
    for case_label, case_rows in by_case.items():
        mf_ms = case_rows.get("mf_immediate", {}).get("ms_per_step")
        for mode in MODES:
            row = case_rows.get(mode)
            if row is None:
                continue
            speed = f"{mf_ms / row['ms_per_step']:.2f}x" if mf_ms else ""
            delta = ""
            if baseline:
                base = baseline.get(f"{case_label}|{mode}")
                if base and base.get("ms_per_step"):
                    pct = 100.0 * (row["ms_per_step"] - base["ms_per_step"]) / base["ms_per_step"]
                    delta = f"{pct:+.1f}%"
            qd1 = row.get("qd1_linf")
            sqd = row.get("settle_qd_linf")
            print(
                f"{case_label:<22} {mode:<18} {row['ms_per_step']:>9.3f} {speed:>7} "
                f"{row['row_mib']:>8.2f} {row['prop_extra_mib']:>7.2f} "
                f"{qd1 if qd1 is None else f'{qd1:.2e}':>9} "
                f"{row['settle_pen_mm']:>10.3f} "
                f"{sqd if sqd is None else f'{sqd:.2e}':>10} {delta:>8}"
            )
        print()


def _profile_kernels(case: TempCase, mode: str, *, device: str, dt: float, pgs_iterations: int, top: int = 20) -> None:
    model = _build_model(case, case.world_count, device, gravity=0.0)
    initial_state = model.state()
    newton.eval_fk(model, initial_state.joint_q, initial_state.joint_qd, initial_state)
    initial_state.clear_forces()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(initial_state, contacts)
    initial = _snapshot_state(initial_state)
    solver = _make_solver(model, case, mode, pgs_iterations)
    # Warm every kernel before recording.
    _timed_loop(model, solver, contacts, initial, repeats=2, warmups=2, dt=dt)

    state_in = model.state()
    state_out = model.state()
    control = model.control()
    steps = 5
    wp.timing_begin(cuda_filter=wp.TIMING_KERNEL | wp.TIMING_MEMSET | wp.TIMING_MEMCPY)
    for _ in range(steps):
        _restore_state(model, state_in, initial)
        _restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
    results = wp.timing_end()

    agg: dict[str, list[float]] = {}
    for r in results:
        name = getattr(r, "name", str(r))
        agg.setdefault(name, []).append(float(getattr(r, "elapsed", 0.0)))
    total = sum(sum(v) for v in agg.values())
    print(
        f"\nTop kernels for {case.label} [{mode}] over {steps} steps "
        f"(total GPU {total:.3f} ms, {total / steps:.3f} ms/step):"
    )
    print(f"{'kernel':<64} {'count':>6} {'total_ms':>9} {'mean_us':>9} {'%':>6}")
    for name, times in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:top]:
        t = sum(times)
        print(f"{name[:64]:<64} {len(times):>6} {t:>9.3f} {1000.0 * t / len(times):>9.1f} {100.0 * t / total:>6.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--pgs-iterations", type=int, default=8)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--settle-tail", type=int, default=10)
    parser.add_argument("--case", default=None, help="substring filter on case label")
    parser.add_argument("--mode", action="append", choices=MODES, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("artifacts/fpgs_articulation_profiles/temperature_baseline.json"),
    )
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument(
        "--kernels", default=None, metavar="CASE|MODE", help="profile one point, e.g. 'contacts:A1,L8,C16|propagation'"
    )
    parser.add_argument("--stages", action="store_true", help="enable FEATHER_PGS_SYNC_TIMINGS stage prints")
    args = parser.parse_args()

    wp.init()
    if not wp.get_device(args.device).is_cuda:
        raise RuntimeError("temperature harness requires a CUDA device")

    cases = [c for c in TEMPERATURE_CASES if args.case is None or args.case in c.label]
    if not cases:
        raise ValueError(f"--case {args.case!r} matched nothing")

    if args.kernels:
        case_label, _, mode = args.kernels.partition("|")
        matches = [c for c in TEMPERATURE_CASES if c.label == case_label]
        if not matches or mode not in MODES:
            raise ValueError(
                f"--kernels expects 'CASE|MODE' with CASE in {[c.label for c in TEMPERATURE_CASES]} and MODE in {MODES}"
            )
        _profile_kernels(matches[0], mode, device=args.device, dt=args.dt, pgs_iterations=args.pgs_iterations)
        return

    modes = tuple(dict.fromkeys(args.mode)) if args.mode else MODES
    rows: list[dict] = []
    timing_model_cache: dict = {}
    settle_model_cache: dict = {}
    started = time.perf_counter()
    for case in cases:
        for mode in modes:
            rows.append(
                run_point(
                    case,
                    mode,
                    device=args.device,
                    dt=args.dt,
                    repeats=args.repeats,
                    warmups=args.warmups,
                    pgs_iterations=args.pgs_iterations,
                    settle_steps=args.settle_steps,
                    settle_tail=args.settle_tail,
                    timing_model_cache=timing_model_cache,
                    settle_model_cache=settle_model_cache,
                )
            )
        # Free cached models between cases to keep peak memory flat.
        timing_model_cache.pop(case.label, None)
        settle_model_cache.pop(case.label, None)

    _attach_reference_errors(rows)

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    _print_table(rows, baseline)
    print(f"total wall time {time.perf_counter() - started:.1f} s")

    payload = {f"{r['case']}|{r['mode']}": r for r in rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"updated baseline {args.baseline}")


if __name__ == "__main__":
    main()
