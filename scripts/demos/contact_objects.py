# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive contact-objects demo using the CENIC adaptive solver.

Runs a scene of 9 spheres and 9 tilted boxes falling onto a ground plane,
contained by invisible walls. Supports both CENIC adaptive stepping and
fixed-dt stepping for comparison.

Usage::

    uv run python -m scripts.demos.contact_objects [--num-worlds N] [--headless] [--fixed-dt DT]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.contact_objects import DT_OUTER, LOG_EVERY, build_model, make_solver

_grid_lines = 0


def _print_status(solver, step):
    global _grid_lines

    n = solver.model.world_count

    if n > 4:
        s = solver.get_status_summary()
        lines = [
            f"  step {step}  tol={solver._tol:.1e}  worlds={n}",
            f"  sim_time  [{s['sim_time_min']:.4f}, {s['sim_time_max']:.4f}] s",
            f"  dt        [{s['dt_min']:.6f}, {s['dt_max']:.6f}] s",
            f"  err_max   {s['error_max']:.3e}",
            f"  accepted  {s['accept_count']}/{n}",
        ]
    else:
        sim_times = solver.sim_time.numpy()
        dts = solver.dt.numpy()
        errors = solver.last_error.numpy()
        accepted = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'L2 error':>{col}}{'status':>{col}}"
        lines = [f"  step {step}  tol={solver._tol:.1e}", bar, hdr, bar]
        for i in range(len(sim_times)):
            lines.append(
                f"{'world ' + str(i):>{col}}"
                f"{sim_times[i]:>{col}.4f}"
                f"{dts[i]:>{col}.6f}"
                f"{errors[i]:>{col}.3e}"
                f"{'ok' if accepted[i] else 'REJECT':>{col}}"
            )
        lines.append(bar)

    if _grid_lines > 0:
        sys.stdout.write(f"\033[{_grid_lines}A")
    sys.stdout.write("\n".join(f"\033[2K{l}" for l in lines) + "\n")
    sys.stdout.flush()
    _grid_lines = len(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-worlds", type=int, default=1, help="parallel worlds")
    parser.add_argument("--num-steps", type=int, default=0, help="0 = run until closed")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--fixed-dt", type=float, default=None, help="Use fixed-step SolverMuJoCo with this dt instead of CENIC"
    )
    args = parser.parse_args()

    model = build_model(args.num_worlds)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    use_fixed = args.fixed_dt is not None

    if use_fixed:
        solver = newton.solvers.SolverMuJoCo(model, separate_worlds=True, nconmax=128, njmax=640)
        contacts = model.contacts()
        n_inner = round(DT_OUTER / args.fixed_dt)
        print(
            f"Fixed-step demo: {args.num_worlds} world(s)  solver=SolverMuJoCo  "
            f"dt={args.fixed_dt:.4e}  substeps/outer={n_inner}",
            flush=True,
        )
    else:
        solver = make_solver(model)
        print(
            f"CENIC contact demo: {args.num_worlds} world(s)  solver=SolverMuJoCoCENIC  "
            f"tol={solver._tol:.1e}  dt_inner_init={solver._dt.numpy()[0]:.4f}  "
            f"dt_inner_max={solver._dt_max:.4f}",
            flush=True,
        )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(
        pos=wp.vec3(1.97, -2.07, 1.07),
        pitch=-22.5,
        yaw=136.3,
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        if use_fixed:
            for _ in range(n_inner):
                state_1 = solver.step(state_0, state_1, control, contacts, args.fixed_dt)
                state_0, state_1 = state_1, state_0
        else:
            state_0, state_1 = solver.step_dt(
                DT_OUTER,
                state_0,
                state_1,
                control,
                apply_forces=viewer.apply_forces,
            )
        t += DT_OUTER
        step += 1

        if not use_fixed and step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        viewer.render(state_0, t)

    wall = time.perf_counter() - t_start
    fps = step / wall if wall > 0 else float("inf")
    print(
        f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps",
        flush=True,
    )


if __name__ == "__main__":
    main()
