# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Interactive dish-rack demo using the CENIC adaptive solver.

Drops a random mix of mugs, forks, spoons, bowls, plates, knives and cups onto
a peg rack.  Object layout per world is deterministic from ``--seed``.

Usage::

    uv run python -m scripts.demos.dish_rack [--num-worlds N] [--headless] [--seed S]
"""

import argparse
import sys
import time

import warp as wp

import newton
import newton.solvers
from scripts.scenes.dish_rack import DT_OUTER, LOG_EVERY, build_model_randomized, make_solver

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
        hdr = f"{'world':>{col}}{'sim_time (s)':>{col}}{'dt (s)':>{col}}{'Linf(q) err':>{col}}{'status':>{col}}"
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
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for object layout")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = build_model_randomized(args.num_worlds, seed=args.seed)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    solver = make_solver(model)
    print(
        f"Dish-rack demo: {args.num_worlds} world(s)  seed={args.seed}  "
        f"solver=SolverMuJoCoCENIC  tol={solver._tol:.1e}  "
        f"dt_inner_init={solver._dt.numpy()[0]:.4f}  dt_inner_max={solver._dt_max:.4f}",
        flush=True,
    )

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(
        pos=wp.vec3(0.85, -0.95, 0.70),
        pitch=-25.0,
        yaw=135.0,
    )

    step = 0
    t = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        state_0, state_1 = solver.step_dt(
            DT_OUTER,
            state_0,
            state_1,
            control,
            apply_forces=viewer.apply_forces,
        )
        t += DT_OUTER
        step += 1

        if step % LOG_EVERY == 0:
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
