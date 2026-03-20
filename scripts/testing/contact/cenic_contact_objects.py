import argparse
import math
import sys
import time

import numpy as np
import warp as wp

import newton
import newton.solvers

DT_OUTER = 0.01    # 100 Hz control / render cadence [s]
LOG_EVERY = 250

SPHERE_RADIUS = 0.050
BOX_HALF      = 0.050
GRID_STEP     = 0.200
GRID_OFFSETS  = [-GRID_STEP, 0.0, GRID_STEP]
Z_SPHERES     = 1.00
Z_BOXES       = 1.25


TOL = 1e-3
DT_INNER_MIN = 1e-6


def build_template() -> newton.ModelBuilder:
    """Single-world template: 9 spheres + 9 tilted boxes."""
    template = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(template)

    cfg_obj = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005)

    for ox in GRID_OFFSETS:
        for oy in GRID_OFFSETS:
            b = template.add_body(
                xform=wp.transform(p=wp.vec3(ox, oy, Z_SPHERES), q=wp.quat_identity()),
            )
            template.add_shape_sphere(b, radius=SPHERE_RADIUS, cfg=cfg_obj)

    _box_angles = [
        ( 15,  0,  0), (-20, 10,  0), ( 35,  0, 15),
        (  0, 25, -10), ( 49,  0,  0), (-30, 20,  5),
        ( 10, -35,  0), (  0, 15, 40), (-15,  0, -25),
    ]
    for (ox, oy), (ax, ay, az) in zip(
        [(ox, oy) for ox in GRID_OFFSETS for oy in GRID_OFFSETS],
        _box_angles,
    ):
        rx, ry, rz = math.radians(ax), math.radians(ay), math.radians(az)
        cx, sx = math.cos(rx / 2), math.sin(rx / 2)
        cy, sy = math.cos(ry / 2), math.sin(ry / 2)
        cz, sz = math.cos(rz / 2), math.sin(rz / 2)
        q = wp.quat(
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )
        b = template.add_body(xform=wp.transform(p=wp.vec3(ox, oy, Z_BOXES), q=q))
        template.add_shape_box(b, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg_obj)

    return template


def build_model(n_worlds: int) -> newton.Model:
    """N replicated worlds + ground plane + invisible walls."""
    template = build_template()
    builder  = newton.ModelBuilder()
    builder.replicate(template, n_worlds)
    builder.add_ground_plane()

    cfg_wall   = newton.ModelBuilder.ShapeConfig(ke=1e4, kd=200, mu=0.3, margin=0.005, is_visible=False)
    half_inner = 0.350
    wt         = 0.025
    wh         = 0.750
    for px, py, hx, hy in [
        (-(half_inner + wt), 0.0,               wt,              half_inner + wt),
        (  half_inner + wt,  0.0,               wt,              half_inner + wt),
        (0.0,               -(half_inner + wt),  half_inner + wt, wt),
        (0.0,                 half_inner + wt,   half_inner + wt, wt),
    ]:
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(px, py, wh), q=wp.quat_identity()),
            hx=hx, hy=hy, hz=wh,
            cfg=cfg_wall,
        )
    return builder.finalize()


def make_solver(model: newton.Model, tol: float = TOL) -> newton.solvers.SolverMuJoCoCENIC:
    """CENIC solver with canonical contact-demo parameters."""
    return newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=tol,
        dt_inner_init=DT_OUTER,
        dt_inner_min=DT_INNER_MIN,
        dt_inner_max=DT_OUTER,
    )


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
        dts       = solver.dt.numpy()
        errors    = solver.last_error.numpy()
        accepted  = solver.accepted.numpy()

        col = 16
        bar = "+" + ("-" * col + "+") * 5
        hdr = (
            f"{'world':>{col}}"
            f"{'sim_time (s)':>{col}}"
            f"{'dt (s)':>{col}}"
            f"{'L2 error':>{col}}"
            f"{'status':>{col}}"
        )
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
    parser.add_argument("--num-steps",  type=int, default=0,  help="0 = run until closed")
    parser.add_argument("--headless",   action="store_true")
    args = parser.parse_args()

    model  = build_model(args.num_worlds)
    solver = make_solver(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    viewer = newton.viewer.ViewerGL(headless=args.headless)
    viewer.set_model(model)
    viewer.set_camera(
        pos=wp.vec3(1.97, -2.07, 1.07),
        pitch=-22.5,
        yaw=136.3,
    )
    print(
        f"CENIC contact demo: {args.num_worlds} world(s)  solver=SolverMuJoCoCENIC  "
        f"tol={solver._tol:.1e}  dt_inner_init={solver._dt.numpy()[0]:.4f}  "
        f"dt_inner_max={solver._dt_max:.4f}",
        flush=True,
    )
    step    = 0
    t       = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        state_0, state_1 = solver.step_dt(
            DT_OUTER, state_0, state_1, control,
            apply_forces=viewer.apply_forces,
        )
        t    += DT_OUTER
        step += 1

        if step % LOG_EVERY == 0:
            _print_status(solver, step)

        if args.num_steps > 0 and step >= args.num_steps:
            break

        viewer.render(state_0, t)

    wall = time.perf_counter() - t_start
    fps  = step / wall if wall > 0 else float("inf")
    print(
        f"\n{step} steps  {t:.3f} s sim  {wall:.2f} s wall  {fps:.1f} fps",
        flush=True,
    )


if __name__ == "__main__":
    main()
