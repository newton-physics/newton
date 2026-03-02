##
#
# Quadruped simulation using the CENIC adaptive-step MuJoCo solver.
#
# Each world adapts its own timestep independently based on integration
# error — no fixed dt, no PCIe transfers in the physics loop.
#
# A status grid is printed every LOG_EVERY_N_STEPS steps showing per-world
# sim time, current dt, RMS error, and accept/reject status.
#
##

import sys

import warp as wp
import newton
import newton.examples

# Number of quadruped simulations to run in parallel
num_worlds = 4

# How often to print the status grid (in simulation steps)
LOG_EVERY_N_STEPS = 10

# --- Build a single quadruped template ---
quadruped = newton.ModelBuilder()
newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(quadruped)

quadruped.default_body_armature = 0.01
quadruped.default_joint_cfg.armature = 0.01
quadruped.default_joint_cfg.target_ke = 2000.0
quadruped.default_joint_cfg.target_kd = 1.0
quadruped.default_shape_cfg.ke = 1.0e4
quadruped.default_shape_cfg.kd = 1.0e2
quadruped.default_shape_cfg.kf = 1.0e2
quadruped.default_shape_cfg.mu = 1.0

quadruped.add_urdf(
    newton.examples.get_asset("quadruped.urdf"),
    xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
    floating=True,
    enable_self_collisions=False,
    ignore_inertial_definitions=True,
)
quadruped.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]
quadruped.joint_target_pos[-12:] = quadruped.joint_q[-12:]

# --- Replicate across worlds and add ground ---
builder = newton.ModelBuilder()
builder.replicate(quadruped, num_worlds)
builder.add_ground_plane()
model = builder.finalize()

# --- Create the CENIC solver ---
solver = newton.solvers.SolverMuJoCoCENIC(
    model,
    tol=1e-3,
    dt_init=0.01,
    dt_min=2e-7,
    dt_max=0.02,
)

state_0 = model.state()
state_1 = model.state()
control = model.control()


# --- Wiping status grid ---
# Tracks how many lines were written last time so we can erase and redraw.
_grid_lines_written = 0


def print_status_grid(solver, step):
    global _grid_lines_written

    sim_times = solver.sim_time.numpy()
    dts       = solver.dt.numpy()
    errors    = solver.last_error.numpy()
    accepted  = solver.accepted.numpy()

    col = 16
    bar = "+" + ("-" * col + "+") * 5
    header = (
        f"{'world':>{col}}"
        f"{'sim_time (s)':>{col}}"
        f"{'dt (s)':>{col}}"
        f"{'RMS error':>{col}}"
        f"{'status':>{col}}"
    )

    lines = [
        f"  step {step}  (tol={solver._tol:.1e})",
        bar,
        header,
        bar,
    ]
    for i in range(len(sim_times)):
        status = "ok" if accepted[i] else "REJECT"
        lines.append(
            f"{f'world {i}':>{col}}"
            f"{sim_times[i]:>{col}.4f}"
            f"{dts[i]:>{col}.6f}"
            f"{errors[i]:>{col}.3e}"
            f"{status:>{col}}"
        )
    lines.append(bar)
    if not any(accepted):
        lines.append("  WARNING: all worlds rejected — tol may be too tight or dt_min too small")

    # Move cursor up over the previous grid and erase to end of screen
    if _grid_lines_written > 0:
        sys.stdout.write(f"\033[{_grid_lines_written}A\033[0J")

    output = "\n".join(lines)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()

    _grid_lines_written = len(lines)


# --- Start the visualizer ---
print(
    f"CENIC ready — {num_worlds} worlds  tol={solver._tol:.1e}  "
    f"dt_init={solver._dt.numpy()[0]:.4f}  dt_max={solver._dt_max:.4f}",
    flush=True,
)
viewer = newton.viewer.ViewerGL(headless=False)
viewer.set_model(model)

step = 0
t = 0.0

while viewer.is_running():
    with wp.ScopedTimer("simulate", active=False):
        state_0.clear_forces()
        viewer.apply_forces(state_0)
        solver.step(state_0, state_1, control, contacts=None)
        state_0, state_1 = state_1, state_0
        t += 0.01
        step += 1

    if step % LOG_EVERY_N_STEPS == 0:
        print_status_grid(solver, step)

    with wp.ScopedTimer("render", active=False):
        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()
