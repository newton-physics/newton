# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Underactuated double pendulum — LQR gain sweep with CENIC adaptive stepping.
#
# Each world gets a different (Q, R) cost matrix sampled in log-space. Running many
# worlds in parallel lets you rank cost weights by how long each world stays balanced.
#
# Run (live viewer, 9 pendulums):
#   uv run scripts/control/cenic_double_pendulum_lqr.py
#
# Run (headless gain sweep, 10 000 pendulums):
#   uv run scripts/control/cenic_double_pendulum_lqr.py --headless \
#       --num-worlds 10000 --sim-steps 2000 --output-csv gains.csv

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.solvers

MASS           = 1.0    # kg per link
LENGTH         = 0.5    # m per link
GRAVITY        = 9.81   # m/s²
ROD_HALF_WIDTH = 0.025  # m, visual half-extent for box shape

PIVOT_Z = LENGTH * 2 + 0.4  # 1.4 m — clearance for both links fully extended downward

# LQR fires every CONTROL_DT seconds of simulation time regardless of adaptive step size.
CONTROL_DT = 0.002  # 2 ms


@wp.kernel
def _control_kernel(
    joint_q:        wp.array(dtype=wp.float32),   # [num_worlds * 2] joint positions θ₁, θ₂
    joint_qd:       wp.array(dtype=wp.float32),   # [num_worlds * 2] joint velocities θ̇₁, θ̇₂
    K:              wp.array(dtype=wp.float32),   # [num_worlds * 4] flat row-major LQR gains
    torque_limit:   float,                        # N·m
    sim_time:       wp.array(dtype=wp.float32),   # [num_worlds] per-world simulation time [s]
    last_ctrl_time: wp.array(dtype=wp.float32),   # [num_worlds] sim_time of last control update
    stored_tau:     wp.array(dtype=wp.float32),   # [num_worlds] zero-order-hold torque
    control_dt:     float,                        # controller period [s]
    joint_f:        wp.array(dtype=wp.float32),   # [num_worlds * 2] output generalized forces
):
    """Shoulder-only LQR with zero-order hold — one GPU thread per world.

    τ₁ = −K · [η₁, η₂, θ̇₁, θ̇₂], recomputed only when CONTROL_DT has elapsed.
    Elbow (joint 1) is always passive. Angles are wrapped to [−π, π] via atan2.
    K layout (flat, row-major): K[w*4 + 0..3] = [K_η₁, K_η₂, K_θ̇₁, K_θ̇₂].
    """
    w = wp.tid()

    theta1  = joint_q[w * 2]
    theta2  = joint_q[w * 2 + 1]
    dtheta1 = joint_qd[w * 2]
    dtheta2 = joint_qd[w * 2 + 1]

    eta1 = wp.atan2(wp.sin(theta1 - wp.float32(wp.pi)),
                    wp.cos(theta1 - wp.float32(wp.pi)))
    eta2 = wp.atan2(wp.sin(theta2), wp.cos(theta2))

    kg = w * 4

    if sim_time[w] - last_ctrl_time[w] >= control_dt:
        tau1 = -(K[kg + 0] * eta1 + K[kg + 1] * eta2
                 + K[kg + 2] * dtheta1 + K[kg + 3] * dtheta2)
        stored_tau[w]     = tau1
        last_ctrl_time[w] = sim_time[w]

    joint_f[w * 2]     = wp.clamp(stored_tau[w], -torque_limit, torque_limit)
    joint_f[w * 2 + 1] = wp.float32(0.0)


@wp.kernel
def _tally_balanced_kernel(
    joint_q:        wp.array(dtype=wp.float32),
    joint_qd:       wp.array(dtype=wp.float32),
    balanced_count: wp.array(dtype=wp.int32),
):
    """Increment per-world counter each physics step the pendulum is near upright.

    Thresholds: |η₁| < 0.5 rad, |η₂| < 0.5 rad, |θ̇₁| + |θ̇₂| < 4.0 rad/s.
    """
    w    = wp.tid()
    eta1 = wp.atan2(wp.sin(joint_q[w * 2]     - wp.float32(wp.pi)),
                    wp.cos(joint_q[w * 2]     - wp.float32(wp.pi)))
    eta2 = wp.atan2(wp.sin(joint_q[w * 2 + 1]),
                    wp.cos(joint_q[w * 2 + 1]))

    is_balanced = (wp.abs(eta1) < wp.float32(0.5)
                   and wp.abs(eta2) < wp.float32(0.5)
                   and wp.abs(joint_qd[w * 2]) + wp.abs(joint_qd[w * 2 + 1]) < wp.float32(4.0))
    if is_balanced:
        wp.atomic_add(balanced_count, w, 1)


def compute_lqr_gains(num_worlds: int, seed: int = 0) -> np.ndarray:
    """Compute per-world LQR gain matrices K ∈ ℝ^{1×4} for the shoulder actuator.

    Linearises the uniform-rod double pendulum at the upright equilibrium
    (θ₁=π, θ₂=0) and solves the continuous-time algebraic Riccati equation
    for ``num_worlds`` different (Q, R) cost weights sampled in log-space.

    Returns:
        K_all: shape [num_worlds, 1, 4], float32.
    """
    from scipy.linalg import solve_continuous_are

    m, l, g = MASS, LENGTH, GRAVITY

    M11 = m * l**2 * (4.0 / 3 + 1.0 / 3 + 1.0)
    M12 = m * l**2 * (1.0 / 3 + 0.5)
    M22 = m * l**2 * (1.0 / 3)
    M0 = np.array([[M11, M12], [M12, M22]])

    G11 = 2.0 * m * g * l
    G12 = 0.5 * m * g * l
    G22 = 0.5 * m * g * l
    K_g = np.array([[G11, G12], [G12, G22]])

    M0_inv = np.linalg.inv(M0)
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = M0_inv @ K_g

    # Shoulder-only actuation: B is 4×1, first column of M0_inv.
    B = np.zeros((4, 1))
    B[2:, :] = M0_inv[:, 0:1]

    rng = np.random.default_rng(seed)
    K_all = np.zeros((num_worlds, 1, 4), dtype=np.float32)

    q_pos = np.exp(rng.uniform(np.log(1.0),  np.log(1000.0), (num_worlds, 2)))
    q_vel = np.exp(rng.uniform(np.log(0.1),  np.log(100.0),  (num_worlds, 2)))
    r_act = np.exp(rng.uniform(np.log(0.01), np.log(10.0),   (num_worlds, 1)))

    K_default = np.array([[20.0, 10.0, 8.0, 4.0]], dtype=np.float32)

    print(f"Computing {num_worlds} LQR gain matrices ... ", end="", flush=True)
    for i in range(num_worlds):
        Q_i = np.diag(np.concatenate([q_pos[i], q_vel[i]]))
        R_i = np.diag(r_act[i])
        try:
            P = solve_continuous_are(A, B, Q_i, R_i)
            K_all[i] = (np.linalg.inv(R_i) @ B.T @ P).astype(np.float32)
        except Exception:
            K_all[i] = K_default

    print("done.", flush=True)
    return K_all


def build_model(num_worlds: int) -> newton.Model:
    """Build N parallel double pendulums, each replicated as a separate world."""
    m, l = MASS, LENGTH

    pendulum = newton.ModelBuilder()
    newton.solvers.SolverMuJoCoCENIC.register_custom_attributes(pendulum)

    I_bend = m * l**2 / 12.0
    inertia = wp.mat33(
        I_bend, 0.0,    0.0,
        0.0,    I_bend, 0.0,
        0.0,    0.0,    I_bend * 0.01,
    )

    hw = ROD_HALF_WIDTH

    link_0 = pendulum.add_link(mass=m, inertia=inertia)
    pendulum.add_shape_box(link_0, hx=hw, hy=hw, hz=l / 2)

    link_1 = pendulum.add_link(mass=m, inertia=inertia)
    pendulum.add_shape_box(link_1, hx=hw, hy=hw, hz=l / 2)

    j0 = pendulum.add_joint_revolute(
        parent=-1,
        child=link_0,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, PIVOT_Z), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, l / 2), wp.quat_identity()),
        target_ke=0.0,
        target_kd=0.0,
    )

    j1 = pendulum.add_joint_revolute(
        parent=link_0,
        child=link_1,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -l / 2), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, l / 2), wp.quat_identity()),
        target_ke=0.0,
        target_kd=0.0,
    )

    pendulum.add_articulation([j0, j1], label="pendulum")

    pendulum.joint_q[0] = float(np.pi) - 0.2   # ≈11° off upright

    scene = newton.ModelBuilder()
    scene.replicate(pendulum, num_worlds, spacing=(0.4, 0.0, 0.0))
    return scene.finalize()


_grid_lines_written = 0


def print_status_grid(solver, step, balanced_counts, total_steps, viewer_t=None, num_show=20):
    global _grid_lines_written

    sim_times = solver.sim_time.numpy()
    dts       = solver.dt.numpy()
    errors    = solver.last_error.numpy()
    counts    = balanced_counts.numpy()

    n    = min(num_show, len(sim_times))
    frac = counts / max(total_steps, 1)
    order = np.argsort(frac)[::-1]

    col = 14
    bar = "+" + ("-" * col + "+") * 5
    header = (
        f"{'world':>{col}}"
        f"{'sim_t (s)':>{col}}"
        f"{'dt (s)':>{col}}"
        f"{'RMS err':>{col}}"
        f"{'balanced%':>{col}}"
    )

    viewer_str = f"  viewer_t={viewer_t:.3f}s" if viewer_t is not None else ""
    lines = [
        f"  step {step}  tol={solver._tol:.1e}{viewer_str}  "
        f"showing top {n} of {len(sim_times)} worlds",
        bar, header, bar,
    ]

    for i in order[:n]:
        lines.append(
            f"{f'world {i}':>{col}}"
            f"{sim_times[i]:>{col}.3f}"
            f"{dts[i]:>{col}.6f}"
            f"{errors[i]:>{col}.2e}"
            f"{frac[i]:>{col}.1%}"
        )
    lines.append(bar)

    if _grid_lines_written > 0:
        sys.stdout.write(f"\033[{_grid_lines_written}A\033[0J")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
    _grid_lines_written = len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CENIC underactuated double pendulum LQR gain sweep"
    )
    parser.add_argument("--num-worlds", type=int, default=9,    help="parallel pendulums")
    parser.add_argument("--headless",   action="store_true",   help="skip viewer, exit after --num-steps")
    parser.add_argument("--num-steps",  type=int, default=2000, help="steps in headless mode")
    args = parser.parse_args()

    device = wp.get_device()

    K_np = compute_lqr_gains(args.num_worlds)
    K_wp = wp.from_numpy(K_np.reshape(-1), dtype=wp.float32, device=device)

    # last_ctrl_time starts at −CONTROL_DT so the first physics step triggers a control update.
    last_ctrl_time = wp.array(
        np.full(args.num_worlds, -CONTROL_DT, dtype=np.float32), device=device
    )
    stored_tau = wp.zeros(args.num_worlds, dtype=wp.float32, device=device)

    model  = build_model(args.num_worlds)
    solver = newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=1e-3,
        dt_init=0.005,
        dt_min=1e-6,
        dt_max=0.02,
        solver="newton",
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    balanced_counts = wp.zeros(args.num_worlds, dtype=wp.int32, device=device)

    LOG_EVERY = max(1, (args.num_steps if args.headless else 2000) // 20)

    print(
        f"CENIC LQR gain sweep — {args.num_worlds} worlds  "
        f"tol=1e-3  dt_init=0.005  "
        f"start θ₁ = π − 0.2 rad  control_dt={CONTROL_DT*1000:.0f} ms",
        flush=True,
    )

    def _physics_step() -> None:
        wp.launch(
            _control_kernel,
            dim=args.num_worlds,
            inputs=[
                state_0.joint_q, state_0.joint_qd, K_wp, 50.0,
                solver.sim_time, last_ctrl_time, stored_tau, CONTROL_DT,
            ],
            outputs=[control.joint_f],
            device=device,
        )

        wp.launch(
            _tally_balanced_kernel,
            dim=args.num_worlds,
            inputs=[state_0.joint_q, state_0.joint_qd],
            outputs=[balanced_counts],
            device=device,
        )

        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts=None)

    if args.headless:
        for step in range(args.num_steps):
            _physics_step()
            state_0, state_1 = state_1, state_0

            if step % LOG_EVERY == 0:
                print_status_grid(solver, step, balanced_counts, step + 1,
                                  num_show=min(20, args.num_worlds))

        counts_np = balanced_counts.numpy()
        times_np  = solver.sim_time.numpy()
        frac      = counts_np / max(args.num_steps, 1)
        order     = np.argsort(frac)[::-1]
        K_flat    = K_wp.numpy().reshape(args.num_worlds, 1, 4)

        print("\n=== Gain Ranking (top 20 by balanced fraction) ===")
        print(f"{'Rank':>5}  {'World':>6}  {'Balanced%':>10}  {'sim_t':>8}  K[0,:4]")
        for rank in range(min(20, args.num_worlds)):
            i = order[rank]
            print(f"{rank+1:>5}  {i:>6}  {frac[i]:>10.1%}  {times_np[i]:>8.3f}  "
                  f"{K_flat[i, 0]}")

    else:
        viewer = newton.viewer.ViewerGL(headless=False)
        viewer.set_model(model)

        center_x = 0.4 * (args.num_worlds - 1) / 2.0
        cam_dist = max(3.0, 0.5 * args.num_worlds)
        viewer.set_camera(
            pos=wp.vec3(center_x, -cam_dist, PIVOT_Z),
            pitch=0.0,
            yaw=90.0,
        )

        step             = 0
        t                = 0.0
        next_render_time = np.zeros(args.num_worlds, dtype=np.float32)

        while viewer.is_running():
            while True:
                wp.launch(
                    _control_kernel,
                    dim=args.num_worlds,
                    inputs=[
                        state_0.joint_q, state_0.joint_qd, K_wp, 50.0,
                        solver.sim_time, last_ctrl_time, stored_tau, CONTROL_DT,
                    ],
                    outputs=[control.joint_f],
                    device=device,
                )
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, contacts=None)
                state_0, state_1 = state_1, state_0
                if np.all(solver.sim_time.numpy() >= next_render_time):
                    break

            wp.launch(
                _tally_balanced_kernel,
                dim=args.num_worlds,
                inputs=[state_0.joint_q, state_0.joint_qd],
                outputs=[balanced_counts],
                device=device,
            )
            next_render_time += CONTROL_DT
            t    += CONTROL_DT
            step += 1

            if step % LOG_EVERY == 0:
                print_status_grid(solver, step, balanced_counts, step,
                                  viewer_t=t,
                                  num_show=min(12, args.num_worlds))

            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()


if __name__ == "__main__":
    main()
