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

##
#
# N parallel double pendulums near upright, each stabilised by a different
# LQR gain matrix K.  Find the best K by ranking balanced fraction.
#
# All control is computed in a single Warp kernel — no CPU round-trips in
# the hot path.  LQR gain matrices (2×4 each) are pre-computed on CPU once
# using scipy.linalg.solve_continuous_are, then uploaded to the GPU and
# never touched again.
#
# Physical model: two uniform rods, identical mass m and length l, gravity g.
#   θ₁ — angle of link 1 from hanging-down  (0 = hanging, π = upright)
#   θ₂ — angle of link 2 relative to link 1 (0 = co-linear)
#
# All pendulums start at θ₁ = π − 0.2 rad (≈11° off upright) from rest.
# Each world applies u = −K · [η₁, η₂, θ̇₁, θ̇₂] where η₁, η₂ are
# angle errors wrapped to [−π, π].  Worlds that stay upright longest
# with lowest error have the best gains.
#
# Run (live viewer, 9 pendulums):
#   uv run scripts/cenic_double_pendulum_lqr.py
#
# Run (headless gain sweep, 10 000 pendulums):
#   uv run scripts/cenic_double_pendulum_lqr.py --headless \
#       --num-worlds 10000 --sim-steps 2000 --output-csv gains.csv
#
##

from __future__ import annotations

import argparse
import sys

import numpy as np
import warp as wp

wp.config.enable_backward = False

import newton
import newton.solvers

# ---------------------------------------------------------------------------
# Physical constants (uniform-rod model)
# ---------------------------------------------------------------------------

MASS = 1.0      # kg per link
LENGTH = 0.5    # m per link
GRAVITY = 9.81  # m/s²
ROD_HALF_WIDTH = 0.025  # m, visual half-extent for box shape

# Pivot height: enough clearance for both links fully extended downward
PIVOT_Z = LENGTH * 2 + 0.4  # 1.4 m

# ---------------------------------------------------------------------------
# Warp kernels — compiled once at import time
# ---------------------------------------------------------------------------


@wp.kernel
def _control_kernel(
    joint_q:      wp.array(dtype=wp.float32),
    joint_qd:     wp.array(dtype=wp.float32),
    # Per-world LQR gain matrix K [2×4], flat row-major (K[w*8 .. w*8+7])
    K:            wp.array(dtype=wp.float32),
    torque_limit: float,
    # Output: generalized forces → qfrc_applied in MuJoCo
    joint_f:      wp.array(dtype=wp.float32),
):
    """One thread per world — pure LQR stabilisation.

    Computes u = −K · η  where η = [η₁, η₂, θ̇₁, θ̇₂] is the linearised
    error state at the upright equilibrium.  Both angle errors are wrapped
    to [−π, π] via atan2 so the controller is correct regardless of
    cumulative joint wind-up.

    Angle convention:
        θ₁ = 0  → link 1 hanging straight down  (stable equilibrium)
        θ₁ = π  → link 1 pointing straight up   (upright target)
        θ₂ = 0  → link 2 co-linear with link 1
    """
    w = wp.tid()

    theta1  = joint_q[w * 2]
    theta2  = joint_q[w * 2 + 1]
    dtheta1 = joint_qd[w * 2]
    dtheta2 = joint_qd[w * 2 + 1]

    # Angle errors normalised to [−π, π]
    eta1 = wp.atan2(wp.sin(theta1 - wp.float32(wp.pi)),
                    wp.cos(theta1 - wp.float32(wp.pi)))
    eta2 = wp.atan2(wp.sin(theta2), wp.cos(theta2))

    b    = w * 8
    tau1 = -(K[b + 0] * eta1 + K[b + 1] * eta2
             + K[b + 2] * dtheta1 + K[b + 3] * dtheta2)
    tau2 = -(K[b + 4] * eta1 + K[b + 5] * eta2
             + K[b + 6] * dtheta1 + K[b + 7] * dtheta2)

    joint_f[w * 2]     = wp.clamp(tau1, -torque_limit, torque_limit)
    joint_f[w * 2 + 1] = wp.clamp(tau2, -torque_limit, torque_limit)


@wp.kernel
def _tally_balanced_kernel(
    joint_q:        wp.array(dtype=wp.float32),
    joint_qd:       wp.array(dtype=wp.float32),
    balanced_count: wp.array(dtype=wp.int32),
):
    """Increment per-world counter each step the world is near upright.

    Uses atan2-wrapped angle errors (same as _control_kernel) so theta2
    wind-up does not cause false negatives.
    """
    w    = wp.tid()
    eta1 = wp.atan2(wp.sin(joint_q[w * 2]     - wp.float32(wp.pi)),
                    wp.cos(joint_q[w * 2]     - wp.float32(wp.pi)))
    eta2 = wp.atan2(wp.sin(joint_q[w * 2 + 1]),
                    wp.cos(joint_q[w * 2 + 1]))
    near = (wp.abs(eta1) < wp.float32(0.5)
            and wp.abs(eta2) < wp.float32(0.5)
            and wp.abs(joint_qd[w * 2]) + wp.abs(joint_qd[w * 2 + 1]) < wp.float32(4.0))
    if near:
        wp.atomic_add(balanced_count, w, 1)


# ---------------------------------------------------------------------------
# Offline LQR gain computation (CPU, runs once before the simulation loop)
# ---------------------------------------------------------------------------


def compute_lqr_gains(num_worlds: int, seed: int = 0) -> np.ndarray:
    """Compute per-world LQR gain matrices K ∈ R^{2×4}.

    Linearises the uniform-rod double pendulum at the upright equilibrium
    (θ₁=π, θ₂=0) and solves the continuous-time algebraic Riccati equation
    for ``num_worlds`` different (Q, R) cost weights sampled in log-space.

    Returns:
        K_all: shape [num_worlds, 2, 4], float32.
    """
    from scipy.linalg import solve_continuous_are

    m, l, g = MASS, LENGTH, GRAVITY

    # ---- Mass matrix M₀ at θ₂ = 0 ----------------------------------------
    M11 = m * l**2 * (4.0 / 3 + 1.0 / 3 + 1.0)   # = 8/3 · m·l²
    M12 = m * l**2 * (1.0 / 3 + 0.5)              # = 5/6 · m·l²
    M22 = m * l**2 * (1.0 / 3)                    # = 1/3 · m·l²
    M0 = np.array([[M11, M12], [M12, M22]])

    # ---- Gravity Hessian at (π, 0) -----------------------------------------
    # Positive (upright is a PE maximum → destabilising spring)
    G11 = 2.0 * m * g * l
    G12 = 0.5 * m * g * l
    G22 = 0.5 * m * g * l
    K_g = np.array([[G11, G12], [G12, G22]])

    # ---- State-space: ẋ = Ax + Bu,  x = [η₁, η₂, θ̇₁, θ̇₂] -----------------
    M0_inv = np.linalg.inv(M0)
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = M0_inv @ K_g   # positive → unstable upright eigenvalues
    B = np.zeros((4, 2))
    B[2:, :] = M0_inv

    # ---- Sample (Q, R) and solve Riccati -----------------------------------
    rng = np.random.default_rng(seed)
    K_all = np.zeros((num_worlds, 2, 4), dtype=np.float32)

    q_pos = np.exp(rng.uniform(np.log(1.0),  np.log(1000.0), (num_worlds, 2)))
    q_vel = np.exp(rng.uniform(np.log(0.1),  np.log(100.0),  (num_worlds, 2)))
    r_act = np.exp(rng.uniform(np.log(0.01), np.log(10.0),   (num_worlds, 2)))

    K_default = np.array([[20.0, 10.0, 8.0, 4.0],
                           [10.0, 20.0, 4.0, 8.0]], dtype=np.float32)

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


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


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

    # Start near upright from rest — tests LQR stabilisation ability
    pendulum.joint_q[0] = float(np.pi) - 0.2   # ≈11° off upright

    scene = newton.ModelBuilder()
    scene.replicate(pendulum, num_worlds, spacing=(0.4, 0.0, 0.0))
    return scene.finalize()


# ---------------------------------------------------------------------------
# Terminal status grid (wipes and redraws in place)
# ---------------------------------------------------------------------------

_grid_lines_written = 0


def print_status_grid(solver, step, balanced_counts, total_steps, num_show=20):
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

    lines = [
        f"  step {step}  tol={solver._tol:.1e}  "
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CENIC double pendulum LQR gain sweep"
    )
    parser.add_argument("--num-worlds",   type=int,   default=9,     help="parallel pendulums")
    parser.add_argument("--headless",     action="store_true",        help="skip viewer, exit after --sim-steps")
    parser.add_argument("--sim-steps",    type=int,   default=2000,  help="steps in headless mode")
    parser.add_argument("--tol",          type=float, default=1e-3,  help="CENIC error tolerance")
    parser.add_argument("--dt-init",      type=float, default=0.005, help="initial timestep [s]")
    parser.add_argument("--dt-min",       type=float, default=1e-6,  help="minimum timestep [s]")
    parser.add_argument("--dt-max",       type=float, default=0.02,  help="maximum timestep [s]")
    parser.add_argument("--torque-limit", type=float, default=50.0,  help="per-joint torque clamp [N·m]")
    parser.add_argument("--seed",         type=int,   default=0,     help="RNG seed for gain sampling")
    parser.add_argument("--output-csv",   type=str,   default="",    help="save gain ranking to CSV")
    args = parser.parse_args()

    device = wp.get_device()

    # ---- LQR gains: CPU only, runs once ------------------------------------
    K_np = compute_lqr_gains(args.num_worlds, seed=args.seed)
    K_wp = wp.from_numpy(K_np.reshape(-1), dtype=wp.float32, device=device)

    # ---- Model + solver ----------------------------------------------------
    model  = build_model(args.num_worlds)
    solver = newton.solvers.SolverMuJoCoCENIC(
        model,
        tol=args.tol,
        dt_init=args.dt_init,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        solver="newton",
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    balanced_counts = wp.zeros(args.num_worlds, dtype=wp.int32, device=device)

    LOG_EVERY = max(1, (args.sim_steps if args.headless else 2000) // 20)

    print(
        f"CENIC LQR gain sweep — {args.num_worlds} worlds  "
        f"tol={args.tol:.1e}  dt_init={args.dt_init:.4f}  "
        f"start θ₁ = π − 0.2 rad",
        flush=True,
    )

    def _physics_step() -> None:
        wp.launch(
            _control_kernel,
            dim=args.num_worlds,
            inputs=[state_0.joint_q, state_0.joint_qd, K_wp, args.torque_limit],
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

    # ---- Headless mode -----------------------------------------------------
    if args.headless:
        for step in range(args.sim_steps):
            _physics_step()
            state_0, state_1 = state_1, state_0

            if step % LOG_EVERY == 0:
                print_status_grid(solver, step, balanced_counts, step + 1,
                                  num_show=min(20, args.num_worlds))

        counts_np = balanced_counts.numpy()
        times_np  = solver.sim_time.numpy()
        frac      = counts_np / max(args.sim_steps, 1)
        order     = np.argsort(frac)[::-1]
        K_flat    = K_wp.numpy().reshape(args.num_worlds, 2, 4)

        print("\n=== Gain Ranking (top 20 by balanced fraction) ===")
        print(f"{'Rank':>5}  {'World':>6}  {'Balanced%':>10}  {'sim_t':>8}  K[0,:4]")
        for rank in range(min(20, args.num_worlds)):
            i = order[rank]
            print(f"{rank+1:>5}  {i:>6}  {frac[i]:>10.1%}  {times_np[i]:>8.3f}  "
                  f"{K_flat[i, 0]}")

        if args.output_csv:
            import csv
            with open(args.output_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["world", "balanced_frac", "sim_time",
                     "K00", "K01", "K02", "K03",
                     "K10", "K11", "K12", "K13"]
                )
                for i in order:
                    writer.writerow(
                        [i, frac[i], times_np[i]]
                        + K_flat[i].flatten().tolist()
                    )
            print(f"Results saved to {args.output_csv}", flush=True)

    # ---- Live viewer mode --------------------------------------------------
    else:
        viewer = newton.viewer.ViewerGL(headless=False)
        viewer.set_model(model)

        step = 0
        t    = 0.0

        while viewer.is_running():
            _physics_step()
            state_0, state_1 = state_1, state_0
            t    += args.dt_init
            step += 1

            if step % LOG_EVERY == 0:
                print_status_grid(solver, step, balanced_counts, step,
                                  num_show=min(12, args.num_worlds))

            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.end_frame()


if __name__ == "__main__":
    main()
