# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical Coulomb-friction tests.

Two complementary tests share this file as the canonical friction benchmark:

  * ``test_friction_ramp`` — (mu, theta) grid of static ramps with a box on
    each. Each cell's expected behavior follows from its critical friction
    angle theta_crit = atan(mu):

      * theta < crit - margin: box at rest (~zero velocity, no displacement).
      * theta > crit + margin: box slides at least ``min_slide`` metres.

  * ``test_friction_stopping_distance`` — sliding boxes on flat ground decelerate
    under kinetic Coulomb friction and stop at d = v0^2 / (2 mu g). Provides
    the precise kinetic-friction oracle that the ramp grid does not exercise.
"""

import math
import time
import unittest
from typing import NamedTuple

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# --- Scene / sim configuration ---

GRAVITY = -9.81
UP_AXIS = newton.Axis.Z

COL_PITCH = 2.5
ROW_PITCH = 6.0
GRID_Z = 2.0

RAMP_HX = 0.5
RAMP_HY = 2.5  # long enough that fast cells (low mu, high theta) stay on the ramp through measurement
RAMP_HZ = 0.05
# Flat slab so the box doesn't tip over on steep slopes — tipping needs tan(theta) > BOX_HY / BOX_HZ.
BOX_HX = 0.2
BOX_HY = 0.2
BOX_HZ = 0.05
BOX_GAP = 0.001  # initial offset above the ramp surface to avoid penalty pop-out

SIM_DT = 1.0 / 60.0
SIM_SUBSTEPS = 30
SETTLE_FRAMES = 30  # 0.5 s, lets the contact-stiffness transient decay
MEASURE_FRAMES = 15  # 0.25 s window for sliding cells to accumulate a measurable displacement
VIEWER_FRAMES = 600

# Sweeps. Non-VBD solvers cover a wide mu range; VBD's penalty friction
# saturates above mu ~ 0.30, so it gets a narrower sweep with looser
# thresholds (see _VBD_THRESHOLDS). Angles are capped at 40 deg because
# constraint-solver friction enforcement on a steep slope from rest is
# noisy near 50 deg. mu=1.00 therefore exercises only the static side.
# Quantitative kinetic-friction validation lives in
# test_friction_stopping_distance, which is unaffected by these caps.
_DEFAULT_MUS = (0.10, 0.30, 0.50, 0.70, 1.00)
_DEFAULT_ANGLES_DEG = (3.0, 10.0, 20.0, 30.0, 40.0)
_VBD_MUS = (0.10, 0.15, 0.20, 0.25, 0.30)
_VBD_ANGLES_DEG = (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)


# Thresholds. Below-crit cells: |v| < v_rest AND post-settle disp < eps_pos.
# Above-crit cells: post-settle disp >= min_slide (a sanity bound — the precise
# kinetic-friction oracle is in test_friction_stopping_distance). VBD gets a
# wider deadband and looser static thresholds because AVBD's penalty friction
# is fuzzy near the static/kinetic boundary.
class _Thresholds(NamedTuple):
    margin_deg: float
    v_rest: float
    eps_pos: float
    min_slide: float


_DEFAULT_THRESHOLDS = _Thresholds(margin_deg=2.0, v_rest=0.10, eps_pos=0.02, min_slide=0.02)
# VBD's min_slide is loose: AVBD penalty-friction creeps borderline cells a few mm
# in 0.25 s rather than sliding fully. The tighter kinetic-friction oracle on these
# solvers is in test_friction_stopping_distance.
_VBD_THRESHOLDS = _Thresholds(margin_deg=5.0, v_rest=0.12, eps_pos=0.10, min_slide=0.005)

# --- Stopping-distance config ---

# Each box slides on its own static ground patch with matching mu (Newton averages
# mu at contact, so a shared ground would dilute the per-box mu).
STOPPING_V0 = 2.0
STOPPING_MUS = (0.20, 0.40, 0.70)
STOPPING_BOX_HALF = 0.25
STOPPING_PATCH_HX = 5.0  # comfortably exceeds d_stop(mu_min) ~ 1.02 m
STOPPING_PATCH_HY = 0.6
STOPPING_PATCH_HZ = 0.05
STOPPING_BOX_PITCH_Y = 5.0
STOPPING_V_FINAL_MAX = 0.05  # m/s — sanity bound: box must have come to rest

_ROW_COLORS = (
    (0.90, 0.30, 0.30),
    (0.90, 0.65, 0.20),
    (0.85, 0.85, 0.20),
    (0.30, 0.75, 0.35),
    (0.30, 0.55, 0.90),
)


def build_friction_grid(device, mus, angles_deg):
    builder = newton.ModelBuilder(gravity=GRAVITY, up_axis=UP_AXIS)

    box_ids = []
    for row, mu in enumerate(mus):
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = mu
        cfg.ke = 1.0e5
        cfg.kd = 1.0e3
        cfg.kf = 0.0  # validate Coulomb friction only — disable viscous component
        cfg.color = _ROW_COLORS[row % len(_ROW_COLORS)]

        row_box_ids = []
        for col, angle_deg in enumerate(angles_deg):
            angle = math.radians(angle_deg)
            ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
            ramp_center = wp.vec3(float(col * COL_PITCH), float(row * ROW_PITCH), float(GRID_Z))

            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=ramp_center, q=ramp_quat),
                hx=RAMP_HX,
                hy=RAMP_HY,
                hz=RAMP_HZ,
                cfg=cfg,
            )

            ramp_up = wp.quat_rotate(ramp_quat, wp.vec3(0.0, 0.0, 1.0))
            box_center = ramp_center + (RAMP_HZ + BOX_HZ + BOX_GAP) * ramp_up
            box_id = builder.add_body(
                xform=wp.transform(p=box_center, q=ramp_quat),
                label=f"box_r{row}_c{col}",
            )
            builder.add_shape_box(body=box_id, hx=BOX_HX, hy=BOX_HY, hz=BOX_HZ, cfg=cfg)
            row_box_ids.append(box_id)

        box_ids.append(row_box_ids)

    builder.color()  # required for VBD
    return builder.finalize(device=device), box_ids


def simulate(solver, model, state_0, state_1, control, contacts, num_frames):
    dt_sub = SIM_DT / SIM_SUBSTEPS
    for _ in range(num_frames):
        for _ in range(SIM_SUBSTEPS):
            state_0.clear_forces()
            if contacts is not None:
                model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt_sub)
            state_0, state_1 = state_1, state_0
    return state_0, state_1


def assert_grid_behavior(test, settle_q, final_q, final_qd, mus, angles_deg, box_ids, thresholds):
    failures = []

    for row, mu in enumerate(mus):
        crit_deg = math.degrees(math.atan(mu))
        for col, theta_deg in enumerate(angles_deg):
            bid = box_ids[row][col]
            v_final = float(np.linalg.norm(final_qd[bid, :3]))
            disp = float(np.linalg.norm(final_q[bid, :3] - settle_q[bid, :3]))
            tag = f"(mu={mu:.2f}, theta={theta_deg:.1f}deg, crit={crit_deg:.1f}deg)"

            if theta_deg < crit_deg - thresholds.margin_deg:
                if v_final >= thresholds.v_rest:
                    failures.append(f"{tag}: expected static but |v|={v_final:.4f} >= {thresholds.v_rest}")
                if disp >= thresholds.eps_pos:
                    failures.append(f"{tag}: expected static but disp={disp:.4f} >= {thresholds.eps_pos}")
            elif theta_deg > crit_deg + thresholds.margin_deg:
                if disp < thresholds.min_slide:
                    failures.append(f"{tag}: expected sliding but disp={disp:.4f} < {thresholds.min_slide}")

    if failures:
        test.fail("\n  ".join([f"{len(failures)} friction-ramp cell(s) failed:", *failures]))


def test_friction_ramp(test, device, solver_fn, mus, angles_deg, thresholds):
    model, box_ids = build_friction_grid(device, mus, angles_deg)

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts() if not isinstance(solver, newton.solvers.SolverMuJoCo) else None

    state_0, state_1 = simulate(solver, model, state_0, state_1, control, contacts, SETTLE_FRAMES)
    settle_q = state_0.body_q.numpy().copy()

    state_0, state_1 = simulate(solver, model, state_0, state_1, control, contacts, MEASURE_FRAMES)
    final_q = state_0.body_q.numpy()
    final_qd = state_0.body_qd.numpy()

    if np.any(np.isnan(final_q)) or np.any(np.isnan(final_qd)):
        test.fail("Simulation produced NaN values (numerical instability)")

    assert_grid_behavior(test, settle_q, final_q, final_qd, mus, angles_deg, box_ids, thresholds)


def build_stopping_distance_scene(device):
    """Boxes on per-box static ground patches with matching mu.

    Newton averages mu across the two contact shapes, so a shared ground would
    give effective mu = (mu_box + mu_patch) / 2. Per-box patches keep the
    effective mu equal to the per-box value.
    """
    builder = newton.ModelBuilder(gravity=GRAVITY, up_axis=UP_AXIS)

    box_ids = []
    for i, mu in enumerate(STOPPING_MUS):
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = mu
        cfg.ke = 1.0e5
        cfg.kd = 1.0e3
        cfg.kf = 0.0
        cfg.color = _ROW_COLORS[i % len(_ROW_COLORS)]

        patch_y = float(i * STOPPING_BOX_PITCH_Y)
        # Shift the patch forward so the box (starting at x=0) has the full
        # patch length ahead of it for stopping room.
        patch_x = STOPPING_PATCH_HX - 0.5
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(p=wp.vec3(patch_x, patch_y, -STOPPING_PATCH_HZ), q=wp.quat_identity()),
            hx=STOPPING_PATCH_HX,
            hy=STOPPING_PATCH_HY,
            hz=STOPPING_PATCH_HZ,
            cfg=cfg,
        )
        box_id = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, patch_y, STOPPING_BOX_HALF + BOX_GAP), q=wp.quat_identity()),
            label=f"box_mu{mu:.2f}",
        )
        builder.add_shape_box(
            body=box_id,
            hx=STOPPING_BOX_HALF,
            hy=STOPPING_BOX_HALF,
            hz=STOPPING_BOX_HALF,
            cfg=cfg,
        )
        box_ids.append(box_id)

    builder.color()  # required for VBD
    return builder.finalize(device=device), box_ids


def test_friction_stopping_distance(test, device, solver_fn, rel_tol):
    """Kinetic-friction oracle: a sliding box stops at d = v0^2 / (2 mu g).

    Three boxes at mu in STOPPING_MUS each start with v0 along world-X on a
    matching ground patch. Run for 1.5 * t_stop(mu_min) so every box has come
    to rest, then compare measured stopping distance against the analytical
    value with a per-solver relative tolerance.
    """
    model, box_ids = build_stopping_distance_scene(device)
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    is_mujoco = isinstance(solver, newton.solvers.SolverMuJoCo)
    contacts = model.contacts() if not is_mujoco else None

    initial_q = state_0.body_q.numpy().copy()

    qd = state_0.body_qd.numpy()
    for bid in box_ids:
        qd[bid, 0] = STOPPING_V0  # body_qd: [v_lin (0:3), omega (3:6)]
    state_0.body_qd.assign(qd)

    # MuJoCo integrates in generalized coordinates; sync joint state from body state
    # so the imposed body_qd takes effect.
    if is_mujoco:
        q_ik = wp.zeros_like(model.joint_q, device=device)
        qd_ik = wp.zeros_like(model.joint_qd, device=device)
        newton.eval_ik(model, state_0, q_ik, qd_ik)
        state_0.joint_q.assign(q_ik)
        state_0.joint_qd.assign(qd_ik)

    g = abs(GRAVITY)
    t_stop_max = STOPPING_V0 / (min(STOPPING_MUS) * g)
    num_frames = int(math.ceil(1.5 * t_stop_max / SIM_DT))

    state_0, state_1 = simulate(solver, model, state_0, state_1, control, contacts, num_frames)

    final_q = state_0.body_q.numpy()
    final_qd = state_0.body_qd.numpy()

    if np.any(np.isnan(final_q)) or np.any(np.isnan(final_qd)):
        test.fail("Simulation produced NaN values (numerical instability)")

    failures = []
    for bid, mu in zip(box_ids, STOPPING_MUS, strict=True):
        d_expected = STOPPING_V0 * STOPPING_V0 / (2.0 * mu * g)
        dx = final_q[bid, 0] - initial_q[bid, 0]
        dy = final_q[bid, 1] - initial_q[bid, 1]
        d_measured = float(math.sqrt(dx * dx + dy * dy))
        rel_err = (d_measured - d_expected) / d_expected
        v_final = float(np.linalg.norm(final_qd[bid, :3]))
        tag = f"(mu={mu:.2f})"
        if abs(rel_err) > rel_tol:
            failures.append(
                f"{tag}: d_measured={d_measured:.4f} m vs d_expected={d_expected:.4f} m "
                f"(rel_err={rel_err:+.2%}, tol={rel_tol:.0%})"
            )
        if v_final >= STOPPING_V_FINAL_MAX:
            failures.append(f"{tag}: |v_final|={v_final:.4f} m/s >= {STOPPING_V_FINAL_MAX} (box did not stop)")

    if failures:
        test.fail("\n  ".join([f"{len(failures)} stopping-distance failure(s):", *failures]))


# --- Solver matrix ---

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

# Featherstone and SemiImplicit use viscous (kf) friction rather than Coulomb,
# so the critical-angle criterion does not apply; excluded here.
# stopping_distance_rel_tol: per-solver tolerance on d_measured/d_expected. Coulomb-cone
# solvers (XPBD, MuJoCo) hit ~0.05-0.2% in practice; VBD systematically overshoots
# 13-18% because AVBD's low-velocity friction regularization lets the box coast past
# the Coulomb stopping point.
_SOLVERS = {
    "xpbd": {
        "factory": lambda model: newton.solvers.SolverXPBD(model, iterations=10),
        "mus": _DEFAULT_MUS,
        "angles_deg": _DEFAULT_ANGLES_DEG,
        "thresholds": _DEFAULT_THRESHOLDS,
        "stopping_distance_rel_tol": 0.01,
    },
    "mujoco_warp": {
        "factory": lambda model: newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            njmax=800,
            nconmax=500,
            cone="elliptic",
            impratio=10.0,
            iterations=200,
            ls_iterations=100,
        ),
        "mus": _DEFAULT_MUS,
        "angles_deg": _DEFAULT_ANGLES_DEG,
        "thresholds": _DEFAULT_THRESHOLDS,
        "stopping_distance_rel_tol": 0.01,
    },
    "mujoco_cpu": {
        "factory": lambda model: newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=True,
            cone="elliptic",
            impratio=10.0,
            iterations=200,
            ls_iterations=100,
        ),
        "mus": _DEFAULT_MUS,
        "angles_deg": _DEFAULT_ANGLES_DEG,
        "thresholds": _DEFAULT_THRESHOLDS,
        "stopping_distance_rel_tol": 0.01,
    },
    "vbd": {
        "factory": lambda model: newton.solvers.SolverVBD(model, iterations=40, rigid_contact_k_start=1.0e5),
        "mus": _VBD_MUS,
        "angles_deg": _VBD_ANGLES_DEG,
        "thresholds": _VBD_THRESHOLDS,
        "stopping_distance_rel_tol": 0.25,
    },
}


class TestRigidFrictionRamp(unittest.TestCase):
    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_friction_grid_xpbd(self):
        self._run_viewer("xpbd")

    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_friction_grid_vbd(self):
        self._run_viewer("vbd")

    @unittest.skip("Visual debugging - run manually to view simulation")
    def test_view_friction_grid_mujoco_warp(self):
        self._run_viewer("mujoco_warp")

    def _run_viewer(self, solver_name):
        device = wp.get_device("cuda:0")
        cfg = _SOLVERS[solver_name]

        model, _ = build_friction_grid(device, cfg["mus"], cfg["angles_deg"])
        solver = cfg["factory"](model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts() if not isinstance(solver, newton.solvers.SolverMuJoCo) else None

        try:
            viewer = newton.viewer.ViewerGL()
            viewer.set_model(model)
            viewer.set_camera(pos=wp.vec3(6.0, -14.0, 10.0), pitch=-22.0, yaw=90.0)
        except Exception as e:
            self.skipTest(f"ViewerGL not available: {e}")
            return

        print(f"\nFriction-ramp grid with '{solver_name}' solver for {VIEWER_FRAMES} frames...")
        print("Close the viewer window or press Ctrl+C to stop.")

        sim_time = 0.0
        try:
            for _ in range(VIEWER_FRAMES):
                viewer.begin_frame(sim_time)
                viewer.log_state(state_0)
                if contacts is not None:
                    viewer.log_contacts(contacts, state_0)
                viewer.end_frame()

                state_0, state_1 = simulate(solver, model, state_0, state_1, control, contacts, 1)
                sim_time += SIM_DT
                time.sleep(SIM_DT)
        except KeyboardInterrupt:
            print("\nStopped by user.")


for device in devices:
    for solver_name, cfg in _SOLVERS.items():
        if device.is_cpu and solver_name == "mujoco_warp":
            continue
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        add_function_test(
            TestRigidFrictionRamp,
            f"test_friction_ramp_{solver_name}",
            test_friction_ramp,
            devices=[device],
            check_output=False,
            solver_fn=cfg["factory"],
            mus=cfg["mus"],
            angles_deg=cfg["angles_deg"],
            thresholds=cfg["thresholds"],
        )
        add_function_test(
            TestRigidFrictionRamp,
            f"test_friction_stopping_distance_{solver_name}",
            test_friction_stopping_distance,
            devices=[device],
            check_output=False,
            solver_fn=cfg["factory"],
            rel_tol=cfg["stopping_distance_rel_tol"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
