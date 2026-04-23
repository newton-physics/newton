# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction-on-ramp grid test.

Builds a grid of static ramps (one per mu, theta cell) with a box on each,
then asserts each box matches the expected behavior given its critical
friction angle theta_crit = atan(mu).
"""

import math
import time
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# --- Scene configuration ---

# Rows (mu) and columns (theta). Kept below mu ~ 0.35 so AVBD's effective
# rigid friction is not saturating.
MUS = (0.10, 0.15, 0.20, 0.25, 0.30)
ANGLES_DEG = (5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0)

COL_PITCH = 2.5
ROW_PITCH = 5.0
GRID_Z = 2.0

RAMP_HX = 0.5
RAMP_HY = 1.5
RAMP_HZ = 0.05
BOX_HALF = 0.2

SIM_DT = 1.0 / 60.0
SIM_SUBSTEPS = 30
NUM_FRAMES = 60
VIEWER_FRAMES = 600

MIN_SLIDE = 0.15

# (margin_deg, v_rest_m_s, eps_pos_m). VBD thresholds widened to absorb
# the slight creep from AVBD's penalty friction on borderline cells.
_DEFAULT_THRESHOLDS = (3.0, 0.10, 0.05)
_VBD_THRESHOLDS = (5.0, 0.12, 0.10)

_ROW_COLORS = (
    (0.90, 0.30, 0.30),
    (0.90, 0.65, 0.20),
    (0.85, 0.85, 0.20),
    (0.30, 0.75, 0.35),
    (0.30, 0.55, 0.90),
)


def build_friction_grid(device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3

    box_ids = []
    for row, mu in enumerate(MUS):
        builder.default_shape_cfg.mu = mu
        builder.default_shape_cfg.color = _ROW_COLORS[row]

        row_box_ids = []
        for col, angle_deg in enumerate(ANGLES_DEG):
            angle = math.radians(angle_deg)
            ramp_quat = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
            ramp_center = wp.vec3(float(col * COL_PITCH), float(row * ROW_PITCH), float(GRID_Z))

            builder.add_shape_box(
                body=-1,
                xform=wp.transform(p=ramp_center, q=ramp_quat),
                hx=RAMP_HX,
                hy=RAMP_HY,
                hz=RAMP_HZ,
            )

            ramp_up = wp.quat_rotate(ramp_quat, wp.vec3(0.0, 0.0, 1.0))
            box_center = ramp_center + (RAMP_HZ + BOX_HALF) * ramp_up
            box_id = builder.add_body(
                xform=wp.transform(p=box_center, q=ramp_quat),
                label=f"box_r{row}_c{col}",
            )
            builder.add_shape_box(body=box_id, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
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


def assert_grid_behavior(test, initial_q, final_q, final_qd, box_ids, thresholds):
    margin_deg, v_rest, eps_pos = thresholds
    failures = []

    for row, mu in enumerate(MUS):
        crit_deg = math.degrees(math.atan(mu))
        for col, theta_deg in enumerate(ANGLES_DEG):
            bid = box_ids[row][col]
            v = float(np.linalg.norm(final_qd[bid, :3]))
            disp = float(np.linalg.norm(final_q[bid, :3] - initial_q[bid, :3]))
            tag = f"(mu={mu:.2f}, theta={theta_deg:.0f}deg, crit={crit_deg:.1f}deg)"

            if theta_deg < crit_deg - margin_deg:
                if v >= v_rest:
                    failures.append(f"{tag}: expected static but v={v:.3f} >= {v_rest}")
                if disp >= eps_pos:
                    failures.append(f"{tag}: expected static but disp={disp:.3f} >= {eps_pos}")
            elif theta_deg > crit_deg + margin_deg:
                if disp <= MIN_SLIDE:
                    failures.append(f"{tag}: expected sliding but disp={disp:.3f} <= {MIN_SLIDE}")

    if failures:
        test.fail("\n  ".join([f"{len(failures)} friction-ramp cell(s) failed:", *failures]))


def test_friction_ramp(test, device, solver_fn, thresholds):
    model, box_ids = build_friction_grid(device)

    solver = solver_fn(model)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts() if not isinstance(solver, newton.solvers.SolverMuJoCo) else None

    initial_q = state_0.body_q.numpy().copy()
    state_0, state_1 = simulate(solver, model, state_0, state_1, control, contacts, NUM_FRAMES)
    final_q = state_0.body_q.numpy()
    final_qd = state_0.body_qd.numpy()

    if np.any(np.isnan(final_q)) or np.any(np.isnan(final_qd)):
        test.fail("Simulation produced NaN values (numerical instability)")

    assert_grid_behavior(test, initial_q, final_q, final_qd, box_ids, thresholds)


# --- Solver matrix ---

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()

# Featherstone and SemiImplicit use viscous (kf) friction rather than Coulomb,
# so the critical-angle criterion does not apply; excluded here.
solvers = {
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, iterations=10),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, njmax=800, nconmax=500),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True),
    "vbd": lambda model: newton.solvers.SolverVBD(model, iterations=40, rigid_contact_k_start=1.0e5),
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
        solver_fn = solvers[solver_name]

        model, _ = build_friction_grid(device)
        solver = solver_fn(model)
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
                time.sleep(0.016)
        except KeyboardInterrupt:
            print("\nStopped by user.")


for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cpu and solver_name == "mujoco_warp":
            continue
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue
        thresholds = _VBD_THRESHOLDS if solver_name == "vbd" else _DEFAULT_THRESHOLDS
        add_function_test(
            TestRigidFrictionRamp,
            f"test_friction_ramp_{solver_name}",
            test_friction_ramp,
            devices=[device],
            check_output=False,
            solver_fn=solver_fn,
            thresholds=thresholds,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
