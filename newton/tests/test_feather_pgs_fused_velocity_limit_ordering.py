# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordering of the fused joint velocity-limit clamp against contact rows.

A joint velocity limit must have the last word in every solver iteration. The
dedicated velocity-limit rows run after the contact rows for that reason, and the
fused clamp of driven DOFs (``fuse_joint_velocity_limits``) has to run there as well:
clamped inside the drive-row visit instead, a contact impulse later in the same
iteration would push the driven DOF past its limit with nothing left to clamp it.

The scene is a stiff driven horizontal arm with a low velocity limit and a heavy
free box slamming into its tip. The impact drives the joint far past the limit,
which must still hold at the end of every step.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

QDOT_MAX = 0.5
# Slack on the limit for an under-converged sweep; the final pass leaves the DOF at the limit.
LIMIT_TOL = 1.05


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip, without gravity or friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    # Arm spanning x in [0, 0.8] at z = 0.5, hinged at its left end about y.
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_arm])
    # Heavy box, bottom face slightly penetrating the arm top near the tip.
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    return builder.finalize(device=device)


def _max_arm_speed_over_impact(device, *, enable_joint_velocity_limits: bool, fuse: bool) -> tuple[float, int]:
    model = _build_arm_and_box_model(device)
    n = model.joint_dof_count
    target_ke = np.zeros(n, dtype=np.float32)
    target_kd = np.zeros(n, dtype=np.float32)
    vel_limit = np.full(n, np.inf, dtype=np.float32)
    target_ke[0] = 200.0
    target_kd[0] = 5.0
    vel_limit[0] = QDOT_MAX
    model.joint_target_ke.assign(target_ke)
    model.joint_target_kd.assign(target_kd)
    model.joint_velocity_limit.assign(vel_limit)
    # Deliberately under-converged: ordering only matters when the sweep does not converge.
    # A clamp inside the drive-row visit lets this impact reach about 1.5x the limit at 8
    # iterations; dedicated rows and the end-of-iteration clamp hold it.
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        drive_mode="physx_pgs",
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        fuse_joint_velocity_limits=fuse,
        pgs_iterations=8,
        dense_max_constraints=32,
        mf_max_constraints=32,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0  # box linear z: slam into the arm tip
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    max_arm_speed = 0.0
    for _ in range(20):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
    return max_arm_speed, int(solver.fuse_joint_velocity_limits)


def test_dedicated_rows_hold_limit_through_impact(test, device):
    """Hold a PGS-driven joint's velocity limit through the impact with dedicated velocity-limit rows."""
    unlimited, _ = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=False, fuse=False)
    limited, fused = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=True, fuse=False)
    test.assertFalse(fused)
    test.assertGreater(unlimited, 2.0 * QDOT_MAX, "the impact must exceed the limit when it is not enforced")
    test.assertLessEqual(limited, QDOT_MAX * LIMIT_TOL)


def test_fused_clamp_holds_limit_through_impact(test, device):
    """Hold the same limit with the fused end-of-iteration clamp that replaces the rows."""
    speed, fused = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=True, fuse=True)
    test.assertTrue(fused)
    test.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)


class TestFeatherPGSFusedVelocityLimitOrdering(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_dedicated_rows_hold_limit_through_impact", test_dedicated_rows_hold_limit_through_impact),
    ("test_fused_clamp_holds_limit_through_impact", test_fused_clamp_holds_limit_through_impact),
):
    add_function_test(TestFeatherPGSFusedVelocityLimitOrdering, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
