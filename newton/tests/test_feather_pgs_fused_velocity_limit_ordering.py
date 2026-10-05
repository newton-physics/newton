# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ordering of the fused joint velocity-limit clamp vs. contact rows.

A joint velocity limit must get the last word of every solver iteration:
PhysX solves joint max-velocity after link contacts (see
PxSceneFlag::eSOLVE_ARTICULATION_CONTACT_LAST and the
eAFTER_STATIC_CONSTRAINTS path in DyFeatherstoneArticulation.cpp), and the
dedicated velocity-limit rows run in the final phase of each FPGS iteration
for the same reason. If the fused clamp instead runs inside the drive-row
visit (before contacts), a contact impulse landing later in the iteration
re-accelerates the driven DOF past its limit and nothing re-clamps it.

The scene makes that failure concrete: a stiff PD-driven horizontal arm with
a low joint velocity limit, and a heavy free box slamming into the arm's tip.
The impact pushes the arm's joint speed far past the limit; the limit must
still hold at the end of every step, exactly as it does with the dedicated
velocity-limit rows the fused clamp replaces.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

QDOT_MAX = 0.5
# GS-convergence slack on the limit; the final clamp leaves the DOF at the
# limit, later float noise stays well under 5%.
LIMIT_TOL = 1.05


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip.

    Gravity is zero and the box carries a large downward velocity, so the
    contact impulse into the arm tip is the only thing fighting the velocity
    limit — fully deterministic.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
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


class TestFeatherPGSFusedVelocityLimitOrdering(unittest.TestCase):
    def _max_arm_speed_over_impact(self, fuse: bool, articulated_contact_response: str) -> float:
        model = _build_arm_and_box_model(device="cuda:0")

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

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            drive_mode="physx_pgs",
            articulated_contact_response=articulated_contact_response,
            enable_joint_velocity_limits=True,
            fuse_joint_velocity_limits=fuse,
            enable_contact_friction=False,
            # Deliberately under-converged (franka-class training configs run
            # 8): ordering only matters when the sweep does NOT converge — at
            # convergence the last contact visits apply ~zero delta and any
            # clamp placement passes. A drive-visit clamp lets this impact
            # reach 1.53x the limit at 8 iterations (7.1x at 1); dedicated
            # rows and the end-of-iteration clamp hold the limit exactly at
            # every iteration count.
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

        collision_pipeline = newton.CollisionPipeline(model)

        contacts = collision_pipeline.contacts()
        max_arm_speed = 0.0
        for _ in range(20):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
        wp.synchronize()
        return max_arm_speed

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free FPGS requires CUDA")
    def test_dedicated_rows_hold_limit_through_impact(self):
        # Reference behavior: dedicated velocity-limit rows run in the final
        # phase of every iteration and hold the limit through the impact.
        speed = self._max_arm_speed_over_impact(fuse=False, articulated_contact_response="immediate")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free FPGS requires CUDA")
    def test_fused_clamp_holds_limit_through_impact(self):
        # The fused clamp must match the dedicated rows it replaces: the
        # contact impulse lands mid-iteration, so the clamp has to run after
        # contacts to keep the limit at the end of the step.
        speed = self._max_arm_speed_over_impact(fuse=True, articulated_contact_response="immediate")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)

    @unittest.skipUnless(wp.is_cuda_available(), "propagation-fused FPGS requires CUDA")
    def test_fused_clamp_holds_limit_through_impact_propagation_fused(self):
        # Same invariant on the propagation full-iteration kernel, where the
        # box-arm contact is solved in the propagation family with the tree
        # response inside the same kernel.
        speed = self._max_arm_speed_over_impact(fuse=True, articulated_contact_response="propagation-fused")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
