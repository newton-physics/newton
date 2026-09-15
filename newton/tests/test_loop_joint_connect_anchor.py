# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for joint-synthesized CONNECT anchors of loop-closure joints.

A ball loop-closure joint (``joint_articulation == -1``) is exported to MuJoCo
as an ``mjEQ_CONNECT`` equality. Its body2-side anchor must equal the joint's
authored child frame (``joint_X_c``), *independent* of whether the model is
assembled (loop closed) at the reference joint configuration. Earlier the anchor
was inferred from the bodies' reference pose, which discards the authored child
anchor and collapses distinct loop closures whenever the model is not assembled
at the reference pose (e.g. a closed-loop leg whose default pose does not close
the loop).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


class TestLoopJointConnectAnchor(unittest.TestCase):
    # Anchors on the two loop bodies, each in its own body frame. Body B sits 1 m from the root,
    # so the anchors are 0.5 m apart in world space at the zero pose: the loop is deliberately
    # NOT closed at the reference configuration.
    POS_A = wp.vec3(0.2, 0.0, 0.0)
    POS_B = wp.vec3(-0.3, 0.0, 0.0)
    BODY_B_OFFSET = wp.vec3(1.0, 0.0, 0.0)

    def _build_model(self) -> newton.Model:
        inertia = wp.mat33(np.eye(3))

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        SolverMuJoCo.register_custom_attributes(builder)

        root = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        root_joint = builder.add_joint_fixed(parent=-1, child=root)

        body_a = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        joint_a = builder.add_joint_revolute(parent=root, child=body_a, axis=wp.vec3(0.0, 0.0, 1.0))

        body_b = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, 0.0), inertia=inertia)
        joint_b = builder.add_joint_revolute(
            parent=root,
            child=body_b,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(self.BODY_B_OFFSET, wp.quat_identity()),
        )

        # Left out of the articulation so it stays a loop closure that MuJoCo receives as a CONNECT.
        builder.add_joint_ball(
            parent=body_a,
            child=body_b,
            parent_xform=wp.transform(self.POS_A, wp.quat_identity()),
            child_xform=wp.transform(self.POS_B, wp.quat_identity()),
        )

        builder.add_articulation(joints=[root_joint, joint_a, joint_b])
        return builder.finalize()

    def _assert_anchors(self, use_mujoco_cpu: bool):
        import mujoco

        solver = SolverMuJoCo(self._build_model(), use_mujoco_cpu=use_mujoco_cpu)

        m = solver.mj_model
        self.assertEqual(m.neq, 1, "expected exactly one synthesized equality constraint")
        self.assertEqual(int(m.eq_type[0]), int(mujoco.mjtEq.mjEQ_CONNECT))

        if use_mujoco_cpu:
            eq_data = np.array(m.eq_data)  # [neq, 11]
        else:
            eq_data = solver.mjw_model.eq_data.numpy()[0]  # [neq, 11]

        anchor1 = eq_data[0][0:3]
        anchor2 = eq_data[0][3:6]

        # anchor1 is the parent-side anchor (bodyA-local).
        np.testing.assert_allclose(anchor1, list(self.POS_A), atol=1e-5)
        # anchor2 is the child-side anchor (bodyB-local) and must equal the authored child frame,
        # NOT the reference-pose projection POS_A - BODY_B_OFFSET = (-0.8, 0, 0).
        np.testing.assert_allclose(anchor2, list(self.POS_B), atol=1e-5)

    def test_ball_loop_connect_honors_child_anchor(self):
        """Check that the MuJoCo-Warp path writes the authored child anchor into eq_data."""
        self._assert_anchors(use_mujoco_cpu=False)

    def test_ball_loop_connect_honors_child_anchor_cpu(self):
        """Check that the MuJoCo CPU path writes the authored child anchor into eq_data."""
        self._assert_anchors(use_mujoco_cpu=True)


if __name__ == "__main__":
    unittest.main()
