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
    def test_ball_loop_connect_honors_child_anchor(self):
        # Two sibling bodies hang off a fixed root, deliberately placed so that
        # the loop is NOT closed at the zero (reference) pose: bodyA's anchor and
        # bodyB's anchor are 0.5 m apart in world space.
        pos0 = wp.vec3(0.2, 0.0, 0.0)  # anchor on bodyA (parent), bodyA-local
        pos1 = wp.vec3(-0.3, 0.0, 0.0)  # anchor on bodyB (child), bodyB-local
        body_b_offset = wp.vec3(1.0, 0.0, 0.0)  # bodyB sits 1 m from the root

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
            parent_xform=wp.transform(body_b_offset, wp.quat_identity()),
        )

        # Loop-closure ball joint between the two siblings. Crucially it is NOT
        # added to the articulation, so joint_articulation stays -1 and the
        # MuJoCo solver synthesizes a CONNECT equality from it.
        builder.add_joint_ball(
            parent=body_a,
            child=body_b,
            parent_xform=wp.transform(pos0, wp.quat_identity()),
            child_xform=wp.transform(pos1, wp.quat_identity()),
        )

        builder.add_articulation(joints=[root_joint, joint_a, joint_b])

        model = builder.finalize()
        solver = SolverMuJoCo(model)

        import mujoco

        m = solver.mj_model
        self.assertEqual(m.neq, 1, "expected exactly one synthesized equality constraint")
        self.assertEqual(int(m.eq_type[0]), int(mujoco.mjtEq.mjEQ_CONNECT))

        if solver.use_mujoco_cpu:
            eq_data = np.array(m.eq_data)  # [neq, 11]
        else:
            eq_data = solver.mjw_model.eq_data.numpy()[0]  # [neq, 11]

        anchor1 = eq_data[0][0:3]
        anchor2 = eq_data[0][3:6]

        # anchor1 is the parent-side anchor (bodyA-local) == pos0.
        np.testing.assert_allclose(anchor1, [pos0[0], pos0[1], pos0[2]], atol=1e-5)
        # anchor2 is the child-side anchor (bodyB-local) and must equal the
        # authored child frame pos1 -- NOT the reference-pose projection
        # (which would be pos0 - body_b_offset = (-0.8, 0, 0)).
        np.testing.assert_allclose(anchor2, [pos1[0], pos1[1], pos1[2]], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
