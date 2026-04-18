# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _build_model():
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=0.0)
    inertia = wp.mat33((0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1))
    body = builder.add_link(armature=0.0, inertia=inertia, mass=1.0)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=0.0,
        target_ke=100.0,
        target_kd=10.0,
        effort_limit=5.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([joint])
    builder.request_state_attributes("mujoco:qfrc_actuator")
    model = builder.finalize()
    model.ground = False
    return model


class TestStateAssignNamespacedAttributes(unittest.TestCase):
    def test_copies_namespaced_attribute(self):
        model = _build_model()
        state_0 = model.state()
        state_1 = model.state()

        sentinel = np.array([3.14], dtype=np.float32)
        state_1.mujoco.qfrc_actuator.assign(sentinel)

        state_0.assign(state_1)

        np.testing.assert_allclose(state_0.mujoco.qfrc_actuator.numpy(), sentinel)

    def test_raises_when_src_missing_namespaced_attribute(self):
        model = _build_model()
        state_0 = model.state()
        state_1 = model.state()
        delattr(state_1, "mujoco")

        with self.assertRaises(ValueError):
            state_0.assign(state_1)

    def test_raises_when_dst_missing_namespaced_attribute(self):
        model = _build_model()
        state_0 = model.state()
        state_1 = model.state()
        delattr(state_0, "mujoco")

        with self.assertRaises(ValueError):
            state_0.assign(state_1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
