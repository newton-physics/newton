# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _build_model(*, custom_attrs: tuple[str, ...] = ()):
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
    for name in custom_attrs:
        builder.add_custom_attribute(
            newton.ModelBuilder.CustomAttribute(
                name=name,
                frequency=newton.Model.AttributeFrequency.BODY,
                dtype=wp.float32,
                default=0.0,
                assignment=newton.Model.AttributeAssignment.STATE,
                namespace="my_namespace",
            )
        )
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

    def test_copies_custom_namespaced_attribute(self):
        model = _build_model(custom_attrs=("my_attribute",))
        state_0 = model.state()
        state_1 = model.state()

        sentinel = np.array([2.71], dtype=np.float32)
        state_1.my_namespace.my_attribute.assign(sentinel)

        state_0.assign(state_1)

        np.testing.assert_allclose(state_0.my_namespace.my_attribute.numpy(), sentinel)

    def test_raises_when_src_missing_custom_namespaced_attribute(self):
        model = _build_model(custom_attrs=("my_attribute",))
        state_0 = model.state()
        state_1 = model.state()
        delattr(state_1, "my_namespace")

        with self.assertRaises(ValueError):
            state_0.assign(state_1)

    def test_raises_when_dst_missing_custom_namespaced_attribute(self):
        model = _build_model(custom_attrs=("my_attribute",))
        state_0 = model.state()
        state_1 = model.state()
        delattr(state_0, "my_namespace")

        with self.assertRaises(ValueError):
            state_0.assign(state_1)

    def test_copies_multiple_custom_namespaced_attributes(self):
        model = _build_model(custom_attrs=("attr_one", "attr_two"))
        state_0 = model.state()
        state_1 = model.state()

        sentinel_one = np.array([1.23], dtype=np.float32)
        sentinel_two = np.array([4.56], dtype=np.float32)
        state_1.my_namespace.attr_one.assign(sentinel_one)
        state_1.my_namespace.attr_two.assign(sentinel_two)

        state_0.assign(state_1)

        np.testing.assert_allclose(state_0.my_namespace.attr_one.numpy(), sentinel_one)
        np.testing.assert_allclose(state_0.my_namespace.attr_two.numpy(), sentinel_two)

    def test_raises_when_one_of_multiple_custom_attributes_missing(self):
        model = _build_model(custom_attrs=("attr_one", "attr_two"))
        state_0 = model.state()
        state_1 = model.state()
        # Remove a single attribute inside the namespace container (not the
        # container itself) to exercise per-attribute presence checks.
        delattr(state_1.my_namespace, "attr_two")

        with self.assertRaises(ValueError):
            state_0.assign(state_1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
