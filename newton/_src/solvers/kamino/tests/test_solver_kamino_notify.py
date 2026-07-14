# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SolverKamino runtime model-property propagation."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context


def _build_limited_revolute() -> newton.Model:
    """Build a tiny model: world to a single body via a limited revolute joint."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)

    builder.begin_world()
    bid = builder.add_link(
        label="link",
        mass=1.0,
        xform=wp.transformf(wp.vec3f(0.0, 0.0, 1.0), wp.quat_identity(dtype=wp.float32)),
        lock_inertia=True,
    )
    builder.add_shape_box(label="box", body=bid, hx=0.1, hy=0.1, hz=0.1)
    jid = builder.add_joint_revolute(
        label="world_to_link",
        parent=-1,
        child=bid,
        axis=newton.Axis.Y,
        limit_lower=-1.0,
        limit_upper=1.0,
    )
    builder.add_articulation([jid])
    builder.end_world()

    return builder.finalize()


class TestKaminoNotifyModelChanged(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_joint_dof_limits_reference_newton(self):
        """Joint DoF limits alias Newton's arrays, so runtime changes need no notify."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        joints = solver._model_kamino.joints

        # The Kamino limit arrays share storage with Newton's model arrays.
        self.assertEqual(joints.q_j_min.ptr, model.joint_limit_lower.ptr)
        self.assertEqual(joints.q_j_max.ptr, model.joint_limit_upper.ptr)
        self.assertEqual(joints.dq_j_max.ptr, model.joint_velocity_limit.ptr)
        self.assertEqual(joints.tau_j_max.ptr, model.joint_effort_limit.ptr)

        # In-place updates to Newton's arrays are reflected without any notify call.
        new_lower = np.full_like(model.joint_limit_lower.numpy(), -0.3)
        new_upper = np.full_like(model.joint_limit_upper.numpy(), 0.3)
        new_vel = np.full_like(model.joint_velocity_limit.numpy(), 7.0)
        new_effort = np.full_like(model.joint_effort_limit.numpy(), 70.0)
        model.joint_limit_lower.assign(new_lower)
        model.joint_limit_upper.assign(new_upper)
        model.joint_velocity_limit.assign(new_vel)
        model.joint_effort_limit.assign(new_effort)

        np.testing.assert_allclose(joints.q_j_min.numpy(), new_lower)
        np.testing.assert_allclose(joints.q_j_max.numpy(), new_upper)
        np.testing.assert_allclose(joints.dq_j_max.numpy(), new_vel)
        np.testing.assert_allclose(joints.tau_j_max.numpy(), new_effort)

        # notify_model_changed(JOINT_DOF_PROPERTIES) is a harmless no-op.
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(joints.q_j_min.numpy(), new_lower)

    def test_body_inertial_properties_reference_newton(self):
        """Body inertial arrays alias Newton's, so runtime changes need no notify."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)
        bodies = solver._model_kamino.bodies

        # The Kamino arrays share storage with Newton's model arrays.
        self.assertEqual(bodies.i_r_com_i.ptr, model.body_com.ptr)
        self.assertEqual(bodies.i_I_i.ptr, model.body_inertia.ptr)
        self.assertEqual(bodies.inv_i_I_i.ptr, model.body_inv_inertia.ptr)
        self.assertEqual(bodies.m_i.ptr, model.body_mass.ptr)

        # In-place updates to Newton's arrays are reflected without any notify call.
        new_com = model.body_com.numpy() + np.array([0.1, 0.2, 0.3], dtype=np.float32)
        new_inertia = model.body_inertia.numpy() * 2.0
        new_inv_inertia = model.body_inv_inertia.numpy() * 0.5
        model.body_com.assign(new_com)
        model.body_inertia.assign(new_inertia)
        model.body_inv_inertia.assign(new_inv_inertia)

        np.testing.assert_allclose(bodies.i_r_com_i.numpy(), new_com, atol=1e-6)
        np.testing.assert_allclose(bodies.i_I_i.numpy(), new_inertia, atol=1e-6)
        np.testing.assert_allclose(bodies.inv_i_I_i.numpy(), new_inv_inertia, atol=1e-6)

        # notify_model_changed(BODY_INERTIAL_PROPERTIES) is a harmless no-op.
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        np.testing.assert_allclose(bodies.i_I_i.numpy(), new_inertia, atol=1e-6)

    def test_notify_does_not_mutate_newton_arrays(self):
        """``notify_model_changed`` must only read Newton's arrays, never write them."""
        model = _build_limited_revolute()
        solver = SolverKamino(model)

        model.joint_limit_lower.assign(np.full_like(model.joint_limit_lower.numpy(), -0.3))
        model.joint_limit_upper.assign(np.full_like(model.joint_limit_upper.numpy(), 0.3))
        model.body_inertia.assign(model.body_inertia.numpy() * 2.0)

        snapshot = {
            "joint_limit_lower": model.joint_limit_lower.numpy().copy(),
            "joint_limit_upper": model.joint_limit_upper.numpy().copy(),
            "joint_velocity_limit": model.joint_velocity_limit.numpy().copy(),
            "joint_effort_limit": model.joint_effort_limit.numpy().copy(),
            "body_com": model.body_com.numpy().copy(),
            "body_inertia": model.body_inertia.numpy().copy(),
            "body_inv_inertia": model.body_inv_inertia.numpy().copy(),
        }

        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        for name, before in snapshot.items():
            after = getattr(model, name).numpy()
            np.testing.assert_array_equal(after, before, err_msg=f"notify_model_changed mutated model.{name}")


if __name__ == "__main__":
    unittest.main()
