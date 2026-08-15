# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for regularized Kamino joint friction."""

from __future__ import annotations

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.models.builders.basics import build_boxes_fourbar
from newton._src.solvers.kamino._src.models.builders.utils import make_homogeneous_builder
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context


def _add_body(builder: newton.ModelBuilder, label: str) -> int:
    """Add a unit-inertia body centered on its joint frame."""
    return builder.add_link(
        label=label,
        mass=1.0,
        inertia=wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )


def _build_heterogeneous_model(device: wp.DeviceLike) -> newton.Model:
    """Build revolute and mixed D6 worlds with distinct friction values."""
    builder = newton.ModelBuilder(gravity=0.0)
    SolverKamino.register_custom_attributes(builder)

    builder.begin_world()
    body = _add_body(builder, "revolute_body")
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z, friction=2.0)
    builder.add_articulation([joint])
    builder.end_world()

    builder.begin_world()
    body = _add_body(builder, "d6_body")
    joint = builder.add_joint_d6(
        parent=-1,
        child=body,
        linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, friction=3.0)],
        angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, friction=4.0)],
    )
    builder.add_articulation([joint])
    builder.end_world()

    return builder.finalize(device=device)


def _build_revolute_model(device: wp.DeviceLike, friction: float | None) -> newton.Model:
    """Build an unforced unit-inertia revolute joint."""
    builder = newton.ModelBuilder(gravity=0.0)
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = _add_body(builder, "body")
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z, friction=friction)
    builder.add_articulation([joint])
    builder.end_world()
    model = builder.finalize(device=device)
    model.joint_qd.assign([1.0])
    return model


def _make_consistent_state(model: newton.Model) -> newton.State:
    """Create a state whose body motion matches the model joint state."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


class TestKaminoJointFriction(unittest.TestCase):
    def setUp(self):
        """Initialize Warp and select the configured test device."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_velocity_threshold_requires_positive_finite_value(self):
        """Reject non-positive and non-finite friction velocity thresholds."""
        self.assertGreater(SolverKamino.Config().joint_friction_velocity_threshold, 0.0)
        for threshold in (0.0, -1.0, math.inf, -math.inf, math.nan):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    SolverKamino.Config(joint_friction_velocity_threshold=threshold)

    def test_internal_builder_defaults_to_zero_friction(self):
        """Allocate zero joint friction for low-level Kamino model builders."""
        builder = make_homogeneous_builder(num_worlds=1, build_fn=build_boxes_fourbar)
        model = builder.finalize(device=self.device)

        np.testing.assert_array_equal(
            model.joints.friction_j.numpy(),
            np.zeros(model.size.sum_of_num_joint_dofs, dtype=np.float32),
        )

    def test_effective_effort_is_signed_bounded_and_does_not_mutate_control(self):
        """Apply friction once across heterogeneous worlds while preserving raw control."""
        model = _build_heterogeneous_model(self.device)
        model.joint_qd.assign([2.0, -0.05, 0.0])
        state_in = _make_consistent_state(model)
        state_out = model.state()
        control = model.control()
        control.joint_f.assign([0.5, -1.0, 2.0])
        control_before = control.joint_f.numpy().copy()

        solver = SolverKamino(
            model,
            config=SolverKamino.Config(joint_friction_velocity_threshold=0.1),
        )
        solver.step(state_in, state_out, control, contacts=None, dt=1.0e-3)

        np.testing.assert_allclose(
            solver._solver_kamino._data.joints.tau_j.numpy(),
            [-1.5, 0.5, 2.0],
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_array_equal(control.joint_f.numpy(), control_before)
        self.assertEqual(solver._model_kamino.joints.friction_j.ptr, model.joint_friction.ptr)

    def test_runtime_property_update_changes_friction_effort(self):
        """Use updated model friction after a joint-property notification."""
        model = _build_revolute_model(self.device, friction=1.0)
        state_in = _make_consistent_state(model)
        state_out = model.state()
        control = model.control()
        solver = SolverKamino(model, config=SolverKamino.Config(joint_friction_velocity_threshold=0.1))

        solver.step(state_in, state_out, control, contacts=None, dt=1.0e-3)
        self.assertAlmostEqual(float(solver._solver_kamino._data.joints.tau_j.numpy()[0]), -1.0, places=6)

        model.joint_friction.assign([0.25])
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        solver.step(state_in, state_out, control, contacts=None, dt=1.0e-3)
        self.assertAlmostEqual(float(solver._solver_kamino._data.joints.tau_j.numpy()[0]), -0.25, places=6)

    def test_backend_and_layout_paths_share_effective_effort(self):
        """Prepare identical friction effort for PADMM/DVI and dense/sparse Jacobians."""
        for dynamics_solver in ("padmm", "dvi"):
            for sparse_jacobian in (False, True):
                with self.subTest(dynamics_solver=dynamics_solver, sparse_jacobian=sparse_jacobian):
                    model = _build_revolute_model(self.device, friction=0.75)
                    state_in = _make_consistent_state(model)
                    state_out = model.state()
                    control = model.control()
                    control.joint_f.assign([0.2])
                    solver = SolverKamino(
                        model,
                        config=SolverKamino.Config(
                            dynamics_solver=dynamics_solver,
                            sparse_jacobian=sparse_jacobian,
                            joint_friction_velocity_threshold=0.1,
                        ),
                    )

                    solver.step(state_in, state_out, control, contacts=None, dt=1.0e-3)

                    self.assertAlmostEqual(
                        float(solver._solver_kamino._data.joints.tau_j.numpy()[0]),
                        -0.55,
                        places=6,
                    )

    def test_unforced_revolute_joint_slows_without_reversing(self):
        """Dissipate unforced revolute motion without crossing zero velocity."""
        for dynamics_solver in ("padmm", "dvi"):
            with self.subTest(dynamics_solver=dynamics_solver):
                model = _build_revolute_model(self.device, friction=2.0)
                state_in = _make_consistent_state(model)
                state_out = model.state()
                solver = SolverKamino(
                    model,
                    config=SolverKamino.Config(
                        dynamics_solver=dynamics_solver,
                        joint_friction_velocity_threshold=0.1,
                    ),
                )
                velocities = [1.0]

                for _ in range(100):
                    solver.step(state_in, state_out, control=None, contacts=None, dt=0.01)
                    velocities.append(float(state_out.joint_qd.numpy()[0]))
                    state_in, state_out = state_out, state_in

                self.assertTrue(all(velocity >= -1.0e-6 for velocity in velocities))
                self.assertLess(abs(velocities[-1]), 1.0e-3)
                self.assertLess(abs(velocities[-1]), abs(velocities[0]))

    def test_zero_friction_preserves_existing_results(self):
        """Match the existing solver result when joint friction is zero."""
        states = []
        for friction in (0.0, None):
            model = _build_revolute_model(self.device, friction=friction)
            state_in = _make_consistent_state(model)
            state_out = model.state()
            solver = SolverKamino(model)
            solver.step(state_in, state_out, control=None, contacts=None, dt=0.01)
            states.append((state_out.body_q.numpy(), state_out.body_qd.numpy(), state_out.joint_qd.numpy()))

        for actual, expected in zip(states[1], states[0], strict=True):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(states[0][2], [1.0], rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
