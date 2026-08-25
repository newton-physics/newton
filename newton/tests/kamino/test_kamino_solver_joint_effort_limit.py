# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Kamino joint effort limits."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.joints import JOINT_TAUMAX
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context

_BODY_MASS = 1.0
_BODY_INERTIA = 1.0
_BODY_COM_X = 0.5
_EFFECTIVE_JOINT_INERTIA = _BODY_INERTIA + _BODY_MASS * _BODY_COM_X**2
DT = 0.01


def _build_revolute(
    effort_limit: float,
    *,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
    armature: float = 0.0,
) -> newton.Model:
    """Build a single world-to-body revolute model."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=_BODY_MASS,
        inertia=[
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
        ],
        com=wp.vec3f(_BODY_COM_X, 0.0, 0.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=newton.Axis.Y,
        effort_limit=effort_limit,
        target_ke=target_ke,
        target_kd=target_kd,
        armature=armature,
        actuator_mode=newton.JointTargetMode.POSITION if target_ke > 0.0 else newton.JointTargetMode.NONE,
    )
    builder.add_articulation([joint])
    builder.end_world()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_two_dof_effort_limit() -> newton.Model:
    """Build a two-DoF joint with one effort-limited implicit PD axis."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=_BODY_MASS,
        inertia=[
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
        ],
        com=wp.vec3f(_BODY_COM_X, 0.0, 0.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_d6(
        -1,
        body,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.X,
                effort_limit=1.0,
                target_ke=100.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            ),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, effort_limit=math.inf),
        ],
    )
    builder.add_articulation([joint])
    builder.end_world()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_two_world_effort_limit() -> newton.Model:
    """Build two worlds, each with one effort-limited implicit-PD joint."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    for _ in range(2):
        builder.begin_world()
        body = builder.add_link(
            mass=_BODY_MASS,
            inertia=[
                _BODY_INERTIA,
                0.0,
                0.0,
                0.0,
                _BODY_INERTIA,
                0.0,
                0.0,
                0.0,
                _BODY_INERTIA,
            ],
            com=wp.vec3f(_BODY_COM_X, 0.0, 0.0),
            lock_inertia=True,
        )
        joint = builder.add_joint_revolute(
            -1,
            body,
            axis=newton.Axis.Y,
            effort_limit=1.0,
            target_ke=100.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        builder.add_articulation([joint])
        builder.end_world()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _build_three_axis_constraint_topology() -> newton.Model:
    """Build a D6 joint with dynamic, effort, and friction rows on separate axes."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=_BODY_MASS,
        inertia=[
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_INERTIA,
        ],
        com=wp.vec3f(_BODY_COM_X, 0.0, 0.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_d6(
        -1,
        body,
        angular_axes=[
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, armature=1.0),
            newton.ModelBuilder.JointDofConfig(
                axis=newton.Axis.Y,
                effort_limit=1.0,
                target_ke=100.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            ),
            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, friction=0.5),
        ],
    )
    builder.add_articulation([joint])
    builder.end_world()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, 0.0))
    return model


def _config(dynamics_solver: str) -> SolverKamino.Config:
    """Create a dense Kamino configuration for one dynamics backend."""
    return SolverKamino.Config(
        dynamics_solver=dynamics_solver,
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
    )


_BACKENDS = ("padmm", "dvi")


def _initial_state(model: newton.Model) -> newton.State:
    """Initialize generalized and maximal state consistently."""
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


def _step(
    solver: SolverKamino,
    model: newton.Model,
    control: newton.Control,
) -> newton.State:
    """Advance the model by one step from rest."""
    state_in = _initial_state(model)
    state_out = model.state()
    solver.step(state_in, state_out, control=control, contacts=None, dt=DT)
    return state_out


class TestSolverKaminoJointEffortLimit(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_pack_separate_axes_for_each_constraint_type(self):
        """Pack dynamic, effort, and friction rows in ascending local-axis order."""
        solver = SolverKamino(_build_three_axis_constraint_topology(), _config("padmm"))
        joints = solver._model_kamino.joints

        np.testing.assert_array_equal(joints.dynamic_cts_axis.numpy(), [0])
        np.testing.assert_array_equal(joints.effort_cts_axis.numpy(), [1])
        np.testing.assert_array_equal(joints.friction_cts_axis.numpy(), [2])
        np.testing.assert_array_equal(
            joints.actuation_path_dof.numpy(),
            [
                solver._kamino.JointActuationPath.DYNAMIC_CTS,
                solver._kamino.JointActuationPath.EFFORT_CTS,
                solver._kamino.JointActuationPath.BODY_WRENCHES,
            ],
        )

    def test_allocate_selected_effort_row_for_partial_two_dof_limit(self):
        """Allocate only the bounded implicit-PD axis using its per-DoF mode."""
        solver = SolverKamino(_build_two_dof_effort_limit(), _config("padmm"))
        kamino = solver._model_kamino

        np.testing.assert_array_equal(
            kamino.joints.act_type_dof.numpy(),
            [
                solver._kamino.JointActuationType.POSITION,
                solver._kamino.JointActuationType.PASSIVE,
            ],
        )
        self.assertEqual(kamino.size.sum_of_num_effort_joint_cts, 1)
        np.testing.assert_array_equal(kamino.info.num_joint_effort_cts.numpy(), [1])
        np.testing.assert_array_equal(kamino.joints.effort_cts_offset.numpy(), [0, 1])
        np.testing.assert_array_equal(kamino.joints.effort_cts_axis.numpy(), [0])

    def test_actuation_path_dof_for_partial_two_dof_limit(self):
        """Route bounded implicit PD through effort rows and leave the other axis explicit."""
        solver = SolverKamino(_build_two_dof_effort_limit(), _config("padmm"))
        kamino = solver._model_kamino
        effort = solver._kamino.JointActuationPath.EFFORT_CTS
        body = solver._kamino.JointActuationPath.BODY_WRENCHES

        np.testing.assert_array_equal(kamino.joints.actuation_path_dof.numpy(), [effort, body])

    def test_allocate_implicit_effort_constraint(self):
        """Allocate an effort row only for a finite-limit implicit PD actuator."""
        for effort_limit, target_ke, expected_count in (
            (math.inf, 100.0, 0),
            (JOINT_TAUMAX, 100.0, 0),
            (1.0, 0.0, 0),
            (1.0, 100.0, 1),
        ):
            with self.subTest(effort_limit=effort_limit, target_ke=target_ke):
                model = _build_revolute(effort_limit, target_ke=target_ke)
                solver = SolverKamino(model, _config("padmm"))
                kamino = solver._model_kamino

                self.assertEqual(kamino.size.sum_of_num_effort_joint_cts, expected_count)
                np.testing.assert_array_equal(kamino.info.num_joint_effort_cts.numpy(), [expected_count])
                np.testing.assert_array_equal(kamino.info.joint_effort_cts_offset.numpy(), [0])
                np.testing.assert_array_equal(kamino.joints.effort_cts_offset.numpy(), [0, expected_count])
                np.testing.assert_array_equal(
                    kamino.joints.effort_cts_offset_total_cts.numpy(),
                    kamino.info.total_cts_offset.numpy() + kamino.info.joint_effort_cts_group_offset.numpy(),
                )
                np.testing.assert_array_equal(
                    kamino.joints.effort_cts_axis.numpy(),
                    [] if expected_count == 0 else [0],
                )

    def test_passive_dynamic_without_effort_limit_has_dynamic_row(self):
        """Allocate a dynamic row for passive armature."""
        model = _build_revolute(math.inf, target_ke=100.0, armature=1.0)
        solver = SolverKamino(model, _config("padmm"))

        np.testing.assert_array_equal(
            solver._model_kamino.joints.dynamic_cts_axis.numpy(),
            [0],
        )

    def test_split_dynamic_constraint_uses_same_axis(self):
        """Keep passive and bounded actuator rows on their shared axis."""
        model = _build_revolute(1.0, target_ke=100.0, armature=1.0)
        solver = SolverKamino(model, _config("padmm"))

        np.testing.assert_array_equal(
            solver._model_kamino.joints.dynamic_cts_axis.numpy(),
            [0],
        )
        np.testing.assert_array_equal(solver._model_kamino.joints.effort_cts_axis.numpy(), [0])

    def test_explicit_effort_is_clamped(self):
        """Clamp explicit joint effort before solving in both dynamics backends."""
        effort_limit = 1.0
        requested_effort = 10.0
        expected_velocity = DT * effort_limit / _EFFECTIVE_JOINT_INERTIA
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                model = _build_revolute(effort_limit)
                solver = SolverKamino(model, _config(backend))
                control = model.control()
                control.joint_f.assign([requested_effort])

                state_out = _step(solver, model, control)

                self.assertAlmostEqual(float(state_out.joint_qd.numpy()[0]), expected_velocity, delta=2.0e-4)

    def test_implicit_pd_effort_saturates_and_is_reported(self):
        """Apply and report saturated implicit PD torque with the commanded sign."""
        effort_limit = 1.0
        target_ke = 100.0
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                model = _build_revolute(effort_limit, target_ke=target_ke)
                solver = SolverKamino(model, _config(backend))
                control = model.control()
                control.joint_target_q.assign([1.0])

                state_out = _step(solver, model, control)

                self.assertGreater(float(state_out.joint_qd.numpy()[0]), 0.0)
                self.assertAlmostEqual(
                    float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0]),
                    effort_limit,
                    delta=2.0e-4,
                )
                self.assertAlmostEqual(
                    float(state_out.joint_lambdas_tau.numpy()[0]),
                    effort_limit,
                    delta=2.0e-4,
                )
                self.assertAlmostEqual(
                    float(solver._solver_kamino.data.bodies.w_a_i.numpy()[0, 4]),
                    effort_limit,
                    delta=2.0e-4,
                )

    def test_implicit_pd_effort_saturates_with_armature(self):
        """Apply the saturated torque through the split passive dynamic row."""
        effort_limit = 1.0
        armature = 1.0
        for backend in _BACKENDS:
            with self.subTest(backend=backend):
                model = _build_revolute(effort_limit, target_ke=100.0, armature=armature)
                solver = SolverKamino(model, _config(backend))
                control = model.control()
                control.joint_target_q.assign([1.0])

                state_out = _step(solver, model, control)

                self.assertGreater(float(state_out.joint_qd.numpy()[0]), 0.0)
                self.assertAlmostEqual(
                    float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0]),
                    effort_limit,
                    delta=2.0e-4,
                )

    def test_finite_pd_effort_limit_update_preserves_topology(self):
        """Apply finite PD effort-bound edits without reallocating constraint rows."""
        initial_limit = 1.0
        updated_limit = 0.25
        model = _build_revolute(initial_limit, target_ke=100.0)
        solver = SolverKamino(model, _config("padmm"))
        kamino = solver._model_kamino
        initial_count = int(kamino.info.num_joint_effort_cts.numpy()[0])
        initial_offset = int(kamino.joints.effort_cts_offset.numpy()[0])
        control = model.control()
        control.joint_target_q.assign([1.0])

        _step(solver, model, control)
        model.joint_effort_limit.assign([updated_limit])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _step(solver, model, control)

        self.assertEqual(int(kamino.info.num_joint_effort_cts.numpy()[0]), initial_count)
        self.assertEqual(int(kamino.joints.effort_cts_offset.numpy()[0]), initial_offset)
        self.assertAlmostEqual(
            float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0]),
            updated_limit,
            delta=2.0e-4,
        )

    def test_effort_limit_finiteness_change_requires_recreation(self):
        """Reject effort-limit edits that change implicit-PD row topology."""
        model = _build_revolute(1.0, target_ke=100.0)
        solver = SolverKamino(model, _config("padmm"))
        model.joint_effort_limit.assign([math.inf])

        with self.assertRaisesRegex(RuntimeError, "dynamic constraint topology"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_actuator_mode_change_that_removes_effort_row_requires_recreation(self):
        """Reject an actuator-only update that removes an effort row."""
        model = _build_revolute(1.0, target_ke=100.0)
        solver = SolverKamino(model, _config("padmm"))
        model.joint_target_mode.assign([newton.JointTargetMode.EFFORT])

        with self.assertRaisesRegex(RuntimeError, "effort-limit row topology"):
            solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)

    def test_joint_actuation_aggregates_mapped_per_dof_types(self):
        """Aggregate DoF actuation types instead of raw Newton target-mode values."""
        model = _build_two_dof_effort_limit()
        model.joint_target_mode.assign(
            [
                newton.JointTargetMode.VELOCITY,
                newton.JointTargetMode.EFFORT,
            ]
        )
        solver = SolverKamino(model, _config("padmm"))

        self.assertEqual(
            solver._model_kamino.joints.act_type.numpy()[0],
            solver._kamino.JointActuationType.VELOCITY,
        )

    def test_multiworld_effort_offsets_follow_global_row_order(self):
        """Prefix effort rows globally while retaining per-world offsets."""
        solver = SolverKamino(_build_two_world_effort_limit(), _config("padmm"))
        kamino = solver._model_kamino

        np.testing.assert_array_equal(kamino.info.num_joint_effort_cts.numpy(), [1, 1])
        np.testing.assert_array_equal(kamino.info.joint_effort_cts_offset.numpy(), [0, 1])
        np.testing.assert_array_equal(kamino.joints.effort_cts_offset.numpy(), [0, 1, 2])
        np.testing.assert_array_equal(
            kamino.joints.effort_cts_offset_total_cts.numpy(),
            kamino.info.total_cts_offset.numpy() + kamino.info.joint_effort_cts_group_offset.numpy(),
        )


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
