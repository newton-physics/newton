# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Kamino joint effort limits."""

import unittest

import warp as wp

import newton
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.solver_configs import KAMINO_CONFIGS, PADMM_CONFIG_NAMES, make_single_iteration_config

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

    def test_explicit_effort_is_clamped(self):
        """Clamp explicit effort with the expected signed joint acceleration."""
        effort_limit = 1.0
        requested_effort = 10.0
        for config_name, config_factory in KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                if config.padmm is not None:
                    config.padmm.use_acceleration = use_acceleration
                for direction in (-1.0, 1.0):
                    with self.subTest(
                        config=config_name,
                        use_acceleration=use_acceleration,
                        direction=direction,
                    ):
                        model = _build_revolute(effort_limit)
                        solver = SolverKamino(model, config)
                        control = model.control()
                        control.joint_f.assign([direction * requested_effort])

                        state_out = _step(solver, model, control)

                        expected_effort = direction * effort_limit
                        expected_velocity = DT * expected_effort / _EFFECTIVE_JOINT_INERTIA
                        self.assertAlmostEqual(
                            float(state_out.joint_qd.numpy()[0]),
                            expected_velocity,
                            delta=3.0e-4,
                        )


class TestSolverKaminoJointEffortLimitImplicitPd(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_implicit_pd_effort_saturates_and_is_reported(self):
        """Apply and report saturated implicit-PD effort with the commanded sign."""
        effort_limit = 1.0
        target_ke = 100.0
        for config_name, config_factory in KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                if config.padmm is not None:
                    config.padmm.use_acceleration = use_acceleration
                for armature in (0.0, 1.0):
                    for direction in (-1.0, 1.0):
                        with self.subTest(
                            config=config_name,
                            use_acceleration=use_acceleration,
                            armature=armature,
                            direction=direction,
                        ):
                            model = _build_revolute(effort_limit, target_ke=target_ke, armature=armature)
                            solver = SolverKamino(model, config)
                            control = model.control()
                            control.joint_target_q.assign([direction])

                            state_out = _step(solver, model, control)

                            expected_effort = direction * effort_limit
                            expected_velocity = DT * expected_effort / (_EFFECTIVE_JOINT_INERTIA + armature)
                            actual_effort = float(solver._solver_kamino.data.joints.lambda_tau_j.numpy()[0])
                            self.assertAlmostEqual(actual_effort, expected_effort, delta=3.0e-4)
                            self.assertAlmostEqual(
                                float(state_out.joint_lambdas_tau.numpy()[0]),
                                expected_effort,
                                delta=3.0e-4,
                            )
                            self.assertAlmostEqual(
                                float(solver._solver_kamino.data.bodies.w_a_i.numpy()[0, 4]),
                                expected_effort,
                                delta=3.0e-4,
                            )
                            self.assertAlmostEqual(
                                float(state_out.joint_qd.numpy()[0]),
                                expected_velocity,
                                delta=3.0e-4,
                            )


class TestSolverKaminoJointEffortLimitWarmstart(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_container_warmstart_reduces_effort_residual(self):
        """Reduce one-iteration residuals using cached implicit-PD effort."""
        effort_limit = 1.0
        target_ke = 100.0
        for config_name, config_factory in KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                if config.padmm is not None:
                    config.padmm.use_acceleration = use_acceleration
                for direction in (-1.0, 1.0):
                    with self.subTest(
                        config=config_name,
                        use_acceleration=use_acceleration,
                        direction=direction,
                    ):
                        model = _build_revolute(effort_limit, target_ke=target_ke, armature=1.0)
                        solver = SolverKamino(model, config)
                        control = model.control()
                        control.joint_target_q.assign([direction])
                        state_1 = _step(solver, model, control)
                        cached_effort = float(state_1.joint_lambdas_tau.numpy()[0])
                        self.assertNotAlmostEqual(cached_effort, 0.0, delta=1.0e-6)

                        cold_solver = SolverKamino(
                            model,
                            make_single_iteration_config(
                                config_factory,
                                warmstart_mode="none",
                                use_acceleration=use_acceleration,
                            ),
                        )
                        warm_solver = SolverKamino(
                            model,
                            make_single_iteration_config(
                                config_factory,
                                warmstart_mode="containers",
                                use_acceleration=use_acceleration,
                            ),
                        )
                        cold_solver.step(state_1, model.state(), control=control, contacts=None, dt=DT)
                        warm_solver.step(state_1, model.state(), control=control, contacts=None, dt=DT)

                        cold_residual = float(cold_solver._solver_kamino.metrics.data.r_vi_natmap.numpy()[0])
                        warm_residual = float(warm_solver._solver_kamino.metrics.data.r_vi_natmap.numpy()[0])
                        if not (cold_residual < 1.0e-6 and warm_residual < 1.0e-6):
                            self.assertLess(warm_residual, cold_residual)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
