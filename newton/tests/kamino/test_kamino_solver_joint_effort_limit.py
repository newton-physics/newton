# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Kamino joint effort limits."""

import unittest

import warp as wp

import newton
from newton._src.solvers.kamino._src.core.joints import JOINT_TAUMAX
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
    armatures: tuple[float, ...] = (0.0,),
    damping: float = 0.0,
    body_inertia: float = _BODY_INERTIA,
) -> newton.Model:
    """Build one independent revolute joint per world."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    for armature in armatures:
        builder.begin_world()
        body = builder.add_link(
            mass=_BODY_MASS,
            inertia=[
                body_inertia,
                0.0,
                0.0,
                0.0,
                body_inertia,
                0.0,
                0.0,
                0.0,
                body_inertia,
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
            damping=damping,
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
        directions = (-1.0, 1.0)
        for config_name, config_factory in KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                if config.padmm is not None:
                    config.padmm.use_acceleration = use_acceleration
                model = _build_revolute(effort_limit, armatures=(0.0, 0.0))
                solver = SolverKamino(model, config)
                control = model.control()
                control.joint_f.assign([direction * requested_effort for direction in directions])
                state_out = _step(solver, model, control)
                velocities = state_out.joint_qd.numpy()

                for world, direction in enumerate(directions):
                    with self.subTest(
                        config=config_name,
                        use_acceleration=use_acceleration,
                        direction=direction,
                    ):
                        expected_effort = direction * effort_limit
                        expected_velocity = DT * expected_effort / _EFFECTIVE_JOINT_INERTIA
                        self.assertAlmostEqual(float(velocities[world]), expected_velocity, delta=3.0e-4)

    def test_unbounded_explicit_effort_is_preserved(self):
        """Preserve commands above the unbounded sentinel on the body-wrench and dynamic paths."""
        requested_effort = 3.0 * JOINT_TAUMAX
        for config_name, config_factory in KAMINO_CONFIGS:
            # Zero armature routes through body wrenches; a positive one through joint dynamics.
            for armature in (0.0, JOINT_TAUMAX):
                inertia = JOINT_TAUMAX + armature + _BODY_MASS * _BODY_COM_X**2
                for limit in (JOINT_TAUMAX, 2.0 * JOINT_TAUMAX, float("inf")):
                    model = _build_revolute(limit, armatures=(armature,) * 2, body_inertia=JOINT_TAUMAX)
                    solver = SolverKamino(model, config_factory())
                    control = model.control()
                    control.joint_f.assign([-requested_effort, requested_effort])
                    velocities = _step(solver, model, control).joint_qd.numpy()
                    for world, direction in enumerate((-1.0, 1.0)):
                        with self.subTest(config=config_name, armature=armature, limit=limit, direction=direction):
                            expected = DT * direction * requested_effort / inertia
                            self.assertAlmostEqual(float(velocities[world]), expected, delta=3.0e-5)

    def test_explicit_effort_with_passive_dynamics(self):
        """Limit actuator torque while retaining passive inertia and damping reactions."""
        cases = [(armature, direction) for armature in (0.0, 1.0) for direction in (-1.0, 1.0)]
        for config_name, config_factory in KAMINO_CONFIGS:
            for damping in (0.0, 20.0):
                model = _build_revolute(1.0, armatures=tuple(a for a, _ in cases), damping=damping)
                model.joint_qd.fill_(1.0)
                solver = SolverKamino(model, config_factory())
                control = model.control()
                control.joint_f.assign([10.0 * direction for _, direction in cases])
                velocities = _step(solver, model, control).joint_qd.numpy()
                for world, (armature, direction) in enumerate(cases):
                    with self.subTest(config=config_name, damping=damping, armature=armature, direction=direction):
                        inertia = _EFFECTIVE_JOINT_INERTIA + armature
                        expected = (inertia + DT * direction) / (inertia + DT * damping)
                        self.assertAlmostEqual(float(velocities[world]), expected, delta=3.0e-4)


class TestSolverKaminoJointEffortLimitImplicitPd(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_implicit_pd_effort_saturates_and_is_reported(self):
        """Apply and report saturated implicit-PD effort with the commanded sign."""
        effort_limit = 1.0
        target_ke = 100.0
        cases = [(armature, direction) for armature in (0.0, 1.0) for direction in (-1.0, 1.0)]
        for config_name, config_factory in KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                if config.padmm is not None:
                    config.padmm.use_acceleration = use_acceleration
                model = _build_revolute(
                    effort_limit, target_ke=target_ke, armatures=tuple(armature for armature, _ in cases)
                )
                solver = SolverKamino(model, config)
                control = model.control()
                control.joint_target_q.assign([direction for _, direction in cases])
                state_out = _step(solver, model, control)
                actual_efforts = solver._solver_kamino.data.joints.lambda_tau_j.numpy()
                reported_efforts = state_out.joint_lambdas_tau.numpy()
                body_efforts = solver._solver_kamino.data.bodies.w_a_i.numpy()
                velocities = state_out.joint_qd.numpy()

                for world, (armature, direction) in enumerate(cases):
                    with self.subTest(
                        config=config_name,
                        use_acceleration=use_acceleration,
                        armature=armature,
                        direction=direction,
                    ):
                        expected_effort = direction * effort_limit
                        expected_velocity = DT * expected_effort / (_EFFECTIVE_JOINT_INERTIA + armature)
                        self.assertAlmostEqual(float(actual_efforts[world]), expected_effort, delta=3.0e-4)
                        self.assertAlmostEqual(float(reported_efforts[world]), expected_effort, delta=3.0e-4)
                        self.assertAlmostEqual(float(body_efforts[world, 4]), expected_effort, delta=3.0e-4)
                        self.assertAlmostEqual(float(velocities[world]), expected_velocity, delta=3.0e-4)

    def test_schur_default_follows_bounded_pd(self):
        """Resolve the automatic Schur default from bounded implicit-PD rows and honor overrides."""
        cases = [
            # (effort limit, use_schur_complement, expected resolved mode)
            (10.0, None, True),
            (float("inf"), None, False),
            (10.0, False, False),
            (float("inf"), True, True),
        ]
        for config_name, config_factory in KAMINO_CONFIGS:
            if config_name in PADMM_CONFIG_NAMES:
                continue
            for limit, requested, expected in cases:
                config = config_factory()
                config.dvi.use_schur_complement = requested
                model = _build_revolute(limit, target_ke=10.0, target_kd=2.0, armatures=(0.011,) * 2)
                solver = SolverKamino(model, config)
                with self.subTest(config=config_name, limit=limit, requested=requested):
                    self.assertIs(solver._solver_kamino._solver_fd._use_schur_complement, expected)

    def test_bounded_pd_with_small_body_inertia(self):
        """Solve coupled drive and joint reactions for interior and saturated efforts."""
        # The offset COM makes the joint inertia much larger than the body inertia.
        # Alternating drive and bilateral rows then converges very slowly.
        body_inertia = 1.0e-4
        armature, damping = 0.011, 0.044
        kp, kd, limit = 10.0, 2.0, 10.0
        targets = (0.2, -0.2, 2.0, -2.0)
        inertia = body_inertia + _BODY_MASS * _BODY_COM_X**2 + armature + DT * damping
        configs = [(name, factory()) for name, factory in KAMINO_CONFIGS]
        for config_name, config_factory in KAMINO_CONFIGS:
            if config_name not in PADMM_CONFIG_NAMES:
                config = config_factory()
                config.dvi.unilateral_solver = "apgd"
                configs.append((f"{config_name}_apgd", config))
        for config_name, config in configs:
            model = _build_revolute(
                limit,
                target_ke=kp,
                target_kd=kd,
                armatures=(armature,) * len(targets),
                damping=damping,
                body_inertia=body_inertia,
            )
            if config_name in PADMM_CONFIG_NAMES:
                # Fixed-penalty ADMM needs more iterations for saturated, coupled rows.
                config.padmm.max_iterations = 5000
            solver = SolverKamino(model, config)
            control = model.control()
            control.joint_target_q.assign(targets)
            state = _step(solver, model, control)
            efforts = state.joint_lambdas_tau.numpy()
            velocities = state.joint_qd.numpy()
            for world, target in enumerate(targets):
                with self.subTest(config=config_name, target=target):
                    # Backward PD and momentum balance solved together, then saturated.
                    effort = kp * target * inertia / (inertia + DT * kd + DT * DT * kp)
                    effort = max(-limit, min(limit, effort))
                    self.assertAlmostEqual(float(efforts[world]), effort, delta=0.02)
                    self.assertAlmostEqual(float(velocities[world]), DT * effort / inertia, delta=0.002)


class TestSolverKaminoJointEffortLimitWarmstart(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_bounded_pd_graph_replay_after_reset_and_limit_edit(self):
        """Preserve implicit saturation across captured solves, reset, and finite limit edits."""
        if not wp.get_device().is_cuda:
            self.skipTest("CUDA graph replay requires a CUDA device")
        targets = (0.2, -0.2, 2.0, -2.0)
        inertia = 1.0e-4 + _BODY_MASS * _BODY_COM_X**2 + 0.011
        for config_name, config_factory in KAMINO_CONFIGS:
            if config_name in PADMM_CONFIG_NAMES:
                continue
            model = _build_revolute(
                10.0, target_ke=10.0, target_kd=2.0, armatures=(0.011,) * len(targets), body_inertia=1.0e-4
            )
            solver = SolverKamino(model, config_factory())
            state_in, state_out = _initial_state(model), model.state()
            control = model.control()
            control.joint_target_q.assign(targets)
            solver.step(state_in, state_out, control, None, DT)
            with wp.ScopedCapture() as capture:
                solver.step(state_in, state_out, control, None, DT)
            for limit in (10.0, 0.5):
                model.joint_effort_limit.fill_(limit)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                solver.reset(state=state_in)
                wp.capture_launch(capture.graph)
                efforts = state_out.joint_lambdas_tau.numpy()
                velocities = state_out.joint_qd.numpy()
                for world, target in enumerate(targets):
                    with self.subTest(config=config_name, limit=limit, target=target):
                        effort = 10.0 * target * inertia / (inertia + DT * 2.0 + DT * DT * 10.0)
                        effort = max(-limit, min(limit, effort))
                        self.assertAlmostEqual(float(efforts[world]), effort, delta=0.02)
                        self.assertAlmostEqual(float(velocities[world]), DT * effort / inertia, delta=0.002)

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
                        model = _build_revolute(effort_limit, target_ke=target_ke, armatures=(1.0,))
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
