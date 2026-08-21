# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for Coulomb joint friction in SolverKamino."""

import unittest

import numpy as np
import warp as wp

import newton
import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton.tests.kamino import setup_tests, test_context

_BODY_MASS = 1.0
_BODY_PRINCIPAL_INERTIA = 1.0
_BODY_COM_X = 0.5
_EFFECTIVE_JOINT_INERTIA = _BODY_PRINCIPAL_INERTIA + _BODY_MASS * _BODY_COM_X**2
DT = 0.01


def _build_revolute(
    friction: float | None,
    *,
    armature: float = 0.0,
    limit: tuple[float, float] | None = None,
    target_ke: float = 0.0,
    target_kd: float = 0.0,
) -> newton.Model:
    """Build a single world-to-body revolute model."""
    builder = newton.ModelBuilder()
    SolverKamino.register_custom_attributes(builder)
    builder.begin_world()
    body = builder.add_link(
        mass=_BODY_MASS,
        inertia=[
            _BODY_PRINCIPAL_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_PRINCIPAL_INERTIA,
            0.0,
            0.0,
            0.0,
            _BODY_PRINCIPAL_INERTIA,
        ],
        com=wp.vec3f(_BODY_COM_X, 0.0, 0.0),
        lock_inertia=True,
    )
    joint = builder.add_joint_revolute(
        -1,
        body,
        axis=newton.Axis.Y,
        friction=friction,
        armature=armature,
        limit_lower=limit[0] if limit else None,
        limit_upper=limit[1] if limit else None,
        target_ke=target_ke,
        target_kd=target_kd,
        actuator_mode=newton.JointTargetMode.POSITION if target_ke > 0.0 else newton.JointTargetMode.NONE,
    )
    builder.add_articulation([joint])
    builder.end_world()
    model = builder.finalize()
    return model


def _dense_config(**kwargs) -> SolverKamino.Config:
    """Create a dense PADMM configuration."""
    return SolverKamino.Config(
        dynamics_solver="padmm",
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
        **kwargs,
    )


def _sparse_config(**kwargs) -> SolverKamino.Config:
    """Create a PADMM configuration with sparse dynamics."""
    return SolverKamino.Config(
        dynamics_solver="padmm",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        dynamics=kamino_config.ConstrainedDynamicsConfig(linear_solver_type="CR"),
        **kwargs,
    )


def _dvi_dense_config(**kwargs) -> SolverKamino.Config:
    """Create a dense DVI configuration."""
    return SolverKamino.Config(
        dynamics_solver="dvi",
        use_fk_solver=False,
        sparse_jacobian=False,
        sparse_dynamics=False,
        use_collision_detector=False,
        **kwargs,
    )


def _dvi_sparse_config(**kwargs) -> SolverKamino.Config:
    """Create a DVI configuration with the matrix-free sparse solve path."""
    return SolverKamino.Config(
        dynamics_solver="dvi",
        use_fk_solver=False,
        sparse_jacobian=True,
        sparse_dynamics=True,
        use_collision_detector=False,
        **kwargs,
    )


_KAMINO_CONFIGS = (
    ("dense", _dense_config),
    ("sparse", _sparse_config),
    ("dvi_dense", _dvi_dense_config),
    ("dvi_sparse", _dvi_sparse_config),
)

# DVI has no `use_acceleration` knob, so only PADMM configs vary over it.
_PADMM_CONFIG_NAMES = ("dense", "sparse")


def _initialize_state(model: newton.Model, q: float = 0.0, qd: float = 0.0) -> newton.State:
    """Initialize generalized and maximal state consistently."""
    model.joint_q.assign([q])
    model.joint_qd.assign([qd])
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return state


def _rollout(
    solver: SolverKamino,
    model: newton.Model,
    state_in: newton.State,
    *,
    steps: int,
    dt: float,
    control: newton.Control | None = None,
    joint_force: float | None = None,
) -> tuple[newton.State, list[float], list[float], list[float]]:
    """Advance a model and record its revolute coordinate, velocity, and friction torque."""
    state_out = model.state()
    if joint_force is not None:
        control = model.control() if control is None else control
        control.joint_f.assign([joint_force])
    coordinates = []
    velocities = []
    friction_torques = []
    for _ in range(steps):
        solver.step(state_in, state_out, control=control, contacts=None, dt=dt)
        state_in, state_out = state_out, state_in
        coordinates.append(float(state_in.joint_q.numpy()[0]))
        velocities.append(float(state_in.joint_qd.numpy()[0]))
        friction_torques.append(float(solver._solver_kamino.data.joints.lambda_f_j.numpy()[0]))
    return state_in, coordinates, velocities, friction_torques


def _run_spin_down_test(
    test_case: unittest.TestCase,
    config_name: str,
    config: SolverKamino.Config,
) -> None:
    """Verify spin-down behavior for a Kamino configuration."""
    friction = 2.0
    for armature in (0.0, 1.0):
        for direction in (-1.0, 1.0):
            with test_case.subTest(
                config=config_name,
                armature=armature,
                use_acceleration=config.padmm.use_acceleration,
                direction=direction,
            ):
                initial_velocity = direction
                model = _build_revolute(friction, armature=armature)
                model.set_gravity((0.0, 0.0, 0.0))
                solver = SolverKamino(model, config)

                decrement = DT * friction / (_EFFECTIVE_JOINT_INERTIA + armature)
                spin_down_steps = int(np.ceil(abs(initial_velocity) / decrement))
                expected = direction * np.maximum(
                    abs(initial_velocity) - decrement * np.arange(1, spin_down_steps + 1),
                    0.0,
                )

                state = _initialize_state(model, qd=initial_velocity)
                state, _, velocities, _ = _rollout(solver, model, state, steps=spin_down_steps, dt=DT)

                np.testing.assert_allclose(velocities, expected, atol=3.0e-4, rtol=0.0)
                np.testing.assert_allclose(velocities[-1], 0.0, atol=1.0e-6)

                stopped_q = float(state.joint_q.numpy()[0])
                state, coordinates, velocities, _ = _rollout(solver, model, state, steps=20, dt=DT)
                np.testing.assert_allclose(velocities, 0.0, atol=1.0e-6)
                np.testing.assert_allclose(coordinates, stopped_q, atol=1.0e-6)


def _run_hold_and_breakaway_test(
    test_case: unittest.TestCase,
    config_name: str,
    config: SolverKamino.Config,
) -> None:
    """Verify static friction holding and breakaway."""
    friction = 2.0
    for armature in (0.0, 1.0):
        for direction in (-1.0, 1.0):
            for control_type in ("feedforward", "PD"):
                with test_case.subTest(
                    config=config_name,
                    armature=armature,
                    use_acceleration=config.padmm.use_acceleration,
                    direction=direction,
                    control_type=control_type,
                ):
                    kp = 10.0 if control_type == "PD" else 0.0
                    kd = 1.0 if control_type == "PD" else 0.0
                    model = _build_revolute(friction, armature=armature, target_ke=kp, target_kd=kd)
                    model.set_gravity((0.0, 0.0, 0.0))
                    solver = SolverKamino(model, config)
                    state = _initialize_state(model)
                    control = model.control()

                    def set_control_from_desired_torque(
                        desired_torque: float,
                        *,
                        control_type: str = control_type,
                        control=control,
                        kp: float = kp,
                    ) -> None:
                        if control_type == "feedforward":
                            control.joint_f.assign([desired_torque])
                        else:
                            control.joint_target_q.assign([desired_torque / kp])

                    num_levels = 4
                    for applied_torque in (direction * i * friction / num_levels for i in range(1, num_levels)):
                        set_control_from_desired_torque(applied_torque)
                        state, _, velocities, friction_torques = _rollout(
                            solver, model, state, steps=1, dt=DT, control=control
                        )
                        test_case.assertAlmostEqual(friction_torques[-1], -applied_torque, delta=1.0e-3)
                        test_case.assertAlmostEqual(velocities[-1], 0.0, delta=1e-6)

                    set_control_from_desired_torque(direction * (friction + 10.0))
                    _, _, velocities, friction_torques = _rollout(
                        solver, model, state, steps=10, dt=DT, control=control
                    )
                    np.testing.assert_allclose(friction_torques, -direction * friction, atol=2.0e-4, rtol=0.0)
                    test_case.assertTrue(all(direction * velocity > 0.0 for velocity in velocities[1:]))


class TestSolverKaminoJointFriction(unittest.TestCase):
    def setUp(self):
        """Initialize the shared public Kamino test context."""
        if not test_context.setup_done:
            setup_tests(clear_cache=False)

    def test_convert_newton_joint_friction(self):
        """Map positive friction to a bounded row and no friction to none."""
        for friction, num_friction_cts in ((2.0, 1), (0.0, 0), (None, 0)):
            with self.subTest(friction=friction):
                model = _build_revolute(friction)
                solver = SolverKamino(model, _dense_config())
                kamino = solver._model_kamino
                self.assertIs(kamino.joints.f_j, model.joint_friction)
                self.assertEqual(kamino.size.sum_of_num_dynamic_joint_cts, 0)
                self.assertEqual(kamino.size.sum_of_num_friction_joint_cts, num_friction_cts)
                self.assertEqual(kamino.size.sum_of_num_bounded_joint_cts, num_friction_cts)
                np.testing.assert_array_equal(kamino.info.num_joint_friction_cts.numpy(), [num_friction_cts])
                np.testing.assert_array_equal(kamino.info.num_joint_bounded_cts.numpy(), [num_friction_cts])
                if num_friction_cts:
                    self.assertEqual(
                        int(kamino.info.joint_friction_cts_group_offset.numpy()[0]),
                        int(kamino.info.num_joint_bilateral_cts.numpy()[0]),
                    )
                    np.testing.assert_array_equal(
                        kamino.info.joint_friction_cts_offset.numpy(),
                        kamino.info.joint_bounded_cts_offset.numpy(),
                    )
                    np.testing.assert_array_equal(
                        kamino.info.joint_friction_cts_group_offset.numpy(),
                        kamino.info.joint_bounded_cts_group_offset.numpy(),
                    )

    def test_ignore_free_joint_friction(self):
        """Ignore friction values attached to a free joint."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        builder.begin_world()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])
        builder.end_world()
        model = builder.finalize()
        model.joint_friction.fill_(1.0)
        with self.assertLogs("root", level="WARNING") as logs:
            solver = SolverKamino(model, _dense_config())
        self.assertTrue(any("Ignoring joint friction on FREE joint" in record.getMessage() for record in logs.records))
        self.assertEqual(solver._model_kamino.size.sum_of_num_friction_joint_cts, 0)
        self.assertEqual(solver._model_kamino.size.sum_of_num_bounded_joint_cts, 0)

    def test_multiworld_sparse_friction_offsets(self):
        """Place sparse friction rows in their owning world's bounded group."""
        builder = newton.ModelBuilder()
        SolverKamino.register_custom_attributes(builder)
        for _ in range(2):
            builder.begin_world()
            body = builder.add_link(
                mass=_BODY_MASS,
                inertia=[
                    _BODY_PRINCIPAL_INERTIA,
                    0.0,
                    0.0,
                    0.0,
                    _BODY_PRINCIPAL_INERTIA,
                    0.0,
                    0.0,
                    0.0,
                    _BODY_PRINCIPAL_INERTIA,
                ],
            )
            joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Y, friction=1.0)
            builder.add_articulation([joint])
            builder.end_world()
        model = builder.finalize()
        solver = SolverKamino(model, _sparse_config())
        kamino = solver._model_kamino

        friction_prefix = kamino.info.joint_friction_cts_offset.numpy()
        friction_group = kamino.info.joint_friction_cts_group_offset.numpy()
        total_prefix = kamino.info.total_cts_offset.numpy()
        friction_offset = kamino.joints.friction_cts_offset.numpy()
        friction_total_offset = kamino.joints.friction_cts_offset_total_cts.numpy()
        joint_world = kamino.joints.wid.numpy()
        self.assertListEqual(friction_prefix.tolist(), [0, 1])
        for jid, wid in enumerate(joint_world):
            self.assertEqual(
                friction_total_offset[jid],
                total_prefix[wid] + friction_group[wid] + friction_offset[jid] - friction_prefix[wid],
            )

        sparse_jacobians = solver._solver_kamino._jacobians
        expected_rows = kamino.info.num_joint_bilateral_cts.numpy() + kamino.info.num_joint_bounded_cts.numpy()
        self.assertTrue(np.all(sparse_jacobians._J_cts.bsm.max_dims.numpy()[:, 0] >= expected_rows))
        world_max_inequalities = (
            kamino.info.num_joint_bounded_cts.numpy()
            + kamino.info.max_limits.numpy()
            + kamino.info.max_contacts.numpy()
        )
        self.assertEqual(kamino.size.sum_of_max_inequalities, int(np.sum(world_max_inequalities)))
        self.assertEqual(kamino.size.max_of_max_inequalities, int(np.max(world_max_inequalities)))
        np.testing.assert_array_equal(
            kamino.info.inequalities_offset.numpy(),
            np.cumsum(np.concatenate(([0], world_max_inequalities[:-1]))),
        )

        state_in = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver.step(state_in, model.state(), control=None, contacts=None, dt=DT)
        np.testing.assert_array_equal(sparse_jacobians._J_cts.bsm.dims.numpy()[:, 0], expected_rows)

    def test_spin_down(self):
        """Match linear spin-down and settle without creep absent a dynamic row.

        Sets a revolute joint with zero gravity to an initial velocity and verifies that it spins down linearly.
        Test covers armatures with and without a dynamic joint constraint.
        The accelerated / non-accelerated PADMM since their implementation use their own specialized kernels.
        """
        for config_name, config_factory in _KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in _PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                config.padmm.use_acceleration = use_acceleration
                _run_spin_down_test(self, config_name, config)

    def test_hold_and_breakaway(self):
        """Hold below the friction limit and break away above it.

        Test both PD and feedforward control.
        Test covers with and without armatures to cover combination with dynamic joint constraint.
        The accelerated / non-accelerated PADMM since their implementation use their own specialized kernels.
        """
        for config_name, config_factory in _KAMINO_CONFIGS:
            use_acceleration_options = (True, False) if config_name in _PADMM_CONFIG_NAMES else (False,)
            for use_acceleration in use_acceleration_options:
                config = config_factory()
                config.padmm.use_acceleration = use_acceleration
                _run_hold_and_breakaway_test(self, config_name, config)

    def test_limit_stops_motion_with_bounded_friction(self):
        """Test that friction is well behaved when interacting with limits.

        Drive the joint into a limit with initial velocity and feedforward
        torque, and verify that friction remains within its bounds. Runs
        against every Kamino dynamics-solver configuration.
        """
        limit = 0.1
        friction = 0.5
        feedforward_torque = 1.0
        initial_velocity = 2.0
        for config_name, config_factory in _KAMINO_CONFIGS:
            with self.subTest(config=config_name):
                model = _build_revolute(friction, limit=(-limit, limit))
                model.set_gravity((0.0, 0.0, 0.0))
                solver = SolverKamino(model, config_factory())
                state = _initialize_state(model, qd=initial_velocity)
                _, coordinates, velocities, friction_torques = _rollout(
                    solver,
                    model,
                    state,
                    steps=500,
                    dt=0.001,
                    joint_force=feedforward_torque,
                )
                self.assertLessEqual(max(coordinates), limit + 1e-2)
                self.assertAlmostEqual(coordinates[-1], limit, delta=1.0e-4)
                self.assertAlmostEqual(velocities[-1], 0.0, delta=1e-4)
                np.testing.assert_array_less(
                    np.abs(friction_torques),
                    friction + 1.0e-4,
                )

    def test_box_natural_map_metrics(self):
        """Include bounded friction rows in iteration-info and solution natural maps."""
        model = _build_revolute(friction=2.0)
        model.set_gravity((0.0, 0.0, 0.0))
        solver = SolverKamino(
            model,
            _dense_config(collect_solver_info=True, compute_solution_metrics=True),
        )
        state_in = _initialize_state(model, qd=1.0)
        solver.step(state_in, model.state(), control=None, contacts=None, dt=DT)

        solver_impl = solver._solver_kamino
        solver_fd = solver_impl.solver_fd
        metrics = solver_impl.metrics
        self.assertIsNotNone(metrics)
        self.assertEqual(int(solver_impl.problem_fd.data.nbc.numpy()[0]), 1)

        problem_data = solver_impl.problem_fd.data
        bid = 0
        row = int(problem_data.vio.numpy()[0] + problem_data.bcgo.numpy()[0] + bid)
        bound_idx = int(problem_data.bcio.numpy()[0] + bid)
        lower = float(problem_data.P.numpy()[row] * problem_data.bound_lower.numpy()[bound_idx])
        upper = float(problem_data.P.numpy()[row] * problem_data.bound_upper.numpy()[bound_idx])

        iterations = int(solver_fd.data.status.numpy()[0]["iterations"])
        info_idx = int(solver_fd.data.info.offsets.numpy()[0]) + iterations - 1
        lambda_info = float(solver_fd.data.info.lambdas.numpy()[row])
        v_aug_info = float(solver_fd.data.info.v_aug.numpy()[row])
        expected_info = abs(lambda_info - np.clip(lambda_info - v_aug_info, lower, upper))
        self.assertAlmostEqual(float(solver_fd.data.info.r_ncp_natmap.numpy()[info_idx]), expected_info, places=6)

        lambda_solution = float(solver_fd.data.solution.lambdas.numpy()[row])
        v_aug_metrics = float(metrics._buffer_v.numpy()[row])
        expected_metrics = abs(lambda_solution - np.clip(lambda_solution - v_aug_metrics, lower, upper))
        self.assertAlmostEqual(float(metrics.data.r_vi_natmap.numpy()[0]), expected_metrics, places=6)


if __name__ == "__main__":
    setup_tests()
    unittest.main(verbosity=2)
