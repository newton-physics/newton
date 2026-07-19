# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for implicit PD behavior with actuator force Jacobians."""

import types
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.actuators import Actuator, ClampingMaxEffort, ControllerPD


def _make_pd_model(kp_model: float, kd_model: float, use_actuator: bool, clamp_actuator: bool = False):
    builder = newton.ModelBuilder(gravity=0.0)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    builder.add_shape_sphere(
        body,
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False),
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=body,
        axis=newton.Axis.Z,
        target_ke=kp_model,
        target_kd=kd_model,
        armature=0.1,
        friction=0.0,
        effort_limit=1.0e9,
        velocity_limit=1.0e9,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([joint])

    model = builder.finalize()
    model.rigid_contact_max = 0
    if use_actuator:
        indices = wp.array([0], dtype=wp.uint32, device=model.device)
        clamping = None
        if clamp_actuator:
            clamping = [ClampingMaxEffort(wp.array([1.0e9], dtype=wp.float32, device=model.device))]
        model.actuators = [
            Actuator(
                indices=indices,
                controller=ControllerPD(
                    kp=wp.array([kp_model], dtype=wp.float32, device=model.device),
                    kd=wp.array([kd_model], dtype=wp.float32, device=model.device),
                ),
                clamping=clamping,
                control_target_pos_attr="joint_target_q",
                control_target_vel_attr="joint_target_qd",
            )
        ]
    return model


def _run_kamino_pd_case(
    kp: float,
    kd: float,
    dt: float,
    use_model_pd: bool,
    use_actuator: bool,
    use_actuator_jacobians: bool,
    clamp_actuator: bool = False,
):
    model = _make_pd_model(
        kp if use_model_pd else 0.0,
        kd if use_model_pd else 0.0,
        use_actuator=use_actuator,
        clamp_actuator=clamp_actuator,
    )
    config = newton.solvers.SolverKamino.Config.from_model(model)
    config.use_actuator_jacobians = use_actuator_jacobians
    config.use_collision_detector = False
    solver = newton.solvers.SolverKamino(model, config=config)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    for _ in range(3):
        control.clear(model)
        control.joint_target_q.assign(np.array([1.0], dtype=np.float32))
        control.joint_target_qd.assign(np.array([0.0], dtype=np.float32))
        if use_actuator:
            for actuator in model.actuators:
                actuator.step(state_0, control, dt=dt, write_force_jacobians=use_actuator_jacobians)
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0

    return state_0.joint_q.numpy().copy(), state_0.joint_qd.numpy().copy()


# ---------------------------------------------------------------------------
# 1. Actuator behavior
# ---------------------------------------------------------------------------


class TestImplicitPDActuators(unittest.TestCase):
    def test_actuator_jacobians(self):
        device = wp.get_device()
        indices = wp.array([0], dtype=wp.uint32, device=device)
        kp = 50.0
        kd = 5.0

        actuator = Actuator(
            indices=indices,
            controller=ControllerPD(
                kp=wp.array([kp], dtype=wp.float32, device=device),
                kd=wp.array([kd], dtype=wp.float32, device=device),
            ),
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        state = types.SimpleNamespace(
            joint_q=wp.array([0.25], dtype=wp.float32, device=device),
            joint_qd=wp.array([0.5], dtype=wp.float32, device=device),
        )
        control = newton.Control()
        control.joint_f = wp.zeros(1, dtype=wp.float32, device=device)
        control.joint_f_dq = wp.zeros(1, dtype=wp.float32, device=device)
        control.joint_f_dqd = wp.zeros(1, dtype=wp.float32, device=device)
        control.joint_target_q = wp.array([1.0], dtype=wp.float32, device=device)
        control.joint_target_qd = wp.array([0.0], dtype=wp.float32, device=device)
        control.joint_act = wp.zeros(1, dtype=wp.float32, device=device)

        control.clear()
        self.assertFalse(actuator.step(state, control, dt=0.01))
        np.testing.assert_array_equal(np.isnan(control.joint_f_dq.numpy()), [True])
        np.testing.assert_array_equal(np.isnan(control.joint_f_dqd.numpy()), [True])

        control.clear()
        self.assertTrue(actuator.step(state, control, dt=0.01, write_force_jacobians=True))
        self.assertAlmostEqual(control.joint_f_dq.numpy()[0], -kp, places=4)
        self.assertAlmostEqual(control.joint_f_dqd.numpy()[0], -kd, places=4)

        clamped_actuator = Actuator(
            indices=indices,
            controller=ControllerPD(
                kp=wp.array([kp], dtype=wp.float32, device=device),
                kd=wp.array([kd], dtype=wp.float32, device=device),
            ),
            clamping=[ClampingMaxEffort(wp.array([10.0], dtype=wp.float32, device=device))],
            control_target_pos_attr="joint_target_q",
            control_target_vel_attr="joint_target_qd",
        )
        control.clear()
        with self.assertWarnsRegex(RuntimeWarning, "cannot provide"):
            self.assertFalse(clamped_actuator.step(state, control, dt=0.01, write_force_jacobians=True))
        np.testing.assert_array_equal(np.isnan(control.joint_f_dq.numpy()), [True])
        np.testing.assert_array_equal(np.isnan(control.joint_f_dqd.numpy()), [True])


# ---------------------------------------------------------------------------
# 2. Solver behavior
# ---------------------------------------------------------------------------


class TestImplicitPDSolvers(unittest.TestCase):
    def test_kamino_cases(self):
        kp = 200.0
        kd = 20.0
        dt = 1.0 / 30.0

        implicit_q, implicit_qd = _run_kamino_pd_case(
            kp, kd, dt, use_model_pd=True, use_actuator=False, use_actuator_jacobians=False
        )
        jacobian_q, jacobian_qd = _run_kamino_pd_case(
            kp, kd, dt, use_model_pd=True, use_actuator=True, use_actuator_jacobians=True
        )
        explicit_q, explicit_qd = _run_kamino_pd_case(
            kp, kd, dt, use_model_pd=False, use_actuator=True, use_actuator_jacobians=False
        )

        np.testing.assert_allclose(jacobian_q, implicit_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(jacobian_qd, implicit_qd, rtol=1e-5, atol=1e-5)
        self.assertGreater(np.linalg.norm(explicit_q - implicit_q), 1e-4)

    def test_kamino_fallback(self):
        kp = 200.0
        kd = 20.0
        dt = 1.0 / 30.0

        explicit_q, explicit_qd = _run_kamino_pd_case(
            kp, kd, dt, use_model_pd=False, use_actuator=True, use_actuator_jacobians=False
        )
        with self.assertWarnsRegex(RuntimeWarning, "cannot provide"):
            fallback_q, fallback_qd = _run_kamino_pd_case(
                kp,
                kd,
                dt,
                use_model_pd=False,
                use_actuator=True,
                use_actuator_jacobians=True,
                clamp_actuator=True,
            )

        np.testing.assert_allclose(fallback_q, explicit_q, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fallback_qd, explicit_qd, rtol=1e-5, atol=1e-5)

    def test_kamino_warning(self):
        kp = 200.0
        kd = 20.0
        dt = 1.0 / 30.0
        model = _make_pd_model(kp, kd, use_actuator=True)
        config = newton.solvers.SolverKamino.Config.from_model(model)
        config.use_collision_detector = False
        solver = newton.solvers.SolverKamino(model, config=config)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        control.clear(model)
        control.joint_target_q.assign(np.array([1.0], dtype=np.float32))
        control.joint_target_qd.assign(np.array([0.0], dtype=np.float32))
        self.assertTrue(model.actuators[0].step(state_0, control, dt=dt, write_force_jacobians=True))

        with self.assertWarnsRegex(RuntimeWarning, "does not consume actuator force Jacobians"):
            solver.step(state_0, state_1, control, None, dt)

    def test_solver_warning(self):
        model = _make_pd_model(200.0, 20.0, use_actuator=True)
        solver = newton.solvers.SolverSemiImplicit(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        control.clear(model)
        control.joint_target_q.assign(np.array([1.0], dtype=np.float32))
        control.joint_target_qd.assign(np.array([0.0], dtype=np.float32))
        self.assertTrue(model.actuators[0].step(state_0, control, dt=0.01, write_force_jacobians=True))

        with self.assertWarnsRegex(RuntimeWarning, "does not consume actuator force Jacobians"):
            solver.step(state_0, state_1, control, None, 0.01)

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            solver.step(state_0, state_1, control, None, 0.01)
        self.assertFalse(recorded)


if __name__ == "__main__":
    unittest.main()
