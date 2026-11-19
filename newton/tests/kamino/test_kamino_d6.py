# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise D6 layouts and row assembly through the Kamino backends."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.core.joints import D6JointAxes, JointCorrectionMode
from newton._src.solvers.kamino._src.kinematics.joints import d6_angular_coordinates, write_d6_joint_motion
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context
from newton.tests.kamino.utils.extract import extract_cts_jacobians, extract_dofs_jacobians
from newton.tests.unittest_utils import requires

_LINEAR = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float32)
_ANGULAR = _LINEAR[[2, 1, 0]]
_INERTIA = wp.mat33f(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


@wp.kernel
def _measure_angular_derivatives(analytic: wp.array[wp.vec3f], finite_difference: wp.array[wp.vec3f]):
    axes = D6JointAxes()
    count = wp.tid()
    axes.angular_count = count
    axes.angular = wp.identity(3, dtype=wp.float32)
    axes.third_axis_sign = -1.0
    orientation = wp.quat_from_axis_angle(wp.normalize(wp.vec3f(1.0, -2.0, 3.0)), 0.8)
    direction = wp.normalize(wp.vec3f(-3.0, 1.0, 2.0))
    epsilon = 1.0e-3
    minus = wp.quat_from_axis_angle(direction, -epsilon) * orientation
    plus = wp.quat_from_axis_angle(direction, epsilon) * orientation
    reference = wp.vec3f(0.0)
    _angles, covectors = d6_angular_coordinates(axes, orientation, reference, False)
    values_minus, _covectors_minus = d6_angular_coordinates(axes, minus, reference, False)
    values_plus, _covectors_plus = d6_angular_coordinates(axes, plus, reference, False)
    analytic[count] = wp.transpose(covectors) @ direction
    finite_difference[count] = (values_plus - values_minus) / (2.0 * epsilon)


@wp.kernel
def _measure_wrapped_angles(
    correction: int, reference: wp.array[float], result: wp.array[float], rates: wp.array[float]
):
    axes = D6JointAxes()
    axes.linear = wp.identity(3, dtype=wp.float32)
    axes.angular = wp.identity(3, dtype=wp.float32)
    axes.angular_count = 3
    axes.third_axis_sign = 1.0
    write_d6_joint_motion(
        axes,
        correction,
        0,
        0,
        wp.vec3f(0.0),
        wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), 0.1),
        wp.spatial_vectorf(0.0),
        reference,
        result,
        rates,
    )


class TestKaminoD6(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.device = wp.get_device(test_context.device)

    def test_angular_jacobians_differentiate_coordinates(self):
        """Match angular velocity covectors to finite differences away from the chart singularities."""
        analytic = wp.empty(4, dtype=wp.vec3f, device=self.device)
        finite_difference = wp.empty_like(analytic)
        wp.launch(_measure_angular_derivatives, dim=4, outputs=[analytic, finite_difference], device=self.device)
        np.testing.assert_allclose(analytic.numpy(), finite_difference.numpy(), atol=8.0e-5, rtol=0.0)

    def test_layouts_project_body_wrench(self):
        """Project independent linear and angular axes in all 16 layouts on every backend."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        layouts = list(itertools.product(range(4), repeat=2))
        wrench = np.array((1.0, 2.0, 3.0, -2.0, 1.0, 3.0), dtype=np.float32)
        dt = 0.01
        expected_twists = []
        expected_rates = []
        for linear_count, angular_count in layouts:
            builder.begin_world()
            body = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
            linear = _LINEAR[:linear_count]
            angular = _ANGULAR[:angular_count]
            joint = builder.add_joint_d6(
                -1,
                body,
                linear_axes=[builder.JointDofConfig(axis=axis, armature=0.0) for axis in linear],
                angular_axes=[builder.JointDofConfig(axis=axis, armature=0.0) for axis in angular],
            )
            builder.add_articulation([joint])
            builder.end_world()
            linear_rates = linear @ wrench[:3] * dt
            angular_rates = angular @ wrench[3:] * dt
            expected_rates.extend((*linear_rates, *angular_rates))
            expected_twists.append(np.concatenate((linear.T @ linear_rates, angular.T @ angular_rates)))

        model = builder.finalize(device=self.device)
        control = model.control()
        for backend in ("scc", "lox", "padmm"):
            with self.subTest(backend=backend):
                solver = SolverKamino(model, SolverKamino.Config(dynamics_solver=backend, use_collision_detector=False))
                state, result = model.state(), model.state()
                state.body_f.assign(np.tile(wrench, (len(layouts), 1)))
                solver.step(state, result, control, contacts=None, dt=dt)
                np.testing.assert_allclose(result.body_qd.numpy(), expected_twists, atol=4.0e-4, rtol=0.0)
                np.testing.assert_allclose(result.joint_qd.numpy(), expected_rates, atol=4.0e-4, rtol=0.0)

    def test_rotation_correction_matches_reset_and_step(self):
        """Honor the configured angular chart during reset and bound all TWOPI coordinates."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[builder.JointDofConfig(axis=(1, 0, 0))],
            angular_axes=[builder.JointDofConfig(axis=(0, 0, 1))],
        )
        builder.add_articulation([joint])
        builder.joint_q[1] = 3.2
        model = builder.finalize(device=self.device)
        for mode, expected in (("none", 3.2 - 2 * np.pi), ("continuous", 3.2), ("twopi", 3.2)):
            with self.subTest(mode=mode):
                solver = SolverKamino(
                    model, SolverKamino.Config(rotation_correction=mode, use_collision_detector=False)
                )
                state = model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver.reset(state, config=SolverKamino.ResetConfig.preserve())
                self.assertAlmostEqual(state.joint_q.numpy()[1], expected, delta=2.0e-6)
                result = model.state()
                solver.step(state, result, model.control(), contacts=None, dt=0.01)
                self.assertAlmostEqual(result.joint_q.numpy()[1], expected, delta=2.0e-6)

        reference = wp.array((2 * np.pi - 0.1, 0.0, 0.0), dtype=float, device=self.device)
        result = wp.empty_like(reference)
        rates = wp.empty_like(reference)
        for mode, expected in ((JointCorrectionMode.TWOPI, 0.1), (JointCorrectionMode.CONTINUOUS, 2 * np.pi + 0.1)):
            wp.launch(
                _measure_wrapped_angles, dim=1, inputs=[mode, reference], outputs=[result, rates], device=self.device
            )
            self.assertAlmostEqual(result.numpy()[0], expected, delta=2.0e-6)

    def test_antiparallel_hinge_axes_recover(self):
        """Reduce a 180-degree swing error with a bounded correction on every backend."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[builder.JointDofConfig(axis=(1, 0, 0))],
            angular_axes=[builder.JointDofConfig(axis=(0, 0, 1))],
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=self.device)
        dt = 0.01
        for backend in ("scc", "lox", "padmm"):
            with self.subTest(backend=backend):
                config = SolverKamino.Config(dynamics_solver=backend, use_collision_detector=False)
                if backend == "lox":
                    # Make the expected separation from antiparallel visible in one step.
                    config.lox.joint_max_correction = 0.05
                solver = SolverKamino(model, config)
                state, result = model.state(), model.state()
                state.body_q.assign(np.array(((0, 0, 0, 1, 0, 0, 0),), dtype=np.float32))
                solver.step(state, result, model.control(), contacts=None, dt=dt)
                angular_speed = np.linalg.norm(result.body_qd.numpy()[body, 3:])
                self.assertLess(angular_speed * dt, np.pi)
                rotation = wp.quatf(*result.body_q.numpy()[body, 3:])
                follower_axis = wp.quat_rotate(rotation, wp.vec3f(0, 0, 1))
                self.assertGreater(follower_axis[2], -0.9999)

    def test_d6_closes_a_kinematic_loop(self):
        """Couple two world sliders through a D6 loop and constrain only the locked motion."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        child = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        builder.add_articulation([builder.add_joint_prismatic(-1, parent, axis=(1, 0, 0))])
        builder.add_articulation([builder.add_joint_prismatic(-1, child, axis=(0, 1, 0))])
        builder.add_joint_d6(
            parent,
            child,
            linear_axes=[builder.JointDofConfig(axis=axis) for axis in _LINEAR[1:]],
            angular_axes=[builder.JointDofConfig(axis=axis) for axis in _ANGULAR],
        )
        model = builder.finalize(device=self.device, skip_validation_joints=True)
        wrench = np.array(((3, 0, 0, 0, 0, 0), (0, 2, 0, 0, 0, 0)), dtype=np.float32)
        expected = ((0, 0, 0, 0, 0, 0), (0, 0.02, 0, 0, 0, 0))
        for backend in ("scc", "lox", "padmm"):
            with self.subTest(backend=backend):
                solver = SolverKamino(model, SolverKamino.Config(dynamics_solver=backend, use_collision_detector=False))
                state, result = model.state(), model.state()
                state.body_f.assign(wrench)
                solver.step(state, result, model.control(), contacts=None, dt=0.01)
                np.testing.assert_allclose(result.body_qd.numpy(), expected, atol=4.0e-4, rtol=0.0)

    def test_lox_proximal_preserves_translation_recovery_bounds(self):
        """Respect recovery caps and dead zones when feeding back exact D6 anchor errors."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[builder.JointDofConfig(axis=(1, 0, 0))],
            angular_axes=[builder.JointDofConfig(axis=(0, 0, 1))],
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=self.device)
        for displacement, cap, dead_zone, correction in (
            (0.1, 0.01, 0.0, 0.01),
            (0.0005, 0.01, 0.001, 0.0),
            (0.1, 1.0, 0.0, 0.1),
        ):
            with self.subTest(displacement=displacement, cap=cap, dead_zone=dead_zone):
                config = SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                config.lox.joint_proximal_relaxation = 0.5
                config.lox.joint_max_correction = cap
                config.lox.position_tolerance = max(dead_zone, 1.0e-7)
                solver = SolverKamino(model, config)
                state, result = model.state(), model.state()
                pose = state.body_q.numpy()
                pose[body, 1] = displacement
                state.body_q.assign(pose)
                solver.step(state, result, model.control(), contacts=None, dt=0.01)
                self.assertAlmostEqual(result.body_q.numpy()[body, 1], displacement - correction, delta=3.0e-6)

    def test_reset_recovers_rates_from_moving_anchors(self):
        """Recover authored rates and match them to both Jacobians at separated anchors."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        child = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        root = builder.add_joint_free(parent)
        joint = builder.add_joint_d6(
            parent,
            child,
            parent_xform=wp.transformf(wp.vec3f(0.2, -0.1, 0.3), wp.quat_identity()),
            child_xform=wp.transformf(wp.vec3f(-0.4, 0.2, 0.1), wp.quat_identity()),
            linear_axes=[builder.JointDofConfig(axis=axis) for axis in _LINEAR[:2]],
            angular_axes=[builder.JointDofConfig(axis=(0.0, 0.0, 1.0))],
        )
        builder.add_articulation([root, joint])
        model = builder.finalize(device=self.device)
        state = model.state()
        coord_start = int(model.joint_q_start.numpy()[joint])
        dof_start = int(model.joint_qd_start.numpy()[joint])
        q = state.joint_q.numpy()
        qd = state.joint_qd.numpy()
        q[coord_start:] = (0.7, -0.4, 0.5)
        qd[:6] = (0.5, 0.1, 0.2, -0.4, 0.3, 0.8)
        qd[dof_start:] = (0.8, -0.3, 0.4)
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for sparse_jacobian in (False, True):
            with self.subTest(sparse_jacobian=sparse_jacobian):
                solver = SolverKamino(
                    model,
                    SolverKamino.Config(use_collision_detector=False, sparse_jacobian=sparse_jacobian),
                )
                state.joint_q.zero_()
                state.joint_qd.zero_()
                solver.reset(state, config=SolverKamino.ResetConfig.preserve())
                np.testing.assert_allclose(state.joint_q.numpy()[coord_start:], q[coord_start:], atol=2.0e-6)
                np.testing.assert_allclose(state.joint_qd.numpy()[dof_start:], qd[dof_start:], atol=2.0e-6)

                impl = solver._solver_kamino
                impl._data.bodies.q_i.assign(state.body_q)
                impl._data.bodies.u_i.assign(state.body_qd)
                impl._update_joints_data(q_j_p=state.joint_q)
                impl._update_jacobians()
                body_twists = state.body_qd.numpy().reshape(-1)
                j_dofs = extract_dofs_jacobians(solver._model_kamino, impl._jacobians)[0]
                j_cts = extract_cts_jacobians(solver._model_kamino, None, None, impl._jacobians)[0]
                np.testing.assert_allclose(j_dofs[dof_start:] @ body_twists, qd[dof_start:], atol=2.0e-6)
                np.testing.assert_allclose(j_cts @ body_twists, 0.0, atol=2.0e-6)

        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        np.testing.assert_allclose(state.joint_q.numpy()[coord_start:], q[coord_start:], atol=2.0e-6)
        np.testing.assert_allclose(state.joint_qd.numpy()[dof_start:], qd[dof_start:], atol=2.0e-6)

    def test_effort_cap_and_position_limit(self):
        """Cap a linear drive and independently enforce an angular stop on every backend."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        drive = builder.JointDofConfig(axis=(1.0, 0.0, 0.0), target_q=0.5, target_ke=100.0, effort_limit=2.0)
        stop = builder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-0.5, limit_upper=0.05)
        joint = builder.add_joint_d6(-1, body, linear_axes=[drive], angular_axes=[stop])
        builder.add_articulation([joint])
        model = builder.finalize(device=self.device)
        dt = 0.01
        for backend in ("scc", "lox", "padmm"):
            with self.subTest(backend=backend):
                solver = SolverKamino(model, SolverKamino.Config(dynamics_solver=backend, use_collision_detector=False))
                state, result = model.state(), model.state()
                state.joint_q.assign(np.array((0.0, 0.15), dtype=np.float32))
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                solver.step(state, result, model.control(), contacts=None, dt=dt)
                rates = result.joint_qd.numpy()
                self.assertAlmostEqual(float(rates[0]), 2.0 * dt, delta=3.0e-4)
                self.assertLess(float(rates[1]), -0.01)

    @requires("pxr.Usd")
    def test_usd_planar_joint_matches_builder(self):
        """Simulate the same planar joint when authored through USD and the builder."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        body = UsdGeom.Xform.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
        joint = UsdPhysics.Joint.Define(stage, "/Planar")
        joint.CreateBody1Rel().SetTargets([body.GetPath()])
        for axis, bound in (("transX", 1.0), ("transY", 1.0), ("rotZ", 90.0)):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(-bound)
            limit.CreateHighAttr(bound)
        usd_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        usd_builder.add_usd(stage)
        self.assertEqual(usd_builder.joint_type, [newton.JointType.D6])

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        child = builder.add_link(mass=1.0, inertia=_INERTIA, lock_inertia=True)
        joint_id = builder.add_joint_d6(
            -1,
            child,
            linear_axes=[builder.JointDofConfig(axis=axis, limit_lower=-1.0, limit_upper=1.0) for axis in _LINEAR[:2]],
            angular_axes=[
                builder.JointDofConfig(axis=(0.0, 0.0, 1.0), limit_lower=-np.pi / 2.0, limit_upper=np.pi / 2.0)
            ],
        )
        builder.add_articulation([joint_id])

        velocities = []
        for source in (usd_builder, builder):
            model = source.finalize(device=self.device)
            solver = SolverKamino(model, SolverKamino.Config(dynamics_solver="scc", use_collision_detector=False))
            control, state, result = model.control(), model.state(), model.state()
            control.joint_f.assign(np.array((2.0, -3.0, 1.0), dtype=np.float32))
            solver.step(state, result, control, contacts=None, dt=0.01)
            velocities.append(result.joint_qd.numpy())
        np.testing.assert_allclose(velocities[0], velocities[1], atol=2.0e-6)
        np.testing.assert_allclose(velocities[0], (0.02, -0.03, 0.01), atol=2.0e-6)
