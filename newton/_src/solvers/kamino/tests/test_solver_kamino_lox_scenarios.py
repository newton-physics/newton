# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Canonical multi-step scenarios for the LOX Kamino backend."""

import unittest

import numpy as np
import warp as wp

import newton._src.solvers.kamino.config as kamino_config
from newton._src.solvers.kamino._src.core.model import ModelKamino
from newton._src.solvers.kamino._src.geometry.detector import CollisionDetector
from newton._src.solvers.kamino._src.solver_kamino_impl import SolverKaminoImpl
from newton._src.solvers.kamino.solver_kamino import SolverKamino
from newton._src.solvers.kamino.tests import setup_tests, test_context
from newton.tests.utils.basics import (
    build_box_on_plane,
    build_box_pendulum,
    build_box_pendulum_vertical,
    build_boxes_hinged,
)


class TestSolverKaminoLOXScenarios(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(device="cpu", clear_cache=False)
        self.device = wp.get_device(test_context.device)

    @staticmethod
    def make_config(*, solve_tightly: bool = True) -> SolverKamino.Config:
        tolerance = 1.0e-9 if solve_tightly else 1.0e-5
        return SolverKamino.Config(
            dynamics_solver="lox",
            lox=kamino_config.LOXSolverConfig(
                max_iterations=25,
                projection_iterations=10,
                position_tolerance=tolerance,
                rotation_tolerance=tolerance,
            ),
        )

    def test_free_fall_matches_discrete_solution(self):
        model = ModelKamino.from_newton(build_box_on_plane(z_offset=0.5, ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        initial_position = state_previous.q_i.numpy()[0, :3].copy()
        time_step = 0.005
        step_count = 80

        for _ in range(step_count):
            solver.step(state_previous, state_next, control, dt=time_step)
            state_previous, state_next = state_next, state_previous

        gravity = model.gravity.vector.numpy()[0]
        acceleration = gravity
        expected_velocity = step_count * time_step * acceleration
        expected_position = initial_position + 0.5 * step_count * (step_count + 1) * time_step**2 * acceleration
        np.testing.assert_allclose(state_previous.u_i.numpy()[0, :3], expected_velocity, rtol=2.0e-5, atol=2.0e-5)
        np.testing.assert_allclose(state_previous.q_i.numpy()[0, :3], expected_position, rtol=2.0e-5, atol=2.0e-5)

    def test_default_tolerance_free_fall_uses_direct_unconstrained_solve(self):
        model = ModelKamino.from_newton(build_box_on_plane(z_offset=0.5, ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config(solve_tightly=False))
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        time_step = 0.005
        step_count = 80

        for _ in range(step_count):
            solver.step(state_previous, state_next, control, dt=time_step)
            state_previous, state_next = state_next, state_previous

        gravity = model.gravity.vector.numpy()[0]
        expected_velocity = step_count * time_step * gravity
        velocity_error = float(np.linalg.norm(state_previous.u_i.numpy()[0, :3] - expected_velocity))
        self.assertLess(velocity_error, 3.0e-4, msg=f"default-tolerance velocity error: {velocity_error:.6g} m/s")

    def test_resting_box_supports_weight_without_drift(self):
        model = ModelKamino.from_newton(build_box_on_plane(ground=True).finalize(device=self.device))
        detector = CollisionDetector(model, config=kamino_config.CollisionDetectorConfig(pipeline="unified"))
        contacts = detector.contacts
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        rest_height = float(state_previous.q_i.numpy()[0, 2])
        height_history = []
        speed_history = []
        normal_force_history = []

        for _ in range(200):
            solver.step(
                state_previous,
                state_next,
                control,
                contacts=contacts,
                detector=detector,
                dt=0.01,
            )
            state_previous, state_next = state_next, state_previous
            height_history.append(float(state_previous.q_i.numpy()[0, 2]))
            speed_history.append(float(np.linalg.norm(state_previous.u_i.numpy()[0, :3])))
            active_contact_count = int(contacts.model_active_contacts.numpy()[0])
            normal_force_history.append(float(np.sum(contacts.reaction.numpy()[:active_contact_count, 2])))

        late_heights = np.asarray(height_history[-50:])
        late_speeds = np.asarray(speed_history[-50:])
        late_normal_forces = np.asarray(normal_force_history[-50:])
        self.assertLess(
            float(np.max(np.abs(late_heights - rest_height))),
            2.0e-3,
            msg=f"late height range: [{late_heights.min():.6g}, {late_heights.max():.6g}] m",
        )
        self.assertLess(
            float(np.max(late_speeds)),
            2.0e-2,
            msg=f"maximum late translational speed: {late_speeds.max():.6g} m/s",
        )
        self.assertAlmostEqual(float(np.mean(late_normal_forces)), 9.81, delta=0.15)

    def test_dropped_box_settles_after_impact(self):
        model = ModelKamino.from_newton(build_box_on_plane(z_offset=0.5, ground=True).finalize(device=self.device))
        detector = CollisionDetector(model, config=kamino_config.CollisionDetectorConfig(pipeline="unified"))
        contacts = detector.contacts
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        rest_height = 0.1
        height_history = []
        speed_history = []
        had_contact = False

        for _ in range(400):
            solver.step(
                state_previous,
                state_next,
                control,
                contacts=contacts,
                detector=detector,
                dt=0.01,
            )
            state_previous, state_next = state_next, state_previous
            height_history.append(float(state_previous.q_i.numpy()[0, 2]))
            speed_history.append(float(np.linalg.norm(state_previous.u_i.numpy()[0, :3])))
            had_contact = had_contact or int(contacts.model_active_contacts.numpy()[0]) > 0

        self.assertTrue(
            had_contact,
            msg=(
                f"no contact; max spatial speed={max(speed_history):.6g}, "
                f"final pose={state_previous.q_i.numpy()[0].tolist()}, "
                f"final joint position={solver.data.joints.q_j.numpy().tolist()}"
            ),
        )
        late_heights = np.asarray(height_history[-100:])
        late_speeds = np.asarray(speed_history[-100:])
        self.assertLess(
            float(np.max(np.abs(late_heights - rest_height))),
            5.0e-3,
            msg=f"late height range: [{late_heights.min():.6g}, {late_heights.max():.6g}] m",
        )
        self.assertLess(
            float(np.max(late_speeds)),
            5.0e-2,
            msg=f"maximum late translational speed: {late_speeds.max():.6g} m/s",
        )

    def test_vertical_pendulum_remains_at_rest(self):
        model = ModelKamino.from_newton(build_box_pendulum_vertical(ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        initial_pose = state_previous.q_i.numpy().copy()
        speed_history = []
        residual_history = []

        for _ in range(200):
            solver.step(state_previous, state_next, control, dt=0.01)
            state_previous, state_next = state_next, state_previous
            speed_history.append(float(np.linalg.norm(state_previous.u_i.numpy()[0])))
            residual_history.append(float(np.max(np.abs(solver.data.joints.r_j.numpy()), initial=0.0)))
            self.assertFalse(bool(solver.solver_fd.world_failed.numpy()[0]))

        late_speeds = np.asarray(speed_history[-50:])
        late_residuals = np.asarray(residual_history[-50:])
        pose_error = float(np.max(np.abs(state_previous.q_i.numpy() - initial_pose)))
        self.assertLess(pose_error, 2.0e-3, msg=f"final pose component error: {pose_error:.6g}")
        self.assertLess(float(np.max(late_speeds)), 2.0e-2)
        self.assertLess(float(np.max(late_residuals)), 2.0e-3)

    def test_horizontal_pendulum_preserves_joint_constraint(self):
        model = ModelKamino.from_newton(build_box_pendulum(ground=False).finalize(device=self.device))
        solver = SolverKaminoImpl(model=model, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        residual_history = []

        for _ in range(400):
            solver.step(state_previous, state_next, control, dt=0.005)
            state_previous, state_next = state_next, state_previous
            residual_history.append(float(np.max(np.abs(solver.data.joints.r_j.numpy()), initial=0.0)))
            self.assertFalse(bool(solver.solver_fd.world_failed.numpy()[0]))

        late_residuals = np.asarray(residual_history[-100:])
        self.assertTrue(np.isfinite(state_previous.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_previous.u_i.numpy()).all())
        self.assertLess(
            float(np.max(late_residuals)),
            5.0e-3,
            msg=f"maximum late structural residual: {late_residuals.max():.6g} m or rad",
        )

    def test_hinged_boxes_resting_contacts_preserve_joint_constraint(self):
        model = ModelKamino.from_newton(build_boxes_hinged(ground=True).finalize(device=self.device))
        detector = CollisionDetector(model, config=kamino_config.CollisionDetectorConfig(pipeline="unified"))
        contacts = detector.contacts
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        residual_history = []
        speed_history = []
        had_contact = False

        for _ in range(300):
            solver.step(
                state_previous,
                state_next,
                control,
                contacts=contacts,
                detector=detector,
                dt=0.005,
            )
            state_previous, state_next = state_next, state_previous
            residual_history.append(float(np.max(np.abs(solver.data.joints.r_j.numpy()), initial=0.0)))
            speed_history.append(float(np.linalg.norm(state_previous.u_i.numpy()[0])))
            had_contact = had_contact or int(contacts.model_active_contacts.numpy()[0]) > 0
            self.assertFalse(bool(solver.solver_fd.world_failed.numpy()[0]))

        self.assertTrue(
            had_contact,
            msg=(
                f"no contact; max spatial speed={max(speed_history):.6g}, "
                f"final pose={state_previous.q_i.numpy()[0].tolist()}, "
                f"final joint position={solver.data.joints.q_j.numpy().tolist()}"
            ),
        )
        late_residuals = np.asarray(residual_history[-100:])
        late_speeds = np.asarray(speed_history[-100:])
        self.assertTrue(np.isfinite(state_previous.q_i.numpy()).all())
        self.assertTrue(np.isfinite(state_previous.u_i.numpy()).all())
        self.assertLess(
            float(np.max(late_residuals)),
            5.0e-3,
            msg=f"maximum late structural residual: {late_residuals.max():.6g} m or rad",
        )
        self.assertLess(
            float(np.max(late_speeds)),
            1.0e-1,
            msg=f"maximum late spatial speed: {late_speeds.max():.6g}",
        )

    def test_hinged_boxes_drop_settles_with_bounded_joint_error(self):
        model = ModelKamino.from_newton(build_boxes_hinged(z_offset=0.5, ground=True).finalize(device=self.device))
        detector = CollisionDetector(model, config=kamino_config.CollisionDetectorConfig(pipeline="unified"))
        contacts = detector.contacts
        solver = SolverKaminoImpl(model=model, contacts=contacts, config=self.make_config())
        state_previous = model.state()
        state_next = model.state()
        control = model.control()
        residual_history = []
        speed_history = []
        height_history = []
        had_contact = False

        for _ in range(600):
            solver.step(
                state_previous,
                state_next,
                control,
                contacts=contacts,
                detector=detector,
                dt=0.005,
            )
            state_previous, state_next = state_next, state_previous
            residual_history.append(float(np.max(np.abs(solver.data.joints.r_j.numpy()), initial=0.0)))
            speed_history.append(float(np.max(np.linalg.norm(state_previous.u_i.numpy(), axis=1), initial=0.0)))
            height_history.append(state_previous.q_i.numpy()[:, 2].copy())
            had_contact = had_contact or int(contacts.model_active_contacts.numpy()[0]) > 0
            self.assertFalse(bool(solver.solver_fd.world_failed.numpy()[0]))

        self.assertTrue(had_contact)
        late_residuals = np.asarray(residual_history[-100:])
        late_speeds = np.asarray(speed_history[-100:])
        late_heights = np.asarray(height_history[-100:])
        self.assertLess(
            float(np.max(late_residuals)),
            5.0e-3,
            msg=f"maximum late structural residual: {late_residuals.max():.6g} m or rad",
        )
        self.assertLess(
            float(np.max(late_speeds)),
            1.0e-1,
            msg=f"maximum late spatial speed: {late_speeds.max():.6g}",
        )
        self.assertLess(
            float(np.max(np.abs(late_heights - 0.05))),
            1.0e-2,
            msg=f"late body height range: [{late_heights.min():.6g}, {late_heights.max():.6g}] m",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
