# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise experimental LOX contact laws through the public solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverKamino


class TestSolverKaminoLOXContacts(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device()

    def _sphere(
        self,
        *,
        gap=0.0,
        velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        gravity=-10.0,
        friction=0.0,
        torsional=0.0,
        rolling=0.0,
        restitution=0.0,
        worlds=1,
    ):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
        SolverKamino.register_custom_attributes(builder)
        shape = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=0.0,
            gap=0.05,
            mu=friction,
            mu_torsional=torsional,
            mu_rolling=rolling,
            restitution=restitution,
        )
        for _ in range(worlds):
            builder.begin_world()
            body = builder.add_body(
                mass=1.0,
                inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
                lock_inertia=True,
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.1 + gap), wp.quat_identity()),
            )
            builder.body_qd[body] = wp.spatial_vector(*velocity)
            builder.add_shape_sphere(body, radius=0.1, cfg=shape)
            builder.add_ground_plane(cfg=shape)
            builder.end_world()
        model = builder.finalize(device=self.device)
        model.rigid_contact_max = 16 * worlds
        return model

    def _solver(self, model, *, method="jacobi", internal=False, compute_solution_metrics=False, **options):
        config = SolverKamino.Config(
            dynamics_solver="lox", use_collision_detector=internal, compute_solution_metrics=compute_solution_metrics
        )
        config.angular_velocity_damping = 0.0
        config.lox.max_iterations = 50
        config.lox.projection_iterations = 5
        config.lox.projection_method = method
        config.lox.position_tolerance = 1.0e-7
        config.lox.velocity_tolerance = 1.0e-7
        for name, value in options.items():
            setattr(config.lox, name, value)
        return SolverKamino(model, config=config)

    def _step(self, model, solver, state_in=None, *, dt=0.01):
        state_in = model.state() if state_in is None else state_in
        state_out = model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver.step(state_in, state_out, model.control(), contacts, dt=dt)
        backend = solver._solver_kamino.solver_fd
        self.assertFalse(backend.world_failed.numpy().any())
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())
        return state_out

    def test_compliance_static_deflection_all_schedules(self):
        """Maintain the analytic spring deflection under gravity for every schedule."""
        compliance = 0.002
        for method in ("jacobi", "gauss_seidel", "apgd"):
            for dt in (0.01, 0.02):
                with self.subTest(method=method, dt=dt):
                    model = self._sphere(gap=-10.0 * compliance)
                    solver = self._solver(model, method=method, contact_compliance=compliance)
                    state = self._step(model, solver, dt=dt)
                    self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), 0.0, delta=2.0e-4)
                    self.assertAlmostEqual(float(state.body_q.numpy()[0, 2]), 0.08, delta=1.0e-5)

    def test_restitution_speculative_trial_and_closed_contact(self):
        """Bounce closing trials immediately while leaving open speculative trials free."""
        for method in ("jacobi", "gauss_seidel", "apgd"):
            for gap, velocity, expected in ((0.0, -2.0, 1.0), (0.01, -2.0, 1.0), (0.01, -0.2, -0.2)):
                with self.subTest(method=method, gap=gap, velocity=velocity):
                    model = self._sphere(
                        gap=gap, gravity=0.0, velocity=(0.0, 0.0, velocity, 0.0, 0.0, 0.0), restitution=0.5
                    )
                    solver = self._solver(model, method=method, contact_restitution=True)
                    state = self._step(model, solver)
                    self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), expected, delta=2.0e-4)

    def test_spatial_friction_reduces_spin_and_roll(self):
        """Dissipate spinning and rolling motion within the shared friction budget."""
        for method in ("jacobi", "gauss_seidel", "apgd"):
            for internal in (False, True):
                with self.subTest(method=method, internal=internal):
                    model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01)
                    solver = self._solver(model, method=method, internal=internal, contact_spatial_friction=True)
                    state = self._step(model, solver)
                    velocity = state.body_qd.numpy()[0]
                    self.assertGreater(velocity[3], 0.0)
                    self.assertLess(velocity[3], 1.99)
                    self.assertGreater(velocity[5], 0.0)
                    self.assertLess(velocity[5], 2.99)
                    problem = solver._solver_kamino.solver_fd.problem
                    law = problem.contact_law.data
                    angular = law.angular_reaction.numpy()[0]
                    normal = problem.contact_reaction.numpy()[0, 2]
                    scaled_norm = np.linalg.norm(angular / np.array([0.02, 0.01, 0.01]))
                    self.assertAlmostEqual(float(scaled_norm), float(normal), delta=2.0e-4)

    def test_disabled_spatial_friction_preserves_legacy_motion(self):
        """Retain the legacy response when angular material coefficients are ignored."""
        outcomes = []
        for coefficient in (0.0, 0.02):
            model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=coefficient, rolling=coefficient)
            solver = self._solver(model)
            outcomes.append(self._step(model, solver).body_qd.numpy())
        np.testing.assert_allclose(outcomes[0], outcomes[1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outcomes[1][0, 3:], [2.0, 0.0, 3.0], rtol=0.0, atol=1.0e-6)

    def test_extended_contact_metrics_use_canonical_law(self):
        """Report the extended contact residual and mark unsupported hard-contact metrics unavailable."""
        for enabled in (False, True):
            with self.subTest(compute_solution_metrics=enabled):
                model = self._sphere(gap=-0.01, velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01)
                solver = self._solver(
                    model, compute_solution_metrics=enabled, contact_compliance=0.001, contact_spatial_friction=True
                )
                self._step(model, solver)
                status = solver.status.numpy()
                self.assertTrue(np.isfinite(status["r_contact"]).all())
                self.assertLess(float(status["r_contact"][0]), 1.0e-4)
                for name in ("r_p", "r_d", "r_c"):
                    self.assertTrue(np.isnan(status[name]).all(), name)
                if enabled:
                    metrics = solver.metrics.data
                    self.assertLess(float(metrics.r_eom.numpy()[0]), 1.0e-3)
                    for name in ("r_v_plus", "r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap"):
                        self.assertTrue(np.isnan(getattr(metrics, name).numpy()).all(), name)
                        np.testing.assert_array_equal(getattr(metrics, f"{name}_argmax").numpy(), -1)
                    for name in ("f_ncp", "f_ccp"):
                        self.assertTrue(np.isnan(getattr(metrics, name).numpy()).all(), name)

    def test_sliding_rolling_and_spin_share_one_budget(self):
        """Couple translational sliding and angular friction through one cone."""
        initial_velocity = np.array([1.0, -0.5, 0.0, 2.0, 1.0, 3.0])
        for method in ("jacobi", "gauss_seidel", "apgd"):
            with self.subTest(method=method):
                model = self._sphere(velocity=initial_velocity, friction=0.3, torsional=0.02, rolling=0.01)
                solver = self._solver(model, method=method, contact_spatial_friction=True)
                state = self._step(model, solver)
                problem = solver._solver_kamino.solver_fd.problem
                linear = problem.contact_reaction.numpy()[0]
                angular = problem.contact_law.data.angular_reaction.numpy()[0]
                sliding_budget = np.dot(linear[:2], linear[:2]) / 0.3**2
                angular_budget = np.sum((angular / np.array([0.02, 0.01, 0.01])) ** 2)
                self.assertGreater(sliding_budget, 1.0e-5)
                self.assertGreater(angular_budget, 1.0e-5)
                self.assertAlmostEqual(float(np.sqrt(sliding_budget + angular_budget)), float(linear[2]), delta=2.0e-4)
                velocity = state.body_qd.numpy()[0]
                metric = np.array([1.0, 1.0, 1.0, 0.004, 0.004, 0.004])
                self.assertLess(np.dot(metric, velocity**2), np.dot(metric, initial_velocity**2))
                np.testing.assert_allclose(
                    state.body_f_total.numpy()[0, 3:],
                    0.004 * (velocity[3:] - initial_velocity[3:]) / 0.01,
                    rtol=0.0,
                    atol=2.0e-4,
                )

    def test_spatial_contact_cuda_graph_replay(self):
        """Replay combined compliant spatial contact laws from a CUDA graph."""
        if not self.device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        for method in ("jacobi", "gauss_seidel", "apgd"):
            with self.subTest(method=method):
                model = self._sphere(gap=-0.002, velocity=(0.0, 0.0, -0.2, 1.0, 0.0, 2.0), torsional=0.02, rolling=0.01)
                solver = self._solver(
                    model,
                    method=method,
                    contact_compliance=0.001,
                    contact_restitution=True,
                    contact_spatial_friction=True,
                )
                state_in, state_out = model.state(), model.state()
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                control = model.control()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, dt=0.01)
                expected = state_out.body_qd.numpy()
                with wp.ScopedCapture(device=self.device) as capture:
                    solver.step(state_in, state_out, control, contacts, dt=0.01)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=0.0, atol=2.0e-4)

    def test_spatial_contact_partial_reset(self):
        """Clear angular impulses only in the world selected for reset."""
        model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01, worlds=2)
        solver = self._solver(model, contact_spatial_friction=True)
        state = self._step(model, solver)
        problem = solver._solver_kamino.solver_fd.problem
        before = problem.contact_law.data.angular_reaction.numpy()
        worlds = problem.contact_world.numpy()
        self.assertGreater(np.linalg.norm(before[worlds == 0]), 0.0)
        self.assertGreater(np.linalg.norm(before[worlds == 1]), 0.0)
        mask = wp.array([True, False, False], dtype=wp.bool, device=self.device)
        solver.reset(state, world_mask=mask, flags=newton.StateFlags.BODY_QD)
        after = problem.contact_law.data.angular_reaction.numpy()
        np.testing.assert_array_equal(after[worlds == 0], 0.0)
        np.testing.assert_array_equal(after[worlds == 1], before[worlds == 1])

    def test_spatial_contact_permutation(self):
        """Keep each world's angular response when collision records change order."""
        model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01, worlds=2)
        solver = self._solver(model, contact_spatial_friction=True)
        state_in, state_out = model.state(), model.state()
        velocities = state_in.body_qd.numpy()
        velocities[1, 3:] *= -2.0
        state_in.body_qd.assign(velocities)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        control = model.control()
        pipeline.collide(state_in, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
        solver.step(state_in, state_out, control, contacts, dt=0.01)
        expected = state_out.body_qd.numpy()
        for name in (
            "point_id",
            "shape0",
            "shape1",
            "point0",
            "point1",
            "offset0",
            "offset1",
            "normal",
            "margin0",
            "margin1",
        ):
            array = getattr(contacts, f"rigid_contact_{name}")
            values = array.numpy()
            values[:2] = values[:2][::-1]
            array.assign(values)
        solver.step(state_in, state_out, control, contacts, dt=0.01)
        np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=0.0, atol=2.0e-4)

    def test_disappearing_contact_clears_angular_impulses(self):
        """Discard spatial contact impulses after a body separates from the plane."""
        model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01)
        solver = self._solver(model, contact_spatial_friction=True)
        state_in = self._step(model, solver)
        problem = solver._solver_kamino.solver_fd.problem
        self.assertGreater(np.linalg.norm(problem.contact_law.data.angular_reaction.numpy()), 0.0)
        positions = state_in.body_q.numpy()
        positions[0, 2] = 1.0
        state_in.body_q.assign(positions)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), contacts, dt=0.01)
        np.testing.assert_allclose(state_out.body_qd.numpy()[0, 3:], state_in.body_qd.numpy()[0, 3:], atol=1.0e-6)
        np.testing.assert_array_equal(problem.contact_law.data.angular_reaction.numpy(), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
