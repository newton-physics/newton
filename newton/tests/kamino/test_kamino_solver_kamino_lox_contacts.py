# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise experimental LOX contact laws through the public solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverKamino
from newton.tests.kamino import setup_tests, test_context

_CPU_QR_SKIP_REASON = (
    "warp.fem.linalg.symmetric_eigenvalues_qr can return NaN eigenvalues on the CPU for denormal "
    "off-diagonal entries, which invalidates spatial contact metrics; fixed in newer Warp."
)


_PROJECTION_SCHEDULES = {
    "jacobi": {"gauss_seidel_max_colors": 1},
    "gauss_seidel": {"gauss_seidel_max_colors": 4},
    "apgd": {"gauss_seidel_max_colors": 1, "projection_acceleration": "apgd"},
    "jacobi_anderson": {"gauss_seidel_max_colors": 1, "projection_acceleration": "anderson"},
    "gauss_seidel_anderson": {"gauss_seidel_max_colors": 4, "projection_acceleration": "anderson"},
}


class TestSolverKaminoLOXContacts(unittest.TestCase):
    def setUp(self):
        if not test_context.setup_done:
            setup_tests(clear_cache=False)
        self.default_device = wp.get_device(test_context.device)

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
        model = builder.finalize(device=self.default_device)
        model.rigid_contact_max = 16 * worlds
        return model

    def _solver(self, model, *, method="jacobi", internal=False, compute_solution_metrics=False, **options):
        config = SolverKamino.Config(
            dynamics_solver="lox", use_collision_detector=internal, compute_solution_metrics=compute_solution_metrics
        )
        config.lox.max_iterations = 50
        config.lox.projection_iterations = 5
        for name, value in _PROJECTION_SCHEDULES[method].items():
            setattr(config.lox, name, value)
        config.lox.position_tolerance = 1.0e-7
        config.lox.velocity_tolerance = 1.0e-7
        for name, value in options.items():
            setattr(config.lox, name, value)
        return SolverKamino(model, config=config)

    def _collide(self, model, state, *, stiffness=None, damping=0.0):
        """Detect the contacts of ``state``, optionally with a uniform per-contact stiffness [N/m] and damping [N s/m]."""
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        if stiffness is not None:
            # Allocate the per-contact properties together, as hydroelastic contacts do
            for name, value in (
                ("rigid_contact_stiffness", stiffness),
                ("rigid_contact_damping", damping),
                ("rigid_contact_friction", 0.0),
            ):
                setattr(
                    contacts,
                    name,
                    wp.full(contacts.rigid_contact_max, value, dtype=wp.float32, device=self.default_device),
                )
        return contacts

    def _step(self, model, solver, state_in=None, *, dt=0.01, stiffness=None, damping=0.0):
        state_in = model.state() if state_in is None else state_in
        state_out = model.state()
        contacts = self._collide(model, state_in, stiffness=stiffness, damping=damping)
        solver.step(state_in, state_out, model.control(), contacts, dt=dt)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], 0)
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()).all())
        return state_out

    def test_compliance_static_deflection_all_schedules(self):
        """Maintain the analytic spring deflection under gravity for every schedule."""
        compliance = 0.002
        for method in _PROJECTION_SCHEDULES:
            for dt in (0.01, 0.02):
                with self.subTest(method=method, dt=dt):
                    model = self._sphere(gap=-10.0 * compliance)
                    solver = self._solver(model, method=method, contact_compliance=True)
                    state = self._step(model, solver, dt=dt, stiffness=1.0 / compliance)
                    self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), 0.0, delta=2.0e-4)
                    self.assertAlmostEqual(float(state.body_q.numpy()[0, 2]), 0.08, delta=1.0e-5)

    def test_compliance_without_contact_stiffness_is_hard(self):
        """Keep contacts without a positive per-contact stiffness hard when compliance is enabled."""
        model = self._sphere()
        hard = self._step(model, self._solver(model), stiffness=500.0).body_q.numpy()
        for stiffness, damping in ((None, 0.0), (0.0, 0.0), (0.0, 20.0)):
            with self.subTest(stiffness=stiffness, damping=damping):
                solver = self._solver(model, contact_compliance=True)
                state = self._step(model, solver, stiffness=stiffness, damping=damping)
                np.testing.assert_allclose(state.body_q.numpy(), hard, rtol=0.0, atol=1.0e-6)

    def test_compliant_contact_damping(self):
        """Sink by the implicit spring-damper step from rest in touching contact."""
        model = self._sphere()
        hard = self._step(model, self._solver(model)).body_q.numpy()
        for damping in (0.0, 20.0):
            with self.subTest(damping=damping):
                solver = self._solver(model, contact_compliance=True)
                state = self._step(model, solver, stiffness=500.0, damping=damping)
                # Backward Euler gives v = -m g dt / (m + k dt^2 + d dt)
                expected = 0.1 / (1.0 + 500.0 * 1.0e-4 + damping * 0.01) * 0.01
                self.assertAlmostEqual(float(hard[0, 2] - state.body_q.numpy()[0, 2]), expected, delta=1.0e-5)

    def test_compliant_contact_ignores_restitution(self):
        """Give compliant contacts the same response for any restitution coefficient."""
        for in_kernel in (False, True):
            outcomes = []
            for restitution in (0.0, 0.5):
                model = self._sphere(gap=-0.001, velocity=(0.0, 0.0, -1.0, 0.0, 0.0, 0.0), restitution=restitution)
                solver = self._solver(model, contact_compliance=True, contact_restitution=in_kernel)
                outcomes.append(self._step(model, solver, stiffness=500.0).body_qd.numpy())
            with self.subTest(contact_restitution=in_kernel):
                np.testing.assert_allclose(outcomes[1], outcomes[0], rtol=0.0, atol=1.0e-6)
                self.assertLess(float(outcomes[0][0, 2]), 0.0)

    def test_restitution_speculative_trial_and_closed_contact(self):
        """Bounce closing trials immediately while leaving open speculative trials free."""
        for method in _PROJECTION_SCHEDULES:
            for gap, velocity, expected in ((0.0, -2.0, 1.0), (0.01, -2.0, 1.0), (0.01, -0.2, -0.2)):
                with self.subTest(method=method, gap=gap, velocity=velocity):
                    model = self._sphere(
                        gap=gap, gravity=0.0, velocity=(0.0, 0.0, velocity, 0.0, 0.0, 0.0), restitution=0.5
                    )
                    solver = self._solver(model, method=method, contact_restitution=True)
                    state = self._step(model, solver)
                    self.assertAlmostEqual(float(state.body_qd.numpy()[0, 2]), expected, delta=2.0e-4)

    def test_non_finite_twist_fails_only_its_world(self):
        """Fail a world whose twists become non-finite and keep the other worlds valid."""
        model = self._sphere(worlds=2)
        solver = self._solver(model)
        state_in, state_out = model.state(), model.state()
        velocity = state_in.body_qd.numpy()
        velocity[1] = np.nan
        state_in.body_qd.assign(velocity)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, dt=0.01)
        np.testing.assert_array_equal(solver.status.numpy()["failed"], [0, 1])
        self.assertTrue(np.isfinite(state_out.body_qd.numpy()[0]).all())

    def test_failed_world_recovers_from_finite_input(self):
        """Do not warm start a failed world from its non-finite reactions."""
        for method in _PROJECTION_SCHEDULES:
            with self.subTest(method=method):
                model = self._sphere(worlds=2)
                solver = self._solver(model, method=method)
                state_in, state_out = model.state(), model.state()
                # On the CPU, the NumPy view aliases the state
                finite = state_in.body_qd.numpy().copy()
                velocity = finite.copy()
                velocity[1] = np.nan
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                for qd, failed in ((velocity, [0, 1]), (finite, [0, 0])):
                    state_in.body_qd.assign(qd)
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, model.control(), contacts, dt=0.01)
                    np.testing.assert_array_equal(solver.status.numpy()["failed"], failed)

    def test_spatial_friction_shares_one_budget(self):
        """Couple sliding, torsional, and rolling friction through one elliptic cone."""
        friction, torsional, rolling, mass, inertia, radius, dt = 0.3, 0.02, 0.01, 1.0, 0.004, 0.1, 0.01
        initial_velocity = np.array([1.0, -0.5, 0.0, 2.0, 1.0, 3.0])
        for method in _PROJECTION_SCHEDULES:
            for internal in (False, True):
                with self.subTest(method=method, internal=internal):
                    model = self._sphere(
                        velocity=initial_velocity, friction=friction, torsional=torsional, rolling=rolling
                    )
                    solver = self._solver(model, method=method, internal=internal, contact_spatial_friction=True)
                    velocity = self._step(model, solver, dt=dt).body_qd.numpy()[0]

                    # Recover the contact impulses from the momentum change, with the contact
                    # force applied at the bottom of the sphere
                    force = mass * (velocity[:3] - initial_velocity[:3]) - mass * dt * np.array([0.0, 0.0, -10.0])
                    torque = inertia * (velocity[3:] - initial_velocity[3:]) - np.cross([0.0, 0.0, -radius], force)
                    sliding_budget = np.dot(force[:2], force[:2]) / friction**2
                    angular_budget = torque[2] ** 2 / torsional**2 + np.dot(torque[:2], torque[:2]) / rolling**2
                    self.assertGreater(sliding_budget, 1.0e-5)
                    self.assertGreater(angular_budget, 1.0e-5)
                    self.assertAlmostEqual(
                        float(np.sqrt(sliding_budget + angular_budget)), float(force[2]), delta=2.0e-4
                    )
                    metric = np.array([mass, mass, mass, inertia, inertia, inertia])
                    self.assertLess(np.dot(metric, velocity**2), np.dot(metric, initial_velocity**2))

    def test_sliding_and_rolling_converge_from_the_warm_start(self):
        """Converge saturated sliding and rolling friction in about one iteration from the predicted velocity."""
        if self.default_device.is_cpu:
            self.skipTest(_CPU_QR_SKIP_REASON)
        for motion, velocity in (
            ("sliding", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ("rolling", (1.0, 0.0, 0.0, 0.0, 10.0, 0.0)),
        ):
            with self.subTest(motion=motion):
                model = self._sphere(velocity=velocity, friction=0.5, torsional=0.005, rolling=0.002)
                config = SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                config.lox.contact_spatial_friction = True
                config.lox.inertial_warmstart_fraction = 1.0
                solver = SolverKamino(model, config=config)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                state_in, state_out, control = model.state(), model.state(), model.control()
                iterations = []

                for _ in range(50):
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, 0.002)
                    state_in, state_out = state_out, state_in
                    iterations.append(int(solver.status.numpy()["iterations"][0]))

                # The first step has no warm start
                self.assertLessEqual(np.mean(iterations[1:]), 2.0)

    def test_disabled_spatial_friction_preserves_legacy_motion(self):
        """Retain the legacy response when angular material coefficients are ignored."""
        outcomes = []
        for coefficient in (0.0, 0.02):
            model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=coefficient, rolling=coefficient)
            solver = self._solver(model)
            outcomes.append(self._step(model, solver).body_qd.numpy())
        np.testing.assert_allclose(outcomes[0], outcomes[1], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outcomes[1][0, 3:], [2.0, 0.0, 3.0], rtol=0.0, atol=1.0e-6)

    def test_solution_metrics_exclude_extended_contacts(self):
        """Exclude compliant and spatial-friction contacts from the dual residuals, but not restitutive ones."""
        dual_residuals = ("r_ncp_primal", "r_ncp_dual", "r_ncp_compl", "r_vi_natmap")
        world_residuals = ("r_v_plus", "f_ncp", "f_ccp")
        for enabled in (False, True):
            with self.subTest(case="all_extended", compute_solution_metrics=enabled):
                model = self._sphere(gap=-0.01, velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01)
                solver = self._solver(
                    model, compute_solution_metrics=enabled, contact_compliance=True, contact_spatial_friction=True
                )
                self._step(model, solver, stiffness=1000.0)
                status = solver.status.numpy()
                for name in ("r_p", "r_d", "r_c"):
                    self.assertTrue(np.isfinite(status[name]).all(), name)
                    self.assertTrue((status[name] >= 0.0).all(), name)
                self.assertLess(float(status["r_c"][0]), 1.0e-8)
                if enabled:
                    metrics = solver.metrics.data
                    self.assertLess(float(metrics.r_eom.numpy()[0]), 1.0e-3)
                    for name in dual_residuals:
                        np.testing.assert_array_equal(getattr(metrics, name).numpy(), 0.0, err_msg=name)
                    for name in world_residuals:
                        self.assertTrue(np.isfinite(getattr(metrics, name).numpy()).all(), name)

        with self.subTest(case="per_contact"):
            # Only the contact of world 1 has angular friction and drops out of its dual residuals
            model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01, worlds=2)
            world_0 = model.shape_world.numpy() == 0
            for name in ("shape_material_mu_torsional", "shape_material_mu_rolling"):
                values = getattr(model, name).numpy()
                values[world_0] = 0.0
                getattr(model, name).assign(values)
            solver = self._solver(model, compute_solution_metrics=True, contact_spatial_friction=True)
            self._step(model, solver)
            metrics = solver.metrics.data
            for name in world_residuals + dual_residuals:
                self.assertTrue(np.isfinite(getattr(metrics, name).numpy()).all(), name)
            for name in dual_residuals:
                self.assertEqual(float(getattr(metrics, name).numpy()[1]), 0.0, name)

        with self.subTest(case="restitution"):
            # Restitution is a solver strategy, measured against the hard-contact problem
            model = self._sphere(gravity=0.0, velocity=(0.0, 0.0, -2.0, 0.0, 0.0, 0.0), restitution=0.5)
            solver = self._solver(model, compute_solution_metrics=True, contact_restitution=True)
            self._step(model, solver)
            metrics = solver.metrics.data
            for name in world_residuals + dual_residuals:
                self.assertTrue(np.isfinite(getattr(metrics, name).numpy()).all(), name)

    def test_spatial_contact_cuda_graph_replay(self):
        """Replay combined compliant spatial contact laws from a CUDA graph."""
        if not self.default_device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device.")
        for method in _PROJECTION_SCHEDULES:
            with self.subTest(method=method):
                model = self._sphere(gap=-0.002, velocity=(0.0, 0.0, -0.2, 1.0, 0.0, 2.0), torsional=0.02, rolling=0.01)
                solver = self._solver(
                    model,
                    method=method,
                    contact_compliance=True,
                    contact_restitution=True,
                    contact_spatial_friction=True,
                )
                state_in, state_out = model.state(), model.state()
                contacts = self._collide(model, state_in, stiffness=1000.0)
                control = model.control()
                solver.step(state_in, state_out, control, contacts, dt=0.01)
                expected = state_out.body_qd.numpy()
                with wp.ScopedCapture(device=self.default_device) as capture:
                    solver.step(state_in, state_out, control, contacts, dt=0.01)
                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state_out.body_qd.numpy(), expected, rtol=0.0, atol=2.0e-4)

    def test_selective_reset_cold_starts_reset_worlds(self):
        """Solve a reset world exactly as a new solver would, whatever its previous contact impulses."""
        for spatial in (False, True):
            with self.subTest(contact_spatial_friction=spatial):
                model = self._sphere(
                    velocity=(1.0, -0.5, 0.0, 2.0, 1.0, 3.0), friction=0.3, torsional=0.02, rolling=0.01, worlds=2
                )
                solver = self._solver(model, contact_spatial_friction=spatial)
                state_in, state_out, control = model.state(), model.state(), model.control()
                for _ in range(3):
                    contacts = self._collide(model, state_in)
                    solver.step(state_in, state_out, control, contacts, dt=0.01)
                    state_in, state_out = state_out, state_in

                mask = wp.array([True, False, False], dtype=wp.bool, device=self.default_device)
                solver.reset(state_in, world_mask=mask)
                contacts = self._collide(model, state_in)
                results = []
                for candidate in (solver, self._solver(model, contact_spatial_friction=spatial)):
                    candidate.step(state_in, state_out, control, contacts, dt=0.01)
                    results.append((state_out.body_qd.numpy()[0], int(candidate.status.numpy()["iterations"][0])))
                np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=1.0e-6)
                self.assertEqual(results[0][1], results[1][1])

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

    def test_spinning_sphere_stops_at_torsional_friction_angle(self):
        """Converge the torsional friction of a spinning sphere to its discrete stopping angle."""
        if self.default_device.is_cpu:
            self.skipTest(_CPU_QR_SKIP_REASON)
        spin, torsional, gravity, inertia, time_step = 3.0, 0.003, 10.0, 0.004, 1.0 / 240.0
        model = self._sphere(velocity=(0.0, 0.0, 0.0, 0.0, 0.0, spin), friction=0.5, torsional=torsional)
        config = SolverKamino.Config(dynamics_solver="lox")
        config.lox.contact_spatial_friction = True
        solver = SolverKamino(model, config=config)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out, control = model.state(), model.state(), model.control()

        for _ in range(int(0.5 / time_step)):
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, time_step)
            state_in, state_out = state_out, state_in

        # The contact point does not slide, so the torque mu_t m g alone decelerates the spin by a
        # constant step per time step. The angular friction impulse must converge at the velocity
        # tolerance, not at the looser rotation tolerance of the body-space residuals
        deceleration = torsional * gravity / inertia
        steps = np.floor(spin / (deceleration * time_step) + 1.0e-6)
        expected = time_step * (steps * spin - deceleration * time_step * steps * (steps + 1.0) / 2.0)
        orientation = state_in.body_q.numpy()[0, 3:]
        angle = 2.0 * np.arctan2(orientation[2], orientation[3])
        self.assertAlmostEqual(float(state_in.body_qd.numpy()[0, 5]), 0.0, delta=1.0e-3)
        self.assertAlmostEqual(angle, expected, delta=5.0e-3 * expected)

    def test_disappearing_contact_clears_angular_impulses(self):
        """Discard spatial contact impulses after a body separates from the plane."""
        model = self._sphere(velocity=(0.0, 0.0, 0.0, 2.0, 0.0, 3.0), torsional=0.02, rolling=0.01)
        solver = self._solver(model, contact_spatial_friction=True)
        state_in = self._step(model, solver)
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


if __name__ == "__main__":
    # Test setup
    setup_tests()

    # Run all tests
    unittest.main(verbosity=2)
