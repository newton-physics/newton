# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numerical checks for the coupled spatial LOX contact law."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.kamino._src.solvers.lox.spatial_contact import (
    compute_spatial_contact_residual,
    mat66f,
    prepare_spatial_contact,
    solve_spatial_contact,
    vec6f,
)
from newton.solvers import SolverKamino


@wp.kernel
def _solve(
    matrices: wp.array[mat66f],
    velocities: wp.array[vec6f],
    frictions: wp.array[wp.vec3f],
    reactions: wp.array[vec6f],
    residuals: wp.array[vec6f],
    statuses: wp.array[wp.int32],
):
    i = wp.tid()
    prepared = prepare_spatial_contact(matrices[i], frictions[i])
    result = solve_spatial_contact(prepared, velocities[i])
    reactions[i] = result.reaction
    statuses[i] = result.status
    residuals[i] = compute_spatial_contact_residual(
        prepared, result.reaction, matrices[i] * result.reaction + velocities[i]
    )


class TestLOXSpatialContact(unittest.TestCase):
    def _run(self, matrices, velocities, frictions, expected=None, *, status=0):
        for device in wp.get_devices():
            with self.subTest(device=str(device)):
                count = len(matrices)
                reactions = wp.empty(count, dtype=vec6f, device=device)
                residuals = wp.empty(count, dtype=vec6f, device=device)
                statuses = wp.empty(count, dtype=wp.int32, device=device)
                wp.launch(
                    _solve,
                    dim=count,
                    inputs=[
                        wp.array(matrices, dtype=mat66f, device=device),
                        wp.array(velocities, dtype=vec6f, device=device),
                        wp.array(frictions, dtype=wp.vec3f, device=device),
                        reactions,
                        residuals,
                        statuses,
                    ],
                    device=device,
                )
                np.testing.assert_array_equal(statuses.numpy(), status)
                if expected is not None:
                    np.testing.assert_allclose(reactions.numpy(), expected, atol=3e-5, rtol=3e-5)
                    metric_scales = np.sqrt(np.max(np.abs(matrices), axis=(1, 2)))
                    np.testing.assert_allclose(residuals.numpy() / metric_scales[:, None], 0.0, atol=3e-5)

    def test_coupled_sticking_and_sliding(self):
        """Recover constructed contact solutions with full normal and angular coupling."""
        rng = np.random.default_rng(894)
        matrices, velocities, frictions, expected = [], [], [], []
        for sliding in (False, True):
            for _ in range(24):
                factor = rng.normal(size=(6, 6))
                canonical = factor @ factor.T + 4.0 * np.eye(6)
                friction = np.array([0.7, 0.08, 0.12])
                scaling = np.array([1.0, 0.7, 0.7, 0.08, 0.12, 0.12])
                rho = rng.normal(size=6)
                rho[0] = 2.0
                rho[1:] *= (2.0 if sliding else 0.5) / np.linalg.norm(rho[1:])
                gamma = np.zeros(6)
                if sliding:
                    gamma[1:] = -3.0 * rho[1:]
                matrix = canonical / np.outer(scaling, scaling)
                reaction = scaling * rho
                free = gamma / scaling - matrix @ reaction
                if free[0] >= 0.0:
                    continue
                matrices.append(matrix)
                velocities.append(free)
                frictions.append(friction)
                expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_shared_friction_budget(self):
        """Make simultaneous sliding, spinning and rolling share one cone boundary."""
        self._run(
            [np.eye(6)],
            [[-1.0, 2.0, 0.0, 2.0, 2.0, 0.0]],
            [[1.0, 1.0, 1.0]],
            [[1.0, -1.0 / np.sqrt(3), 0.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(3), 0.0]],
        )

    def test_zero_coefficients(self):
        """Omit disabled rows including singular unused angular blocks."""
        matrices, velocities, frictions, expected = [], [], [], []
        for friction in ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]):
            scaling = np.array([1.0, friction[0], friction[0], friction[1], friction[2], friction[2]])
            active = scaling > 0
            matrix = np.diag(active.astype(float))
            free = np.array([-1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
            reaction = np.zeros(6)
            reaction[0] = 1.0
            if friction[0]:
                reaction[1] = -friction[0]
            elif friction[1]:
                reaction[3] = -friction[1]
            elif friction[2]:
                reaction[4] = -friction[2]
            matrices.append(matrix)
            velocities.append(free)
            frictions.append(friction)
            expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_separation_and_metric_scaling(self):
        """Preserve the law under scaling and return zero for separating contact."""
        matrices, velocities, frictions, expected = [], [], [], []
        for scale in (1e-8, 1.0, 1e8):
            for normal in (-1.0, 0.1):
                matrices.append(scale * np.eye(6))
                velocities.append(scale * np.array([normal, 2.0, 0.0, 0.0, 0.0, 0.0]))
                frictions.append([0.5, 0.1, 0.2])
                expected.append([1.0, -0.5, 0.0, 0.0, 0.0, 0.0] if normal < 0 else np.zeros(6))
        self._run(matrices, velocities, frictions, expected)

    def test_mixed_active_rows(self):
        """Retain coupling when only some friction families are enabled."""
        rng = np.random.default_rng(401)
        matrices, velocities, frictions, expected = [], [], [], []
        for friction in ([0.0, 0.2, 0.3], [0.7, 0.0, 0.3], [0.7, 0.2, 0.0], [0.7, 0.001, 0.002]):
            for _ in range(8):
                factor = rng.normal(size=(6, 6))
                matrix = factor @ factor.T + 5.0 * np.eye(6)
                scaling = np.array([1.0, friction[0], friction[0], friction[1], friction[2], friction[2]])
                active = scaling > 0.0
                rho = rng.normal(size=6)
                rho[~active] = 0.0
                rho[0] = 2.0
                rho[1:] *= rho[0] / np.linalg.norm(rho[1:])
                velocity = np.zeros(6)
                velocity[1:][active[1:]] = -2.0 * rho[1:][active[1:]] / scaling[1:][active[1:]]
                reaction = scaling * rho
                matrices.append(matrix)
                velocities.append(velocity - matrix @ reaction)
                frictions.append(friction)
                expected.append(reaction)
        self._run(matrices, velocities, frictions, expected)

    def test_nonmonotone_root(self):
        """Solve a strongly coupled case whose scalar root is not monotone."""
        schur = np.array([0.1, 10.0, 2.0, 2.0, 2.0])
        coupling = np.array([0.0, 2.0, 0.0, 0.0, 0.0])
        rhs = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        normal_rhs = -0.01
        matrix = np.zeros((6, 6))
        matrix[0, 0] = 1.0
        matrix[0, 1:] = coupling
        matrix[1:, 0] = coupling
        matrix[1:, 1:] = np.diag(schur) + np.outer(coupling, coupling)
        lower, upper = 0.0, 100.0
        for _ in range(90):
            alpha = 0.5 * (lower + upper)
            solution = rhs / (schur + alpha)
            value = np.linalg.norm(solution) - coupling @ solution + normal_rhs
            if value > 0.0:
                lower = alpha
            else:
                upper = alpha
        reaction = np.r_[coupling @ solution - normal_rhs, -solution]
        free = np.r_[normal_rhs, rhs + normal_rhs * coupling]
        self._run([matrix], [free], [[1.0, 1.0, 1.0]], [reaction])

    def test_unbracketed_root_status(self):
        """Report an exhausted root bracket instead of silently accepting an impulse."""
        self._run([np.eye(6)], [[-1e-30, 1.0, 0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]], status=2)

    def test_fast_path_metric_and_coefficient_scaling(self):
        """Avoid determinant underflow in the three-dimensional fast path."""
        matrices, velocities, frictions, expected = [], [], [], []
        for scale in (1e-16, 1.0, 1e16):
            matrices.append(scale * np.eye(6))
            velocities.append(scale * np.array([-1.0, 2.0, 0.0, 0.0, 0.0, 0.0]))
            frictions.append([0.5, 0.0, 0.0])
            expected.append([1.0, -0.5, 0.0, 0.0, 0.0, 0.0])
        matrices.append(np.eye(6))
        velocities.append([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        frictions.append([1e-12, 0.0, 0.0])
        expected.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._run(matrices, velocities, frictions, expected)

    def test_off_center_contact_all_schedules(self):
        """Couple the normal impulse to angular friction in an offset rigid contact."""
        initial = np.array([1.0, -0.5, -1.0, 2.0, 1.0, 3.0])
        inertia = np.array([0.004, 0.006, 0.008])
        dt = 0.001
        for device in wp.get_devices():
            for method in ("jacobi", "gauss_seidel", "apgd"):
                with self.subTest(device=str(device), method=method):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    SolverKamino.register_custom_attributes(builder)
                    builder.begin_world()
                    body = builder.add_body(
                        mass=1.0,
                        inertia=wp.mat33(*np.diag(inertia).reshape(-1)),
                        lock_inertia=True,
                        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
                    )
                    builder.body_qd[body] = wp.spatial_vector(*initial)
                    shape = newton.ModelBuilder.ShapeConfig(
                        density=0.0, margin=0.0, gap=0.05, mu=0.3, mu_torsional=0.02, mu_rolling=0.01
                    )
                    builder.add_shape_sphere(
                        body,
                        radius=0.1,
                        xform=wp.transform(wp.vec3(0.03, 0.0, 0.0), wp.quat_identity()),
                        cfg=shape,
                    )
                    builder.add_ground_plane(cfg=shape)
                    builder.end_world()
                    model = builder.finalize(device=device)
                    model.rigid_contact_max = 16
                    config = SolverKamino.Config(dynamics_solver="lox", use_collision_detector=False)
                    config.angular_velocity_damping = 0.0
                    config.lox.contact_spatial_friction = True
                    config.lox.projection_method = method
                    config.lox.max_iterations = 50
                    config.lox.projection_iterations = 5
                    config.lox.velocity_tolerance = 1e-7
                    config.lox.position_tolerance = 1e-7
                    solver = SolverKamino(model, config=config)
                    state_in, state_out = model.state(), model.state()
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state_in, contacts)
                    self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
                    solver.step(state_in, state_out, model.control(), contacts, dt=dt)
                    backend = solver._solver_kamino.solver_fd
                    self.assertFalse(backend.world_failed.numpy().any())
                    velocity = state_out.body_qd.numpy()[body]
                    self.assertTrue(np.isfinite(velocity).all())
                    problem = backend.problem
                    law = problem.contact_law.data
                    matrix = problem.contact_law.physical.mechanical.numpy()[0]
                    self.assertGreater(np.linalg.norm(matrix[0, 3:]), 1e-4)
                    linear = problem.contact_reaction.numpy()[0]
                    angular = law.angular_reaction.numpy()[0]
                    rho = np.r_[linear[2], linear[:2] / 0.3, angular / np.array([0.02, 0.01, 0.01])]
                    self.assertGreater(rho[0], 0.0)
                    self.assertGreater(np.linalg.norm(rho[1:3]), 1e-3)
                    self.assertGreater(np.linalg.norm(rho[3:]), 1e-3)
                    self.assertAlmostEqual(float(np.linalg.norm(rho[1:])), float(rho[0]), delta=2e-5)
                    self.assertLess(float(problem.world_contact_residual_max.numpy()[0]), 2e-5)
                    metric = np.r_[np.ones(3), inertia]
                    self.assertLess(np.dot(metric, velocity**2), np.dot(metric, initial**2))
                    wrench = state_out.body_f_total.numpy()[body]
                    np.testing.assert_allclose(wrench[:3], (velocity[:3] - initial[:3]) / dt, rtol=2e-5, atol=2e-3)
                    expected_torque = inertia * (velocity[3:] - initial[3:]) / dt
                    expected_torque += np.cross(initial[3:], inertia * initial[3:])
                    np.testing.assert_allclose(wrench[3:], expected_torque, rtol=2e-5, atol=2e-3)

    def test_invalid_preparation(self):
        """Report invalid active metrics and coefficients explicitly."""
        invalid = np.eye(6)
        invalid[3, 3] = -1.0
        self._run(
            [invalid, np.eye(6), np.eye(6)],
            [[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
            [[0.5, 0.1, 0.1], [-1.0, 0.1, 0.1], [0.5, np.nan, 0.1]],
            status=1,
        )


if __name__ == "__main__":
    unittest.main()
