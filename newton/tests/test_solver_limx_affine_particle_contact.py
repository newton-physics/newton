# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx import AffineBodyModel
from newton._src.solvers.limx.affine_types import mat1212, vec12
from newton._src.solvers.limx.constraints.affine_particle_contact import ConstraintAffineParticleContact


def _unit_tetrahedron() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    tetrahedra = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    surface_triangles = np.asarray(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    return vertices, tetrahedra, surface_triangles


def _make_body_model(translation: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> AffineBodyModel:
    vertices, tetrahedra, surface_triangles = _unit_tetrahedron()
    return AffineBodyModel(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=0.0,
        initial_transform=wp.transform(wp.vec3(*translation), wp.quat_identity()),
        device="cpu",
    )


def _make_particle_model(positions: np.ndarray, triangles: np.ndarray):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particles(
        pos=positions,
        vel=[wp.vec3(0.0)] * len(positions),
        mass=[1.0] * len(positions),
        radius=[0.0] * len(positions),
    )
    builder.add_triangles(triangles[:, 0], triangles[:, 1], triangles[:, 2])
    return builder.finalize(device="cpu")


def _make_contact(particle_model, body_model, **overrides) -> ConstraintAffineParticleContact:
    parameters = {
        "particle_model": particle_model,
        "body_model": body_model,
        "thickness": 0.01,
        "stiffness": 10.0,
        "normal_damping": 0.0,
        "friction": 0.0,
        "friction_epsilon": 1.0e-4,
        "max_contacts": 64,
    }
    parameters.update(overrides)
    contact = ConstraintAffineParticleContact(**parameters)
    contact.begin_step(
        particle_model.particle_q,
        particle_model.particle_qd,
        body_model.q,
        body_model.qd,
        0.01,
    )
    contact.prepare(particle_model.particle_q, body_model.q)
    return contact


def _active_count(buffer) -> int:
    return min(int(buffer.count.numpy()[0]), buffer.capacity)


def _find_contact(buffer, particle_ids: tuple[int, ...], affine_ids: tuple[int, ...]) -> int:
    count = _active_count(buffer)
    stored_particle_ids = buffer.particle_ids.numpy()[:count]
    stored_affine_ids = buffer.affine_ids.numpy()[:count]
    particle_target = tuple(sorted(particle_ids))
    affine_target = tuple(sorted(affine_ids))
    for contact in range(count):
        actual_particles = tuple(sorted(int(index) for index in stored_particle_ids[contact] if index >= 0))
        actual_affine = tuple(sorted(int(index) for index in stored_affine_ids[contact] if index >= 0))
        if actual_particles == particle_target and actual_affine == affine_target:
            return contact
    return -1


def _point_jacobian(rest_position: np.ndarray) -> np.ndarray:
    x, y, z = rest_position
    return np.asarray(
        [
            [1.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, x, y, z, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x, y, z],
        ],
        dtype=np.float64,
    )


class TestConstraintAffineParticleContactDetection(unittest.TestCase):
    def test_retains_affine_edge_and_vertex_vf_regions(self):
        """Retain cloth-vertex VF contact in affine PE and PP closest regions."""
        particle_model = _make_particle_model(
            np.asarray(
                [[0.5, -0.002, -0.002], [-0.002, -0.002, -0.002], [3.0, 3.0, 3.0]],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        contact = _make_contact(particle_model, _make_body_model())
        count = _active_count(contact.cloth_vertex_face_contacts)
        particle_ids = contact.cloth_vertex_face_contacts.particle_ids.numpy()[:count]
        affine_weights = contact.cloth_vertex_face_contacts.affine_weights.numpy()[:count]
        directions = contact.cloth_vertex_face_contacts.directions.numpy()[:count]

        edge_rows = [
            row
            for row in range(count)
            if particle_ids[row, 0] == 0 and np.count_nonzero(np.abs(affine_weights[row]) > 1.0e-6) == 2
        ]
        vertex_rows = [
            row
            for row in range(count)
            if particle_ids[row, 0] == 1 and np.count_nonzero(np.abs(affine_weights[row]) > 1.0e-6) == 1
        ]

        self.assertTrue(edge_rows)
        self.assertTrue(vertex_rows)
        np.testing.assert_allclose(
            directions[edge_rows[0]],
            np.asarray([0.0, -1.0, -1.0]) / np.sqrt(2.0),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            directions[vertex_rows[0]],
            -np.ones(3) / np.sqrt(3.0),
            atol=1.0e-6,
        )

    def test_orients_cloth_vertex_against_affine_face(self):
        """Push a crossed cloth vertex along the affine face outward normal."""
        particle_model = _make_particle_model(
            np.asarray([[0.33, 0.33, 0.338], [3.0, 3.0, 3.0], [4.0, 3.0, 3.0]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        contact = _make_contact(particle_model, _make_body_model())
        row = _find_contact(contact.cloth_vertex_face_contacts, (0,), (1, 2, 3))

        self.assertGreaterEqual(row, 0)
        expected_direction = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
        signed_distance = (0.33 + 0.33 + 0.338 - 1.0) / np.sqrt(3.0)
        np.testing.assert_allclose(
            contact.cloth_vertex_face_contacts.directions.numpy()[row],
            expected_direction,
            atol=5.0e-5,
        )
        self.assertAlmostEqual(
            float(contact.cloth_vertex_face_contacts.depths.numpy()[row]),
            0.01 - signed_distance,
            places=6,
        )

    def test_uses_affine_surface_to_orient_avf_closest_direction(self):
        """Orient the AVF closest-point direction consistently on both cloth sides."""
        positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        triangles = np.asarray([[0, 1, 2]], dtype=np.int32)
        expected_direction = np.asarray([0.0, 0.0, 1.0])

        for z in (0.002, -0.002):
            with self.subTest(z=z):
                particle_model = _make_particle_model(positions, triangles)
                contact = _make_contact(particle_model, _make_body_model((0.25, 0.25, z)))
                row = _find_contact(contact.affine_vertex_face_contacts, (0, 1, 2), (0,))

                self.assertGreaterEqual(row, 0)
                np.testing.assert_allclose(
                    contact.affine_vertex_face_contacts.directions.numpy()[row],
                    expected_direction,
                    atol=1.0e-6,
                )
                self.assertAlmostEqual(
                    float(contact.affine_vertex_face_contacts.depths.numpy()[row]),
                    0.01 - z,
                    places=6,
                )

    def test_uses_affine_surface_to_orient_strict_interior_mixed_edge_contact(self):
        """Orient the mixed EE closest separation from the affine edge pseudo-normal."""
        particle_model = _make_particle_model(
            np.asarray([[0.5, -1.0, -0.002], [0.5, 1.0, -0.002], [1.5, 1.0, -0.002]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        body_model = _make_body_model()
        contact = _make_contact(particle_model, body_model)
        row = _find_contact(contact.edge_edge_contacts, (0, 1), (0, 1))

        self.assertGreaterEqual(row, 0)
        np.testing.assert_allclose(
            contact.edge_edge_contacts.directions.numpy()[row],
            [0.0, 0.0, -1.0],
            atol=1.0e-6,
        )
        self.assertAlmostEqual(float(contact.edge_edge_contacts.depths.numpy()[row]), 0.008, places=6)

    def test_rejects_endpoint_mixed_edge_contact(self):
        """Reject mixed EE when the closest points lie on segment endpoints."""
        particle_model = _make_particle_model(
            np.asarray([[0.0, -1.0, -0.002], [0.0, 0.0, -0.002], [-1.0, 0.0, -0.002]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        contact = _make_contact(particle_model, _make_body_model())

        self.assertEqual(_find_contact(contact.edge_edge_contacts, (0, 1), (0, 1)), -1)

    def test_activates_detected_near_parallel_mixed_edge_mollifier(self):
        """Activate the IPC mollifier for a detected near-parallel mixed EE row."""
        threshold = 1.0e-3
        sine = np.sqrt(0.5 * threshold)
        cosine = np.sqrt(1.0 - sine * sine)
        particle_model = _make_particle_model(
            np.asarray(
                [
                    [0.5 - 0.5 * cosine, -0.5 * sine, -0.002],
                    [0.5 + 0.5 * cosine, 0.5 * sine, -0.002],
                    [1.5, 1.0, -0.002],
                ],
                dtype=np.float32,
            ),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        contact = _make_contact(particle_model, _make_body_model())
        row = _find_contact(contact.edge_edge_contacts, (0, 1), (0, 1))

        self.assertGreaterEqual(row, 0)
        self.assertEqual(int(contact.edge_edge_contacts.mollifier_active.numpy()[row]), 1)
        self.assertAlmostEqual(
            float(contact.edge_edge_contacts.mollifier_thresholds.numpy()[row]),
            threshold,
            places=6,
        )

    def test_counts_mixed_contact_overflow(self):
        """Count detected rows beyond capacity without writing past the buffer."""
        particle_model = _make_particle_model(
            np.asarray([[0.33, 0.33, 0.342], [0.34, 0.32, 0.342], [0.32, 0.345, 0.342]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        contact = _make_contact(particle_model, _make_body_model(), max_contacts=1)
        detected = int(contact.cloth_vertex_face_contacts.count.numpy()[0])
        overflow = int(contact.cloth_vertex_face_contacts.overflow_count.numpy()[0])

        self.assertGreater(detected, contact.cloth_vertex_face_contacts.capacity)
        self.assertEqual(overflow, detected - contact.cloth_vertex_face_contacts.capacity)


class TestConstraintAffineParticleContactOperator(unittest.TestCase):
    @staticmethod
    def _ordinary_contact_fixture():
        particle_model = _make_particle_model(
            np.asarray([[0.33, 0.33, 0.342], [3.0, 3.0, 3.0], [4.0, 3.0, 3.0]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        body_model = _make_body_model()
        contact = _make_contact(particle_model, body_model)
        row = _find_contact(contact.cloth_vertex_face_contacts, (0,), (1, 2, 3))
        if row < 0:
            raise AssertionError("Expected one cloth-vertex affine-face contact")
        return particle_model, body_model, contact, row

    @staticmethod
    def _dense_contact_jacobian(particle_model, body_model, buffer, row: int) -> np.ndarray:
        particle_count = particle_model.particle_count
        jacobian = np.zeros((3, 3 * particle_count + 12 * body_model.body_count), dtype=np.float64)
        particle_ids = buffer.particle_ids.numpy()[row]
        particle_weights = buffer.particle_weights.numpy()[row]
        for particle, weight in zip(particle_ids, particle_weights, strict=True):
            if particle >= 0:
                columns = slice(3 * int(particle), 3 * int(particle) + 3)
                jacobian[:, columns] += float(weight) * np.eye(3)
        rest_positions = body_model.rest_surface_vertices.numpy().astype(np.float64)
        ownership = body_model.surface_ownership.numpy()
        affine_ids = buffer.affine_ids.numpy()[row]
        affine_weights = buffer.affine_weights.numpy()[row]
        for vertex, weight in zip(affine_ids, affine_weights, strict=True):
            if vertex >= 0:
                body = int(ownership[vertex])
                columns = slice(3 * particle_count + 12 * body, 3 * particle_count + 12 * (body + 1))
                jacobian[:, columns] += float(weight) * _point_jacobian(rest_positions[vertex])
        return jacobian

    def test_lifts_force_full_hessian_and_exact_diagonal(self):
        """Match a dense mixed VF force, HVP, and native block diagonals."""
        particle_model, body_model, contact, row = self._ordinary_contact_fixture()
        buffer = contact.cloth_vertex_face_contacts
        jacobian = self._dense_contact_jacobian(particle_model, body_model, buffer, row)
        direction = buffer.directions.numpy()[row].astype(np.float64)
        depth = float(buffer.depths.numpy()[row])
        world_hessian = contact.stiffness * np.outer(direction, direction)
        expected_force = jacobian.T @ (contact.stiffness * depth * direction)
        expected_hessian = jacobian.T @ world_hessian @ jacobian

        particle_force = wp.zeros(particle_model.particle_count, dtype=wp.vec3, device="cpu")
        affine_force = wp.zeros(body_model.body_count, dtype=vec12, device="cpu")
        contact.accumulate_force(
            particle_model.particle_q,
            body_model.q,
            particle_force,
            affine_force,
        )
        actual_force = np.concatenate((particle_force.numpy().reshape(-1), affine_force.numpy().reshape(-1)))
        np.testing.assert_allclose(actual_force, expected_force, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(
            particle_force.numpy().sum(axis=0) + affine_force.numpy()[0, :3],
            np.zeros(3),
            atol=2.0e-6,
        )

        vector = np.linspace(-0.7, 0.9, len(expected_force), dtype=np.float32)
        particle_input = wp.array(
            vector[: 3 * particle_model.particle_count].reshape(-1, 3),
            dtype=wp.vec3,
            device="cpu",
        )
        affine_input = wp.array(
            vector[3 * particle_model.particle_count :].reshape(-1, 12),
            dtype=vec12,
            device="cpu",
        )
        particle_product = wp.zeros_like(particle_input)
        affine_product = wp.zeros_like(affine_input)
        contact.multiply(particle_input, affine_input, particle_product, affine_product)
        actual_product = np.concatenate((particle_product.numpy().reshape(-1), affine_product.numpy().reshape(-1)))
        np.testing.assert_allclose(actual_product, expected_hessian @ vector, rtol=3.0e-5, atol=3.0e-6)

        particle_diagonal = wp.zeros(particle_model.particle_count, dtype=wp.mat33, device="cpu")
        affine_diagonal = wp.zeros(body_model.body_count, dtype=mat1212, device="cpu")
        contact.accumulate_diagonal(particle_diagonal, affine_diagonal)
        for particle in range(particle_model.particle_count):
            block = expected_hessian[3 * particle : 3 * particle + 3, 3 * particle : 3 * particle + 3]
            np.testing.assert_allclose(particle_diagonal.numpy()[particle], block, rtol=3.0e-5, atol=3.0e-6)
        affine_offset = 3 * particle_model.particle_count
        np.testing.assert_allclose(
            affine_diagonal.numpy()[0],
            expected_hessian[affine_offset:, affine_offset:],
            rtol=3.0e-5,
            atol=3.0e-6,
        )

    def test_lifts_mollified_edge_residual_into_both_domains(self):
        """Match a near-parallel mixed EE residual and Gauss-Newton operator."""
        threshold = 1.0e-3
        sine = np.sqrt(0.5 * threshold)
        cosine = np.sqrt(1.0 - sine * sine)
        particle_positions = np.asarray(
            [[0.0, 0.0, 0.05], [cosine, sine, 0.05], [0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        particle_model = _make_particle_model(
            particle_positions,
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        body_model = _make_body_model()
        contact = _make_contact(particle_model, body_model, max_contacts=1)
        contact.cloth_vertex_face_contacts.count.zero_()
        contact.affine_vertex_face_contacts.count.zero_()

        buffer = contact.edge_edge_contacts
        buffer.particle_ids.assign(np.asarray([[0, 1, -1]], dtype=np.int32))
        buffer.particle_weights.assign(np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32))
        buffer.affine_ids.assign(np.asarray([[0, 1, -1]], dtype=np.int32))
        buffer.affine_weights.assign(np.asarray([[-0.5, -0.5, 0.0]], dtype=np.float32))
        buffer.directions.assign(np.asarray([[0.0, 0.0, -1.0]], dtype=np.float32))
        buffer.depths.assign(np.asarray([0.05], dtype=np.float32))
        buffer.mollifier_thresholds.assign(np.asarray([threshold], dtype=np.float32))
        buffer.mollifier_active.assign(np.asarray([1], dtype=np.int32))
        buffer.count.assign(np.asarray([1], dtype=np.int32))
        buffer.forces.zero_()
        buffer.hessians.zero_()

        affine_positions = contact.affine_positions.numpy().astype(np.float64)
        rest_positions = body_model.rest_surface_vertices.numpy().astype(np.float64)
        weights = np.asarray([-0.5, -0.5, 0.5, 0.5], dtype=np.float64)
        direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        depth = 0.05
        dof_count = 3 * particle_model.particle_count + 12

        def residual(dofs: np.ndarray) -> np.ndarray:
            particle_delta = dofs[: 3 * particle_model.particle_count].reshape(-1, 3)
            affine_delta = dofs[3 * particle_model.particle_count :]
            current = np.asarray(
                [
                    affine_positions[0] + _point_jacobian(rest_positions[0]) @ affine_delta,
                    affine_positions[1] + _point_jacobian(rest_positions[1]) @ affine_delta,
                    particle_positions[0].astype(np.float64) + particle_delta[0],
                    particle_positions[1].astype(np.float64) + particle_delta[1],
                ]
            )
            displacements = current - np.asarray(
                [affine_positions[0], affine_positions[1], particle_positions[0], particle_positions[1]],
                dtype=np.float64,
            )
            current_depth = depth - np.sum(weights[:, None] * displacements * direction)
            cross_product = np.cross(current[1] - current[0], current[3] - current[2])
            cross_squared = float(np.dot(cross_product, cross_product))
            scale = np.sqrt(2.0 * threshold - cross_squared) / threshold
            return current_depth * scale * cross_product

        jacobian = np.empty((3, dof_count), dtype=np.float64)
        epsilon = 1.0e-6
        zero = np.zeros(dof_count, dtype=np.float64)
        residual_value = residual(zero)
        for column in range(dof_count):
            offset = np.zeros(dof_count, dtype=np.float64)
            offset[column] = epsilon
            jacobian[:, column] = (residual(offset) - residual(-offset)) / (2.0 * epsilon)
        expected_force = -contact.stiffness * jacobian.T @ residual_value
        expected_hessian = contact.stiffness * jacobian.T @ jacobian

        particle_force = wp.zeros(particle_model.particle_count, dtype=wp.vec3, device="cpu")
        affine_force = wp.zeros(body_model.body_count, dtype=vec12, device="cpu")
        contact.accumulate_force(particle_model.particle_q, body_model.q, particle_force, affine_force)
        actual_force = np.concatenate((particle_force.numpy().reshape(-1), affine_force.numpy().reshape(-1)))
        np.testing.assert_allclose(actual_force, expected_force, rtol=3.0e-4, atol=3.0e-4)

        vector = np.linspace(-0.6, 0.8, dof_count, dtype=np.float32)
        particle_input = wp.array(
            vector[: 3 * particle_model.particle_count].reshape(-1, 3),
            dtype=wp.vec3,
            device="cpu",
        )
        affine_input = wp.array(
            vector[3 * particle_model.particle_count :].reshape(-1, 12),
            dtype=vec12,
            device="cpu",
        )
        particle_product = wp.zeros_like(particle_input)
        affine_product = wp.zeros_like(affine_input)
        contact.multiply(particle_input, affine_input, particle_product, affine_product)
        actual_product = np.concatenate((particle_product.numpy().reshape(-1), affine_product.numpy().reshape(-1)))
        np.testing.assert_allclose(actual_product, expected_hessian @ vector, rtol=5.0e-4, atol=5.0e-4)

        particle_diagonal = wp.zeros(particle_model.particle_count, dtype=wp.mat33, device="cpu")
        affine_diagonal = wp.zeros(body_model.body_count, dtype=mat1212, device="cpu")
        contact.accumulate_diagonal(particle_diagonal, affine_diagonal)
        for particle in range(particle_model.particle_count):
            block = expected_hessian[3 * particle : 3 * particle + 3, 3 * particle : 3 * particle + 3]
            np.testing.assert_allclose(particle_diagonal.numpy()[particle], block, rtol=5.0e-4, atol=5.0e-4)
        affine_offset = 3 * particle_model.particle_count
        np.testing.assert_allclose(
            affine_diagonal.numpy()[0],
            expected_hessian[affine_offset:, affine_offset:],
            rtol=5.0e-4,
            atol=5.0e-4,
        )

    def test_keeps_regularized_friction_finite_and_balanced(self):
        """Keep mixed friction finite with equal-and-opposite translation force."""
        particle_model = _make_particle_model(
            np.asarray([[0.33, 0.33, 0.342], [3.0, 3.0, 3.0], [4.0, 3.0, 3.0]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        body_model = _make_body_model()
        contact = ConstraintAffineParticleContact(
            particle_model,
            body_model,
            thickness=0.01,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.4,
            friction_epsilon=1.0e-4,
            max_contacts=64,
        )
        particle_velocity = wp.zeros(particle_model.particle_count, dtype=wp.vec3, device="cpu")
        particle_velocity.assign(np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
        contact.begin_step(particle_model.particle_q, particle_velocity, body_model.q, body_model.qd, 0.01)
        contact.prepare(particle_model.particle_q, body_model.q)

        particle_force = wp.zeros(particle_model.particle_count, dtype=wp.vec3, device="cpu")
        affine_force = wp.zeros(body_model.body_count, dtype=vec12, device="cpu")
        contact.accumulate_force(particle_model.particle_q, body_model.q, particle_force, affine_force)
        force = np.concatenate((particle_force.numpy().reshape(-1), affine_force.numpy().reshape(-1)))

        self.assertTrue(np.isfinite(force).all())
        self.assertGreater(float(np.linalg.norm(particle_force.numpy()[0])), 0.0)
        np.testing.assert_allclose(
            particle_force.numpy().sum(axis=0) + affine_force.numpy()[:, :3].sum(axis=0),
            np.zeros(3),
            atol=2.0e-6,
        )

    def test_requires_valid_step_lifecycle(self):
        """Reject contact assembly before a valid begin-step and prepare sequence."""
        particle_model = _make_particle_model(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        body_model = _make_body_model()
        contact = ConstraintAffineParticleContact(
            particle_model,
            body_model,
            thickness=0.01,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.0,
            friction_epsilon=1.0e-4,
            max_contacts=4,
        )

        with self.assertRaisesRegex(RuntimeError, "begin_step"):
            contact.prepare(particle_model.particle_q, body_model.q)
        with self.assertRaisesRegex(ValueError, "dt"):
            contact.begin_step(
                particle_model.particle_q,
                particle_model.particle_qd,
                body_model.q,
                body_model.qd,
                0.0,
            )
        particle_output = wp.zeros(particle_model.particle_count, dtype=wp.vec3, device="cpu")
        affine_output = wp.zeros(body_model.body_count, dtype=vec12, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            contact.accumulate_force(particle_model.particle_q, body_model.q, particle_output, affine_output)


if __name__ == "__main__":
    unittest.main()
