# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.limx import AffineBodyModel
from newton._src.solvers.limx.affine_types import mat1212, vec12
from newton._src.solvers.limx.constraints.affine_body_contact import ConstraintAffineBodyContact


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


def _two_body_model(translation: tuple[float, float, float], device: str = "cpu") -> AffineBodyModel:
    vertices, tetrahedra, surface_triangles = _unit_tetrahedron()
    return AffineBodyModel.from_instances(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=0.0,
        initial_transforms=[
            wp.transform_identity(),
            wp.transform(wp.vec3(*translation), wp.quat_identity()),
        ],
        device=device,
    )


def _near_parallel_model(device: str = "cpu") -> AffineBodyModel:
    vertices, tetrahedra, surface_triangles = _unit_tetrahedron()
    angle = 0.01
    translation = (
        0.5 - 0.5 * np.cos(angle),
        -0.5 * np.sin(angle),
        0.002,
    )
    return AffineBodyModel.from_instances(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=0.0,
        initial_transforms=[
            wp.transform_identity(),
            wp.transform(
                wp.vec3(*translation),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
            ),
        ],
        device=device,
    )


def _make_contact(model: AffineBodyModel, **overrides) -> ConstraintAffineBodyContact:
    parameters = {
        "body_model": model,
        "thickness": 0.01,
        "stiffness": 10.0,
        "normal_damping": 0.5,
        "friction": 0.2,
        "friction_epsilon": 1.0e-4,
        "max_contacts": 256,
    }
    parameters.update(overrides)
    return ConstraintAffineBodyContact(**parameters)


def _active_rows(buffer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(int(buffer.count.numpy()[0]), buffer.capacity)
    return (
        buffer.ids.numpy()[:count],
        buffer.weights.numpy()[:count],
        buffer.depths.numpy()[:count],
    )


def _find_row(ids: np.ndarray, first: int, others: tuple[int, ...]) -> int:
    target = tuple(sorted(others))
    for row, contact_ids in enumerate(ids):
        if int(contact_ids[0]) == first and tuple(sorted(int(value) for value in contact_ids[1:])) == target:
            return row
    return -1


def _find_edge_row(ids: np.ndarray, edge_0: tuple[int, int], edge_1: tuple[int, int]) -> int:
    target_0 = tuple(sorted(edge_0))
    target_1 = tuple(sorted(edge_1))
    for row, contact_ids in enumerate(ids):
        actual_0 = tuple(sorted(int(value) for value in contact_ids[:2]))
        actual_1 = tuple(sorted(int(value) for value in contact_ids[2:]))
        if (actual_0, actual_1) == (target_0, target_1) or (actual_0, actual_1) == (target_1, target_0):
            return row
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


class TestConstraintAffineBodyContactDetection(unittest.TestCase):
    def test_detects_vf_interior_edge_and_vertex_regions(self):
        """Retain triangle interior, edge, and vertex closest-point VF contacts."""
        placements = (
            ((0.33, 0.33, 0.342), 0),
            ((0.5, 0.5, -0.002), 1),
            ((1.002, -0.002, -0.002), 2),
        )

        for translation, expected_zero_weights in placements:
            with self.subTest(translation=translation):
                model = _two_body_model(translation)
                contact = _make_contact(model)
                contact.begin_step(model.q, model.qd, 0.01)
                contact.prepare(model.q)
                ids, weights, depths = _active_rows(contact.vertex_face_contacts)
                row = _find_row(ids, 4, (1, 2, 3))

                self.assertGreaterEqual(row, 0)
                self.assertGreater(depths[row], 0.0)
                zero_weights = int(np.count_nonzero(np.abs(weights[row, 1:]) < 1.0e-5))
                self.assertEqual(zero_weights, expected_zero_weights)

    def test_detects_only_cross_body_stencils(self):
        """Reject every VF and EE stencil whose features share one affine body."""
        model = _two_body_model((0.002, 0.0, 0.0))
        contact = _make_contact(model)
        contact.begin_step(model.q, model.qd, 0.01)
        contact.prepare(model.q)
        ownership = model.surface_ownership.numpy()
        vf_ids, _vf_weights, _vf_depths = _active_rows(contact.vertex_face_contacts)
        ee_ids, _ee_weights, _ee_depths = _active_rows(contact.edge_edge_contacts)

        self.assertGreater(len(vf_ids) + len(ee_ids), 0)
        self.assertTrue(np.all(ownership[vf_ids[:, 0]] != ownership[vf_ids[:, 1]]))
        self.assertTrue(np.all(ownership[ee_ids[:, 0]] != ownership[ee_ids[:, 2]]))

    def test_accepts_strict_interior_ee_and_rejects_endpoint_ee(self):
        """Accept EE only when both closest parameters lie strictly inside."""
        interior_model = _two_body_model((0.25, -0.5, 0.002))
        interior_contact = _make_contact(interior_model)
        interior_contact.begin_step(interior_model.q, interior_model.qd, 0.01)
        interior_contact.prepare(interior_model.q)
        interior_ids, interior_weights, _depths = _active_rows(interior_contact.edge_edge_contacts)
        row = _find_edge_row(interior_ids, (0, 1), (4, 6))

        self.assertGreaterEqual(row, 0)
        self.assertGreater(interior_weights[row, 1], 0.0)
        self.assertLess(interior_weights[row, 1], 1.0)
        self.assertGreater(-interior_weights[row, 3], 0.0)
        self.assertLess(-interior_weights[row, 3], 1.0)

        endpoint_model = _two_body_model((0.0, -0.5, 0.002))
        endpoint_contact = _make_contact(endpoint_model)
        endpoint_contact.begin_step(endpoint_model.q, endpoint_model.qd, 0.01)
        endpoint_contact.prepare(endpoint_model.q)
        endpoint_ids, _weights, _depths = _active_rows(endpoint_contact.edge_edge_contacts)

        self.assertEqual(_find_edge_row(endpoint_ids, (0, 1), (4, 6)), -1)

    def test_counts_contact_buffer_overflow(self):
        """Count excess contacts without writing beyond fixed buffer capacity."""
        model = _two_body_model((0.002, 0.0, 0.0))
        contact = _make_contact(model, max_contacts=1)
        contact.begin_step(model.q, model.qd, 0.01)
        contact.prepare(model.q)

        overflow = int(contact.vertex_face_contacts.overflow_count.numpy()[0])
        overflow += int(contact.edge_edge_contacts.overflow_count.numpy()[0])
        self.assertGreater(overflow, 0)

    def test_rejects_invalid_contact_configuration(self):
        """Reject invalid affine contact models, coefficients, and capacities."""
        model = _two_body_model((2.0, 0.0, 0.0))
        one_body_vertices, one_body_tets, one_body_surface = _unit_tetrahedron()
        one_body_model = AffineBodyModel(
            one_body_vertices,
            one_body_tets,
            one_body_surface,
            density=1.0,
            rigidity=0.0,
            initial_transform=wp.transform_identity(),
            device="cpu",
        )
        cases = [
            ({"body_model": object()}, "body_model"),
            ({"body_model": one_body_model}, "two affine bodies"),
            ({"thickness": 0.0}, "thickness"),
            ({"stiffness": np.nan}, "stiffness"),
            ({"normal_damping": -1.0}, "normal_damping"),
            ({"friction": -1.0}, "friction"),
            ({"friction_epsilon": 0.0}, "friction_epsilon"),
            ({"max_contacts": 0}, "max_contacts"),
        ]

        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex((TypeError, ValueError), message):
                _make_contact(model, **overrides)


class TestConstraintAffineBodyContactOperator(unittest.TestCase):
    def test_lifts_vf_force_full_hessian_and_exact_diagonal(self):
        """Lift one VF response into the complete two-body affine operator."""
        model = _two_body_model((0.33, 0.33, 0.342))
        contact = _make_contact(
            model,
            stiffness=10.0,
            normal_damping=2.0,
            friction=0.5,
            friction_epsilon=0.01,
        )
        velocity_values = np.zeros((2, 12), dtype=np.float32)
        velocity_values[1, :3] = [0.2, -0.1, -0.5]
        velocities = wp.array(velocity_values, dtype=vec12, device="cpu")
        input_values = np.linspace(-0.6, 0.8, 24, dtype=np.float32).reshape(2, 12)
        affine_input = wp.array(input_values, dtype=vec12, device="cpu")
        force = wp.zeros(2, dtype=vec12, device="cpu")
        product = wp.zeros_like(force)
        diagonal = wp.zeros(2, dtype=mat1212, device="cpu")
        empty_particles = wp.empty(0, dtype=wp.vec3, device="cpu")
        dt = 0.1

        contact.begin_step(model.q, velocities, dt)
        contact.prepare(model.q)
        contact.accumulate_force(model.q, force)
        contact.multiply(empty_particles, affine_input, empty_particles, product)
        contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device="cpu"), diagonal)

        ids, weights, depths = _active_rows(contact.vertex_face_contacts)
        self.assertEqual(len(ids), 1)
        self.assertEqual(min(int(contact.edge_edge_contacts.count.numpy()[0]), contact.max_contacts), 0)
        rest_positions = model.rest_surface_vertices.numpy().astype(np.float64)
        ownership = model.surface_ownership.numpy()
        body_jacobians = np.zeros((2, 3, 12), dtype=np.float64)
        for particle, weight in zip(ids[0], weights[0], strict=True):
            body_jacobians[ownership[particle]] += float(weight) * _point_jacobian(rest_positions[particle])

        direction = contact.vertex_face_contacts.directions.numpy()[0].astype(np.float64)
        depth = float(depths[0])
        relative_velocity = sum(body_jacobians[body] @ velocity_values[body] for body in range(2))
        normal_velocity = float(direction @ relative_velocity)
        tangent = np.eye(3) - np.outer(direction, direction)
        tangent_displacement = dt * tangent @ relative_velocity
        tangent_length = float(np.linalg.norm(tangent_displacement))
        inverse_length = 1.0 / tangent_length if tangent_length > 0.01 else (2.0 - tangent_length / 0.01) / 0.01
        alpha = 0.5 * 10.0 * depth * inverse_length
        world_force = 10.0 * depth * direction - alpha * tangent_displacement
        world_hessian = 10.0 * np.outer(direction, direction) + alpha * tangent
        if normal_velocity < 0.0:
            world_force -= 2.0 * normal_velocity * direction
            world_hessian += 2.0 / dt * np.outer(direction, direction)

        expected_force = np.stack([jacobian.T @ world_force for jacobian in body_jacobians])
        dense_hessian = np.block(
            [
                [body_jacobians[row].T @ world_hessian @ body_jacobians[column] for column in range(2)]
                for row in range(2)
            ]
        )
        expected_product = (dense_hessian @ input_values.reshape(-1)).reshape(2, 12)
        expected_diagonal = np.stack([jacobian.T @ world_hessian @ jacobian for jacobian in body_jacobians])

        np.testing.assert_allclose(force.numpy(), expected_force, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(product.numpy(), expected_product, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(diagonal.numpy(), expected_diagonal, rtol=3.0e-5, atol=3.0e-6)
        np.testing.assert_allclose(force.numpy()[:, :3].sum(axis=0), np.zeros(3), atol=2.0e-6)
        np.testing.assert_allclose(dense_hessian, dense_hessian.T, atol=1.0e-10)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(dense_hessian)[0]), -1.0e-9)

    def test_regularizes_friction_and_requires_prepared_lifecycle(self):
        """Keep small-slip friction finite and require prepared affine buffers."""
        model = _two_body_model((0.33, 0.33, 0.342))
        contact = _make_contact(model, normal_damping=0.0, friction=0.5, friction_epsilon=0.01)
        velocity_values = np.zeros((2, 12), dtype=np.float32)
        velocity_values[1, :3] = [1.0e-5, -1.0e-5, 0.0]
        velocities = wp.array(velocity_values, dtype=vec12, device="cpu")
        force = wp.zeros(2, dtype=vec12, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "prepare"):
            contact.accumulate_force(model.q, force)
        contact.begin_step(model.q, velocities, 0.1)
        contact.prepare(model.q)
        contact.accumulate_force(model.q, force)

        force_values = force.numpy()
        self.assertTrue(np.isfinite(force_values).all())
        self.assertLess(float(np.sum(force_values * velocity_values)), 0.0)

    def test_lifts_near_parallel_ee_mollifier(self):
        """Lift the EE mollifier residual and full Gauss-Newton operator."""
        model = _near_parallel_model()
        contact = _make_contact(
            model,
            thickness=0.0025,
            stiffness=7.0,
            normal_damping=0.0,
            friction=0.0,
        )
        input_values = np.linspace(-0.4, 0.7, 24, dtype=np.float32).reshape(2, 12)
        affine_input = wp.array(input_values, dtype=vec12, device="cpu")
        force = wp.zeros(2, dtype=vec12, device="cpu")
        product = wp.zeros_like(force)
        diagonal = wp.zeros(2, dtype=mat1212, device="cpu")
        empty_particles = wp.empty(0, dtype=wp.vec3, device="cpu")

        contact.begin_step(model.q, model.qd, 0.1)
        contact.prepare(model.q)
        contact.accumulate_force(model.q, force)
        contact.multiply(empty_particles, affine_input, empty_particles, product)
        contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device="cpu"), diagonal)

        rest_positions = model.rest_surface_vertices.numpy().astype(np.float64)
        ownership = model.surface_ownership.numpy()
        base_positions = contact.positions.numpy().astype(np.float64)
        base_states = model.q.numpy().astype(np.float64)
        expected_force = np.zeros(24, dtype=np.float64)
        expected_hessian = np.zeros((24, 24), dtype=np.float64)

        def contact_body_jacobian(contact_ids, contact_weights):
            jacobian = np.zeros((3, 24), dtype=np.float64)
            for particle, weight in zip(contact_ids, contact_weights, strict=True):
                body = ownership[particle]
                jacobian[:, 12 * body : 12 * (body + 1)] += float(weight) * _point_jacobian(rest_positions[particle])
            return jacobian

        vf_ids, vf_weights, vf_depths = _active_rows(contact.vertex_face_contacts)
        vf_directions = contact.vertex_face_contacts.directions.numpy()[: len(vf_ids)].astype(np.float64)
        for ids, weights, direction, depth in zip(
            vf_ids,
            vf_weights,
            vf_directions,
            vf_depths,
            strict=True,
        ):
            jacobian = contact_body_jacobian(ids, weights)
            expected_force += jacobian.T @ (7.0 * float(depth) * direction)
            expected_hessian += 7.0 * jacobian.T @ np.outer(direction, direction) @ jacobian

        ee_ids, ee_weights, ee_depths = _active_rows(contact.edge_edge_contacts)
        ee_directions = contact.edge_edge_contacts.directions.numpy()[: len(ee_ids)].astype(np.float64)
        thresholds = contact.edge_edge_contacts.mollifier_thresholds.numpy()[: len(ee_ids)].astype(np.float64)
        expected_active = []
        epsilon = 2.0e-5

        def residual(
            flat_states,
            contact_ids,
            contact_base_positions,
            contact_depth,
            contact_weights,
            contact_direction,
            contact_threshold,
        ):
            states = flat_states.reshape(2, 12)
            current = np.stack(
                [_point_jacobian(rest_positions[particle]) @ states[ownership[particle]] for particle in contact_ids]
            )
            displacement = current - contact_base_positions
            current_depth = contact_depth - float(np.sum(contact_weights[:, None] * displacement * contact_direction))
            edge_0 = current[1] - current[0]
            edge_1 = current[3] - current[2]
            cross_product = np.cross(edge_0, edge_1)
            cross_squared = float(cross_product @ cross_product)
            beta = np.sqrt(max(2.0 * contact_threshold - cross_squared, 0.0)) / contact_threshold
            return current_depth * beta * cross_product

        for ids, weights, direction, depth, threshold in zip(
            ee_ids,
            ee_weights,
            ee_directions,
            ee_depths,
            thresholds,
            strict=True,
        ):
            base_contact_positions = base_positions[ids]
            base_cross = np.cross(
                base_contact_positions[1] - base_contact_positions[0],
                base_contact_positions[3] - base_contact_positions[2],
            )
            active = float(base_cross @ base_cross) < float(threshold)
            expected_active.append(int(active))
            if not active:
                jacobian = contact_body_jacobian(ids, weights)
                expected_force += jacobian.T @ (7.0 * float(depth) * direction)
                expected_hessian += 7.0 * jacobian.T @ np.outer(direction, direction) @ jacobian
                continue

            flat_states = base_states.reshape(-1)
            residual_arguments = (
                ids,
                base_contact_positions,
                float(depth),
                weights,
                direction,
                float(threshold),
            )
            residual_value = residual(flat_states, *residual_arguments)
            residual_jacobian = np.empty((3, 24), dtype=np.float64)
            for column in range(24):
                offset = np.zeros(24, dtype=np.float64)
                offset[column] = epsilon
                residual_jacobian[:, column] = (
                    residual(flat_states + offset, *residual_arguments)
                    - residual(flat_states - offset, *residual_arguments)
                ) / (2.0 * epsilon)
            expected_force += -7.0 * residual_jacobian.T @ residual_value
            expected_hessian += 7.0 * residual_jacobian.T @ residual_jacobian

        self.assertIn(1, expected_active)
        np.testing.assert_array_equal(
            contact.edge_edge_contacts.mollifier_active.numpy()[: len(ee_ids)],
            expected_active,
        )
        expected_product = (expected_hessian @ input_values.reshape(-1)).reshape(2, 12)
        expected_diagonal = np.stack(
            [expected_hessian[:12, :12], expected_hessian[12:, 12:]],
        )
        np.testing.assert_allclose(force.numpy().reshape(-1), expected_force, rtol=4.0e-4, atol=4.0e-5)
        np.testing.assert_allclose(product.numpy(), expected_product, rtol=5.0e-4, atol=5.0e-5)
        np.testing.assert_allclose(diagonal.numpy(), expected_diagonal, rtol=5.0e-4, atol=5.0e-5)
        np.testing.assert_allclose(expected_hessian, expected_hessian.T, atol=1.0e-9)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(expected_hessian)[0]), -1.0e-8)

    def test_scales_near_parallel_ee_friction_load(self):
        """Scale near-parallel EE friction by the active mollifier value."""
        model = _near_parallel_model()
        velocity_values = np.zeros((2, 12), dtype=np.float32)
        velocity_values[1, :3] = [0.0, 0.1, 0.0]
        velocities = wp.array(velocity_values, dtype=vec12, device="cpu")
        frictionless = _make_contact(
            model,
            thickness=0.0025,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.0,
        )
        frictional = _make_contact(
            model,
            thickness=0.0025,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.4,
            friction_epsilon=1.0e-4,
        )
        forces = []
        for contact in (frictionless, frictional):
            force = wp.zeros(2, dtype=vec12, device="cpu")
            contact.begin_step(model.q, velocities, 0.1)
            contact.prepare(model.q)
            contact.accumulate_force(model.q, force)
            forces.append(force.numpy())

        rest_positions = model.rest_surface_vertices.numpy().astype(np.float64)
        ownership = model.surface_ownership.numpy()
        expected = np.zeros((2, 12), dtype=np.float64)
        expected_active_count = 0
        for buffer, use_mollifier in (
            (frictional.vertex_face_contacts, False),
            (frictional.edge_edge_contacts, True),
        ):
            ids, weights, depths = _active_rows(buffer)
            directions = buffer.directions.numpy()[: len(ids)].astype(np.float64)
            thresholds = (
                buffer.mollifier_thresholds.numpy()[: len(ids)].astype(np.float64)
                if use_mollifier
                else np.zeros(len(ids))
            )
            for contact_ids, contact_weights, direction, depth, threshold in zip(
                ids,
                weights,
                directions,
                depths,
                thresholds,
                strict=True,
            ):
                body_jacobians = np.zeros((2, 3, 12), dtype=np.float64)
                for particle, weight in zip(contact_ids, contact_weights, strict=True):
                    body_jacobians[ownership[particle]] += float(weight) * _point_jacobian(rest_positions[particle])
                relative_velocity = sum(body_jacobians[body] @ velocity_values[body] for body in range(2))
                tangent = np.eye(3) - np.outer(direction, direction)
                tangent_displacement = 0.1 * tangent @ relative_velocity
                tangent_length = float(np.linalg.norm(tangent_displacement))
                inverse_length = (
                    1.0 / tangent_length if tangent_length > 1.0e-4 else (2.0 - tangent_length / 1.0e-4) / 1.0e-4
                )
                load_scale = 1.0
                if use_mollifier:
                    positions = frictional.positions.numpy()[contact_ids]
                    cross_product = np.cross(positions[1] - positions[0], positions[3] - positions[2])
                    cross_squared = float(cross_product @ cross_product)
                    if cross_squared < threshold:
                        expected_active_count += 1
                        load_scale = np.clip(
                            cross_squared * (2.0 * threshold - cross_squared) / (threshold * threshold),
                            0.0,
                            1.0,
                        )
                alpha = 0.4 * 10.0 * float(depth) * inverse_length * load_scale
                world_force = -alpha * tangent_displacement
                for body in range(2):
                    expected[body] += body_jacobians[body].T @ world_force

        self.assertGreater(expected_active_count, 0)
        np.testing.assert_allclose(forces[1] - forces[0], expected, rtol=5.0e-4, atol=5.0e-6)


if __name__ == "__main__":
    unittest.main()
