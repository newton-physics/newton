# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.limx.affine_types import mat1212, vec12


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


def _unit_tetrahedron_model(device: str):
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
    return newton.solvers.AffineBodyModel(
        vertices,
        tetrahedra,
        surface_triangles,
        density=6.0,
        rigidity=0.0,
        initial_transform=wp.transform_identity(),
        device=device,
    )


class TestConstraintAffineStaticPlaneContact(unittest.TestCase):
    @staticmethod
    def _devices() -> list[str]:
        devices = ["cpu"]
        if wp.is_cuda_available():
            devices.append("cuda:0")
        return devices

    def test_lifts_force_hessian_product_and_exact_diagonal(self):
        """Lift world contact force and curvature into affine coordinates."""
        direction_values = np.linspace(-0.6, 0.5, 12, dtype=np.float32)
        dt = 0.1

        for device in self._devices():
            with self.subTest(device=device):
                model = _unit_tetrahedron_model(device)
                contact = newton.solvers.ConstraintAffineStaticPlaneContact(
                    model,
                    normal=(0.0, 0.0, 1.0),
                    offset=0.0,
                    thickness=0.3,
                    stiffness=10.0,
                    normal_damping=2.0,
                    friction=0.5,
                    friction_epsilon=0.1,
                )
                velocity_values = np.zeros(12, dtype=np.float32)
                velocity_values[:3] = [0.2, -0.1, -0.5]
                velocities = wp.array([velocity_values], dtype=vec12, device=device)
                direction = wp.array([direction_values], dtype=vec12, device=device)
                force = wp.zeros(1, dtype=vec12, device=device)
                product = wp.zeros(1, dtype=vec12, device=device)
                diagonal = wp.zeros(1, dtype=mat1212, device=device)
                empty_particles = wp.empty(0, dtype=wp.vec3, device=device)

                contact.begin_step(model.q, velocities, dt)
                contact.prepare(model.q)
                contact.accumulate_force(model.q, force)
                contact.multiply(empty_particles, direction, empty_particles, product)
                contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device=device), diagonal)

                expected_force = np.zeros(12, dtype=np.float64)
                expected_product = np.zeros(12, dtype=np.float64)
                expected_diagonal = np.zeros((12, 12), dtype=np.float64)
                state = model.q.numpy()[0].astype(np.float64)
                generalized_velocity = velocity_values.astype(np.float64)
                for rest_position in model.rest_surface_vertices.numpy().astype(np.float64):
                    jacobian = _point_jacobian(rest_position)
                    position = jacobian @ state
                    depth = 0.3 - position[2]
                    if depth <= 0.0:
                        continue
                    velocity = jacobian @ generalized_velocity
                    normal_velocity = velocity[2]
                    tangent_displacement = dt * np.asarray([velocity[0], velocity[1], 0.0])
                    tangent_length = np.linalg.norm(tangent_displacement)
                    friction_over_length = (2.0 - tangent_length / 0.1) / 0.1
                    alpha = 0.5 * 10.0 * depth * friction_over_length
                    world_force = np.asarray(
                        [
                            -alpha * tangent_displacement[0],
                            -alpha * tangent_displacement[1],
                            10.0 * depth - 2.0 * normal_velocity,
                        ]
                    )
                    world_hessian = np.diag([alpha, alpha, 10.0 + 2.0 / dt])
                    lifted_hessian = jacobian.T @ world_hessian @ jacobian
                    expected_force += jacobian.T @ world_force
                    expected_product += lifted_hessian @ direction_values
                    expected_diagonal += lifted_hessian

                np.testing.assert_allclose(force.numpy()[0], expected_force, rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(product.numpy()[0], expected_product, rtol=2.0e-5, atol=2.0e-6)
                np.testing.assert_allclose(diagonal.numpy()[0], expected_diagonal, rtol=2.0e-5, atol=2.0e-6)

    def test_zeros_inactive_contact_contributions(self):
        """Zero every lifted contribution when all surface points clear the plane."""
        for device in self._devices():
            with self.subTest(device=device):
                model = _unit_tetrahedron_model(device)
                state_values = model.q.numpy()
                state_values[0, 2] += 2.0
                state = wp.array(state_values, dtype=vec12, device=device)
                contact = newton.solvers.ConstraintAffineStaticPlaneContact(
                    model,
                    normal=(0.0, 0.0, 1.0),
                    offset=0.0,
                    thickness=0.1,
                    stiffness=10.0,
                    normal_damping=2.0,
                    friction=0.5,
                    friction_epsilon=0.1,
                )
                velocities = wp.zeros(1, dtype=vec12, device=device)
                direction = wp.array([np.linspace(0.1, 1.2, 12)], dtype=vec12, device=device)
                force = wp.zeros(1, dtype=vec12, device=device)
                product = wp.zeros_like(force)
                diagonal = wp.zeros(1, dtype=mat1212, device=device)
                empty_particles = wp.empty(0, dtype=wp.vec3, device=device)

                contact.begin_step(state, velocities, 0.1)
                contact.prepare(state)
                contact.accumulate_force(state, force)
                contact.multiply(empty_particles, direction, empty_particles, product)
                contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device=device), diagonal)

                np.testing.assert_array_equal(force.numpy(), np.zeros((1, 12), dtype=np.float32))
                np.testing.assert_array_equal(product.numpy(), np.zeros((1, 12), dtype=np.float32))
                np.testing.assert_array_equal(diagonal.numpy(), np.zeros((1, 12, 12), dtype=np.float32))

    def test_applies_normal_damping_only_while_approaching(self):
        """Apply normal damping force and curvature only to approaching points."""
        model = _unit_tetrahedron_model("cpu")
        rest_positions = model.rest_surface_vertices.numpy().astype(np.float64)
        state = model.q.numpy()[0].astype(np.float64)
        expected_lift = np.zeros(12, dtype=np.float64)
        expected_hessian = np.zeros((12, 12), dtype=np.float64)
        for rest_position in rest_positions:
            jacobian = _point_jacobian(rest_position)
            if 0.3 - (jacobian @ state)[2] <= 0.0:
                continue
            expected_lift += jacobian.T @ np.asarray([0.0, 0.0, 1.0])
            expected_hessian += jacobian.T @ np.diag([0.0, 0.0, 20.0]) @ jacobian

        results = []
        for normal_velocity in (-0.5, 0.5):
            contact = newton.solvers.ConstraintAffineStaticPlaneContact(
                model,
                normal=(0.0, 0.0, 1.0),
                offset=0.0,
                thickness=0.3,
                stiffness=10.0,
                normal_damping=2.0,
                friction=0.0,
                friction_epsilon=0.1,
            )
            velocity_values = np.zeros(12, dtype=np.float32)
            velocity_values[2] = normal_velocity
            velocities = wp.array([velocity_values], dtype=vec12, device="cpu")
            force = wp.zeros(1, dtype=vec12, device="cpu")
            diagonal = wp.zeros(1, dtype=mat1212, device="cpu")
            contact.begin_step(model.q, velocities, 0.1)
            contact.prepare(model.q)
            contact.accumulate_force(model.q, force)
            contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device="cpu"), diagonal)
            results.append((force.numpy()[0], diagonal.numpy()[0]))

        approaching, separating = results
        np.testing.assert_allclose(approaching[0] - separating[0], expected_lift, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(approaching[1] - separating[1], expected_hessian, rtol=2.0e-5, atol=2.0e-6)

    def test_regularizes_friction_and_opposes_tangential_motion(self):
        """Keep small-slip friction finite and oppose tangential affine motion."""
        model = _unit_tetrahedron_model("cpu")
        contact = newton.solvers.ConstraintAffineStaticPlaneContact(
            model,
            normal=(0.0, 0.0, 1.0),
            offset=0.0,
            thickness=0.3,
            stiffness=10.0,
            normal_damping=0.0,
            friction=0.5,
            friction_epsilon=0.1,
        )
        velocity_values = np.zeros(12, dtype=np.float32)
        velocity_values[:2] = [1.0e-4, -2.0e-4]
        velocities = wp.array([velocity_values], dtype=vec12, device="cpu")
        force = wp.zeros(1, dtype=vec12, device="cpu")
        diagonal = wp.zeros(1, dtype=mat1212, device="cpu")

        contact.begin_step(model.q, velocities, 0.1)
        contact.prepare(model.q)
        contact.accumulate_force(model.q, force)
        contact.accumulate_diagonal(wp.empty(0, dtype=wp.mat33, device="cpu"), diagonal)

        force_values = force.numpy()[0]
        self.assertTrue(np.isfinite(force_values).all())
        self.assertTrue(np.isfinite(diagonal.numpy()).all())
        self.assertLess(float(np.dot(force_values, velocity_values)), 0.0)

    def test_rejects_invalid_parameters(self):
        """Reject invalid affine contact parameters before allocating runtime state."""
        model = _unit_tetrahedron_model("cpu")
        defaults = {
            "body_model": model,
            "normal": (0.0, 0.0, 1.0),
            "offset": 0.0,
            "thickness": 0.1,
            "stiffness": 10.0,
            "normal_damping": 2.0,
            "friction": 0.5,
            "friction_epsilon": 0.1,
        }
        cases = [
            ({"body_model": object()}, "body_model"),
            ({"normal": (0.0, 0.0, 0.0)}, "normal"),
            ({"normal": (0.0, np.nan, 1.0)}, "normal"),
            ({"offset": np.inf}, "offset"),
            ({"thickness": 0.0}, "thickness"),
            ({"stiffness": 0.0}, "stiffness"),
            ({"normal_damping": -1.0}, "normal_damping"),
            ({"friction": -1.0}, "friction"),
            ({"friction_epsilon": 0.0}, "friction_epsilon"),
        ]

        contact_type = newton.solvers.ConstraintAffineStaticPlaneContact
        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex((TypeError, ValueError), message):
                contact_type(**(defaults | overrides))

    def test_requires_lifecycle_and_matching_affine_vectors(self):
        """Require prepared matching affine vectors and empty particle vectors."""
        model = _unit_tetrahedron_model("cpu")
        contact = newton.solvers.ConstraintAffineStaticPlaneContact(
            model,
            normal=(0.0, 0.0, 1.0),
            offset=0.0,
            thickness=0.1,
            stiffness=10.0,
            normal_damping=2.0,
            friction=0.5,
            friction_epsilon=0.1,
        )
        velocities = wp.zeros(1, dtype=vec12, device="cpu")

        with self.assertRaisesRegex(RuntimeError, "begin_step"):
            contact.prepare(model.q)
        with self.assertRaisesRegex(ValueError, "dt"):
            contact.begin_step(model.q, velocities, 0.0)
        with self.assertRaisesRegex(ValueError, "1"):
            contact.begin_step(wp.empty(0, dtype=vec12, device="cpu"), velocities, 0.1)
        with self.assertRaisesRegex(TypeError, "vec12"):
            contact.begin_step(wp.zeros(1, dtype=wp.vec3, device="cpu"), velocities, 0.1)

        contact.begin_step(model.q, velocities, 0.1)
        contact.prepare(model.q)
        nonempty_particles = wp.zeros(1, dtype=wp.vec3, device="cpu")
        affine_output = wp.zeros(1, dtype=vec12, device="cpu")
        with self.assertRaisesRegex(ValueError, "particle_input"):
            contact.multiply(nonempty_particles, velocities, nonempty_particles, affine_output)
        with self.assertRaisesRegex(ValueError, "particle_diagonal"):
            contact.accumulate_diagonal(
                wp.zeros(1, dtype=wp.mat33, device="cpu"), wp.zeros(1, dtype=mat1212, device="cpu")
            )

        if wp.is_cuda_available():
            with self.assertRaisesRegex(ValueError, "device"):
                contact.begin_step(
                    wp.zeros(1, dtype=vec12, device="cuda:0"),
                    wp.zeros(1, dtype=vec12, device="cuda:0"),
                    0.1,
                )


if __name__ == "__main__":
    unittest.main()
