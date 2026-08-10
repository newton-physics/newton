# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintGroupDynamic(unittest.TestCase):
    def test_public_export_constructs_ordered_group(self):
        device = wp.get_device("cuda:0")
        events = []

        class RecordingConstraint:
            def __init__(self, name):
                self.name = name
                self.particle_count = 2
                self.device = device

            def begin_step(self, positions, velocities, dt):
                events.append((self.name, "begin_step", dt))

            def prepare(self, positions):
                events.append((self.name, "prepare"))

            def accumulate_force(self, positions, output):
                events.append((self.name, "force"))

            def hessian_multiply(self, positions, vector, output):
                events.append((self.name, "hessian"))

            def accumulate_diagonal(self, positions, output):
                events.append((self.name, "diagonal"))

        group_type = newton.solvers.ConstraintGroupDynamic
        group = group_type([RecordingConstraint("self"), RecordingConstraint("plane")])
        positions = wp.zeros(2, dtype=wp.vec3, device=device)
        velocities = wp.zeros_like(positions)
        diagonal = wp.zeros(2, dtype=wp.mat33, device=device)

        group.begin_step(positions, velocities, 0.01)
        group.prepare(positions)
        group.accumulate_force(positions, velocities)
        group.hessian_multiply(positions, velocities, velocities)
        group.accumulate_diagonal(positions, diagonal)

        self.assertEqual(group.particle_count, 2)
        self.assertEqual(group.device, device)
        self.assertEqual(
            events,
            [
                ("self", "begin_step", 0.01),
                ("plane", "begin_step", 0.01),
                ("self", "prepare"),
                ("plane", "prepare"),
                ("self", "force"),
                ("plane", "force"),
                ("self", "hessian"),
                ("plane", "hessian"),
                ("self", "diagonal"),
                ("plane", "diagonal"),
            ],
        )

    def test_rejects_empty_or_mismatched_particle_domains(self):
        group_type = newton.solvers.ConstraintGroupDynamic
        device = wp.get_device("cuda:0")

        class Domain:
            def __init__(self, particle_count, device):
                self.particle_count = particle_count
                self.device = wp.get_device(device)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            group_type([])
        with self.assertRaisesRegex(ValueError, "particle count"):
            group_type([Domain(2, device), Domain(3, device)])

    def test_static_system_binding_is_forwarded_only_to_consumers(self):
        device = wp.get_device("cuda:0")
        bindings = []

        class Constraint:
            def __init__(self):
                self.particle_count = 2
                self.device = device

        class AdaptiveConstraint(Constraint):
            def bind_static_system(self, static_diagonal, masses):
                bindings.append((static_diagonal, masses))

        group = newton.solvers.ConstraintGroupDynamic([AdaptiveConstraint(), Constraint()])
        static_diagonal = wp.zeros(2, dtype=wp.mat33, device=device)
        masses = wp.ones(2, dtype=float, device=device)

        group.bind_static_system(static_diagonal, masses)

        self.assertEqual(bindings, [(static_diagonal, masses)])


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestConstraintStaticPlaneContact(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device("cuda:0")

    def _make_contact(self, **overrides):
        parameters = {
            "normal": (0.0, 0.0, 1.0),
            "offset": 0.0,
            "thickness": 0.1,
            "stiffness": 10.0,
            "normal_damping": 2.0,
            "friction": 0.5,
            "friction_epsilon": 0.1,
            "particle_count": 3,
            "device": self.device,
        }
        parameters.update(overrides)
        contact_type = newton.solvers.ConstraintStaticPlaneContact
        return contact_type(**parameters)

    def test_force_hessian_product_and_exact_diagonal(self):
        contact = self._make_contact()
        positions = wp.array(
            [(0.0, 0.0, 0.05), (0.0, 0.0, -0.02), (0.0, 0.0, 0.2)],
            dtype=wp.vec3,
            device=self.device,
        )
        velocities = wp.array(
            [(1.0, 0.0, -2.0), (0.0, 0.0, 0.0), (5.0, 0.0, -5.0)],
            dtype=wp.vec3,
            device=self.device,
        )
        direction = wp.array([(2.0, 3.0, 4.0)] * 3, dtype=wp.vec3, device=self.device)
        force = wp.zeros_like(positions)
        product = wp.zeros_like(positions)
        diagonal = wp.zeros(3, dtype=wp.mat33, device=self.device)

        contact.begin_step(positions, velocities, 0.1)
        contact.prepare(positions)
        contact.accumulate_force(positions, force)
        contact.hessian_multiply(positions, direction, product)
        contact.accumulate_diagonal(positions, diagonal)

        np.testing.assert_allclose(
            force.numpy(),
            [[-0.25, 0.0, 4.5], [0.0, 0.0, 1.2], [0.0, 0.0, 0.0]],
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            product.numpy(),
            [[5.0, 7.5, 120.0], [24.0, 36.0, 40.0], [0.0, 0.0, 0.0]],
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            diagonal.numpy(),
            [np.diag([2.5, 2.5, 30.0]), np.diag([12.0, 12.0, 10.0]), np.zeros((3, 3))],
            atol=1.0e-5,
        )

    def test_particle_indices_restrict_contact_to_surface_subset(self):
        """Apply plane forces and Hessians only to selected surface particles."""
        contact = self._make_contact(
            particle_indices=[0, 2],
            normal_damping=0.0,
            friction=0.0,
        )
        positions = wp.array([(0.0, 0.0, 0.0)] * 3, dtype=wp.vec3, device=self.device)
        velocities = wp.zeros_like(positions)
        direction = wp.array([(0.0, 0.0, 1.0)] * 3, dtype=wp.vec3, device=self.device)
        force = wp.zeros_like(positions)
        product = wp.zeros_like(positions)
        diagonal = wp.zeros(3, dtype=wp.mat33, device=self.device)

        contact.begin_step(positions, velocities, 0.1)
        contact.prepare(positions)
        contact.accumulate_force(positions, force)
        contact.hessian_multiply(positions, direction, product)
        contact.accumulate_diagonal(positions, diagonal)

        np.testing.assert_allclose(force.numpy(), [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(product.numpy(), [[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
        np.testing.assert_allclose(
            diagonal.numpy(),
            [np.diag([0.0, 0.0, 10.0]), np.zeros((3, 3)), np.diag([0.0, 0.0, 10.0])],
        )

    def test_rejects_invalid_parameters(self):
        cases = [
            ({"normal": (0.0, 0.0, 0.0)}, "normal"),
            ({"normal": (0.0, np.nan, 1.0)}, "normal"),
            ({"offset": np.inf}, "offset"),
            ({"thickness": 0.0}, "thickness"),
            ({"stiffness": 0.0}, "stiffness"),
            ({"normal_damping": -1.0}, "normal_damping"),
            ({"friction": -1.0}, "friction"),
            ({"friction_epsilon": 0.0}, "friction_epsilon"),
            ({"particle_count": 0}, "particle_count"),
            ({"particle_indices": []}, "particle_indices"),
            ({"particle_indices": [0.0, 1.0]}, "integers"),
            ({"particle_indices": [0, 3]}, "out-of-range"),
            ({"particle_indices": [0, 0]}, "unique"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    self._make_contact(**overrides)

    def test_requires_step_data_and_matching_particle_count(self):
        contact = self._make_contact()
        positions = wp.zeros(3, dtype=wp.vec3, device=self.device)
        velocities = wp.zeros_like(positions)

        with self.assertRaisesRegex(RuntimeError, "begin_step"):
            contact.prepare(positions)
        with self.assertRaisesRegex(ValueError, "dt"):
            contact.begin_step(positions, velocities, 0.0)
        with self.assertRaisesRegex(ValueError, "3"):
            contact.begin_step(wp.zeros(2, dtype=wp.vec3, device=self.device), velocities, 0.1)


if __name__ == "__main__":
    unittest.main()
