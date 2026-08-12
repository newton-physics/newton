# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton._src.solvers.limx.affine_types import mat1212, vec12


class TestConstraintGroupAffine(unittest.TestCase):
    def test_forwards_affine_constraint_lifecycle_in_order(self):
        """Forward every affine dynamic operation in declaration order."""
        events = []

        class RecordingConstraint:
            body_count = 2
            device = wp.get_device("cpu")

            def __init__(self, name):
                self.name = name

            def begin_step(self, q, qd, dt):
                events.append((self.name, "begin", dt))

            def prepare(self, q):
                events.append((self.name, "prepare"))

            def accumulate_force(self, q, output):
                events.append((self.name, "force"))

            def multiply(self, particle_input, affine_input, particle_output, affine_output):
                events.append((self.name, "multiply"))

            def accumulate_diagonal(self, particle_diagonal, affine_diagonal):
                events.append((self.name, "diagonal"))

        group = newton.solvers.ConstraintGroupAffine([RecordingConstraint("body"), RecordingConstraint("ground")])
        q = wp.zeros(2, dtype=vec12, device="cpu")
        empty_particles = wp.empty(0, dtype=wp.vec3, device="cpu")
        group.begin_step(q, q, 0.01)
        group.prepare(q)
        group.accumulate_force(q, q)
        group.multiply(empty_particles, q, empty_particles, q)
        group.accumulate_diagonal(
            wp.empty(0, dtype=wp.mat33, device="cpu"),
            wp.zeros(2, dtype=mat1212, device="cpu"),
        )

        self.assertEqual(group.body_count, 2)
        self.assertEqual(group.device, wp.get_device("cpu"))
        self.assertEqual(
            events,
            [
                ("body", "begin", 0.01),
                ("ground", "begin", 0.01),
                ("body", "prepare"),
                ("ground", "prepare"),
                ("body", "force"),
                ("ground", "force"),
                ("body", "multiply"),
                ("ground", "multiply"),
                ("body", "diagonal"),
                ("ground", "diagonal"),
            ],
        )

    def test_rejects_empty_or_mismatched_affine_domains(self):
        """Reject empty groups and children over different affine domains."""

        class Domain:
            def __init__(self, body_count, device):
                self.body_count = body_count
                self.device = wp.get_device(device)

        group_type = newton.solvers.ConstraintGroupAffine
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            group_type([])
        with self.assertRaisesRegex(ValueError, "body count"):
            group_type([Domain(2, "cpu"), Domain(3, "cpu")])
        if wp.is_cuda_available():
            with self.assertRaisesRegex(ValueError, "device"):
                group_type([Domain(2, "cpu"), Domain(2, "cuda:0")])


if __name__ == "__main__":
    unittest.main()
