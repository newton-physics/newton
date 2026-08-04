# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

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

        group_type = getattr(newton.solvers, "ConstraintGroupDynamic")
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
        group_type = getattr(newton.solvers, "ConstraintGroupDynamic")
        device = wp.get_device("cuda:0")

        class Domain:
            def __init__(self, particle_count, device):
                self.particle_count = particle_count
                self.device = wp.get_device(device)

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            group_type([])
        with self.assertRaisesRegex(ValueError, "particle count"):
            group_type([Domain(2, device), Domain(3, device)])


if __name__ == "__main__":
    unittest.main()
