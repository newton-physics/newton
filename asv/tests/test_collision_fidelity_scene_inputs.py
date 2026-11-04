# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check physical inputs and ordinary solver construction without a GPU."""

import sys
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

sys.path.insert(0, str(Path(__file__).parents[1]))

from collision_fidelity.public_vbd import create_solver
from collision_fidelity.scenes import build_scene


class TestCollisionFidelitySceneInputs(unittest.TestCase):
    def test_public_solver_caps_contact_without_changing_material_or_state(self):
        """Apply only the declared cap and keep the ordinary solver and pipeline."""
        builder = newton.ModelBuilder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=100.0,
            tri_ka=200.0,
        )
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=5000.0))
        builder.color()
        model = builder.finalize(device="cpu")
        model.soft_contact_ke = 4000.0
        positions = model.particle_q.numpy().copy()
        material = model.tri_materials.numpy().copy()
        solver, pipeline = create_solver(model, iterations=2, particle_enable_self_contact=True)
        self.assertIs(type(solver), SolverVBD)
        self.assertIs(type(pipeline), newton.CollisionPipeline)
        self.assertIs(solver.collision_pipeline, pipeline)
        self.assertTrue(solver.rigid_compliant_alm)
        self.assertTrue(model.collision_fidelity_solver_options["rigid_compliant_alm"])
        self.assertEqual(model.soft_contact_ke, 1000.0)
        np.testing.assert_array_equal(model.shape_material_ke.numpy(), [1000.0])
        np.testing.assert_array_equal(model.particle_q.numpy(), positions)
        np.testing.assert_array_equal(model.tri_materials.numpy(), material)

    def test_freefall_has_no_supports_or_pins(self):
        """Keep the analytic control free of physical supports and driven points."""
        scene = build_scene("s3_cloth_10substeps", device="cpu")
        self.assertEqual(scene.model.shape_count, 0)
        self.assertEqual(scene.model.body_count, 0)
        self.assertEqual(scene.model.particle_count, 4 * 11 * 11)
        self.assertEqual(scene.driven_indices, [])
        self.assertTrue(np.all(scene.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)))
        self.assertTrue(np.all(scene.model.particle_mass.numpy() > 0))
        np.testing.assert_allclose(scene.model.gravity.numpy(), [[0.0, 0.0, -9.81]])
        self.assertAlmostEqual(scene.dt, 1.0 / 600.0)

    def test_duck_remains_explicitly_unsupported(self):
        """Do not silently replace the archived second-solver articulated recipe."""
        with self.assertRaisesRegex(NotImplementedError, "second dynamics solver"):
            build_scene("r2_franka_duck", device="cpu")


if __name__ == "__main__":
    unittest.main()
