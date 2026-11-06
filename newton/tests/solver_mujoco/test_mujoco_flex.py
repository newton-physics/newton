# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mapping Newton particles to MuJoCo flexes in SolverMuJoCo."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo


def _builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    # MuJoCo Warp applies geom gap as a contact-inclusion offset for flex-plane contacts.
    builder.default_shape_cfg.gap = 0.0
    return builder


class TestMuJoCoFlex(unittest.TestCase):
    def test_element_dimensions_map_to_flexes(self):
        """Tetrahedra, triangles, and springs become 3D, 2D, and 1D flexes without duplicates."""
        builder = _builder()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
        )
        builder.add_cloth_grid(
            pos=wp.vec3(1.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
        )
        rope = [builder.add_particle((2.0 + 0.1 * i, 0.0, 1.0), (0.0, 0.0, 0.0), 0.01) for i in range(3)]
        builder.particle_mass[rope[0]] = 0.0
        builder.add_spring(rope[0], rope[1], 50.0, 0.5, 0.0)
        builder.add_spring(rope[1], rope[2], 50.0, 0.5, 0.0)
        model = builder.finalize()

        mj_model = SolverMuJoCo(model, use_flex=True).mj_model

        np.testing.assert_array_equal(mj_model.flex_dim, [3, 2, 1])
        # The soft grid's surface triangles are covered by its 3D flex, so the 2D flex holds only the cloth.
        np.testing.assert_array_equal(mj_model.flex_vertnum, [8, 9, 3])
        np.testing.assert_array_equal(mj_model.flex_elemnum, [5, 8, 2])
        np.testing.assert_allclose(mj_model.flex_edgestiffness, [0.0, 0.0, 50.0])
        np.testing.assert_allclose(mj_model.flex_edgedamping, [0.0, 0.0, 0.5])
        # Every particle except the pinned rope end has three slide DOFs.
        self.assertEqual(mj_model.nv, 3 * (model.particle_count - 1))

    def test_cloth_falls_onto_ground_in_each_world(self):
        """Replicated cloths rest on the ground while a massless particle stays pinned."""
        cloth = _builder()
        cloth.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            particle_radius=0.02,
        )
        pinned = cloth.add_particle((0.5, 0.0, 0.5), (0.0, 0.0, 0.0), 0.0)
        builder = _builder()
        builder.replicate(cloth, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize()

        solver = SolverMuJoCo(model, use_flex=True)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(150):
            solver.step(state_0, state_1, control, None, 2.0e-3)
            state_0, state_1 = state_1, state_0

        q0 = model.particle_q.numpy().reshape(2, -1, 3)
        q = state_0.particle_q.numpy().reshape(2, -1, 3)
        np.testing.assert_allclose(q[:, pinned], q0[:, pinned])
        np.testing.assert_allclose(q[:, :pinned, 2], 0.02, atol=5.0e-3)
        np.testing.assert_allclose(q[1, :, 0] - q[0, :, 0], 1.0, atol=1.0e-3)


if __name__ == "__main__":
    unittest.main()
