# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mapping Newton particles to MuJoCo flexes in SolverMuJoCo."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, mujoco


def _builder() -> newton.ModelBuilder:
    builder = newton.ModelBuilder()
    # MuJoCo Warp applies geom gap as a contact-inclusion offset for flex-plane contacts.
    builder.default_shape_cfg.gap = 0.0
    return builder


def _add_cloth(builder: newton.ModelBuilder, x: float) -> range:
    """Add a 2x2 cloth at ``x`` and return its triangle indices."""
    start = builder.tri_count
    builder.add_cloth_grid(
        pos=wp.vec3(x, 0.0, 1.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=2,
        dim_y=2,
        cell_x=0.1,
        cell_y=0.1,
        mass=0.01,
    )
    return range(start, builder.tri_count)


def _add_rope(builder: newton.ModelBuilder, x: float, **particle_kwargs) -> range:
    """Add a three-particle rope with a pinned first particle and return its spring indices."""
    rope = [builder.add_particle((x + 0.1 * i, 0.0, 1.0), (0.0, 0.0, 0.0), 0.01, **particle_kwargs) for i in range(3)]
    builder.particle_mass[rope[0]] = 0.0
    start = builder.spring_count
    builder.add_spring(rope[0], rope[1], 50.0, 0.5, 0.0)
    builder.add_spring(rope[1], rope[2], 50.0, 0.5, 0.0)
    return range(start, builder.spring_count)


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
        _add_cloth(builder, 1.0)
        _add_rope(builder, 2.0)
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

    def test_authored_flexes_keep_settings_in_each_world(self):
        """Authored flexes keep their settings per world, and the remaining elements are grouped automatically."""
        world = _builder()
        mujoco.add_flex(
            world,
            triangles=_add_cloth(world, 0.0),
            label="sheet",
            young=1.0e5,
            poisson=0.3,
            elastic2d="both",
            thickness=1.0e-3,
            selfcollide="auto",
            gap=0.01,
        )
        _add_cloth(world, 1.0)
        # Zero-radius particles still compile, since their inertia is clamped.
        springs = _add_rope(world, 2.0, radius=0.0)
        mujoco.add_flex(world, springs=springs, young=1.0e4, edgestiffness=7.0, radius=0.03, friction=(0.5, 0.0, 0.0))
        builder = _builder()
        builder.replicate(world, 2)
        model = builder.finalize()

        # Element assignments are remapped to each world's own flex rows.
        np.testing.assert_array_equal(model.mujoco.tri_flex.numpy().reshape(2, -1)[:, 0], [0, 2])
        np.testing.assert_array_equal(model.mujoco.spring_flex.numpy(), [1, 1, 3, 3])

        solver = SolverMuJoCo(model, use_flex=True)
        mj_model = solver.mj_model
        np.testing.assert_array_equal(mj_model.flex_dim, [2, 2, 1])
        self.assertEqual(solver._mujoco.mj_id2name(mj_model, solver._mujoco.mjtObj.mjOBJ_FLEX, 0), "sheet")
        np.testing.assert_array_equal(mj_model.flex_selfcollide, [4, 0, 0])
        np.testing.assert_allclose(mj_model.flex_gap, [0.01, 0.0, 0.0])
        np.testing.assert_allclose(mj_model.flex_radius, [0.1, 0.1, 0.03])
        np.testing.assert_allclose(mj_model.flex_edgestiffness, [0.0, 0.0, 7.0])
        np.testing.assert_allclose(mj_model.flex_friction[2], [0.5, 0.0, 0.0])
        # Only the sheet (elastic2d="both") and the rope (young > 0) resist bending.
        np.testing.assert_array_equal(mj_model.flex_bendingadr >= 0, [True, False, True])

    def test_add_flex_validates_input(self):
        """add_flex rejects ambiguous sources, mismatched options, reused elements, and other worlds."""
        builder = _builder()
        triangles = _add_cloth(builder, 0.0)
        springs = _add_rope(builder, 1.0)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles, springs=springs)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, springs=springs, thickness=0.1)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles, edgestiffness=1.0)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles, elastic2d="wrinkle")
        with self.assertRaises(IndexError):
            mujoco.add_flex(builder, springs=[builder.spring_count])
        mujoco.add_flex(builder, springs=springs)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, springs=springs)
        builder.begin_world()
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles)
        builder.end_world()

    def test_use_flex_requires_mujoco_contacts(self):
        """Flex particles cannot use Newton contacts, so the combination is rejected."""
        builder = _builder()
        _add_cloth(builder, 0.0)
        model = builder.finalize()
        with self.assertRaises(ValueError):
            SolverMuJoCo(model, use_flex=True, use_mujoco_contacts=False)


if __name__ == "__main__":
    unittest.main()
