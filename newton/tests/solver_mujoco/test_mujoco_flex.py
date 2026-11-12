# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mapping Newton particles to MuJoCo flexes in SolverMuJoCo."""

import itertools
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

        mj_model = SolverMuJoCo(model).mj_model

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

        solver = SolverMuJoCo(model)
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

        solver = SolverMuJoCo(model)
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
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles, poisson=0.3)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, springs=springs, edgeequality=True, edgestiffness=1.0)
        with self.assertRaises(IndexError):
            mujoco.add_flex(builder, springs=[builder.spring_count])
        mujoco.add_flex(builder, springs=springs)
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, springs=springs)
        builder.begin_world()
        with self.assertRaises(ValueError):
            mujoco.add_flex(builder, triangles=triangles)
        builder.end_world()

    def test_stiff_cable_rests_on_another_flex(self):
        """Flexes collide with each other, and edge equality keeps a stiff cable stable and inextensible."""
        builder = _builder()
        below = [builder.add_particle((0.1 * i - 0.2, 0.0, 0.1), (0.0, 0.0, 0.0), 0.0, radius=0.01) for i in range(5)]
        for a, b in itertools.pairwise(below):
            builder.add_spring(a, b, 1.0e3, 0.0, 0.0)
        above = [builder.add_particle((0.0, 0.1 * i - 0.2, 0.2), (0.0, 0.0, 0.0), 0.01, radius=0.01) for i in range(5)]
        start = builder.spring_count
        for a, b in itertools.pairwise(above):
            builder.add_spring(a, b, 5.0e5, 0.0, 0.0)
        mujoco.add_flex(builder, springs=range(start, builder.spring_count), edgeequality=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        self.assertEqual(solver.mj_model.neq, 1)
        self.assertEqual(solver.mj_model.eq_type[0], int(solver._mujoco.mjtEq.mjEQ_FLEX))
        self.assertEqual(solver.mj_model.flex_edgestiffness[0], 0.0)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(250):
            solver.step(state_0, state_1, control, None, 2.0e-3)
            state_0, state_1 = state_1, state_0

        q = state_0.particle_q.numpy()[above]
        self.assertTrue(np.isfinite(q).all())
        # The middle of the falling cable rests on the pinned cable instead of passing through it.
        self.assertGreater(q[2, 2], 0.1 + 0.015)
        # Explicit edge springs this stiff diverge at this step; the soft edge constraint stays near rest length.
        np.testing.assert_allclose(np.linalg.norm(np.diff(q, axis=0), axis=1), 0.1, rtol=5.0e-2)

    def test_flexes_require_mujoco_contacts(self):
        """Flex particles cannot use Newton contacts, so the combination is rejected."""
        builder = _builder()
        _add_cloth(builder, 0.0)
        model = builder.finalize()
        with self.assertRaises(ValueError):
            SolverMuJoCo(model, use_mujoco_contacts=False)

    def test_cloth_springs_are_kept(self):
        """Cloth springs form their own 1D flex even though the cloth covers their particles."""
        builder = _builder()
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.01,
            add_springs=True,
            spring_ke=10.0,
        )
        model = builder.finalize()

        mj_model = SolverMuJoCo(model).mj_model
        np.testing.assert_array_equal(mj_model.flex_dim, [2, 1])
        self.assertEqual(mj_model.flex_elemnum[1], model.spring_count)
        np.testing.assert_allclose(mj_model.flex_edgestiffness[1], 10.0)

    def test_pins_and_contacts_in_each_world(self):
        """Pins stay at each world's position, particle-only colliders hold the cloth, and
        flex contacts export without a Newton shape on the flex side."""
        cloth = _builder()
        cloth.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.1),
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
            fix_left=True,
        )
        builder = _builder()
        builder.replicate(cloth, 2, spacing=(1.0, 0.0, 0.0))
        ground_cfg = newton.ModelBuilder.ShapeConfig(gap=0.0, has_shape_collision=False, has_particle_collision=True)
        ground = builder.add_ground_plane(cfg=ground_cfg)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(150):
            solver.step(state_0, state_1, control, None, 2.0e-3)
            state_0, state_1 = state_1, state_0

        q0 = model.particle_q.numpy().reshape(2, -1, 3)
        q = state_0.particle_q.numpy().reshape(2, -1, 3)
        pinned = np.flatnonzero(model.particle_mass.numpy()[: q.shape[1]] == 0.0)
        np.testing.assert_allclose(q[:, pinned], q0[:, pinned])
        # Each cloth hangs from its own world's pins, so the worlds stay one spacing apart.
        np.testing.assert_allclose(q[1, :, 0] - q[0, :, 0], 1.0, atol=1.0e-3)
        free = np.setdiff1d(np.arange(q.shape[1]), pinned)
        self.assertGreater(q[:, free, 2].min(), 0.0)

        contacts = newton.Contacts(solver.get_max_contact_count(), 0, device=model.device)
        solver.update_contacts(contacts, state_0)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        shapes = np.stack(
            [contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]], axis=1
        )
        np.testing.assert_array_equal(np.sort(shapes, axis=1), np.tile([-1, ground], (count, 1)))

        solver.reset(state_0)
        np.testing.assert_allclose(state_0.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_allclose(state_0.particle_qd.numpy(), 0.0)

    def test_only_active_element_particles_are_simulated(self):
        """Free particles and elements of inactive particles are left to other solvers."""
        builder = _builder()
        _add_cloth(builder, 0.0)
        cloth_particles = builder.particle_count
        _add_cloth(builder, 1.0)
        for p in range(cloth_particles, builder.particle_count):
            builder.particle_flags[p] &= ~int(newton.ParticleFlags.ACTIVE)
        builder.add_particle((2.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.01)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        self.assertEqual(solver.mj_model.nflex, 1)
        self.assertEqual(solver.mj_model.nv, 3 * cloth_particles)

        state_0, state_1 = model.state(), model.state()
        state_0.particle_qd.fill_(1.0)
        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        q, qd = state_1.particle_q.numpy(), state_1.particle_qd.numpy()
        self.assertFalse(np.allclose(q[:cloth_particles], model.particle_q.numpy()[:cloth_particles]))
        np.testing.assert_array_equal(q[cloth_particles:], model.particle_q.numpy()[cloth_particles:])
        np.testing.assert_array_equal(qd[cloth_particles:], 1.0)

        solver.reset(state_1)
        np.testing.assert_array_equal(state_1.particle_qd.numpy()[cloth_particles:], 1.0)

        # Models with only free particles, such as MPM sand, convert no particles.
        builder = _builder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_particle((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.01)
        model = builder.finalize()
        self.assertEqual(SolverMuJoCo(model, use_mujoco_contacts=False).mj_model.nv, 6)

    def test_worlds_must_share_flex_layout(self):
        """Worlds whose elements or authored flexes differ are rejected."""
        builder = _builder()
        for authored in (True, False):
            builder.begin_world()
            triangles = _add_cloth(builder, 0.0)
            if authored:
                mujoco.add_flex(builder, triangles=triangles)
            builder.end_world()
        model = builder.finalize()
        with self.assertRaises(ValueError):
            SolverMuJoCo(model)


if __name__ == "__main__":
    unittest.main()
