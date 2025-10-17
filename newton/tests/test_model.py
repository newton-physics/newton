# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelBuilder
from newton._src.geometry.utils import create_box_mesh, transform_points
from newton.tests.unittest_utils import assert_np_equal


class TestModel(unittest.TestCase):
    def test_add_triangles(self):
        rng = np.random.default_rng(123)

        pts = np.array(
            [
                [-0.00585869, 0.34189449, -1.17415233],
                [-1.894547, 0.1788074, 0.9251329],
                [-1.26141048, 0.16140787, 0.08823282],
                [-0.08609255, -0.82722546, 0.65995427],
                [0.78827592, -1.77375711, -0.55582718],
            ]
        )
        tris = np.array([[0, 3, 4], [0, 2, 3], [2, 1, 3], [1, 4, 3]])

        builder1 = ModelBuilder()
        builder2 = ModelBuilder()
        for pt in pts:
            builder1.add_particle(wp.vec3(pt), wp.vec3(), 1.0)
            builder2.add_particle(wp.vec3(pt), wp.vec3(), 1.0)

        # test add_triangle(s) with default arguments:
        areas = builder2.add_triangles(tris[:, 0], tris[:, 1], tris[:, 2])
        for i, t in enumerate(tris):
            area = builder1.add_triangle(t[0], t[1], t[2])
            self.assertAlmostEqual(area, areas[i], places=6)

        # test add_triangle(s) with non default arguments:
        tri_ke = rng.standard_normal(size=pts.shape[0])
        tri_ka = rng.standard_normal(size=pts.shape[0])
        tri_kd = rng.standard_normal(size=pts.shape[0])
        tri_drag = rng.standard_normal(size=pts.shape[0])
        tri_lift = rng.standard_normal(size=pts.shape[0])
        for i, t in enumerate(tris):
            builder1.add_triangle(
                t[0],
                t[1],
                t[2],
                tri_ke[i],
                tri_ka[i],
                tri_kd[i],
                tri_drag[i],
                tri_lift[i],
            )
        builder2.add_triangles(tris[:, 0], tris[:, 1], tris[:, 2], tri_ke, tri_ka, tri_kd, tri_drag, tri_lift)

        assert_np_equal(np.array(builder1.tri_indices), np.array(builder2.tri_indices))
        assert_np_equal(np.array(builder1.tri_poses), np.array(builder2.tri_poses), tol=1.0e-6)
        assert_np_equal(np.array(builder1.tri_activations), np.array(builder2.tri_activations))
        assert_np_equal(np.array(builder1.tri_materials), np.array(builder2.tri_materials))

    def test_add_edges(self):
        rng = np.random.default_rng(123)

        pts = np.array(
            [
                [-0.00585869, 0.34189449, -1.17415233],
                [-1.894547, 0.1788074, 0.9251329],
                [-1.26141048, 0.16140787, 0.08823282],
                [-0.08609255, -0.82722546, 0.65995427],
                [0.78827592, -1.77375711, -0.55582718],
            ]
        )
        edges = np.array([[0, 4, 3, 1], [3, 2, 4, 1]])

        builder1 = ModelBuilder()
        builder2 = ModelBuilder()
        for pt in pts:
            builder1.add_particle(wp.vec3(pt), wp.vec3(), 1.0)
            builder2.add_particle(wp.vec3(pt), wp.vec3(), 1.0)

        # test defaults:
        for i in range(2):
            builder1.add_edge(edges[i, 0], edges[i, 1], edges[i, 2], edges[i, 3])
        builder2.add_edges(edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3])

        # test non defaults:
        rest = rng.standard_normal(size=2)
        edge_ke = rng.standard_normal(size=2)
        edge_kd = rng.standard_normal(size=2)
        for i in range(2):
            builder1.add_edge(edges[i, 0], edges[i, 1], edges[i, 2], edges[i, 3], rest[i], edge_ke[i], edge_kd[i])
        builder2.add_edges(edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3], rest, edge_ke, edge_kd)

        assert_np_equal(np.array(builder1.edge_indices), np.array(builder2.edge_indices))
        assert_np_equal(np.array(builder1.edge_rest_angle), np.array(builder2.edge_rest_angle), tol=1.0e-4)
        assert_np_equal(np.array(builder1.edge_bending_properties), np.array(builder2.edge_bending_properties))

    def test_collapse_fixed_joints(self):
        shape_cfg = ModelBuilder.ShapeConfig(density=1.0)

        def add_three_cubes(builder: ModelBuilder, parent_body=-1):
            unit_cube = {"hx": 0.5, "hy": 0.5, "hz": 0.5, "cfg": shape_cfg}
            b0 = builder.add_body()
            builder.add_shape_box(body=b0, **unit_cube)
            builder.add_joint_fixed(parent=parent_body, child=b0, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
            b1 = builder.add_body()
            builder.add_shape_box(body=b1, **unit_cube)
            builder.add_joint_fixed(parent=parent_body, child=b1, parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0)))
            b2 = builder.add_body()
            builder.add_shape_box(body=b2, **unit_cube)
            builder.add_joint_fixed(parent=parent_body, child=b2, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0)))
            return b2

        builder = ModelBuilder()
        # only fixed joints
        builder.add_articulation()
        add_three_cubes(builder)
        assert builder.joint_count == 3
        assert builder.body_count == 3

        # fixed joints followed by a non-fixed joint
        builder.add_articulation()
        last_body = add_three_cubes(builder)
        assert builder.joint_count == 6
        assert builder.body_count == 6
        assert builder.articulation_count == 2
        b3 = builder.add_body()
        builder.add_shape_box(
            body=b3, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg, xform=wp.transform(wp.vec3(1.0, 2.0, 3.0))
        )
        builder.add_joint_revolute(parent=last_body, child=b3, axis=wp.vec3(0.0, 1.0, 0.0))

        # a non-fixed joint followed by fixed joints
        builder.add_articulation()
        free_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.4, 0.5, 0.6))
        b4 = builder.add_body(xform=free_xform)
        builder.add_shape_box(body=b4, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        builder.add_joint_free(parent=-1, child=b4, parent_xform=wp.transform(wp.vec3(0.0, -1.0, 0.0)))
        assert_np_equal(builder.body_q[b4], np.array(free_xform))
        assert_np_equal(builder.joint_q[-7:], np.array(free_xform))
        assert builder.joint_count == 8
        assert builder.body_count == 8
        assert builder.articulation_count == 3
        add_three_cubes(builder, parent_body=b4)

        builder.collapse_fixed_joints()

        assert builder.joint_count == 2
        assert builder.articulation_count == 2
        assert builder.articulation_start == [0, 1]
        assert builder.joint_type == [newton.JointType.REVOLUTE, newton.JointType.FREE]
        assert builder.shape_count == 11
        assert builder.shape_body == [-1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1]
        assert builder.body_count == 2
        assert builder.body_com[0] == wp.vec3(1.0, 2.0, 3.0)
        assert builder.body_com[1] == wp.vec3(0.25, 0.25, 0.25)
        assert builder.body_mass == [1.0, 4.0]
        assert builder.body_inv_mass == [1.0, 0.25]

        # create another builder, test add_builder function
        builder2 = ModelBuilder()
        builder2.add_builder(builder)
        assert builder2.articulation_count == builder.articulation_count
        assert builder2.joint_count == builder.joint_count
        assert builder2.body_count == builder.body_count
        assert builder2.shape_count == builder.shape_count
        assert builder2.articulation_start == builder.articulation_start
        # add the same builder again
        builder2.add_builder(builder)
        assert builder2.articulation_count == 2 * builder.articulation_count
        assert builder2.articulation_start == [0, 1, 2, 3]

    def test_add_builder_with_open_edges(self):
        builder = ModelBuilder()

        dim_x = 16
        dim_y = 16

        world_builder = ModelBuilder()
        world_builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.25),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
        )

        num_worlds = 2
        world_offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        builder_open_edge_count = np.sum(np.array(builder.edge_indices) == -1)
        world_builder_open_edge_count = np.sum(np.array(world_builder.edge_indices) == -1)

        for i in range(num_worlds):
            xform = wp.transform(world_offsets[i], wp.quat_identity())
            builder.add_builder(
                world_builder,
                xform,
                update_num_world_count=True,
            )

        self.assertEqual(
            np.sum(np.array(builder.edge_indices) == -1),
            builder_open_edge_count + num_worlds * world_builder_open_edge_count,
            "builder does not have the expected number of open edges",
        )

    def test_mesh_approximation(self):
        def box_mesh(scale=(1.0, 1.0, 1.0), transform: wp.transform | None = None):
            vertices, indices = create_box_mesh(scale)
            if transform is not None:
                vertices = transform_points(vertices, transform)
            return newton.Mesh(vertices, indices)

        def npsorted(x):
            return np.array(sorted(x))

        builder = ModelBuilder()
        tf = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_identity())
        scale = wp.vec3(1.0, 3.0, 0.2)
        mesh = box_mesh(scale=scale, transform=tf)
        mesh.maxhullvert = 5
        s0 = builder.add_shape_mesh(body=-1, mesh=mesh)
        s1 = builder.add_shape_mesh(body=-1, mesh=mesh)
        s2 = builder.add_shape_mesh(body=-1, mesh=mesh)
        builder.approximate_meshes(method="convex_hull", shape_indices=[s0])
        builder.approximate_meshes(method="bounding_box", shape_indices=[s1])
        builder.approximate_meshes(method="bounding_sphere", shape_indices=[s2])
        # convex hull
        self.assertEqual(len(builder.shape_source[s0].vertices), 5)
        # the convex hull maintains the original transform
        assert_np_equal(np.array(builder.shape_transform[s0]), np.array(wp.transform_identity()), tol=1.0e-4)
        # bounding box
        self.assertIsNone(builder.shape_source[s1])
        self.assertEqual(builder.shape_type[s1], newton.GeoType.BOX)
        assert_np_equal(npsorted(builder.shape_scale[s1]), npsorted(scale), tol=1.0e-5)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[s1].p), np.array(tf.p), tol=1.0e-4)
        # bounding sphere
        self.assertIsNone(builder.shape_source[s2])
        self.assertEqual(builder.shape_type[s2], newton.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[s2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[s2]), np.array(tf), tol=1.0e-4)
        # make sure the original mesh is not modified
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.indices), 36)

    def test_add_particles_grouping(self):
        """Test that add_particles correctly assigns world groups."""
        builder = ModelBuilder()

        # Test with default group (-1)
        builder.add_particles(
            pos=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 3, mass=[1.0] * 3
        )

        # Change to group 0 and add more particles
        builder.current_world = 0
        builder.add_particles(pos=[(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 2, mass=[1.0] * 2)

        # Finalize and check groups
        model = builder.finalize()
        particle_groups = model.particle_world.numpy()

        # First 3 particles should be in group -1
        self.assertTrue(np.all(particle_groups[0:3] == -1))
        # Next 2 particles should be in group 0
        self.assertTrue(np.all(particle_groups[3:5] == 0))

    def test_world_grouping(self):
        """Test world grouping functionality for Model entities."""
        main_builder = ModelBuilder()

        # Create global entities (group -1)
        main_builder.current_world = -1
        ground_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_box(
            body=ground_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=5.0, hy=5.0, hz=0.1
        )
        main_builder.add_particle((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), mass=1.0)

        # Create a simple builder for worlds
        def create_world_builder():
            world_builder = ModelBuilder()
            # Add particles
            p1 = world_builder.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0)
            p2 = world_builder.add_particle((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), mass=1.0)
            world_builder.add_spring(p1, p2, ke=100.0, kd=1.0, control=0.0)

            # Add articulated body
            world_builder.add_articulation()
            b1 = world_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=10.0)
            b2 = world_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=5.0)
            world_builder.add_joint_revolute(parent=b1, child=b2, axis=(0, 1, 0))
            world_builder.add_shape_sphere(
                body=b1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1
            )
            world_builder.add_shape_sphere(
                body=b2, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.05
            )

            return world_builder

        # Add world 0
        world0_builder = create_world_builder()
        main_builder.add_builder(
            world0_builder, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), world=0
        )

        # Add world 1
        world1_builder = create_world_builder()
        main_builder.add_builder(
            world1_builder, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), world=1
        )

        # Add world 2 (testing auto-assignment)
        world2_builder = create_world_builder()
        main_builder.add_builder(
            world2_builder, xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity())
        )  # should get world 2

        # Finalize the model
        model = main_builder.finalize()

        # Verify counts
        self.assertEqual(model.num_worlds, 3)
        self.assertEqual(model.particle_count, 7)  # 1 global + 2*3 = 7
        self.assertEqual(model.body_count, 7)  # 1 global + 2*3 = 7
        self.assertEqual(model.shape_count, 7)  # 1 global + 2*3 = 7
        self.assertEqual(model.joint_count, 3)  # 0 global + 1*3 = 3
        self.assertEqual(model.articulation_count, 3)  # 0 global + 1*3 = 3

        # Verify group assignments
        particle_groups = model.particle_world.numpy() if model.particle_world is not None else []
        body_groups = model.body_world.numpy() if model.body_world is not None else []
        shape_worlds = model.shape_world.numpy() if model.shape_world is not None else []
        joint_worlds = model.joint_world.numpy() if model.joint_world is not None else []
        articulation_groups = model.articulation_world.numpy() if model.articulation_world is not None else []

        if len(particle_groups) > 0:
            # Check global entities
            self.assertEqual(particle_groups[0], -1)  # global particle

            # Check world 0 entities (indices 1-2 for particles)
            self.assertTrue(np.all(particle_groups[1:3] == 0))

            # Check world 1 entities
            self.assertTrue(np.all(particle_groups[3:5] == 1))

            # Check world 2 entities (auto-assigned)
            self.assertTrue(np.all(particle_groups[5:7] == 2))

        if len(body_groups) > 0:
            self.assertEqual(body_groups[0], -1)  # ground body
            self.assertTrue(np.all(body_groups[1:3] == 0))
            self.assertTrue(np.all(body_groups[3:5] == 1))
            self.assertTrue(np.all(body_groups[5:7] == 2))

        if len(shape_worlds) > 0:
            self.assertEqual(shape_worlds[0], -1)  # ground shape
            self.assertTrue(np.all(shape_worlds[1:3] == 0))
            self.assertTrue(np.all(shape_worlds[3:5] == 1))
            self.assertTrue(np.all(shape_worlds[5:7] == 2))

        if len(joint_worlds) > 0:
            self.assertEqual(joint_worlds[0], 0)
            self.assertEqual(joint_worlds[1], 1)
            self.assertEqual(joint_worlds[2], 2)

        if len(articulation_groups) > 0:
            self.assertEqual(articulation_groups[0], 0)
            self.assertEqual(articulation_groups[1], 1)
            self.assertEqual(articulation_groups[2], 2)

    def test_num_worlds_tracking(self):
        """Test that num_worlds is properly tracked when using add_builder with worlds."""
        main_builder = ModelBuilder()

        # Create a simple sub-builder
        sub_builder = ModelBuilder()
        sub_builder.add_body(mass=1.0)

        # Test 1: Global entities should not increment num_worlds
        self.assertEqual(main_builder.num_worlds, 0)
        main_builder.add_builder(sub_builder, world=-1, update_num_world_count=True)
        self.assertEqual(main_builder.num_worlds, 0)  # Should still be 0

        # Test 2: Auto-increment with world=None
        main_builder.add_builder(sub_builder, world=None, update_num_world_count=True)
        self.assertEqual(main_builder.num_worlds, 1)

        main_builder.add_builder(sub_builder, world=None, update_num_world_count=True)
        self.assertEqual(main_builder.num_worlds, 2)

        # Test 3: Explicit world indices
        main_builder2 = ModelBuilder()

        # Add world 3 directly (skipping 0, 1, 2)
        main_builder2.add_builder(sub_builder, world=3, update_num_world_count=True)
        self.assertEqual(main_builder2.num_worlds, 4)  # Should be 3+1

        # Add world 1 (should not change num_worlds since 4 > 1+1)
        main_builder2.add_builder(sub_builder, world=1, update_num_world_count=True)
        self.assertEqual(main_builder2.num_worlds, 4)  # Should still be 4

        # Add world 5 (should increase to 6)
        main_builder2.add_builder(sub_builder, world=5, update_num_world_count=True)
        self.assertEqual(main_builder2.num_worlds, 6)  # Should be 5+1

        # Test 4: update_num_world_count=False should not change num_worlds
        main_builder3 = ModelBuilder()
        main_builder3.add_builder(sub_builder, world=2, update_num_world_count=False)
        self.assertEqual(main_builder3.num_worlds, 0)  # Should remain 0

    def test_collapse_fixed_joints_with_groups(self):
        """Test that collapse_fixed_joints correctly preserves world groups."""
        builder = ModelBuilder()

        # World 0: Chain with fixed joints
        builder.current_world = 0
        b0_0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_2 = builder.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world so collapse_fixed_joints processes this chain
        builder.add_joint_revolute(
            parent=-1,
            child=b0_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Add fixed joint (will be collapsed)
        builder.add_joint_fixed(
            parent=b0_0, child=b0_1, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )

        # Add revolute joint (will be retained)
        builder.add_joint_revolute(
            parent=b0_1,
            child=b0_2,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 1.0, 0.0),
        )

        # World 1: Another chain
        builder.current_world = 1
        b1_0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)
        b1_1 = builder.add_body(xform=wp.transform(wp.vec3(1.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world
        builder.add_joint_revolute(
            parent=-1,
            child=b1_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(1.0, 0.0, 0.0),
        )

        # Add revolute joint
        builder.add_joint_revolute(
            parent=b1_0,
            child=b1_1,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Global body (not connected to world via joints, will be ignored by collapse)
        builder.current_world = -1
        builder.add_body(xform=wp.transform(wp.vec3(0.0, -5.0, 0.0), wp.quat_identity()), mass=0.0)

        # Check groups before collapse
        self.assertEqual(builder.body_world, [0, 0, 0, 1, 1, -1])
        self.assertEqual(builder.joint_world, [0, 0, 0, 1, 1])  # 5 joints now

        # Collapse fixed joints
        builder.collapse_fixed_joints(verbose=False)

        # After collapse:
        # - b0_0 and b0_1 are merged (b0_1 removed)
        # - Fixed joint is removed
        # - Remaining bodies: b0_0 (merged), b0_2, b1_0, b1_1
        # - Note: global_body is removed because it's not connected to world
        # - Remaining joints: world->b0_0, b0_0->b0_2, world->b1_0, b1_0->b1_1

        self.assertEqual(builder.body_count, 4)  # Two bodies removed (b0_1 merged, global_body removed)
        self.assertEqual(builder.joint_count, 4)  # One joint removed (fixed joint)

        # Check that groups are preserved correctly
        self.assertEqual(builder.body_world, [0, 0, 1, 1])  # Groups preserved for retained bodies
        self.assertEqual(builder.joint_world, [0, 0, 1, 1])  # Groups preserved for retained joints

        # Finalize and verify
        model = builder.finalize()
        body_groups = model.body_world.numpy()
        joint_worlds = model.joint_world.numpy()

        # Verify body groups
        self.assertEqual(body_groups[0], 0)  # Merged b0_0
        self.assertEqual(body_groups[1], 0)  # b0_2
        self.assertEqual(body_groups[2], 1)  # b1_0
        self.assertEqual(body_groups[3], 1)  # b1_1

        # Verify joint groups (world connections and body-to-body joints)
        self.assertEqual(joint_worlds[0], 0)  # world->b0_0 from world 0
        self.assertEqual(joint_worlds[1], 0)  # b0_0->b0_2 from world 0
        self.assertEqual(joint_worlds[2], 1)  # world->b1_0 from world 1
        self.assertEqual(joint_worlds[3], 1)  # b1_0->b1_1 from world 1

    def test_add_builder(self):
        orig_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.5, 0.6, 0.7))
        offset_xform = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_rpy(-0.7, 0.8, -0.9))

        fixed_base = ModelBuilder()
        fixed_base.add_body(xform=orig_xform)
        fixed_base.add_joint_revolute(parent=-1, child=0, parent_xform=orig_xform)
        fixed_base.add_shape_sphere(body=0, xform=orig_xform)

        floating_base = ModelBuilder()
        floating_base.add_body(xform=orig_xform)
        floating_base.add_joint_free(parent=-1, child=0)
        floating_base.add_shape_sphere(body=0, xform=orig_xform)

        static_shape = ModelBuilder()
        static_shape.add_shape_sphere(body=-1, xform=orig_xform)

        builder = ModelBuilder()
        builder.add_builder(fixed_base, xform=offset_xform)
        builder.add_builder(floating_base, xform=offset_xform)
        builder.add_builder(static_shape, xform=offset_xform)

        self.assertEqual(builder.body_count, 2)
        self.assertEqual(builder.joint_count, 2)
        self.assertEqual(builder.articulation_count, 2)
        self.assertEqual(builder.shape_count, 3)
        self.assertEqual(builder.body_world, [0, 1])
        self.assertEqual(builder.joint_world, [0, 1])
        self.assertEqual(builder.joint_type, [newton.JointType.REVOLUTE, newton.JointType.FREE])
        self.assertEqual(builder.joint_parent, [-1, -1])
        self.assertEqual(builder.joint_child, [0, 1])
        self.assertEqual(builder.joint_q_start, [0, 1])
        self.assertEqual(builder.joint_qd_start, [0, 1])
        self.assertEqual(builder.shape_world, [0, 1, 2])
        self.assertEqual(builder.shape_body, [0, 1, -1])
        self.assertEqual(builder.body_shapes, {0: [0], 1: [1], -1: [2]})
        self.assertEqual(builder.body_q[0], offset_xform * orig_xform)
        self.assertEqual(builder.body_q[1], offset_xform * orig_xform)
        # fixed base has updated parent transform
        assert_np_equal(np.array(builder.joint_X_p[0]), np.array(offset_xform * orig_xform), tol=1.0e-6)
        # floating base has updated joint coordinates
        assert_np_equal(np.array(builder.joint_q[1:]), np.array(offset_xform * orig_xform), tol=1.0e-6)
        # shapes with a parent body keep the original transform
        assert_np_equal(np.array(builder.shape_transform[0]), np.array(orig_xform), tol=1.0e-6)
        assert_np_equal(np.array(builder.shape_transform[1]), np.array(orig_xform), tol=1.0e-6)
        # static shape receives the offset transform
        assert_np_equal(np.array(builder.shape_transform[2]), np.array(offset_xform * orig_xform), tol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
