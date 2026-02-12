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
import warnings

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
            b0 = builder.add_link()
            builder.add_shape_box(body=b0, **unit_cube)
            j0 = builder.add_joint_fixed(
                parent=parent_body, child=b0, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0))
            )
            b1 = builder.add_link()
            builder.add_shape_box(body=b1, **unit_cube)
            j1 = builder.add_joint_fixed(
                parent=parent_body, child=b1, parent_xform=wp.transform(wp.vec3(0.0, 1.0, 0.0))
            )
            b2 = builder.add_link()
            builder.add_shape_box(body=b2, **unit_cube)
            j2 = builder.add_joint_fixed(
                parent=parent_body, child=b2, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0))
            )
            return b2, [j0, j1, j2]

        builder = ModelBuilder()
        # only fixed joints
        last_body, joints = add_three_cubes(builder)
        builder.add_articulation(joints)
        assert builder.joint_count == 3
        assert builder.body_count == 3

        # fixed joints followed by a non-fixed joint
        last_body, joints = add_three_cubes(builder)
        assert builder.joint_count == 6
        assert builder.body_count == 6
        assert builder.articulation_count == 1  # Only one articulation created so far
        b3 = builder.add_link()
        builder.add_shape_box(
            body=b3, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg, xform=wp.transform(wp.vec3(1.0, 2.0, 3.0))
        )
        joints.append(builder.add_joint_revolute(parent=last_body, child=b3, axis=wp.vec3(0.0, 1.0, 0.0)))
        builder.add_articulation(joints)
        assert builder.articulation_count == 2  # Now we have two articulations

        # a non-fixed joint followed by fixed joints
        free_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.4, 0.5, 0.6))
        b4 = builder.add_link(xform=free_xform)
        builder.add_shape_box(body=b4, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        j_free = builder.add_joint_free(parent=-1, child=b4, parent_xform=wp.transform(wp.vec3(0.0, -1.0, 0.0)))
        assert_np_equal(builder.body_q[b4], np.array(free_xform))
        assert_np_equal(builder.joint_q[-7:], np.array(free_xform))
        assert builder.joint_count == 8
        assert builder.body_count == 8
        _last_body2, joints2 = add_three_cubes(builder, parent_body=b4)
        all_joints = [j_free, *joints2]
        builder.add_articulation(all_joints)
        assert builder.articulation_count == 3  # Three articulations total

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

    def test_lock_inertia_on_shape_addition(self):
        builder = ModelBuilder()
        shape_cfg = ModelBuilder.ShapeConfig(density=1000.0)
        base_com = wp.vec3(0.1, 0.2, 0.3)
        base_inertia = wp.mat33(0.2, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4)

        locked_body = builder.add_link(mass=2.0, com=base_com, inertia=base_inertia, lock_inertia=True)
        unlocked_body = builder.add_link(mass=2.0, com=base_com, inertia=base_inertia, lock_inertia=False)

        locked_mass = builder.body_mass[locked_body]
        locked_com = builder.body_com[locked_body]
        locked_inertia = builder.body_inertia[locked_body]

        unlocked_mass = builder.body_mass[unlocked_body]

        builder.add_shape_box(body=locked_body, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)
        builder.add_shape_box(body=unlocked_body, hx=0.5, hy=0.5, hz=0.5, cfg=shape_cfg)

        self.assertEqual(builder.body_mass[locked_body], locked_mass)
        assert_np_equal(np.array(builder.body_com[locked_body]), np.array(locked_com))
        assert_np_equal(np.array(builder.body_inertia[locked_body]), np.array(locked_inertia))
        self.assertNotEqual(builder.body_mass[unlocked_body], unlocked_mass)

    def test_collapse_fixed_joints_with_locked_inertia(self):
        builder = ModelBuilder()
        b0 = builder.add_link(mass=1.0, lock_inertia=True)
        j0 = builder.add_joint_free(b0)
        b1 = builder.add_link(mass=2.0, lock_inertia=True)
        j1 = builder.add_joint_fixed(parent=b0, child=b1)
        builder.add_articulation([j0, j1])

        builder.collapse_fixed_joints()

        self.assertEqual(builder.body_count, 1)
        self.assertAlmostEqual(builder.body_mass[0], 3.0)
        self.assertTrue(builder.body_lock_inertia[0])

    def test_add_world_with_open_edges(self):
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
            builder.add_world(world_builder, xform)

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
        self.assertEqual(builder.shape_type[s0], newton.GeoType.CONVEX_MESH)
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

        # test keep_visual_shapes
        s3 = builder.add_shape_mesh(body=-1, mesh=mesh)
        builder.approximate_meshes(method="convex_hull", shape_indices=[s3], keep_visual_shapes=True)
        # approximation is created, but not visible
        self.assertEqual(len(builder.shape_source[s3].vertices), 5)
        self.assertEqual(builder.shape_type[s3], newton.GeoType.CONVEX_MESH)
        self.assertEqual(builder.shape_flags[s3] & newton.ShapeFlags.VISIBLE, 0)
        # a new visual shape is created
        self.assertIs(builder.shape_source[s3 + 1], mesh)
        self.assertEqual(builder.shape_flags[s3 + 1] & newton.ShapeFlags.VISIBLE, newton.ShapeFlags.VISIBLE)

        # make sure the original mesh is not modified
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.indices), 36)

    def test_approximate_meshes_collision_filter_child_bodies(self):
        def normalize_pair(a, b):
            return (min(a, b), max(a, b))

        def get_filter_set(builder):
            return {normalize_pair(a, b) for a, b in builder.shape_collision_filter_pairs}

        builder = ModelBuilder()

        # Create a chain of 3 bodies (like an articulation)
        body0 = builder.add_link()
        body1 = builder.add_link()
        body2 = builder.add_link()

        # Add initial shapes to each body (like mesh shapes before decomposition)
        shape0_initial = builder.add_shape_sphere(body=body0, radius=0.1)
        shape1_initial = builder.add_shape_sphere(body=body1, radius=0.1)
        shape2_initial = builder.add_shape_sphere(body=body2, radius=0.1)

        # Create joints (establishes parent->child relationships)
        # body0 is parent of body1, body1 is parent of body2
        joint_free = builder.add_joint_free(parent=-1, child=body0)
        joint0 = builder.add_joint_revolute(parent=body0, child=body1, axis=(0, 0, 1))
        joint1 = builder.add_joint_revolute(parent=body1, child=body2, axis=(0, 0, 1))
        builder.add_articulation(joints=[joint_free, joint0, joint1])

        # At this point, initial shapes should be filtered between adjacent bodies
        filter_set = get_filter_set(builder)
        self.assertIn(
            normalize_pair(shape0_initial, shape1_initial),
            filter_set,
            "Initial body0-body1 shapes should be filtered",
        )
        self.assertIn(
            normalize_pair(shape1_initial, shape2_initial),
            filter_set,
            "Initial body1-body2 shapes should be filtered",
        )

        # Now simulate what approximate_meshes() does: add additional shapes to bodies
        # after joints are already created (like convex decomposition adding multiple parts)
        shape0_extra1 = builder.add_shape_box(body=body0, hx=0.1, hy=0.1, hz=0.1)
        shape0_extra2 = builder.add_shape_capsule(body=body0, radius=0.05, half_height=0.1)
        shape1_extra1 = builder.add_shape_box(body=body1, hx=0.1, hy=0.1, hz=0.1)

        filter_set = get_filter_set(builder)

        # Verify: new body0 shapes should filter with ALL body1 shapes (including initial)
        for parent_shape in [shape0_extra1, shape0_extra2]:
            for child_shape in [shape1_initial, shape1_extra1]:
                expected_pair = normalize_pair(parent_shape, child_shape)
                self.assertIn(
                    expected_pair,
                    filter_set,
                    f"New parent body0 shape {parent_shape} should filter with body1 shape {child_shape}",
                )

        # Verify: new body1 shapes should filter with ALL body0 shapes (parent)
        for child_shape in [shape1_extra1]:
            for parent_shape in [shape0_initial, shape0_extra1, shape0_extra2]:
                expected_pair = normalize_pair(parent_shape, child_shape)
                self.assertIn(
                    expected_pair,
                    filter_set,
                    f"New body1 shape {child_shape} should filter with parent body0 shape {parent_shape}",
                )

        # Verify: new body1 shapes should filter with ALL body2 shapes (child)
        for parent_shape in [shape1_extra1]:
            expected_pair = normalize_pair(parent_shape, shape2_initial)
            self.assertIn(
                expected_pair,
                filter_set,
                f"New body1 shape {parent_shape} should filter with child body2 shape {shape2_initial}",
            )

    def test_add_particles_grouping(self):
        """Test that add_particles correctly assigns world groups."""
        builder = ModelBuilder()

        # Test with default group (-1)
        builder.add_particles(
            pos=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 3, mass=[1.0] * 3
        )

        # Change to world 0 and add more particles
        builder.begin_world()
        builder.add_particles(pos=[(3.0, 0.0, 0.0), (4.0, 0.0, 0.0)], vel=[(0.0, 0.0, 0.0)] * 2, mass=[1.0] * 2)
        builder.end_world()

        # Finalize and check groups
        model = builder.finalize()
        particle_groups = model.particle_world.numpy()

        # First 3 particles should be in group -1
        self.assertTrue(np.all(particle_groups[0:3] == -1))
        # Next 2 particles should be in group 0
        self.assertTrue(np.all(particle_groups[3:5] == 0))

    def test_world_grouping(self):
        """Test world grouping functionality for Model entities."""
        # Optionally enable debug printing
        verbose = False  # Set to True to enable debug output

        # Create builder with a mix of global and world-specific entities
        main_builder = ModelBuilder()

        # Create global entities (world -1)
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
            b1 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=10.0)
            b2 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()), mass=5.0)
            b3 = world_builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=2.5)
            j1 = world_builder.add_joint_revolute(parent=b1, child=b2, axis=(0, 1, 0))
            j2 = world_builder.add_joint_revolute(parent=b2, child=b3, axis=(0, 1, 0))
            world_builder.add_articulation([j1, j2])
            world_builder.add_shape_sphere(
                body=b1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.1
            )
            world_builder.add_shape_sphere(
                body=b2, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.05
            )
            world_builder.add_shape_sphere(
                body=b3, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.025
            )

            return world_builder

        # Add world 0
        world0_builder = create_world_builder()
        main_builder.add_world(world0_builder, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()))

        # Add world 1
        world1_builder = create_world_builder()
        main_builder.add_world(world1_builder, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))

        # Add world 2
        world2_builder = create_world_builder()
        main_builder.add_world(world2_builder, xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))

        # Add more global entities to end of the model
        floor_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_box(
            body=floor_body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), hx=5.0, hy=5.0, hz=0.1
        )
        ball_body = main_builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.0)
        main_builder.add_shape_sphere(
            body=ball_body, xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), radius=0.5
        )
        main_builder.add_particle((0.0, 0.0, 5.0), (0.0, 0.0, 0.0), mass=1.0)
        main_builder.add_particle((0.0, 0.0, 5.5), (0.0, 0.0, 0.0), mass=1.0)

        # Finalize the model
        model = main_builder.finalize()

        # Verify counts
        self.assertEqual(model.num_worlds, 3)
        self.assertEqual(model.particle_count, 9)  # 3 global + 2*3 = 9
        self.assertEqual(model.body_count, 12)  # 3 global + 3*3 = 12
        self.assertEqual(model.shape_count, 12)  # 3 global + 3*3 = 12
        self.assertEqual(model.joint_count, 9)  # 3 global + 2*3 = 9
        self.assertEqual(model.articulation_count, 6)  # 3 global + 1*3 = 6

        # Verify group assignments
        particle_world = model.particle_world.numpy() if model.particle_world is not None else []
        body_world = model.body_world.numpy() if model.body_world is not None else []
        shape_world = model.shape_world.numpy() if model.shape_world is not None else []
        joint_world = model.joint_world.numpy() if model.joint_world is not None else []
        articulation_world = model.articulation_world.numpy() if model.articulation_world is not None else []

        if len(particle_world) > 0:
            # Check global entities
            self.assertEqual(particle_world[0], -1)  # global particle at front
            self.assertEqual(particle_world[-2], -1)  # global particle at back
            self.assertEqual(particle_world[-1], -1)  # global particle at back

            # Check world 0 entities (indices for particles)
            self.assertTrue(np.all(particle_world[1:3] == 0))

            # Check world 1 entities (auto-assigned)
            self.assertTrue(np.all(particle_world[3:5] == 1))

            # Check world 2 entities (auto-assigned)
            self.assertTrue(np.all(particle_world[5:7] == 2))

        if len(body_world) > 0:
            self.assertEqual(body_world[0], -1)  # ground body
            self.assertTrue(np.all(body_world[1:4] == 0))
            self.assertTrue(np.all(body_world[4:7] == 1))
            self.assertTrue(np.all(body_world[7:10] == 2))
            self.assertEqual(body_world[10], -1)  # floor body
            self.assertEqual(body_world[11], -1)  # ball body

        if len(shape_world) > 0:
            self.assertEqual(shape_world[0], -1)  # ground shape
            self.assertTrue(np.all(shape_world[1:4] == 0))
            self.assertTrue(np.all(shape_world[4:7] == 1))
            self.assertTrue(np.all(shape_world[7:10] == 2))
            self.assertEqual(shape_world[10], -1)  # floor shape
            self.assertEqual(shape_world[11], -1)  # ball shape

        if len(joint_world) > 0:
            self.assertEqual(joint_world[0], -1)  # ground body's free joint
            self.assertEqual(joint_world[1], 0)
            self.assertEqual(joint_world[2], 0)
            self.assertEqual(joint_world[3], 1)
            self.assertEqual(joint_world[4], 1)
            self.assertEqual(joint_world[5], 2)
            self.assertEqual(joint_world[6], 2)
            self.assertEqual(joint_world[7], -1)  # floor body's free joint
            self.assertEqual(joint_world[8], -1)  # ball body's free joint

        if len(articulation_world) > 0:
            self.assertEqual(articulation_world[0], -1)  # ground body's articulation
            self.assertEqual(articulation_world[1], 0)
            self.assertEqual(articulation_world[2], 1)
            self.assertEqual(articulation_world[3], 2)
            self.assertEqual(articulation_world[4], -1)  # floor body's articulation
            self.assertEqual(articulation_world[5], -1)  # ball body's articulation

        # Verify world start indices
        particle_world_start = model.particle_world_start.numpy() if model.particle_world_start is not None else []
        body_world_start = model.body_world_start.numpy() if model.body_world_start is not None else []
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else []
        joint_world_start = model.joint_world_start.numpy() if model.joint_world_start is not None else []
        articulation_world_start = (
            model.articulation_world_start.numpy() if model.articulation_world_start is not None else []
        )
        equality_constraint_world_start = (
            model.equality_constraint_world_start.numpy() if model.equality_constraint_world_start is not None else []
        )
        joint_dof_world_start = model.joint_dof_world_start.numpy() if model.joint_dof_world_start is not None else []
        joint_coord_world_start = (
            model.joint_coord_world_start.numpy() if model.joint_coord_world_start is not None else []
        )
        joint_constraint_world_start = (
            model.joint_constraint_world_start.numpy() if model.joint_constraint_world_start is not None else []
        )

        # Optional console-output for debugging
        if verbose:
            print(f"particle_world_start: {particle_world_start}")
            print(f"body_world_start: {body_world_start}")
            print(f"shape_world_start: {shape_world_start}")
            print(f"joint_world_start: {joint_world_start}")
            print(f"articulation_world_start: {articulation_world_start}")
            print(f"equality_constraint_world_start: {equality_constraint_world_start}")
            print(f"joint_dof_world_start: {joint_dof_world_start}")
            print(f"joint_coord_world_start: {joint_coord_world_start}")
            print(f"joint_constraint_world_start: {joint_constraint_world_start}")

        # Check that sizes match num_worlds + 2, i.e. conforms to spec
        self.assertEqual(particle_world_start.size, model.num_worlds + 2)
        self.assertEqual(body_world_start.size, model.num_worlds + 2)
        self.assertEqual(shape_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_world_start.size, model.num_worlds + 2)
        self.assertEqual(articulation_world_start.size, model.num_worlds + 2)
        self.assertEqual(equality_constraint_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_dof_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_coord_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_constraint_world_start.size, model.num_worlds + 2)

        # Check that the last elements match total counts
        self.assertEqual(particle_world_start[-1], model.particle_count)
        self.assertEqual(body_world_start[-1], model.body_count)
        self.assertEqual(shape_world_start[-1], model.shape_count)
        self.assertEqual(joint_world_start[-1], model.joint_count)
        self.assertEqual(articulation_world_start[-1], model.articulation_count)
        self.assertEqual(equality_constraint_world_start[-1], model.equality_constraint_count)
        self.assertEqual(joint_dof_world_start[-1], model.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], model.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], model.joint_constraint_count)

        # Check that world starts are non-decreasing
        for i in range(model.num_worlds + 1):
            self.assertLessEqual(particle_world_start[i], particle_world_start[i + 1])
            self.assertLessEqual(body_world_start[i], body_world_start[i + 1])
            self.assertLessEqual(shape_world_start[i], shape_world_start[i + 1])
            self.assertLessEqual(joint_world_start[i], joint_world_start[i + 1])
            self.assertLessEqual(articulation_world_start[i], articulation_world_start[i + 1])
            self.assertLessEqual(equality_constraint_world_start[i], equality_constraint_world_start[i + 1])
            self.assertLessEqual(joint_dof_world_start[i], joint_dof_world_start[i + 1])
            self.assertLessEqual(joint_coord_world_start[i], joint_coord_world_start[i + 1])
            self.assertLessEqual(joint_constraint_world_start[i], joint_constraint_world_start[i + 1])

        # Check exact values of world starts for this specific case
        self.assertTrue(np.array_equal(particle_world_start, np.array([1, 3, 5, 7, 9])))
        self.assertTrue(np.array_equal(body_world_start, np.array([1, 4, 7, 10, 12])))
        self.assertTrue(np.array_equal(shape_world_start, np.array([1, 4, 7, 10, 12])))
        self.assertTrue(np.array_equal(joint_world_start, np.array([1, 3, 5, 7, 9])))
        self.assertTrue(np.array_equal(articulation_world_start, np.array([1, 2, 3, 4, 6])))
        self.assertTrue(np.array_equal(equality_constraint_world_start, np.array([0, 0, 0, 0, 0])))
        self.assertTrue(np.array_equal(joint_dof_world_start, np.array([6, 8, 10, 12, 24])))
        self.assertTrue(np.array_equal(joint_coord_world_start, np.array([7, 9, 11, 13, 27])))
        self.assertTrue(np.array_equal(joint_constraint_world_start, np.array([0, 10, 20, 30, 30])))

    def test_num_worlds_tracking(self):
        """Test that num_worlds is properly tracked when using add_world."""
        main_builder = ModelBuilder()

        # Create a simple sub-builder
        sub_builder = ModelBuilder()
        sub_builder.add_body(mass=1.0)

        # Test 1: Global entities should not increment num_worlds
        self.assertEqual(main_builder.num_worlds, 0)
        main_builder.add_builder(sub_builder)  # Adds to global world (-1)
        self.assertEqual(main_builder.num_worlds, 0)  # Should still be 0

        # Test 2: Using add_world() for automatic world management
        main_builder.add_world(sub_builder)
        self.assertEqual(main_builder.num_worlds, 1)

        main_builder.add_world(sub_builder)
        self.assertEqual(main_builder.num_worlds, 2)

        # Test 3: Using begin_world/end_world
        main_builder2 = ModelBuilder()

        # Add worlds in sequence
        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)
        main_builder2.end_world()
        self.assertEqual(main_builder2.num_worlds, 1)

        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)
        main_builder2.end_world()
        self.assertEqual(main_builder2.num_worlds, 2)

        # Test 4: Adding to same world using begin_world with existing index
        main_builder2.begin_world()
        main_builder2.add_builder(sub_builder)  # Adds to world 2
        main_builder2.add_builder(sub_builder)  # Also adds to world 2
        main_builder2.end_world()
        self.assertEqual(main_builder2.num_worlds, 3)  # Should now be 3

    def test_world_validation_errors(self):
        """Test that world validation catches non-contiguous and non-monotonic world indices."""
        # Test non-contiguous worlds
        builder1 = ModelBuilder()
        sub_builder = ModelBuilder()
        sub_builder.add_body(mass=1.0)

        # Create world 0 and world 2, skipping world 1
        # We need to manually manipulate world indices to create invalid cases
        builder1.add_world(sub_builder)  # Creates world 0
        # Manually skip world 1 by incrementing num_worlds
        builder1.num_worlds = 2
        builder1.begin_world()  # This will be world 2
        builder1.add_builder(sub_builder)
        builder1.end_world()

        # Should raise error about non-contiguous worlds
        with self.assertRaises(ValueError) as cm:
            builder1.finalize()
        self.assertIn("not contiguous", str(cm.exception))

        # Test non-monotonic worlds
        # This is harder to create with the new API since worlds are always added in order
        # We'll have to directly manipulate the world arrays
        builder2 = ModelBuilder()
        builder2.add_world(sub_builder)  # World 0
        builder2.add_world(sub_builder)  # World 1
        # Manually swap world indices to create non-monotonic ordering
        builder2.body_world[0], builder2.body_world[1] = builder2.body_world[1], builder2.body_world[0]

        # Should raise error about non-monotonic ordering
        with self.assertRaises(ValueError) as cm:
            builder2.finalize()
        self.assertIn("monotonic", str(cm.exception))

    def test_world_context_errors(self):
        """Test error handling for begin_world() and end_world()."""
        # Test calling begin_world() twice without end_world()
        builder1 = ModelBuilder()
        builder1.begin_world()
        with self.assertRaises(RuntimeError) as cm:
            builder1.begin_world()
        self.assertIn("Cannot begin a new world", str(cm.exception))
        self.assertIn("already in world context", str(cm.exception))

        # Test calling end_world() without begin_world()
        builder2 = ModelBuilder()
        with self.assertRaises(RuntimeError) as cm:
            builder2.end_world()
        self.assertIn("Cannot end world", str(cm.exception))
        self.assertIn("not currently in a world context", str(cm.exception))

        # Test that we can still use the builder correctly after proper usage
        builder3 = ModelBuilder()
        builder3.begin_world()
        builder3.add_body()
        builder3.end_world()
        model = builder3.finalize()
        self.assertEqual(model.num_worlds, 1)

        # Test world index out of range (above num_worlds-1)
        builder4 = ModelBuilder()
        builder4.begin_world()  # Creates world 0
        builder4.add_body()
        builder4.end_world()
        # Manually set world index above valid range
        builder4.body_world[0] = 5  # num_worlds=1, so valid range is -1 to 0
        with self.assertRaises(ValueError) as cm:
            builder4.finalize()
        self.assertIn("Invalid world index", str(cm.exception))

        # Test world index below -1 (invalid)
        builder5 = ModelBuilder()
        builder5.begin_world()
        builder5.add_body()
        builder5.end_world()
        # Manually set an invalid world index below -1
        builder5.body_world[0] = -2
        with self.assertRaises(ValueError) as cm:
            builder5.finalize()
        self.assertIn("Invalid world index", str(cm.exception))

    def test_collapse_fixed_joints_with_groups(self):
        """Test that collapse_fixed_joints correctly preserves world groups."""
        # Optionally enable debug printing
        verbose = False  # Set to True to enable debug output

        # Create builder with multiple worlds and fixed joints
        builder = ModelBuilder()

        # World 0: Chain with fixed joints
        builder.begin_world()
        b0_0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_1 = builder.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)
        b0_2 = builder.add_link(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world so collapse_fixed_joints processes this chain
        j0_0 = builder.add_joint_revolute(
            parent=-1,
            child=b0_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Add fixed joint (will be collapsed)
        j0_1 = builder.add_joint_fixed(
            parent=b0_0, child=b0_1, parent_xform=wp.transform_identity(), child_xform=wp.transform_identity()
        )

        # Add revolute joint (will be retained)
        j0_2 = builder.add_joint_revolute(
            parent=b0_1,
            child=b0_2,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 1.0, 0.0),
        )
        # Create articulation for world 0
        builder.add_articulation([j0_0, j0_1, j0_2])

        builder.end_world()

        # World 1: Another chain
        builder.begin_world()
        b1_0 = builder.add_link(xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)
        b1_1 = builder.add_link(xform=wp.transform(wp.vec3(1.0, 2.0, 0.0), wp.quat_identity()), mass=1.0)

        # Connect to world
        j1_0 = builder.add_joint_revolute(
            parent=-1,
            child=b1_0,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(1.0, 0.0, 0.0),
        )

        # Add revolute joint
        j1_1 = builder.add_joint_revolute(
            parent=b1_0,
            child=b1_1,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=(0.0, 0.0, 1.0),
        )

        # Create articulation for world 1
        builder.add_articulation([j1_0, j1_1])

        builder.end_world()

        # Global body (connected to world via free joint)
        # Using add_body for a standalone body with free joint
        builder.add_body(xform=wp.transform(wp.vec3(0.0, -5.0, 0.0), wp.quat_identity()), mass=0.0)

        # Check worlds before collapse
        self.assertEqual(builder.body_world, [0, 0, 0, 1, 1, -1])
        self.assertEqual(builder.joint_world, [0, 0, 0, 1, 1, -1])  # 6 joints now (includes free joint from add_body)

        # Collapse fixed joints
        builder.collapse_fixed_joints(verbose=verbose)

        # After collapse:
        # - b0_0 and b0_1 are merged (b0_1 removed)
        # - Fixed joint is removed
        # - Remaining bodies: b0_0 (merged), b0_2, b1_0, b1_1, global_body
        # - Note: global_body is now retained because it's connected to world via free joint
        # - Remaining joints: world->b0_0, b0_0->b0_2, world->b1_0, b1_0->b1_1, world->global_body (free joint)

        self.assertEqual(builder.body_count, 5)  # One body removed (b0_1 merged)
        self.assertEqual(builder.joint_count, 5)  # One joint removed (fixed joint)

        # Check that groups are preserved correctly
        self.assertEqual(builder.body_world, [0, 0, 1, 1, -1])  # Groups preserved for retained bodies
        self.assertEqual(builder.joint_world, [0, 0, 1, 1, -1])  # Groups preserved for retained joints

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

        # Verify world start indices
        particle_world_start = model.particle_world_start.numpy() if model.particle_world_start is not None else []
        body_world_start = model.body_world_start.numpy() if model.body_world_start is not None else []
        shape_world_start = model.shape_world_start.numpy() if model.shape_world_start is not None else []
        joint_world_start = model.joint_world_start.numpy() if model.joint_world_start is not None else []
        articulation_world_start = (
            model.articulation_world_start.numpy() if model.articulation_world_start is not None else []
        )
        equality_constraint_world_start = (
            model.equality_constraint_world_start.numpy() if model.equality_constraint_world_start is not None else []
        )
        joint_dof_world_start = model.joint_dof_world_start.numpy() if model.joint_dof_world_start is not None else []
        joint_coord_world_start = (
            model.joint_coord_world_start.numpy() if model.joint_coord_world_start is not None else []
        )
        joint_constraint_world_start = (
            model.joint_constraint_world_start.numpy() if model.joint_constraint_world_start is not None else []
        )

        # Optional console-output for debugging
        if verbose:
            print(f"particle_world_start: {particle_world_start}")
            print(f"body_world_start: {body_world_start}")
            print(f"shape_world_start: {shape_world_start}")
            print(f"joint_world_start: {joint_world_start}")
            print(f"articulation_world_start: {articulation_world_start}")
            print(f"equality_constraint_world_start: {equality_constraint_world_start}")
            print(f"joint_dof_world_start: {joint_dof_world_start}")
            print(f"joint_coord_world_start: {joint_coord_world_start}")
            print(f"joint_constraint_world_start: {joint_constraint_world_start}")

        # Verify total counts
        self.assertEqual(builder.particle_count, 0)
        self.assertEqual(builder.body_count, 5)
        self.assertEqual(builder.shape_count, 0)
        self.assertEqual(builder.joint_count, 5)
        self.assertEqual(builder.articulation_count, 3)
        self.assertEqual(len(builder.equality_constraint_world), 0)
        self.assertEqual(builder.joint_dof_count, 10)
        self.assertEqual(builder.joint_coord_count, 11)
        self.assertEqual(builder.joint_constraint_count, 20)
        self.assertEqual(particle_world_start[-1], builder.particle_count)
        self.assertEqual(body_world_start[-1], builder.body_count)
        self.assertEqual(shape_world_start[-1], builder.shape_count)
        self.assertEqual(joint_world_start[-1], builder.joint_count)
        self.assertEqual(articulation_world_start[-1], builder.articulation_count)
        self.assertEqual(equality_constraint_world_start[-1], len(builder.equality_constraint_world))
        self.assertEqual(joint_dof_world_start[-1], builder.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], builder.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], builder.joint_constraint_count)

        # Check that sizes match num_worlds + 2, i.e. conforms to spec
        self.assertEqual(particle_world_start.size, model.num_worlds + 2)
        self.assertEqual(body_world_start.size, model.num_worlds + 2)
        self.assertEqual(shape_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_world_start.size, model.num_worlds + 2)
        self.assertEqual(articulation_world_start.size, model.num_worlds + 2)
        self.assertEqual(equality_constraint_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_dof_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_coord_world_start.size, model.num_worlds + 2)
        self.assertEqual(joint_constraint_world_start.size, model.num_worlds + 2)

        # Check that the last elements match total counts
        self.assertEqual(particle_world_start[-1], model.particle_count)
        self.assertEqual(body_world_start[-1], model.body_count)
        self.assertEqual(shape_world_start[-1], model.shape_count)
        self.assertEqual(joint_world_start[-1], model.joint_count)
        self.assertEqual(articulation_world_start[-1], model.articulation_count)
        self.assertEqual(equality_constraint_world_start[-1], model.equality_constraint_count)
        self.assertEqual(joint_dof_world_start[-1], model.joint_dof_count)
        self.assertEqual(joint_coord_world_start[-1], model.joint_coord_count)
        self.assertEqual(joint_constraint_world_start[-1], model.joint_constraint_count)

        # Check that world starts are non-decreasing
        for i in range(model.num_worlds + 1):
            self.assertLessEqual(particle_world_start[i], particle_world_start[i + 1])
            self.assertLessEqual(body_world_start[i], body_world_start[i + 1])
            self.assertLessEqual(shape_world_start[i], shape_world_start[i + 1])
            self.assertLessEqual(joint_world_start[i], joint_world_start[i + 1])
            self.assertLessEqual(articulation_world_start[i], articulation_world_start[i + 1])
            self.assertLessEqual(equality_constraint_world_start[i], equality_constraint_world_start[i + 1])
            self.assertLessEqual(joint_dof_world_start[i], joint_dof_world_start[i + 1])
            self.assertLessEqual(joint_coord_world_start[i], joint_coord_world_start[i + 1])
            self.assertLessEqual(joint_constraint_world_start[i], joint_constraint_world_start[i + 1])

        # Check exact values of world starts for this specific case
        self.assertTrue(np.array_equal(particle_world_start, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(body_world_start, np.array([0, 2, 4, 5])))
        self.assertTrue(np.array_equal(shape_world_start, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(joint_world_start, np.array([0, 2, 4, 5])))
        self.assertTrue(np.array_equal(articulation_world_start, np.array([0, 1, 2, 3])))
        self.assertTrue(np.array_equal(equality_constraint_world_start, np.array([0, 0, 0, 0])))
        self.assertTrue(np.array_equal(joint_dof_world_start, np.array([0, 2, 4, 10])))
        self.assertTrue(np.array_equal(joint_coord_world_start, np.array([0, 2, 4, 11])))
        self.assertTrue(np.array_equal(joint_constraint_world_start, np.array([0, 10, 20, 20])))

    def test_add_world(self):
        orig_xform = wp.transform(wp.vec3(1.0, 2.0, 3.0), wp.quat_rpy(0.5, 0.6, 0.7))
        offset_xform = wp.transform(wp.vec3(4.0, 5.0, 6.0), wp.quat_rpy(-0.7, 0.8, -0.9))

        fixed_base = ModelBuilder()
        b0 = fixed_base.add_link(xform=orig_xform)
        j0 = fixed_base.add_joint_revolute(parent=-1, child=b0, parent_xform=orig_xform)
        fixed_base.add_articulation([j0])
        fixed_base.add_shape_sphere(body=b0, xform=orig_xform)

        floating_base = ModelBuilder()
        b1 = floating_base.add_link(xform=orig_xform)
        j1 = floating_base.add_joint_free(parent=-1, child=b1)
        floating_base.add_articulation([j1])
        floating_base.add_shape_sphere(body=b1, xform=orig_xform)

        static_shape = ModelBuilder()
        static_shape.add_shape_sphere(body=-1, xform=orig_xform)

        builder = ModelBuilder()
        builder.add_world(fixed_base, xform=offset_xform)
        builder.add_world(floating_base, xform=offset_xform)
        builder.add_world(static_shape, xform=offset_xform)

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

    def test_articulation_validation_contiguous(self):
        """Test that articulation requires contiguous joint indices"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)
        link3 = builder.add_link(mass=1.0)
        link4 = builder.add_link(mass=1.0)

        # Create joints
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)
        joint3 = builder.add_joint_revolute(parent=link2, child=link3)
        joint4 = builder.add_joint_revolute(parent=link3, child=link4)

        # Test valid contiguous articulation
        builder.add_articulation([joint1, joint2, joint3, joint4])  # Should work

        # Test non-contiguous articulation should fail
        builder2 = ModelBuilder()
        link1 = builder2.add_link(mass=1.0)
        link2 = builder2.add_link(mass=1.0)
        link3 = builder2.add_link(mass=1.0)

        j1 = builder2.add_joint_revolute(parent=-1, child=link1)
        j2 = builder2.add_joint_revolute(parent=link1, child=link2)
        # Create a joint for another articulation to create a gap
        other_link = builder2.add_link(mass=1.0)
        _j_other = builder2.add_joint_revolute(parent=-1, child=other_link)
        j3 = builder2.add_joint_revolute(parent=link2, child=link3)

        # This should fail because [j1, j2, j3] are not contiguous (j_other is in between)
        with self.assertRaises(ValueError) as context:
            builder2.add_articulation([j1, j2, j3])
        self.assertIn("contiguous", str(context.exception))

    def test_articulation_validation_monotonic(self):
        """Test that articulation requires monotonically increasing joint indices"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)

        # Create joints
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)

        # Test joints in wrong order (not monotonic)
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint2, joint1])  # Wrong order
        self.assertIn("monotonically increasing", str(context.exception))

    def test_articulation_validation_empty(self):
        """Test that articulation requires at least one joint"""
        builder = ModelBuilder()

        # Test empty articulation should fail
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([])
        self.assertIn("no joints", str(context.exception))

    def test_articulation_validation_world_mismatch(self):
        """Test that all joints in articulation must belong to same world"""
        builder = ModelBuilder()

        # Create joints in world 0
        builder.begin_world()
        link1 = builder.add_link(mass=1.0)
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        builder.end_world()

        # Create joint in world 1
        builder.begin_world()
        link2 = builder.add_link(mass=1.0)
        joint2 = builder.add_joint_revolute(parent=-1, child=link2)

        # Try to create articulation from joints in different worlds (while still in world 1)
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint1, joint2])
        self.assertIn("world", str(context.exception).lower())
        builder.end_world()

    def test_articulation_validation_tree_structure(self):
        """Test that articulation validates tree structure (no multiple parents)"""
        builder = ModelBuilder()

        # Create links
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)
        link3 = builder.add_link(mass=1.0)

        # Create joints that would form invalid tree (link2 has two parents)
        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)
        joint3 = builder.add_joint_revolute(parent=link3, child=link2)  # link2 already has parent link1

        # This should fail because link2 has multiple parents
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint1, joint2, joint3])
        self.assertIn("multiple parents", str(context.exception))

    def test_articulation_validation_duplicate_joint(self):
        """Test that adding a joint to multiple articulations raises an error"""
        builder = ModelBuilder()

        # Create links and joints
        link1 = builder.add_link(mass=1.0)
        link2 = builder.add_link(mass=1.0)

        joint1 = builder.add_joint_revolute(parent=-1, child=link1)
        joint2 = builder.add_joint_revolute(parent=link1, child=link2)

        # Add joints to first articulation
        builder.add_articulation([joint1, joint2])

        # Create another joint
        link3 = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_revolute(parent=link2, child=link3)

        # Try to add joint2 (already in articulation) to a new articulation
        with self.assertRaises(ValueError) as context:
            builder.add_articulation([joint2, joint3])
        self.assertIn("already belongs to articulation", str(context.exception))
        self.assertIn("joint_2", str(context.exception))  # joint2's key

    def test_joint_world_validation(self):
        """Test that joints validate parent/child bodies belong to current world"""
        builder = ModelBuilder()

        # Create body in world 0
        builder.begin_world()
        link1 = builder.add_link(mass=1.0)
        builder.end_world()

        # Switch to world 1 and try to create joint with body from world 0
        builder.begin_world()
        link2 = builder.add_link(mass=1.0)

        # This should fail because link1 is in world 0 but we're in world 1
        with self.assertRaises(ValueError) as context:
            builder.add_joint_revolute(parent=link1, child=link2)
        self.assertIn("world", str(context.exception).lower())
        builder.end_world()

    def test_articulation_validation_orphan_joint(self):
        """Test that joints not belonging to an articulation raise an error on finalize."""
        builder = ModelBuilder()
        body = builder.add_link()

        # Add joint but do NOT add it to an articulation
        builder.add_joint_revolute(parent=-1, child=body, key="orphan_joint")

        # finalize() should raise ValueError about orphan joints
        with self.assertRaises(ValueError) as context:
            builder.finalize()

        self.assertIn("not belonging to any articulation", str(context.exception))
        self.assertIn("orphan_joint", str(context.exception))

    def test_articulation_validation_multiple_orphan_joints(self):
        """Test error message shows multiple orphan joints."""
        builder = ModelBuilder()
        body1 = builder.add_link()
        body2 = builder.add_link()

        # Add multiple joints without articulations
        builder.add_joint_revolute(parent=-1, child=body1, key="first_joint")
        builder.add_joint_revolute(parent=body1, child=body2, key="second_joint")

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("2 joint(s)", error_msg)
        self.assertIn("first_joint", error_msg)
        self.assertIn("second_joint", error_msg)

    def test_shape_thickness_exceeds_contact_margin_warning(self):
        """Test that a warning is raised when shape thickness > contact_margin."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create a shape with thickness > contact_margin (should trigger warning)
        cfg = ModelBuilder.ShapeConfig()
        cfg.thickness = 0.01
        cfg.contact_margin = 0.005  # Less than thickness
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg, key="bad_sphere")

        # Should warn about thickness > contact_margin
        with self.assertWarns(UserWarning) as cm:
            builder.finalize()

        warning_msg = str(cm.warning)
        self.assertIn("thickness > contact_margin", warning_msg)
        self.assertIn("bad_sphere", warning_msg)
        self.assertIn("missed collisions", warning_msg)

    def test_shape_thickness_within_contact_margin_no_warning(self):
        """Test that no warning is raised when shape thickness <= contact_margin."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create a shape with thickness <= contact_margin (should not trigger warning)
        cfg = ModelBuilder.ShapeConfig()
        cfg.thickness = 0.005
        cfg.contact_margin = 0.01  # Greater than thickness
        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg)

        # Should NOT warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.finalize()
            # Filter for our specific warning about thickness
            thickness_warnings = [warning for warning in w if "thickness > contact_margin" in str(warning.message)]
            self.assertEqual(len(thickness_warnings), 0, "Unexpected warning about thickness > contact_margin")

    def test_shape_thickness_warning_multiple_shapes(self):
        """Test that the warning correctly reports multiple shapes with thickness > contact_margin."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)

        # Create multiple shapes with thickness > contact_margin
        cfg_bad = ModelBuilder.ShapeConfig()
        cfg_bad.thickness = 0.02
        cfg_bad.contact_margin = 0.01

        builder.add_shape_sphere(body=body, radius=0.5, cfg=cfg_bad, key="sphere1")
        builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5, cfg=cfg_bad, key="box1")

        # One good shape that should not be in the warning
        cfg_good = ModelBuilder.ShapeConfig()
        cfg_good.thickness = 0.005
        cfg_good.contact_margin = 0.01
        builder.add_shape_capsule(body=body, radius=0.2, half_height=0.5, cfg=cfg_good, key="good_capsule")

        with self.assertWarns(UserWarning) as cm:
            builder.finalize()

        warning_msg = str(cm.warning)
        self.assertIn("2 shape(s)", warning_msg)
        self.assertIn("sphere1", warning_msg)
        self.assertIn("box1", warning_msg)
        self.assertNotIn("good_capsule", warning_msg)

    def test_collision_filter_pairs_canonical_order(self):
        """Test that collision filter pairs are stored in canonical order (s1 < s2)."""
        builder = ModelBuilder()

        # Create a body with multiple shapes
        body = builder.add_body()
        shape0 = builder.add_shape_sphere(body=body, radius=0.5)
        shape1 = builder.add_shape_box(body=body, hx=1.0, hy=1.0, hz=1.0)
        shape2 = builder.add_shape_capsule(body=body, radius=0.3, half_height=1.0)

        # Add collision filter pairs in non-canonical order to test normalization
        builder.shape_collision_filter_pairs.append((shape1, shape0))  # reversed order
        builder.shape_collision_filter_pairs.append((shape0, shape2))  # correct order
        builder.shape_collision_filter_pairs.append((shape2, shape1))  # reversed order

        # Finalize the model
        model = builder.finalize()

        # Verify all collision filter pairs are in canonical order (s1 < s2)
        for s1, s2 in model.shape_collision_filter_pairs:
            self.assertLess(s1, s2, f"Collision filter pair ({s1}, {s2}) is not in canonical order")

        # Verify we have the expected pairs (should be normalized to canonical order)
        self.assertIn((shape0, shape1), model.shape_collision_filter_pairs)
        self.assertIn((shape0, shape2), model.shape_collision_filter_pairs)
        self.assertIn((shape1, shape2), model.shape_collision_filter_pairs)

    def test_validate_structure_invalid_shape_body(self):
        """Test that _validate_structure catches invalid shape_body references."""
        builder = ModelBuilder()
        body = builder.add_body(mass=1.0)
        builder.add_shape_sphere(body=body, radius=0.5, key="test_shape")

        # Manually set invalid body reference
        builder.shape_body[0] = 999  # Invalid body index

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("shape_body", error_msg)
        self.assertIn("test_shape", error_msg)
        self.assertIn("999", error_msg)

    def test_validate_structure_invalid_joint_parent(self):
        """Test that _validate_structure catches invalid joint_parent references."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, key="test_joint")
        builder.add_articulation([joint])

        # Manually set invalid parent body reference
        builder.joint_parent[0] = 999  # Invalid body index

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("joint_parent", error_msg)
        self.assertIn("test_joint", error_msg)

    def test_validate_structure_invalid_joint_child(self):
        """Test that _validate_structure catches invalid joint_child references."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, key="test_joint")
        builder.add_articulation([joint])

        # Manually set invalid child body reference (child cannot be -1)
        builder.joint_child[0] = -1  # Invalid: child cannot be world

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("joint_child", error_msg)
        self.assertIn("Child cannot be the world", error_msg)

    def test_validate_structure_self_referential_joint(self):
        """Test that _validate_structure catches self-referential joints."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body, key="self_ref_joint")
        builder.add_articulation([joint])

        # Manually set parent == child (self-referential)
        builder.joint_parent[0] = body
        builder.joint_child[0] = body

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Self-referential joint", error_msg)
        self.assertIn("self_ref_joint", error_msg)

    def test_validate_structure_invalid_equality_constraint_body(self):
        """Test that _validate_structure catches invalid equality constraint body references."""
        builder = ModelBuilder()
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        builder.add_equality_constraint_weld(
            body1=body1,
            body2=body2,
            key="test_constraint",
        )

        # Manually set invalid body reference
        builder.equality_constraint_body1[0] = 999

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid body reference", error_msg)
        self.assertIn("equality_constraint_body1", error_msg)
        self.assertIn("test_constraint", error_msg)

    def test_validate_structure_invalid_equality_constraint_joint(self):
        """Test that _validate_structure catches invalid equality constraint joint references."""
        builder = ModelBuilder()
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint2])

        # Add a joint equality constraint
        builder.add_equality_constraint_joint(
            joint1=joint1,
            joint2=joint2,
            key="joint_constraint",
        )

        # Manually set invalid joint reference
        builder.equality_constraint_joint1[0] = 999

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Invalid joint reference", error_msg)
        self.assertIn("equality_constraint_joint1", error_msg)
        self.assertIn("joint_constraint", error_msg)

    def test_validate_structure_array_length_mismatch(self):
        """Test that _validate_structure catches array length mismatches."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body)
        builder.add_articulation([joint])

        # Manually corrupt array length
        builder.joint_armature.append(0.0)  # Add extra element

        with self.assertRaises(ValueError) as context:
            builder.finalize()

        error_msg = str(context.exception)
        self.assertIn("Array length mismatch", error_msg)
        self.assertIn("joint_armature", error_msg)

    def test_validate_joint_ordering_correct_order(self):
        """Test that validate_joint_ordering passes for correctly ordered joints."""
        builder = ModelBuilder()

        # Create a simple chain in DFS order
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)
        builder.add_articulation([joint1, joint2, joint3])

        # Should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = builder.validate_joint_ordering()
            ordering_warnings = [warning for warning in w if "DFS topological order" in str(warning.message)]
            self.assertEqual(len(ordering_warnings), 0)

        self.assertTrue(result)

    def test_validate_joint_ordering_incorrect_order(self):
        """Test that validate_joint_ordering warns for incorrectly ordered joints."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order: joint3 (body2->body3) comes BEFORE joint2 (body1->body2)
        # This is invalid because body2 hasn't been processed yet when we try to process joint3
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order - parent not processed
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])  # Wrong order: should be [joint1, joint2, joint3]

        # Should warn about non-DFS order
        with self.assertWarns(UserWarning) as cm:
            result = builder.validate_joint_ordering()

        self.assertFalse(result)
        self.assertIn("DFS topological order", str(cm.warning))

    def test_skip_all_validations(self):
        """Test that skip_all_validations skips all validation checks."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, key="orphan_joint")
        # Don't add articulation - this would normally fail _validate_joints

        # Without skip_all_validations, should raise ValueError about orphan joint
        with self.assertRaises(ValueError) as context:
            builder.finalize(skip_all_validations=False)
        self.assertIn("orphan_joint", str(context.exception))

        # With skip_all_validations=True, should NOT raise the validation error
        # Create a fresh builder for clean test
        builder2 = ModelBuilder()
        body2 = builder2.add_link(mass=1.0)
        builder2.add_joint_revolute(parent=-1, child=body2, key="orphan_joint2")
        # This should succeed (validation skipped)
        model = builder2.finalize(skip_all_validations=True)
        self.assertIsNotNone(model)

    def test_skip_validation_structure(self):
        """Test that skip_validation_structure skips structural validation."""
        builder = ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body)
        builder.add_articulation([joint])

        # Manually corrupt array length to trigger structure validation error
        builder.joint_armature.append(0.0)  # Add extra element

        # Without skip_validation_structure, should raise ValueError
        with self.assertRaises(ValueError) as context:
            builder.finalize(skip_validation_structure=False)
        self.assertIn("Array length mismatch", str(context.exception))

        # Create fresh builder with same corruption
        builder2 = ModelBuilder()
        body2 = builder2.add_link(mass=1.0)
        joint2 = builder2.add_joint_revolute(parent=-1, child=body2)
        builder2.add_articulation([joint2])
        builder2.joint_armature.append(0.0)

        # With skip_validation_structure=True, should skip the structure check
        # Model creation will likely fail, but not from structure validation
        try:
            builder2.finalize(skip_validation_structure=True)
        except ValueError as e:
            # If it raises ValueError, it should NOT be about array length mismatch
            self.assertNotIn("Array length mismatch", str(e))

    def test_skip_validation_joint_ordering_default(self):
        """Test that joint ordering validation is skipped by default."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order for the chain
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])

        # By default (skip_validation_joint_ordering=True), should not warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            builder.finalize()
            ordering_warnings = [warning for warning in w if "DFS topological order" in str(warning.message)]
            self.assertEqual(len(ordering_warnings), 0)

    def test_enable_validation_joint_ordering(self):
        """Test that joint ordering validation can be enabled."""
        builder = ModelBuilder()

        # Create a chain: world -> body1 -> body2 -> body3
        body1 = builder.add_link(mass=1.0)
        body2 = builder.add_link(mass=1.0)
        body3 = builder.add_link(mass=1.0)

        # Create joints in WRONG order for the chain
        joint1 = builder.add_joint_revolute(parent=-1, child=body1)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3)  # Out of order
        joint2 = builder.add_joint_revolute(parent=body1, child=body2)
        builder.add_articulation([joint1, joint3, joint2])

        # With skip_validation_joint_ordering=False, should warn
        with self.assertWarns(UserWarning) as cm:
            builder.finalize(skip_validation_joint_ordering=False)

        self.assertIn("DFS topological order", str(cm.warning))

    def test_mimic_constraint_programmatic(self):
        """Test programmatic creation of mimic constraints."""
        builder = newton.ModelBuilder()

        # Create two joints
        b0 = builder.add_body()
        b1 = builder.add_body()
        b2 = builder.add_body()

        j1 = builder.add_joint_revolute(
            parent=-1,
            child=b0,
            axis=(0, 0, 1),
            key="j1",
        )
        j2 = builder.add_joint_revolute(
            parent=-1,
            child=b1,
            axis=(0, 0, 1),
            key="j2",
        )
        j3 = builder.add_joint_revolute(
            parent=-1,
            child=b2,
            axis=(0, 0, 1),
            key="j3",
        )

        # Add mimic constraints
        _c1 = builder.add_constraint_mimic(
            joint0=j2,
            joint1=j1,
            coef0=-0.25,
            coef1=1.5,
            key="mimic1",
        )
        _c2 = builder.add_constraint_mimic(
            joint0=j3,
            joint1=j1,
            coef0=0.0,
            coef1=-1.0,
            enabled=False,
            key="mimic2",
        )

        model = builder.finalize()

        self.assertEqual(model.constraint_mimic_count, 2)

        # Check first constraint
        self.assertEqual(model.constraint_mimic_joint0.numpy()[0], j2)
        self.assertEqual(model.constraint_mimic_joint1.numpy()[0], j1)
        self.assertAlmostEqual(model.constraint_mimic_coef0.numpy()[0], -0.25)
        self.assertAlmostEqual(model.constraint_mimic_coef1.numpy()[0], 1.5)
        self.assertTrue(model.constraint_mimic_enabled.numpy()[0])
        self.assertEqual(model.constraint_mimic_key[0], "mimic1")

        # Check second constraint
        self.assertEqual(model.constraint_mimic_joint0.numpy()[1], j3)
        self.assertEqual(model.constraint_mimic_joint1.numpy()[1], j1)
        self.assertAlmostEqual(model.constraint_mimic_coef0.numpy()[1], 0.0)
        self.assertAlmostEqual(model.constraint_mimic_coef1.numpy()[1], -1.0)
        self.assertFalse(model.constraint_mimic_enabled.numpy()[1])
        self.assertEqual(model.constraint_mimic_key[1], "mimic2")

    def test_control_clear(self):
        """Test that Control.clear() works without errors."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint])

        model = builder.finalize()
        control = model.control()
        try:
            control.clear()
        except Exception as e:
            self.fail(f"control.clear() raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
