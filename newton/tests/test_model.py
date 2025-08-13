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
from newton.geometry.utils import create_box_mesh, transform_points
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
        assert builder.joint_type == [newton.JOINT_REVOLUTE, newton.JOINT_FREE]
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

        env_builder = ModelBuilder()
        env_builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.1, 0.1, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.25),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
        )

        num_envs = 2
        env_offsets = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        builder_open_edge_count = np.sum(np.array(builder.edge_indices) == -1)
        env_builder_open_edge_count = np.sum(np.array(env_builder.edge_indices) == -1)

        for i in range(num_envs):
            xform = wp.transform(env_offsets[i], wp.quat_identity())
            builder.add_builder(
                env_builder,
                xform,
                update_num_env_count=True,
                separate_collision_group=True,
            )

        self.assertEqual(
            np.sum(np.array(builder.edge_indices) == -1),
            builder_open_edge_count + num_envs * env_builder_open_edge_count,
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
        self.assertEqual(builder.shape_type[s1], newton.geometry.GeoType.BOX)
        assert_np_equal(npsorted(builder.shape_scale[s1]), npsorted(scale), tol=1.0e-6)
        # only compare the position since the rotation is not guaranteed to be the same
        assert_np_equal(np.array(builder.shape_transform[s1].p), np.array(tf.p), tol=1.0e-4)
        # bounding sphere
        self.assertIsNone(builder.shape_source[s2])
        self.assertEqual(builder.shape_type[s2], newton.geometry.GeoType.SPHERE)
        self.assertAlmostEqual(builder.shape_scale[s2][0], wp.length(scale))
        assert_np_equal(np.array(builder.shape_transform[s2]), np.array(tf), tol=1.0e-4)
        # make sure the original mesh is not modified
        self.assertEqual(len(mesh.vertices), 8)
        self.assertEqual(len(mesh.indices), 36)


if __name__ == "__main__":
    unittest.main()
