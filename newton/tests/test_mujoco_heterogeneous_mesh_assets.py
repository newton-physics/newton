# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve native collision metadata for every heterogeneous mesh asset."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_mujoco_heterogeneous_native import (
    _advance,
    _build_stack_model,
    _create_simulation,
    _read_contact_pairs,
)
from newton.tests.unittest_utils import get_test_devices


def _make_prism_mesh(polygon, cap_triangles):
    """Extrude an authored polygon with outward triangle faces and nonzero volume."""
    count = len(polygon)
    vertices = np.array([(x, y, z) for z in (-0.04, 0.04) for x, y in polygon], dtype=np.float32)
    faces = [(a, c, b) for a, b, c in cap_triangles]
    faces.extend((a + count, b + count, c + count) for a, b, c in cap_triangles)
    for first in range(count):
        second = (first + 1) % count
        faces.extend(((first, second, second + count), (first, second + count, first + count)))
    return newton.Mesh(vertices, np.asarray(faces, dtype=np.int32).reshape(-1), compute_inertia=False)


def _build_asset_model(meshes, device):
    """Reserve representative slots in world zero while later worlds use distinct mesh assets."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    for world, mesh in enumerate(meshes):
        builder.begin_world()
        body = builder.add_link(
            mass=2.0,
            inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])),
            com=wp.vec3(0.01, -0.02, 0.03),
        )
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        for part in range(2 if world == 0 else 1):
            builder.add_shape_convex_hull(
                body,
                mesh=mesh,
                cfg=cfg,
                scale=(0.8 + world * 0.1, 1.1, 0.9),
                xform=wp.transform((0.6 * part, 0.03, -0.02), wp.quat_rpy(0.2, -0.1, 0.3)),
            )
        builder.end_world()
    return builder.finalize(device=device)


class TestMuJoCoHeterogeneousMeshAssets(unittest.TestCase):
    def test_nonrepresentative_meshes_preserve_multicontact_manifolds(self):
        """Match native contact manifolds for meshes absent from the compiled slot representatives."""
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _build_stack_model([(1, 1, False), (2, 2, False), (1, 1, False)], device)
                coordinates = model.joint_q.numpy().reshape(3, 2, 7)
                coordinates[:, 1, 2] = 0.27
                model.joint_q.assign(coordinates.reshape(-1))
                for multiccd in (False, True):
                    with self.subTest(multiccd=multiccd):
                        simulation = _advance(_create_simulation(model, enable_multiccd=multiccd), 1)
                        pairs = _read_contact_pairs(simulation)
                        body_pairs = pairs[np.all(model.shape_body.numpy()[pairs] >= 0, axis=1)]
                        worlds = model.shape_world.numpy()[body_pairs[:, 0]]
                        counts = np.bincount(worlds, minlength=3)
                        # Worlds zero and two have identical geometry, but only world
                        # zero contributes representative meshes to the compiled geoms.
                        expected_count = 4 if multiccd else 1
                        np.testing.assert_array_equal(counts[[0, 2]], [expected_count, expected_count])

                        solver = simulation[0]
                        data = solver.mjw_data
                        count = int(data.nacon.numpy()[0])
                        worlds = data.contact.worldid.numpy()[:count]
                        shape_map = solver.mjc_geom_to_newton_shape.numpy()
                        shapes = shape_map[worlds[:, None], data.contact.geom.numpy()[:count]]
                        body_contacts = np.all(model.shape_body.numpy()[shapes] >= 0, axis=1)
                        manifold = np.column_stack(
                            (
                                data.contact.pos.numpy()[:count],
                                data.contact.dist.numpy()[:count],
                                data.contact.frame.numpy()[:count].reshape(count, 9),
                            )
                        )
                        reference = manifold[body_contacts & (worlds == 0)]
                        actual = manifold[body_contacts & (worlds == 2)]
                        reference = reference[np.lexsort(reference[:, :3].T[::-1])]
                        actual = actual[np.lexsort(actual[:, :3].T[::-1])]
                        np.testing.assert_allclose(actual, reference, atol=2.0e-6)

                        data_ids = solver.mjw_model.geom_dataid.numpy()
                        mesh_ids = np.unique(data_ids[data_ids >= 0])
                        self.assertTrue(np.all(solver.mjw_model.mesh_graphadr.numpy()[mesh_ids] >= 0))
                        self.assertTrue(np.all(solver.mjw_model.mesh_polynum.numpy()[mesh_ids] > 0))

    def test_nonrepresentative_mesh_sets_multicontact_capacity(self):
        """Size polygon scratch storage for a twelve-sided mesh absent from representative geoms."""
        angles = np.arange(12) * (2.0 * np.pi / 12)
        polygon = np.column_stack((0.14 * np.cos(angles), 0.14 * np.sin(angles)))
        prism = _make_prism_mesh(polygon, [(0, index, index + 1) for index in range(1, 11)])
        box = newton.Mesh.create_box(0.12, 0.09, 0.04, compute_inertia=False)
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _build_asset_model([box, prism], device)
                solver = _create_simulation(model, enable_multiccd=True)[0]
                shape_map = solver.mjc_geom_to_newton_shape.numpy()
                geom = np.flatnonzero(shape_map[1] >= 0)[0]
                mesh_id = solver.mjw_model.geom_dataid.numpy()[1, geom]
                first_polygon = solver.mjw_model.mesh_polyadr.numpy()[mesh_id]
                polygon_count = solver.mjw_model.mesh_polynum.numpy()[mesh_id]
                vertices_per_polygon = solver.mjw_model.mesh_polyvertnum.numpy()[
                    first_polygon : first_polygon + polygon_count
                ]
                self.assertGreater(polygon_count, 0)
                self.assertGreaterEqual(np.max(vertices_per_polygon), 12)
                self.assertGreaterEqual(solver.mjw_model.npolygonmax, 12)
                self.assertGreaterEqual(solver.mjw_model.nmeshdegmax, 3)

    def test_mesh_inertia_choice_preserves_authored_geometry_and_body_properties(self):
        """Preserve asymmetric mesh placement and explicit body inertia through shape pose updates."""
        polygon = [(0.0, 0.0), (0.3, 0.0), (0.3, 0.1), (0.1, 0.1), (0.1, 0.2), (0.0, 0.2)]
        concave = _make_prism_mesh(polygon, [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)])
        box = newton.Mesh.create_box(0.12, 0.09, 0.04, compute_inertia=False)
        for device in get_test_devices():
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _build_asset_model([box, concave], device)
                original_mass = model.body_mass.numpy().copy()
                original_inertia = model.body_inertia.numpy().copy()
                original_com = model.body_com.numpy().copy()
                original_vertices = [mesh.vertices.copy() for mesh in (box, concave)]
                solver = _create_simulation(model)[0]
                shape_map = solver.mjc_geom_to_newton_shape.numpy()
                mesh_ids = solver.mjw_model.geom_dataid.numpy()
                vertices = solver.mjw_model.mesh_vert.numpy()
                vertex_starts = solver.mjw_model.mesh_vertadr.numpy()
                vertex_counts = solver.mjw_model.mesh_vertnum.numpy()
                for updated in (False, True):
                    with self.subTest(updated=updated):
                        if updated:
                            transforms = model.shape_transform.numpy()
                            for shape in range(model.shape_count):
                                transforms[shape] = np.asarray(
                                    wp.transform((0.05 * shape, -0.03, 0.07), wp.quat_rpy(-0.25, 0.15, -0.2))
                                )
                            model.shape_transform.assign(transforms)
                            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
                        shape_poses = model.shape_transform.numpy()
                        scales = model.shape_scale.numpy()
                        geom_positions = solver.mjw_model.geom_pos.numpy()
                        geom_rotations = solver.mjw_model.geom_quat.numpy()
                        for world, geom in np.argwhere(shape_map >= 0):
                            shape = shape_map[world, geom]
                            mesh_id = mesh_ids[world, geom]
                            first = vertex_starts[mesh_id]
                            compiled_vertices = vertices[first : first + vertex_counts[mesh_id]]
                            rotation = geom_rotations[world, geom]
                            geom_pose = wp.transform(geom_positions[world, geom], wp.quat(*rotation[[1, 2, 3, 0]]))
                            shape_pose = wp.transform(*shape_poses[shape])
                            actual = np.array(
                                [wp.transform_point(geom_pose, wp.vec3(vertex)) for vertex in compiled_vertices]
                            )
                            expected = np.array(
                                [
                                    wp.transform_point(shape_pose, wp.vec3(vertex * scales[shape]))
                                    for vertex in model.shape_source[shape].vertices
                                ]
                            )
                            np.testing.assert_allclose(actual, expected, atol=2.0e-6)

                        masses = solver.mjw_model.body_mass.numpy()
                        inertias = solver.mjw_model.body_inertia.numpy()
                        com_positions = solver.mjw_model.body_ipos.numpy()
                        inertia_rotations = solver.mjw_model.body_iquat.numpy()
                        body_map = solver.mjc_body_to_newton.numpy()
                        for world, body in np.argwhere(body_map >= 0):
                            original = body_map[world, body]
                            rotation = inertia_rotations[world, body]
                            matrix = np.asarray(wp.quat_to_matrix(wp.quat(*rotation[[1, 2, 3, 0]]))).reshape(3, 3)
                            np.testing.assert_allclose(masses[world, body], original_mass[original])
                            np.testing.assert_allclose(com_positions[world, body], original_com[original], atol=1.0e-7)
                            np.testing.assert_allclose(
                                matrix @ np.diag(inertias[world, body]) @ matrix.T,
                                original_inertia[original],
                                atol=1.0e-7,
                            )
                np.testing.assert_array_equal(model.body_mass.numpy(), original_mass)
                np.testing.assert_array_equal(model.body_inertia.numpy(), original_inertia)
                for mesh, expected in zip((box, concave), original_vertices, strict=True):
                    np.testing.assert_array_equal(mesh.vertices, expected)


if __name__ == "__main__":
    unittest.main()
