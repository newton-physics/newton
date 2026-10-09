# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover top-face contacts on a wide box with an off-center partner."""

import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_box_contact_nearby_face(test, device, *, split=False):
    """Use the nearby slab face for primitive and convex-mesh partners in either order."""
    half = (0.0126257, 0.0126257, 0.0252514)
    mesh = newton.Mesh.create_box(*half, duplicate_vertices=False, compute_inertia=False)
    rotations = (wp.quat_identity(), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7))
    cases = product((0.03, 1.0), (False, True), (False, True), rotations, (0.0, 0.17), (0.2326, 0.22, 0.216, 0.2552494))
    for thickness, use_mesh, swap, rotation, tilt, height in cases:
        with test.subTest(thickness=thickness, mesh=use_mesh, swap=swap, rotation=rotation, tilt=tilt, height=height):
            frame = wp.transform(wp.vec3(0.0), rotation)
            table_pose = frame * wp.transform(wp.vec3(0.585, 0.0, 0.23 - thickness / 2), wp.quat_identity())
            object_pose = frame * wp.transform(
                wp.vec3(0.735, -0.35, height), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), tilt)
            )
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=object_pose)
            table_shape = -1
            for is_table in (False, True) if swap else (True, False):
                if is_table:
                    table_shape = builder.add_shape_box(
                        body=-1,
                        xform=table_pose,
                        hx=0.3675,
                        hy=0.61,
                        hz=thickness / 2,
                        cfg=builder.ShapeConfig(gap=0.01),
                    )
                elif use_mesh:
                    builder.add_shape_convex_hull(body=body, mesh=mesh, cfg=builder.ShapeConfig(gap=0.001))
                else:
                    builder.add_shape_box(
                        body=body, hx=half[0], hy=half[1], hz=half[2], cfg=builder.ShapeConfig(gap=0.001)
                    )
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            normals = contacts.rigid_contact_normal.numpy()[:count]
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            upward = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)))
            signs = np.where(shape0 == table_shape, 1.0, -1.0)
            np.testing.assert_allclose(normals, signs[:, None] * upward, atol=1.0e-4)
            points = [contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_point1.numpy()[:count]]
            depths = []
            for i in range(count):
                table_side = 0 if shape0[i] == table_shape else 1
                table_point = points[table_side][i]
                object_local = points[1 - table_side][i]
                object_point = np.asarray(wp.transform_point(object_pose, wp.vec3(object_local)))
                table_local = np.asarray(wp.transform_point(wp.transform_inverse(table_pose), wp.vec3(table_point)))
                test.assertAlmostEqual(float(table_local[2]), thickness / 2, delta=1.0e-5)
                test.assertLessEqual(abs(float(np.max(np.abs(object_local) - half))), 1.0e-5)
                depths.append(float(np.dot(table_point - object_point, upward)))
            expected_depth = 0.23 - height + half[2] * np.cos(tilt) + half[0] * np.sin(tilt)
            test.assertAlmostEqual(max(depths), expected_depth, delta=1.0e-5)


class TestBoxContactNormal(unittest.TestCase):
    pass


def test_box_mesh_minimum_depth(test, device):
    """Match an independent Minkowski hull for oblique and edge contacts."""
    from scipy.spatial import ConvexHull

    vertices = np.array([[-0.4, -0.3, -0.2], [0.6, -0.2, -0.25], [0.15, 0.7, -0.3], [0.05, 0.1, 0.7]])
    hull = ConvexHull(vertices)
    mesh = newton.Mesh(vertices, hull.simplices.flatten(), compute_inertia=False)
    half = np.array([0.45, 0.35, 0.3])
    corners = np.array(list(product(*[(-value, value) for value in half])))
    rotations = (wp.quat_identity(), wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, -1.0, 0.5)), 0.39))
    for position, rotation, swap in product(
        ((0.35, 0.15, 0.2), (0.2, 0.25, 0.3), (0.55, 0.2, 0.2)), rotations, (False, True)
    ):
        with test.subTest(position=position, rotation=rotation, swap=swap):
            partner = np.array([wp.quat_rotate(rotation, wp.vec3(vertex)) for vertex in vertices]) + position
            difference = (corners[:, None] - partner[None, :]).reshape(-1, 3)
            planes = ConvexHull(difference).equations
            closest = planes[np.argmax(planes[:, 3])]
            expected_depth = -closest[3]
            test.assertGreater(expected_depth, 0.0)
            expected_normal = closest[:3]
            builder = newton.ModelBuilder()
            pose = wp.transform(wp.vec3(position), rotation)
            body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            box_shape = -1
            for is_box in (False, True) if swap else (True, False):
                if is_box:
                    box_shape = builder.add_shape_box(body=-1, hx=half[0], hy=half[1], hz=half[2])
                else:
                    builder.add_shape_convex_hull(body=body, mesh=mesh, cfg=builder.ShapeConfig(density=0.0))
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            normals = contacts.rigid_contact_normal.numpy()[:count]
            signs = np.where(shape0 == box_shape, 1.0, -1.0)
            np.testing.assert_allclose(normals, signs[:, None] * expected_normal, atol=1.0e-4)
            points = [contacts.rigid_contact_point0.numpy()[:count], contacts.rigid_contact_point1.numpy()[:count]]
            depths = []
            for i in range(count):
                side = 0 if shape0[i] == box_shape else 1
                box_point = points[side][i]
                local = points[1 - side][i]
                mesh_point = np.asarray(wp.transform_point(pose, wp.vec3(local)))
                test.assertLessEqual(abs(float(np.max(np.abs(box_point) - half))), 1.0e-5)
                test.assertLessEqual(abs(float(np.max(hull.equations[:, :3] @ local + hull.equations[:, 3]))), 1.0e-5)
                depths.append(np.dot(box_point - mesh_point, expected_normal))
            test.assertAlmostEqual(max(depths), expected_depth, delta=1.0e-5)


add_function_test(
    TestBoxContactNormal, "test_box_mesh_minimum_depth", test_box_mesh_minimum_depth, devices=get_test_devices()
)


def test_box_mesh_minimum_depth_split(test, device):
    """Check the cooperative axis search against the independent hull oracle."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_box_mesh_minimum_depth(test, device)


add_function_test(
    TestBoxContactNormal,
    "test_box_mesh_minimum_depth_split",
    test_box_mesh_minimum_depth_split,
    devices=get_test_devices(),
)


add_function_test(
    TestBoxContactNormal, "test_box_contact_nearby_face", test_box_contact_nearby_face, devices=get_test_devices()
)


def test_box_contact_nearby_face_split(test, device):
    """Use the nearby slab face through the split CUDA collision kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_box_contact_nearby_face(test, device, split=True)


add_function_test(
    TestBoxContactNormal,
    "test_box_contact_nearby_face_split",
    test_box_contact_nearby_face_split,
    devices=get_test_devices(),
)


def test_box_contact_split_reuse(test, device):
    """Reuse captured box work across overlap, separation, and empty queues."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    half = (0.0126257, 0.0126257, 0.0252514)
    for use_mesh in (False, True):
        builder = newton.ModelBuilder()
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.585, 0.0, 0.215)),
            hx=0.3675,
            hy=0.61,
            hz=0.015,
            cfg=builder.ShapeConfig(gap=0.01),
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0.735, -0.35, 0.2326)))
        if use_mesh:
            builder.add_shape_convex_hull(
                body, mesh=newton.Mesh.create_box(*half, duplicate_vertices=False), cfg=builder.ShapeConfig(gap=0.001)
            )
        else:
            builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=builder.ShapeConfig(gap=0.001))
        model = builder.finalize(device)
        with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
            pipeline = newton.CollisionPipeline(model, broad_phase="sap")
        state = model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        with wp.ScopedCapture(device=device) as capture:
            pipeline.collide(state, contacts)
        poses = state.body_q.numpy()
        for height in (0.2326, 0.256, 0.27, 0.2326):
            with test.subTest(mesh=use_mesh, height=height):
                poses[body, 2] = height
                state.body_q.assign(poses)
                wp.capture_launch(capture.graph)
                count = int(contacts.rigid_contact_count.numpy()[0])
                if height == 0.27:
                    test.assertEqual(count, 0)
                    np.testing.assert_array_equal(pipeline.narrow_phase.split_gjk_work_count.numpy(), (0, 0))
                    continue
                test.assertGreater(count, 0)
                normals = contacts.rigid_contact_normal.numpy()[:count]
                # Normalizing a submillimeter GJK gap amplifies float32
                # cancellation in the world-space witness subtraction.
                normal_tolerance = 3.0e-4 if height == 0.256 else 1.0e-4
                np.testing.assert_allclose(
                    normals, np.broadcast_to((0.0, 0.0, 1.0), normals.shape), atol=normal_tolerance
                )
                table = contacts.rigid_contact_point0.numpy()[:count]
                partner = contacts.rigid_contact_point1.numpy()[:count] + poses[body, :3]
                depth = np.sum((table - partner) * normals, axis=1)
                np.testing.assert_allclose(depth, 0.23 - height + half[2], atol=1.0e-5)


add_function_test(
    TestBoxContactNormal, "test_box_contact_split_reuse", test_box_contact_split_reuse, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
