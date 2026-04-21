# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.raycast import (
    ray_intersect_box,
    ray_intersect_capsule,
    ray_intersect_cone,
    ray_intersect_cylinder,
    ray_intersect_ellipsoid,
    ray_intersect_geom,
    ray_intersect_mesh,
    ray_intersect_plane,
    ray_intersect_sphere,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestRaycast(unittest.TestCase):
    pass


# Kernels to test ray intersection functions
@wp.kernel
def kernel_test_sphere(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
):
    tid = wp.tid()
    t, _n = ray_intersect_sphere(geom_to_world, ray_origin, ray_direction, r)
    out_t[tid] = t


@wp.kernel
def kernel_test_box(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
):
    tid = wp.tid()
    t, _n = ray_intersect_box(geom_to_world, ray_origin, ray_direction, size)
    out_t[tid] = t


@wp.kernel
def kernel_test_capsule(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    t, _n = ray_intersect_capsule(geom_to_world, ray_origin, ray_direction, r, h)
    out_t[tid] = t


@wp.kernel
def kernel_test_cylinder(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    t, _n = ray_intersect_cylinder(geom_to_world, ray_origin, ray_direction, r, h)
    out_t[tid] = t


@wp.kernel
def kernel_test_cone(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    t, _n = ray_intersect_cone(geom_to_world, ray_origin, ray_direction, r, h)
    out_t[tid] = t


@wp.kernel
def kernel_test_ellipsoid(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    semi_axes: wp.vec3,
):
    tid = wp.tid()
    t, _n = ray_intersect_ellipsoid(geom_to_world, ray_origin, ray_direction, semi_axes)
    out_t[tid] = t


@wp.kernel
def kernel_test_plane(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
):
    tid = wp.tid()
    t, _n = ray_intersect_plane(geom_to_world, ray_origin, ray_direction, size)
    out_t[tid] = t


@wp.kernel
def kernel_test_geom(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
):
    tid = wp.tid()
    t, _n = ray_intersect_geom(geom_to_world, size, geomtype, ray_origin, ray_direction, mesh_id)
    out_t[tid] = t


@wp.kernel
def kernel_test_mesh(
    out_t: wp.array[float],
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
    mesh_id: wp.uint64,
):
    tid = wp.tid()
    t, _n, _u, _v, _f = ray_intersect_mesh(geom_to_world, ray_origin, ray_direction, size, mesh_id, False, 1.0e6)
    out_t[tid] = t


# Test functions
def test_ray_intersect_sphere(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 1.0

    with test.subTest("hit"):
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("miss"):
        ray_origin = wp.vec3(-2.0, 2.0, 0.0)
        wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("inside"):
        ray_origin = wp.vec3(0.0, 0.0, 0.0)
        wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)


def test_ray_intersect_box(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)

    with test.subTest("hit"):
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("miss"):
        ray_origin = wp.vec3(-2.0, 2.0, 0.0)
        wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("inside"):
        ray_origin = wp.vec3(0.0, 0.0, 0.0)
        wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("rotated"):
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.785398)  # pi/4
        geom_to_world = wp.transform(wp.vec3(0.0, 0.0, 0.0), rot)
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
        test.assertAlmostEqual(out_t.numpy()[0], 2.0 - 1.41421, delta=1e-5)


def test_ray_intersect_capsule(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 0.5
    h = 1.0

    with test.subTest("hit_cylinder"):
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    with test.subTest("hit_cap"):
        ray_origin = wp.vec3(0.0, 0.0, -2.0)
        ray_direction = wp.vec3(0.0, 0.0, 1.0)
        wp.launch(
            kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 2.0 - 1.0 - 0.5, delta=1e-5)

    with test.subTest("miss"):
        ray_origin = wp.vec3(-2.0, 2.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)


def test_ray_intersect_cylinder(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 0.5
    h = 1.0

    with test.subTest("hit_body"):
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_cylinder, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    with test.subTest("hit_cap"):
        ray_origin = wp.vec3(0.0, 0.0, -2.0)
        ray_direction = wp.vec3(0.0, 0.0, 1.0)
        wp.launch(
            kernel_test_cylinder, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("miss"):
        ray_origin = wp.vec3(-2.0, 2.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_cylinder, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)


def test_ray_intersect_cone(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 1.0  # base radius
    h = 1.0  # half height (so total height is 2.0)

    with test.subTest("hit_body"):
        ray_origin = wp.vec3(-2.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-3)

    with test.subTest("hit_base"):
        ray_origin = wp.vec3(0.0, 0.0, -2.0)
        ray_direction = wp.vec3(0.0, 0.0, 1.0)
        wp.launch(
            kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-3)  # hits base at z=-1

    with test.subTest("miss"):
        ray_origin = wp.vec3(-2.0, 2.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("hit_tip"):
        ray_origin = wp.vec3(0.0, 0.0, 2.0)
        ray_direction = wp.vec3(0.0, 0.0, -1.0)
        wp.launch(
            kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-3)  # hits tip at z=1


def test_ray_intersect_ellipsoid(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    semi_axes = wp.vec3(1.0, 0.5, 0.5)  # non-uniform to exercise ellipsoid-specific logic

    with test.subTest("hit"):
        ray_origin = wp.vec3(-3.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_ellipsoid,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, semi_axes],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 2.0, delta=1e-5)

    with test.subTest("miss"):
        ray_origin = wp.vec3(-3.0, 1.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_ellipsoid,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, semi_axes],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("inside"):
        ray_origin = wp.vec3(0.0, 0.0, 0.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_ellipsoid,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, semi_axes],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)


def test_ray_intersect_plane(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(0.0, 0.0, 0.0)

    with test.subTest("hit_from_above"):
        ray_origin = wp.vec3(0.0, 0.0, 4.0)
        ray_direction = wp.vec3(3.0, 0.0, -4.0)  # 3-4-5 triple
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("parallel_miss"):
        ray_origin = wp.vec3(0.0, 0.0, 2.0)
        ray_direction = wp.vec3(1.0, 0.0, 0.0)
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("backward_miss"):
        ray_origin = wp.vec3(0.0, 0.0, 5.0)
        ray_direction = wp.vec3(0.0, 0.0, 1.0)
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("translated_plane"):
        geom_to_world = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
        ray_origin = wp.vec3(0.0, 0.0, 7.0)
        ray_direction = wp.vec3(3.0, 0.0, -4.0)  # 3-4-5 triple, local z-offset = 4
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("finite_miss_half_extent"):
        geom_to_world = wp.transform_identity()
        size_finite = wp.vec3(4.0, 4.0, 0.0)  # full width 4, half-extent 2
        ray_origin = wp.vec3(0.0, 0.0, 4.0)
        ray_direction = wp.vec3(3.0, 0.0, -4.0)  # 3-4-5 triple, crosses plane at (3, 0, 0) -- |3| > 2
        wp.launch(
            kernel_test_plane,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, size_finite],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("finite_miss_x"):
        geom_to_world = wp.transform_identity()
        size_finite = wp.vec3(2.0, 2.0, 0.0)  # full width 2, half-extent 1
        ray_origin = wp.vec3(0.0, 0.0, 4.0)
        ray_direction = wp.vec3(3.0, 0.0, -4.0)  # 3-4-5 triple, hits at (3, 0, 0) -- |3| > 1
        wp.launch(
            kernel_test_plane,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, size_finite],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("finite_miss_y"):
        geom_to_world = wp.transform_identity()
        size_finite = wp.vec3(10.0, 2.0, 0.0)
        ray_origin = wp.vec3(0.0, 0.0, 4.0)
        ray_direction = wp.vec3(0.0, 3.0, -4.0)  # 3-4-5 triple, hits at (0, 3, 0) -- |3| > half 2 = 1
        wp.launch(
            kernel_test_plane,
            dim=1,
            inputs=[out_t, geom_to_world, ray_origin, ray_direction, size_finite],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    with test.subTest("hit_from_below"):
        geom_to_world = wp.transform_identity()
        ray_origin = wp.vec3(0.0, 0.0, -4.0)
        ray_direction = wp.vec3(0.0, 3.0, 4.0)  # 3-4-5 triple
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("rotated_plane"):
        q_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 3.14159265 / 2.0)
        geom_to_world = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_rot)
        ray_origin = wp.vec3(0.0, -5.0, 0.0)
        ray_direction = wp.vec3(0.0, 1.0, 0.0)
        wp.launch(
            kernel_test_plane, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device
        )
        test.assertAlmostEqual(out_t.numpy()[0], 5.0, delta=1e-5)


def test_geom_ray_intersect(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    mesh_id = wp.uint64(0)  # No mesh for primitive shapes

    with test.subTest("plane"):
        size = wp.vec3(0.0, 0.0, 0.0)
        ray_origin_plane = wp.vec3(0.0, 0.0, 5.0)
        ray_direction_plane = wp.vec3(0.0, 0.0, -1.0)
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.PLANE, ray_origin_plane, ray_direction_plane, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 5.0, delta=1e-5)

    with test.subTest("sphere"):
        size = wp.vec3(1.0, 0.0, 0.0)  # r
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.SPHERE, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("box"):
        size = wp.vec3(1.0, 1.0, 1.0)  # half-extents
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.BOX, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    with test.subTest("capsule"):
        size = wp.vec3(0.5, 1.0, 0.0)  # r, h
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.CAPSULE, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    with test.subTest("cylinder"):
        size = wp.vec3(0.5, 1.0, 0.0)  # r, h
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.CYLINDER, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    with test.subTest("cone"):
        size = wp.vec3(1.0, 1.0, 0.0)  # r, h
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.CONE, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-3)

    with test.subTest("ellipsoid"):
        size = wp.vec3(1.0, 0.5, 0.5)  # semi-axes (rx, ry, rz)
        wp.launch(
            kernel_test_geom,
            dim=1,
            inputs=[out_t, geom_to_world, size, GeoType.ELLIPSOID, ray_origin, ray_direction, mesh_id],
            device=device,
        )
        test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)


def test_ray_intersect_mesh(test: TestRaycast, device: str):
    """Test mesh raycasting using a simple quad made of two triangles."""
    out_t = wp.zeros(1, dtype=float, device=device)

    # Create a simple quad mesh (2x2 quad at z=0)
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],  # bottom left
            [1.0, -1.0, 0.0],  # bottom right
            [1.0, 1.0, 0.0],  # top right
            [-1.0, 1.0, 0.0],  # top left
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            [0, 1, 2],  # first triangle
            [0, 2, 3],  # second triangle
        ],
        dtype=np.int32,
    ).flatten()

    # Create Newton mesh and finalize to get Warp mesh
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    # Test cases
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)  # no scaling

    # Case 1: Ray hits the quad from above
    ray_origin = wp.vec3(0.0, 0.0, 2.0)
    ray_direction = wp.vec3(0.0, 0.0, -1.0)
    wp.launch(
        kernel_test_mesh,
        dim=1,
        inputs=[out_t, geom_to_world, ray_origin, ray_direction, size, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 2.0, delta=1e-3)  # Should hit at z=0, distance=2

    # Case 2: Ray hits the quad from below
    ray_origin = wp.vec3(0.0, 0.0, -2.0)
    ray_direction = wp.vec3(0.0, 0.0, 1.0)
    wp.launch(
        kernel_test_mesh,
        dim=1,
        inputs=[out_t, geom_to_world, ray_origin, ray_direction, size, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 2.0, delta=1e-3)  # Should hit at z=0, distance=2

    # Case 3: Ray misses the quad
    ray_origin = wp.vec3(2.0, 2.0, 2.0)  # Outside quad bounds
    ray_direction = wp.vec3(0.0, 0.0, -1.0)
    wp.launch(
        kernel_test_mesh,
        dim=1,
        inputs=[out_t, geom_to_world, ray_origin, ray_direction, size, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)  # Should miss

    # Case 4: Ray hits quad at angle
    ray_origin = wp.vec3(-2.0, 0.0, 1.0)
    ray_direction = wp.vec3(1.0, 0.0, -0.5)  # Angled ray
    ray_direction = wp.normalize(ray_direction)
    wp.launch(
        kernel_test_mesh,
        dim=1,
        inputs=[out_t, geom_to_world, ray_origin, ray_direction, size, mesh_id],
        device=device,
    )
    # Calculate expected distance: ray hits quad at x=0, z=0
    # Ray equation: (-2, 0, 1) + t*(1, 0, -0.5) = (0, 0, 0)
    # -2 + t = 0 -> t = 2
    # 1 - 0.5*t = 0 -> t = 2
    expected_dist = 2.0 * np.sqrt(1.0**2 + 0.5**2)  # |t| * |direction|
    test.assertAlmostEqual(out_t.numpy()[0], expected_dist, delta=1e-3)


def test_mesh_ray_intersect_via_geom(test: TestRaycast, device: str):
    """Test mesh raycasting through the ray_intersect_geom interface."""
    out_t = wp.zeros(1, dtype=float, device=device)

    # Create a simple triangle mesh
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],  # Triangle pointing up
        ],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2], dtype=np.int32)

    # Create and finalize mesh
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    # Test ray hitting the triangle
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)
    ray_origin = wp.vec3(0.0, 0.0, 2.0)
    ray_direction = wp.vec3(0.0, 0.0, -1.0)

    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.MESH, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 2.0, delta=1e-3)  # Should hit triangle at z=0


def test_convex_hull_ray_intersect_via_geom(test: TestRaycast, device: str):
    """Test convex hull raycasting through the ray_intersect_geom interface (uses mesh path)."""
    out_t = wp.zeros(1, dtype=float, device=device)

    # Create a simple triangle mesh (convex by definition)
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    indices = np.array([0, 1, 2], dtype=np.int32)

    # Create and finalize mesh
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    # Test ray hitting the triangle
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)
    ray_origin = wp.vec3(0.0, 0.0, 2.0)
    ray_direction = wp.vec3(0.0, 0.0, -1.0)

    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.CONVEX_MESH, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 2.0, delta=1e-3)  # Should hit triangle at z=0


devices = get_test_devices()
add_function_test(TestRaycast, "test_ray_intersect_plane", test_ray_intersect_plane, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_sphere", test_ray_intersect_sphere, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_box", test_ray_intersect_box, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_capsule", test_ray_intersect_capsule, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_cylinder", test_ray_intersect_cylinder, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_cone", test_ray_intersect_cone, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_ellipsoid", test_ray_intersect_ellipsoid, devices=devices)
add_function_test(TestRaycast, "test_geom_ray_intersect", test_geom_ray_intersect, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_mesh", test_ray_intersect_mesh, devices=devices)
add_function_test(TestRaycast, "test_mesh_ray_intersect_via_geom", test_mesh_ray_intersect_via_geom, devices=devices)
add_function_test(
    TestRaycast, "test_convex_hull_ray_intersect_via_geom", test_convex_hull_ray_intersect_via_geom, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
