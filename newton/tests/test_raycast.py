# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.geometry.raycast import ray_intersect_geom, ray_intersect_mesh
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestRaycast(unittest.TestCase):
    pass


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


def test_ray_intersect_sphere(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 0.0, 0.0)  # r
    direction = wp.vec3(1.0, 0.0, 0.0)

    cases = [
        ("hit", wp.vec3(-2.0, 0.0, 0.0), 1.0),
        ("miss", wp.vec3(-2.0, 2.0, 0.0), -1.0),
        ("inside", wp.vec3(0.0, 0.0, 0.0), 1.0),
    ]

    for name, origin, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, geom_to_world, size, GeoType.SPHERE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_box(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    size = wp.vec3(1.0, 1.0, 1.0)  # half-extents
    direction = wp.vec3(1.0, 0.0, 0.0)

    identity = wp.transform_identity()
    rot_45_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 4.0)

    # (name, xform, origin, expected)
    cases = [
        ("hit", identity, wp.vec3(-2.0, 0.0, 0.0), 1.0),
        ("miss", identity, wp.vec3(-2.0, 2.0, 0.0), -1.0),
        ("inside", identity, wp.vec3(0.0, 0.0, 0.0), 1.0),
        ("rotated", wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_45_z), wp.vec3(-2.0, 0.0, 0.0), 2.0 - wp.sqrt(2.0)),
    ]

    for name, xform, origin, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom, dim=1, inputs=[out_t, xform, size, GeoType.BOX, origin, direction, 0], device=device
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_capsule(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h

    # (name, origin, direction, expected)
    cases = [
        ("hit_cylinder", wp.vec3(-2.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), 1.5),
        ("hit_cap", wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0), 0.5),
        ("miss", wp.vec3(-2.0, 2.0, 0.0), wp.vec3(1.0, 0.0, 0.0), -1.0),
    ]

    for name, origin, direction, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, geom_to_world, size, GeoType.CAPSULE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_cylinder(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h

    # (name, origin, direction, expected)
    cases = [
        ("hit_body", wp.vec3(-2.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), 1.5),
        ("hit_cap", wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0), 1.0),
        ("miss", wp.vec3(-2.0, 2.0, 0.0), wp.vec3(1.0, 0.0, 0.0), -1.0),
    ]

    for name, origin, direction, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, geom_to_world, size, GeoType.CYLINDER, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_cone(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 0.0)  # r, h (total height = 2*h)

    # (name, origin, direction, expected, delta)
    cases = [
        ("hit_body", wp.vec3(-2.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), 1.5, 1e-3),
        ("hit_base", wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0), 1.0, 1e-3),  # base at z=-1
        ("hit_tip", wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, -1.0), 1.0, 1e-3),  # tip at z=+1
        ("miss", wp.vec3(-2.0, 2.0, 0.0), wp.vec3(1.0, 0.0, 0.0), -1.0, 1e-5),
    ]

    for name, origin, direction, expected, delta in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, geom_to_world, size, GeoType.CONE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=delta)


def test_ray_intersect_ellipsoid(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 0.5, 0.5)  # semi-axes; non-uniform to exercise ellipsoid-specific logic
    direction = wp.vec3(1.0, 0.0, 0.0)

    cases = [
        ("hit", wp.vec3(-3.0, 0.0, 0.0), 2.0),
        ("miss", wp.vec3(-3.0, 1.0, 0.0), -1.0),
        ("inside", wp.vec3(0.0, 0.0, 0.0), 1.0),
    ]

    for name, origin, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, geom_to_world, size, GeoType.ELLIPSOID, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_plane(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)

    identity = wp.transform_identity()
    infinite = wp.vec3(0.0, 0.0, 0.0)  # unbounded plane

    # Transforms for non-identity cases.
    xform_z3 = wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())
    xform_rot_x = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi / 2.0))

    # (name, xform, size, origin, direction, expected)
    cases = [
        ("hit_from_above", identity, infinite, wp.vec3(0.0, 0.0, 4.0), wp.vec3(3.0, 0.0, -4.0), 1.0),  # 3-4-5 triple
        ("parallel_miss", identity, infinite, wp.vec3(0.0, 0.0, 2.0), wp.vec3(1.0, 0.0, 0.0), -1.0),
        ("backward_miss", identity, infinite, wp.vec3(0.0, 0.0, 5.0), wp.vec3(0.0, 0.0, 1.0), -1.0),
        ("translated_plane", xform_z3, infinite, wp.vec3(0.0, 0.0, 7.0), wp.vec3(3.0, 0.0, -4.0), 1.0),
        # Finite planes: hit point (3, 0, 0) lies outside the half-extent.
        (
            "finite_miss_half_extent",
            identity,
            wp.vec3(4.0, 4.0, 0.0),
            wp.vec3(0.0, 0.0, 4.0),
            wp.vec3(3.0, 0.0, -4.0),
            -1.0,
        ),
        ("finite_miss_x", identity, wp.vec3(2.0, 2.0, 0.0), wp.vec3(0.0, 0.0, 4.0), wp.vec3(3.0, 0.0, -4.0), -1.0),
        # Hit at (0, 3, 0) lies outside half-extent 1 in y.
        ("finite_miss_y", identity, wp.vec3(10.0, 2.0, 0.0), wp.vec3(0.0, 0.0, 4.0), wp.vec3(0.0, 3.0, -4.0), -1.0),
        ("hit_from_below", identity, infinite, wp.vec3(0.0, 0.0, -4.0), wp.vec3(0.0, 3.0, 4.0), 1.0),
        ("rotated_plane", xform_rot_x, infinite, wp.vec3(0.0, -5.0, 0.0), wp.vec3(0.0, 1.0, 0.0), 5.0),
        ("axial_hit", identity, infinite, wp.vec3(0.0, 0.0, 5.0), wp.vec3(0.0, 0.0, -1.0), 5.0),
    ]

    for name, xform, size, origin, direction, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom, dim=1, inputs=[out_t, xform, size, GeoType.PLANE, origin, direction, 0], device=device
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_mesh(test: TestRaycast, device: str):
    """Test mesh raycasting using a simple quad made of two triangles."""
    out_t = wp.zeros(1, dtype=float, device=device)

    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32).flatten()
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    xform = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)  # no scaling

    # Angled ray: (-2, 0, 1) + t*(1, 0, -0.5) hits the quad at (0, 0, 0).
    angled_dir = wp.normalize(wp.vec3(1.0, 0.0, -0.5))
    angled_expected = 2.0 * wp.sqrt(1.0**2 + 0.5**2)  # pre-normalize length * t=2

    # (name, origin, direction, expected, delta)
    cases = [
        ("hit_from_above", wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, -1.0), 2.0, 1e-3),
        ("hit_from_below", wp.vec3(0.0, 0.0, -2.0), wp.vec3(0.0, 0.0, 1.0), 2.0, 1e-3),
        ("miss_outside_bounds", wp.vec3(2.0, 2.0, 2.0), wp.vec3(0.0, 0.0, -1.0), -1.0, 1e-5),
        ("hit_angled", wp.vec3(-2.0, 0.0, 1.0), angled_dir, angled_expected, 1e-3),
    ]

    for name, origin, direction, expected, delta in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_mesh,
                dim=1,
                inputs=[out_t, xform, origin, direction, size, mesh_id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=delta)


def test_mesh_ray_intersect(test: TestRaycast, device: str):
    """Test mesh raycasting through the ray_intersect_geom interface."""
    out_t = wp.zeros(1, dtype=float, device=device)

    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    xform = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)

    cases = [
        ("hit", wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, -1.0), 2.0),
    ]

    for name, origin, direction, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, xform, size, GeoType.MESH, origin, direction, mesh_id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-3)


def test_convex_hull_ray_intersect_via_geom(test: TestRaycast, device: str):
    """Test convex hull raycasting through the ray_intersect_geom interface (uses mesh path)."""
    out_t = wp.zeros(1, dtype=float, device=device)

    vertices = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.int32)
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
        mesh_id = mesh.finalize(device=device)

    xform = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)

    cases = [
        ("hit", wp.vec3(0.0, 0.0, 2.0), wp.vec3(0.0, 0.0, -1.0), 2.0),
    ]

    for name, origin, direction, expected in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, xform, size, GeoType.CONVEX_MESH, origin, direction, mesh_id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-3)


devices = get_test_devices()
add_function_test(TestRaycast, "test_ray_intersect_plane", test_ray_intersect_plane, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_sphere", test_ray_intersect_sphere, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_box", test_ray_intersect_box, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_capsule", test_ray_intersect_capsule, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_cylinder", test_ray_intersect_cylinder, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_cone", test_ray_intersect_cone, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_ellipsoid", test_ray_intersect_ellipsoid, devices=devices)
add_function_test(TestRaycast, "test_ray_intersect_mesh", test_ray_intersect_mesh, devices=devices)
add_function_test(TestRaycast, "test_mesh_ray_intersect", test_mesh_ray_intersect, devices=devices)
add_function_test(
    TestRaycast, "test_convex_hull_ray_intersect_via_geom", test_convex_hull_ray_intersect_via_geom, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
