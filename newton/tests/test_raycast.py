# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, Heightfield
from newton._src.geometry.raycast import (
    ray_intersect_geom,
    ray_intersect_mesh,
    raycast_kernel,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestRaycast(unittest.TestCase):
    pass


@wp.kernel
def kernel_test_geom(
    out_t: wp.array[float],
    out_n: wp.array[wp.vec3],
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
):
    """Invoke :func:`ray_intersect_geom` and write hit distance + normal."""
    tid = wp.tid()
    t, n = ray_intersect_geom(
        geom_to_world,
        size,
        geomtype,
        ray_origin,
        ray_direction,
        mesh_id,
    )
    out_t[tid] = t
    out_n[tid] = n


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
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                inputs=[out_t, out_n, geom_to_world, size, GeoType.SPHERE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_box(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                kernel_test_geom,
                dim=1,
                inputs=[out_t, out_n, xform, size, GeoType.BOX, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_capsule(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                inputs=[out_t, out_n, geom_to_world, size, GeoType.CAPSULE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_cylinder(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                inputs=[out_t, out_n, geom_to_world, size, GeoType.CYLINDER, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_cone(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                inputs=[out_t, out_n, geom_to_world, size, GeoType.CONE, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=delta)


def test_ray_intersect_ellipsoid(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
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
                inputs=[out_t, out_n, geom_to_world, size, GeoType.ELLIPSOID, origin, direction, 0],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-5)


def test_ray_intersect_plane(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)

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
                kernel_test_geom,
                dim=1,
                inputs=[out_t, out_n, xform, size, GeoType.PLANE, origin, direction, 0],
                device=device,
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
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)

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
                inputs=[out_t, out_n, xform, size, GeoType.MESH, origin, direction, mesh_id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-3)


def test_convex_hull_ray_intersect_via_geom(test: TestRaycast, device: str):
    """Test convex hull raycasting through the ray_intersect_geom interface (uses mesh path)."""
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)

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
                inputs=[out_t, out_n, xform, size, GeoType.CONVEX_MESH, origin, direction, mesh_id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=1e-3)


def _hfield_mesh(device: str, data: np.ndarray, hx: float, hy: float, min_z: float, max_z: float) -> wp.Mesh:
    """Build a ``wp.Mesh`` from raw heightfield data the same way the builder does.

    Uses the same triangulation (two CCW triangles per cell) so test results are
    directly comparable to collision-kernel behaviour.
    """
    nrow, ncol = data.shape
    d_min, d_max = float(data.min()), float(data.max())
    normalized = (data - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(data)

    z_range = max_z - min_z
    dx = 2.0 * hx / (ncol - 1)
    dy = 2.0 * hy / (nrow - 1)

    verts = np.empty((nrow * ncol, 3), dtype=np.float32)
    for r in range(nrow):
        for c in range(ncol):
            idx = r * ncol + c
            verts[idx] = [-hx + c * dx, -hy + r * dy, min_z + float(normalized[r, c]) * z_range]

    indices = np.empty((nrow - 1) * (ncol - 1) * 6, dtype=np.int32)
    i = 0
    for r in range(nrow - 1):
        for c in range(ncol - 1):
            v00, v10 = r * ncol + c, r * ncol + (c + 1)
            v01, v11 = (r + 1) * ncol + c, (r + 1) * ncol + (c + 1)
            indices[i : i + 6] = [v00, v10, v11, v00, v11, v01]
            i += 6

    with wp.ScopedDevice(device):
        points = wp.array(verts, dtype=wp.vec3)
        idx_arr = wp.array(indices, dtype=wp.int32)
        return wp.Mesh(points=points, velocities=wp.zeros_like(points), indices=idx_arr)


def test_ray_intersect_heightfield(test: TestRaycast, device: str):
    """Heightfield raycasts via wp.Mesh BVH query. Regression for issue #2412."""
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
    identity = wp.transform_identity()

    # 1) Flat heightfield at z=1 on a 3x3 grid over [-2, 2]^2.
    flat = np.full((3, 3), 1.0, dtype=np.float32)
    mesh_flat = _hfield_mesh(device, flat, hx=2.0, hy=2.0, min_z=1.0, max_z=1.0)

    # 2) Tilted 2x2 cell: corner (1,1) raised to z=1, the rest at z=0 over [-1, 1]^2.
    tilt = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh_tilt = _hfield_mesh(device, tilt, hx=1.0, hy=1.0, min_z=0.0, max_z=1.0)

    # Translated flat heightfield (shifted +2 in z) to exercise geom_to_world.
    xform_shift_z = wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity())

    size = wp.vec3(1.0, 1.0, 1.0)
    # (name, xform, mesh, origin, direction, expected_t, delta)
    cases = [
        (
            "flat_hit_from_above",
            identity,
            mesh_flat,
            wp.vec3(0.0, 0.0, 5.0),
            wp.vec3(0.0, 0.0, -1.0),
            4.0,
            1e-4,
        ),
        (
            "tilt_hit_sloped_face",
            identity,
            mesh_tilt,
            wp.vec3(0.5, -0.5, 2.0),
            wp.vec3(0.0, 0.0, -1.0),
            # analytic: triangle (p00=(-1,-1,0), p10=(1,-1,0), p11=(1,1,1)) contains
            # XY=(0.5,-0.5); plane z at that XY is 0.25, so t = 2.0 - 0.25.
            1.75,
            1e-4,
        ),
        (
            "miss_outside_extent",
            identity,
            mesh_flat,
            wp.vec3(5.0, 5.0, 5.0),
            wp.vec3(0.0, 0.0, -1.0),
            -1.0,
            1e-5,
        ),
        (
            "miss_parallel_above",
            identity,
            mesh_flat,
            wp.vec3(0.0, 0.0, 5.0),
            wp.vec3(1.0, 0.0, 0.0),
            -1.0,
            1e-5,
        ),
        (
            "translated_flat_hit",
            xform_shift_z,
            mesh_flat,
            wp.vec3(0.0, 0.0, 6.0),
            wp.vec3(0.0, 0.0, -1.0),
            3.0,  # surface now at z=3
            1e-4,
        ),
    ]

    for name, xform, mesh, origin, direction, expected, delta in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, out_n, xform, size, GeoType.HFIELD, origin, direction, mesh.id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected, delta=delta)


def test_ray_intersect_heightfield_normals(test: TestRaycast, device: str):
    """Validate surface normals returned for HFIELD hits via wp.Mesh query.

    For a flat heightfield the normal is exactly world +Z. For the tilted
    (p00, p10, p11) triangle with p11 raised to z=1 over a [-1, 1]^2 cell, the
    plane normal is proportional to ``(0, -1, 2)`` -- we check the unit-length
    version.
    """
    out_t = wp.zeros(1, dtype=float, device=device)
    out_n = wp.zeros(1, dtype=wp.vec3, device=device)
    identity = wp.transform_identity()

    flat = np.full((3, 3), 1.0, dtype=np.float32)
    mesh_flat = _hfield_mesh(device, flat, hx=2.0, hy=2.0, min_z=1.0, max_z=1.0)
    tilt = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mesh_tilt = _hfield_mesh(device, tilt, hx=1.0, hy=1.0, min_z=0.0, max_z=1.0)

    sqrt5 = float(np.sqrt(5.0))
    size = wp.vec3(1.0, 1.0, 1.0)
    # (name, mesh, origin, direction, expected_t, expected_normal)
    cases = [
        (
            "flat_normal_z",
            mesh_flat,
            wp.vec3(0.0, 0.0, 5.0),
            wp.vec3(0.0, 0.0, -1.0),
            4.0,
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ),
        (
            "sloped_normal",
            mesh_tilt,
            wp.vec3(0.5, -0.5, 2.0),
            wp.vec3(0.0, 0.0, -1.0),
            1.75,
            np.array([0.0, -1.0 / sqrt5, 2.0 / sqrt5], dtype=np.float32),
        ),
    ]

    for name, mesh, origin, direction, expected_t, expected_n in cases:
        with test.subTest(name):
            wp.launch(
                kernel_test_geom,
                dim=1,
                inputs=[out_t, out_n, identity, size, GeoType.HFIELD, origin, direction, mesh.id],
                device=device,
            )
            test.assertAlmostEqual(out_t.numpy()[0], expected_t, delta=1e-4)
            got_n = out_n.numpy()[0]
            # Normal must be unit length (ray_intersect_geom normalises it).
            test.assertAlmostEqual(float(np.linalg.norm(got_n)), 1.0, delta=1e-4)
            # Match the analytic normal component-wise.
            for axis, expected_val in enumerate(expected_n):
                test.assertAlmostEqual(float(got_n[axis]), float(expected_val), delta=1e-4)


def test_ray_intersect_heightfield_scaled(test: TestRaycast, device: str):
    """Per-instance ``scale`` on ``add_shape_heightfield`` is honored by the raycast.

    Exercises the full ``ModelBuilder -> finalize -> raycast_kernel`` pipeline.
    A ray at ``x=1.5`` (outside the unit-scale extent ``[-1, 1]`` but inside the
    scaled extent ``[-2, 2]``) must hit on the scaled instance and miss on the
    unscaled one. The scaled instance must also hit at the correctly scaled z.
    """
    # ``min_z = max_z = 1`` pins the flat surface at local z=1 regardless of
    # the Heightfield constructor's elevation normalization.
    flat = np.ones((3, 3), dtype=np.float32)
    hf = Heightfield(data=flat, nrow=3, ncol=3, hx=1.0, hy=1.0, min_z=1.0, max_z=1.0)

    builder_scaled = newton.ModelBuilder()
    builder_scaled.add_shape_heightfield(heightfield=hf, scale=(2.0, 2.0, 2.0))
    model_scaled = builder_scaled.finalize(device=device)
    state_scaled = model_scaled.state()

    builder_unscaled = newton.ModelBuilder()
    builder_unscaled.add_shape_heightfield(heightfield=hf)
    model_unscaled = builder_unscaled.finalize(device=device)
    state_unscaled = model_unscaled.state()

    def cast(model, state, origin, direction):
        min_dist = wp.array([1.0e10], dtype=float, device=device)
        min_index = wp.array([-1], dtype=int, device=device)
        min_body_index = wp.array([-1], dtype=int, device=device)
        lock = wp.array([0], dtype=wp.int32, device=device)
        empty_world = wp.array([], dtype=int, device=device)
        empty_offsets = wp.array([], dtype=wp.vec3, device=device)
        empty_mask = wp.array([], dtype=int, device=device)
        wp.launch(
            raycast_kernel,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_body,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.shape_source_ptr,
                origin,
                direction,
                lock,
            ],
            outputs=[min_dist, min_index, min_body_index, empty_world, empty_offsets, empty_mask],
            device=device,
        )
        dist = float(min_dist.numpy()[0])
        return dist if dist < 1.0e10 else -1.0

    direction = wp.vec3(0.0, 0.0, -1.0)

    # Inside scaled extent [-2, 2] at x=1.5 but outside unit extent [-1, 1].
    with test.subTest("scaled_hit_outside_unit_extent"):
        t_scaled = cast(model_scaled, state_scaled, wp.vec3(1.5, 0.0, 5.0), direction)
        test.assertAlmostEqual(t_scaled, 3.0, delta=1e-4)  # mesh z=1 * scale_z=2 → surface at 2; 5-2=3
    with test.subTest("unscaled_miss_outside_unit_extent"):
        t_unscaled = cast(model_unscaled, state_unscaled, wp.vec3(1.5, 0.0, 5.0), direction)
        test.assertAlmostEqual(t_unscaled, -1.0, delta=1e-5)

    # Center ray confirms the scaled z-range: hit at z=2 vs z=1 unscaled.
    with test.subTest("scaled_center_z"):
        t_scaled_center = cast(model_scaled, state_scaled, wp.vec3(0.0, 0.0, 5.0), direction)
        test.assertAlmostEqual(t_scaled_center, 3.0, delta=1e-4)
    with test.subTest("unscaled_center_z"):
        t_unscaled_center = cast(model_unscaled, state_unscaled, wp.vec3(0.0, 0.0, 5.0), direction)
        test.assertAlmostEqual(t_unscaled_center, 4.0, delta=1e-4)


def _make_intersection_model(device: str):
    builder = newton.ModelBuilder()
    builder.begin_world()
    sphere_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_sphere(body=sphere_body, radius=0.5)

    box_body = builder.add_body(xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))
    builder.add_shape_box(body=box_body, hx=0.5, hy=0.5, hz=0.5)
    builder.end_world()

    return builder.finalize(device=device)


def test_intersect_ray(test: TestRaycast, device: str):
    model = _make_intersection_model(device)

    origins = wp.array(
        np.array(
            [
                [-2.0, 0.0, 0.0],
                [3.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    directions = wp.array(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    worlds = wp.array(np.array([0, 0, 0], dtype=np.int32), dtype=wp.int32, device=device)

    out_dist = wp.empty(shape=3, dtype=float, device=device)
    out_shape_id = wp.empty(shape=3, dtype=wp.int32, device=device)
    out_normal = wp.empty(shape=3, dtype=wp.vec3, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=out_dist,
        out_shape_id=out_shape_id,
        out_normal=out_normal,
    )

    np.testing.assert_allclose(out_dist.numpy(), np.array([1.5, 1.5, -1.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_array_equal(out_shape_id.numpy(), np.array([0, 1, -1], dtype=np.int32))
    np.testing.assert_allclose(
        out_normal.numpy(),
        np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )


def _make_global_world_model(device: str):
    """Two worlds each with a sphere, plus a global-world box shared by both."""
    builder = newton.ModelBuilder()

    # Global world (-1): box at the origin, accessible from every world.
    global_body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    global_box = builder.add_shape_box(body=global_body, hx=0.5, hy=0.5, hz=0.5)

    builder.begin_world()
    body0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 5.0, 0.0), wp.quat_identity()))
    sphere0 = builder.add_shape_sphere(body=body0, radius=0.5)
    builder.end_world()

    builder.begin_world()
    body1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 10.0, 0.0), wp.quat_identity()))
    sphere1 = builder.add_shape_sphere(body=body1, radius=0.5)
    builder.end_world()

    return builder.finalize(device=device), global_box, sphere0, sphere1


def test_intersect_ray_global_world(test: TestRaycast, device: str):
    model, global_box, sphere0, sphere1 = _make_global_world_model(device)

    # Rays fired along -y toward each world's sphere and the global box.
    origins = wp.array(
        np.array(
            [
                [0.0, 5.0, 5.0],  # world 0 sphere
                [0.0, 10.0, 5.0],  # world 1 sphere
                [0.0, 0.0, 5.0],  # global box, queried from world 0
                [0.0, 0.0, 5.0],  # global box, queried from world -1
                [0.0, 10.0, 5.0],  # world 1 sphere is invisible from world 0
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device=device,
    )
    directions = wp.array(
        np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
        dtype=wp.vec3,
        device=device,
    )
    worlds = wp.array(np.array([0, 1, 0, -1, 0], dtype=np.int32), dtype=wp.int32, device=device)

    out_dist = wp.empty(shape=5, dtype=float, device=device)
    out_shape_id = wp.empty(shape=5, dtype=wp.int32, device=device)
    out_normal = wp.empty(shape=5, dtype=wp.vec3, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        enable_global_world=True,
        out_dist=out_dist,
        out_shape_id=out_shape_id,
        out_normal=out_normal,
    )

    np.testing.assert_allclose(out_dist.numpy(), np.array([4.5, 4.5, 4.5, 4.5, -1.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_array_equal(
        out_shape_id.numpy(), np.array([sphere0, sphere1, global_box, global_box, -1], dtype=np.int32)
    )


def test_intersect_ray_heightfield_uses_finalize_bvh(test: TestRaycast, device: str):
    builder = newton.ModelBuilder()
    heightfield = Heightfield(data=np.zeros((2, 2), dtype=np.float32), nrow=2, ncol=2, hx=1.0, hy=1.0)
    shape_id = builder.add_shape_heightfield(heightfield=heightfield)
    model = builder.finalize(device=device)

    test.assertEqual(model.bvh_shape_count_enabled, 1)

    origins = wp.array(np.array([[0.0, 0.0, 2.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    directions = wp.array(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), dtype=wp.vec3, device=device)
    worlds = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
    out_dist = wp.empty(shape=1, dtype=float, device=device)
    out_shape_id = wp.empty(shape=1, dtype=wp.int32, device=device)

    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=out_dist,
        out_shape_id=out_shape_id,
    )

    np.testing.assert_allclose(out_dist.numpy(), np.array([2.0], dtype=np.float32), atol=1e-5)
    np.testing.assert_array_equal(out_shape_id.numpy(), np.array([shape_id], dtype=np.int32))


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
add_function_test(
    TestRaycast,
    "test_ray_intersect_heightfield",
    test_ray_intersect_heightfield,
    devices=devices,
)
add_function_test(
    TestRaycast,
    "test_ray_intersect_heightfield_normals",
    test_ray_intersect_heightfield_normals,
    devices=devices,
)
add_function_test(
    TestRaycast,
    "test_ray_intersect_heightfield_scaled",
    test_ray_intersect_heightfield_scaled,
    devices=devices,
)
add_function_test(TestRaycast, "test_intersect_ray", test_intersect_ray, devices=devices)
add_function_test(TestRaycast, "test_intersect_ray_global_world", test_intersect_ray_global_world, devices=devices)
add_function_test(
    TestRaycast,
    "test_intersect_ray_heightfield_uses_finalize_bvh",
    test_intersect_ray_heightfield_uses_finalize_bvh,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
