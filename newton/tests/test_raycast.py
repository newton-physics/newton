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
    ray_intersect_geom,
    ray_intersect_mesh,
    ray_intersect_sphere,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

wp.config.quiet = True


class TestRaycast(unittest.TestCase):
    pass


# Kernels to test ray intersection functions
@wp.kernel
def kernel_test_sphere(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_sphere(geom_to_world, ray_origin, ray_direction, r)


@wp.kernel
def kernel_test_box(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_box(geom_to_world, ray_origin, ray_direction, size)


@wp.kernel
def kernel_test_capsule(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_capsule(geom_to_world, ray_origin, ray_direction, r, h)


@wp.kernel
def kernel_test_cylinder(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_cylinder(geom_to_world, ray_origin, ray_direction, r, h)


@wp.kernel
def kernel_test_cone(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    r: float,
    h: float,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_cone(geom_to_world, ray_origin, ray_direction, r, h)


@wp.kernel
def kernel_test_geom(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    mesh_id: wp.uint64,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_geom(geom_to_world, size, geomtype, ray_origin, ray_direction, mesh_id)


@wp.kernel
def kernel_test_mesh(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
    size: wp.vec3,
    mesh_id: wp.uint64,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_mesh(geom_to_world, ray_origin, ray_direction, size, mesh_id)


# Test functions
def test_ray_intersect_sphere(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 1.0

    # Case 1: Ray hits sphere
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Case 2: Ray misses sphere
    ray_origin = wp.vec3(-2.0, 2.0, 0.0)
    wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    # Case 3: Ray starts inside
    ray_origin = wp.vec3(0.0, 0.0, 0.0)
    wp.launch(kernel_test_sphere, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)


def test_ray_intersect_box(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    size = wp.vec3(1.0, 1.0, 1.0)

    # Case 1: Ray hits box
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Case 2: Ray misses box
    ray_origin = wp.vec3(-2.0, 2.0, 0.0)
    wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    # Case 3: Ray starts inside
    ray_origin = wp.vec3(0.0, 0.0, 0.0)
    wp.launch(kernel_test_box, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, size], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Case 4: Rotated box
    # Rotate 45 degrees around Z axis
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

    # Case 1: Hit cylinder part
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    # Case 2: Hit top cap
    ray_origin = wp.vec3(0.0, 0.0, -2.0)
    ray_direction = wp.vec3(0.0, 0.0, 1.0)
    wp.launch(kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 2.0 - 1.0 - 0.5, delta=1e-5)

    # Case 3: Miss
    ray_origin = wp.vec3(-2.0, 2.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_capsule, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)


def test_ray_intersect_cylinder(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    r = 0.5
    h = 1.0

    # Case 1: Hit cylinder body
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(
        kernel_test_cylinder, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    # Case 2: Hit top cap
    ray_origin = wp.vec3(0.0, 0.0, -2.0)
    ray_direction = wp.vec3(0.0, 0.0, 1.0)
    wp.launch(
        kernel_test_cylinder, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Case 3: Miss
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

    # Case 1: Hit cone body from the side
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-3)

    # Case 2: Hit cone base from below
    ray_origin = wp.vec3(0.0, 0.0, -2.0)
    ray_direction = wp.vec3(0.0, 0.0, 1.0)
    wp.launch(kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-3)  # hits base at z=-1

    # Case 3: Miss cone completely
    ray_origin = wp.vec3(-2.0, 2.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    wp.launch(kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], -1.0, delta=1e-5)

    # Case 4: Ray from above hitting the tip area
    ray_origin = wp.vec3(0.0, 0.0, 2.0)
    ray_direction = wp.vec3(0.0, 0.0, -1.0)
    wp.launch(kernel_test_cone, dim=1, inputs=[out_t, geom_to_world, ray_origin, ray_direction, r, h], device=device)
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-3)  # hits tip at z=1


def test_geom_ray_intersect(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)
    mesh_id = wp.uint64(0)  # No mesh for primitive shapes

    # Sphere
    size = wp.vec3(1.0, 0.0, 0.0)  # r
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.SPHERE, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Box
    size = wp.vec3(1.0, 1.0, 1.0)  # half-extents
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.BOX, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Capsule
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.CAPSULE, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    # Cylinder
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.CYLINDER, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    # Cone
    size = wp.vec3(1.0, 1.0, 0.0)  # r, h
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GeoType.CONE, ray_origin, ray_direction, mesh_id],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-3)


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
for device in devices:
    add_function_test(TestRaycast, f"test_ray_intersect_sphere_{device}", test_ray_intersect_sphere, devices=[device])
    add_function_test(TestRaycast, f"test_ray_intersect_box_{device}", test_ray_intersect_box, devices=[device])
    add_function_test(TestRaycast, f"test_ray_intersect_capsule_{device}", test_ray_intersect_capsule, devices=[device])
    add_function_test(
        TestRaycast, f"test_ray_intersect_cylinder_{device}", test_ray_intersect_cylinder, devices=[device]
    )
    add_function_test(TestRaycast, f"test_ray_intersect_cone_{device}", test_ray_intersect_cone, devices=[device])
    add_function_test(TestRaycast, f"test_geom_ray_intersect_{device}", test_geom_ray_intersect, devices=[device])
    add_function_test(TestRaycast, f"test_ray_intersect_mesh_{device}", test_ray_intersect_mesh, devices=[device])
    add_function_test(
        TestRaycast, f"test_mesh_ray_intersect_via_geom_{device}", test_mesh_ray_intersect_via_geom, devices=[device]
    )
    add_function_test(
        TestRaycast,
        f"test_convex_hull_ray_intersect_via_geom_{device}",
        test_convex_hull_ray_intersect_via_geom,
        devices=[device],
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
