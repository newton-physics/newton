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

import warp as wp

from newton.geometry import GEO_BOX, GEO_CAPSULE, GEO_CYLINDER, GEO_SPHERE
from newton.geometry.raycast import (
    ray_intersect_box,
    ray_intersect_capsule,
    ray_intersect_cylinder,
    ray_intersect_geom,
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
def kernel_test_geom(
    out_t: wp.array(dtype=float),
    geom_to_world: wp.transform,
    size: wp.vec3,
    geomtype: int,
    ray_origin: wp.vec3,
    ray_direction: wp.vec3,
):
    tid = wp.tid()
    out_t[tid] = ray_intersect_geom(geom_to_world, size, geomtype, ray_origin, ray_direction)


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


def test_geom_ray_intersect(test: TestRaycast, device: str):
    out_t = wp.zeros(1, dtype=float, device=device)
    geom_to_world = wp.transform_identity()
    ray_origin = wp.vec3(-2.0, 0.0, 0.0)
    ray_direction = wp.vec3(1.0, 0.0, 0.0)

    # Sphere
    size = wp.vec3(1.0, 0.0, 0.0)  # r
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GEO_SPHERE, ray_origin, ray_direction],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Box
    size = wp.vec3(1.0, 1.0, 1.0)  # half-extents
    wp.launch(
        kernel_test_geom, dim=1, inputs=[out_t, geom_to_world, size, GEO_BOX, ray_origin, ray_direction], device=device
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.0, delta=1e-5)

    # Capsule
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GEO_CAPSULE, ray_origin, ray_direction],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)

    # Cylinder
    size = wp.vec3(0.5, 1.0, 0.0)  # r, h
    wp.launch(
        kernel_test_geom,
        dim=1,
        inputs=[out_t, geom_to_world, size, GEO_CYLINDER, ray_origin, ray_direction],
        device=device,
    )
    test.assertAlmostEqual(out_t.numpy()[0], 1.5, delta=1e-5)


devices = get_test_devices()
for device in devices:
    add_function_test(TestRaycast, f"test_ray_intersect_sphere_{device}", test_ray_intersect_sphere, devices=[device])
    add_function_test(TestRaycast, f"test_ray_intersect_box_{device}", test_ray_intersect_box, devices=[device])
    add_function_test(TestRaycast, f"test_ray_intersect_capsule_{device}", test_ray_intersect_capsule, devices=[device])
    add_function_test(
        TestRaycast, f"test_ray_intersect_cylinder_{device}", test_ray_intersect_cylinder, devices=[device]
    )
    add_function_test(TestRaycast, f"test_geom_ray_intersect_{device}", test_geom_ray_intersect, devices=[device])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
