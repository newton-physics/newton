# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

from newton._src.core.types import Axis
from newton._src.geometry import kernels
from newton.tests.unittest_utils import add_function_test, get_test_devices

PRIMITIVE_SPHERE = wp.constant(0)
PRIMITIVE_BOX = wp.constant(1)
PRIMITIVE_CAPSULE = wp.constant(2)
PRIMITIVE_CYLINDER = wp.constant(3)
PRIMITIVE_ELLIPSOID = wp.constant(4)
PRIMITIVE_CONE = wp.constant(5)
PRIMITIVE_PLANE = wp.constant(6)


@wp.func
def _safe_normalize(v: wp.vec3, fallback: wp.vec3):
    eps = 1.0e-8
    v_len = wp.length(v)
    if v_len > eps:
        return v / v_len
    return fallback


@wp.func
def _eval_sdf(primitive: int, point: wp.vec3, p0: float, p1: float, p2: float, up_axis: int):
    if primitive == PRIMITIVE_SPHERE:
        return kernels.sdf_sphere(point, p0)
    if primitive == PRIMITIVE_BOX:
        return kernels.sdf_box(point, p0, p1, p2)
    if primitive == PRIMITIVE_CAPSULE:
        return kernels.sdf_capsule(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_CYLINDER:
        return kernels.sdf_cylinder(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_ELLIPSOID:
        return kernels.sdf_ellipsoid(point, wp.vec3(p0, p1, p2))
    if primitive == PRIMITIVE_CONE:
        return kernels.sdf_cone(point, p0, p1, up_axis)
    return kernels.sdf_plane(point, p0, p1)


@wp.func
def _eval_grad(primitive: int, point: wp.vec3, p0: float, p1: float, p2: float, up_axis: int):
    if primitive == PRIMITIVE_SPHERE:
        return kernels.sdf_sphere_grad(point, p0)
    if primitive == PRIMITIVE_BOX:
        return kernels.sdf_box_grad(point, p0, p1, p2)
    if primitive == PRIMITIVE_CAPSULE:
        return kernels.sdf_capsule_grad(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_CYLINDER:
        return kernels.sdf_cylinder_grad(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_ELLIPSOID:
        return kernels.sdf_ellipsoid_grad(point, wp.vec3(p0, p1, p2))
    if primitive == PRIMITIVE_CONE:
        return kernels.sdf_cone_grad(point, p0, p1, up_axis)
    return kernels.sdf_plane_grad(point, p0, p1)


@wp.kernel
def evaluate_gradient_error_kernel(
    primitive: int,
    points: wp.array(dtype=wp.vec3),
    p0: float,
    p1: float,
    p2: float,
    up_axis: int,
    eps: float,
    dot_alignment: wp.array(dtype=float),
    analytic_norm: wp.array(dtype=float),
):
    tid = wp.tid()
    point = points[tid]

    grad_analytic = _eval_grad(primitive, point, p0, p1, p2, up_axis)
    analytic_norm[tid] = wp.length(grad_analytic)
    grad_analytic = _safe_normalize(grad_analytic, wp.vec3(0.0, 0.0, 1.0))

    dx = wp.vec3(eps, 0.0, 0.0)
    dy = wp.vec3(0.0, eps, 0.0)
    dz = wp.vec3(0.0, 0.0, eps)

    fxp = _eval_sdf(primitive, point + dx, p0, p1, p2, up_axis)
    fxm = _eval_sdf(primitive, point - dx, p0, p1, p2, up_axis)
    fyp = _eval_sdf(primitive, point + dy, p0, p1, p2, up_axis)
    fym = _eval_sdf(primitive, point - dy, p0, p1, p2, up_axis)
    fzp = _eval_sdf(primitive, point + dz, p0, p1, p2, up_axis)
    fzm = _eval_sdf(primitive, point - dz, p0, p1, p2, up_axis)

    grad_fd = wp.vec3(
        (fxp - fxm) / (2.0 * eps),
        (fyp - fym) / (2.0 * eps),
        (fzp - fzm) / (2.0 * eps),
    )
    grad_fd = _safe_normalize(grad_fd, wp.vec3(0.0, 0.0, 1.0))

    dot_alignment[tid] = wp.dot(grad_analytic, grad_fd)


def _assert_gradient_matches_fd(
    test: unittest.TestCase,
    device,
    primitive: int,
    points_np: np.ndarray,
    p0: float,
    p1: float,
    p2: float,
    up_axis: int,
    dot_tol: float = 0.995,
):
    points_wp = wp.array(points_np.astype(np.float32), dtype=wp.vec3, device=device)
    dot_alignment_wp = wp.zeros(points_np.shape[0], dtype=float, device=device)
    analytic_norm_wp = wp.zeros(points_np.shape[0], dtype=float, device=device)

    wp.launch(
        evaluate_gradient_error_kernel,
        dim=points_np.shape[0],
        inputs=[
            primitive,
            points_wp,
            p0,
            p1,
            p2,
            up_axis,
            1.0e-4,
            dot_alignment_wp,
            analytic_norm_wp,
        ],
        device=device,
    )

    dot_alignment = dot_alignment_wp.numpy()
    analytic_norm = analytic_norm_wp.numpy()
    np.testing.assert_array_less(np.full_like(dot_alignment, dot_tol), dot_alignment)
    np.testing.assert_allclose(analytic_norm, np.ones_like(analytic_norm), atol=1.0e-5, rtol=0.0)


def test_sdf_sphere_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [0.8, -0.2, 0.5],
            [-1.7, 0.4, 0.2],
            [0.3, 0.9, -1.2],
        ],
        dtype=np.float32,
    )
    _assert_gradient_matches_fd(test, device, PRIMITIVE_SPHERE, points, 1.3, 0.0, 0.0, int(Axis.Y))


def test_sdf_box_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [1.4, 0.2, 0.1],
            [0.1, -1.8, 0.0],
            [0.2, 0.1, 1.3],
            [0.85, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    _assert_gradient_matches_fd(test, device, PRIMITIVE_BOX, points, 0.9, 1.2, 0.7, int(Axis.Y))


def test_sdf_capsule_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [0.6, 0.2, 0.3],
            [-0.5, -0.6, 0.4],
            [0.2, 1.6, -0.1],
        ],
        dtype=np.float32,
    )
    for axis in (Axis.X, Axis.Y, Axis.Z):
        _assert_gradient_matches_fd(test, device, PRIMITIVE_CAPSULE, points, 0.4, 1.1, 0.0, int(axis))


def test_sdf_cylinder_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [1.3, 0.1, 0.0],
            [0.2, -1.6, 0.1],
            [0.1, 1.5, -0.2],
            [1.3, 1.3, 0.0],
        ],
        dtype=np.float32,
    )
    _assert_gradient_matches_fd(test, device, PRIMITIVE_CYLINDER, points, 0.7, 1.0, 0.0, int(Axis.Y))


def test_sdf_ellipsoid_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [1.5, -0.2, 0.3],
            [-0.4, 0.7, 1.1],
            [0.2, -0.9, -0.4],
        ],
        dtype=np.float32,
    )
    _assert_gradient_matches_fd(test, device, PRIMITIVE_ELLIPSOID, points, 1.2, 0.8, 0.6, int(Axis.Y), dot_tol=0.95)


def test_sdf_cone_grad_matches_finite_difference(test, device):
    points = np.array(
        [
            [0.7, -0.3, 0.2],
            [1.1, -1.0, 0.1],
            [0.2, 1.2, 0.1],
        ],
        dtype=np.float32,
    )
    for axis in (Axis.X, Axis.Y, Axis.Z):
        _assert_gradient_matches_fd(test, device, PRIMITIVE_CONE, points, 1.0, 1.5, 0.0, int(axis), dot_tol=0.99)


def test_sdf_plane_grad_matches_finite_difference_for_infinite_plane(test, device):
    points = np.array(
        [
            [0.3, -0.2, 1.1],
            [-2.0, 0.4, -0.6],
            [1.7, 3.0, 0.2],
        ],
        dtype=np.float32,
    )
    _assert_gradient_matches_fd(test, device, PRIMITIVE_PLANE, points, 0.0, 0.0, 0.0, int(Axis.Y))


class TestSdfPrimitive(unittest.TestCase):
    pass


_devices = get_test_devices(mode="basic")
add_function_test(
    TestSdfPrimitive,
    "test_sdf_sphere_grad_matches_finite_difference",
    test_sdf_sphere_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_box_grad_matches_finite_difference",
    test_sdf_box_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_capsule_grad_matches_finite_difference",
    test_sdf_capsule_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_cylinder_grad_matches_finite_difference",
    test_sdf_cylinder_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_ellipsoid_grad_matches_finite_difference",
    test_sdf_ellipsoid_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_cone_grad_matches_finite_difference",
    test_sdf_cone_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_plane_grad_matches_finite_difference_for_infinite_plane",
    test_sdf_plane_grad_matches_finite_difference_for_infinite_plane,
    devices=_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
