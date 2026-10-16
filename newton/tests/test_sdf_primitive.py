# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

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
# cylinder with a distinct top radius: the tapered capped-cone branch
PRIMITIVE_TRUNCATED_CONE = wp.constant(7)


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
        return kernels.sdf_cylinder(point, p0, p1, up_axis, -1.0, p2)
    if primitive == PRIMITIVE_ELLIPSOID:
        return kernels.sdf_ellipsoid(point, wp.vec3(p0, p1, p2))
    if primitive == PRIMITIVE_CONE:
        return kernels.sdf_cone(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_TRUNCATED_CONE:
        return kernels.sdf_cylinder(point, p0, p1, up_axis, p2, 0.0)
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
        return kernels.sdf_cylinder_grad(point, p0, p1, up_axis, -1.0, p2)
    if primitive == PRIMITIVE_ELLIPSOID:
        return kernels.sdf_ellipsoid_grad(point, wp.vec3(p0, p1, p2))
    if primitive == PRIMITIVE_CONE:
        return kernels.sdf_cone_grad(point, p0, p1, up_axis)
    if primitive == PRIMITIVE_TRUNCATED_CONE:
        return kernels.sdf_cylinder_grad(point, p0, p1, up_axis, p2, 0.0)
    return kernels.sdf_plane_grad(point, p0, p1)


@wp.kernel
def evaluate_gradient_kernel(
    primitive: int,
    points: wp.array[wp.vec3],
    p0: float,
    p1: float,
    p2: float,
    up_axis: int,
    gradient: wp.array[wp.vec3],
):
    tid = wp.tid()
    gradient[tid] = _eval_grad(primitive, points[tid], p0, p1, p2, up_axis)


@wp.kernel
def evaluate_gradient_error_kernel(
    primitive: int,
    points: wp.array[wp.vec3],
    p0: float,
    p1: float,
    p2: float,
    up_axis: int,
    eps: float,
    dot_alignment: wp.array[float],
    analytic_norm: wp.array[float],
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
    eps: float = 1.0e-4,
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
            eps,
            dot_alignment_wp,
            analytic_norm_wp,
        ],
        device=device,
    )

    dot_alignment = dot_alignment_wp.numpy()
    analytic_norm = analytic_norm_wp.numpy()
    test.assertTrue(
        np.all(dot_alignment > dot_tol),
        msg=(
            f"Gradient alignment below tolerance {dot_tol}: min={dot_alignment.min():.6f} "
            f"at {points_np[int(np.argmin(dot_alignment))]} (params {p0}, {p1}, {p2}, eps {eps})"
        ),
    )
    test.assertTrue(
        np.allclose(analytic_norm, np.ones_like(analytic_norm), atol=1.0e-5, rtol=0.0),
        msg=f"Analytic gradient norm deviates from 1: min={analytic_norm.min():.6f}, max={analytic_norm.max():.6f}",
    )


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


def test_sdf_barrel_cylinder_grad_matches_finite_difference(test, device):
    """Verify barrel-cylinder SDF gradients in every axis orientation."""
    points = np.array(
        [
            [0.9, 0.2, 0.1],
            [-0.8, -0.6, 0.3],
            [0.2, 1.3, -0.1],
            [0.7, 0.8, 0.4],
        ],
        dtype=np.float32,
    )
    for axis in (Axis.X, Axis.Y, Axis.Z):
        _assert_gradient_matches_fd(test, device, PRIMITIVE_CYLINDER, points, 0.7, 1.0, 2.0, int(axis))


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


def _cone_rim_ring(radius, half_height):
    """Sample rings around a cone cap for gradient comparisons."""
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rings = []
    for radial_offset in (0.15, 0.4, 1.0):
        for axial_offset in (-0.4, -0.12, 0.12, 0.4):
            ring_radius = radius * (1.0 + radial_offset)
            rings.append(
                np.stack(
                    [
                        ring_radius * np.cos(angles),
                        ring_radius * np.sin(angles),
                        np.full_like(angles, -half_height * (1.0 + axial_offset)),
                    ],
                    axis=-1,
                )
            )
    return np.concatenate(rings).astype(np.float32)


def test_sdf_cone_grad_near_base_rim_across_scales(test, device):
    """Verify the cone gradient beside the base rim over several aspect ratios and shape scales."""
    for radius, half_height in ((1.0, 1.5), (1.0, 0.1), (0.1, 1.0), (1.0, 1.0)):
        for scale in (1.0e-3, 1.0, 1.0e3):
            r = radius * scale
            h = half_height * scale
            points = _cone_rim_ring(r, h)
            for axis, order in ((Axis.X, (2, 0, 1)), (Axis.Y, (0, 2, 1)), (Axis.Z, (0, 1, 2))):
                _assert_gradient_matches_fd(
                    test,
                    device,
                    PRIMITIVE_CONE,
                    points[:, order],
                    r,
                    h,
                    0.0,
                    int(axis),
                    dot_tol=0.999,
                    eps=1.0e-3 * max(r, h),
                )


def test_sdf_truncated_cone_grad_matches_finite_difference(test, device):
    """Verify the tapered cylinder gradient at both rims over several tapers and shape scales."""
    for bottom, top, half_height in ((1.0, 0.4, 1.0), (0.4, 1.0, 1.0), (1.0, 0.05, 0.2)):
        for scale in (1.0e-3, 1.0, 1.0e3):
            r0 = bottom * scale
            r1 = top * scale
            h = half_height * scale
            points = np.concatenate([_cone_rim_ring(r0, h), _cone_rim_ring(r1, -h)]).astype(np.float32)
            for axis, order in ((Axis.X, (2, 0, 1)), (Axis.Y, (0, 2, 1)), (Axis.Z, (0, 1, 2))):
                _assert_gradient_matches_fd(
                    test,
                    device,
                    PRIMITIVE_TRUNCATED_CONE,
                    points[:, order],
                    r0,
                    h,
                    r1,
                    int(axis),
                    dot_tol=0.999,
                    eps=1.0e-3 * max(r0, r1, h),
                )


def test_sdf_cone_grad_known_normals(test, device):
    """Verify the cone gradient against directions known without differencing."""
    radius = 1.0
    half_height = 1.5
    slant = np.array([2.0 * half_height, 0.0, radius])
    slant /= np.linalg.norm(slant)
    cases = (
        ("above apex", (0.0, 0.0, half_height + 0.7), (0.0, 0.0, 1.0)),
        ("below base", (0.0, 0.0, -half_height - 0.7), (0.0, 0.0, -1.0)),
        ("beside rim", (radius + 0.3, 0.0, -half_height - 0.4), (0.6, 0.0, -0.8)),
        ("beside lateral face", (radius + 0.25, 0.0, 0.0), tuple(slant)),
        ("inside near lateral face", (0.45, 0.0, 0.0), tuple(slant)),
        ("inside near base", (0.1, 0.0, -half_height + 0.1), (0.0, 0.0, -1.0)),
        ("on lateral face", (radius / 2.0, 0.0, 0.0), tuple(slant)),
        ("on base", (radius / 2.0, 0.0, -half_height), (0.0, 0.0, -1.0)),
    )
    points = np.array([c[1] for c in cases], dtype=np.float32)
    expected = np.array([c[2] for c in cases], dtype=np.float64)
    for axis, order in ((Axis.X, (2, 0, 1)), (Axis.Y, (0, 2, 1)), (Axis.Z, (0, 1, 2))):
        points_wp = wp.array(points[:, order], dtype=wp.vec3, device=device)
        gradient_wp = wp.zeros(len(cases), dtype=wp.vec3, device=device)
        wp.launch(
            evaluate_gradient_kernel,
            dim=len(cases),
            inputs=[PRIMITIVE_CONE, points_wp, radius, half_height, 0.0, int(axis), gradient_wp],
            device=device,
        )
        gradient = gradient_wp.numpy()
        for (name, _point, _want), got, want in zip(cases, gradient, expected[:, order], strict=True):
            test.assertAlmostEqual(float(np.linalg.norm(got)), 1.0, delta=1.0e-5, msg=f"{axis}: {name}")
            test.assertGreater(float(np.dot(got, want)), 0.9999, msg=f"{axis}: {name}: got {got}, expected {want}")


def test_sdf_truncated_cone_grad_zero_height(test, device):
    """Verify collapsed tapered cylinders use the disk's closest-point normals."""
    points = np.array([[1.3, 0.0, 0.0], [1.3, 0.0, 0.4], [0.0, 0.0, -0.4]], dtype=np.float32)
    expected = np.array([[1.0, 0.0, 0.0], [0.6, 0.0, 0.8], [0.0, 0.0, -1.0]])
    for bottom, top in ((1.0, 0.4), (0.4, 1.0)):
        for axis, order in ((Axis.X, (2, 0, 1)), (Axis.Y, (0, 2, 1)), (Axis.Z, (0, 1, 2))):
            points_wp = wp.array(points[:, order], dtype=wp.vec3, device=device)
            gradient_wp = wp.zeros(len(points), dtype=wp.vec3, device=device)
            wp.launch(
                evaluate_gradient_kernel,
                dim=len(points),
                inputs=[PRIMITIVE_TRUNCATED_CONE, points_wp, bottom, 0.0, top, int(axis), gradient_wp],
                device=device,
            )
            np.testing.assert_allclose(gradient_wp.numpy(), expected[:, order], atol=1.0e-3, rtol=0.0)


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


_devices = get_test_devices()
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
    "test_sdf_barrel_cylinder_grad_matches_finite_difference",
    test_sdf_barrel_cylinder_grad_matches_finite_difference,
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
    "test_sdf_cone_grad_near_base_rim_across_scales",
    test_sdf_cone_grad_near_base_rim_across_scales,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_truncated_cone_grad_matches_finite_difference",
    test_sdf_truncated_cone_grad_matches_finite_difference,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_cone_grad_known_normals",
    test_sdf_cone_grad_known_normals,
    devices=_devices,
)
add_function_test(
    TestSdfPrimitive,
    "test_sdf_truncated_cone_grad_zero_height",
    test_sdf_truncated_cone_grad_zero_height,
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
