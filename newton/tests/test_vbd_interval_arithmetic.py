# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Containment tests for the isolated VBD interval-arithmetic utilities."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.vbd.interval_arithmetic import (
    interval,
    interval_add,
    interval_mul,
    next_float_down,
    next_float_up,
    rigid_point_plane_signed_distance_derivative_interval,
    rigid_point_plane_signed_distance_interval,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestVBDIntervalArithmetic(unittest.TestCase):
    pass


@wp.kernel(enable_backward=False)
def _next_float_kernel(values: wp.array[float], down: wp.array[float], up: wp.array[float]):
    tid = wp.tid()
    down[tid] = next_float_down(values[tid])
    up[tid] = next_float_up(values[tid])


@wp.kernel(enable_backward=False)
def _basic_interval_kernel(
    a_lower: wp.array[float],
    a_upper: wp.array[float],
    b_lower: wp.array[float],
    b_upper: wp.array[float],
    add_lower: wp.array[float],
    add_upper: wp.array[float],
    mul_lower: wp.array[float],
    mul_upper: wp.array[float],
):
    tid = wp.tid()
    a = interval(a_lower[tid], a_upper[tid])
    b = interval(b_lower[tid], b_upper[tid])
    add_result = interval_add(a, b)
    mul_result = interval_mul(a, b)
    add_lower[tid] = add_result.lower
    add_upper[tid] = add_result.upper
    mul_lower[tid] = mul_result.lower
    mul_upper[tid] = mul_result.upper


@wp.kernel(enable_backward=False)
def _rigid_signed_distance_interval_kernel(
    t_lower: wp.array[float],
    t_upper: wp.array[float],
    n: wp.array[wp.vec3],
    d: wp.array[wp.vec3],
    c0: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    axis: wp.array[wp.vec3],
    angle: wp.array[float],
    offset0: wp.array[wp.vec3],
    result_lower: wp.array[float],
    result_upper: wp.array[float],
):
    tid = wp.tid()
    result = rigid_point_plane_signed_distance_interval(
        t_lower[tid],
        t_upper[tid],
        n[tid],
        d[tid],
        c0[tid],
        dx[tid],
        axis[tid],
        angle[tid],
        offset0[tid],
    )
    result_lower[tid] = result.lower
    result_upper[tid] = result.upper


@wp.kernel(enable_backward=False)
def _rigid_signed_distance_derivative_interval_kernel(
    t_lower: wp.array[float],
    t_upper: wp.array[float],
    n: wp.array[wp.vec3],
    dx: wp.array[wp.vec3],
    axis: wp.array[wp.vec3],
    angle: wp.array[float],
    offset0: wp.array[wp.vec3],
    result_lower: wp.array[float],
    result_upper: wp.array[float],
):
    tid = wp.tid()
    result = rigid_point_plane_signed_distance_derivative_interval(
        t_lower[tid], t_upper[tid], n[tid], dx[tid], axis[tid], angle[tid], offset0[tid]
    )
    result_lower[tid] = result.lower
    result_upper[tid] = result.upper


def _empty_outputs(count, device, number):
    return [wp.empty(count, dtype=float, device=device) for _ in range(number)]


def test_next_float_matches_numpy(test, device):
    values = np.array(
        [
            -np.inf,
            -np.finfo(np.float32).max,
            -1.0,
            -np.finfo(np.float32).tiny,
            -0.0,
            0.0,
            np.finfo(np.float32).tiny,
            1.0,
            np.finfo(np.float32).max,
            np.inf,
            np.nan,
        ],
        dtype=np.float32,
    )
    down, up = _empty_outputs(len(values), device, 2)
    wp.launch(
        _next_float_kernel,
        dim=len(values),
        inputs=[wp.array(values, dtype=float, device=device)],
        outputs=[down, up],
        device=device,
    )

    with np.errstate(over="ignore"):
        expected_down = np.nextafter(values, np.float32(-np.inf), dtype=np.float32)
        expected_up = np.nextafter(values, np.float32(np.inf), dtype=np.float32)
    actual_down = down.numpy()
    actual_up = up.numpy()
    finite_or_infinite = ~np.isnan(values)
    test.assertTrue(
        np.array_equal(
            actual_down[finite_or_infinite].view(np.uint32),
            expected_down[finite_or_infinite].view(np.uint32),
        )
    )
    test.assertTrue(
        np.array_equal(actual_up[finite_or_infinite].view(np.uint32), expected_up[finite_or_infinite].view(np.uint32))
    )
    test.assertTrue(np.isnan(actual_down[-1]))
    test.assertTrue(np.isnan(actual_up[-1]))


def test_basic_interval_operations_contain_exact_results(test, device):
    rng = np.random.default_rng(104729)
    endpoints = rng.uniform(-20.0, 20.0, size=(256, 4)).astype(np.float32)
    a_lower = np.minimum(endpoints[:, 0], endpoints[:, 1])
    a_upper = np.maximum(endpoints[:, 0], endpoints[:, 1])
    b_lower = np.minimum(endpoints[:, 2], endpoints[:, 3])
    b_upper = np.maximum(endpoints[:, 2], endpoints[:, 3])
    outputs = _empty_outputs(len(endpoints), device, 4)
    wp.launch(
        _basic_interval_kernel,
        dim=len(endpoints),
        inputs=[
            wp.array(a_lower, dtype=float, device=device),
            wp.array(a_upper, dtype=float, device=device),
            wp.array(b_lower, dtype=float, device=device),
            wp.array(b_upper, dtype=float, device=device),
        ],
        outputs=outputs,
        device=device,
    )
    add_lower, add_upper, mul_lower, mul_upper = [output.numpy() for output in outputs]

    a_lower64, a_upper64 = a_lower.astype(np.float64), a_upper.astype(np.float64)
    b_lower64, b_upper64 = b_lower.astype(np.float64), b_upper.astype(np.float64)
    exact_add_lower = a_lower64 + b_lower64
    exact_add_upper = a_upper64 + b_upper64
    products = np.stack(
        [
            a_lower64 * b_lower64,
            a_lower64 * b_upper64,
            a_upper64 * b_lower64,
            a_upper64 * b_upper64,
        ]
    )
    exact_mul_lower = products.min(axis=0)
    exact_mul_upper = products.max(axis=0)

    test.assertTrue(np.all(add_lower.astype(np.float64) <= exact_add_lower))
    test.assertTrue(np.all(add_upper.astype(np.float64) >= exact_add_upper))
    test.assertTrue(np.all(mul_lower.astype(np.float64) <= exact_mul_lower))
    test.assertTrue(np.all(mul_upper.astype(np.float64) >= exact_mul_upper))


def _sample_rigid_signed_distance(t, n, d, c0, dx, axis, angle, offset0):
    axis_dot_offset = axis @ offset0
    parallel = axis * axis_dot_offset
    perpendicular = offset0 - parallel
    cross = np.cross(axis, offset0)
    u = angle * t[:, None]
    point = c0 + t[:, None] * dx + parallel + np.cos(u) * perpendicular + np.sin(u) * cross
    return (point - d) @ n


def _sample_rigid_signed_distance_derivative(t, n, dx, axis, angle, offset0):
    axis_dot_offset = axis @ offset0
    perpendicular = offset0 - axis * axis_dot_offset
    cross = np.cross(axis, offset0)
    u = angle * t[:, None]
    velocity = dx + angle * (-np.sin(u) * perpendicular + np.cos(u) * cross)
    return velocity @ n


def test_rigid_signed_distance_intervals_contain_samples(test, device):
    rng = np.random.default_rng(130363)
    count = 64
    t_ranges = np.sort(rng.uniform(0.0, 1.0, size=(count, 2)).astype(np.float32), axis=1)
    n = rng.normal(size=(count, 3)).astype(np.float32)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    d = rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32)
    c0 = rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32)
    dx = rng.uniform(-1.0, 1.0, size=(count, 3)).astype(np.float32)
    axis = rng.normal(size=(count, 3)).astype(np.float32)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = rng.uniform(0.0, np.pi, size=count).astype(np.float32)
    offset0 = rng.uniform(-2.0, 2.0, size=(count, 3)).astype(np.float32)

    # Include a cross-and-return arc whose endpoints are both behind x=0.99.
    t_ranges[0] = (0.0, 1.0)
    n[0] = (1.0, 0.0, 0.0)
    d[0] = (0.99, 0.0, 0.0)
    c0[0] = (0.0, 0.0, 0.0)
    dx[0] = (0.0, 0.0, 0.0)
    axis[0] = (0.0, 0.0, 1.0)
    angle[0] = np.float32(np.pi)
    offset0[0] = (0.0, -1.0, 0.0)

    result_lower, result_upper = _empty_outputs(count, device, 2)
    wp.launch(
        _rigid_signed_distance_interval_kernel,
        dim=count,
        inputs=[
            wp.array(t_ranges[:, 0], dtype=float, device=device),
            wp.array(t_ranges[:, 1], dtype=float, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(d, dtype=wp.vec3, device=device),
            wp.array(c0, dtype=wp.vec3, device=device),
            wp.array(dx, dtype=wp.vec3, device=device),
            wp.array(axis, dtype=wp.vec3, device=device),
            wp.array(angle, dtype=float, device=device),
            wp.array(offset0, dtype=wp.vec3, device=device),
        ],
        outputs=[result_lower, result_upper],
        device=device,
    )
    result_lower = result_lower.numpy().astype(np.float64)
    result_upper = result_upper.numpy().astype(np.float64)

    derivative_lower, derivative_upper = _empty_outputs(count, device, 2)
    wp.launch(
        _rigid_signed_distance_derivative_interval_kernel,
        dim=count,
        inputs=[
            wp.array(t_ranges[:, 0], dtype=float, device=device),
            wp.array(t_ranges[:, 1], dtype=float, device=device),
            wp.array(n, dtype=wp.vec3, device=device),
            wp.array(dx, dtype=wp.vec3, device=device),
            wp.array(axis, dtype=wp.vec3, device=device),
            wp.array(angle, dtype=float, device=device),
            wp.array(offset0, dtype=wp.vec3, device=device),
        ],
        outputs=[derivative_lower, derivative_upper],
        device=device,
    )
    derivative_lower = derivative_lower.numpy().astype(np.float64)
    derivative_upper = derivative_upper.numpy().astype(np.float64)

    for i in range(count):
        samples = np.linspace(float(t_ranges[i, 0]), float(t_ranges[i, 1]), 4097, dtype=np.float64)
        values = _sample_rigid_signed_distance(
            samples,
            n[i].astype(np.float64),
            d[i].astype(np.float64),
            c0[i].astype(np.float64),
            dx[i].astype(np.float64),
            axis[i].astype(np.float64),
            float(angle[i]),
            offset0[i].astype(np.float64),
        )
        test.assertLessEqual(result_lower[i], values.min(), f"lower containment failed for case {i}")
        test.assertGreaterEqual(result_upper[i], values.max(), f"upper containment failed for case {i}")
        derivative_values = _sample_rigid_signed_distance_derivative(
            samples,
            n[i].astype(np.float64),
            dx[i].astype(np.float64),
            axis[i].astype(np.float64),
            float(angle[i]),
            offset0[i].astype(np.float64),
        )
        test.assertLessEqual(
            derivative_lower[i], derivative_values.min(), f"derivative lower containment failed for case {i}"
        )
        test.assertGreaterEqual(
            derivative_upper[i], derivative_values.max(), f"derivative upper containment failed for case {i}"
        )

    test.assertGreater(result_upper[0], 0.0, "the interval must expose the hidden positive arc")


devices = get_test_devices()

add_function_test(
    TestVBDIntervalArithmetic,
    "test_next_float_matches_numpy",
    test_next_float_matches_numpy,
    devices=devices,
)
add_function_test(
    TestVBDIntervalArithmetic,
    "test_basic_interval_operations_contain_exact_results",
    test_basic_interval_operations_contain_exact_results,
    devices=devices,
)
add_function_test(
    TestVBDIntervalArithmetic,
    "test_rigid_signed_distance_intervals_contain_samples",
    test_rigid_signed_distance_intervals_contain_samples,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
