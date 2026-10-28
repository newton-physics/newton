# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material-sweep regressions independent of rigid-body convergence."""

import unittest

import numpy as np
import warp as wp

from newton._src.sim.tendon import TendonGuideType
from newton._src.solvers.tendon_kernels import solve_tendon_material
from newton._src.solvers.vbd.tendon_kernels import _rolling_spin_axis_component
from newton._src.solvers.xpbd.tendon_kernels import solve_tendon_slip
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_xpbd_spin_reaction_bounds_accumulated_impulse(test, device):
    """Bound the total friction impulse and retract it when the cable unloads."""
    with wp.ScopedDevice(device):
        for mu in (0.0, 0.1, 10.0):
            with test.subTest(mu=mu):
                lambdas = wp.zeros(2)
                tension = wp.full(2, 100.0)
                reaction = wp.zeros(3, dtype=wp.vec3)
                deltas = wp.zeros(1, dtype=wp.spatial_vector)
                inputs = [
                    wp.array([wp.transform_identity()], dtype=wp.transform),
                    wp.array([-1, 0, -1], dtype=int),
                    wp.array([-1, 1, -1], dtype=int),
                    wp.zeros(3, dtype=int),
                    wp.array(
                        [int(TendonGuideType.ANCHOR), int(TendonGuideType.ROLLER), int(TendonGuideType.ANCHOR)],
                        dtype=int,
                    ),
                    wp.array([0.0, 1.0, 0.0], dtype=float),
                    wp.array([0.0, mu, 0.0], dtype=float),
                    wp.ones(3, dtype=bool),
                    wp.zeros(3, dtype=wp.vec3),
                    wp.array([[0.0, 1.0, 0.0]] * 3, dtype=wp.vec3),
                    wp.array([[-1.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=wp.vec3),
                    wp.array([[-1.0, 0.0, 0.0], [1.0, 0.0, -1.0]], dtype=wp.vec3),
                    wp.full(2, 0.01),
                    tension,
                    wp.zeros(2),
                    lambdas,
                    0.01,
                    reaction,
                ]
                accumulated = 0.0
                for pair in ([-1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0], [-0.5, -0.5], [0.0, -1.0]):
                    lambdas.assign(np.asarray(pair, dtype=np.float32))
                    deltas.zero_()
                    wp.launch(solve_tendon_slip, dim=3, inputs=inputs, outputs=[deltas])
                    accumulated += float(deltas.numpy()[0, 4])
                    rho = np.exp(min(mu * np.pi, 20.0))
                    bound = 2.0 * (rho - 1.0) / (rho + 1.0)
                    expected = np.clip(pair[0] - pair[1], -bound, bound)
                    test.assertAlmostEqual(accumulated, float(expected), delta=1.0e-6)
                tension.zero_()
                deltas.zero_()
                wp.launch(solve_tendon_slip, dim=3, inputs=inputs, outputs=[deltas])
                accumulated += float(deltas.numpy()[0, 4])
                test.assertAlmostEqual(accumulated, 0.0, delta=1.0e-6)
                # A world roller must never write its reaction to the last body.
                inputs[3].fill_(-1)
                tension.fill_(100.0)
                deltas.zero_()
                wp.launch(solve_tendon_slip, dim=3, inputs=inputs, outputs=[deltas])
                np.testing.assert_array_equal(deltas.numpy(), 0.0)


def solve_material_case(
    device,
    tensions,
    ratios,
    rolling_left=None,
    rolling_right=None,
    sweeps=256,
    tol=1.0e-6,
    damping_tensions=None,
):
    """Hold geometry fixed and exercise the production material-transfer kernel."""
    count = len(tensions)
    length = np.ones(count, dtype=np.float32)
    compliance = np.full(count, 0.01, dtype=np.float32)
    rest = length - compliance * np.asarray(tensions, dtype=np.float32)
    damping_values = (
        np.zeros(count, dtype=np.float32)
        if damping_tensions is None
        else np.asarray(damping_tensions, dtype=np.float32)
    )
    rest += compliance * damping_values
    active = wp.ones(count, dtype=int, device=device)
    length_gpu = wp.array(length, device=device)
    rest_gpu = wp.array(rest, device=device)
    stretch = wp.zeros(count, device=device)
    damping = wp.zeros(count, device=device)
    points_l = wp.array([[i, 0, 0] for i in range(count)], dtype=wp.vec3, device=device)
    points_r = wp.array([[i + 1, 0, 0] for i in range(count)], dtype=wp.vec3, device=device)
    poses = wp.array([wp.transform_identity()] * (count + 1), dtype=wp.transform, device=device)
    guide_active = wp.ones(count + 1, dtype=bool, device=device)
    sweep_count = wp.zeros(1, dtype=int, device=device)
    # Identity poses and unit +x span directions make each length rate the
    # difference of adjacent x velocities. Use damping coefficient one.
    velocities = np.zeros((count + 1, 6), dtype=np.float32)
    velocities[1:, 0] = np.cumsum(damping_values)
    inputs = [
        poses,
        wp.array(velocities, dtype=wp.spatial_vector, device=device),
        poses,
        wp.zeros(count + 1, dtype=wp.vec3, device=device),
        wp.array([0, count + 1], dtype=int, device=device),
        wp.array(np.arange(count + 1), dtype=int, device=device),
        wp.array(
            [int(TendonGuideType.ANCHOR)] + [int(TendonGuideType.ROLLER)] * (count - 1) + [int(TendonGuideType.ANCHOR)],
            dtype=int,
            device=device,
        ),
        wp.ones(count + 1, device=device),
        wp.zeros(count + 1, dtype=wp.vec3, device=device),
        wp.array([[0, 0, 1]] * (count + 1), dtype=wp.vec3, device=device),
        rest_gpu,
        wp.array(rest, device=device),
        wp.array(rest, device=device),
        stretch,
        damping,
        active,
        wp.array(np.arange(count), dtype=int, device=device),
        wp.array(np.arange(1, count + 1), dtype=int, device=device),
        wp.array(compliance, device=device),
        wp.ones(count, device=device),
        guide_active,
        guide_active,
        wp.zeros(count + 1, device=device),
        points_l,
        points_r,
        length_gpu,
        points_l,
        points_r,
        wp.array(np.zeros(count) if rolling_left is None else rolling_left, dtype=float, device=device),
        wp.array(np.zeros(count) if rolling_right is None else rolling_right, dtype=float, device=device),
        wp.array([-1, *range(count - 1), -1], dtype=int, device=device),
        wp.array([-1, *range(1, count), -1], dtype=int, device=device),
        wp.array([1.0, *ratios, 1.0], dtype=float, device=device),
        sweep_count,
        0,
        1.0 / 60.0,
        1,
        1,
        1,
        sweeps,
        tol,
        wp.zeros(count + 1, device=device),
    ]
    wp.launch(solve_tendon_material, dim=1, inputs=inputs, device=device)
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(solve_tendon_material, dim=1, inputs=inputs, device=device)
        wp.capture_launch(capture.graph)
    stretch_np = stretch.numpy()
    result = np.maximum(stretch_np / compliance + damping.numpy(), 0.0)
    return result, rest_gpu.numpy(), rest, int(sweep_count.numpy()[0])


def test_net_slip_reaches_capstan_boundary(test, device):
    """Neighbor updates must retract excess slip instead of stopping inside the cone."""
    rho = np.array([1.4, 1.0004])
    for reverse in (False, True):
        ratios = rho[::-1] if reverse else rho
        for tol in (0.0, 1.0e-6):
            with test.subTest(reverse=reverse, tol=tol):
                tension, rest, initial, _ = solve_material_case(device, [9, 1, 9], ratios, tol=tol)
                initial_tension = (1.0 - initial.astype(float)) / float(np.float32(0.01))
                # Both end spans feed the middle. Thus both contacts must finish
                # on their slipping boundaries, not merely somewhere in the cone.
                middle = initial_tension.sum() / (1 + ratios[0] + ratios[1])
                expected = np.array([ratios[0] * middle, middle, ratios[1] * middle])
                np.testing.assert_allclose(tension, expected, atol=3.0e-5, rtol=1.0e-5)
                np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)


def test_rolling_trial_precedes_friction_projection(test, device):
    """A sticking roller must preserve each endpoint's full material displacement."""
    for reverse in (False, True):
        left = [0.0, 0.002] if not reverse else [0.0, 0.0]
        right = [0.0, 0.0] if not reverse else [0.002, 0.0]
        for sweeps in (1, 4, 256):
            with test.subTest(reverse=reverse, sweeps=sweeps):
                tension, rest, initial, _ = solve_material_case(
                    device, [4, 4], [2], left, right, sweeps=sweeps, tol=0.0
                )
                expected_rest = initial + np.array([right[0], left[1]])
                expected_tension = (1.0 - expected_rest) / float(np.float32(0.01))
                np.testing.assert_allclose(rest, expected_rest, atol=1.0e-7, rtol=0)
                np.testing.assert_allclose(tension, expected_tension, atol=1.0e-5, rtol=0)


def test_damped_net_slip(test, device):
    """Project net slip using total tension, including the damping term."""
    damping = np.array([0.5, -0.2, 0.3])
    rho = np.array([1.4, 1.0004])
    tension, rest, initial, _ = solve_material_case(device, [9, 1, 9], rho, damping_tensions=damping)
    initial_tension = (1.0 - initial.astype(float)) / float(np.float32(0.01)) + damping
    middle = initial_tension.sum() / (1 + rho[0] + rho[1])
    np.testing.assert_allclose(tension, [rho[0] * middle, middle, rho[1] * middle], atol=3.0e-5, rtol=1.0e-5)
    np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)


def test_slack_span_and_sticking_contact(test, device):
    """Preserve slack material and sticking contacts while taking up slack before tension."""
    for initial, rho, expected in (
        ([-4, 0, 9], [1, 1], [5 / 3] * 3),
        ([-4, -2, -1], [1.2, 1.5], [0, 0, 0]),
        ([3, 4, 3], [1.5, 1.5], [3, 4, 3]),
    ):
        tension, rest, old_rest, _ = solve_material_case(device, initial, rho)
        np.testing.assert_allclose(tension, expected, atol=2.0e-5, rtol=0)
        np.testing.assert_allclose(rest.sum(), old_rest.sum(), atol=3.0e-7, rtol=0)
        if min(initial) >= 0 or max(initial) <= 0:
            np.testing.assert_allclose(rest, old_rest, atol=1.0e-7, rtol=0)


def test_rolling_grip_requires_tension(test, device):
    """Grip a loaded cable, but cancel rolling transport without tension or friction."""
    for initial, rho, grip in (([4, 4], 2.0, True), ([0, 0], 2.0, False), ([4, 4], 1.0, False)):
        with test.subTest(initial=initial, rho=rho):
            tension, rest, before, _ = solve_material_case(
                device, initial, [rho], rolling_left=[0, 0.002], rolling_right=[-0.002, 0]
            )
            transport = np.array([-0.002, 0.002]) if grip else np.zeros(2)
            np.testing.assert_allclose(rest, before + transport, atol=1.0e-7, rtol=0)
            np.testing.assert_allclose(tension, (1.0 - before - transport) / 0.01, atol=2.0e-5, rtol=0)


def test_rolling_trial_respects_rest_floor(test, device):
    """Project an exhausted no-slip trial as a pair without creating material."""
    tension, rest, before, _ = solve_material_case(
        device, [99.9998, 3], [1.0e8], rolling_left=[0, 0.01], rolling_right=[-0.01, 0]
    )
    test.assertTrue(np.isfinite(tension).all())
    np.testing.assert_allclose(rest[0], 1.0e-6, atol=3.0e-8, rtol=0)
    np.testing.assert_allclose(rest.astype(float).sum(), before.astype(float).sum(), atol=1.0e-7, rtol=0)


class TestTendonMaterialSweeps(unittest.TestCase):
    pass


@wp.kernel
def _evaluate_roller_spin(
    poses: wp.array[wp.transform],
    guide_body: wp.array[int],
    guide_type: wp.array[int],
    offsets: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    cone_l: wp.array[int],
    cone_r: wp.array[int],
    cap_ratio: wp.array[float],
    rest: wp.array[float],
    point_l: wp.array[wp.vec3],
    point_r: wp.array[wp.vec3],
    compliance: wp.array[float],
    damping: wp.array[float],
    active: wp.array[int],
    guide_l: wp.array[int],
    guide_r: wp.array[int],
    result: wp.array[wp.vec3],
):
    seg = wp.tid()
    tension = wp.max((1.0 - rest[seg]) / compliance[seg], 0.0)
    result[seg] = _rolling_spin_axis_component(
        1.0 / 60.0,
        poses,
        poses,
        offsets,
        guide_body,
        guide_type,
        offsets,
        axes,
        cone_l,
        cone_r,
        cap_ratio,
        rest,
        point_l,
        point_r,
        compliance,
        damping,
        active,
        guide_l,
        guide_r,
        1,
        seg,
        tension,
        wp.vec3(0.5, 1.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
    )


def test_vbd_spin_reaction_matches_capstan_projection(test, device):
    """Transmit the full sticking moment and limit only an out-of-cone trial."""
    for rho, tensions in ((1.0, [4, 3]), (2.0, [4, 3]), (2.0, [9, 1]), (2.0, [1, 9]), (2.0, [9, 0])):
        with test.subTest(rho=rho, tensions=tensions):
            rest = 1.0 - 0.01 * np.array(tensions, dtype=np.float32)
            output = wp.zeros(2, dtype=wp.vec3, device=device)
            wp.launch(
                _evaluate_roller_spin,
                dim=2,
                inputs=[
                    wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device),
                    wp.array([-1, 1, -1], dtype=int, device=device),
                    wp.array(
                        [int(TendonGuideType.ANCHOR), int(TendonGuideType.ROLLER), int(TendonGuideType.ANCHOR)],
                        dtype=int,
                        device=device,
                    ),
                    wp.array([[0, 0, 0], [0.5, 0, 0], [0, 0, 0]], dtype=wp.vec3, device=device),
                    wp.array([[0, 0, 1]] * 3, dtype=wp.vec3, device=device),
                    wp.array([-1, 0, -1], dtype=int, device=device),
                    wp.array([-1, 1, -1], dtype=int, device=device),
                    wp.array([1, rho, 1], dtype=float, device=device),
                    wp.array(rest, device=device),
                    wp.array([[-0.5, 1, 0], [1.5, 0, 0]], dtype=wp.vec3, device=device),
                    wp.array([[0.5, 1, 0], [1.5, -1, 0]], dtype=wp.vec3, device=device),
                    wp.full(2, 0.01, device=device),
                    wp.zeros(2, device=device),
                    wp.ones(2, dtype=int, device=device),
                    wp.array([0, 1], dtype=int, device=device),
                    wp.array([1, 2], dtype=int, device=device),
                    output,
                ],
                device=device,
            )
            scale = min(1.0, (rho - 1.0) / (rho + 1.0) * sum(tensions) / abs(tensions[0] - tensions[1]))
            np.testing.assert_allclose(output.numpy(), [[0, 0, -(1 - scale)]] * 2, atol=2.0e-6, rtol=0)


devices = get_test_devices()
add_function_test(
    TestTendonMaterialSweeps, "test_rolling_grip_requires_tension", test_rolling_grip_requires_tension, devices=devices
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_rolling_trial_respects_rest_floor",
    test_rolling_trial_respects_rest_floor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_net_slip_reaches_capstan_boundary",
    test_net_slip_reaches_capstan_boundary,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_rolling_trial_precedes_friction_projection",
    test_rolling_trial_precedes_friction_projection,
    devices=devices,
)
add_function_test(TestTendonMaterialSweeps, "test_damped_net_slip", test_damped_net_slip, devices=devices)
add_function_test(
    TestTendonMaterialSweeps,
    "test_slack_span_and_sticking_contact",
    test_slack_span_and_sticking_contact,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_vbd_spin_reaction_matches_capstan_projection",
    test_vbd_spin_reaction_matches_capstan_projection,
    devices=devices,
)


add_function_test(
    TestTendonMaterialSweeps,
    "test_xpbd_spin_reaction_bounds_accumulated_impulse",
    test_xpbd_spin_reaction_bounds_accumulated_impulse,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main()
