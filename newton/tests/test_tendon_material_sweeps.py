# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material-sweep regressions independent of rigid-body convergence."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.tendon import TendonGuideType
from newton._src.solvers.tendon_kernels import _material_sweep_contact_error, solve_tendon_material
from newton.tests.unittest_utils import add_function_test, get_test_devices


def solve_material_case(
    device,
    tensions,
    ratios,
    rolling_left=None,
    rolling_right=None,
    sweeps=256,
    tol=1.0e-6,
    atol=1.0e-6,
    damping_tensions=None,
    segment_compliance=0.01,
    segment_lengths=None,
    capture=True,
):
    """Hold geometry fixed and exercise the production material-transfer kernel."""
    count = len(tensions)
    length = (
        np.ones(count, dtype=np.float32) if segment_lengths is None else np.asarray(segment_lengths, dtype=np.float32)
    )
    compliance = np.full(count, segment_compliance, dtype=np.float32)
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
    positions = np.concatenate(([0.0], np.cumsum(length)))
    points_l = wp.array([[x, 0, 0] for x in positions[:-1]], dtype=wp.vec3, device=device)
    points_r = wp.array([[x, 0, 0] for x in positions[1:]], dtype=wp.vec3, device=device)
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
        atol,
        wp.zeros(count + 1, device=device),
    ]
    wp.launch(solve_tendon_material, dim=1, inputs=inputs, device=device)
    if capture and wp.get_device(device).is_cuda:
        eager_stretch, eager_rest = stretch.numpy(), rest_gpu.numpy()
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(solve_tendon_material, dim=1, inputs=inputs, device=device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(stretch.numpy(), eager_stretch)
        np.testing.assert_array_equal(rest_gpu.numpy(), eager_rest)
    stretch_np = stretch.numpy()
    result = np.maximum(stretch_np / compliance + damping.numpy(), 0.0)
    return result, rest_gpu.numpy(), rest, int(sweep_count.numpy()[0])


def test_adaptive_sweeps_exhausted_span(test, device):
    """Accept a material-bound optimum, not an impossible capstan equality."""
    for reverse in (False, True):
        for capture in (False, True) if device.is_cuda else (False,):
            lengths, tensions = [1.0, 0.001], [5.0, 0.0]
            if reverse:
                lengths, tensions = lengths[::-1], tensions[::-1]
            with test.subTest(reverse=reverse, capture=capture):
                force, rest, initial, count = solve_material_case(
                    device, tensions, [2.0], segment_lengths=lengths, capture=capture
                )
                donor = 0 if reverse else 1
                receiver = 1 - donor
                expected_rest = initial.copy()
                expected_rest[receiver] += expected_rest[donor] - 1.0e-6
                expected_rest[donor] = 1.0e-6
                np.testing.assert_allclose(rest, expected_rest, atol=1.0e-7, rtol=0)
                test.assertGreater(force[receiver], 2.0 * force[donor])
                test.assertLessEqual(count, 2, "The donor bound already prevents any further transfer")
                full, full_rest, _, full_count = solve_material_case(
                    device, tensions, [2.0], segment_lengths=lengths, tol=0.0, atol=100.0
                )
                test.assertEqual(full_count, 256, "Zero relative tolerance disables early-out, including atol")
                np.testing.assert_array_equal(force, full)
                np.testing.assert_array_equal(rest, full_rest)


def test_adaptive_sweeps_replenished_donor(test, device):
    """Recheck a depleted middle span after its other neighbor replenishes it."""
    for reverse in (False, True):
        lengths, tensions, ratios = [1.0, 0.1, 1.0], [50.0, 0.0, -25.0], [1.2, 1.4]
        expected = 25.0 * np.array([1.68, 1.4, 1.0]) / 4.08
        if reverse:
            lengths, tensions, ratios, expected = lengths[::-1], tensions[::-1], ratios[::-1], expected[::-1]
        with test.subTest(reverse=reverse):
            force, rest, initial, count = solve_material_case(
                device, tensions, ratios, segment_lengths=lengths, tol=1.0e-6
            )
            test.assertGreater(count, 1)
            test.assertLess(count, 256)
            np.testing.assert_allclose(force, expected, atol=3.0e-5, rtol=0)
            np.testing.assert_allclose(rest.sum(), initial.sum(), atol=2.0e-7, rtol=0)
            test.assertGreater(rest[1], 1.0e-6)


def test_material_convergence_bound_directions(test, device):
    """An exhausted donor can stop transfer, but cannot prevent release from its bound."""
    rows = [
        [3.0, 1.0, 2.0, 0.1, 1.0, 0.0],  # upper bound, force drives into it
        [1.0, 3.0, 2.0, -0.1, 0.0, 1.0],  # lower bound, force drives into it
        [1.5, 1.0, 2.0, 0.1, 1.0, 0.0],  # upper bound must release
        [1.0, 1.5, 2.0, -0.1, 0.0, 1.0],  # lower bound must release
        [3.0, 1.0, 2.0, 0.0, 1.0, 0.0],  # no slip, upper bound
        [1.0, 3.0, 2.0, 0.0, 1.0, 0.0],  # no slip, can move away
        [3.0, 1.0, 2.0, 0.1, 1.0, 1.0],  # neighbor replenished donor
        [1.0, 3.0, 2.0, -0.1, 1.0, 1.0],
    ]
    errors = wp.empty(len(rows), dtype=float, device=device)
    wp.launch(
        _evaluate_contact_error,
        dim=len(rows),
        inputs=[wp.array(rows, dtype=float, device=device)],
        outputs=[errors],
        device=device,
    )
    np.testing.assert_allclose(errors.numpy(), [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0], atol=1e-7, rtol=0)


def test_adaptive_sweeps_check_final_force_scale(test, device):
    """Resolve small final tensions after a much larger transient material load."""
    for reverse in (False, True):
        initial = [-40.0, 40.0, 0.03]
        ratios = [1.2, 1.4]
        if reverse:
            initial = initial[::-1]
            ratios = ratios[::-1]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(device, initial, ratios, tol=1.0e-3)
            full, _, _, _ = solve_material_case(device, initial, ratios, tol=0.0)
            for left, right, rho in zip(tension[:-1], tension[1:], ratios, strict=True):
                error = max(left - rho * right, right - rho * left, 0.0)
                test.assertLessEqual(error, 1.0e-6 + 1.0e-3 * max(left, right))
            np.testing.assert_allclose(tension, full, atol=3.0e-6, rtol=5.0e-3)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)
            test.assertLess(count, 256)


def test_adaptive_sweeps_preserve_small_corrections(test, device):
    """Resolve small force corrections after transferring much more material."""
    for reverse in (False, True):
        initial = [-1.0e5, 1.0e5, 100.0]
        ratios = [1.2, 1.4]
        if reverse:
            initial = initial[::-1]
            ratios = ratios[::-1]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(
                device, initial, ratios, tol=1.0e-6, segment_compliance=1.0e-8
            )
            for left, right, rho in zip(tension[:-1], tension[1:], ratios, strict=True):
                error = max(left - rho * right, right - rho * left, 0.0)
                test.assertLessEqual(error, 1.0e-6 + 1.0e-6 * max(left, right))
            test.assertLess(count, 256)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)


def test_adaptive_sweeps_stop_at_satisfied_projection(test, device):
    """Stop once the final friction law is satisfied, even after a large update."""
    for reverse in (False, True):
        initial = [30.0, 0.0] if not reverse else [0.0, 30.0]
        expected = [20.0, 10.0] if not reverse else [10.0, 20.0]
        with test.subTest(reverse=reverse):
            tension, rest, initial_rest, count = solve_material_case(device, initial, [2.0], tol=1.0e-3)
            np.testing.assert_allclose(tension, expected, atol=1.0e-5, rtol=0)
            np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=3.0e-7, rtol=0)
            test.assertEqual(count, 1, "A satisfied two-span projection needs no further sweep")


def test_adaptive_sweeps_absolute_tolerance(test, device):
    """Stop an unloading cable without chasing a shrinking relative force scale."""
    tension, rest, initial, count = solve_material_case(
        device, [-40.0, 40.0, -0.01], [1.2, 1.4], tol=1.0e-3, atol=1.0e-6
    )
    test.assertLessEqual(float(tension.max()), 2.0e-6)
    test.assertLess(count, 128)
    np.testing.assert_allclose(rest.sum(), initial.sum(), atol=3.0e-7, rtol=0)


@wp.kernel
def _evaluate_contact_error(values: wp.array2d[float], result: wp.array[float]):
    """Evaluate force residuals for independently specified contact states."""
    i = wp.tid()
    result[i] = _material_sweep_contact_error(
        values[i, 0], values[i, 1], values[i, 2], values[i, 3], values[i, 4] != 0.0, values[i, 5] != 0.0
    )


def test_material_convergence_requires_slip_boundary(test, device):
    """Reject excess net slip even when its tension lies inside the friction band."""
    rows = [
        [1.5, 1.0, 2.0, 0.0],
        [1.5, 1.0, 2.0, 0.1],
        [1.0, 1.5, 2.0, -0.1],
        [2.0, 1.0, 2.0, 0.1],
        [1.0, 2.0, 2.0, -0.1],
        [3.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 0.1],
    ]
    rows = [[*row, 1.0, 1.0] for row in rows]
    errors = wp.empty(len(rows), dtype=float, device=device)
    wp.launch(
        _evaluate_contact_error,
        dim=len(rows),
        inputs=[wp.array(rows, dtype=float, device=device)],
        outputs=[errors],
        device=device,
    )
    np.testing.assert_allclose(errors.numpy(), [0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0], atol=1e-7, rtol=0)


def test_material_sweep_tolerances_are_validated(test, device):
    """Reject invalid sweep controls consistently in both solvers."""
    model = newton.ModelBuilder().finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        for name in ("tendon_settle_tol", "tendon_settle_atol"):
            for value in (-1.0, float("nan"), float("inf")):
                with test.subTest(solver=solver_type.__name__, parameter=name, value=value):
                    with test.assertRaisesRegex(ValueError, name):
                        solver_type(model, **{name: value})
        test.assertEqual(solver_type(model, tendon_max_sweeps=4096).tendon_max_sweeps, 4096)
        for value in (0, -1, 1.5, 2**31):
            with test.subTest(solver=solver_type.__name__, budget=value):
                with test.assertRaisesRegex(ValueError, "tendon_max_sweeps"):
                    solver_type(model, tendon_max_sweeps=value)


def test_solver_entry_points_honor_sweep_budget(test, device):
    """Execute the requested budget through both solvers, including CUDA graph replay."""
    builder = newton.ModelBuilder()
    builder.add_tendon(
        [
            newton.TendonGuide(body=-1, offset=(-1.0, 0.0, 0.0)),
            newton.TendonGuide(body=-1, guide_type=newton.TendonGuideType.PINHOLE, compliance=0.01),
            newton.TendonGuide(body=-1, offset=(0.0, 1.0, 0.0), compliance=0.01),
        ]
    )
    model = builder.finalize(device=device)
    for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
        with test.subTest(solver=solver_type.__name__):
            solver = solver_type(model, iterations=1, tendon_max_sweeps=512, tendon_settle_tol=0.0)
            state, output, control = model.state(), model.state(), model.control()
            solver.step(state, output, control, None, 0.01)
            np.testing.assert_array_equal(solver.tendon_cone_sweep_count.numpy(), [512])
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as captured:
                    solver.step(state, output, control, None, 0.01)
                wp.capture_launch(captured.graph)
                np.testing.assert_array_equal(solver.tendon_cone_sweep_count.numpy(), [512])


def test_adaptive_sweeps_honor_large_budget(test, device):
    """Resolve a finite-friction chain beyond the former 256-sweep cap."""
    span_count, rho = 32, 1.01
    tension, rest, initial_rest, count = solve_material_case(
        device,
        [10.0] + [0.0] * (span_count - 1),
        [rho] * (span_count - 1),
        sweeps=4096,
        tol=1.0e-5,
        atol=1.0e-7,
    )
    scale = np.power(rho, -np.arange(span_count))
    extension = np.sum(1.0 - initial_rest.astype(np.float64))
    expected = scale * extension / (0.01 * scale.sum())
    test.assertGreater(count, 256)
    test.assertLess(count, 4096)
    np.testing.assert_allclose(tension, expected, atol=2.0e-4, rtol=0.0)
    np.testing.assert_allclose(rest.sum(), initial_rest.sum(), atol=4.0e-6, rtol=0.0)
    _, _, _, count = solve_material_case(device, [1.0, 1.0], [rho], sweeps=512, tol=0.0)
    test.assertEqual(count, 512, "Disabling early-out must execute the requested budget")


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


devices = get_test_devices()
add_function_test(
    TestTendonMaterialSweeps,
    "test_solver_entry_points_honor_sweep_budget",
    test_solver_entry_points_honor_sweep_budget,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_exhausted_span",
    test_adaptive_sweeps_exhausted_span,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_replenished_donor",
    test_adaptive_sweeps_replenished_donor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_convergence_bound_directions",
    test_material_convergence_bound_directions,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_absolute_tolerance",
    test_adaptive_sweeps_absolute_tolerance,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_convergence_requires_slip_boundary",
    test_material_convergence_requires_slip_boundary,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_material_sweep_tolerances_are_validated",
    test_material_sweep_tolerances_are_validated,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_check_final_force_scale",
    test_adaptive_sweeps_check_final_force_scale,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_preserve_small_corrections",
    test_adaptive_sweeps_preserve_small_corrections,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_stop_at_satisfied_projection",
    test_adaptive_sweeps_stop_at_satisfied_projection,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_honor_large_budget",
    test_adaptive_sweeps_honor_large_budget,
    devices=devices,
)
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


if __name__ == "__main__":
    unittest.main()
