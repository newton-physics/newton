# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material-sweep regressions independent of rigid-body convergence."""

import math
import sys
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.tendon import TendonGuideType
from newton._src.solvers.tendon_kernels import _material_sweep_contact_error, solve_tendon_material
from newton.tests.test_tendon_capstan import build_explicit_inactive_dynamic_route, build_kinematic_rolling_transport
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices


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
    """Check material projection and stored-force precision while holding geometry fixed."""
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
        1,
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
    stored_rest = rest_gpu.numpy()
    stored_force = np.maximum((length - stored_rest) / compliance + damping.numpy(), 0.0)
    # Cover writeback and the production len-rest force evaluation, not just
    # scratch stretch. Allow length-scale roundoff, including a clamped rest floor.
    force_resolution = (np.abs(np.spacing(length)) + np.abs(np.spacing(stored_rest))) / compliance
    force_resolution += 2.0 * np.abs(np.spacing(result))
    np.testing.assert_array_less(np.abs(stored_force - result), force_resolution)
    return result, stored_rest, rest, int(sweep_count.numpy()[0])


def test_rolling_material_exhaustion_diagnostic(test, device):
    """Report an exhausted wrap budget once, at the accepted pose, including graph replay."""
    for exhausted in (False, True):
        with test.subTest(exhausted=exhausted):
            capture = StdOutCapture()
            capture.begin()
            try:
                _run_material_exhaustion_cases(test, device, exhausted)
            finally:
                diagnostic = capture.end()
            # Windows cannot reliably capture native CPU/CUDA printf. Numerical
            # conservation and exhaustion assertions still run on every device.
            if sys.platform != "win32":
                expected = (8 if device.is_cuda else 4) if exhausted else 0
                test.assertEqual(diagnostic.count("ERROR: Tendon 0 material rest-length floor"), expected, diagnostic)
                test.assertNotIn("supported wrap range", diagnostic)


def _run_material_exhaustion_cases(test, device, exhausted):
    """Exercise changing wraps and dynamic activation with valid and exhausted budgets."""
    for dynamic in (False, True):
        with wp.ScopedDevice(device):
            if dynamic:
                model, candidate = build_explicit_inactive_dynamic_route(device)
                moved_body = int(model.tendon_guide_body.numpy()[candidate])
            else:
                model, _ = build_kinematic_rolling_transport(mu=0.2)
                moved_body = int(model.tendon_guide_body.numpy()[2])
        model.body_color_groups = [wp.array([i], dtype=int, device=device) for i in range(model.body_count)]
        model.tendon_segment_compliance.fill_(0.1)
        for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
            solver = solver_type(model, iterations=4)
            state, output, control = model.state(), model.state(), model.control()
            solver.reset(state)
            solver.step(state, output, control, None, 1.0 / 60.0)
            state.assign(output)
            baseline = model.state()
            baseline.assign(state)
            initial_arc = float(0.0)
            if not dynamic:
                radial_l = solver.tendon_segment_attachment_r.numpy()[0]
                radial_r = solver.tendon_segment_attachment_l.numpy()[1]
                initial_arc = 0.1 * abs(math.atan2(np.cross(radial_l, radial_r)[2], np.dot(radial_l, radial_r)))
            graph = None
            if device.is_cuda:
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state, output, control, None, 1.0 / 60.0)
                graph = capture.graph

            # The first trial needs material from the other span, but has enough
            # overall. The second exceeds the entire budget. The third releases
            # material from the arc. All three use supported positive wraps.
            cases = ((0.001, 0.02),) if exhausted else ((0.003, 0.02), (0.001, -0.02))
            if dynamic:
                cases = ((0.001 if exhausted else 1.0, 0.0),)
            for free_rest, angle in cases:
                for use_graph in (False, True) if graph is not None else (False,):
                    with test.subTest(solver=solver_type.__name__, dynamic=dynamic, rest=free_rest, graph=use_graph):
                        solver.reset(state, flags=0)
                        state.assign(baseline)
                        state.tendon_segment_rest_length.assign([free_rest, 0.0] if dynamic else [free_rest / 2.0] * 2)
                        poses = state.body_q.numpy()
                        poses[moved_body, :3] = (
                            (0.0, 0.0, 0.0) if dynamic else (0.4 * math.cos(angle), 0.4 * math.sin(angle), 0.0)
                        )
                        state.body_q.assign(poses)
                        if use_graph:
                            wp.capture_launch(graph)
                        else:
                            solver.step(state, output, control, None, 1.0 / 60.0)
                        if dynamic:
                            test.assertTrue(bool(output.tendon_guide_active.numpy()[candidate]))
                        arc = initial_arc + 0.1 * angle
                        if dynamic:
                            radial_l = solver.tendon_segment_attachment_r.numpy()[0]
                            radial_r = solver.tendon_segment_attachment_l.numpy()[1]
                            arc = 0.1 * abs(math.atan2(np.cross(radial_l, radial_r)[1], np.dot(radial_l, radial_r)))
                        total_material = float(output.tendon_segment_rest_length.numpy().sum()) + arc
                        if exhausted:
                            test.assertGreater(total_material - (free_rest + initial_arc), 1.0e-4)
                        else:
                            test.assertAlmostEqual(total_material, free_rest + initial_arc, delta=2.0e-7)
    wp.synchronize_device(device)


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

    # A nanometre-scale refill can still change a stiff neighbor's force by
    # several newtons. It must not be classified as an exhausted donor.
    for reverse in (False, True):
        tensions = [40.0, 19.99, 0.0]
        expected = np.array([23.988, 19.99, 19.99 / 1.2])
        if reverse:
            tensions, expected = tensions[::-1], expected[::-1]
        with test.subTest(tiny_refill=True, reverse=reverse):
            force, rest, initial, count = solve_material_case(
                device,
                tensions,
                [1.2, 1.2],
                segment_lengths=[0.001, 0.002, 0.001],
                segment_compliance=[1.0e-10, 1.0e-4, 1.0e-10],
                tol=1.0e-3,
            )
            np.testing.assert_allclose(force, expected, atol=1.0e-4, rtol=1.0e-3)
            np.testing.assert_allclose(rest.sum(), initial.sum(), atol=1.0e-9, rtol=0)
            test.assertLess(count, 256)


def test_adaptive_sweeps_rest_floor_roundoff(test, device):
    """Accept depleted-span convergence without a false material-exhaustion error."""
    cases = [
        {
            "tensions": [15.58, 2.69, 10.72, 10.28, -0.266, 9.26, 7.70, 12.79],
            "ratios": [1.148, 1.483, 1.423, 1.240, 1.374, 1.373, 1.334],
            "segment_lengths": [1, 1, 1, 1, 0.002, 1, 1, 1],
            "segment_compliance": [0.000404, 0.003513, 0.000907, 0.002589, 0.000396, 0.000792, 0.0000732, 0.001602],
            "rolling_left": [0, -0.00723, 0.01114, -0.01219, 0.02693, 0.02498, -0.00115, -0.01030],
            "rolling_right": [0.00723, -0.01114, 0.01219, -0.02693, -0.02498, 0.00115, 0.01030, 0],
        },
        {
            # Retain these float32-sensitive inputs: draining the 2 mm slack
            # span used to overshoot its floor enough to print an ERROR.
            "tensions": [14.751089715672919, 19.9, -0.00589664940819655, -0.9815034265618756, -3.573101602159288],
            "ratios": [1.4849929919374645, 1.9556574016447539, 1.4258567912671964, 1.3688375035007576],
            "segment_lengths": [1, 1, 0.002, 1, 1],
            "segment_compliance": 0.01,
        },
    ]
    for case in cases:
        with test.subTest(spans=len(case["tensions"])):
            tension, rest, initial, count = solve_material_case(device, **case, tol=1.0e-3)
            reference, _, _, _ = solve_material_case(device, **case, tol=0.0, sweeps=4096, capture=False)
            test.assertLess(count, 256, "A depleted span must not force the full sweep budget")
            np.testing.assert_allclose(tension, reference, atol=1.0e-6, rtol=2.0e-3)
            np.testing.assert_allclose(rest.sum(), initial.sum(), atol=8.0e-7, rtol=0)
            donor = len(tension) // 2
            compliance = np.broadcast_to(np.float32(case["segment_compliance"]), tension.shape)
            # Recover the unclamped rest from the reported stretch tension.
            # It must respect the floor before reconstruction clamps it.
            raw_rest = float(np.float32(0.002)) - float(tension[donor]) * float(compliance[donor])
            test.assertAlmostEqual(raw_rest, 1.0e-6, delta=8.0e-10)


def test_adaptive_sweeps_coincident_pinholes(test, device):
    """Certify shared spans after all coincident-pinhole rows have updated them."""
    # The zero-length span makes three cone rows share the next span. Checking
    # only the immediately preceding row can miss the last row's final residual.
    points = [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    friction = [0.0, 0.1, 0.2, 0.01, 0.0]
    rest = [0.91, 1.0e-6, 0.99, 0.91]
    rho = np.exp(0.01 * np.pi / 2.0)
    # Only the last pinhole slips at equilibrium: its neighboring tensions
    # sum to 10 N and have ratio rho. The coincident pair stays inside its cone.
    expected = np.array([9.0, 0.0, 10.0 / (1.0 + rho), 10.0 * rho / (1.0 + rho)])
    for reverse in (False, True):
        positions = points[::-1] if reverse else points
        coefficients = friction[::-1] if reverse else friction
        rest_lengths = rest[::-1] if reverse else rest
        builder = newton.ModelBuilder()
        guides = [newton.TendonGuide(body=-1, offset=positions[0])]
        for i in range(1, len(positions)):
            guides.append(
                newton.TendonGuide(
                    body=-1,
                    offset=positions[i],
                    guide_type=newton.TendonGuideType.ANCHOR if i == 4 else newton.TendonGuideType.PINHOLE,
                    mu=coefficients[i],
                    compliance=0.01,
                    rest_length=rest_lengths[i - 1],
                )
            )
        builder.add_tendon(guides)
        model = builder.finalize(device=device)
        for solver_type in (newton.solvers.SolverXPBD, newton.solvers.SolverVBD):
            with test.subTest(reverse=reverse, solver=solver_type.__name__):
                solver = solver_type(model, iterations=1, tendon_settle_tol=1.0e-6)
                state, output, control = model.state(), model.state(), model.control()
                solver.step(state, output, control, None, 0.01)
                tension = np.maximum(solver.tendon_segment_stretch.numpy() / 0.01, 0.0)
                np.testing.assert_allclose(tension, expected[::-1] if reverse else expected, atol=2.0e-5, rtol=0)
                test.assertLess(int(solver.tendon_cone_sweep_count.numpy()[0]), 256)
                if device.is_cuda:
                    with wp.ScopedCapture(device=device) as captured:
                        solver.step(state, output, control, None, 0.01)
                    solver.tendon_segment_stretch.zero_()
                    wp.capture_launch(captured.graph)
                    np.testing.assert_array_equal(
                        np.maximum(solver.tendon_segment_stretch.numpy() / 0.01, 0.0), tension
                    )


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
            stored_tension = np.maximum((np.float32(1.0) - rest) / np.float32(1.0e-8), 0.0)
            force_resolution = 0.5 * np.abs(np.spacing(rest)) / np.float32(1.0e-8)
            force_resolution += 2.0 * np.abs(np.spacing(tension))
            np.testing.assert_array_less(np.abs(stored_tension - tension), force_resolution)
            for i, rho in enumerate(ratios):
                left, right = stored_tension[i : i + 2]
                error = max(left - rho * right, right - rho * left, 0.0)
                rounding = max(
                    force_resolution[i] + rho * force_resolution[i + 1],
                    force_resolution[i + 1] + rho * force_resolution[i],
                )
                test.assertLessEqual(error, 1.0e-6 + 1.0e-6 * max(tension[i : i + 2]) + rounding)
            # Spending the full sweep budget cannot recover precision lost on writeback.
            _, full_rest, _, full_count = solve_material_case(
                device, initial, ratios, tol=0.0, segment_compliance=1.0e-8, capture=False
            )
            test.assertEqual(full_count, 256)
            np.testing.assert_array_equal(rest, full_rest)


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
    "test_rolling_material_exhaustion_diagnostic",
    test_rolling_material_exhaustion_diagnostic,
    devices=devices,
    check_output=False,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_replenished_donor",
    test_adaptive_sweeps_replenished_donor,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_rest_floor_roundoff",
    test_adaptive_sweeps_rest_floor_roundoff,
    devices=devices,
)
add_function_test(
    TestTendonMaterialSweeps,
    "test_adaptive_sweeps_coincident_pinholes",
    test_adaptive_sweeps_coincident_pinholes,
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
