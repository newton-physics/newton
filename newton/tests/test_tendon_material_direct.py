# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the finite-friction direct material return map."""

import math
import unittest
from dataclasses import dataclass

import numpy as np
import warp as wp

from newton._src.solvers.tendon_material import TendonMaterialStatus, solve_tendon_material_component
from newton.tests.unittest_utils import add_function_test, get_test_devices


@dataclass(frozen=True)
class _Piece:
    lo: float
    hi: float
    a: float
    b: float


def _prefix_minimizer(pieces: list[_Piece]) -> float:
    """Locate the prefix minimum in T=exp(u), returning zero for u=-infinity."""
    if pieces[0].b <= 0.0:
        return 0.0
    for piece in pieces:
        if piece.b <= 0.0:
            continue
        root = piece.b / piece.a
        if piece.lo * (1.0 - 1.0e-12) <= root <= piece.hi * (1.0 + 1.0e-12):
            return root
    raise ArithmeticError("No tension-domain derivative root found")


def _reference_taut(x0, compliance, cap_ratio):
    """Solve the same DP without any logarithm or exponential evaluations.

    _Pieces represent dF/du evaluated at T=exp(u), NOT dF/dT. Thus the
    derivative is still A*T-B and remains monotone. The monotone coordinate
    change preserves every sign and minimizer used by the recurrence.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    compliance = np.asarray(compliance, dtype=np.float64)
    cap_ratio = np.asarray(cap_ratio, dtype=np.float64)
    n = len(x0)
    if n == 0 or compliance.shape != x0.shape or cap_ratio.shape != (n - 1,):
        raise ValueError("Inconsistent chain dimensions")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(compliance)):
        raise ValueError("Nonfinite material data")
    if np.any(compliance <= 0) or np.any(cap_ratio < 1) or not np.all(np.isfinite(cap_ratio)):
        raise ValueError("Positive compliance and finite capstan ratios >= 1 required")
    if float(x0.sum()) <= 0:
        raise ValueError("The reference requires positive total extension")

    pieces = [_Piece(0.0, math.inf, float(compliance[0]), float(x0[0]))]
    minima = [_prefix_minimizer(pieces)]
    piece_visits = 1
    max_pieces = 1
    for i in range(1, n):
        previous_minimum = minima[-1]
        cap = float(cap_ratio[i - 1])
        transformed = []
        if previous_minimum > 0.0:
            for piece in pieces:
                if piece.lo < previous_minimum:
                    transformed.append(
                        _Piece(
                            piece.lo / cap,
                            min(piece.hi, previous_minimum) / cap,
                            piece.a * cap,
                            piece.b,
                        )
                    )
            if cap > 1.0:
                transformed.append(_Piece(previous_minimum / cap, previous_minimum * cap, 0.0, 0.0))
        for piece in pieces:
            if piece.hi > previous_minimum:
                transformed.append(
                    _Piece(
                        max(piece.lo, previous_minimum) * cap,
                        piece.hi * cap,
                        piece.a / cap,
                        piece.b,
                    )
                )
        pieces = [_Piece(p.lo, p.hi, p.a + compliance[i], p.b + x0[i]) for p in transformed]
        minima.append(_prefix_minimizer(pieces))
        piece_visits += len(pieces)
        max_pieces = max(max_pieces, len(pieces))

    tension = np.empty(n)
    tension[-1] = minima[-1]
    for i in range(n - 2, -1, -1):
        tension[i] = max(
            tension[i + 1] / cap_ratio[i],
            min(tension[i + 1] * cap_ratio[i], minima[i]),
        )
    return compliance * tension, {"max_pieces": max_pieces, "piece_visits": piece_visits}


def _reference_slack(initial, upper, weights=None):
    """Minimize weighted squared net transfer as an independent slack oracle.

    With b=x0-min(U,0), p_i=sum_(j<=i)b_j, y_i=z_i-p_i, feasibility becomes
    0<=y_0<=...<=y_(n-2)<=-sum(b). Weighted PAVA fits y to -p, then clips to
    that common interval. An already feasible slack allocation remains unchanged.
    """
    x0 = np.asarray(initial, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if x0.ndim != 1 or upper.shape != x0.shape or len(x0) == 0:
        raise ValueError("Inconsistent slack chain dimensions")
    if not np.isfinite(x0).all() or not np.isfinite(upper).all():
        raise ValueError("Finite slack inputs required")
    slack_upper = np.minimum(upper, 0.0)
    if x0.sum() > slack_upper.sum():
        raise ValueError("No legal zero-tension allocation preserves this total")
    n = len(x0)
    if n == 1:
        return x0.copy(), np.empty(0)
    weights = np.ones(n - 1) if weights is None else np.asarray(weights, dtype=np.float64)
    if weights.shape != (n - 1,) or np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("Positive transfer weights required")
    prefix = np.cumsum(x0 - slack_upper)
    capacity = -float(prefix[-1])
    targets = -prefix[:-1]
    blocks = []
    for i in range(n - 1):
        blocks.append([i, i + 1, weights[i], weights[i] * targets[i]])
        while len(blocks) >= 2 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(n - 1)
    for first, last, weight, weighted_target in blocks:
        fitted[first:last] = np.clip(weighted_target / weight, 0.0, capacity)
    transfer = fitted + prefix[:-1]
    output = x0.copy()
    output[:-1] -= transfer
    output[1:] += transfer
    return output, transfer


@wp.kernel(enable_backward=False)
def solve_material_direct(
    initial: wp.array[float],
    compliance: wp.array[float],
    cap_ratio: wp.array[float],
    upper: wp.array[float],
    output: wp.array[float],
    transfer: wp.array[wp.float64],
    status: wp.array[int],
    faces: wp.array[int],
    cache_valid: wp.array[int],
    piece_count: wp.array[int],
    n: int,
    allow_warm: int,
):
    solve_tendon_material_component(
        initial,
        compliance,
        cap_ratio,
        upper,
        output,
        transfer,
        status,
        faces,
        cache_valid,
        piece_count,
        n,
        allow_warm,
        wp.tid(),
        n,
        n - 1,
    )


@wp.kernel(enable_backward=False)
def solve_material_packed(
    initial: wp.array[float],
    compliance: wp.array[float],
    cap_ratio: wp.array[float],
    upper: wp.array[float],
    output: wp.array[float],
    transfer: wp.array[wp.float64],
    status: wp.array[int],
    faces: wp.array[int],
    cache_valid: wp.array[int],
    piece_count: wp.array[int],
    starts: wp.array[int],
    counts: wp.array[int],
):
    component = wp.tid()
    solve_tendon_material_component(
        initial,
        compliance,
        cap_ratio,
        upper,
        output,
        transfer,
        status,
        faces,
        cache_valid,
        piece_count,
        counts[component],
        1,
        starts[component],
        1,
        1,
    )


def _solve(device, initial, compliance, caps, upper=None, faces=None, allow_warm=True):
    initial = np.atleast_2d(initial).astype(np.float32)
    compliance = np.broadcast_to(compliance, initial.shape).astype(np.float32)
    count, n = initial.shape
    caps = np.broadcast_to(caps, (count, n - 1)).astype(np.float32)
    upper = np.full_like(initial, 1.0e20) if upper is None else np.broadcast_to(upper, initial.shape).astype(np.float32)
    arrays = [wp.array(value.ravel(), dtype=float, device=device) for value in (initial, compliance, caps, upper)]
    output = wp.full(count * n, -123.0, dtype=float, device=device)
    transfer = wp.full(count * n, -123.0, dtype=wp.float64, device=device)
    status = wp.zeros(count, dtype=int, device=device)
    cache_faces = (
        np.zeros((count, n - 1), dtype=np.int32)
        if faces is None
        else np.broadcast_to(faces, (count, n - 1)).astype(np.int32)
    )
    cache = wp.array(cache_faces.ravel(), dtype=int, device=device)
    valid = wp.full(count, int(faces is not None), dtype=int, device=device)
    pieces = wp.zeros(count, dtype=int, device=device)
    wp.launch(
        solve_material_direct,
        dim=count,
        inputs=[*arrays, output, transfer, status, cache, valid, pieces, n, int(allow_warm)],
        device=device,
    )
    return output.numpy().reshape(count, n), status.numpy(), cache.numpy().reshape(count, n - 1)


def _assert_reference(test, x, c, k, output, rtol=3.0e-6):
    x, c, k = (np.asarray(a, dtype=np.float32) for a in (x, c, k))
    expected, _ = _reference_taut(x, c, k)
    peak = np.max(expected / c)
    test.assertLessEqual(float(np.max(np.abs((output - expected) / c))), rtol * peak)


def test_cold_three_span_return_map(test, device):
    """Verify cold three span return map."""
    y, status, _ = _solve(device, [9, 1, 10], [1, 1, 1], [2, 2])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_allclose(y[0], [8, 4, 8])


def test_correct_warm_faces(test, device):
    """Verify correct warm faces."""
    y, status, _ = _solve(device, [9, 1, 10], [1, 1, 1], [2, 2], faces=[1, -1])
    test.assertEqual(status[0], TendonMaterialStatus.WARM)
    np.testing.assert_allclose(y[0], [8, 4, 8])


def test_stale_warm_face_is_recomputed(test, device):
    """Verify stale warm face is recomputed."""
    x, c, k = [10, 1.0e-8], [1, 1.0e-9], [2]
    y, status, _ = _solve(device, x, c, k, faces=[1])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    _assert_reference(test, x, c, k, y[0])
    np.testing.assert_allclose(y[0] / np.float32(c), [10, 10], rtol=2.0e-6)


def test_slip_reversal_invalidates_warm_face(test, device):
    """Verify slip reversal invalidates warm face."""
    y, status, _ = _solve(device, [1, 9], [1, 1], [2], faces=[1])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    _assert_reference(test, [1, 9], [1, 1], [2], y[0])


def test_slack_redistribution_moves_least_material(test, device):
    """Verify slack redistribution moves least material."""
    y, status, faces = _solve(device, [1, -2], [1, 1], [2], faces=[1])
    test.assertEqual(status[0], TendonMaterialStatus.SLACK_REDISTRIBUTED)
    np.testing.assert_array_equal(y[0], [0, -1])
    np.testing.assert_array_equal(faces, 0)


def test_infeasible_slack_does_not_partially_write(test, device):
    """Verify infeasible slack does not partially write."""
    y, status, faces = _solve(device, [-3, 2], [1, 1], [2], upper=[-2, 5], faces=[1])
    test.assertEqual(status[0], TendonMaterialStatus.SLACK_INFEASIBLE)
    np.testing.assert_array_equal(y, -123.0)
    np.testing.assert_array_equal(faces, 1)


def test_slack_bound_deficit_survives_large_background(test, device):
    """Verify slack bound deficit survives large background."""
    a = np.float32(1.0e20)
    y, status, faces = _solve(device, [0, -a, 0], [1, 1, 1], [2, 2], upper=[-a, -1, 0], faces=[1, 1])
    test.assertEqual(status[0], TendonMaterialStatus.SLACK_INFEASIBLE)
    np.testing.assert_array_equal(y, -123.0)
    np.testing.assert_array_equal(faces, 1)


def test_slack_bilateral_transfer_and_reversal(test, device):
    """Verify slack bilateral transfer and reversal."""
    for x, upper, expected in (
        ([-1, 1, -1], [3, 3, 3], [-0.5, 0, -0.5]),
        ([-5, 2, -1], [-2, 2, 2], [-4, 0, 0]),
        ([3, -1, -2], [5, 5, 5], [0, 0, 0]),
        ([2, -4, 1], [-1, 2, 2], [-1, 0, 0]),
    ):
        with test.subTest(x=x):
            y, status, _ = _solve(device, x, [1, 1.0e-12, 1.0e-4], [1.1, 2], upper=upper)
            test.assertEqual(status[0], TendonMaterialStatus.SLACK_REDISTRIBUTED)
            np.testing.assert_allclose(y[0], expected)
            reverse, _, _ = _solve(device, x[::-1], [1, 1, 1], [2, 1.1], upper=upper[::-1])
            np.testing.assert_array_equal(y[0], reverse[0, ::-1])


def test_random_slack_matches_isotonic_oracle(test, device):
    """Verify random slack matches isotonic oracle."""
    rng = np.random.default_rng(2026090904)
    for n in (2, 3, 8, 32):
        count = 128
        scale = np.power(10.0, rng.uniform(-8, 2, (count, 1)))
        upper = (rng.uniform(-1, 2, (count, n)) * scale).astype(np.float32)
        feasible = np.minimum(upper, 0) - rng.uniform(0, 2, (count, n)) * scale
        transfer = rng.uniform(-3, 3, (count, n - 1)) * scale
        x = feasible.copy()
        x[:, :-1] += transfer
        x[:, 1:] -= transfer
        x = x.astype(np.float32)
        c = np.power(10.0, rng.uniform(-12, -3, (count, n))).astype(np.float32)
        k = np.exp(rng.uniform(0.01, 0.3, (count, n - 1))).astype(np.float32)
        y, codes, _ = _solve(device, x, c, k, upper=upper)
        for i in range(count):
            with test.subTest(n=n, route=i):
                test.assertIn(
                    codes[i], (TendonMaterialStatus.SLACK_UNCHANGED, TendonMaterialStatus.SLACK_REDISTRIBUTED)
                )
                expected, _ = _reference_slack(x[i], upper[i])
                test.assertTrue(np.all(y[i] <= np.minimum(upper[i], 0)))
                np.testing.assert_allclose(y[i], expected, rtol=2.0e-7, atol=1.0e-12 * scale[i, 0])
                test.assertLessEqual(
                    abs(y[i].sum(dtype=np.float64) - x[i].sum(dtype=np.float64)),
                    1.0e-7 * np.sum(np.abs(y[i]), dtype=np.float64) + 1.0e-12 * scale[i, 0],
                )


def test_slack_single_span_and_zero_total(test, device):
    """Verify slack single span and zero total."""
    for x, upper in (([-1], [-1]), ([0], [0])):
        y, status, _ = _solve(device, x, [1], [], upper=upper)
        test.assertEqual(status[0], TendonMaterialStatus.SLACK_UNCHANGED)
        np.testing.assert_array_equal(y[0], x)


def test_slack_total_survives_large_cancellation(test, device):
    """Verify slack total survives large cancellation."""
    x = np.array([1.0e20, -1, -1.0e20], dtype=np.float32)
    for trial, expected in ((x, [0, 0, -1]), (x[::-1], [-1, 0, 0])):
        y, status, _ = _solve(device, trial, [1, 1, 1], [2, 2])
        test.assertEqual(status[0], TendonMaterialStatus.SLACK_REDISTRIBUTED)
        np.testing.assert_array_equal(y[0], expected)


def test_slack_block_merging_preserves_prefix_low_parts(test, device):
    """Verify slack block merging preserves prefix low parts."""
    for magnitude, slack in ((1.0e20, 1.0), (1.0e8, 1.0e-8)):
        x = np.array([-magnitude, -slack, 2 * magnitude, 0, -magnitude], dtype=np.float32)
        expected = np.array([-0.75, 0, 0, 0, -0.25]) * np.float32(slack)
        for trial, reference in ((x, expected), (x[::-1], expected[::-1])):
            with test.subTest(magnitude=magnitude, reversed=trial is not x):
                y, status, _ = _solve(device, trial, np.ones(5), np.full(4, 2))
                test.assertEqual(status[0], TendonMaterialStatus.SLACK_REDISTRIBUTED)
                np.testing.assert_allclose(y[0], reference, rtol=1.0e-7, atol=0)


def test_slack_reconstruction_preserves_small_neighbor_allocation(test, device):
    """Verify slack reconstruction preserves small neighbor allocation."""
    a = np.float32(1.0e20)
    x = np.array([-a, -1, 2, -1, -1], dtype=np.float32)
    expected = np.array([-a, 0, 0, 0, -1], dtype=np.float32)
    for trial, reference in ((x, expected), (x[::-1], expected[::-1])):
        y, status, _ = _solve(device, trial, np.ones(5), np.full(4, 2))
        test.assertEqual(status[0], TendonMaterialStatus.SLACK_REDISTRIBUTED)
        np.testing.assert_array_equal(y[0], reference)


def test_three_scale_slack_cancellation_is_rejected_explicitly(test, device):
    """Verify three scale slack cancellation is rejected explicitly."""
    x = [0, 0, 2.0**90, -(2.0**-56), 2.0**36, -(2.0**90), -(2.0**36)]
    y, status, faces = _solve(device, x, np.ones(7), np.full(6, 2), faces=np.ones(6))
    test.assertEqual(status[0], TendonMaterialStatus.NUMERICAL_FAILURE)
    np.testing.assert_array_equal(y, -123.0)
    np.testing.assert_array_equal(faces, 1)
    # No arithmetic is needed to retain a feasible history, even at this scale range.
    feasible = -np.abs(x)
    y, status, faces = _solve(device, feasible, np.ones(7), np.full(6, 2), faces=np.ones(6))
    test.assertEqual(status[0], TendonMaterialStatus.SLACK_UNCHANGED)
    np.testing.assert_array_equal(y[0], np.asarray(feasible, dtype=np.float32))
    np.testing.assert_array_equal(faces, 0)


def test_slack_invalidates_cache_before_reactivation(test, device):
    """Verify slack invalidates cache before reactivation."""
    snapshots = [[9, 1, 10], [-1, 1, -1], [-0.5, 0, -0.5], [9, 1, 10], [9, 1, 10]]
    trials = [wp.array(x, dtype=float, device=device) for x in snapshots]
    c = wp.ones(3, dtype=float, device=device)
    k = wp.full(2, 2.0, dtype=float, device=device)
    upper = wp.full(3, 100.0, dtype=float, device=device)
    output = wp.zeros(3, dtype=float, device=device)
    transfer = wp.zeros(3, dtype=wp.float64, device=device)
    status = wp.zeros(1, dtype=int, device=device)
    faces = wp.zeros(2, dtype=int, device=device)
    valid = wp.zeros(1, dtype=int, device=device)
    pieces = wp.zeros(1, dtype=int, device=device)
    saved_codes = wp.zeros(5, dtype=int, device=device)
    saved_valid = wp.zeros(5, dtype=int, device=device)
    saved_output = wp.zeros(15, dtype=float, device=device)

    def trace():
        valid.zero_()
        for i, trial in enumerate(trials):
            wp.launch(
                solve_material_direct,
                dim=1,
                inputs=[trial, c, k, upper, output, transfer, status, faces, valid, pieces, 3, 1],
                device=device,
            )
            wp.copy(saved_codes, status, dest_offset=i, count=1)
            wp.copy(saved_valid, valid, dest_offset=i, count=1)
            wp.copy(saved_output, output, dest_offset=3 * i, count=3)

    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            trace()
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
    else:
        trace()
    np.testing.assert_array_equal(
        saved_codes.numpy(),
        [
            TendonMaterialStatus.COLD,
            TendonMaterialStatus.SLACK_REDISTRIBUTED,
            TendonMaterialStatus.SLACK_UNCHANGED,
            TendonMaterialStatus.COLD,
            TendonMaterialStatus.WARM,
        ],
    )
    np.testing.assert_array_equal(saved_valid.numpy(), [1, 0, 0, 1, 1])
    np.testing.assert_allclose(
        saved_output.numpy().reshape(5, 3), [[8, 4, 8], [-0.5, 0, -0.5], [-0.5, 0, -0.5], [8, 4, 8], [8, 4, 8]]
    )


def test_feasible_slack_history_is_preserved(test, device):
    """Verify feasible slack history is preserved."""
    y, status, faces = _solve(device, [-1, -2], [1, 1], [2], faces=[1])
    test.assertEqual(status[0], TendonMaterialStatus.SLACK_UNCHANGED)
    np.testing.assert_array_equal(y[0], [-1, -2])
    np.testing.assert_array_equal(faces, 0)


def test_impossible_bounds_are_explicit(test, device):
    """Verify impossible bounds are explicit."""
    y, status, _ = _solve(device, [10, 1, 1], [1, 1, 1], [2, 2], upper=[100, 1, 100])
    test.assertEqual(status[0], TendonMaterialStatus.BOUND_INCOMPATIBLE)
    np.testing.assert_array_equal(y, -123.0)


def test_history_incompatible_but_cone_feasible_bounds(test, device):
    """Verify history incompatible but cone feasible bounds."""
    y, status, _ = _solve(device, [1, 9, 10], [1, 1, 1], [2, 2], upper=[3, 100, 100])
    test.assertEqual(status[0], TendonMaterialStatus.BOUND_INCOMPATIBLE)
    np.testing.assert_array_equal(y, -123.0)


def test_compatible_exact_bound_touch(test, device):
    """Verify compatible exact bound touch."""
    y, status, _ = _solve(device, [1, 8, 11], [1, 1, 1], [2, 2], upper=[3, 100, 100])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_allclose(y[0], [3, 6, 11])


def test_invalid_cached_face_is_only_a_failed_guess(test, device):
    """Verify invalid cached face is only a failed guess."""
    y, status, _ = _solve(device, [9, 1, 10], [1, 1, 1], [2, 2], faces=[99, -1])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_allclose(y[0], [8, 4, 8])


def test_nearly_zero_wrap_keeps_sticking_history(test, device):
    """Verify nearly zero wrap keeps sticking history."""
    x, c, k = np.ones(32), np.ones(32), np.full(31, np.float32(1.000001))
    y, status, _ = _solve(device, x, c, k, faces=np.ones(31, dtype=np.int32))
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_allclose(y[0], 1.0, rtol=1.0e-7)


def test_bound_touch_allows_only_double_roundoff(test, device):
    """Verify bound touch allows only double roundoff."""
    x = np.float32(0.0001070217476808466)
    c = np.float32(0.00010297950211679563)
    y, status, _ = _solve(device, [x], [c], [], upper=[x])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_array_equal(y[0], [x])


def test_random_wide_compliance_cold_and_warm(test, device):
    """Verify random wide compliance cold and warm."""
    rng = np.random.default_rng(20260909)
    count, n = 128, 32
    c = np.power(10.0, rng.uniform(-12.0, -3.0, (count, n))).astype(np.float32)
    x = (c * rng.uniform(-1.0, 20.0, (count, n))).astype(np.float32)
    k = np.exp(rng.uniform(0.01, 0.3, (count, n - 1))).astype(np.float32)
    y, codes, faces = _solve(device, x, c, k)
    for i in range(count):
        with test.subTest(route=i):
            test.assertEqual(codes[i], TendonMaterialStatus.COLD)
            _assert_reference(test, x[i], c[i], k[i], y[i])
    y2, codes2, _ = _solve(device, x, c, k, faces=faces)
    test.assertTrue(np.all(np.isin(codes2, [TendonMaterialStatus.COLD, TendonMaterialStatus.WARM])))
    for i in range(count):
        _assert_reference(test, x[i], c[i], k[i], y2[i])


def test_packed_components_do_not_overwrite_neighbors(test, device):
    """Keep component scratch and cached faces inside their packed intervals."""
    size = 11
    initial = np.zeros(size, dtype=np.float32)
    initial[1:4] = [9, 1, 10]
    initial[5:7] = [1, 9]
    arrays = [
        wp.array(initial, dtype=float, device=device),
        wp.ones(size, dtype=float, device=device),
        wp.full(size, 2.0, dtype=float, device=device),
        wp.full(size, 100.0, dtype=float, device=device),
    ]
    output = wp.full(size, -123.0, dtype=float, device=device)
    transfer = wp.full(size, -123.0, dtype=wp.float64, device=device)
    status = wp.full(size, 73, dtype=int, device=device)
    faces = wp.full(size, 77, dtype=int, device=device)
    valid = wp.zeros(size, dtype=int, device=device)
    pieces = wp.full(size, 74, dtype=int, device=device)
    starts = wp.array([1, 5], dtype=int, device=device)
    counts = wp.array([3, 2], dtype=int, device=device)
    inputs = [*arrays, output, transfer, status, faces, valid, pieces, starts, counts]
    for expected_status in (TendonMaterialStatus.COLD, TendonMaterialStatus.WARM):
        wp.launch(solve_material_packed, dim=2, inputs=inputs, device=device)
        actual = output.numpy()
        np.testing.assert_allclose(actual[1:4], [8, 4, 8])
        np.testing.assert_allclose(actual[5:7], [10 / 3, 20 / 3])
        np.testing.assert_array_equal(actual[[0, 4, 7, 8, 9, 10]], -123.0)
        np.testing.assert_array_equal(transfer.numpy()[[0, 4, 7, 8, 9, 10]], -123.0)
        np.testing.assert_array_equal(transfer.numpy()[[3, 6]], 0.0)
        np.testing.assert_allclose(transfer.numpy()[1:4], [1.0, -2.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(transfer.numpy()[5:7], [-7.0 / 3.0, 0.0], atol=1e-14)
        np.testing.assert_array_equal(status.numpy()[[1, 5]], expected_status)
        np.testing.assert_array_equal(status.numpy()[[0, 2, 3, 4, 6, 7, 8, 9, 10]], 73)
        np.testing.assert_array_equal(faces.numpy()[[0, 3, 4, 6, 7, 8, 9, 10]], 77)
        np.testing.assert_array_equal(valid.numpy()[[1, 5]], 1)
        np.testing.assert_array_equal(valid.numpy()[[0, 2, 3, 4, 6, 7, 8, 9, 10]], 0)
        np.testing.assert_array_equal(pieces.numpy()[[0, 2, 3, 4, 6, 7, 8, 9, 10]], 74)


def test_invalid_inputs_preserve_output_and_cache(test, device):
    """Reject invalid numeric inputs without partially accepting a component."""
    cases = (
        ([np.nan, 1], [1, 1], [2], [100, 100]),
        ([1, np.inf], [1, 1], [2], [100, 100]),
        ([9, 1], [0, 1], [2], [100, 100]),
        ([9, 1], [1, -1], [2], [100, 100]),
        ([9, 1], [1.0e-26, 1], [2], [100, 100]),
        ([9, 1], [1, np.nan], [2], [100, 100]),
        ([9, 1], [1, 1], [0.999], [100, 100]),
        ([9, 1], [1, 1], [4.001], [100, 100]),
        ([9, 1], [1, 1], [np.inf], [100, 100]),
        ([9, 1], [1, 1], [2], [np.nan, 100]),
    )
    for initial, compliance, caps, upper in cases:
        with test.subTest(initial=initial, compliance=compliance, caps=caps, upper=upper):
            arrays = [wp.array(values, dtype=float, device=device) for values in (initial, compliance, caps, upper)]
            output = wp.full(2, -123.0, dtype=float, device=device)
            transfer = wp.full(2, -123.0, dtype=wp.float64, device=device)
            status = wp.zeros(1, dtype=int, device=device)
            faces = wp.ones(1, dtype=int, device=device)
            valid = wp.ones(1, dtype=int, device=device)
            pieces = wp.full(1, 77, dtype=int, device=device)
            wp.launch(
                solve_material_direct,
                dim=1,
                inputs=[*arrays, output, transfer, status, faces, valid, pieces, 2, 1],
                device=device,
            )
            test.assertEqual(status.numpy()[0], TendonMaterialStatus.BAD_INPUT)
            np.testing.assert_array_equal(output.numpy(), -123.0)
            np.testing.assert_array_equal(transfer.numpy(), -123.0)
            np.testing.assert_array_equal(faces.numpy(), 1)
            np.testing.assert_array_equal(valid.numpy(), 1)
            np.testing.assert_array_equal(pieces.numpy(), 0)


def test_unsupported_component_sizes_do_not_access_span_data(test, device):
    """Reject empty or oversized components before reading their span arrays."""
    size = 2
    arrays = [wp.zeros(size, dtype=float, device=device) for _ in range(4)]
    output = wp.full(size, -123.0, dtype=float, device=device)
    transfer = wp.full(size, -123.0, dtype=wp.float64, device=device)
    status = wp.zeros(size, dtype=int, device=device)
    faces = wp.ones(size, dtype=int, device=device)
    valid = wp.ones(size, dtype=int, device=device)
    pieces = wp.full(size, 77, dtype=int, device=device)
    starts = wp.array([0, 1], dtype=int, device=device)
    counts = wp.array([0, 33], dtype=int, device=device)
    wp.launch(
        solve_material_packed,
        dim=2,
        inputs=[*arrays, output, transfer, status, faces, valid, pieces, starts, counts],
        device=device,
    )
    np.testing.assert_array_equal(status.numpy(), TendonMaterialStatus.BAD_INPUT)
    np.testing.assert_array_equal(output.numpy(), -123.0)
    np.testing.assert_array_equal(transfer.numpy(), -123.0)
    np.testing.assert_array_equal(faces.numpy(), 1)
    np.testing.assert_array_equal(valid.numpy(), 1)
    np.testing.assert_array_equal(pieces.numpy(), 0)


def test_failed_solves_preserve_warm_cache(test, device):
    """Retain cached state and output for every post-validation failure code."""
    cases = (
        ([-3, 2], [-2, 5], TendonMaterialStatus.SLACK_INFEASIBLE),
        ([10, 1, 1], [100, 1, 100], TendonMaterialStatus.BOUND_INCOMPATIBLE),
        (
            [0, 0, 2.0**90, -(2.0**-56), 2.0**36, -(2.0**90), -(2.0**36)],
            [1.0e30] * 7,
            TendonMaterialStatus.NUMERICAL_FAILURE,
        ),
    )
    for initial, upper, expected_status in cases:
        with test.subTest(status=expected_status):
            n = len(initial)
            arrays = [
                wp.array(initial, dtype=float, device=device),
                wp.ones(n, dtype=float, device=device),
                wp.full(n - 1, 2.0, dtype=float, device=device),
                wp.array(upper, dtype=float, device=device),
            ]
            output = wp.full(n, -123.0, dtype=float, device=device)
            transfer = wp.full(n, -123.0, dtype=wp.float64, device=device)
            status = wp.zeros(1, dtype=int, device=device)
            faces = wp.ones(n - 1, dtype=int, device=device)
            valid = wp.ones(1, dtype=int, device=device)
            pieces = wp.zeros(1, dtype=int, device=device)
            wp.launch(
                solve_material_direct,
                dim=1,
                inputs=[*arrays, output, transfer, status, faces, valid, pieces, n, 1],
                device=device,
            )
            test.assertEqual(status.numpy()[0], expected_status)
            np.testing.assert_array_equal(output.numpy(), -123.0)
            np.testing.assert_array_equal(transfer.numpy(), -123.0)
            np.testing.assert_array_equal(faces.numpy(), 1)
            np.testing.assert_array_equal(valid.numpy(), 1)


def test_zero_friction_limit_uses_same_return_map(test, device):
    """Equalize tension at a unit cap ratio without a separate solve path."""
    y, status, _ = _solve(device, [9, 1], [1, 4], [1])
    test.assertEqual(status[0], TendonMaterialStatus.COLD)
    np.testing.assert_allclose(y[0], [2, 8])


def test_two_span_projection_matches_closed_form(test, device):
    """Match the independent two-span capstan solution over varied stiffnesses."""
    rng = np.random.default_rng(2026091001)
    compliance = np.power(10.0, rng.uniform(-12, -3, (256, 2))).astype(np.float32)
    initial = (compliance * rng.uniform(0.1, 20, (256, 2))).astype(np.float32)
    caps = rng.uniform(1.01, 4, (256, 1)).astype(np.float32)
    actual, status, _ = _solve(device, initial, compliance, caps)
    np.testing.assert_array_equal(status, TendonMaterialStatus.COLD)
    for x, c, cap, result in zip(
        initial.astype(np.float64), compliance.astype(np.float64), caps[:, 0].astype(np.float64), actual, strict=True
    ):
        trial = x / c
        if trial[0] > cap * trial[1]:
            tension = np.array([cap, 1]) * (x.sum() / (cap * c[0] + c[1]))
        elif trial[1] > cap * trial[0]:
            tension = np.array([1, cap]) * (x.sum() / (c[0] + cap * c[1]))
        else:
            tension = trial
        np.testing.assert_allclose(result / c, tension, rtol=2.0e-6)


class TestTendonMaterialDirect(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(
    TestTendonMaterialDirect,
    "test_two_span_projection_matches_closed_form",
    test_two_span_projection_matches_closed_form,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_failed_solves_preserve_warm_cache",
    test_failed_solves_preserve_warm_cache,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_zero_friction_limit_uses_same_return_map",
    test_zero_friction_limit_uses_same_return_map,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_packed_components_do_not_overwrite_neighbors",
    test_packed_components_do_not_overwrite_neighbors,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_invalid_inputs_preserve_output_and_cache",
    test_invalid_inputs_preserve_output_and_cache,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_unsupported_component_sizes_do_not_access_span_data",
    test_unsupported_component_sizes_do_not_access_span_data,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect, "test_cold_three_span_return_map", test_cold_three_span_return_map, devices=devices
)
add_function_test(TestTendonMaterialDirect, "test_correct_warm_faces", test_correct_warm_faces, devices=devices)
add_function_test(
    TestTendonMaterialDirect, "test_stale_warm_face_is_recomputed", test_stale_warm_face_is_recomputed, devices=devices
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slip_reversal_invalidates_warm_face",
    test_slip_reversal_invalidates_warm_face,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_redistribution_moves_least_material",
    test_slack_redistribution_moves_least_material,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_infeasible_slack_does_not_partially_write",
    test_infeasible_slack_does_not_partially_write,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_bound_deficit_survives_large_background",
    test_slack_bound_deficit_survives_large_background,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_bilateral_transfer_and_reversal",
    test_slack_bilateral_transfer_and_reversal,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_random_slack_matches_isotonic_oracle",
    test_random_slack_matches_isotonic_oracle,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_single_span_and_zero_total",
    test_slack_single_span_and_zero_total,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_total_survives_large_cancellation",
    test_slack_total_survives_large_cancellation,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_block_merging_preserves_prefix_low_parts",
    test_slack_block_merging_preserves_prefix_low_parts,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_reconstruction_preserves_small_neighbor_allocation",
    test_slack_reconstruction_preserves_small_neighbor_allocation,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_three_scale_slack_cancellation_is_rejected_explicitly",
    test_three_scale_slack_cancellation_is_rejected_explicitly,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_slack_invalidates_cache_before_reactivation",
    test_slack_invalidates_cache_before_reactivation,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_feasible_slack_history_is_preserved",
    test_feasible_slack_history_is_preserved,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_impossible_bounds_are_explicit",
    test_impossible_bounds_are_explicit,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_history_incompatible_but_cone_feasible_bounds",
    test_history_incompatible_but_cone_feasible_bounds,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect, "test_compatible_exact_bound_touch", test_compatible_exact_bound_touch, devices=devices
)
add_function_test(
    TestTendonMaterialDirect,
    "test_invalid_cached_face_is_only_a_failed_guess",
    test_invalid_cached_face_is_only_a_failed_guess,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_nearly_zero_wrap_keeps_sticking_history",
    test_nearly_zero_wrap_keeps_sticking_history,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_bound_touch_allows_only_double_roundoff",
    test_bound_touch_allows_only_double_roundoff,
    devices=devices,
)
add_function_test(
    TestTendonMaterialDirect,
    "test_random_wide_compliance_cold_and_warm",
    test_random_wide_compliance_cold_and_warm,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
