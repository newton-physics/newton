# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.implicit_mpm.rheology_solver_kernels import (
    YieldParamVec,
    get_dilatancy,
    make_solve_flow_rule,
    normal_yield_bounds,
    shear_yield_stress,
    shear_yield_stress_camclay,
    solve_flow_rule_camclay,
    vec6,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Base flow rule solver (no viscosity); used by most tests.
solve_flow_rule = make_solve_flow_rule(has_viscosity=False, has_dilatancy=True)
# Viscosity-aware variant; used by the dispatch/viscosity tests.
solve_flow_rule_viscous = make_solve_flow_rule(has_viscosity=True, has_dilatancy=True)

devices = get_test_devices()


# ---------------------------------------------------------------------------
# Wrapper kernels
# ---------------------------------------------------------------------------


@wp.kernel
def test_flow_rule_impl_kernel(
    D: wp.array[vec6],
    b: wp.array[vec6],
    r: wp.array[vec6],
    yp: wp.array[YieldParamVec],
    u_out: wp.array[vec6],
):
    i = wp.tid()
    u_out[i] = solve_flow_rule(D[i], b[i], r[i], yp[i], 1.0)


@wp.kernel
def test_flow_rule_camclay_kernel(
    D: wp.array[vec6],
    b: wp.array[vec6],
    r: wp.array[vec6],
    yp: wp.array[YieldParamVec],
    u_out: wp.array[vec6],
):
    i = wp.tid()
    u_out[i] = solve_flow_rule_camclay(D[i], b[i], r[i], yp[i])


@wp.kernel
def test_flow_rule_dispatch_kernel(
    D: wp.array[vec6],
    b: wp.array[vec6],
    r: wp.array[vec6],
    yp: wp.array[YieldParamVec],
    volume: wp.array[float],
    u_out: wp.array[vec6],
):
    i = wp.tid()
    u_out[i] = solve_flow_rule_viscous(D[i], b[i], r[i], yp[i], volume[i])


@wp.kernel
def eval_shear_yield_kernel(
    yp: wp.array[YieldParamVec],
    r_N: wp.array[float],
    ys_out: wp.array[float],
    pmin_out: wp.array[float],
    pmax_out: wp.array[float],
):
    i = wp.tid()
    ys, _dys, pmin, pmax = shear_yield_stress(yp[i], r_N[i])
    ys_out[i] = ys
    pmin_out[i] = pmin
    pmax_out[i] = pmax


@wp.kernel
def eval_shear_yield_camclay_kernel(
    yp: wp.array[YieldParamVec],
    r_N: wp.array[float],
    ys_out: wp.array[float],
    pmin_out: wp.array[float],
    pmax_out: wp.array[float],
):
    i = wp.tid()
    ys, _dys, pmin, pmax = shear_yield_stress_camclay(yp[i], r_N[i])
    ys_out[i] = ys
    pmin_out[i] = pmin
    pmax_out[i] = pmax


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_arrays(D_np, b_np, r_np, yp_np, device):
    """Create warp arrays from numpy data (each shape (1, 6))."""
    D_wp = wp.array(D_np.reshape(1, 6), dtype=vec6, device=device)
    b_wp = wp.array(b_np.reshape(1, 6), dtype=vec6, device=device)
    r_wp = wp.array(r_np.reshape(1, 6), dtype=vec6, device=device)
    yp_wp = wp.array(yp_np.reshape(1, 6), dtype=YieldParamVec, device=device)
    u_wp = wp.zeros(1, dtype=vec6, device=device)
    return D_wp, b_wp, r_wp, yp_wp, u_wp


def _run_impl(D_np, b_np, r_np, yp_np, device):
    D_wp, b_wp, r_wp, yp_wp, u_wp = _make_arrays(D_np, b_np, r_np, yp_np, device)
    wp.launch(test_flow_rule_impl_kernel, dim=1, inputs=[D_wp, b_wp, r_wp, yp_wp, u_wp], device=device)
    return u_wp.numpy()[0]


def _run_camclay(D_np, b_np, r_np, yp_np, device):
    D_wp, b_wp, r_wp, yp_wp, u_wp = _make_arrays(D_np, b_np, r_np, yp_np, device)
    wp.launch(test_flow_rule_camclay_kernel, dim=1, inputs=[D_wp, b_wp, r_wp, yp_wp, u_wp], device=device)
    return u_wp.numpy()[0]


def _run_dispatch(D_np, b_np, r_np, yp_np, volume, device):
    D_wp, b_wp, r_wp, yp_wp, u_wp = _make_arrays(D_np, b_np, r_np, yp_np, device)
    vol_wp = wp.array([volume], dtype=float, device=device)
    wp.launch(test_flow_rule_dispatch_kernel, dim=1, inputs=[D_wp, b_wp, r_wp, yp_wp, vol_wp, u_wp], device=device)
    return u_wp.numpy()[0]


def _eval_shear_yield(yp_np, r_N_val, device):
    yp_wp = wp.array(yp_np.reshape(1, 6), dtype=YieldParamVec, device=device)
    r_N_wp = wp.array([r_N_val], dtype=float, device=device)
    ys_wp = wp.zeros(1, dtype=float, device=device)
    pmin_wp = wp.zeros(1, dtype=float, device=device)
    pmax_wp = wp.zeros(1, dtype=float, device=device)
    wp.launch(eval_shear_yield_kernel, dim=1, inputs=[yp_wp, r_N_wp, ys_wp, pmin_wp, pmax_wp], device=device)
    return ys_wp.numpy()[0], pmin_wp.numpy()[0], pmax_wp.numpy()[0]


def _eval_shear_yield_camclay(yp_np, r_N_val, device):
    yp_wp = wp.array(yp_np.reshape(1, 6), dtype=YieldParamVec, device=device)
    r_N_wp = wp.array([r_N_val], dtype=float, device=device)
    ys_wp = wp.zeros(1, dtype=float, device=device)
    pmin_wp = wp.zeros(1, dtype=float, device=device)
    pmax_wp = wp.zeros(1, dtype=float, device=device)
    wp.launch(eval_shear_yield_camclay_kernel, dim=1, inputs=[yp_wp, r_N_wp, ys_wp, pmin_wp, pmax_wp], device=device)
    return ys_wp.numpy()[0], pmin_wp.numpy()[0], pmax_wp.numpy()[0]


def _make_yield_params(
    friction=0.5, yield_pressure=100.0, tensile_ratio=0.0, yield_stress=10.0, dilatancy=0.0, viscosity=0.0
):
    """Build a YieldParamVec numpy array using the same layout as YieldParamVec.from_values."""
    ps = np.sqrt(3.0 / 2.0)
    return np.array(
        [
            yield_pressure * ps,  # p_max * sqrt(3/2)
            tensile_ratio * yield_pressure * ps,  # p_min * sqrt(3/2)
            yield_stress,  # s_max
            friction * yield_pressure,  # mu * p_max
            dilatancy,
            viscosity,
        ],
        dtype=np.float32,
    )


def _yield_surface_normal_impl(r, yp):
    """Compute the yield surface normal for the impl flow rule at stress r.

    The yield surface is |r_T| = s + mu * f(r_N) with piecewise-linear f.
    Returns the outward normal ∇g where g = |r_T| - ys(r_N).
    """
    r_N = float(r[0])
    r_T = r.copy()
    r_T[0] = 0.0
    r_T_norm = np.linalg.norm(r_T)

    p_max = float(yp[0])
    p_min = -max(0.0, float(yp[1]))
    mu = max(0.0, float(yp[3]) / p_max) if p_max > 0 else 0.0

    p1 = p_min + 0.5 * p_max
    p2 = 0.5 * p_max

    if r_N < p1:
        dys = mu
    elif r_N > p2:
        dys = -mu
    else:
        dys = 0.0

    grad = np.zeros(6, dtype=np.float64)
    grad[0] = -dys
    if r_T_norm > 1e-10:
        grad[1:] = r_T[1:] / r_T_norm
    return grad


def _yield_surface_normal_camclay(r, yp):
    """Compute the yield surface normal for the cam-clay flow rule at stress r.

    The yield surface is |r_T|^2 + beta^2 * (r_N - r_mid)^2 = c^2.
    Returns the outward normal ∇g (up to scale).
    """
    r_N = float(r[0])
    r_T = r.copy()
    r_T[0] = 0.0

    r_N_max = float(yp[0])
    r_N_min = -max(0.0, float(yp[1]))
    mu = max(0.0, float(yp[3]) / r_N_max) if r_N_max > 0 else 0.0

    ratio = r_N_min / r_N_max if r_N_max > 0 else 0.0
    beta_sq = mu * mu / (1.0 - 2.0 * ratio)
    r_mid = 0.5 * (r_N_min + r_N_max)

    grad = np.zeros(6, dtype=np.float64)
    grad[0] = 2.0 * beta_sq * (r_N - r_mid)
    grad[1:] = 2.0 * r_T[1:]
    return grad


def check_yield_normal_alignment(test, u, grad_f, tol=1e-2):
    """Check that velocity u is collinear with the yield surface normal grad_f."""
    u_norm = np.linalg.norm(u)
    g_norm = np.linalg.norm(grad_f)
    if u_norm < 1e-4 or g_norm < 1e-4:
        return
    cos_angle = float(np.dot(u, grad_f)) / (u_norm * g_norm)
    test.assertAlmostEqual(
        abs(cos_angle),
        1.0,
        places=2,
        msg=f"u not aligned with yield surface normal: |cos|={abs(cos_angle):.6f}",
    )


def check_flow_rule_invariants(test, u, D, b, yp, shear_yield_fn, device, tol=1.0e-3, check_alignment=True):
    """Check mathematical invariants of the flow-rule solution.

    Given u = solve_*(D, b, r_guess, yp), compute r = (u - b) / D and verify:
    1. Normal stress within yield bounds
    2. Deviatoric stress on or inside yield surface
    3. Complementarity: elastic => u_T ~ 0, sliding => on yield surface
    """
    r = (u - b) / D

    r_N = r[0]
    r_T = r.copy()
    r_T[0] = 0.0
    r_T_norm = np.linalg.norm(r_T)

    # Evaluate yield bounds using warp kernel
    r_N_clamped = float(np.clip(r_N, -max(0.0, yp[1]), yp[0]))
    ys, pmin, pmax = shear_yield_fn(yp, r_N_clamped, device)

    # 1. Normal stress in bounds
    test.assertGreaterEqual(r_N, pmin - tol, f"r_N={r_N} below pmin={pmin}")
    test.assertLessEqual(r_N, pmax + tol, f"r_N={r_N} above pmax={pmax}")

    # 2. Deviatoric on/inside yield surface
    test.assertLessEqual(r_T_norm, ys + tol, f"|r_T|={r_T_norm} exceeds yield stress={ys}")

    # 3. Complementarity
    u_T = u.copy()
    u_T[0] = 0.0
    u_T_norm = np.linalg.norm(u_T)
    if r_T_norm < ys - tol:
        # Elastic: tangential velocity should be ~zero
        test.assertLess(u_T_norm, tol, f"Elastic but |u_T|={u_T_norm} > 0")

    # 4. Alignment: if u_T is non-zero, it should be collinear with r_T
    #    (holds for impl and cam-clay NACC, but not cam-clay associated flow with anisotropic D)
    if check_alignment and u_T_norm > tol and r_T_norm > tol:
        cos_angle = np.dot(u_T, r_T) / (u_T_norm * r_T_norm)
        test.assertAlmostEqual(
            abs(float(cos_angle)),
            1.0,
            places=3,
            msg=f"u_T and r_T not collinear: |cos(angle)|={abs(cos_angle):.6f}",
        )


# ---------------------------------------------------------------------------
# Test scenarios — solve_flow_rule
# ---------------------------------------------------------------------------


def test_flow_rule_impl_elastic(test, device):
    """Small b so unconstrained stress is inside yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.5, 0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=50.0)

    u = _run_impl(D, b, r, yp, device)

    # r_0 = -b/D, should be well inside yield surface
    # u_T should be ~0 (elastic)
    u_T = u.copy()
    u_T[0] = 0.0
    test.assertLess(np.linalg.norm(u_T), 1e-3, "Elastic case: u_T should be ~zero")

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


def test_flow_rule_impl_sliding(test, device):
    """Large tangential b so stress hits yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0)

    u = _run_impl(D, b, r, yp, device)

    # Should be sliding: u_T != 0
    u_T = u.copy()
    u_T[0] = 0.0
    test.assertGreater(np.linalg.norm(u_T), 1.0, "Sliding case: u_T should be non-zero")

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


def test_flow_rule_impl_normal_clamping(test, device):
    """Large normal b pushes past yield pressure bounds."""
    D = np.full(6, 1.0, dtype=np.float32)
    # Large positive b[0] pushes r_N = -b/D far negative (past tensile limit)
    b = np.array([-500.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0)

    u = _run_impl(D, b, r, yp, device)

    # Normal component of r should be clamped
    r_out = (u - b) / D
    ps = np.sqrt(3.0 / 2.0)
    pmax = 100.0 * ps
    test.assertLessEqual(r_out[0], pmax + 1e-2, "r_N should be clamped to pmax")

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


def test_flow_rule_impl_anisotropic(test, device):
    """Non-uniform D to test anisotropic response."""
    D = np.array([2.0, 0.5, 1.0, 3.0, 0.1, 1.5], dtype=np.float32)
    b = np.array([0.0, 100.0, -50.0, 30.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0)

    u = _run_impl(D, b, r, yp, device)

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


def test_flow_rule_impl_zero_yield(test, device):
    """yield_pressure = 0, should return b (fully plastic)."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([10.0, 5.0, -3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.0, yield_pressure=0.0, yield_stress=0.0)

    u = _run_impl(D, b, r, yp, device)

    # With zero yield, stress should be ~zero => u ~ b
    np.testing.assert_allclose(u, b, atol=1e-3, err_msg="Zero yield: u should equal b")


def test_flow_rule_impl_dilatancy(test, device):
    """Non-zero dilatancy, check normal-tangential coupling."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp_no_dil = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0, dilatancy=0.0)
    yp_dil = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0, dilatancy=0.5)

    u_no_dil = _run_impl(D, b, r, yp_no_dil, device)
    u_dil = _run_impl(D, b, r, yp_dil, device)

    # With dilatancy, the normal velocity should differ due to coupling
    test.assertNotAlmostEqual(
        float(u_dil[0]),
        float(u_no_dil[0]),
        places=2,
        msg="Dilatancy should affect normal velocity",
    )

    check_flow_rule_invariants(test, u_dil, D, b, yp_dil, _eval_shear_yield, device)


def test_flow_rule_impl_dilatancy_orthogonal(test, device):
    """With dilatancy=1.0 (associated flow), u should be normal to the yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    # Use large yield_pressure so stress stays on the conical part (away from the cap)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=1000.0, yield_stress=0.0, dilatancy=1.0, tensile_ratio=0.0)

    u = _run_impl(D, b, r, yp, device)
    r_out = (u - b) / D

    grad_f = _yield_surface_normal_impl(r_out, yp)
    check_yield_normal_alignment(test, u, grad_f)
    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


def test_flow_rule_impl_dilatancy_orthogonal_aniso(test, device):
    """Associated flow with dilatancy=1.0 under anisotropic D: u normal to yield surface."""
    D = np.array([2.0, 0.5, 1.0, 3.0, 0.1, 1.5], dtype=np.float32)
    b = np.array([0.0, 100.0, -50.0, 30.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=1000.0, yield_stress=0.0, dilatancy=1.0, tensile_ratio=0.0)

    u = _run_impl(D, b, r, yp, device)
    r_out = (u - b) / D

    grad_f = _yield_surface_normal_impl(r_out, yp)
    check_yield_normal_alignment(test, u, grad_f)
    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield, device)


# ---------------------------------------------------------------------------
# Test scenarios — solve_flow_rule_camclay
# ---------------------------------------------------------------------------


def test_flow_rule_camclay_elastic(test, device):
    """Small b so unconstrained stress is inside cam-clay yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.5, 0.1, -0.1, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=0.0, tensile_ratio=0.1)

    u = _run_camclay(D, b, r, yp, device)

    # Elastic => u should be zero
    test.assertLess(np.linalg.norm(u), 1e-2, "Cam-clay elastic: u should be ~zero")


def test_flow_rule_camclay_sliding(test, device):
    """Large tangential b so stress hits cam-clay yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=0.0, tensile_ratio=0.1)

    u = _run_camclay(D, b, r, yp, device)

    # Should be sliding: u_T != 0
    u_T = u.copy()
    u_T[0] = 0.0
    test.assertGreater(np.linalg.norm(u_T), 1.0, "Cam-clay sliding: u_T should be non-zero")

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield_camclay, device)


def test_flow_rule_camclay_normal_clamping(test, device):
    """Large normal b pushes past cam-clay yield pressure bounds."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([-500.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=0.0, tensile_ratio=0.1)

    u = _run_camclay(D, b, r, yp, device)

    # r_N should be clamped
    r_out = (u - b) / D
    ps = np.sqrt(3.0 / 2.0)
    pmax = 100.0 * ps
    test.assertLessEqual(r_out[0], pmax + 1e-2, "Cam-clay: r_N should be clamped to pmax")


def test_flow_rule_camclay_anisotropic(test, device):
    """Non-uniform D to test anisotropic cam-clay response."""
    D = np.array([2.0, 0.5, 1.0, 3.0, 0.1, 1.5], dtype=np.float32)
    b = np.array([0.0, 100.0, -50.0, 30.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=0.0, tensile_ratio=0.1)

    u = _run_camclay(D, b, r, yp, device)

    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield_camclay, device)


def test_flow_rule_camclay_zero_yield(test, device):
    """yield_pressure = 0, cam-clay should return b (fully plastic)."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([10.0, 5.0, -3.0, 1.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.0, yield_pressure=0.0, yield_stress=0.0)

    u = _run_camclay(D, b, r, yp, device)

    # With zero yield, u ~ b
    np.testing.assert_allclose(u, b, atol=1e-3, err_msg="Cam-clay zero yield: u should equal b")


def test_flow_rule_camclay_dilatancy_orthogonal(test, device):
    """Cam-clay with dilatancy=1.0 (associated): stress displacement normal to yield surface."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=1000.0, yield_stress=0.0, dilatancy=1.0, tensile_ratio=0.0)

    u = _run_camclay(D, b, r, yp, device)
    r_proj = (u - b) / D

    # For dilatancy!=0, r_0 = r_guess - (D*r_guess + b) / max(D); with r_guess=0: r_0 = -b/max(D)
    r_0 = -b / np.max(D)
    delta_r = r_proj - r_0

    grad_f = _yield_surface_normal_camclay(r_proj, yp)
    check_yield_normal_alignment(test, delta_r, grad_f)

    # For isotropic D, u equals delta_r so u is also aligned with the normal
    check_yield_normal_alignment(test, u, grad_f)
    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield_camclay, device)


def test_flow_rule_camclay_dilatancy_orthogonal_aniso(test, device):
    """Cam-clay with dilatancy=1.0 (associated), anisotropic D: stress displacement normal to yield surface."""
    D = np.array([2.0, 0.5, 1.0, 3.0, 0.1, 1.5], dtype=np.float32)
    b = np.array([0.0, 100.0, -50.0, 30.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    yp = _make_yield_params(friction=0.5, yield_pressure=1000.0, yield_stress=0.0, dilatancy=1.0, tensile_ratio=0.0)

    u = _run_camclay(D, b, r, yp, device)
    r_proj = (u - b) / D

    # For dilatancy!=0, r_0 = r_guess - (D*r_guess + b) / max(D); with r_guess=0: r_0 = -b/max(D)
    r_0 = -b / np.max(D)
    delta_r = r_proj - r_0

    grad_f = _yield_surface_normal_camclay(r_proj, yp)
    check_yield_normal_alignment(test, delta_r, grad_f)
    # u_T / r_T alignment doesn't hold for cam-clay associated flow with anisotropic D
    check_flow_rule_invariants(test, u, D, b, yp, _eval_shear_yield_camclay, device, check_alignment=False)


# ---------------------------------------------------------------------------
# Test scenario — solve_flow_rule_viscous dispatcher (viscosity)
# ---------------------------------------------------------------------------


def test_flow_rule_dispatch_viscosity(test, device):
    """Verify that high viscosity attenuates velocity magnitude."""
    D = np.full(6, 1.0, dtype=np.float32)
    b = np.array([0.0, 200.0, 200.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    volume = 1.0

    yp_no_visc = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0, viscosity=0.0)
    yp_visc = _make_yield_params(friction=0.5, yield_pressure=100.0, yield_stress=10.0, viscosity=10.0)

    u_no_visc = _run_dispatch(D, b, r, yp_no_visc, volume, device)
    u_visc = _run_dispatch(D, b, r, yp_visc, volume, device)

    # Viscosity should reduce velocity magnitude
    test.assertLess(
        np.linalg.norm(u_visc),
        np.linalg.norm(u_no_visc),
        "Viscosity should attenuate velocity",
    )


# ---------------------------------------------------------------------------
# Bipotential residual (adapted from bench_flow_rule.py eval_residual)
# ---------------------------------------------------------------------------


@wp.func
def eval_flow_rule_residual(
    D: vec6,
    b: vec6,
    yield_params: YieldParamVec,
    u: vec6,
):
    """Evaluate the flow-rule residual for a given velocity u.

    The residual measures how well the solution satisfies:
    1. Stress on yield surface (r_proj_err)
    2. Flow rule constraint on normal velocity (u_proj_err)
    3. Bipotential condition: dot(u, r) + phi*(u) = 0 (slack_err)

    Returns the sum of all residual components.
    """
    r = wp.cw_div(u - b, D)

    dilatancy = get_dilatancy(yield_params)
    ys, dys, pmin_local, pmax_local = shear_yield_stress(yield_params, r[0])

    r_T = r
    r_T[0] = 0.0
    u_T = u
    u_T[0] = 0.0

    rT_n = wp.length(r_T)
    uT_n = wp.length(u_T)

    # 1. Stress on yield surface
    rproj_err = wp.max(0.0, rT_n - ys) + wp.abs(r[0] - wp.clamp(r[0], pmin_local, pmax_local))

    # 2. Flow rule constraint on normal velocity
    uproj_err = wp.max(0.0, wp.sign(dys) * (dilatancy * dys * uT_n - u[0]))

    # 3. Bipotential condition: dot(u, r) + support_function(u) = 0
    ys0 = wp.max(0.0, yield_params[2])
    pmin, pmax = normal_yield_bounds(yield_params)
    mu = wp.where(pmax > 0.0, wp.max(0.0, yield_params[3] / pmax), 0.0)

    # Support function of the yield surface evaluated at u
    u_conj_min = -pmin * u[0] + 0.5 * pmax * wp.max(0.0, dilatancy * mu * uT_n - u[0])
    u_conj_max = -pmax * u[0] + 0.5 * pmax * wp.max(0.0, dilatancy * mu * uT_n + u[0])
    u_conj = wp.max(u_conj_min, u_conj_max) + ys0 * uT_n

    slack_err = wp.abs(wp.dot(u, r) + u_conj + (1.0 - dilatancy) * (ys - ys0) * uT_n)

    return rproj_err + uproj_err + slack_err


@wp.kernel
def test_random_flow_rule_impl_kernel(errors: wp.array[float]):
    tid = wp.tid()

    rng = wp.rand_init(42, tid)
    b = 100.0 * vec6(
        wp.randf(rng) - 0.5,
        wp.randf(rng) - 0.5,
        wp.randf(rng) - 0.5,
        wp.randf(rng) - 0.5,
        wp.randf(rng) - 0.5,
        wp.randf(rng) - 0.5,
    )

    D = 100.0 * vec6(
        0.0001 + wp.randf(rng),
        0.0001 + wp.randf(rng),
        0.0001 + wp.randf(rng),
        0.0001 + wp.randf(rng),
        0.0001 + wp.randf(rng),
        0.0001 + wp.randf(rng),
    )

    dilatancy = wp.randf(rng)
    mu = wp.randf(rng)
    ys = wp.randf(rng)
    yp = 1.0 + 1.0e6 * wp.randf(rng)
    tyr = 0.00001 * wp.randf(rng)

    yield_params = YieldParamVec.from_values(mu, yp, tyr, ys, dilatancy, 0.0)

    r = vec6(0.0)
    u = solve_flow_rule(D, b, r, yield_params, 1.0)

    errors[tid] = eval_flow_rule_residual(D, b, yield_params, u)


def test_flow_rule_impl_random(test, device):
    """Bipotential residual check over random inputs for solve_flow_rule."""
    n = 4096
    errors = wp.zeros(n, dtype=float, device=device)
    wp.launch(test_random_flow_rule_impl_kernel, dim=n, inputs=[], outputs=[errors], device=device)

    err_np = errors.numpy()
    max_err = float(np.max(err_np))
    test.assertLess(max_err, 1.0, f"impl random: max residual {max_err:.6f} too large")


# ---------------------------------------------------------------------------
# Test class and registration
# ---------------------------------------------------------------------------


class TestSolveFlowRule(unittest.TestCase):
    pass


# solve_flow_rule tests
add_function_test(TestSolveFlowRule, "test_flow_rule_impl_elastic", test_flow_rule_impl_elastic, devices=devices)
add_function_test(TestSolveFlowRule, "test_flow_rule_impl_sliding", test_flow_rule_impl_sliding, devices=devices)
add_function_test(
    TestSolveFlowRule, "test_flow_rule_impl_normal_clamping", test_flow_rule_impl_normal_clamping, devices=devices
)
add_function_test(
    TestSolveFlowRule, "test_flow_rule_impl_anisotropic", test_flow_rule_impl_anisotropic, devices=devices
)
add_function_test(TestSolveFlowRule, "test_flow_rule_impl_zero_yield", test_flow_rule_impl_zero_yield, devices=devices)
add_function_test(TestSolveFlowRule, "test_flow_rule_impl_dilatancy", test_flow_rule_impl_dilatancy, devices=devices)
add_function_test(
    TestSolveFlowRule,
    "test_flow_rule_impl_dilatancy_orthogonal",
    test_flow_rule_impl_dilatancy_orthogonal,
    devices=devices,
)
add_function_test(
    TestSolveFlowRule,
    "test_flow_rule_impl_dilatancy_orthogonal_aniso",
    test_flow_rule_impl_dilatancy_orthogonal_aniso,
    devices=devices,
)

# solve_flow_rule_camclay tests
add_function_test(TestSolveFlowRule, "test_flow_rule_camclay_elastic", test_flow_rule_camclay_elastic, devices=devices)
add_function_test(TestSolveFlowRule, "test_flow_rule_camclay_sliding", test_flow_rule_camclay_sliding, devices=devices)
add_function_test(
    TestSolveFlowRule,
    "test_flow_rule_camclay_normal_clamping",
    test_flow_rule_camclay_normal_clamping,
    devices=devices,
)
add_function_test(
    TestSolveFlowRule, "test_flow_rule_camclay_anisotropic", test_flow_rule_camclay_anisotropic, devices=devices
)
add_function_test(
    TestSolveFlowRule, "test_flow_rule_camclay_zero_yield", test_flow_rule_camclay_zero_yield, devices=devices
)
add_function_test(
    TestSolveFlowRule,
    "test_flow_rule_camclay_dilatancy_orthogonal",
    test_flow_rule_camclay_dilatancy_orthogonal,
    devices=devices,
)
add_function_test(
    TestSolveFlowRule,
    "test_flow_rule_camclay_dilatancy_orthogonal_aniso",
    test_flow_rule_camclay_dilatancy_orthogonal_aniso,
    devices=devices,
)

# Dispatcher viscosity test
add_function_test(
    TestSolveFlowRule, "test_flow_rule_dispatch_viscosity", test_flow_rule_dispatch_viscosity, devices=devices
)

# Random bipotential residual tests
add_function_test(TestSolveFlowRule, "test_flow_rule_impl_random", test_flow_rule_impl_random, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
