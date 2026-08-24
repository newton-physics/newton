# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Friction tests for the penalty contact kernel used by SemiImplicit and Featherstone.

The kernel evaluates min(kf * |vt|, mu * |fn|) in the direction of vt, with the
direction smoothed by wp.norm_pseudo_huber. That costs a factor
|vt| / sqrt(delta^2 + |vt|^2), so friction_smoothing has to stay small.

The kernel is launched on hand-built arrays instead of going through a solver,
so a failure here is a failure of the force law and nothing else.
"""

import inspect
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.semi_implicit.kernels_contact import eval_body_contact
from newton.tests.unittest_utils import add_function_test, get_test_devices

PENETRATION = 1.0e-5  # m
CONTACT_KE = 2.0e4  # N/m
CONTACT_KD = 3.0  # N·s/m
CONTACT_KF = 1.0e3  # N·s/m, the ShapeConfig default
CONTACT_MU = 0.9
CONTACT_OFFSET_X = 0.25  # m, lever arm for the spin case

NORMAL_FORCE = CONTACT_KE * PENETRATION  # 0.2 N, exact while the normal velocity is zero
CONE = CONTACT_MU * NORMAL_FORCE  # 0.18 N
TRANSITION = CONE / CONTACT_KF  # 0.18 mm/s, where the stick branch gives way

SLIP_SPEEDS = (0.5, 0.1, 0.05, 0.02, 0.005, 0.001, 0.0)
SHARP_SMOOTHING = 1.0e-5  # small enough that the fade is not measurable


def solver_default():
    """Return the friction_smoothing default, which both solvers must agree on."""
    solvers = (newton.solvers.SolverSemiImplicit, newton.solvers.SolverFeatherstone)
    defaults = {s.__name__: inspect.signature(s.__init__).parameters["friction_smoothing"].default for s in solvers}
    assert len(set(defaults.values())) == 1, defaults
    return next(iter(defaults.values()))


DEFAULT_SMOOTHING = solver_default()


def expected_friction(slip, smoothing, cone=CONE):
    """Return the friction magnitude the force law prescribes."""
    vs = math.hypot(slip, smoothing)
    return min(CONTACT_KF * vs, cone) * slip / vs


def contact_force(device, v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0), smoothing=None, grad=False):
    """Launch the kernel on one contact and return the force on body 0.

    Returns the force plus the input and output arrays, so the gradient test can
    replay the launch under a tape.
    """

    def arr(values, dtype):
        return wp.array(np.array(values, dtype=np.float32), dtype=dtype, device=device, requires_grad=grad)

    def iarr(values):
        return wp.array(np.array(values, dtype=np.int32), dtype=int, device=device)

    body_qd = arr([[*v, *w]], wp.spatial_vector)  # [linear at com; angular]
    body_f = wp.zeros(1, dtype=wp.spatial_vector, device=device, requires_grad=grad)

    # The contact sits off the origin so a spin about z has a lever arm. The
    # penetration d = dot(n, point0 - point1) stays -PENETRATION either way.
    wp.launch(
        eval_body_contact,
        dim=1,
        device=device,
        outputs=[body_f],
        inputs=[
            arr([wp.transform_identity()], wp.transform),
            body_qd,
            arr([[0.0, 0.0, 0.0]], wp.vec3),
            arr([CONTACT_KE] * 2, float),
            arr([CONTACT_KD] * 2, float),
            arr([CONTACT_KF] * 2, float),
            arr([0.0] * 2, float),  # ka, no adhesion
            arr([CONTACT_MU] * 2, float),
            iarr([0, -1]),  # shape 0 on body 0, shape 1 static
            iarr([1]),  # contact count
            arr([[CONTACT_OFFSET_X, 0.0, -PENETRATION]], wp.vec3),
            arr([[CONTACT_OFFSET_X, 0.0, 0.0]], wp.vec3),
            arr([[0.0, 0.0, -1.0]], wp.vec3),  # normal, A to B
            iarr([0]),
            iarr([1]),
            arr([0.0], float),
            arr([0.0], float),
            None,  # no per-contact stiffness, damping or friction
            None,
            None,
            False,  # force_in_world_frame
            DEFAULT_SMOOTHING if smoothing is None else smoothing,
        ],
    )
    return body_f.numpy()[0][:3].copy(), body_qd, body_f


def sliding_force(device, slip, smoothing):
    """Return the normal and friction magnitudes for a purely tangential slip."""
    force, _, _ = contact_force(device, v=(slip, 0.0, 0.0), smoothing=smoothing)
    return float(force[2]), float(np.linalg.norm(force[:2]))


def test_friction_matches_the_force_law(test, device):
    """Verify friction equals min(kf * |vt|, mu * |fn|) after the smoothing fade."""
    for smoothing in (DEFAULT_SMOOTHING, SHARP_SMOOTHING):
        for slip in SLIP_SPEEDS:
            normal, friction = sliding_force(device, slip, smoothing)
            test.assertAlmostEqual(normal, NORMAL_FORCE, delta=1.0e-6)
            expected = expected_friction(slip, smoothing)
            test.assertAlmostEqual(
                friction,
                expected,
                delta=max(1.0e-4 * CONE, 1.0e-3 * expected),
                msg=f"{friction:.6g} N, expected {expected:.6g} N at {slip * 1e3:g} mm/s, delta={smoothing:g}",
            )


def test_friction_stays_inside_the_cone(test, device):
    """Verify friction never exceeds mu * |fn|, at any slip speed."""
    for smoothing in (DEFAULT_SMOOTHING, SHARP_SMOOTHING, 1.0):
        for slip in SLIP_SPEEDS:
            _, friction = sliding_force(device, slip, smoothing)
            test.assertLessEqual(
                friction,
                CONE * (1.0 + 1.0e-4),
                msg=f"{friction:.4g} N over the {CONE:.4g} N cone at {slip * 1e3:g} mm/s, delta={smoothing:g}",
            )


def test_default_smoothing_reaches_the_cone(test, device):
    """Verify the shipped default still lets friction saturate at realistic slip speeds."""
    for slip, floor in ((0.02, 0.99), (0.001, 0.50)):
        _, friction = sliding_force(device, slip, DEFAULT_SMOOTHING)
        test.assertGreaterEqual(
            friction,
            floor * CONE,
            msg=f"only {friction / CONE:.1%} of the cone at {slip * 1e3:g} mm/s, delta={DEFAULT_SMOOTHING:g}",
        )


def test_friction_is_linear_below_the_transition(test, device):
    """Verify friction is kf * vt while the stick branch holds."""
    for slip in (0.2 * TRANSITION, 0.5 * TRANSITION, 0.9 * TRANSITION):
        _, friction = sliding_force(device, slip, SHARP_SMOOTHING)
        test.assertAlmostEqual(friction, CONTACT_KF * slip, delta=1.0e-3 * CONTACT_KF * slip)


def test_friction_opposes_the_slip(test, device):
    """Verify friction is antiparallel to vt for an off-axis slip."""
    slip = np.array([0.03, -0.04, 0.0])  # 50 mm/s, both tangential axes in play
    force, _, _ = contact_force(device, v=tuple(slip))
    expected = -slip[:2] / np.linalg.norm(slip) * expected_friction(float(np.linalg.norm(slip)), DEFAULT_SMOOTHING)
    np.testing.assert_allclose(force[:2], expected, rtol=1.0e-3, atol=1.0e-6)
    test.assertAlmostEqual(float(force[2]), NORMAL_FORCE, delta=1.0e-6)


def test_damping_widens_the_cone(test, device):
    """Verify the cone follows the damped normal force fn + fd, not ke * penetration."""
    approach = 0.5  # m/s along -z
    force, _, _ = contact_force(device, v=(0.02, 0.0, -approach))

    normal = NORMAL_FORCE + CONTACT_KD * approach
    test.assertAlmostEqual(float(force[2]), normal, delta=1.0e-5)

    cone = CONTACT_MU * normal
    friction = float(np.linalg.norm(force[:2]))
    test.assertAlmostEqual(friction, expected_friction(0.02, DEFAULT_SMOOTHING, cone), delta=1.0e-3 * cone)
    test.assertGreater(friction, CONE)


def test_spin_produces_friction(test, device):
    """Verify slip from an angular velocity at an offset contact is resisted."""
    spin = 0.08  # rad/s about +z, so vt = (0, spin * offset, 0) = 20 mm/s
    force, _, _ = contact_force(device, w=(0.0, 0.0, spin))
    expected = expected_friction(spin * CONTACT_OFFSET_X, DEFAULT_SMOOTHING)
    np.testing.assert_allclose(force[:2], [0.0, -expected], rtol=1.0e-3, atol=1.0e-6)


def test_friction_falls_off_as_the_contact_slows(test, device):
    """Verify friction shrinks towards zero as vt does, instead of growing."""
    for smoothing in (DEFAULT_SMOOTHING, 1.0):
        previous = float("inf")
        for slip in SLIP_SPEEDS:  # descending
            _, friction = sliding_force(device, slip, smoothing)
            test.assertLessEqual(
                friction,
                previous + 1.0e-9,
                msg=f"grew from {previous:.4g} to {friction:.4g} N at {slip * 1e3:g} mm/s, delta={smoothing:g}",
            )
            previous = friction
        test.assertAlmostEqual(previous, 0.0, delta=1.0e-9)


def test_friction_gradient_is_finite_at_zero_slip(test, device):
    """Verify the friction gradient stays finite at and around vt = 0."""
    seed = wp.array(
        np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.spatial_vector, device=device
    )
    ceiling = 10.0 * max(CONTACT_KF, CONE / DEFAULT_SMOOTHING)
    for slip in (0.0, 1.0e-9, SHARP_SMOOTHING, 0.01):
        tape = wp.Tape()
        with tape:
            _, body_qd, body_f = contact_force(device, v=(slip, 0.0, 0.0), grad=True)
        tape.backward(grads={body_f: seed})
        grad = body_qd.grad.numpy()[0]
        test.assertTrue(np.all(np.isfinite(grad)), f"{grad} at {slip:g} m/s")
        test.assertLess(float(np.max(np.abs(grad))), ceiling, f"{grad} at {slip:g} m/s")
        tape.zero()


devices = get_test_devices()


class TestRigidContactFrictionCone(unittest.TestCase):
    pass


for _test in (
    test_friction_matches_the_force_law,
    test_friction_stays_inside_the_cone,
    test_default_smoothing_reaches_the_cone,
    test_friction_is_linear_below_the_transition,
    test_friction_opposes_the_slip,
    test_damping_widens_the_cone,
    test_spin_produces_friction,
    test_friction_falls_off_as_the_contact_slows,
    test_friction_gradient_is_finite_at_zero_slip,
):
    add_function_test(TestRigidContactFrictionCone, _test.__name__, _test, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
