# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Coulomb friction updates of the SolverFeatherPGS matrix-free solve (``friction_mode``)."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

MODES = ("current", "bisection", "bisection_desaxce", "coulomb_newton")
DT = 1.0 / 240.0
G = 9.81


def _box_on_slope(device, slope_angle, mu, articulated):
    """A box on the ground under gravity tilted by ``slope_angle``, free or on three prismatic joints."""
    builder = newton.ModelBuilder(gravity=(G * np.sin(slope_angle), 0.0, -G * np.cos(slope_angle)))
    builder.default_shape_cfg.mu = mu
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
    if articulated:
        link = builder.add_link(xform=xform)
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        axis = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            link,
            linear_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
            parent_xform=xform,
        )
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=xform)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _slide(device, mode, slope_angle, mu, articulated, steps=120):
    """Return the box's final body velocity and height."""
    model = _box_on_slope(device, slope_angle, mu, articulated)
    solver = SolverFeatherPGS(model, pgs_iterations=32, friction_mode=mode, friction_anchor_beta=0.0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0.body_qd.numpy()[0], float(state_0.body_q.numpy()[0, 2])


def test_every_mode_holds_inside_the_cone(test, device):
    """Hold a free and an articulated box at rest inside the friction cone in every mode."""
    for articulated in (False, True):
        for mode in MODES:
            with test.subTest(articulated=articulated, mode=mode):
                velocity, height = _slide(device, mode, slope_angle=0.3, mu=0.5, articulated=articulated)
                test.assertLess(float(np.abs(velocity).max()), 1.0e-3)
                test.assertAlmostEqual(height, 0.1, delta=1.0e-3)


def test_sliding_follows_each_modes_coulomb_update(test, device):
    """Slide outside the friction cone: exact for current and coulomb_newton, approximate for bisection.

    ``g (sin a - mu cos a)`` is the exact sliding acceleration. The bisection modes solve the normal
    impulse to a fixed tolerance per sweep and slide slightly faster on free bodies; the de Saxce bias
    also lifts a sliding free body off the ground.
    """
    slope_angle, mu, steps = 0.6, 0.3, 120
    expected = G * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    results = {}
    for articulated in (False, True):
        for mode in MODES:
            results[(articulated, mode)] = _slide(device, mode, slope_angle, mu, articulated, steps)
    for articulated in (False, True):
        for mode in ("current", "coulomb_newton"):
            with test.subTest(articulated=articulated, mode=mode):
                velocity, height = results[(articulated, mode)]
                test.assertAlmostEqual(float(velocity[0]), expected, delta=0.005 * expected)
                test.assertAlmostEqual(height, 0.1, delta=1.0e-3)
    for mode in ("bisection", "bisection_desaxce"):
        with test.subTest(articulated=True, mode=mode):
            velocity, height = results[(True, mode)]
            test.assertAlmostEqual(float(velocity[0]), expected, delta=0.005 * expected)
        with test.subTest(articulated=False, mode=mode):
            velocity, height = results[(False, mode)]
            test.assertGreater(float(velocity[0]), expected)
            test.assertAlmostEqual(float(velocity[0]), expected, delta=0.08 * expected)
    # The de Saxce bias raises the normal target velocity of a sliding free-body contact.
    test.assertAlmostEqual(results[(False, "bisection")][1], 0.1, delta=1.0e-4)
    test.assertGreater(results[(False, "bisection_desaxce")][1], 0.1 + 1.0e-3)
    test.assertGreater(float(results[(False, "bisection_desaxce")][0][2]), 1.0e-3)


def test_modes_select_distinct_kernels(test, device):
    """Build one matrix-free kernel per mode and keep the default kernel's name."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=False)
    names = set()
    for mode in MODES:
        solver = SolverFeatherPGS(model, friction_mode=mode, friction_anchor_beta=0.0)
        name = solver._pgs_solve_mf_gs_kernel.key
        test.assertEqual(mode == "current", not name.endswith(mode), name)
        names.add(name)
    test.assertEqual(len(names), len(MODES))


def test_unsupported_combinations_raise(test, device):
    """Reject the options the non-default modes do not implement."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=True)
    with test.assertRaisesRegex(ValueError, "friction_mode"):
        SolverFeatherPGS(model, friction_mode="bad")
    for mode in MODES[1:]:
        with test.subTest(mode=mode):
            # An omitted friction_anchor_beta selects point friction, with a warning.
            with test.assertWarnsRegex(UserWarning, "point friction"):
                solver = SolverFeatherPGS(model, friction_mode=mode)
            test.assertEqual(solver.friction_anchor_beta, 0.0)
            with test.assertRaisesRegex(ValueError, "friction_mode='current'"):
                SolverFeatherPGS(model, friction_mode=mode, friction_anchor_beta=0.2)
            with test.assertRaisesRegex(ValueError, "friction_mode='current'"):
                SolverFeatherPGS(model, friction_mode=mode, friction_anchor_beta=0.0, contact_torsion_radius=0.01)
            with test.assertRaisesRegex(ValueError, "friction_mode"):
                SolverFeatherPGS(model, friction_mode=mode, friction_anchor_beta=0.0, contact_compliance=True)
            for response in ("propagation", "propagation-fused"):
                with test.assertRaisesRegex(NotImplementedError, "friction_mode"):
                    SolverFeatherPGS(model, friction_mode=mode, articulated_contact_response=response)


def test_sparse_factors_require_the_current_mode(test, device):
    """Select sparse mass factors only for the default friction update."""
    builder = newton.ModelBuilder()
    root = builder.add_link()
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    for side in (-1.0, 1.0):
        leg = builder.add_link()
        builder.add_shape_box(leg, hx=0.05, hy=0.05, hz=0.05)
        joints.append(
            builder.add_joint_revolute(
                root, leg, axis=newton.Axis.Y, parent_xform=wp.transform(wp.vec3(side * 0.2, 0, 0), wp.quat_identity())
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    test.assertIsNotNone(SolverFeatherPGS(model, friction_anchor_beta=0.0)._sparse_mass_matrix_size)
    for mode in MODES[1:]:
        with test.subTest(mode=mode):
            solver = SolverFeatherPGS(model, friction_mode=mode, friction_anchor_beta=0.0)
            test.assertIsNone(solver._sparse_mass_matrix_size)


def test_split_rejects_other_modes(test, device):
    """Reject the non-default modes in the split solve; the default omitted friction stays silent."""
    model = _box_on_slope(device, 0.3, 0.5, articulated=True)
    for mode in MODES[1:]:
        with test.subTest(mode=mode), warnings.catch_warnings():
            warnings.simplefilter("error")
            with test.assertRaisesRegex(NotImplementedError, f"friction_mode='{mode}'.*pgs_mode='matrix_free'"):
                SolverFeatherPGS(model, pgs_mode="split", friction_mode=mode)
    test.assertEqual(SolverFeatherPGS(model, pgs_mode="split").friction_mode, "current")


class TestFeatherPGSFrictionMode(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (
    test_every_mode_holds_inside_the_cone,
    test_sliding_follows_each_modes_coulomb_update,
    test_modes_select_distinct_kernels,
    test_unsupported_combinations_raise,
    test_sparse_factors_require_the_current_mode,
):
    add_function_test(TestFeatherPGSFrictionMode, _fn.__name__, _fn, devices=cuda_devices)
add_function_test(
    TestFeatherPGSFrictionMode,
    "test_split_rejects_other_modes",
    test_split_rejects_other_modes,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
