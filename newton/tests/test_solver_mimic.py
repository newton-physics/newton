# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_mimic_model(device, vectorized, *, kinematic_reference=False, small_mismatch=False):
    """Build an inconsistent scalar or vectorized mimic model."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    reference_body = builder.add_link()
    follower_body = builder.add_link()
    builder.add_shape_box(reference_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_shape_box(follower_body, hx=0.1, hy=0.1, hz=0.1)
    if kinematic_reference:
        builder.body_flags[reference_body] = int(newton.BodyFlags.KINEMATIC)

    if vectorized:
        axis = newton.ModelBuilder.JointDofConfig.create_unlimited
        axes = [axis(newton.Axis.X), axis(newton.Axis.Y)]
        reference = builder.add_joint_d6(-1, reference_body, linear_axes=axes)
        follower = builder.add_joint_d6(-1, follower_body, linear_axes=axes)
        initial_q = [0.2, -0.1, -0.18, 0.23] if small_mismatch else [0.2, -0.1, 0.8, 0.5]
    else:
        reference = builder.add_joint_revolute(-1, reference_body, axis=newton.Axis.Z)
        follower = builder.add_joint_prismatic(-1, follower_body, axis=newton.Axis.X)
        initial_q = [0.35, -0.405] if small_mismatch else [0.35, 0.8]

    builder.add_articulation([reference, follower])
    offset = 0.1
    multiplier = -1.5
    builder.set_joint_mimic(follower, reference, (offset, multiplier))
    builder.color()
    model = builder.finalize(device=device)
    model.joint_q.assign(np.asarray(initial_q, dtype=np.float32))
    model.joint_qd.zero_()

    state_in = model.state()
    state_out = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    return model, state_in, state_out, reference, follower, offset, multiplier


def _read_joint_q(model, state, generalized):
    """Return generalized positions from either reduced or maximal solver state."""
    if generalized:
        return state.joint_q.numpy()
    joint_q = wp.empty_like(model.joint_q)
    joint_qd = wp.empty_like(model.joint_qd)
    newton.eval_ik(model, state, joint_q, joint_qd)
    return joint_q.numpy()


def _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier):
    """Assert every follower coordinate satisfies its affine mimic relationship."""
    joint_q_start = model.joint_q_start.numpy()
    reference_slice = slice(joint_q_start[reference], joint_q_start[reference + 1])
    follower_slice = slice(joint_q_start[follower], joint_q_start[follower + 1])
    np.testing.assert_allclose(
        joint_q[follower_slice],
        offset + multiplier * joint_q[reference_slice],
        atol=2.0e-3,
    )


def _test_solver_mimic(test, device, solver_name):
    """Enforce scalar and vectorized mimic relationships with one solver."""
    for vectorized in (False, True):
        with test.subTest(vectorized=vectorized):
            # VBD performs one local mimic projection per solver iteration. Keep
            # its initial pose within the projection's intended convergence basin;
            # large authored inconsistencies should be resolved with eval_mimic()
            # followed by eval_fk() before simulation.
            model, state_in, state_out, reference, follower, offset, multiplier = _build_mimic_model(
                device,
                vectorized,
                small_mismatch=solver_name == "vbd",
            )
            if solver_name == "featherstone":
                solver = newton.solvers.SolverFeatherstone(model)
                generalized = True
            elif solver_name == "semi_implicit":
                if vectorized:
                    solver = newton.solvers.SolverSemiImplicit(
                        model,
                        joint_mimic_ke=1.0e4,
                        joint_mimic_kd=1.0e2,
                    )
                else:
                    solver = newton.solvers.SolverSemiImplicit(
                        model,
                        joint_mimic_ke=1.0e2,
                        joint_mimic_kd=3.0,
                    )
                generalized = False
            elif solver_name == "vbd":
                solver = newton.solvers.SolverVBD(
                    model,
                    iterations=4,
                    rigid_compliant_alm=True,
                )
                generalized = False
            else:
                raise ValueError(f"Unknown solver {solver_name}")

            step_count = 240 if solver_name == "semi_implicit" else 1
            dt = 1.0 / 240.0 if solver_name == "semi_implicit" else 1.0 / 60.0
            for _ in range(step_count):
                solver.step(state_in, state_out, None, None, dt)
                state_in, state_out = state_out, state_in
            joint_q = _read_joint_q(model, state_in, generalized)
            _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier)
            if not generalized and not vectorized:
                reference_q = float(joint_q[model.joint_q_start.numpy()[reference]])
                test.assertNotAlmostEqual(reference_q, 0.35, places=4)


def _test_featherstone_force_transfer(test, device):
    """Transfer a follower force into the Featherstone reference coordinate."""
    model, state_in, state_out, reference, follower, _, _ = _build_mimic_model(device, False)
    control = model.control()
    follower_dof = int(model.joint_qd_start.numpy()[follower])
    reference_dof = int(model.joint_qd_start.numpy()[reference])
    control.joint_f[follower_dof : follower_dof + 1].fill_(1.0)

    solver = newton.solvers.SolverFeatherstone(model)
    solver.step(state_in, state_out, control, None, 1.0 / 60.0)

    reference_velocity = float(state_out.joint_qd.numpy()[reference_dof])
    test.assertGreater(abs(reference_velocity), 1.0e-4)


def test_featherstone_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in Featherstone."""
    _test_solver_mimic(test, device, "featherstone")
    _test_featherstone_force_transfer(test, device)


def test_semi_implicit_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in SemiImplicit."""
    _test_solver_mimic(test, device, "semi_implicit")


def test_semi_implicit_mimic_preserves_kinematic_reference(test, device):
    """Keep a prescribed kinematic reference fixed while its follower responds."""
    model, state_in, state_out, reference, follower, offset, multiplier = _build_mimic_model(
        device,
        False,
        kinematic_reference=True,
    )
    reference_body = int(model.joint_child.numpy()[reference])
    reference_pose = state_in.body_q.numpy()[reference_body].copy()
    reference_velocity = state_in.body_qd.numpy()[reference_body].copy()
    solver = newton.solvers.SolverSemiImplicit(
        model,
        joint_mimic_ke=1.0e3,
        joint_mimic_kd=1.5e2,
    )

    for _ in range(480):
        solver.step(state_in, state_out, None, None, 1.0 / 240.0)
        state_in, state_out = state_out, state_in

    np.testing.assert_allclose(state_in.body_q.numpy()[reference_body], reference_pose, atol=1.0e-7)
    np.testing.assert_allclose(state_in.body_qd.numpy()[reference_body], reference_velocity, atol=1.0e-7)
    joint_q = _read_joint_q(model, state_in, False)
    _assert_mimic_relation(model, joint_q, reference, follower, offset, multiplier)


def test_semi_implicit_mimic_validates_gains(test, device):
    """Reject negative or non-finite global mimic penalty gains."""
    model, *_ = _build_mimic_model(device, False)
    for name in ("joint_mimic_ke", "joint_mimic_kd"):
        for value in (-1.0, float("nan"), float("inf")):
            with test.subTest(name=name, value=value):
                with test.assertRaisesRegex(ValueError, f"{name} must be finite and non-negative"):
                    newton.solvers.SolverSemiImplicit(model, **{name: value})


def test_vbd_mimic(test, device):
    """Enforce scalar and vectorized mimic relationships in VBD."""
    _test_solver_mimic(test, device, "vbd")


class TestSolverMimic(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestSolverMimic, "test_featherstone_mimic", test_featherstone_mimic, devices=devices)
add_function_test(TestSolverMimic, "test_semi_implicit_mimic", test_semi_implicit_mimic, devices=devices)
add_function_test(
    TestSolverMimic,
    "test_semi_implicit_mimic_preserves_kinematic_reference",
    test_semi_implicit_mimic_preserves_kinematic_reference,
    devices=devices,
)
add_function_test(
    TestSolverMimic,
    "test_semi_implicit_mimic_validates_gains",
    test_semi_implicit_mimic_validates_gains,
    devices=devices,
)
add_function_test(TestSolverMimic, "test_vbd_mimic", test_vbd_mimic, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
