# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit (augmented) joint drives of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 120.0
MASS = 2.0
ARMATURE = 0.1
KE = 400.0
KD = 15.0
TARGET_POS = 0.3
TARGET_VEL = 0.2


def _build_slider(device, *, effort_limit=1.0e6, damping=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=MASS, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.X,
        target_ke=KE,
        target_kd=KD,
        target_pos=TARGET_POS,
        target_vel=TARGET_VEL,
        armature=ARMATURE,
        effort_limit=effort_limit,
        damping=damping,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _reference_trajectory(steps, *, effort_limit=1.0e6, damping=0.0):
    """Backward-Euler PD drive: ``(m + a + dt kd + dt^2 ke) qdd = clamp(u0) - c qd``."""
    q = 0.0
    qd = 0.0
    history = []
    effective_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    for _ in range(steps):
        u0 = -(KE * (q - TARGET_POS + DT * qd) + KD * (qd - TARGET_VEL))
        u0 = float(np.clip(u0, -effort_limit, effort_limit))
        qdd = (u0 - damping * qd) / effective_mass
        qd = qd + DT * qdd
        q = q + DT * qd
        history.append((q, qd))
    return np.asarray(history)


def _solver_trajectory(model, steps, pgs_mode):
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        history.append((float(state_0.joint_q.numpy()[0]), float(state_0.joint_qd.numpy()[0])))
    return np.asarray(history)


def test_implicit_drive_matches_backward_euler_reference(test, device, pgs_mode="matrix_free"):
    """Integrate the PD drive implicitly through the augmented mass matrix."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider(device), 120, pgs_mode), _reference_trajectory(120), rtol=1.0e-4, atol=1.0e-5
    )


def test_drive_force_is_clamped_to_the_effort_limit(test, device, pgs_mode="matrix_free"):
    """Clamp the explicit drive force to the joint effort limit before the implicit solve."""
    reference = _reference_trajectory(120, effort_limit=5.0)
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider(device, effort_limit=5.0), 120, pgs_mode), reference, rtol=1.0e-4, atol=1.0e-5
    )
    test.assertGreater(float(np.abs(reference - _reference_trajectory(120)).max()), 1.0e-2, "the clamp must be active")


def test_passive_joint_damping_is_applied(test, device, pgs_mode="matrix_free"):
    """Apply Model.joint_damping as a passive joint force."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider(device, damping=3.0), 120, pgs_mode),
        _reference_trajectory(120, damping=3.0),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
split_devices = get_test_devices()
for _name, _func in (
    ("test_implicit_drive_matches_backward_euler_reference", test_implicit_drive_matches_backward_euler_reference),
    ("test_drive_force_is_clamped_to_the_effort_limit", test_drive_force_is_clamped_to_the_effort_limit),
    ("test_passive_joint_damping_is_applied", test_passive_joint_damping_is_applied),
):
    add_function_test(TestFeatherPGSAugmentedDrives, _name, _func, devices=devices)
    add_function_test(TestFeatherPGSAugmentedDrives, f"{_name}_split", _func, devices=split_devices, pgs_mode="split")


if __name__ == "__main__":
    unittest.main()
