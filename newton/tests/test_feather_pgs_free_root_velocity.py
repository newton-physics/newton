# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for FeatherPGS floating-root velocity handling."""

import unittest

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import (
    convert_root_free_qd_local_to_world,
    convert_root_free_qd_world_to_local,
    integrate_generalized_joints,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherPGSFreeRootVelocity(unittest.TestCase):
    pass


def run_free_root_velocity_roundtrip(test: TestFeatherPGSFreeRootVelocity, device):
    """Floating-root qd should round-trip between public CoM and internal origin conventions."""
    articulation_root_is_free = wp.array(np.array([1, 0], dtype=np.int32), dtype=int, device=device)
    articulation_root_dof_start = wp.array(np.array([0, 6], dtype=np.int32), dtype=int, device=device)
    articulation_root_com_offset = wp.array(
        np.array([[0.2, -0.3, 0.1], [0.0, 0.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device=device,
    )

    qd_public_np = np.array(
        [
            1.5,
            -2.0,
            0.25,
            0.4,
            -0.6,
            0.8,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        dtype=np.float32,
    )
    qd = wp.array(qd_public_np, dtype=float, device=device)

    wp.launch(
        convert_root_free_qd_world_to_local,
        dim=2,
        inputs=[articulation_root_is_free, articulation_root_dof_start, articulation_root_com_offset],
        outputs=[qd],
        device=device,
    )

    qd_local_np = qd.numpy()
    omega = qd_public_np[3:6]
    com_offset = np.array([0.2, -0.3, 0.1], dtype=np.float32)
    expected_local_linear = qd_public_np[:3] - np.cross(omega, com_offset)
    np.testing.assert_allclose(qd_local_np[:3], expected_local_linear, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(qd_local_np[3:6], qd_public_np[3:6], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(qd_local_np[6:], qd_public_np[6:], rtol=1e-6, atol=1e-6)

    wp.launch(
        convert_root_free_qd_local_to_world,
        dim=2,
        inputs=[articulation_root_is_free, articulation_root_dof_start, articulation_root_com_offset],
        outputs=[qd],
        device=device,
    )

    np.testing.assert_allclose(qd.numpy(), qd_public_np, rtol=1e-6, atol=1e-6)


def run_free_root_integration_uses_origin_velocity(test: TestFeatherPGSFreeRootVelocity, device):
    """Free-joint position integration should convert CoM velocity back to origin velocity."""
    joint_type = wp.array(np.array([int(JointType.FREE)], dtype=np.int32), dtype=int, device=device)
    joint_child = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_q_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_qd_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[3, 3]], dtype=np.int32), dtype=int, device=device)
    body_com = wp.array(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), dtype=wp.vec3, device=device)

    joint_q = wp.array(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=float, device=device)
    joint_qd = wp.array(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32), dtype=float, device=device)
    joint_qdd = wp.zeros(6, dtype=float, device=device)
    joint_q_new = wp.zeros(7, dtype=float, device=device)
    joint_qd_new = wp.zeros(6, dtype=float, device=device)

    wp.launch(
        integrate_generalized_joints,
        dim=1,
        inputs=[
            joint_type,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_dof_dim,
            body_com,
            joint_q,
            joint_qd,
            joint_qdd,
            0.1,
        ],
        outputs=[joint_q_new, joint_qd_new],
        device=device,
    )

    joint_q_new_np = joint_q_new.numpy()
    joint_qd_new_np = joint_qd_new.numpy()

    # qd stores CoM velocity. With r_com = (0, 1, 0) and w = (0, 0, 2),
    # the origin velocity is v_com - w x r_com = (3, 0, 0).
    np.testing.assert_allclose(joint_q_new_np[:3], np.array([10.3, 0.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(joint_qd_new_np[:3], np.array([1.0, 0.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(joint_qd_new_np[3:], np.array([0.0, 0.0, 2.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSFreeRootVelocity,
        f"test_free_root_velocity_roundtrip_{device}",
        run_free_root_velocity_roundtrip,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeRootVelocity,
        f"test_free_root_integration_uses_origin_velocity_{device}",
        run_free_root_integration_uses_origin_velocity,
        devices=[device],
    )
