# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs.kernels import (
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
)
from newton.solvers import SolverFeatherPGS


def _allocated_joint_velocity_slots(qd: float, *, fraction: float, qdot_max: float = 1.0):
    device = "cpu"
    velocity_limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([qdot_max], dtype=wp.float32, device=device),
            wp.array([qd], dtype=wp.float32, device=device),
            fraction,
            wp.array([-1], dtype=wp.int32, device=device),  # drive_slot (unused: skip_driven=0)
            0,  # skip_driven
            wp.array([0], dtype=wp.int32, device=device),
            8,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots(qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0):
    device = "cpu"
    rigid_velocity_limit_slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    rigid_velocity_limit_sign = wp.zeros((12,), dtype=wp.float32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([lin_limit], dtype=wp.float32, device=device),
            wp.array([ang_limit], dtype=wp.float32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[rigid_velocity_limit_slot, rigid_velocity_limit_sign, mf_slot_counter],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


class TestFeatherPGSVelocityLimitActivationFraction(unittest.TestCase):
    def test_solver_rejects_invalid_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "velocity_limit_activation_fraction"):
                    SolverFeatherPGS(model, velocity_limit_activation_fraction=value)

    def test_solver_accepts_and_defaults_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        self.assertEqual(SolverFeatherPGS(model).velocity_limit_activation_fraction, 0.0)
        self.assertEqual(
            SolverFeatherPGS(model, velocity_limit_activation_fraction=0.7).velocity_limit_activation_fraction,
            0.7,
        )

    def test_fraction_zero_preserves_always_allocate_joint_rows(self):
        # fraction == 0.0 must match the historical behavior exactly: both
        # sides of every finitely limited DOF allocate, in the same order,
        # regardless of the current joint velocity.
        for qd in (0.0, 0.5, -2.0):
            with self.subTest(qd=qd):
                slots, signs, count = _allocated_joint_velocity_slots(qd, fraction=0.0)
                self.assertEqual(slots, [0, 1])
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_gates_static_joint_dof_to_zero_rows(self):
        slots, signs, count = _allocated_joint_velocity_slots(0.0, fraction=0.9)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(signs, [0.0, 0.0])
        self.assertEqual(count, 0)

        slots, signs, count = _allocated_joint_velocity_slots(0.85, fraction=0.9)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(count, 0)

    def test_joint_dof_past_threshold_allocates_rows_same_step(self):
        # The gate reads the same-step joint_qd input, so a DOF driven past
        # the threshold gets its row pair allocated in that very step.
        for qd in (0.95, -0.95, 1.5):
            with self.subTest(qd=qd):
                slots, signs, count = _allocated_joint_velocity_slots(qd, fraction=0.9)
                self.assertEqual(slots, [0, 1])
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_zero_preserves_always_allocate_rigid_rows(self):
        slots, signs, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.0)
        self.assertEqual(slots, list(range(12)))
        self.assertEqual(signs, [1.0, -1.0] * 6)
        self.assertEqual(count, 12)

    def test_fraction_gates_static_rigid_body_to_zero_rows(self):
        slots, signs, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.9)
        self.assertEqual(slots, [-1] * 12)
        self.assertEqual(signs, [0.0] * 12)
        self.assertEqual(count, 0)

    def test_rigid_axis_past_threshold_allocates_rows_same_step(self):
        # Only the linear-z axis (axis 2 -> slot indices 4, 5) exceeds the
        # 90% activation threshold; its pair allocates this step.
        slots, signs, count = _allocated_rigid_velocity_slots([0.0, 0.0, 0.95, 0.0, 0.0, 0.0], fraction=0.9)
        expected_slots = [-1] * 12
        expected_slots[4] = 0
        expected_slots[5] = 1
        expected_signs = [0.0] * 12
        expected_signs[4] = 1.0
        expected_signs[5] = -1.0
        self.assertEqual(slots, expected_slots)
        self.assertEqual(signs, expected_signs)
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
