# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint and free-body velocity-limit rows of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs.kernels import (
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

QDOT_MAX = 0.5
# Slack on the limit for an under-converged sweep; the final pass leaves the DOF at the limit.
LIMIT_TOL = 1.05


def _allocated_joint_velocity_slots(device, qd: float, *, fraction: float, qdot_max: float = 1.0):
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
            wp.array([0], dtype=wp.int32, device=device),
            8,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots(device, qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0):
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
        ],
        outputs=[rigid_velocity_limit_slot, rigid_velocity_limit_sign, mf_slot_counter],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip, without gravity or friction."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    # Arm spanning x in [0, 0.8] at z = 0.5, hinged at its left end about y.
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_arm])
    # Heavy box, bottom face slightly penetrating the arm top near the tip.
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    return builder.finalize(device=device)


def _max_arm_speed_over_impact(device, *, enable_joint_velocity_limits: bool) -> float:
    model = _build_arm_and_box_model(device)
    n = model.joint_dof_count
    target_ke = np.zeros(n, dtype=np.float32)
    target_kd = np.zeros(n, dtype=np.float32)
    vel_limit = np.full(n, np.inf, dtype=np.float32)
    target_ke[0] = 200.0
    target_kd[0] = 5.0
    vel_limit[0] = QDOT_MAX
    model.joint_target_ke.assign(target_ke)
    model.joint_target_kd.assign(target_kd)
    model.joint_velocity_limit.assign(vel_limit)
    # Deliberately under-converged: the velocity limit must still have the last word.
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        pgs_iterations=8,
        dense_max_constraints=32,
        mf_max_constraints=32,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0  # box linear z: slam into the arm tip
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    max_arm_speed = 0.0
    for _ in range(20):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
    return max_arm_speed


def test_dedicated_rows_hold_limit_through_impact(test, device):
    """Hold a driven joint's velocity limit through an impact that would exceed it several times over."""
    unlimited = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=False)
    limited = _max_arm_speed_over_impact(device, enable_joint_velocity_limits=True)
    test.assertGreater(unlimited, 2.0 * QDOT_MAX, "the impact must exceed the limit when it is not enforced")
    test.assertLessEqual(limited, QDOT_MAX * LIMIT_TOL)


def test_fraction_zero_allocates_every_joint_row(test, device):
    """Allocate both rows of every limited DOF regardless of its velocity when the fraction is zero."""
    for qd in (0.0, 0.5, -2.0):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots(device, qd, fraction=0.0)
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_gates_static_joint_dof_to_zero_rows(test, device):
    """Allocate no rows for a DOF below the activation fraction of its limit."""
    slots, signs, count = _allocated_joint_velocity_slots(device, 0.0, fraction=0.9)
    test.assertEqual(slots, [-1, -1])
    test.assertEqual(signs, [0.0, 0.0])
    test.assertEqual(count, 0)
    slots, signs, count = _allocated_joint_velocity_slots(device, 0.85, fraction=0.9)
    test.assertEqual(slots, [-1, -1])
    test.assertEqual(count, 0)


def test_joint_dof_past_threshold_allocates_rows_same_step(test, device):
    """Allocate the row pair in the same step a DOF crosses the activation fraction."""
    for qd in (0.95, -0.95, 1.5):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots(device, qd, fraction=0.9)
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_zero_allocates_every_rigid_row(test, device):
    """Allocate all twelve free-body velocity-limit rows when the fraction is zero."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0] * 6, fraction=0.0)
    test.assertEqual(slots, list(range(12)))
    test.assertEqual(signs, [1.0, -1.0] * 6)
    test.assertEqual(count, 12)


def test_fraction_gates_static_rigid_body_to_zero_rows(test, device):
    """Allocate no free-body rows for a body below the activation fraction."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0] * 6, fraction=0.9)
    test.assertEqual(slots, [-1] * 12)
    test.assertEqual(signs, [0.0] * 12)
    test.assertEqual(count, 0)


def test_rigid_axis_past_threshold_allocates_rows_same_step(test, device):
    """Allocate only the row pair of the free-body axis past the activation fraction."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0, 0.0, 0.95, 0.0, 0.0, 0.0], fraction=0.9)
    expected_slots = [-1] * 12
    expected_slots[4] = 0
    expected_slots[5] = 1
    expected_signs = [0.0] * 12
    expected_signs[4] = 1.0
    expected_signs[5] = -1.0
    test.assertEqual(slots, expected_slots)
    test.assertEqual(signs, expected_signs)
    test.assertEqual(count, 2)


def test_free_body_velocity_limits_hold(test, device, pgs_mode="matrix_free"):
    """Hold a free body's authored linear and angular speed limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverFeatherPGS.register_custom_attributes(builder)
    body = builder.add_body(
        custom_attributes={"rigid_body_max_linear_velocity": 0.5, "rigid_body_max_angular_velocity": 1.0}
    )
    builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    qd[3:6] = (5.0, 0.0, 0.0)
    state_0.joint_qd.assign(qd)
    for _ in range(60):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    final = state_0.joint_qd.numpy()
    test.assertLessEqual(float(np.max(np.abs(final[0:3]))), 0.5 * LIMIT_TOL)
    test.assertLessEqual(float(np.max(np.abs(final[3:6]))), 1.0 * LIMIT_TOL)


class TestFeatherPGSVelocityLimitActivationFraction(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_dedicated_rows_hold_limit_through_impact", test_dedicated_rows_hold_limit_through_impact),
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_gates_static_joint_dof_to_zero_rows", test_fraction_gates_static_joint_dof_to_zero_rows),
    ("test_joint_dof_past_threshold_allocates_rows_same_step", test_joint_dof_past_threshold_allocates_rows_same_step),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_fraction_gates_static_rigid_body_to_zero_rows", test_fraction_gates_static_rigid_body_to_zero_rows),
    (
        "test_rigid_axis_past_threshold_allocates_rows_same_step",
        test_rigid_axis_past_threshold_allocates_rows_same_step,
    ),
    ("test_free_body_velocity_limits_hold", test_free_body_velocity_limits_hold),
):
    add_function_test(TestFeatherPGSVelocityLimitActivationFraction, _name, _func, devices=devices)
# The allocators are mode-independent kernels; free-body velocity limits are also rows in split mode.
for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_gates_static_joint_dof_to_zero_rows", test_fraction_gates_static_joint_dof_to_zero_rows),
    ("test_joint_dof_past_threshold_allocates_rows_same_step", test_joint_dof_past_threshold_allocates_rows_same_step),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_fraction_gates_static_rigid_body_to_zero_rows", test_fraction_gates_static_rigid_body_to_zero_rows),
    (
        "test_rigid_axis_past_threshold_allocates_rows_same_step",
        test_rigid_axis_past_threshold_allocates_rows_same_step,
    ),
):
    add_function_test(
        TestFeatherPGSVelocityLimitActivationFraction, _name, _func, devices=[d for d in get_test_devices() if d.is_cpu]
    )
add_function_test(
    TestFeatherPGSVelocityLimitActivationFraction,
    "test_free_body_velocity_limits_hold_split",
    test_free_body_velocity_limits_hold,
    devices=get_test_devices(),
    pgs_mode="split",
)


if __name__ == "__main__":
    unittest.main()
