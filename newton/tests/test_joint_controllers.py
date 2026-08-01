# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.utils.import_mjcf import parse_mjcf
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestJointController(unittest.TestCase):
    pass


def test_revolute_controller(
    test: TestJointController,
    device,
    solver_fn,
    pos_target_val,
    vel_target_val,
    expected_pos,
    expected_vel,
    target_ke,
    target_kd,
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    box_mass = 1.0
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    # easy case: identity transform, zero center of mass
    b = builder.add_link(inertia=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))

    # Create a revolute joint
    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=pos_target_val,
        target_vel=vel_target_val,
        armature=0.0,
        # limit_lower=-wp.pi,
        # limit_upper=wp.pi,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=target_ke,
        target_kd=target_kd,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)

    solver = solver_fn(model)

    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    control.joint_target_q = wp.array([pos_target_val], dtype=wp.float32, device=device)
    control.joint_target_qd = wp.array([vel_target_val], dtype=wp.float32, device=device)

    sim_dt = 1.0 / 60.0
    sim_time = 0.0
    for _ in range(100):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, sim_dt)
        state_0, state_1 = state_1, state_0

        sim_time += sim_dt

    if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

    joint_q = state_0.joint_q.numpy()
    joint_qd = state_0.joint_qd.numpy()
    if expected_pos is not None:
        test.assertAlmostEqual(joint_q[0], expected_pos, delta=1e-2)
    if expected_vel is not None:
        test.assertAlmostEqual(joint_qd[0], expected_vel, delta=1e-2)


def test_ball_controller(
    test: TestJointController,
    device,
    solver_fn,
    pos_target_vals,
    vel_target_vals,
    expected_quat,
    expected_vel,
    target_ke,
    target_kd,
):
    """Test ball joint controller with position and velocity targets."""
    # Ball-joint per-axis targets exercise the legacy DOF layout.
    prev_flag = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = False
    try:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
        box_mass = 1.0
        box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        # easy case: identity transform, zero center of mass
        b = builder.add_link(inertia=box_inertia, mass=box_mass)
        builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))

        # Create a ball joint
        j = builder.add_joint_ball(
            parent=-1,
            child=b,
            parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            armature=0.0,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        builder.add_articulation([j])

        test.assertEqual(builder.joint_count, 1)
        test.assertEqual(builder.joint_dof_count, 3)
        test.assertEqual(builder.joint_coord_count, 4)
        test.assertEqual(builder.joint_type[0], newton.JointType.BALL)
        test.assertEqual(builder.joint_parent[0], -1)
        test.assertEqual(builder.joint_child[0], b)
        test.assertEqual(builder.joint_armature[0], 0.0)
        test.assertEqual(builder.joint_friction[0], 0.0)

        # Set controller gains for the ball joint axes
        # Ball joints have 3 axes (X, Y, Z) that are added to joint_target_ke/kd arrays
        qd_start = builder.joint_qd_start[j]
        for i in range(3):  # 3 angular axes
            builder.joint_target_ke[qd_start + i] = target_ke
            builder.joint_target_kd[qd_start + i] = target_kd

        model = builder.finalize(device=device)

        solver = solver_fn(model)

        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        control = model.control()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            control.joint_target_pos = wp.array(pos_target_vals, dtype=wp.float32, device=device)
            control.joint_target_vel = wp.array(vel_target_vals, dtype=wp.float32, device=device)

        sim_dt = 1.0 / 60.0
        sim_time = 0.0
        for _ in range(100):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

            sim_time += sim_dt

        if not isinstance(solver, newton.solvers.SolverMuJoCo | newton.solvers.SolverFeatherstone):
            newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)

        joint_q = state_0.joint_q.numpy()
        joint_qd = state_0.joint_qd.numpy()

        # Ball joint has 4 position coordinates (quaternion) and 3 velocity coordinates
        if expected_quat is not None:
            # Check quaternion (allowing for sign flip since q and -q represent same rotation)
            # Compute dot product between actual and expected quaternions
            dot = abs(
                joint_q[0] * expected_quat[0]
                + joint_q[1] * expected_quat[1]
                + joint_q[2] * expected_quat[2]
                + joint_q[3] * expected_quat[3]
            )
            test.assertAlmostEqual(dot, 1.0, delta=1e-2)

        if expected_vel is not None:
            for i in range(3):
                test.assertAlmostEqual(joint_qd[i], expected_vel[i], delta=1e-2)
    finally:
        newton.use_coord_layout_targets = prev_flag


def test_ball_controller_coord_layout(
    test: TestJointController,
    device,
    solver_fn,
    target_axis_angle,
    expected_quat,
    target_ke,
    target_kd,
):
    """Ball-joint position target under the coord layout: the user writes a
    target quaternion and the MuJoCo solver must convert it to the matching
    axis-angle component before feeding it to per-axis position actuators.
    Without the conversion the equilibrium for a 90° setpoint sits at ~40.5°.
    """
    prev_flag = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = True
    try:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
        box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        b = builder.add_link(inertia=box_inertia, mass=1.0)
        builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))
        j = builder.add_joint_ball(
            parent=-1,
            child=b,
            parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            armature=0.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        builder.add_articulation([j])
        qd_start = builder.joint_qd_start[j]
        for i in range(3):
            builder.joint_target_ke[qd_start + i] = target_ke
            builder.joint_target_kd[qd_start + i] = target_kd

        model = builder.finalize(device=device)
        solver = solver_fn(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        # Convert axis-angle target to a quaternion and write to joint_target_q.
        ax = np.asarray(target_axis_angle, dtype=np.float32)
        angle = float(np.linalg.norm(ax))
        if angle > 0.0:
            n = ax / angle
            s = float(np.sin(angle / 2.0))
            target_quat = [n[0] * s, n[1] * s, n[2] * s, float(np.cos(angle / 2.0))]
        else:
            target_quat = [0.0, 0.0, 0.0, 1.0]
        control = model.control()
        control.joint_target_q = wp.array(target_quat, dtype=wp.float32, device=device)

        sim_dt = 1.0 / 60.0
        for _ in range(100):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        joint_q = state_0.joint_q.numpy()
        dot = abs(
            joint_q[0] * expected_quat[0]
            + joint_q[1] * expected_quat[1]
            + joint_q[2] * expected_quat[2]
            + joint_q[3] * expected_quat[3]
        )
        test.assertAlmostEqual(dot, 1.0, delta=1e-2)
    finally:
        newton.use_coord_layout_targets = prev_flag


def test_ball_controller_coord_layout_rotated_anchor(
    test: TestJointController,
    device,
    solver_fn,
    child_rot,
    target_axis_angle,
    expected_quat,
    expected_vel,
    target_ke,
    target_kd,
):
    """Ball-joint coord-layout target under non-identity ``child_xform`` rotation.

    The joint_target_q quaternion and joint_target_qd 3-vector live in Newton's
    parent anchor frame; the MuJoCo solver must conjugate them by ``q_cj``
    before handing them to per-axis actuators. With identity ``child_xform`` the conjugation is a
    no-op, so this test only fails under non-identity rotation. A preceding free joint keeps the
    ball target starts nonzero.
    """
    prev_flag = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = True
    try:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
        box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

        base = builder.add_link(inertia=box_inertia, mass=1.0)
        j_free = builder.add_joint_free(child=base)
        builder.add_articulation([j_free])

        b = builder.add_link(inertia=box_inertia, mass=1.0)
        builder.add_shape_box(body=b, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=1))
        j = builder.add_joint_ball(
            parent=-1,
            child=b,
            parent_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 2.0, 0.0), child_rot),
            armature=0.0,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        builder.add_articulation([j])
        qd_start = builder.joint_qd_start[j]
        for i in range(3):
            builder.joint_target_ke[qd_start + i] = target_ke
            builder.joint_target_kd[qd_start + i] = target_kd

        model = builder.finalize(device=device)
        q_start = int(model.joint_q_start.numpy()[j])
        qd_start = int(model.joint_qd_start.numpy()[j])
        test.assertGreater(q_start, 0)
        test.assertGreater(qd_start, 0)

        solver = solver_fn(model)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

        ax = np.asarray(target_axis_angle, dtype=np.float32)
        angle = float(np.linalg.norm(ax))
        if angle > 0.0:
            n = ax / angle
            s = float(np.sin(angle / 2.0))
            target_quat = [n[0] * s, n[1] * s, n[2] * s, float(np.cos(angle / 2.0))]
        else:
            target_quat = [0.0, 0.0, 0.0, 1.0]
        control = model.control()
        joint_target_q = control.joint_target_q.numpy()
        joint_target_q[q_start : q_start + 4] = target_quat
        wp.copy(control.joint_target_q, wp.array(joint_target_q, dtype=wp.float32, device=device))
        joint_target_qd = control.joint_target_qd.numpy()
        joint_target_qd[qd_start : qd_start + 3] = expected_vel
        wp.copy(control.joint_target_qd, wp.array(joint_target_qd, dtype=wp.float32, device=device))

        sim_dt = 1.0 / 60.0
        for _ in range(100):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0

        joint_q = state_0.joint_q.numpy()
        joint_qd = state_0.joint_qd.numpy()

        if expected_quat is not None:
            dot = abs(
                joint_q[q_start + 0] * expected_quat[0]
                + joint_q[q_start + 1] * expected_quat[1]
                + joint_q[q_start + 2] * expected_quat[2]
                + joint_q[q_start + 3] * expected_quat[3]
            )
            test.assertAlmostEqual(dot, 1.0, delta=1e-2)

        if target_ke == 0.0:
            for i in range(3):
                test.assertAlmostEqual(joint_qd[qd_start + i], expected_vel[i], delta=1e-2)
    finally:
        newton.use_coord_layout_targets = prev_flag


def test_free_plus_revolute_position_target(
    test: TestJointController,
    device,
    solver_fn,
    coord_layout: bool,
):
    """Position target on a revolute behind a free joint must be applied under
    both layouts (coord-index 7 vs DOF-index 6). One step + ``qfrc_actuator``
    isolates the indexing path from floating-base dynamics.
    """
    prev_flag = newton.use_coord_layout_targets
    newton.use_coord_layout_targets = coord_layout
    try:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

        base = builder.add_link(inertia=wp.mat33(np.eye(3) * 0.1), mass=1.0)
        builder.add_shape_box(body=base, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        j_free = builder.add_joint_free(child=base)

        child = builder.add_link(inertia=wp.mat33(np.eye(3) * 0.1), mass=1.0)
        builder.add_shape_box(body=child, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        target_pos = wp.pi / 4.0
        target_ke = 100.0
        j_rev = builder.add_joint_revolute(
            parent=base,
            child=child,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_pos=target_pos,
            target_vel=0.0,
            target_ke=target_ke,
            target_kd=0.0,
            armature=0.0,
            limit_ke=0.0,
            limit_kd=0.0,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        builder.add_articulation([j_free, j_rev])
        builder.request_state_attributes("mujoco:qfrc_actuator")

        model = builder.finalize(device=device)
        model.ground = False

        test.assertEqual(model.joint_dof_count, 7)
        test.assertEqual(model.joint_coord_count, 8)

        solver = solver_fn(model)
        if not isinstance(solver, newton.solvers.SolverMuJoCo):
            test.skipTest("qfrc_actuator-based check is MuJoCo-specific")

        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        control = model.control()

        solver.step(state_0, state_1, control, None, dt=0.01)

        qfrc = state_1.mujoco.qfrc_actuator.numpy()
        applied = float(qfrc[6])
        expected = target_ke * target_pos
        test.assertAlmostEqual(
            applied,
            expected,
            delta=expected * 0.1,
            msg=f"expected ~{expected:.2f}, got {applied:.4f} (coord_layout={coord_layout})",
        )
    finally:
        newton.use_coord_layout_targets = prev_flag


def test_effort_limit_clamping(
    test: TestJointController,
    device,
    solver_fn,
):
    """Test that MuJoCo solver correctly clamps actuator forces based on effort_limit."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))

    box_mass = 1.0
    inertia_value = 0.1
    box_inertia = wp.mat33((inertia_value, 0.0, 0.0), (0.0, inertia_value, 0.0), (0.0, 0.0, inertia_value))
    b = builder.add_link(inertia=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    # High PD gains should be clamped by low effort_limit
    high_kp = 10000.0
    high_kd = 1000.0
    effort_limit = 5.0

    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=0.0,
        target_vel=0.0,
        armature=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=high_kp,
        target_kd=high_kd,
        effort_limit=effort_limit,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    model = builder.finalize(device=device)
    model.ground = False
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    initial_q = 1.0
    initial_qd = 0.0
    state_0.joint_q.assign([initial_q])
    state_0.joint_qd.assign([initial_qd])

    control = model.control()
    control.joint_target_q = wp.array([0.0], dtype=wp.float32, device=device)
    control.joint_target_qd = wp.array([0.0], dtype=wp.float32, device=device)

    dt = 0.01

    F_unclamped = -high_kp * initial_q - high_kd * initial_qd
    F_clamped = np.clip(F_unclamped, -effort_limit, effort_limit)
    alpha = F_clamped / inertia_value
    qd_expected = initial_qd + alpha * dt
    q_expected = initial_q + qd_expected * dt

    solver.step(state_0, state_1, control, None, dt=dt)

    q_actual = state_1.joint_q.numpy()[0]
    qd_actual = state_1.joint_qd.numpy()[0]

    alpha_unclamped = F_unclamped / inertia_value
    qd_unclamped = initial_qd + alpha_unclamped * dt
    q_unclamped = initial_q + qd_unclamped * dt

    test.assertGreater(abs(q_unclamped - q_expected), 0.5, "Clamping should significantly affect the motion")

    tolerance = 0.05
    test.assertAlmostEqual(
        q_actual,
        q_expected,
        delta=tolerance,
        msg=f"Position with clamped effort limit: expected {q_expected:.4f}, got {q_actual:.4f}",
    )
    test.assertAlmostEqual(
        qd_actual,
        qd_expected,
        delta=tolerance * 10,
        msg=f"Velocity with clamped effort limit: expected {qd_expected:.4f}, got {qd_actual:.4f}",
    )


def test_qfrc_actuator(
    test: TestJointController,
    device,
    solver_fn,
):
    """Test that mujoco.qfrc_actuator extended state attribute is populated correctly by MuJoCo solver."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))

    box_mass = 1.0
    inertia_value = 0.1
    box_inertia = wp.mat33((inertia_value, 0.0, 0.0), (0.0, inertia_value, 0.0), (0.0, 0.0, inertia_value))
    b = builder.add_link(inertia=box_inertia, mass=box_mass)
    builder.add_shape_box(body=b, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))

    kp = 100.0
    kd = 10.0
    effort_limit = 5.0

    j = builder.add_joint_revolute(
        parent=-1,
        child=b,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_pos=0.0,
        target_vel=0.0,
        armature=0.0,
        limit_ke=0.0,
        limit_kd=0.0,
        target_ke=kp,
        target_kd=kd,
        effort_limit=effort_limit,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([j])

    builder.request_state_attributes("mujoco:qfrc_actuator")

    model = builder.finalize(device=device)
    model.ground = False
    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    # Verify that qfrc_actuator is allocated under the mujoco namespace
    test.assertTrue(hasattr(state_1, "mujoco"), "state should have mujoco namespace")
    test.assertIsNotNone(state_1.mujoco.qfrc_actuator, "mujoco.qfrc_actuator should be allocated")
    test.assertEqual(len(state_1.mujoco.qfrc_actuator), model.joint_dof_count)

    initial_q = 1.0
    initial_qd = 0.0
    state_0.joint_q.assign([initial_q])
    state_0.joint_qd.assign([initial_qd])

    control = model.control()
    control.joint_target_q = wp.array([0.0], dtype=wp.float32, device=device)
    control.joint_target_qd = wp.array([0.0], dtype=wp.float32, device=device)

    dt = 0.01

    # PD force: F = kp * (target - q) + kd * (target_vel - qd)
    F_unclamped = -kp * initial_q - kd * initial_qd
    F_expected = np.clip(F_unclamped, -effort_limit, effort_limit)

    solver.step(state_0, state_1, control, None, dt=dt)

    qfrc = state_1.mujoco.qfrc_actuator.numpy()
    test.assertEqual(len(qfrc), 1, "Should have one DOF")
    test.assertAlmostEqual(
        float(qfrc[0]),
        F_expected,
        delta=0.5,
        msg=f"qfrc_actuator: expected {F_expected:.4f}, got {float(qfrc[0]):.4f}",
    )

    # Verify that qfrc_actuator is NOT allocated when not requested
    builder2 = newton.ModelBuilder(up_axis=newton.Axis.Y, gravity=(0.0, 0.0, 0.0))
    b2 = builder2.add_link(inertia=box_inertia, mass=box_mass)
    builder2.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    j2 = builder2.add_joint_revolute(
        parent=-1,
        child=b2,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 0.0, 1.0),
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder2.add_articulation([j2])
    model2 = builder2.finalize(device=device)
    state_not_requested = model2.state()
    test.assertFalse(
        hasattr(state_not_requested, "mujoco") and hasattr(state_not_requested.mujoco, "qfrc_actuator"),
        "mujoco.qfrc_actuator should not exist when not requested",
    )


def test_free_joint_qfrc_actuator_frame(
    test: TestJointController,
    device,
    solver_fn,
):
    """Test free joint mujoco.qfrc_actuator frame conversion with actuators.

    A free-joint body is rotated 90deg around X (body-z -> world-(-y)).
    Two motor actuators are attached:
    - "thrust" with gear=[0,0,1,0,0,0]: applies force along MuJoCo DOF[2] (world-z linear).
    - "yaw"   with gear=[0,0,0,0,0,1]: applies torque on MuJoCo DOF[5] (body-z angular).

    After conversion to Newton world-frame:
    - thrust ctrl=10 -> qfrc_actuator linear z = 10 (linear DOFs are already world-frame).
    - yaw ctrl=10    -> torque around body-z = world-(-y), so qfrc_actuator angular y < 0.
    """
    # Build via MJCF to get actuators on the free joint
    mjcf = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="drone" pos="0 0 1" quat="0.7071 0.7071 0 0">
          <freejoint name="root"/>
          <geom type="box" size="0.1 0.1 0.05" mass="1"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
        </body>
      </worldbody>
      <actuator>
        <motor name="thrust" joint="root" gear="0 0 1 0 0 0"/>
        <motor name="yaw"   joint="root" gear="0 0 0 0 0 1"/>
      </actuator>
    </mujoco>
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    parse_mjcf(builder, mjcf, ctrl_direct=True, ignore_inertial_definitions=False)
    builder.request_state_attributes("mujoco:qfrc_actuator")

    model = builder.finalize(device=device)
    model.ground = False

    solver = solver_fn(model)

    state_0 = model.state()
    state_1 = model.state()

    control = model.control()
    # Set ctrl: [thrust=10, yaw=10]
    ctrl_vals = np.zeros(2, dtype=np.float32)
    ctrl_vals[0] = 10.0  # thrust along MuJoCo z-linear DOF
    ctrl_vals[1] = 10.0  # yaw torque around MuJoCo body-z angular DOF
    control.mujoco.ctrl = wp.array(ctrl_vals, dtype=wp.float32, device=device)

    dt = 0.01
    solver.step(state_0, state_1, control, None, dt=dt)

    qfrc = state_1.mujoco.qfrc_actuator.numpy()

    # --- Thrust check: linear force along world-z ---
    # Linear DOFs (0,1,2) = (fx, fy, fz) in world frame
    # Thrust gear=[0,0,1,...] -> force in world-z (same in MuJoCo and Newton)
    test.assertAlmostEqual(float(qfrc[0]), 0.0, delta=0.5, msg=f"thrust fx should be ~0, got {qfrc[0]:.2f}")
    test.assertAlmostEqual(float(qfrc[1]), 0.0, delta=0.5, msg=f"thrust fy should be ~0, got {qfrc[1]:.2f}")
    test.assertAlmostEqual(float(qfrc[2]), 10.0, delta=0.5, msg=f"thrust fz should be ~10, got {qfrc[2]:.2f}")

    # --- Yaw check: torque from body-z rotated to world frame ---
    # Body is rotated 90deg around X, so body-z -> world-(-y)
    # MuJoCo: torque around body-z = 10
    # Newton world-frame: torque should be around world-(-y), i.e. qfrc[4] < 0
    angular_qfrc = qfrc[3:6]
    test.assertGreater(
        abs(angular_qfrc[1]),
        abs(angular_qfrc[0]) + abs(angular_qfrc[2]) + 1.0,
        msg=f"yaw torque should be primarily around world-y, got {angular_qfrc}",
    )
    test.assertLess(
        angular_qfrc[1],
        0.0,
        msg=f"yaw torque should be negative-y (body-z -> world-(-y)), got {angular_qfrc}",
    )
    test.assertAlmostEqual(
        float(angular_qfrc[1]), -10.0, delta=1.0, msg=f"yaw torque magnitude should be ~10, got {angular_qfrc[1]:.2f}"
    )


devices = get_test_devices()
solvers = {
    "featherstone": lambda model: newton.solvers.SolverFeatherstone(model, angular_damping=0.0),
    "mujoco_cpu": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True),
    "mujoco_warp": lambda model: newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True),
    "xpbd": lambda model: newton.solvers.SolverXPBD(model, angular_damping=0.0, iterations=5),
    # "semi_implicit": lambda model: newton.solvers.SolverSemiImplicit(model, angular_damping=0.0),
}
for device in devices:
    for solver_name, solver_fn in solvers.items():
        if device.is_cuda and solver_name == "mujoco_cpu":
            continue

        if "mujoco" in solver_name:
            add_function_test(
                TestJointController,
                f"test_effort_limit_clamping_{solver_name}",
                test_effort_limit_clamping,
                devices=[device],
                solver_fn=solver_fn,
            )
            add_function_test(
                TestJointController,
                f"test_qfrc_actuator_{solver_name}",
                test_qfrc_actuator,
                devices=[device],
                solver_fn=solver_fn,
            )
            add_function_test(
                TestJointController,
                f"test_free_joint_qfrc_actuator_frame_{solver_name}",
                test_free_joint_qfrc_actuator_frame,
                devices=[device],
                solver_fn=solver_fn,
            )
        if solver_name in ("mujoco_cpu", "mujoco_warp"):
            for layout_name, layout_value in (("dof_layout", False), ("coord_layout", True)):
                add_function_test(
                    TestJointController,
                    f"test_free_plus_revolute_position_target_{layout_name}_{solver_name}",
                    test_free_plus_revolute_position_target,
                    devices=[device],
                    solver_fn=solver_fn,
                    coord_layout=layout_value,
                )

        # Revolute joint tests
        add_function_test(
            TestJointController,
            f"test_revolute_joint_controller_position_target_{solver_name}",
            test_revolute_controller,
            devices=[device],
            solver_fn=solver_fn,
            pos_target_val=wp.pi / 2.0,
            vel_target_val=0.0,
            expected_pos=wp.pi / 2.0,
            expected_vel=0.0,
            target_ke=2000.0,
            target_kd=500.0,
        )
        add_function_test(
            TestJointController,
            f"test_revolute_joint_controller_velocity_target_{solver_name}",
            test_revolute_controller,
            devices=[device],
            solver_fn=solver_fn,
            pos_target_val=0.0,
            vel_target_val=wp.pi / 2.0,
            expected_pos=None,
            expected_vel=wp.pi / 2.0,
            target_ke=0.0,
            target_kd=500.0,
        )

        if solver_name == "mujoco_cpu" or solver_name == "mujoco_warp":
            # Ball joint tests
            # Test 1: Position control - rotation around Z axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_z_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, wp.pi / 2.0],  # Rotate 90 degrees around Z
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.0, 0.0, 0.7071068, 0.7071068],  # quat for 90 deg around Z
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 2: Position control - rotation around X axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_x_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[wp.pi / 2.0, 0.0, 0.0],  # Rotate 90 degrees around X
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.7071068, 0.0, 0.0, 0.7071068],  # quat for 90 deg around X
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 3: Position control - rotation around Y axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_position_target_y_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, wp.pi / 2.0, 0.0],  # Rotate 90 degrees around Y
                vel_target_vals=[0.0, 0.0, 0.0],
                expected_quat=[0.0, 0.7071068, 0.0, 0.7071068],  # quat for 90 deg around Y
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )

            # Test 4: Velocity control - angular velocity around Z axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_z_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[0.0, 0.0, wp.pi / 2.0],  # Angular velocity around Z
                expected_quat=None,  # Don't check position for velocity control
                expected_vel=[0.0, 0.0, wp.pi / 2.0],
                target_ke=0.0,
                target_kd=500.0,
            )

            # Test 5: Velocity control - angular velocity around X axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_x_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[wp.pi / 2.0, 0.0, 0.0],  # Angular velocity around X
                expected_quat=None,
                expected_vel=[wp.pi / 2.0, 0.0, 0.0],
                target_ke=0.0,
                target_kd=500.0,
            )

            # Test 6: Velocity control - angular velocity around Y axis
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_velocity_target_y_{solver_name}",
                test_ball_controller,
                devices=[device],
                solver_fn=solver_fn,
                pos_target_vals=[0.0, 0.0, 0.0],
                vel_target_vals=[0.0, wp.pi / 2.0, 0.0],  # Angular velocity around Y
                expected_quat=None,
                expected_vel=[0.0, wp.pi / 2.0, 0.0],
                target_ke=0.0,
                target_kd=500.0,
            )

            # Coord layout: target quaternion must be converted to axis-angle
            # before being fed to MuJoCo position actuators on a ball joint.
            for axis_name, axis_angle, quat in (
                ("z", [0.0, 0.0, wp.pi / 2.0], [0.0, 0.0, 0.7071068, 0.7071068]),
                ("x", [wp.pi / 2.0, 0.0, 0.0], [0.7071068, 0.0, 0.0, 0.7071068]),
                ("y", [0.0, wp.pi / 2.0, 0.0], [0.0, 0.7071068, 0.0, 0.7071068]),
            ):
                add_function_test(
                    TestJointController,
                    f"test_ball_joint_controller_coord_layout_{axis_name}_{solver_name}",
                    test_ball_controller_coord_layout,
                    devices=[device],
                    solver_fn=solver_fn,
                    target_axis_angle=axis_angle,
                    expected_quat=quat,
                    target_ke=2000.0,
                    target_kd=500.0,
                )

            # Coord layout with non-identity child_xform.rot: targets in Newton's
            # parent anchor frame must be conjugated by q_cj before reaching MuJoCo.
            # Use a child anchor rotated 30° around Y so the conjugation is non-trivial
            # and the target axis (X or Z) doesn't commute with it.
            child_rot_y30 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 6.0)
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_coord_layout_rotated_anchor_pos_{solver_name}",
                test_ball_controller_coord_layout_rotated_anchor,
                devices=[device],
                solver_fn=solver_fn,
                child_rot=child_rot_y30,
                target_axis_angle=[wp.pi / 2.0, 0.0, 0.0],
                expected_quat=[0.7071068, 0.0, 0.0, 0.7071068],
                expected_vel=[0.0, 0.0, 0.0],
                target_ke=2000.0,
                target_kd=500.0,
            )
            add_function_test(
                TestJointController,
                f"test_ball_joint_controller_coord_layout_rotated_anchor_vel_{solver_name}",
                test_ball_controller_coord_layout_rotated_anchor,
                devices=[device],
                solver_fn=solver_fn,
                child_rot=child_rot_y30,
                target_axis_angle=[0.0, 0.0, 0.0],
                expected_quat=None,
                expected_vel=[wp.pi / 2.0, 0.0, 0.0],
                target_ke=0.0,
                target_kd=500.0,
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
