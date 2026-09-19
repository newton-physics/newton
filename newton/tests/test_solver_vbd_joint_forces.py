# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the optional VBD incoming-joint wrench reconstruction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0


def _make_joint_model(device, joint_type, *, friction=0.0, damping=0.0, **joint_kwargs):
    """Build a single unit-mass or unit-inertia joint with no gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    if joint_type == newton.JointType.REVOLUTE:
        joint = builder.add_joint_revolute(
            -1, body, axis=newton.Axis.Z, friction=friction, damping=damping, **joint_kwargs
        )
    else:
        joint = builder.add_joint_prismatic(
            -1, body, axis=newton.Axis.X, friction=friction, damping=damping, **joint_kwargs
        )
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device)


def _setup(model, *, alm=True, solve="local"):
    """Allocate a solver, initialized states, and force-query buffers."""
    solver = newton.solvers.SolverVBD(
        model, iterations=64, rigid_compliant_alm=alm, rigid_avbd_alpha=0.0, rigid_articulation_solve=solve
    )
    state, next_state = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    previous = wp.clone(state.body_q)
    wrench = wp.empty(model.joint_count, dtype=wp.spatial_vector, device=model.device)
    return solver, state, next_state, previous, wrench


def test_vbd_joint_force_losses(test, device, *, solve="local"):
    """Include actuation, Coulomb friction, and viscous damping in both solver modes."""
    for alm in (False, True):
        for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            for direction in (-1.0, 1.0):
                with test.subTest(alm=alm, joint_type=joint_type, direction=direction):
                    model = _make_joint_model(device, joint_type, friction=0.7, damping=0.3)
                    model.joint_qd.fill_(direction * 2.0)
                    solver, state, next_state, previous, wrench = _setup(model, alm=alm, solve=solve)
                    control = model.control()
                    control.joint_f.fill_(direction * 3.0)
                    solver.step(state, next_state, control, None, _DT)
                    history = solver.body_q_prev.numpy().copy()
                    turns = solver._joint_coordinates.data.history.numpy().copy()
                    duals = solver.joint_lambda_lin.numpy().copy()
                    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
                    rate = float(next_state.joint_q.numpy()[0]) / _DT
                    expected = direction * 3.0 - 0.7 * np.tanh(rate / 0.01) - 0.3 * rate
                    component = 5 if joint_type == newton.JointType.REVOLUTE else 0
                    expected_wrench = np.zeros((1, 6))
                    expected_wrench[0, component] = expected
                    np.testing.assert_allclose(wrench.numpy(), expected_wrench, atol=2.0e-4)
                    # Repeated queries overwrite output and do not advance any history.
                    first = wrench.numpy().copy()
                    wrench.fill_(wp.spatial_vector(123.0))
                    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
                    np.testing.assert_array_equal(wrench.numpy(), first)
                    np.testing.assert_array_equal(solver.body_q_prev.numpy(), history)
                    np.testing.assert_array_equal(solver._joint_coordinates.data.history.numpy(), turns)
                    np.testing.assert_array_equal(solver.joint_lambda_lin.numpy(), duals)
                    # Nonzero armature must not invent a force absent from VBD dynamics.
                    model.joint_armature.fill_(10.0)
                    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
                    np.testing.assert_array_equal(wrench.numpy(), first)


def test_vbd_joint_force_friction_response(test, device, *, solve="local"):
    """Report local ALM reactions or sparse regularized friction without changing history."""
    model = _make_joint_model(device, newton.JointType.REVOLUTE, friction=1.0)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    # Leave a displacement so re-projecting instead of reading lambda is detectable.
    solver.iterations = 1
    control = model.control()
    control.joint_f.fill_(0.5)
    solver.step(state, next_state, control, None, _DT)
    reaction = solver.joint_friction_lambda.numpy().copy()
    if solve == "local":
        test.assertGreater(float(reaction[0]), 0.0)
        test.assertLess(float(reaction[0]), 0.5)
        expected = 0.5 - float(reaction[0])
    else:
        np.testing.assert_array_equal(reaction, 0.0)
        rate = float(next_state.joint_q.numpy()[0]) / _DT
        expected = 0.5 - np.tanh(rate / 0.01)
    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
    test.assertAlmostEqual(float(wrench.numpy()[0, 5]), expected, delta=1.0e-5)
    np.testing.assert_array_equal(solver.joint_friction_lambda.numpy(), reaction)


def test_vbd_joint_force_fixed_frame(test, device, *, solve="local"):
    """Report fixed-joint loads in the child joint frame, shifted away from the COM."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -10.0))
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=wp.vec3(0.4, -0.2, 0.3))
    joint = builder.add_joint_fixed(
        -1,
        body,
        parent_xform=wp.transform(wp.vec3(2.0, 3.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.7)),
        child_xform=wp.transform(wp.vec3(0.1, 0.2, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    for _ in range(80):
        wp.copy(previous, state.body_q)
        solver.step(state, next_state, None, None, _DT)
        state, next_state = next_state, state
    solver.eval_joint_forces(state, wrench, body_q_prev=previous, dt=_DT)
    pose = wp.transform(*state.body_q.numpy()[0])
    anchor = pose * wp.transform(*model.joint_X_c.numpy()[0])
    rotation = wp.transform_get_rotation(anchor)
    com_offset = wp.transform_point(pose, wp.vec3(*model.body_com.numpy()[0])) - wp.transform_get_translation(anchor)
    expected_force = wp.vec3(0.0, 0.0, 20.0)
    expected = np.concatenate(
        (
            wp.quat_rotate_inv(rotation, expected_force),
            wp.quat_rotate_inv(rotation, wp.cross(com_offset, expected_force)),
        )
    )
    np.testing.assert_allclose(wrench.numpy()[0], expected, atol=0.03)


def test_vbd_joint_force_drive_limit(test, device, *, solve="local"):
    """Include converged drive and limit efforts independently of feedforward actuation."""
    for alm in (False, True):
        for limit in (False, True):
            with test.subTest(alm=alm, limit=limit):
                options = {"limit_lower": -0.2, "limit_upper": 0.2, "limit_ke": 200.0, "limit_kd": 1.0}
                if not limit:
                    options = {"target_ke": 20.0, "target_kd": 1.0}
                model = _make_joint_model(device, newton.JointType.REVOLUTE, **options)
                model.joint_q.fill_(0.3 if limit else 0.1)
                model.joint_qd.fill_(0.5)
                solver, state, next_state, previous, wrench = _setup(model, alm=alm, solve=solve)
                previous_qd = wp.clone(state.joint_qd)
                effort = wp.empty_like(state.joint_qd)
                control = model.control()
                control.joint_target_q.fill_(0.4)
                solver.step(state, next_state, control, None, _DT)
                solver.eval_joint_forces(
                    next_state,
                    wrench,
                    body_q_prev=previous,
                    dt=_DT,
                    control=control,
                    joint_effort=effort,
                    joint_qd_prev=previous_qd,
                )
                q = float(next_state.joint_q.numpy()[0])
                rate = (q - float(model.joint_q.numpy()[0])) / _DT
                expected = -200.0 * (q - 0.2) - rate if limit else 20.0 * (0.4 - q) - rate
                test.assertAlmostEqual(float(wrench.numpy()[0, 5]), expected, delta=0.03)
                test.assertAlmostEqual(float(effort.numpy()[0]), 0.0 if limit else expected, delta=0.03)


def test_vbd_joint_force_mimic(test, device, *, solve="local", standalone=False):
    """Include both sides of multiple mimic reactions without double-counting follower loads."""
    for alm in (False, True):
        with test.subTest(alm=alm):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            ratios = np.array([1.0, -1.0, 2.0])
            friction = np.array([0.2, 0.4, 0.6])
            damping = np.array([0.1, 0.2, 0.3])
            joints = []
            for i in range(3):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joints.append(
                    builder.add_joint_prismatic(-1, body, axis=newton.Axis.X, friction=friction[i], damping=damping[i])
                )
            builder.add_articulation(joints)
            builder.set_joint_mimic(joints[1], joints[0], (0.0, -1.0))
            builder.set_joint_mimic(joints[2], joints[0], (0.0, 2.0))
            builder.joint_qd[:] = ratios.tolist()
            if standalone:
                # Exercise the local-body sweep after the sparse assembly.
                builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.color()
            model = builder.finalize(device=device)
            model.joint_armature.assign([0.3, 0.5, 0.7])
            solver, state, next_state, previous, wrench = _setup(model, alm=alm, solve=solve)
            previous_qd = wp.clone(state.joint_qd)
            effort = wp.empty_like(state.joint_qd)
            control = model.control()
            control.joint_f.assign([5.0, 0.0, 0.0])
            solver.step(state, next_state, control, None, _DT)
            solver.eval_joint_forces(
                next_state,
                wrench,
                body_q_prev=previous,
                dt=_DT,
                control=control,
                joint_effort=effort,
                joint_qd_prev=previous_qd,
            )
            velocity = next_state.joint_qd.numpy()
            force = wrench.numpy()[:, 0]
            np.testing.assert_allclose(force, (velocity - ratios) / _DT, atol=0.03)
            passive = -friction * np.tanh(velocity / 0.01) - damping * velocity
            # Mimic forces do no work along the permitted coupled motion.
            test.assertAlmostEqual(float(np.dot(ratios, force)), 5.0 + float(np.dot(ratios, passive)), delta=2.0e-4)
            rotor_effort = np.dot(ratios * model.joint_armature.numpy(), (velocity - ratios) / _DT)
            np.testing.assert_allclose(effort.numpy(), [5.0 + rotor_effort, 0.0, 0.0], atol=2.0e-4)


def test_vbd_joint_force_serial_mimic(test, device, *, solve="local"):
    """Keep incoming and outgoing reactions distinct on a serial mimic/passive chain."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    parent = -1
    for i in range(3):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(parent, body, axis=newton.Axis.Z, friction=0.2 * (i + 1), damping=0.1 * (i + 1))
        )
        parent = body
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])
    builder.joint_qd[:] = [1.0, 1.0, 0.5]
    builder.color()
    model = builder.finalize(device=device)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    initial_omega = state.body_qd.numpy()[:, 5].copy()
    control = model.control()
    control.joint_f.assign([5.0, 0.0, 0.0])
    solver.step(state, next_state, control, None, _DT)
    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
    torque = wrench.numpy()[:, 5]
    acceleration = (next_state.body_qd.numpy()[:, 5] - initial_omega) / _DT
    np.testing.assert_allclose(torque - np.append(torque[1:], 0.0), acceleration, atol=0.03)


def test_vbd_joint_force_d6(test, device, *, solve="local"):
    """Report distinct translational and rotational D6 losses in the moving child frame."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    config = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        -1,
        body,
        linear_axes=[config(axis=newton.Axis.X, friction=0.4, damping=0.1)],
        angular_axes=[config(axis=newton.Axis.Z, friction=0.8, damping=0.2)],
    )
    builder.add_articulation([joint])
    builder.joint_q[:] = [0.2, 0.3]
    builder.joint_qd[:] = [1.0, 2.0]
    builder.color()
    model = builder.finalize(device=device)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    control = model.control()
    control.joint_f.assign([2.0, 3.0])
    solver.step(state, next_state, control, None, _DT)
    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
    rate = (next_state.joint_q.numpy() - model.joint_q.numpy()) / _DT
    effort = np.array([2.0, 3.0]) - np.array([0.4, 0.8]) * np.tanh(rate / 0.01) - np.array([0.1, 0.2]) * rate
    angle = float(next_state.joint_q.numpy()[1])
    expected = [effort[0] * np.cos(angle), -effort[0] * np.sin(angle), 0.0, 0.0, 0.0, effort[1]]
    np.testing.assert_allclose(wrench.numpy()[0], expected, atol=2.0e-4)


def test_vbd_joint_force_contact(test, device, *, solve="local"):
    """Keep transmitted actuator load nonzero when an external contact balances it."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e4, kd=10.0)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.2, cfg=cfg)
    joint = builder.add_joint_prismatic(
        -1, body, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity())
    )
    builder.add_articulation([joint])
    builder.color()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    control = model.control()
    control.joint_f.fill_(-5.0)
    for _ in range(80):
        wp.copy(previous, state.body_q)
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, _DT)
        state, next_state = next_state, state
    solver.eval_joint_forces(state, wrench, body_q_prev=previous, dt=_DT, control=control)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    test.assertLess(float(np.linalg.norm(state.body_qd.numpy())), 0.01)
    test.assertAlmostEqual(float(wrench.numpy()[0, 2]), -5.0, delta=1.0e-4)


def test_vbd_joint_force_validation(test, device, *, solve="local"):
    """Reject invalid query inputs and clear outputs for disabled joints."""
    model = _make_joint_model(device, newton.JointType.PRISMATIC, friction=1.0)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    for dt in (0.0, -1.0, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            solver.eval_joint_forces(state, wrench, body_q_prev=previous, dt=dt)
    with test.assertRaises(ValueError):
        solver.eval_joint_forces(state, wrench, body_q_prev=state.body_q, dt=_DT)
    with test.assertRaises(ValueError):
        solver.eval_joint_forces(state, wrench, body_q_prev=solver.body_q_prev, dt=_DT)
    with test.assertRaises(ValueError):
        wrong = wp.empty(1, dtype=wp.vec3, device=device)
        solver.eval_joint_forces(state, wrong, body_q_prev=previous, dt=_DT)
    effort = wp.empty_like(state.joint_qd)
    with test.assertRaises(ValueError):
        solver.eval_joint_forces(state, wrench, body_q_prev=previous, dt=_DT, joint_effort=effort)
    with test.assertRaises(ValueError):
        solver.eval_joint_forces(
            state, wrench, body_q_prev=previous, dt=_DT, joint_effort=effort, joint_qd_prev=state.joint_qd
        )
    model.joint_enabled.fill_(False)
    control = model.control()
    control.joint_f.fill_(100.0)
    solver.step(state, next_state, control, None, _DT)
    wrench.fill_(wp.spatial_vector(123.0))
    solver.eval_joint_forces(next_state, wrench, body_q_prev=previous, dt=_DT, control=control)
    np.testing.assert_array_equal(wrench.numpy(), np.zeros((1, 6)))


def test_vbd_joint_force_capture(test, device, *, solve="local"):
    """Replay step and force-query kernels together without allocations or history loss."""
    model = _make_joint_model(device, newton.JointType.PRISMATIC, friction=0.7, damping=0.3)
    model.joint_qd.fill_(2.0)
    model.joint_armature.fill_(1.2)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    previous_qd = wp.clone(state.joint_qd)
    effort = wp.empty_like(state.joint_qd)
    control = model.control()
    control.joint_f.fill_(3.0)

    def step_and_query():
        wp.copy(previous, state.body_q)
        wp.copy(previous_qd, state.joint_qd)
        solver.step(state, next_state, control, None, _DT)
        solver.eval_joint_forces(
            next_state,
            wrench,
            body_q_prev=previous,
            dt=_DT,
            control=control,
            joint_effort=effort,
            joint_qd_prev=previous_qd,
        )

    step_and_query()
    graph = None
    if model.device.is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            step_and_query()
        graph = capture.graph
    for _ in range(4):
        if graph is not None:
            wp.capture_launch(graph)
        else:
            step_and_query()
        rate = float(next_state.joint_qd.numpy()[0])
        test.assertAlmostEqual(float(wrench.numpy()[0, 0]), 3.0 - 0.7 * np.tanh(rate / 0.01) - 0.3 * rate, delta=2.0e-4)
        motor_effort = 3.0 + 1.2 * (rate - float(previous_qd.numpy()[0])) / _DT
        test.assertAlmostEqual(float(effort.numpy()[0]), motor_effort, delta=2.0e-4)


def test_vbd_joint_effort_multiaxis(test, device, *, solve="local"):
    """Project D6 actuation onto its moving motion axes, not coordinate gradients."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
    config = newton.ModelBuilder.JointDofConfig
    joint = builder.add_joint_d6(
        -1, body, angular_axes=[config(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)]
    )
    builder.add_articulation([joint])
    builder.joint_q[:] = [0.3, 0.4, 0.5]
    builder.color()
    model = builder.finalize(device=device)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    previous_qd = wp.clone(state.joint_qd)
    effort = wp.empty_like(state.joint_qd)
    control = model.control()
    control.joint_f.assign([2.0, 3.0, 5.0])
    solver.step(state, next_state, control, None, _DT)
    solver.eval_joint_forces(
        next_state,
        wrench,
        body_q_prev=previous,
        dt=_DT,
        control=control,
        joint_effort=effort,
        joint_qd_prev=previous_qd,
    )
    q0, q1, _q2 = next_state.joint_q.numpy()
    expected = [
        2.0,
        3.0 * np.cos(q0) + 5.0 * np.sin(q0),
        2.0 * np.sin(q1) - 3.0 * np.sin(q0) * np.cos(q1) + 5.0 * np.cos(q0) * np.cos(q1),
    ]
    np.testing.assert_allclose(effort.numpy(), expected, atol=2.0e-5)


def test_vbd_joint_effort_rotating_parent(test, device, *, solve="local"):
    """Match linear-drive reporting to the solver with a rotating kinematic parent."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
    child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    root = builder.add_joint_free(parent)
    joint = builder.add_joint_prismatic(parent, child, axis=newton.Axis.X, target_ke=0.0, target_kd=10.0)
    builder.add_articulation([root, joint])
    builder.joint_q[-1] = 0.4
    builder.color()
    model = builder.finalize(device=device)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    control = model.control()
    solver.step(state, next_state, control, None, _DT)
    state, next_state = next_state, state
    wp.copy(previous, solver.body_q_prev)
    previous_qd = wp.clone(state.joint_qd)
    effort = wp.empty_like(state.joint_qd)
    poses = state.body_q.numpy()
    poses[parent, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3))
    state.body_q.assign(poses)
    solver.step(state, next_state, control, None, _DT)
    solver.eval_joint_forces(
        next_state,
        wrench,
        body_q_prev=previous,
        dt=_DT,
        control=control,
        joint_effort=effort,
        joint_qd_prev=previous_qd,
    )
    child_rotation = wp.quat(*next_state.body_q.numpy()[child, 3:])
    force_world = wp.quat_rotate(child_rotation, wp.vec3(*wrench.numpy()[joint, :3]))
    axis_world = wp.vec3(np.cos(0.3), np.sin(0.3), 0.0)
    test.assertAlmostEqual(float(effort.numpy()[-1]), float(wp.dot(axis_world, force_world)), delta=0.01)


def test_vbd_joint_effort_simulated_armature(test, device, *, solve="local"):
    """Avoid adding an armature estimate when sparse dynamics already simulate it."""
    model = _make_joint_model(device, newton.JointType.REVOLUTE, armature=2.0)
    solver, state, next_state, previous, wrench = _setup(model, solve=solve)
    previous_qd = wp.clone(state.joint_qd)
    effort = wp.empty_like(state.joint_qd)
    control = model.control()
    control.joint_f.fill_(3.0)
    solver.step(state, next_state, control, None, _DT)
    solver.eval_joint_forces(
        next_state,
        wrench,
        body_q_prev=previous,
        dt=_DT,
        control=control,
        joint_effort=effort,
        joint_qd_prev=previous_qd,
    )
    acceleration = float(next_state.joint_qd.numpy()[0]) / _DT
    simulated = solve == "block_sparse_joints"
    test.assertAlmostEqual(acceleration, 1.0 if simulated else 3.0, delta=0.005)
    test.assertAlmostEqual(float(wrench.numpy()[0, 5]), 3.0, delta=2.0e-4)
    test.assertAlmostEqual(float(effort.numpy()[0]), 3.0 if simulated else 3.0 + 2.0 * acceleration, delta=0.005)


class TestSolverVBDJointForces(unittest.TestCase):
    pass


for test_function in (
    test_vbd_joint_force_losses,
    test_vbd_joint_force_friction_response,
    test_vbd_joint_force_fixed_frame,
    test_vbd_joint_force_drive_limit,
    test_vbd_joint_force_mimic,
    test_vbd_joint_force_serial_mimic,
    test_vbd_joint_force_d6,
    test_vbd_joint_force_contact,
    test_vbd_joint_force_validation,
    test_vbd_joint_force_capture,
    test_vbd_joint_effort_multiaxis,
    test_vbd_joint_effort_rotating_parent,
    test_vbd_joint_effort_simulated_armature,
):
    add_function_test(
        TestSolverVBDJointForces, test_function.__name__, test_function, devices=get_test_devices(mode="basic")
    )
    add_function_test(
        TestSolverVBDJointForces,
        test_function.__name__ + "_sparse",
        test_function,
        devices=get_test_devices(mode="basic"),
        solve="block_sparse_joints",
    )


add_function_test(
    TestSolverVBDJointForces,
    "test_vbd_joint_force_mimic_sparse_with_local_body",
    test_vbd_joint_force_mimic,
    devices=get_test_devices(mode="basic"),
    solve="block_sparse_joints",
    standalone=True,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
