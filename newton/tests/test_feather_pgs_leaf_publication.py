# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared tree fixtures and serial-versus-parallel publication checks for SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

POISON = -123.25


def _build_model(device="cpu", *, locked_d6=False, chain=False, leaves=7):
    """Build a tree of prismatic leaves with nontrivial anchors and COMs, plus unrelated moving bodies."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))

    def body(index):
        return builder.add_link(
            mass=0.5 + 0.1 * index,
            com=wp.vec3(0.07, -0.03, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )

    base = body(0)
    root_kwargs = {
        "parent": -1,
        "child": base,
        "parent_xform": wp.transform(wp.vec3(2.0, -1.0, 0.6), wp.quat_rpy(0.3, -0.2, 0.4)),
        "child_xform": wp.transform(wp.vec3(0.2, -0.1, 0.05), wp.quat_rpy(-0.1, 0.2, 0.0)),
    }
    root = builder.add_joint_d6(**root_kwargs) if locked_d6 else builder.add_joint_fixed(**root_kwargs)
    joints = [root]
    leaf_bodies = []
    for index in range(leaves):
        child = body(index + 1)
        leaf_bodies.append(child)
        joints.append(
            builder.add_joint_prismatic(
                parent=(base if index == 0 else leaf_bodies[index - 1]) if chain else base,
                child=child,
                axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9),
                parent_xform=wp.transform(wp.vec3(0.1 * index, -0.05, 0.13), wp.quat_rpy(0.2, -0.1 * index, 0.3)),
                child_xform=wp.transform(wp.vec3(-0.09, 0.04, 0.08), wp.quat_rpy(0.4, 0.2, -0.1)),
                target_ke=3.0,
                target_kd=0.2,
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
    builder.add_articulation(joints)
    arm = body(leaves + 1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(-0.8, 0.1, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    free = body(leaves + 2)
    builder.add_articulation([builder.add_joint_free(free)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    q = model.joint_q.numpy()
    q[: leaves + 1] = np.linspace(-0.31, 0.27, leaves + 1)
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    # Joint axes need not be unit length.
    axes = model.joint_axis.numpy()
    axes[0] *= 1.4
    model.joint_axis.assign(axes)
    return model, joints, leaf_bodies


def _build_branched_model(device="cpu", *, floating_root=False):
    """Interleave terminal joints and internal parents with nonzero parent motion."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    bodies = [
        builder.add_link(
            mass=0.7 + 0.1 * index,
            com=wp.vec3(0.03, -0.04, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )
        for index in range(7)
    ]
    root_args = {"parent": -1, "child": bodies[0]}
    root = builder.add_joint_free(**root_args) if floating_root else builder.add_joint_revolute(**root_args)
    joints = [root]
    # Joint 1 is terminal although it precedes the internal joint 2; later leaves use both moving parents.
    for index, kind in enumerate(("revolute", "revolute", "prismatic", "ball", "d6", "fixed"), start=1):
        args = {
            "parent": bodies[2] if index in (4, 5) else bodies[0],
            "child": bodies[index],
            "parent_xform": wp.transform(wp.vec3(0.1 * index, -0.07, 0.13), wp.quat_rpy(0.2, -0.1, 0.3)),
            "child_xform": wp.transform(wp.vec3(-0.03, 0.04, 0.02), wp.quat_rpy(-0.1, 0.2, 0.1)),
        }
        if kind in ("revolute", "prismatic"):
            args["axis"] = wp.vec3(0.2, 0.5, 0.9)
        if kind == "d6":
            args["linear_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
            args["angular_axes"] = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y)]
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
    builder.add_articulation(joints)
    model = builder.finalize(device=device)
    q = model.joint_q.numpy()
    starts = model.joint_q_start.numpy()
    for joint in (1, 2, 3, 5):
        q[starts[joint] : starts[joint + 1]] = 0.07 * joint
    if floating_root:
        q[:7] = (0.4, -0.2, 0.6, *wp.quat_rpy(0.3, -0.2, 0.1))
    else:
        q[0] = 0.31
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.9, 0.8, model.joint_dof_count, dtype=np.float32))
    return model, joints, bodies


def _build_contact_model(device):
    """Load three terminal joints against the ground under a moving root."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1))
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=root,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.185), wp.quat_identity()),
        )
    ]
    for index, kind in enumerate(("revolute", "prismatic", "ball")):
        child = builder.add_link(mass=0.5, inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.03))
        args = {
            "parent": root,
            "child": child,
            "parent_xform": wp.transform(wp.vec3(0.5 * (index - 1), 0.3, 0.0), wp.quat_identity()),
        }
        if kind != "ball":
            args["axis"] = newton.Axis.Y if kind == "revolute" else newton.Axis.Z
        joints.append(getattr(builder, f"add_joint_{kind}")(**args))
        builder.add_shape_sphere(child, radius=0.2)
    builder.add_articulation(joints)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.linspace(-0.1, 0.15, model.joint_dof_count, dtype=np.float32))
    return model


def _solver(model, *, enabled, velocity_limits=False, **options):
    """Build a solver with serial (``enabled=False``) or parallel tree traversal."""
    solver = SolverFeatherPGS(
        model,
        pgs_iterations=8,
        update_mass_matrix_interval=2,
        enable_joint_limits=True,
        enable_joint_velocity_limits=velocity_limits,
        parallel_tree=enabled,
        **options,
    )
    if enabled:
        assert solver._tree_plan is not None
    else:
        assert solver._tree_plan is None
        assert solver._tree_net_wrenches == ()
    solver._prepare_augmented_state(model.state())
    return solver


def _dynamics_fields(solver, state):
    """Arrays written by the forward pass of stage 1."""
    return {
        "body_q": state.body_q,
        "body_q_com": solver.body_q_com,
        "origin": solver.articulation_origin,
        "S": solver.joint_S_s,
        "v": solver.body_v_s,
        "a": solver.body_a_s,
        "I": solver.body_I_s,
        "f": solver.body_f_s,
    }


def _poison(fields, keep=()):
    for name, array in fields.items():
        if name not in keep:
            array.fill_(POISON)


def _run_stage1(solver, state, following):
    """Run stage 1 forward and backward passes; return the predictor input velocity."""
    qd = solver._stage1_fk_id(state, solver, following)
    solver._stage1_joint_tau(state, solver, following, solver.model.control(), 1.0 / 240.0)
    return qd


def _check_graph_publication(test, model):
    """Publish body state eagerly and by CUDA graph replay into poisoned buffers."""
    device = model.device
    solvers = [_solver(model, enabled=value) for value in (False, True)]
    eager = []
    for solver in solvers:
        state = model.state()
        fields = {"body_q": state.body_q, "body_qd": state.body_qd}
        q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()
        _poison(fields)
        solver._stage7_update_kinematics(state)
        expected = {name: array.numpy().copy() for name, array in fields.items()}
        test.assertTrue(all(np.isfinite(value).all() for value in expected.values()))
        with wp.ScopedCapture(device=device) as capture:
            solver._stage7_update_kinematics(state)
        for _ in range(2):
            _poison(fields)
            wp.capture_launch(capture.graph)
            for name, array in fields.items():
                np.testing.assert_array_equal(array.numpy(), expected[name], err_msg=name)
        np.testing.assert_array_equal(state.joint_q.numpy(), q)
        np.testing.assert_array_equal(state.joint_qd.numpy(), qd)
        eager.append(expected)
    reference = model.state()
    newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
    for name in eager[0]:
        np.testing.assert_allclose(eager[1][name], eager[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
        np.testing.assert_allclose(eager[1][name], getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6)


def test_cuda_actual_graph_publication(test, device):
    """Match poisoned-buffer graph replays of the published state for broad and mixed moving trees."""
    _check_graph_publication(test, _build_model(device, locked_d6=True, leaves=108)[0])
    _check_graph_publication(test, _build_branched_model(device, floating_root=True)[0])


def test_interleaved_mixed_leaves_and_moving_parents(test, device):
    """Keep internal joints in order and match the serial forward pass with moving parents."""
    for floating in (False, True):
        with test.subTest(floating_root=floating):
            model, _, bodies = _build_branched_model(device, floating_root=floating)
            solvers = [_solver(model, enabled=value) for value in (False, True)]
            leaves = [bodies[index] for index in (1, 3, 4, 5, 6)]
            snapshots = []
            for solver in solvers:
                state, following = model.state(), model.state()
                _poison(_dynamics_fields(solver, state))
                _run_stage1(solver, state, following)
                snapshots.append(
                    {name: array.numpy().copy() for name, array in _dynamics_fields(solver, state).items()}
                )
            for name in snapshots[0]:
                np.testing.assert_allclose(snapshots[1][name], snapshots[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
            test.assertGreater(float(np.max(np.abs(snapshots[1]["a"][leaves]))), 1e-3)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshots[1]["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)


def test_current_frames_and_inertial_notifications(test, device):
    """Read current joint frames, axes, COMs, masses, inertias and gravity after a notification."""
    model, _, _ = _build_model(device)
    solvers = [_solver(model, enabled=value) for value in (False, True)]
    before = []
    for solver in solvers:
        state, following = model.state(), model.state()
        _run_stage1(solver, state, following)
        before.append(state.body_q.numpy().copy())
    frames = model.joint_X_p.numpy()
    frames[0, :3] += (0.3, -0.7, 0.5)
    frames[1, :3] += (0.2, 0.4, -0.1)
    model.joint_X_p.assign(frames)
    child_frames = model.joint_X_c.numpy()
    child_frames[1, :3] += (-0.1, 0.07, 0.09)
    model.joint_X_c.assign(child_frames)
    axes = model.joint_axis.numpy()
    axes[0] = (0.4, -0.7, 1.2)
    model.joint_axis.assign(axes)
    model.body_com.assign(model.body_com.numpy() + np.array((0.01, 0.02, -0.03), dtype=np.float32))
    model.body_mass.assign(model.body_mass.numpy() * 1.3)
    model.body_inertia.assign(model.body_inertia.numpy() * 1.2)
    model.gravity.assign(np.array([[1.0, -2.0, -8.0]], dtype=np.float32))
    after = []
    for solver in solvers:
        solver.notify_model_changed(newton.ModelFlags.ALL)
        state, following = model.state(), model.state()
        _run_stage1(solver, state, following)
        after.append({name: array.numpy().copy() for name, array in _dynamics_fields(solver, state).items()})
        after[-1]["tau"] = solver.joint_tau.numpy().copy()
    test.assertFalse(np.array_equal(before[1], after[1]["body_q"]))
    for name in after[0]:
        np.testing.assert_allclose(after[1][name], after[0][name], rtol=3e-6, atol=3e-6, err_msg=name)


def _check_full_steps_and_reset(test, model, *, loaded, velocity_limits=False):
    contact_capacity = 32 if loaded else 1
    model.rigid_contact_max = contact_capacity
    if velocity_limits:
        model.joint_velocity_limit.fill_(0.01)
    solvers = [_solver(model, enabled=value, velocity_limits=velocity_limits) for value in (False, True)]
    states = [[model.state(), model.state()] for _ in solvers]
    controls = [model.control() for _ in solvers]
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=contact_capacity)
    contacts = [pipeline.contacts() for _ in solvers]
    saw_contacts = False
    for pair in states:
        newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
    for step in range(6):
        for index, solver in enumerate(solvers):
            current, following = states[index]
            if step == 3:
                solver.reset(current)
            if loaded:
                pipeline.collide(current, contacts[index])
                saw_contacts |= int(contacts[index].rigid_contact_count.numpy()[0]) > 0
            forces = np.zeros((model.body_count, 6), dtype=np.float32)
            forces[1, :3] = (0.2 * step, -0.1, 0.3)
            current.body_f.assign(forces)
            controls[index].joint_f.assign(np.linspace(-0.1, 0.2, model.joint_dof_count, dtype=np.float32))
            solver.step(current, following, controls[index], contacts[index], 1.0 / 240.0)
            if velocity_limits and step == 0:
                test.assertGreater(float(np.max(np.abs(solver.qd_work.numpy() - current.joint_qd.numpy()))), 0.01)
            states[index] = [following, current]
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(states[1][0], name).numpy(),
                getattr(states[0][0], name).numpy(),
                rtol=1e-5,
                atol=2e-6,
                err_msg=f"step {step} {name}",
            )
        np.testing.assert_allclose(solvers[1].v_hat.numpy(), solvers[0].v_hat.numpy(), rtol=1e-5, atol=2e-6)
    if loaded:
        test.assertTrue(saw_contacts, "the loaded fixture must create contact rows")
        graphs = []
        for index, solver in enumerate(solvers):
            pair = states[index]
            with wp.ScopedCapture(device=model.device) as capture:
                for phase in (0, 1):
                    pipeline.collide(pair[phase], contacts[index])
                    solver.step(pair[phase], pair[1 - phase], controls[index], contacts[index], 1.0 / 240.0)
            graphs.append(capture.graph)
        for replay in range(2):
            if replay == 1:
                model.body_mass.assign(model.body_mass.numpy() * 1.03)
                model.body_inertia.assign(model.body_inertia.numpy() * 1.03)
                for index, solver in enumerate(solvers):
                    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
                    solver.reset(states[index][0])
            for graph in graphs:
                wp.capture_launch(graph)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(states[1][0], name).numpy(),
                    getattr(states[0][0], name).numpy(),
                    rtol=1e-5,
                    atol=2e-6,
                    err_msg=f"captured replay {replay}: {name}",
                )


def test_full_steps_and_reset(test, device):
    """Match serial traversal over complete steps, a reset, contacts, velocity limits and graph replay."""
    with test.subTest(fixture="tree"):
        _check_full_steps_and_reset(test, _build_model(device, leaves=4)[0], loaded=False)
    with test.subTest(fixture="contacts"):
        _check_full_steps_and_reset(test, _build_contact_model(device), loaded=True)
    with test.subTest(fixture="contacts", velocity_limits=True):
        _check_full_steps_and_reset(test, _build_contact_model(device), loaded=True, velocity_limits=True)


class TestLeafPublication(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_cuda_actual_graph_publication",
    "test_interleaved_mixed_leaves_and_moving_parents",
    "test_current_frames_and_inertial_notifications",
    "test_full_steps_and_reset",
):
    add_function_test(TestLeafPublication, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
