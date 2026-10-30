# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loop closures, mimic joints, pre-elimination, springs, free roots and resets of SolverSteel."""

import functools
import inspect
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel import kernels as steel_kernels
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    integrate_generalized_joints,
    remove_free_root_transport_from_qdd,
    update_qdd_from_velocity,
)
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverSteel
from newton.tests.steel._fixtures import (
    _CHILD_ANCHORS,
    _REL_P,
    _build_carried_load,
    _build_four_bar,
    _build_two_revolute_chain,
    _loop_anchor_gap,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _run(device, steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, model, state_0


def _build_interleaved_four_bar():
    """Four-bar tree, then a free-jointed object articulation, then the closure BALL joint.

    Importers that parse equalities after the scene tree append loop joints last, so the
    closure falls into a later articulation's joint range.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 0.6
    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

    body = b.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_sphere(body, radius=0.05)
    j_free = b.add_joint_free(child=body)
    b.add_articulation([j_free], label="free_object")

    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b


def _build_standalone_world_root():
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    articulated = b.add_link()
    b.add_shape_box(articulated, hx=0.05, hy=0.05, hz=0.1)
    tree_joint = b.add_joint_revolute(parent=-1, child=articulated, axis=newton.Axis.Y)
    b.add_articulation([tree_joint], label="articulated")
    standalone = b.add_link()
    b.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
    standalone_joint = b.add_joint_fixed(parent=-1, child=standalone)
    return b, standalone_joint


def test_loop_joint_presence_is_stable(test, device):
    """Keep a trailing BALL loop joint out of the tree solve without destabilizing it."""
    _, _, state = _run(device, steps=120, crank_target=0.0)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())
    test.assertTrue(np.isfinite(state.joint_qd.numpy()).all())


def test_loop_closure_is_enforced(test, device):
    """Hold the four-bar's loop anchors together with the connect rows under drive."""
    _, model, state = _run(device, steps=720, crank_target=0.6)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())
    gap = _loop_anchor_gap(model, state)
    test.assertLess(gap, 2.0e-3, f"loop anchor gap {gap * 1e3:.2f} mm: four-bar not closed")


def test_rocker_follows_crank(test, device):
    """Move the undriven rocker coherently with the driven crank through the loop."""
    _, _, state = _run(device, steps=720, crank_target=0.6)
    q = state.joint_q.numpy()
    # Parallel four-bar (equal crank and rocker lengths): the rocker angle tracks the crank angle.
    test.assertAlmostEqual(q[2], q[0], delta=0.08)
    test.assertGreater(abs(q[2]), 0.3, "rocker did not move: loop not transmitting")


def test_four_bar_capture_matches_eager(test, device):
    """Replay the eager four-bar trajectory from a captured step."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    eager = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    captured = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    e0, e1 = model.state(), model.state()
    c0, c1 = model.state(), model.state()

    def step_pair(solver, s0, s1):
        solver.step(s0, s1, control, None, 1.0 / 240.0)
        solver.step(s1, s0, control, None, 1.0 / 240.0)

    step_pair(captured, c0, c1)
    step_pair(eager, e0, e1)
    # Capturing records the launches without running them.
    with wp.ScopedCapture(device=device) as capture:
        step_pair(captured, c0, c1)
    for _ in range(60):
        wp.capture_launch(capture.graph)
        step_pair(eager, e0, e1)
    np.testing.assert_allclose(c0.joint_q.numpy(), e0.joint_q.numpy(), atol=2.0e-5)
    test.assertLess(_loop_anchor_gap(model, c0), 2.0e-3)


def test_connect_ownership_survives_interleaved_articulation(test, device):
    """Keep a deferred closure with its child body's articulation."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    left = b.add_link()
    b.add_shape_box(left, hx=0.05, hy=0.05, hz=0.1)
    j_left = b.add_joint_revolute(parent=-1, child=left, axis=newton.Axis.Y)
    right = b.add_link()
    b.add_shape_box(right, hx=0.05, hy=0.05, hz=0.1)
    j_right = b.add_joint_revolute(parent=-1, child=right, axis=newton.Axis.Y)
    b.add_articulation([j_left, j_right], label="closed_linkage")

    foreign = b.add_link()
    b.add_shape_sphere(foreign, radius=0.05)
    j_foreign = b.add_joint_free(child=foreign)
    b.add_articulation([j_foreign], label="foreign")

    loop_joint = b.add_joint_ball(parent=left, child=right)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free")

    test.assertEqual(int(model.joint_articulation.numpy()[loop_joint]), -1)
    np.testing.assert_array_equal(solver._model_plan.loop_joint_articulation, np.array([-1, -1, -1, 0], dtype=np.int32))
    np.testing.assert_array_equal(solver._model_plan.articulation_joint_end, np.array([2, 3], dtype=np.int32))
    np.testing.assert_array_equal(solver._connect_art.numpy(), np.array([0], dtype=np.int32))


def check_standalone_world_root_is_not_loop_joint(test, device, **solver_kwargs):
    """An unrelated unowned world-root joint is not a closure."""
    b, standalone_joint = _build_standalone_world_root()
    model = b.finalize(device=device)
    test.assertEqual(model.joint_articulation.numpy().tolist(), [0, -1])
    test.assertEqual(standalone_joint, 1)
    solver = SolverSteel(model, solve_mode="matrix_free", **solver_kwargs)
    test.assertEqual(solver._connect_count, 0)
    np.testing.assert_array_equal(solver._model_plan.loop_joint_articulation, np.array([-1, -1], dtype=np.int32))


def test_standalone_world_root_is_not_loop_joint(test, device):
    """Do not treat an unowned world-root joint as a closure on the default path."""
    check_standalone_world_root_is_not_loop_joint(test, device)


def test_propagation_standalone_world_root_is_not_loop_joint(test, device):
    """Do not treat an unowned world-root joint as a closure with propagation responses."""
    check_standalone_world_root_is_not_loop_joint(test, device, articulated_contact_response="propagation")


def test_connect_survives_foreign_joint_between_tree_and_loop(test, device):
    """Attribute a loop joint by body ownership when another articulation's joint precedes it.

    Index-range attribution would assign the closure to the free object's articulation
    and leave the four-bar open.
    """
    model = _build_interleaved_four_bar().finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    test.assertEqual(solver._connect_count, 1, "closure row lost: loop joint attributed to the wrong articulation")

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(360):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    test.assertLess(gap_max, 2.0e-3, f"loop closure not enforced (anchor gap {gap_max:.4f} m)")


def test_unsupported_loop_joints_are_rejected(test, device):
    """Reject loop joints other than BALL and REVOLUTE, and closures between two dynamic articulations."""
    b = _build_four_bar()
    _expect_one_warning(
        test,
        newton.exceptions.NewtonWarning,
        "another joint already connects these bodies",
        lambda: b.add_joint_fixed(parent=1, child=2),
    )
    with test.assertRaisesRegex(NotImplementedError, "only BALL and REVOLUTE loop-closing joints"):
        SolverSteel(b.finalize(device=device), solve_mode="matrix_free")

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    first = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    second = b.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
    for body in (first, second):
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        b.add_joint_ball(parent=first, child=second)
    with test.assertRaisesRegex(NotImplementedError, "connects articulations"):
        SolverSteel(b.finalize(device=device), solve_mode="matrix_free")


def test_closure_row_overflow_is_reported(test, device):
    """Drop a closure that does not fit whole, and flag the overflow."""
    model = _build_four_bar().finalize(device=device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        articulation_constraint_max=2,
        joint_limit_activation_gap=0.0,
        warn_constraint_overflow=False,
    )
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertEqual(int(solver._connect_slot.numpy()[0]), -1)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 0)
    test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))


def _anchor_gaps(model, state, joints):
    """World-space parent/child anchor distances [m] of the model's loop joint frames."""
    bq = state.body_q.numpy().astype(np.float64)
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    xp = model.joint_X_p.numpy()
    xc = model.joint_X_c.numpy()

    def anchor(body, local):
        if body < 0:
            return np.asarray(local, dtype=np.float64)
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        w = wp.transform_point(t, wp.vec3(*local))
        return np.array([w[0], w[1], w[2]])

    return [float(np.linalg.norm(anchor(int(jp[j]), xp[j][:3]) - anchor(int(jc[j]), xc[j][:3]))) for j in joints]


def _set_joint_enabled(model, joints, enabled):
    """Write ``Model.joint_enabled`` in place, without a notification."""
    flags = model.joint_enabled.numpy()
    flags[np.asarray(joints, dtype=np.int64)] = enabled
    model.joint_enabled.assign(flags)


def _prescribe_carrier(model, state, articulation, t, v, omega):
    """Write the carrier's free-joint pose/velocity for time ``t`` and refresh its FK."""
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    angle = float(np.linalg.norm(omega)) * t
    axis = np.asarray(omega, dtype=np.float64) / max(float(np.linalg.norm(omega)), 1.0e-12)
    rot = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
    q[0:3] = np.array([0.0, 0.0, 1.0]) + np.asarray(v) * t
    q[3:7] = [rot[0], rot[1], rot[2], rot[3]]
    qd[0:3] = v
    qd[3:6] = omega
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=articulation)


def test_kinematic_parent_closure_carries_load(test, device):
    """Carry a load pinned to a moving kinematic carrier along with it.

    The three point closures hold the load at its relative pose while the carrier
    translates and spins; the closure target carries the carrier's anchor velocity.
    """
    b, carrier, load, joints = _build_carried_load()
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    articulation = wp.array([int(solver._body_to_articulation.numpy()[carrier])], dtype=wp.int32, device=device)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    dt = 1.0 / 240.0
    v = np.array([0.3, 0.0, 0.1])
    omega = np.array([0.0, 0.0, 1.5])
    for i in range(240):
        _prescribe_carrier(model, state_0, articulation, i * dt, v, omega)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    _prescribe_carrier(model, state_0, articulation, 240 * dt, v, omega)

    bq = state_0.body_q.numpy().astype(np.float64)
    test.assertTrue(np.isfinite(bq).all())
    carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
    expected = wp.transform_point(carrier_t, wp.vec3(*_REL_P))
    error = np.linalg.norm(bq[load, :3] - np.array([expected[0], expected[1], expected[2]]))
    test.assertLess(error, 3.0e-3, f"load lagged the carrier by {error:.4f} m")
    for gap in _anchor_gaps(model, state_0, joints):
        test.assertLess(gap, 3.0e-3)
    # The carrier actually moved, so the closure tracked a nonzero target velocity.
    test.assertGreater(np.linalg.norm(bq[carrier, :3] - np.array([0.0, 0.0, 1.0])), 0.25)


def test_world_parent_closure_holds_hanging_load(test, device):
    """Hold a load hanging from world anchors through three point closures."""
    b, _, load, joints = _build_carried_load(world_parent=True)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(240):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    bq = state_0.body_q.numpy().astype(np.float64)
    test.assertTrue(np.isfinite(bq).all())
    test.assertLess(np.linalg.norm(bq[load, :3] - (np.array([0.0, 0.0, 1.0]) + _REL_P)), 3.0e-3)
    for gap in _anchor_gaps(model, state_0, joints):
        test.assertLess(gap, 3.0e-3)


def test_model_edits_release_and_reanchor_closures(test, device):
    """Release closures through Model.joint_enabled, re-anchor them through joint_X_p and engage them again."""
    b, carrier, load, joints = _build_carried_load()
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    dt = 1.0 / 240.0

    def run(steps):
        nonlocal state_0, state_1
        for _ in range(steps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0

    _set_joint_enabled(model, joints, False)
    z0 = float(state_0.body_q.numpy()[load, 2])
    run(48)
    bq = state_0.body_q.numpy().astype(np.float64)
    test.assertGreater(z0 - float(bq[load, 2]), 0.15, "released load should free-fall")

    carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
    load_t = wp.transform(wp.vec3(*bq[load, :3]), wp.quat(*bq[load, 3:]))
    rel = wp.transform_multiply(wp.transform_inverse(carrier_t), load_t)
    joint_X_p = model.joint_X_p.numpy()
    for j, c in zip(joints, _CHILD_ANCHORS, strict=True):
        p = wp.transform_point(rel, wp.vec3(*c))
        joint_X_p[j][:3] = (p[0], p[1], p[2])
    model.joint_X_p.assign(joint_X_p)
    _set_joint_enabled(model, joints, True)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    held_z = float(bq[load, 2])
    run(120)
    bq = state_0.body_q.numpy().astype(np.float64)
    test.assertTrue(np.isfinite(bq).all())
    test.assertLess(abs(float(bq[load, 2]) - held_z), 3.0e-3, "re-engaged closure should hold the load")
    for gap in _anchor_gaps(model, state_0, joints):
        test.assertLess(gap, 3.0e-3)

    _set_joint_enabled(model, joints, False)
    run(48)
    test.assertGreater(held_z - float(state_0.body_q.numpy()[load, 2]), 0.1, "released again: must fall")


def test_disabled_loop_joint_starts_released(test, device):
    """Start a loop joint disabled in Model.joint_enabled released, and engage it by enabling it."""
    b, _, load, joints = _build_carried_load(enabled=False, world_parent=True)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    z0 = float(state_0.body_q.numpy()[load, 2])
    for _ in range(48):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertGreater(z0 - float(state_0.body_q.numpy()[load, 2]), 0.15)
    test.assertEqual(solver._connect_slot.numpy().tolist(), [-1, -1, -1])

    _set_joint_enabled(model, joints, True)
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertTrue(np.all(solver._connect_slot.numpy() >= 0))


def test_kinematic_parent_turning_dynamic_is_rejected(test, device):
    """Reject a body-flag change that would turn a prescribed-parent closure into a dynamic one.

    The carrier is a kinematic body of a partially dynamic articulation and the closure
    connects it to another articulation's free body. Clearing the carrier's kinematic flag
    would leave a one-way closure between two dynamic articulations, which construction
    rejects, so the notification raises too.
    """
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = b.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    carrier = b.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
    b.add_articulation([b.add_joint_revolute(-1, root), b.add_joint_prismatic(-1, carrier, axis=newton.Axis.X)])
    load = b.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    loop_joint = b.add_joint_ball(carrier, load)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", joint_limit_activation_gap=0.0)
    test.assertEqual(solver._connect_parent_prescribed.numpy().tolist(), [1])
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 0.01)

    # Unrelated flag changes and an unchanged carrier are accepted.
    flags = model.body_flags.numpy()
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    flags[root] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    flags[root] = int(newton.BodyFlags.DYNAMIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)

    flags[carrier] = int(newton.BodyFlags.DYNAMIC)
    model.body_flags.assign(flags)
    with test.assertRaisesRegex(NotImplementedError, f"loop-closing joint {loop_joint} .* no longer kinematic"):
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
    with test.assertRaisesRegex(NotImplementedError, "connects articulations 0 and 1"):
        SolverSteel(model, solve_mode="matrix_free", joint_limit_activation_gap=0.0)


def test_imported_connect_equality_is_enforced(test, device):
    """Enforce an imported MuJoCo CONNECT through its converted loop joint; reject it unconverted."""
    mjcf = """
    <mujoco>
      <worldbody>
        <body name="link" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="box" pos="0.3 0 0" size="0.3 0.05 0.05"/>
        </body>
      </worldbody>
      <equality><connect body1="link" anchor="0.6 0 0"/></equality>
    </mujoco>
    """
    for convert in (True, False):
        b = newton.ModelBuilder()
        if convert:
            # The converted CONNECT becomes a BALL loop joint parallel to the hinge.
            _expect_one_warning(
                test,
                newton.exceptions.NewtonWarning,
                "another joint already connects these bodies",
                functools.partial(b.add_mjcf, mjcf, convert_mjc_equality_constraints=True),
            )
        else:
            b.add_mjcf(mjcf, convert_mjc_equality_constraints=False)
        model = b.finalize(device=device)
        test.assertEqual(model.mujoco.equality_constraint_count, 1)
        if not convert:
            with test.assertRaisesRegex(NotImplementedError, "equality"):
                SolverSteel(model, solve_mode="matrix_free")
            continue
        solver = SolverSteel(model, solve_mode="matrix_free", iterations=16)
        test.assertEqual(solver._connect_count, 1)
        state_0, state_1 = model.state(), model.state()
        for _ in range(120):
            solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        # The closure at the free end holds the hinge against gravity.
        test.assertLess(abs(float(state_0.joint_q.numpy()[0])), 1.0e-2)


_LOOP_JOINT = 3  # The four-bar's BALL loop joint.


def _four_bar_setup(device, world_count=1, **solver_kwargs):
    """Return a driven four-bar model (``world_count`` worlds), its solver, states and control."""
    four_bar = _build_four_bar()
    if world_count == 1:
        builder = four_bar
    else:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(four_bar, world_count)
    model = builder.finalize(device=device)
    options = {"solve_mode": "matrix_free", "iterations": 16, "baumgarte": 0.1, **solver_kwargs}
    solver = SolverSteel(model, **options)
    control = model.control()
    targets = model.joint_target_q.numpy().reshape(world_count, -1)
    targets[:, 0] = 0.6
    control.joint_target_q.assign(targets.reshape(-1))
    return model, solver, model.state(), model.state(), control


def _step_four_bar(solver, state_0, state_1, control, steps):
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0, state_1


def _four_bar_gaps(model, state, world_count):
    """Loop anchor gap of every world's four-bar [m]."""
    joints_per_world = model.joint_count // world_count
    return np.array(_anchor_gaps(model, state, [w * joints_per_world + _LOOP_JOINT for w in range(world_count)]))


def check_in_place_release(test, device, **solver_kwargs):
    """Close the four-bar, then release its closure through an in-place Model.joint_enabled edit."""
    model, solver, state_0, state_1, control = _four_bar_setup(device, **solver_kwargs)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 240)
    test.assertLess(_four_bar_gaps(model, state_0, 1)[0], 2.0e-3)
    _set_joint_enabled(model, [_LOOP_JOINT], False)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 120)
    test.assertEqual(int(solver._connect_slot.numpy()[0]), -1)
    test.assertGreater(_four_bar_gaps(model, state_0, 1)[0], 1.0e-2, "released closure still enforced")
    _set_joint_enabled(model, [_LOOP_JOINT], True)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 480)
    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    test.assertLess(_four_bar_gaps(model, state_0, 1)[0], 2.0e-3, "re-enabled closure did not close")


def test_in_place_release(test, device):
    """Release and re-engage a four-bar closure by editing Model.joint_enabled in place."""
    check_in_place_release(test, device)


def test_in_place_release_with_preelimination(test, device):
    """Release and re-engage a pre-eliminated four-bar closure through Model.joint_enabled."""
    check_in_place_release(test, device, enable_bilateral_preelimination=True)


def test_world_parent_release(test, device):
    """Drop a load hung from world anchors once its closures are disabled in the model."""
    b, _, load, joints = _build_carried_load(world_parent=True)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, model.control(), 60)
    z_held = float(state_0.body_q.numpy()[load, 2])
    _set_joint_enabled(model, joints, False)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, model.control(), 48)
    test.assertGreater(z_held - float(state_0.body_q.numpy()[load, 2]), 0.15)


def test_reanchor_world_parent_closure(test, device):
    """Move a world-parent closure's anchors through joint_X_p and notify_model_changed."""
    b, _, load, joints = _build_carried_load(world_parent=True)
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    shift = np.array([0.05, 0.0, 0.0])
    joint_X_p = model.joint_X_p.numpy()
    joint_X_p[joints, :3] += shift
    model.joint_X_p.assign(joint_X_p)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, model.control(), 360)
    expected = np.array([0.0, 0.0, 1.0]) + _REL_P + shift
    test.assertLess(np.linalg.norm(state_0.body_q.numpy()[load, :3] - expected), 3.0e-3)
    for gap in _anchor_gaps(model, state_0, joints):
        test.assertLess(gap, 3.0e-3)


@wp.kernel
def _release_even_worlds(loop_joint: int, joints_per_world: int, joint_enabled: wp.array[bool]):
    world = wp.tid()
    joint_enabled[world * joints_per_world + loop_joint] = world % 2 != 0


def test_captured_device_release(test, device):
    """Release half of 64 worlds' closures from a kernel between replays of one captured graph."""
    world_count = 64
    model, solver, state_0, state_1, control = _four_bar_setup(device, world_count=world_count)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 2)
    with wp.ScopedCapture(device=device) as capture:
        _step_four_bar(solver, state_0, state_1, control, 2)
    for _ in range(120):
        wp.capture_launch(capture.graph)
    test.assertLess(float(_four_bar_gaps(model, state_0, world_count).max()), 2.0e-3)
    wp.launch(
        _release_even_worlds,
        dim=world_count,
        inputs=[_LOOP_JOINT, model.joint_count // world_count],
        outputs=[model.joint_enabled],
        device=device,
    )
    for _ in range(60):
        wp.capture_launch(capture.graph)
    gaps = _four_bar_gaps(model, state_0, world_count)
    test.assertGreater(float(gaps[0::2].min()), 1.0e-2, "released worlds stayed closed")
    test.assertLess(float(gaps[1::2].max()), 2.0e-3, "enforced worlds opened")
    slots = solver._connect_slot.numpy()
    test.assertTrue(np.all(slots[0::2] == -1))
    test.assertTrue(np.all(slots[1::2] >= 0))


def test_replaced_joint_enabled_array(test, device):
    """Read a replaced Model.joint_enabled array eagerly, and after recapturing a graph."""
    model, solver, state_0, state_1, control = _four_bar_setup(device)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 240)
    released = model.joint_enabled.numpy()
    released[_LOOP_JOINT] = False
    model.joint_enabled = wp.array(released, dtype=bool, device=device)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 120)
    test.assertGreater(_four_bar_gaps(model, state_0, 1)[0], 1.0e-2)

    # A graph records the array it was captured with; recapture after replacing it.
    model.joint_enabled = wp.array(np.ones(model.joint_count, dtype=bool), dtype=bool, device=device)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    with wp.ScopedCapture(device=device) as capture:
        _step_four_bar(solver, state_0, state_1, control, 2)
    for _ in range(240):
        wp.capture_launch(capture.graph)
    test.assertLess(_four_bar_gaps(model, state_0, 1)[0], 2.0e-3)


def test_release_matches_xpbd(test, device):
    """Release world-anchored closures like SolverXPBD: both hold the load, then let it fall alike."""
    dt = 1.0 / 240.0
    trajectories = {}
    for name in ("steel", "xpbd"):
        b, _, load, joints = _build_carried_load(world_parent=True)
        model = b.finalize(device=device)
        if name == "steel":
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
        else:
            solver = newton.solvers.SolverXPBD(model, iterations=8)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 60)
        z = [float(state_0.body_q.numpy()[load, 2])]
        _set_joint_enabled(model, joints, False)
        for _ in range(24):
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
            z.append(float(state_0.body_q.numpy()[load, 2]))
        trajectories[name] = np.asarray(z)
    held = 1.0 + _REL_P[2]
    for name, z in trajectories.items():
        test.assertAlmostEqual(z[0], held, delta=3.0e-3, msg=name)
        test.assertGreater(z[0] - z[-1], 0.02, f"{name}: released load did not fall")
    np.testing.assert_allclose(trajectories["steel"], trajectories["xpbd"], atol=3.0e-3)


def test_closure_owner_stays_awake_and_releases(test, device):
    """Keep a closure's articulation awake under sleeping, so a released closure takes effect."""
    b = _build_four_bar()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    model = b.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=16,
        baumgarte=0.1,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
    )
    control = model.control()
    targets = model.joint_target_q.numpy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    states = [model.state(), model.state()]

    def advance(steps):
        for _ in range(steps):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
            states.reverse()

    advance(480)
    # The box sleeps; the four-bar owns closure rows and stays awake.
    test.assertEqual(solver._sleeping.art_awake.numpy().tolist(), [1, 0])
    test.assertLess(_four_bar_gaps(model, states[0], 1)[0], 2.0e-3)
    _set_joint_enabled(model, [_LOOP_JOINT], False)
    advance(120)
    test.assertGreater(_four_bar_gaps(model, states[0], 1)[0], 1.0e-2)


def test_propagation_rejects_closures(test, device):
    """Reject loop closures with propagation contact responses, enabled or released."""
    for enabled in (True, False):
        model = _build_four_bar().finalize(device=device)
        _set_joint_enabled(model, [_LOOP_JOINT], enabled)
        with test.assertRaisesRegex(NotImplementedError, "Loop-closing joints are not supported"):
            SolverSteel(model, solve_mode="matrix_free", articulated_contact_response="propagation")


def test_disabled_tree_joint_is_reported(test, device):
    """Report a tree joint disabled after construction, eagerly and under capture, until reset."""
    world_count = 4
    model, solver, state_0, state_1, control = _four_bar_setup(device, world_count=world_count)
    joints_per_world = model.joint_count // world_count
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 2)
    solver.check_constraint_capacity()
    _set_joint_enabled(model, [2 * joints_per_world + 1], False)
    state_0, state_1 = _step_four_bar(solver, state_0, state_1, control, 1)
    with test.assertRaisesRegex(RuntimeError, r"tree joint in worlds \[2\]"):
        solver.check_constraint_capacity()
    _set_joint_enabled(model, [2 * joints_per_world + 1], True)
    mask = wp.array([False, False, True, False, False], dtype=wp.bool, device=device)
    solver.reset(state_0, world_mask=mask)
    solver.check_constraint_capacity()

    with wp.ScopedCapture(device=device) as capture:
        _step_four_bar(solver, state_0, state_1, control, 2)
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    _set_joint_enabled(model, [joints_per_world], False)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, r"tree joint in worlds \[1\]"):
        solver.check_constraint_capacity()

    with test.assertRaisesRegex(NotImplementedError, "disabled joints in an articulation tree"):
        SolverSteel(model, solve_mode="matrix_free")


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


def _build_floating_four_bar():
    """Build a free-floating planar four-bar, without gravity, whose crank is driven against a locked rocker."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    z0 = 0.6
    base = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0), wp.quat_identity()))
    b.add_shape_box(base, hx=0.25, hy=0.02, hz=0.02)
    j_free = b.add_joint_free(child=base)
    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=base,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=base,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        limit_lower=0.0,
        limit_upper=0.0,
    )
    b.add_articulation([j_free, j_crank, j_coupler, j_rocker])
    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )
    crank_dof, crank_coord = 6, 7
    b.joint_target_ke[crank_dof] = 50.0
    b.joint_target_kd[crank_dof] = 1.0
    b.joint_target_mode[crank_dof] = int(newton.JointTargetMode.POSITION)
    b.joint_target_q[crank_coord] = 0.8
    return b


def _angular_momentum(model, state) -> np.ndarray:
    """Return the system's angular momentum about the world origin [kg m^2/s]."""
    mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
    body_q, body_qd = state.body_q.numpy(), state.body_qd.numpy()
    total = np.zeros(3)
    for b in range(model.body_count):
        rotation = np.array(wp.quat_to_matrix(wp.quat(*body_q[b, 3:]))).reshape(3, 3)
        x_com = body_q[b, :3] + rotation @ com[b]
        total += np.cross(x_com, mass[b] * body_qd[b, :3]) + rotation @ inertia[b] @ rotation.T @ body_qd[b, 3:]
    return total


def test_open_closure_on_floating_base_conserves_angular_momentum(test, device):
    """Keep a free-floating mechanism's angular momentum while its loop closure is loaded open."""
    model = _build_floating_four_bar().finalize(device=device)
    solver = SolverSteel(model, iterations=2, enable_joint_limits=True)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(300):
        solver.step(state_0, state_1, control, None, 1.0 / 500.0)
        state_0, state_1 = state_1, state_0
    test.assertGreater(_loop_anchor_gap(model, state_0), 5.0e-3, "the closure should be loaded open")
    test.assertLess(np.linalg.norm(_angular_momentum(model, state_0)), 1.0e-3)


def test_open_world_closure_pulls_at_child_anchor(test, device):
    """Pull a free load toward an offset world anchor through its center of mass without spinning it."""
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    load = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
    # The child anchor is the load's center of mass; the world anchor sits 5 cm to the side.
    _expect_one_warning(
        test,
        newton.exceptions.NewtonWarning,
        "another joint already connects these bodies",
        lambda: b.add_joint_ball(
            parent=-1, child=load, parent_xform=wp.transform(wp.vec3(0.05, 0.0, 1.0), wp.quat_identity())
        ),
    )
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    spin = 0.0
    for _ in range(120):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        spin = max(spin, float(np.linalg.norm(state_0.body_qd.numpy()[load, 3:])))
    test.assertLess(np.linalg.norm(state_0.body_q.numpy()[load, :3] - np.array([0.05, 0.0, 1.0])), 3.0e-3)
    test.assertLess(spin, 1.0e-3)


def test_open_closure_from_fixed_root_pulls_at_child_anchor(test, device):
    """Pull a pendulum load toward an anchor on its fixed root through its center of mass without spinning it."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root_xform = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
    root = b.add_link(xform=root_xform)
    b.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    load = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
    b.add_articulation(
        [
            b.add_joint_fixed(parent=-1, child=root, parent_xform=root_xform),
            b.add_joint_ball(
                parent=root, child=load, child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity())
            ),
        ]
    )
    # The child anchor is the load's center of mass; the root anchor is 5 cm away and out of the rod's reach.
    _expect_one_warning(
        test,
        newton.exceptions.NewtonWarning,
        "another joint already connects these bodies",
        lambda: b.add_joint_ball(
            parent=root, child=load, parent_xform=wp.transform(wp.vec3(0.03, 0.04, -0.2), wp.quat_identity())
        ),
    )
    model = b.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    spin = 0.0
    for _ in range(120):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        body_q, body_qd = state_0.body_q.numpy(), state_0.body_qd.numpy()
        rod_axis = np.array(wp.quat_to_matrix(wp.quat(*body_q[load, 3:]))).reshape(3, 3)[:, 2]
        spin = max(spin, abs(float(rod_axis @ body_qd[load, 3:])))
    test.assertGreater(_loop_anchor_gap(model, state_0), 5.0e-3, "the closure should stay open")
    test.assertLess(spin, 1.0e-3)


class TestSteelConnect(unittest.TestCase):
    pass


class TestSteelPrescribedParentConnect(unittest.TestCase):
    pass


class TestSteelConnectPropagation(unittest.TestCase):
    pass


class TestSteelConnectModelEdits(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _name in (
    "test_loop_joint_presence_is_stable",
    "test_loop_closure_is_enforced",
    "test_rocker_follows_crank",
    "test_four_bar_capture_matches_eager",
    "test_connect_ownership_survives_interleaved_articulation",
    "test_standalone_world_root_is_not_loop_joint",
    "test_connect_survives_foreign_joint_between_tree_and_loop",
    "test_unsupported_loop_joints_are_rejected",
    "test_closure_row_overflow_is_reported",
    "test_imported_connect_equality_is_enforced",
    "test_open_closure_on_floating_base_conserves_angular_momentum",
    "test_open_world_closure_pulls_at_child_anchor",
    "test_open_closure_from_fixed_root_pulls_at_child_anchor",
):
    add_function_test(TestSteelConnect, _name, globals()[_name], devices=cuda_devices)


for _name in (
    "test_kinematic_parent_closure_carries_load",
    "test_world_parent_closure_holds_hanging_load",
    "test_model_edits_release_and_reanchor_closures",
    "test_disabled_loop_joint_starts_released",
    "test_kinematic_parent_turning_dynamic_is_rejected",
):
    add_function_test(TestSteelPrescribedParentConnect, _name, globals()[_name], devices=cuda_devices)


for _name in (
    "test_in_place_release",
    "test_in_place_release_with_preelimination",
    "test_world_parent_release",
    "test_reanchor_world_parent_closure",
    "test_captured_device_release",
    "test_replaced_joint_enabled_array",
    "test_release_matches_xpbd",
    "test_closure_owner_stays_awake_and_releases",
    "test_propagation_rejects_closures",
    "test_disabled_tree_joint_is_reported",
):
    add_function_test(TestSteelConnectModelEdits, _name, globals()[_name], devices=cuda_devices)


add_function_test(
    TestSteelConnectPropagation,
    "test_propagation_standalone_world_root_is_not_loop_joint",
    test_propagation_standalone_world_root_is_not_loop_joint,
    devices=cuda_devices,
)


DT = 1.0 / 240.0


class TestPreeliminationSignature(unittest.TestCase):
    def test_options_are_keyword_only(self):
        """Expose the pre-elimination options as keyword-only, with the switch off by default."""
        parameters = inspect.signature(SolverSteel).parameters
        for name, default in (
            ("enable_bilateral_preelimination", False),
            ("bilateral_preelimination_include_mimics", True),
        ):
            with self.subTest(option=name):
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameters[name].default, default)


class TestPreeliminationKernels(unittest.TestCase):
    """Kernel-level contracts of the connect rows and the regularized elimination, on CPU."""

    def test_connect_rows_are_normalized(self):
        """Scale each connect row to a unit Jacobian so a short lever arm does not weaken it."""
        device = "cpu"

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        J = wp.zeros((1, 3, 1), dtype=float, device=device)
        phi = wp.zeros((1, 3), dtype=float, device=device)
        target = wp.zeros((1, 3), dtype=float, device=device)
        # A world anchor and one rotational child DOF with a 2 mm lever arm.
        wp.launch(
            steel_kernels.populate_connect_J_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                1,
                arr([0]),
                arr([0]),
                arr([0]),
                arr([-1]),
                arr([0]),
                arr([3]),
                wp.zeros(1, dtype=wp.vec3, device=device),
                arr([[0.002, 0.001, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.transform),
                arr([[0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.transform),
                arr([1]),
                arr([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.transform),
                wp.zeros(1, dtype=wp.spatial_vector, device=device),
                wp.zeros(1, dtype=wp.vec3, device=device),
                arr([0]),
                arr([-1]),
                arr([0, 1]),
                arr([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], wp.spatial_vector),
                wp.zeros(1, dtype=wp.vec3, device=device),
            ],
            outputs=[
                J,
                wp.zeros((1, 3), dtype=int, device=device),
                wp.zeros((1, 3), dtype=int, device=device),
                wp.zeros((1, 3), dtype=float, device=device),
                phi,
                target,
            ],
            device=device,
        )
        # Unnormalized, the row would be J = -0.002 with phi = 0.001.
        self.assertAlmostEqual(float(J.numpy()[0, 1, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(phi.numpy()[0, 1]), 0.5, places=5)

    def test_regularized_projection_leaves_documented_residual(self):
        """Leave the residual ``R (S + R)^-1 (J_B v + b_B)`` of the regularized block factor.

        With ``H = I``, a bilateral row ``J_B = [1, -1]`` (``S = 2``) and a second row
        ``[1, 0]``, the projection of ``v = [1, 0]`` and the corrected response of the second
        row both keep the factor ``R / (S + R)`` of their bilateral violation, with
        ``R = 1e-3 S + 1e-7``. The elimination is not exact.
        """
        device = "cpu"

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        J = arr([[[1.0, -1.0], [1.0, 0.0]]], float)
        Y = arr(J.numpy(), float)
        slots = wp.full(8, -1, dtype=int, device=device)
        nB = wp.zeros(1, dtype=int, device=device)
        S = wp.zeros(64, dtype=float, device=device)
        reg = wp.zeros(8, dtype=float, device=device)
        LS = wp.zeros(64, dtype=float, device=device)
        reg_rel, reg_floor = 1.0e-3, 1.0e-7
        wp.launch(
            steel_kernels.preelim_setup_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0, 1]),
                arr([0]),
                1,
                arr([-1]),
                arr([-1]),
                arr([3]),
                0,
                J,
                Y,
                2,
                reg_rel,
                reg_floor,
            ],
            outputs=[slots, nB, S, reg, LS],
            device=device,
        )
        v = arr([1.0, 0.0], float)
        wp.launch(
            steel_kernels.preelim_project_velocity_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0]),
                slots,
                nB,
                LS,
                J,
                Y,
                wp.zeros((1, 2), dtype=float, device=device),
                2,
            ],
            outputs=[v],
            device=device,
        )
        wp.launch(
            steel_kernels.preelim_correct_Y_for_size,
            dim=2,
            inputs=[arr([0]), arr([0]), arr([0]), arr([2]), slots, nB, LS, J, 2, 2, 1],
            outputs=[Y],
            device=device,
        )
        s_block = 2.0
        regularizer = reg_rel * s_block + reg_floor
        self.assertAlmostEqual(float(reg.numpy()[0]), regularizer, places=7)
        expected = regularizer / (s_block + regularizer)
        residual = float(v.numpy()[0] - v.numpy()[1])
        leak = float(J.numpy()[0, 0] @ Y.numpy()[0, 1])
        np.testing.assert_allclose([residual, leak], [expected, expected], rtol=1.0e-3)
        self.assertGreater(residual, 0.0)


def _run_four_bar(device, steps: int = 720, crank_target: float = 0.6, iterations: int = 2, **solver_kwargs):
    """Drive the four-bar and track the worst loop-anchor gap.

    Runs at a low iteration count on purpose: with iterative connect rows the anchor gap
    is limited by convergence (about 0.5 mm at 2 iterations), while pre-elimination
    keeps it small at the same budget.
    """
    model = _build_four_bar().finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=iterations, baumgarte=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def _build_overcapacity_mixed_bilateral_model(device):
    """Build one articulation with three connect rows and six mimic rows."""
    builder = _build_four_bar()
    # Six rows on one follower need the deprecated API; joint-owned mimics allow one per joint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for index in range(6):
            builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0, label=f"mimic_{index}")
    return builder.finalize(device=device)


def test_default_preserves_all_bilateral_trajectory(test, device):
    """Match explicit elimination of every bilateral row with the implicit default."""
    builder = _build_four_bar()
    builder.set_joint_mimic(1, 0)
    model = builder.finalize(device=device)
    solvers = [
        SolverSteel(model, solve_mode="matrix_free", iterations=8, enable_bilateral_preelimination=True, **options)
        for options in ({}, {"bilateral_preelimination_include_mimics": True})
    ]
    states = [(model.state(), model.state()) for _ in solvers]
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    for _ in range(32):
        for index, solver in enumerate(solvers):
            state_in, state_out = states[index]
            state_in.clear_forces()
            solver.step(state_in, state_out, control, None, DT)
            states[index] = (state_out, state_in)
        for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
            a, b = [getattr(pair[0], attribute).numpy() for pair in states]
            test.assertTrue(np.isfinite(a).all())
            np.testing.assert_array_equal(a, b)
    for solver in solvers:
        test.assertTrue(solver.bilateral_preelimination_include_mimics)
        test.assertTrue(solver._preelim_active)
        count = int(solver._preelim_nB.numpy()[0])
        slots = solver._preelim_slots.numpy()[:count]
        selected_types = solver._row_type.numpy()[0, slots]
        test.assertEqual(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
        test.assertGreater(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_CONNECT)), 0)


def test_mimics_can_remain_iterative_when_total_rows_exceed_capacity(test, device):
    """Eliminate the connect rows while excess mimic rows stay iterative."""
    model = _build_overcapacity_mixed_bilateral_model(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver_all = SolverSteel(model, solve_mode="matrix_free", enable_bilateral_preelimination=True)
    test.assertFalse(solver_all._preelim_active)
    test.assertTrue(any("9 bilateral rows" in str(w.message) for w in caught))

    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=8,
        enable_bilateral_preelimination=True,
        bilateral_preelimination_include_mimics=False,
    )
    test.assertTrue(solver._preelim_active)
    test.assertEqual(solver._preelim_count, 1)

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(16):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0

    test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
    count = int(solver._preelim_nB.numpy()[0])
    test.assertGreater(count, 0)
    test.assertLessEqual(count, 3)
    selected_types = solver._row_type.numpy()[0, solver._preelim_slots.numpy()[:count]]
    np.testing.assert_array_equal(selected_types, np.full(count, PGS_CONSTRAINT_TYPE_CONNECT))
    counts = solver._constraint_count.numpy()
    rows = solver._row_type.numpy()
    seen = {int(row) for world in range(rows.shape[0]) for row in rows[world, : counts[world]]}
    test.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)
    test.assertIn(int(PGS_CONSTRAINT_TYPE_MIMIC), seen)


def test_closure_held_at_low_iterations(test, device):
    """Hold the loop closure tight at a low iteration count with pre-elimination.

    At 2 iterations the iterative connect rows cannot converge (the anchor gap stays
    near half a millimetre); the regularized elimination holds the closure 100x tighter
    through the same driven stroke.
    """
    solver_off, _, _, gap_off = _run_four_bar(device)
    test.assertFalse(solver_off._preelim_active)
    solver_on, _, state_on, gap_on = _run_four_bar(device, enable_bilateral_preelimination=True)
    test.assertTrue(solver_on._preelim_active)
    test.assertEqual(solver_on._preelim_count, 1)

    test.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
    test.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m too loose")
    test.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")


def test_mimic_held_at_low_iterations(test, device):
    """Hold an eliminated mimic row coupled to a closure at one iteration.

    In the parallel four-bar the coupler joint angle is minus the crank angle; a mimic row
    stating the same relationship competes with the connect rows in an iterative sweep.
    """
    errors = []
    for enabled in (False, True):
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0, coeffs=(0.0, -1.0))
        model = builder.finalize(device=device)
        solver = SolverSteel(
            model, solve_mode="matrix_free", iterations=1, baumgarte=0.1, enable_bilateral_preelimination=enabled
        )
        test.assertEqual(solver._preelim_active, enabled)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        worst = 0.0
        for _ in range(480):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
            q = state_0.joint_q.numpy()
            worst = max(worst, abs(float(q[1] + q[0])))
        test.assertGreater(abs(float(q[0])), 0.5)
        errors.append(worst)
    test.assertGreater(errors[0], 1.0e-5, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(errors[1], 1.0e-6)
    test.assertLess(errors[1], 0.01 * errors[0], f"iterative {errors[0]:.2e} vs eliminated {errors[1]:.2e}")


def test_preelimination_with_free_body_in_world(test, device):
    """Leave an eliminated, contacting four-bar unchanged by a free body elsewhere in its world.

    The four-bar's coupler rests on the ground, so its contact rows are corrected by the
    elimination. Adding a separate free box gives the world two size groups and the
    gathered (not aliased) world response; the correction must reach it.
    """
    trajectories = []
    for with_box in (False, True):
        builder = _build_four_bar()
        # The coupler's underside is at 0.38 m and starts in contact.
        builder.add_ground_plane(height=0.385)
        if with_box:
            box = builder.add_body(xform=wp.transform(wp.vec3(0.8, 0.0, 0.585), wp.quat_rpy(0.2, 0.1, 0.3)))
            builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.06)
        model = builder.finalize(device=device)
        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            iterations=2,
            baumgarte=0.1,
            articulation_constraint_max=64,
            enable_bilateral_preelimination=True,
        )
        test.assertTrue(solver._preelim_active)
        test.assertEqual(solver._jy_world_aliased, not with_box)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        trajectory = []
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            trajectory.append(state_0.joint_q.numpy()[:3].copy())
        test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        test.assertFalse(bool(solver.constraint_overflow.numpy().any()))
        trajectories.append(np.asarray(trajectory))
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1.0e-4)


def test_mechanism_behavior_preserved(test, device):
    """Keep the rocker tracking the crank through the closed loop (parallel four-bar)."""
    _, _, state, _ = _run_four_bar(device, enable_bilateral_preelimination=True, iterations=16)
    q = state.joint_q.numpy()
    test.assertAlmostEqual(q[2], q[0], delta=0.08)
    test.assertGreater(abs(q[2]), 0.3, "rocker did not move: loop not transmitting")


def test_rows_remain_allocated(test, device):
    """Keep eliminated rows in their dense slots and in the sweep."""
    solver, _, _, _ = _run_four_bar(device, steps=60, enable_bilateral_preelimination=True)
    counts = solver._constraint_count.numpy()
    rows = solver._row_type.numpy()
    seen = set()
    for w in range(rows.shape[0]):
        seen.update(rows[w, : counts[w]].tolist())
    test.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)


def test_prescribed_parent_closure_warns_and_falls_back(test, device):
    """Disable elimination for the whole solver when a closure has a kinematic or world parent."""
    b, _, _, _ = _build_carried_load()
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "kinematic or world parent.*whole solver"):
        solver = SolverSteel(b.finalize(device=device), solve_mode="matrix_free", enable_bilateral_preelimination=True)
    test.assertFalse(solver._preelim_active)
    test.assertEqual(solver._connect_count, 3)

    # An eligible four-bar is eliminated alone, but not next to a world-parent closure.
    eligible = SolverSteel(
        _build_four_bar().finalize(device=device), solve_mode="matrix_free", enable_bilateral_preelimination=True
    )
    test.assertTrue(eligible._preelim_active)
    b = _build_four_bar()
    load = b.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*another joint already connects these bodies")
        b.add_joint_ball(-1, load, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "whole solver"):
        mixed = SolverSteel(b.finalize(device=device), solve_mode="matrix_free", enable_bilateral_preelimination=True)
    test.assertFalse(mixed._preelim_active)
    test.assertEqual(mixed._preelim_count, 0)
    test.assertEqual(mixed._connect_count, 2)


def test_preelimination_capture_matches_eager(test, device):
    """Replay the pre-eliminated four-bar's eager trajectory from a CUDA graph."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    options = {"iterations": 2, "baumgarte": 0.1, "enable_bilateral_preelimination": True}
    eager, captured = (
        SolverSteel(model, solve_mode="matrix_free", **options),
        SolverSteel(model, solve_mode="matrix_free", **options),
    )
    e0, e1 = model.state(), model.state()
    c0, c1 = model.state(), model.state()

    def step_pair(solver, s0, s1):
        solver.step(s0, s1, control, None, DT)
        solver.step(s1, s0, control, None, DT)

    step_pair(captured, c0, c1)
    step_pair(eager, e0, e1)
    # Capturing records the launches without running them.
    with wp.ScopedCapture(device=device) as capture:
        step_pair(captured, c0, c1)
    for _ in range(60):
        wp.capture_launch(capture.graph)
        step_pair(eager, e0, e1)
    np.testing.assert_allclose(c0.joint_q.numpy(), e0.joint_q.numpy(), atol=2.0e-5)
    test.assertLess(_loop_anchor_gap(model, c0), 2.0e-5)


def _run_loaded_four_bar(device, contact_warmstart):
    """Rest a 1 kg box on an eliminated four-bar's coupler; return the solver, worst gap and carried impulse."""
    builder = _build_four_bar()
    box = builder.add_body(xform=wp.transform(wp.vec3(0.2, 0.0, 0.4705), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    model = builder.finalize(device=device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=1,
        baumgarte=0.1,
        articulation_constraint_max=64,
        friction_anchor_beta=0.0,
        enable_bilateral_preelimination=True,
        contact_warmstart=contact_warmstart,
    )
    pipeline = newton.CollisionPipeline(model, contact_matching="latest")
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    gap_max = carried = 0.0
    for step in range(240):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        if step >= 20:
            gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
        if contact_warmstart:
            history = solver._ws_prev_impulses.numpy()[0]
            contact_rows = solver._ws_prev_row_type.numpy()[0] == PGS_CONSTRAINT_TYPE_CONTACT
            carried = max(carried, float(np.abs(history[contact_rows]).max(initial=0.0)))
    return solver, model, state_0, box, gap_max, carried


def test_warmstart_with_preelimination_holds_a_loaded_closure(test, device):
    """Warm start a contact that loads an eliminated closure; the closure stays tight and the box rests.

    The carried contact impulse is installed into the velocity before the bilateral projection,
    so it cannot reopen the closure.
    """
    solver, model, state, box, gap_warm, carried = _run_loaded_four_bar(device, contact_warmstart=True)
    _, _, _, _, gap_cold, _ = _run_loaded_four_bar(device, contact_warmstart=False)
    test.assertTrue(solver._preelim_active)
    # The carried contact history is the box's weight impulse, m g dt.
    weight_impulse = float(model.body_mass.numpy()[box]) * 9.81 * DT
    test.assertAlmostEqual(carried, weight_impulse, delta=0.2 * weight_impulse)
    test.assertLess(gap_warm, 1.0e-6)
    test.assertLessEqual(gap_warm, gap_cold)
    test.assertAlmostEqual(float(state.body_q.numpy()[box, 2]), 0.47, delta=1.0e-3)


def test_split_rejects_bilateral_rows(test, device):
    """Reject loop closures in the split solve, with or without pre-elimination."""
    model = _build_four_bar().finalize(device=device)
    for enabled in (False, True):
        with test.subTest(enable_bilateral_preelimination=enabled):
            with test.assertRaisesRegex(NotImplementedError, "require solve_mode='matrix_free'"):
                SolverSteel(model, solve_mode="split", enable_bilateral_preelimination=enabled)


def test_propagation_keeps_iterative_mimic_rows(test, device):
    """Keep iterative mimic rows with the propagation responses; pre-elimination falls back with a warning."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "propagation"):
                solver = SolverSteel(model, articulated_contact_response=response, enable_bilateral_preelimination=True)
            test.assertFalse(solver._preelim_active)
            test.assertGreater(solver._mimic_count, 0)


def test_propagation_rejects_loop_joints(test, device):
    """Reject loop-closing joints with the propagation responses."""
    model = _build_four_bar().finalize(device=device)
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(NotImplementedError, "Loop-closing joints.*articulated_contact_response"):
                SolverSteel(model, articulated_contact_response=response)


class TestSteelPreelimination(unittest.TestCase):
    pass


class TestSteelPreeliminationSplit(unittest.TestCase):
    pass


class TestSteelPreeliminationPropagation(unittest.TestCase):
    pass


for _name in (
    "test_default_preserves_all_bilateral_trajectory",
    "test_mimics_can_remain_iterative_when_total_rows_exceed_capacity",
    "test_closure_held_at_low_iterations",
    "test_mimic_held_at_low_iterations",
    "test_preelimination_with_free_body_in_world",
    "test_mechanism_behavior_preserved",
    "test_rows_remain_allocated",
    "test_prescribed_parent_closure_warns_and_falls_back",
    "test_preelimination_capture_matches_eager",
    "test_warmstart_with_preelimination_holds_a_loaded_closure",
):
    add_function_test(TestSteelPreelimination, _name, globals()[_name], devices=cuda_devices)


add_function_test(
    TestSteelPreeliminationSplit,
    "test_split_rejects_bilateral_rows",
    test_split_rejects_bilateral_rows,
    devices=get_test_devices(),
)


add_function_test(
    TestSteelPreeliminationPropagation,
    "test_propagation_keeps_iterative_mimic_rows",
    test_propagation_keeps_iterative_mimic_rows,
    devices=cuda_devices,
)


add_function_test(
    TestSteelPreeliminationPropagation,
    "test_propagation_rejects_loop_joints",
    test_propagation_rejects_loop_joints,
    devices=cuda_devices,
)


def _build_mimic_layout(joint_type):
    """Build a joint-owned mimic pair after a free base in one articulation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_free(child=base)]
    parent = base
    for _ in range(2):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        linear = []
        angular = []
        if joint_type == newton.JointType.PRISMATIC:
            linear = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
        elif joint_type == newton.JointType.REVOLUTE:
            angular = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)]
        elif joint_type == newton.JointType.D6:
            linear = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
            angular = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
        if joint_type == newton.JointType.BALL:
            joint = builder.add_joint_ball(parent=parent, child=child)
        elif joint_type == newton.JointType.FREE:
            joint = builder.add_joint_free(parent=parent, child=child)
        elif joint_type == newton.JointType.DISTANCE:
            joint = builder.add_joint_distance(parent=parent, child=child)
        else:
            joint = builder.add_joint(joint_type, parent=parent, child=child, linear_axes=linear, angular_axes=angular)
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)
    if joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
        # Bypass the authoring guard to exercise defensive solver handling of
        # unsupported metadata supplied directly by callers.
        builder.joint_mimic_joint[joints[2]] = joints[1]
        builder.joint_mimic_coeffs[joints[2]] = (0.1, -0.5)
    else:
        builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.1, -0.5))
    return builder, joints[1], joints[2]


def _run_model(model, leader_target: float, steps: int, **solver_kwargs):
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = leader_target
    control.joint_target_q.assign(targets)
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy()


def _run_chain(device, coef0: float, coef1: float, leader_target: float, steps: int = 600, legacy: bool = False):
    builder, _, _ = _build_two_revolute_chain(coef0, coef1, legacy=legacy)
    return _run_model(builder.finalize(device=device), leader_target, steps)


def check_reject_quaternion_mimics_at_construction(test, device, **solver_kwargs):
    """Reject joint-owned mimics whose position and velocity layouts differ."""
    for joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
        with test.subTest(joint_type=joint_type):
            builder, _, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            test.assertNotEqual(q_start[follower + 1] - q_start[follower], qd_start[follower + 1] - qd_start[follower])
            with test.assertRaisesRegex(NotImplementedError, "mimic joint .* position and velocity"):
                SolverSteel(model, **solver_kwargs)


def check_scalar_and_d6_mimic_coordinate_maps(test, device, **solver_kwargs):
    """Preserve componentwise mimic maps after a quaternion-layout free base."""
    for joint_type, dimensions in (
        (newton.JointType.REVOLUTE, 1),
        (newton.JointType.PRISMATIC, 1),
        (newton.JointType.D6, 6),
    ):
        with test.subTest(joint_type=joint_type):
            builder, leader, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            solver = SolverSteel(model, **solver_kwargs)
            test.assertEqual(solver._mimic_count, dimensions)
            for suffix, start, joint in (
                ("q0", model.joint_q_start, follower),
                ("q1", model.joint_q_start, leader),
                ("dof0", model.joint_qd_start, follower),
                ("dof1", model.joint_qd_start, leader),
            ):
                expected = np.arange(int(start.numpy()[joint]), int(start.numpy()[joint]) + dimensions)
                np.testing.assert_array_equal(getattr(solver, f"_mimic_{suffix}").numpy(), expected)
            np.testing.assert_array_equal(solver._mimic_legacy.numpy(), np.full(dimensions, -1))


def check_invalid_legacy_mimic_keeps_precedence(test, device, **solver_kwargs):
    """A legacy mimic replaces the joint-owned mimic of its follower, also when it is rejected."""
    builder, leader, follower = _build_mimic_layout(newton.JointType.BALL)
    with test.assertWarns(DeprecationWarning):
        builder.add_constraint_mimic(joint0=follower, joint1=leader)
    model = builder.finalize(device=device)
    # The legacy entry is checked instead of the joint-owned BALL mimic it replaces.
    with test.assertRaisesRegex(NotImplementedError, "must couple REVOLUTE or PRISMATIC"):
        SolverSteel(model, **solver_kwargs)


def test_reject_quaternion_mimics_at_construction(test, device):
    """Reject mimics of BALL, FREE and DISTANCE joints at construction."""
    check_reject_quaternion_mimics_at_construction(test, device, solve_mode="matrix_free")


def test_scalar_and_d6_mimic_coordinate_maps(test, device):
    """Map scalar and D6 mimic followers to their coordinate rows."""
    check_scalar_and_d6_mimic_coordinate_maps(test, device, solve_mode="matrix_free")


def test_invalid_legacy_mimic_keeps_precedence(test, device):
    """Validate a legacy mimic entry in place of the joint-owned mimic it replaces."""
    check_invalid_legacy_mimic_keeps_precedence(test, device, solve_mode="matrix_free")


def test_cross_articulation_mimic_is_rejected(test, device):
    """Reject a mimic between two articulations instead of ignoring it."""
    template, j_leader, j_follower = _build_two_revolute_chain(0.0, 1.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_builder(template)
    builder.add_builder(template)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # The follower of the second copy mimics the leader of the first copy.
        builder.add_constraint_mimic(joint0=j_follower + 2, joint1=j_leader, coef0=0.0, coef1=1.0)
    with test.assertRaisesRegex(NotImplementedError, "different articulations"):
        SolverSteel(builder.finalize(device=device), solve_mode="matrix_free")


def test_identity_mimic_tracks_leader(test, device):
    """Track the driven leader with the undriven follower joint under a 1:1 mimic."""
    _, q = _run_chain(device, coef0=0.0, coef1=1.0, leader_target=0.5)
    test.assertAlmostEqual(q[0], 0.5, delta=0.05)
    test.assertAlmostEqual(q[1], q[0], delta=0.02)


def test_legacy_constraint_mimic_tracks_leader(test, device):
    """Couple the follower through the deprecated mimic constraints."""
    _, q = _run_chain(device, coef0=0.1, coef1=-0.5, leader_target=0.6, legacy=True)
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_legacy_constraint_overrides_joint_mimic(test, device):
    """Replace the follower's joint-owned mimic with a legacy constraint on it, as in SolverMuJoCo."""
    builder, j_leader, j_follower = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-0.5)
    solver, q = _run_model(builder.finalize(device=device), leader_target=0.6, steps=600)
    test.assertEqual(solver._mimic_count, 1)
    test.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_runtime_coefficient_change(test, device):
    """Read joint-owned coefficients every step, also from a captured graph."""
    builder, _, follower = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)

    def simulate():
        for _ in range(2):
            solver.step(state_0, state_1, control, None, DT)
            solver.step(state_1, state_0, control, None, DT)

    simulate()
    with wp.ScopedCapture(device=device) as capture:
        simulate()
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertAlmostEqual(q[1], q[0], delta=0.02)

    coeffs = model.joint_mimic_coeffs.numpy()
    coeffs[follower] = (0.1, -0.5)
    model.joint_mimic_coeffs.assign(coeffs)
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_mimic_row_is_assembled(test, device):
    """Assemble a mimic row that carries the coupling."""
    solver, _ = _run_chain(device, coef0=0.0, coef1=1.0, leader_target=0.5, steps=10)
    row_types = solver._row_type.numpy()
    counts = solver._constraint_count.numpy()
    rows = [int(t) for w in range(row_types.shape[0]) for t in row_types[w, : counts[w]]]
    test.assertIn(PGS_CONSTRAINT_TYPE_MIMIC, rows)


def test_replicated_mimics_have_articulation_local_ranges(test, device):
    """Use one compact lookup range per articulation for replicated mimic rows."""
    template, _, _ = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(template, copy_count=4)
    solver = SolverSteel(builder.finalize(device=device), solve_mode="matrix_free")
    test.assertEqual(solver._mimic_art_start.numpy().tolist(), [0, 1, 2, 3, 4])
    test.assertEqual(solver._mimic_art_list.numpy().tolist(), [0, 1, 2, 3])


def test_disabled_mimic_is_ignored(test, device):
    """Leave the follower joint uncoupled under a disabled legacy mimic constraint."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0, legacy=True)
    builder.constraint_mimic_enabled[0] = False
    _, q = _run_model(builder.finalize(device=device), leader_target=0.5, steps=600)
    test.assertAlmostEqual(q[0], 0.5, delta=0.05)
    # The undriven follower swings freely; it must not sit at the leader angle.
    test.assertNotAlmostEqual(q[1], q[0], delta=0.02)


def test_mimic_row_overflow_is_reported(test, device):
    """Drop and flag a mimic row beyond articulation_constraint_max rather than keep part of it."""
    builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
    model = builder.finalize(device=device)
    # A zero activation gap keeps the (far) joint-limit rows out of the single row slot.
    options = {"articulation_constraint_max": 1, "joint_limit_activation_gap": 0.0, "warn_constraint_overflow": False}
    solver = SolverSteel(model, solve_mode="matrix_free", **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver._mimic_slot.numpy()[0]), 0)
    test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # A second, consistent row on the other joint (identity coupling in both directions).
        builder.add_constraint_mimic(joint0=0, joint1=1, coef0=0.0, coef1=1.0)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode="matrix_free", **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 1)
    test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
    with test.assertRaises(RuntimeError):
        solver.check_constraint_capacity()


def test_split_reject_quaternion_mimics_at_construction(test, device):
    """Reject quaternion mimics at construction in the split solve."""
    check_reject_quaternion_mimics_at_construction(test, device, solve_mode="split")


def test_split_rejects_supported_mimics(test, device):
    """Reject scalar and D6 mimics in the split solve, which has no bilateral rows."""
    for joint_type in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC, newton.JointType.D6):
        with test.subTest(joint_type=joint_type):
            builder, _, _ = _build_mimic_layout(joint_type)
            with test.assertRaisesRegex(NotImplementedError, "require solve_mode='matrix_free'"):
                SolverSteel(builder.finalize(device=device), solve_mode="split")


def test_split_invalid_legacy_mimic_keeps_precedence(test, device):
    """Keep legacy mimic precedence during validation in the split solve."""
    check_invalid_legacy_mimic_keeps_precedence(test, device, solve_mode="split")


class TestSteelMimicLayout(unittest.TestCase):
    pass


class TestSteelMimic(unittest.TestCase):
    pass


class TestSteelMimicLayoutSplit(unittest.TestCase):
    pass


for _name in (
    "test_reject_quaternion_mimics_at_construction",
    "test_scalar_and_d6_mimic_coordinate_maps",
    "test_invalid_legacy_mimic_keeps_precedence",
    "test_cross_articulation_mimic_is_rejected",
):
    add_function_test(TestSteelMimicLayout, _name, globals()[_name], devices=cuda_devices)


for _name in (
    "test_identity_mimic_tracks_leader",
    "test_legacy_constraint_mimic_tracks_leader",
    "test_legacy_constraint_overrides_joint_mimic",
    "test_runtime_coefficient_change",
    "test_mimic_row_is_assembled",
    "test_replicated_mimics_have_articulation_local_ranges",
    "test_disabled_mimic_is_ignored",
    "test_mimic_row_overflow_is_reported",
):
    add_function_test(TestSteelMimic, _name, globals()[_name], devices=cuda_devices)


for _name in (
    "test_split_reject_quaternion_mimics_at_construction",
    "test_split_rejects_supported_mimics",
    "test_split_invalid_legacy_mimic_keeps_precedence",
):
    add_function_test(TestSteelMimicLayoutSplit, _name, globals()[_name], devices=get_test_devices())


SPRING_KEYS = ("mujoco:dof_passive_stiffness", "mujoco:dof_springref", "mujoco:dof_ref")


_SPRING_WARNING = r"does not support passive joint springs yet .*#4516"


def _hinge_model(device, *, damping: float, spring_k: float | None = None, spring_ref: float = 0.5):
    """A fixed-base link on a revolute Z joint, so gravity exerts no torque about the axis.

    ``spring_k`` registers the MuJoCo spring attributes and sets the joint's stiffness.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    custom = {}
    if spring_k is not None:
        SolverMuJoCo.register_custom_attributes(builder)
        custom = {"mujoco:dof_passive_stiffness": spring_k, "mujoco:dof_springref": spring_ref}
    link = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        damping=damping,
        custom_attributes=custom,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _run_springs(model, solver, steps: int, qd0: float = 0.0):
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def _construct(test, model, *, solve_mode, expect_warning):
    """Construct the solver, requiring exactly the spring warning or no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverSteel(model, solve_mode=solve_mode, iterations=8)
    messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    test.assertEqual(len(caught), int(expect_warning), messages)
    if expect_warning:
        test.assertIs(caught[0].category, newton.exceptions.NewtonWarning)
        test.assertRegex(str(caught[0].message), _SPRING_WARNING)
    return solver


def test_passive_damping_decays_velocity(test, device, solve_mode="matrix_free"):
    """Decay joint velocity through passive damping; without damping the joint keeps coasting."""
    damped_model = _hinge_model(device, damping=0.02)
    undamped_model = _hinge_model(device, damping=0.0)
    damped = _run_springs(
        damped_model, _construct(test, damped_model, solve_mode=solve_mode, expect_warning=False), 480, qd0=5.0
    )
    undamped = _run_springs(
        undamped_model, _construct(test, undamped_model, solve_mode=solve_mode, expect_warning=False), 480, qd0=5.0
    )
    qd_damped = abs(float(damped.joint_qd.numpy()[0]))
    qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
    test.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
    test.assertLess(qd_damped, 0.2 * qd_undamped)


def test_nonzero_spring_stiffness_warns_and_is_ignored(test, device, solve_mode="matrix_free"):
    """Warn once at construction for nonzero MuJoCo spring stiffness and apply no spring torque."""
    model = _hinge_model(device, damping=0.0, spring_k=4.0, spring_ref=0.5)
    solver = _construct(test, model, solve_mode=solve_mode, expect_warning=True)
    np.testing.assert_array_equal(solver._passive_spring_stiffness.numpy(), [0.0])
    state = _run_springs(model, solver, 240)
    np.testing.assert_allclose(state.joint_q.numpy(), [0.0], atol=1.0e-6)
    test.assertEqual(float(solver._joint_tau.numpy()[0]), 0.0)


def test_zero_or_absent_spring_stiffness_does_not_warn(test, device, solve_mode="matrix_free"):
    """Construct silently without the spring attributes, and with zero registered stiffness."""
    _construct(test, _hinge_model(device, damping=0.1), solve_mode=solve_mode, expect_warning=False)
    _construct(test, _hinge_model(device, damping=0.1, spring_k=0.0), solve_mode=solve_mode, expect_warning=False)


def test_spring_attribute_edits_leave_damping_refresh_intact(test, device, solve_mode="matrix_free"):
    """Refresh damping on a JOINT_DOF_PROPERTIES notify while still applying no spring torque."""
    model = _hinge_model(device, damping=0.0, spring_k=0.0, spring_ref=0.5)
    solver = _construct(test, model, solve_mode=solve_mode, expect_warning=False)
    model.mujoco.dof_passive_stiffness.assign([4.0])
    model.joint_damping.assign([3.0])
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    state_0, state_1 = model.state(), model.state()
    state_0.joint_qd.fill_(1.0)
    solver.step(state_0, state_1, model.control(), None, DT)
    np.testing.assert_allclose(solver._joint_tau.numpy(), [-3.0], atol=1.0e-6)


def test_registration_adds_no_spring_attributes(test, device):
    """Register only Steel's own body attributes, not the MuJoCo spring attributes."""
    builder = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(builder)
    test.assertFalse(any(key in builder.custom_attributes for key in SPRING_KEYS))


class TestSteelSprings(unittest.TestCase):
    pass


# test_joint_damping covers passive damping on the matrix-free CUDA and split CPU routes.
add_function_test(
    TestSteelSprings,
    "test_passive_damping_decays_velocity_split",
    test_passive_damping_decays_velocity,
    devices=get_cuda_test_devices(),
    solve_mode="split",
)
for _name, _func in (
    ("test_nonzero_spring_stiffness_warns_and_is_ignored", test_nonzero_spring_stiffness_warns_and_is_ignored),
    ("test_zero_or_absent_spring_stiffness_does_not_warn", test_zero_or_absent_spring_stiffness_does_not_warn),
    (
        "test_spring_attribute_edits_leave_damping_refresh_intact",
        test_spring_attribute_edits_leave_damping_refresh_intact,
    ),
):
    add_function_test(TestSteelSprings, _name, _func, devices=get_cuda_test_devices())
    add_function_test(TestSteelSprings, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split")


add_function_test(
    TestSteelSprings,
    "test_registration_adds_no_spring_attributes",
    test_registration_adds_no_spring_attributes,
    devices=None,
)


DT_free_base_momentum = 1.0 / 200.0


STEPS = 60


# Spin about x, transverse to the stack axis, so the composite COM offset is swept through the
# rotation. A spin parallel to the offset cannot excite the term at all.
OMEGA = (20.0, 0.0, 0.0)


LINK_MASS = 1.0


LINK_HALF = 0.10  # half the stack spacing; the composite COM sits this far above the root COM


def _link_inertia(mass, radius=0.05, height=0.20):
    """Solid-cylinder inertia tensor about the link's own centre of mass."""
    i_xy = mass * (3.0 * radius**2 + height**2) / 12.0
    i_z = 0.5 * mass * radius**2
    return wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z)


def _build_welded_pair(device, root_com_z=0.0):
    """Two links welded by a fixed joint: composite COM sits ``LINK_HALF`` above the root COM.

    ``root_com_z`` offsets the root link's own centre of mass from its origin, which is the case
    every real robot link has and no bare primitive does.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, root_com_z),
        inertia=_link_inertia(LINK_MASS),
    )
    upper = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    free = builder.add_joint_free(parent=-1, child=root)
    weld = builder.add_joint_fixed(
        parent=root,
        child=upper,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free, weld])
    return builder.finalize(device=device)


def _build_equivalent_single(device):
    """One link carrying the welded pair's composite mass properties.

    Same total mass, same centre-of-mass offset from the body origin and same inertia about that
    centre of mass as :func:`_build_welded_pair`, so the two models are the same physical object.
    """
    total = 2.0 * LINK_MASS
    single = _link_inertia(LINK_MASS)
    i_xy = 2.0 * (single[0, 0] + LINK_MASS * LINK_HALF**2)
    i_z = 2.0 * single[2, 2]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=total,
        com=wp.vec3(0.0, 0.0, LINK_HALF),
        inertia=wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_serial_chain(device, depth=2):
    """Free root followed by ``depth`` revolute links in series, axes transverse to the spin."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    prev = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    joints = [builder.add_joint_free(parent=-1, child=prev)]
    for level in range(depth):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF * (level + 1)), wp.quat_identity()),
            mass=LINK_MASS,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=_link_inertia(LINK_MASS),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_single_offset_com_tumbling(device):
    """One free link with a triaxial inertia and a centre of mass offset from its origin.

    Distinct principal moments let a non-principal spin tumble, so the angular velocity changes
    within every step. The public-to-internal shift's residual is ``(w_out - w_in) x com_offset``
    per step: it needs both the tumble and the offset to be nonzero, which is why bodies spun about
    a principal axis (and every zero-offset primitive) cannot see it. The offset magnitude matches
    a humanoid pelvis.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0 * LINK_MASS,
        com=wp.vec3(0.0, 0.0, -0.076),
        inertia=wp.mat33(0.04, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.06),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _com_velocity(model, state):
    """Centre-of-mass velocity of the whole articulation [m/s]."""
    mass = model.body_mass.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64).reshape(-1, 6)
    return (mass[:, None] * body_qd[:, :3]).sum(axis=0) / mass.sum()


def _spin_free_articulation(model, omega=OMEGA, solve_mode="matrix_free"):
    """Spin the articulation with its composite centre of mass at rest, then step.

    Returns the largest centre-of-mass velocity drift [m/s] seen over the run.
    """
    state_0, state_1 = model.state(), model.state()

    # The free joint's linear coordinate is the ROOT body's COM velocity, which for a multi-link
    # articulation is not the composite COM. Give it the orbital velocity that leaves the composite
    # COM at rest, so both models start from the same physical state.
    body_q = state_0.body_q.numpy().astype(np.float64).reshape(-1, 7)
    mass = model.body_mass.numpy().astype(np.float64)
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)
    com_world = body_q[:, :3] + com_local  # identity orientation at build time
    composite = (mass[:, None] * com_world).sum(axis=0) / mass.sum()
    root_com = com_world[0]

    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = np.cross(np.array(omega), root_com - composite)
    joint_qd[3:6] = omega
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverSteel(model, solve_mode=solve_mode)
    solver._rigid_body_angular_damping.zero_()
    control = model.control()
    reference = _com_velocity(model, state_0)
    worst = 0.0
    for _ in range(STEPS):
        solver.step(state_0, state_1, control, None, DT_free_base_momentum)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(_com_velocity(model, state_0) - reference)))
    return worst


def test_single_link_conserves_com_velocity(test, device, solve_mode="matrix_free"):
    """Hold the center-of-mass velocity of a lone rotating free body in free fall.

    With no gravity the public-to-internal shift cancels step to step, so this case passes even
    on the defective formulation; it pins the baseline, not the fix. The gravity variant below
    is the discriminating single-body case.
    """
    model = _build_equivalent_single(device)
    test.assertLess(_spin_free_articulation(model, solve_mode=solve_mode), 1e-6)


def test_single_link_offset_com_tumbling(test, device, solve_mode="matrix_free"):
    """Conserve the COM velocity of a lone tumbling free body with an offset center of mass.

    Tumbling changes the angular velocity within each step, so the public-to-internal shift's
    reference-point error stops cancelling step to step: on the defective formulation this body
    drifts at ~0.5 m/s within the run, versus ~1e-7 here, in every ``solve_mode`` and
    ``articulated_contact_response``. This is the single-body case the fix corrects — every
    real robot root link has an offset centre of mass; it fails without the fix.
    """
    model = _build_single_offset_com_tumbling(device)
    test.assertLess(_spin_free_articulation(model, omega=(20.0, 0.5, 0.0), solve_mode=solve_mode), 1e-3)


def test_welded_pair_conserves_com_velocity(test, device, solve_mode="matrix_free"):
    """Hold the COM velocity of a welded two-link articulation to the integrator's own accuracy.

    The pair is the same physical object as the single link above -- same mass, centre of mass
    and inertia -- but its composite centre of mass sits away from the frame origin, so it
    exercises the reference-point bias. Drift over this run is ~0.20 m/s with the per-link
    frame correction and ~1.64 m/s without it; the bound sits between the two.
    """
    model = _build_welded_pair(device)
    test.assertLess(_spin_free_articulation(model, solve_mode=solve_mode), 0.5)


def test_welded_pair_drift_converges_with_timestep(test, device, solve_mode="matrix_free"):
    """Show the residual drift is first-order in dt, not a fixed formulation error.

    Halving the step must roughly halve the drift. A formulation error would sit flat instead,
    which is what distinguished this defect from ordinary truncation error.
    """
    global DT_free_base_momentum  # noqa: PLW0603 - the step size is the quantity under test
    original = DT_free_base_momentum
    try:
        DT_free_base_momentum = 1.0 / 200.0
        coarse = _spin_free_articulation(_build_welded_pair(device), solve_mode=solve_mode)
        DT_free_base_momentum = 1.0 / 400.0
        fine = _spin_free_articulation(_build_welded_pair(device), solve_mode=solve_mode)
    finally:
        DT_free_base_momentum = original
    test.assertGreater(coarse, 0.0)
    test.assertLess(fine, 0.65 * coarse)


def test_offset_root_com_conserves_com_velocity(test, device, solve_mode="matrix_free"):
    """Conserve momentum for a root link whose own COM is offset from its origin.

    This exercises the public-to-internal free-base shift, which is an identity only when the
    frame origin and the coordinate refer to the same point. It is a no-op for a bare primitive
    and the dominant error term for a real robot link.
    """
    model = _build_welded_pair(device, root_com_z=-0.076)
    test.assertLess(_spin_free_articulation(model, solve_mode=solve_mode), 0.5)


def test_matches_featherstone_reference(test, device, solve_mode="matrix_free"):
    """Track SolverFeatherstone on chains deep enough to expose the free base.

    Featherstone solves the same dynamics without Steel's frame re-centring machinery, so
    it is an independent reference. A depth-2 serial chain is the shallowest case that
    distinguishes a correct free-base bias from one that merely looks right on a welded pair.
    """
    model = _build_serial_chain(device, depth=2)
    trajectories = []
    for solver_cls in (SolverFeatherstone, SolverSteel):
        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[3:6] = OMEGA
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        if solver_cls is SolverFeatherstone:
            solver = solver_cls(model, angular_damping=0.0)
        else:
            solver = solver_cls(model, solve_mode=solve_mode)
            solver._default_angular_damping.zero_()
        control = model.control()
        for _ in range(20):
            solver.step(state_0, state_1, control, None, DT_free_base_momentum)
            state_0, state_1 = state_1, state_0
        trajectories.append(state_0.body_qd.numpy().astype(np.float64).copy())
    test.assertLess(float(np.abs(trajectories[0] - trajectories[1]).max()), 1e-4)


class TestSteelFreeBaseMomentum(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _name in (
    "test_single_link_conserves_com_velocity",
    "test_single_link_offset_com_tumbling",
    "test_welded_pair_conserves_com_velocity",
    "test_welded_pair_drift_converges_with_timestep",
    "test_offset_root_com_conserves_com_velocity",
    "test_matches_featherstone_reference",
):
    add_function_test(TestSteelFreeBaseMomentum, _name, globals()[_name], devices=devices)
    add_function_test(
        TestSteelFreeBaseMomentum,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        solve_mode="split",
    )


DT_free_root_predictor = 1.0 / 200.0


STEPS_free_root_predictor = 80


RADIUS = 0.1


# Spin transverse to the slide so omega x v points along +/-z, straight into the contact normal.
OMEGA_Y = 20.0


SLIDE_VX = 2.0


@wp.kernel
def _record_slider_step(
    body_q: wp.array[wp.transform],
    rigid_contact_count: wp.array[int],
    step: int,
    # outputs
    heights: wp.array[float],
    peak_contacts: wp.array[int],
):
    """Record one slider sample without a per-step device synchronization."""
    heights[step] = wp.transform_get_translation(body_q[0])[2]
    peak_contacts[0] = wp.max(peak_contacts[0], rigid_contact_count[0])


def _undamped(model):
    """Disable the per-body angular damping, so the free root keeps its spin."""
    model.steel.rigid_body_angular_damping.zero_()
    return model


def _new_builder(**kwargs):
    builder = newton.ModelBuilder(**kwargs)
    SolverSteel.register_custom_attributes(builder)
    return builder


def _build_slider(device):
    """A frictionless sphere resting on a frictionless ground plane."""
    builder = _new_builder(gravity=(0.0, 0.0, -9.81))
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
    )
    builder.add_shape_sphere(body, radius=RADIUS)
    builder.add_articulation([builder.add_joint_free(body)])
    builder.add_ground_plane()
    return _undamped(builder.finalize(device=device))


def _slide_heights(model, omega_y, solve_mode="matrix_free", response="immediate"):
    """Slide the sphere at ``SLIDE_VX`` while spinning at ``omega_y``; return heights and peak contacts."""
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, omega_y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverSteel(model, solve_mode=solve_mode, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()
    heights = wp.empty(STEPS_free_root_predictor, dtype=wp.float32, device=model.device)
    peak_contacts = wp.zeros(1, dtype=wp.int32, device=model.device)
    for step in range(STEPS_free_root_predictor):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT_free_root_predictor)
        state_0, state_1 = state_1, state_0
        wp.launch(
            _record_slider_step,
            dim=1,
            inputs=[state_0.body_q, contacts.rigid_contact_count, step],
            outputs=[heights, peak_contacts],
            device=model.device,
        )
    return heights.numpy(), int(peak_contacts.numpy()[0])


def _world_com(model, state):
    """World position of the root body's center of mass [m]."""
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)[0]
    q = state.body_q.numpy().astype(np.float64).reshape(-1, 7)[0]
    rot = wp.quat(*(float(c) for c in q[3:7]))
    return np.asarray(q[0:3]) + np.asarray(wp.quat_rotate(rot, wp.vec3(*com_local)))


def test_frictionless_slide_is_spin_invariant(test, device, solve_mode="matrix_free", response="immediate"):
    """Keep a frictionless sliding sphere's height trajectory independent of its spin.

    Without friction the spin exerts no force. A predictor missing the ``omega x v`` term feeds
    the normal rows a phantom vertical velocity of ``dt * omega_y * vx`` (0.2 m/s here) and the
    heights diverge by millimetres within the run.
    """
    still, contacts_still = _slide_heights(_build_slider(device), 0.0, solve_mode, response)
    spinning, contacts_spin = _slide_heights(_build_slider(device), OMEGA_Y, solve_mode, response)
    test.assertGreater(min(contacts_still, contacts_spin), 0, "no contacts were generated")
    for label, heights in (("still", still), ("spinning", spinning)):
        test.assertLess(float(np.abs(heights - RADIUS).max()), 2e-3, f"{label} run left the ground support band")
    divergence = float(np.abs(spinning - still).max())
    test.assertLess(divergence, 1e-4, f"spin changed a frictionless slide by {divergence} m")


def test_anchored_spinner_com_stays_put(test, device, solve_mode="matrix_free"):
    """Keep the center of mass of a force-free spinner with an offset joint anchor in place.

    The free joint's coordinate tracks the child anchor frame, offset from the center of mass by
    ``child_xform``; the anchor must orbit the stationary center of mass.
    """
    configs = (
        ("plain-offset", wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)),
        (
            "rotated-anchor-offset-com",
            wp.transform(wp.vec3(0.1, 0.02, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)),
            wp.vec3(0.03, -0.05, 0.02),
        ),
    )
    for label, child_xform, com in configs:
        with test.subTest(config=label):
            builder = _new_builder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                mass=1.0, com=com, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014)
            )
            builder.add_articulation([builder.add_joint_free(body, child_xform=child_xform)])
            model = _undamped(builder.finalize(device=device))
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[3:6] = (0.0, 0.0, 10.0)
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            start = _world_com(model, state_0)
            solver = SolverSteel(model, solve_mode=solve_mode)
            drift = 0.0
            for _ in range(STEPS_free_root_predictor):
                solver.step(state_0, state_1, model.control(), None, DT_free_root_predictor)
                state_0, state_1 = state_1, state_0
                drift = max(drift, float(np.linalg.norm(_world_com(model, state_0) - start)))
            test.assertLess(drift, 1e-4, f"COM of a force-free spinner drifted {drift} m")


def test_integrator_transport_identities(test, device):
    """Add ``omega x v * dt`` to a root free joint's linear coordinate and pass a descendant's through."""
    builder = _new_builder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    parent = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=inertia)
    root = builder.add_joint_free(parent)
    nested = builder.add_joint_free(child, parent=parent)
    builder.add_articulation([root, nested])
    model = builder.finalize(device=device)

    state = model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[0:3] = (0.4, 0.0, 0.0)
    joint_qd[3:6] = (0.0, 0.0, 3.0)
    joint_qd[6:9] = (0.4, 0.0, 0.0)
    joint_qd[9:12] = (0.0, 0.0, 3.0)
    state.joint_qd.assign(joint_qd)
    q_new = wp.zeros_like(state.joint_q)
    qd_new = wp.zeros_like(state.joint_qd)
    wp.launch(
        integrate_generalized_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            wp.zeros(model.joint_count, dtype=wp.int32, device=device),
            model.joint_dof_dim,
            model.body_com,
            model.joint_X_c,
            state.joint_q,
            state.joint_qd,
            wp.zeros_like(state.joint_qd),
            DT_free_root_predictor,
            wp.zeros(model.body_count, dtype=wp.float32, device=device),
            wp.ones(model.joint_count, dtype=wp.int32, device=device),
        ],
        outputs=[q_new, qd_new],
        device=device,
    )
    out = qd_new.numpy()
    transport = np.cross((0.0, 0.0, 3.0), (0.4, 0.0, 0.0)) * DT_free_root_predictor
    np.testing.assert_allclose(out[0:3], np.array((0.4, 0.0, 0.0)) + transport, rtol=1e-6)
    np.testing.assert_allclose(out[3:6], (0.0, 0.0, 3.0), atol=1e-7)
    np.testing.assert_allclose(out[6:9], (0.4, 0.0, 0.0), atol=1e-7, err_msg="descendant gained transport")
    np.testing.assert_allclose(out[9:12], (0.0, 0.0, 3.0), atol=1e-7)


def test_predictor_and_qdd_transport_identities(test, device, solve_mode="matrix_free"):
    """Apply the transport term in the predictor and remove it in the solved-velocity conversion."""
    model = _build_slider(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    test.assertEqual(solver._free_root_joint_count, 1)
    state_in, state_out = model.state(), model.state()
    state_aug = solver._prepare_augmented_state(state_in)

    qd_np = np.array((2.0, 0.0, 0.0, 0.0, 20.0, 0.0), dtype=np.float32)
    qdd_np = np.array((0.3, -0.4, 0.5, 0.1, -0.2, 0.25), dtype=np.float32)
    state_in.joint_qd.assign(qd_np)
    solver._qd_work.assign(qd_np)
    state_aug._joint_qdd.assign(qdd_np)
    solver._stage3_compute_v_hat(state_in, state_aug, DT_free_root_predictor, solver._qd_work)
    transport = np.cross(qd_np[3:6], qd_np[0:3])
    expected_v_hat = qd_np + qdd_np * DT_free_root_predictor
    expected_v_hat[0:3] += transport * DT_free_root_predictor
    np.testing.assert_allclose(solver._v_hat.numpy(), expected_v_hat, rtol=1e-6, atol=1e-7)

    v_out_np = np.array((1.8, 0.1, -0.15, 0.05, 19.5, 0.25), dtype=np.float32)
    solver._v_out.assign(v_out_np)
    wp.launch(
        update_qdd_from_velocity,
        dim=model.joint_dof_count,
        inputs=[
            state_in.joint_qd,
            solver._kinematic_dof_mask,
            1.0 / DT_free_root_predictor,
            solver._dynamics_dof_active,
        ],
        outputs=[solver._v_out, state_aug._joint_qdd],
        device=device,
    )
    wp.launch(
        remove_free_root_transport_from_qdd,
        dim=solver._free_root_joint_count,
        inputs=[
            solver._free_root_joint_indices,
            model.joint_qd_start,
            solver._kinematic_joint_mask,
            state_in.joint_qd,
            solver._dynamics_joint_active,
        ],
        outputs=[state_aug._joint_qdd],
        device=device,
    )
    expected_qdd = (v_out_np - qd_np) / DT_free_root_predictor
    expected_qdd[0:3] -= transport
    np.testing.assert_allclose(state_aug._joint_qdd.numpy(), expected_qdd, rtol=1e-6, atol=1e-5)
    wp.launch(
        integrate_generalized_joints,
        dim=model.joint_count,
        inputs=[
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            solver._kinematic_joint_mask,
            model.joint_dof_dim,
            model.body_com,
            model.joint_X_c,
            state_in.joint_q,
            state_in.joint_qd,
            state_aug._joint_qdd,
            DT_free_root_predictor,
            solver._rigid_body_angular_damping,
            solver._dynamics_joint_active,
        ],
        outputs=[state_out.joint_q, state_out.joint_qd],
        device=device,
    )
    np.testing.assert_allclose(state_out.joint_qd.numpy(), v_out_np, rtol=1e-6, atol=1e-6)


def test_compact_root_metadata_includes_standalone_joint(test, device, solve_mode="matrix_free"):
    """Keep every world-rooted free joint, but no descendant, in the compact root launch."""
    builder = _new_builder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    root_a = builder.add_link(mass=1.0, inertia=inertia)
    child = builder.add_link(mass=1.0, inertia=inertia)
    root_b = builder.add_link(mass=1.0, inertia=inertia)
    joint_a = builder.add_joint_free(root_a)
    nested = builder.add_joint_free(child, parent=root_a)
    joint_b = builder.add_joint_free(root_b)
    builder.add_articulation([joint_a, nested])
    solver = SolverSteel(builder.finalize(device=device), solve_mode=solve_mode)
    np.testing.assert_array_equal(solver._free_root_joint_indices.numpy(), (joint_a, joint_b))


def test_captured_step_matches_uncaptured(test, device, solve_mode="matrix_free"):
    """Reproduce the eager trajectory by replaying a captured collide-and-step graph."""
    heights_ref, _ = _slide_heights(_build_slider(device), OMEGA_Y, solve_mode)

    model = _build_slider(device)
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, OMEGA_Y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    solver = SolverSteel(model, solve_mode=solve_mode)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()

    def one_step():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT_free_root_predictor)
        # Copy back instead of swapping so the captured pointers stay valid.
        wp.copy(state_0.body_q, state_1.body_q)
        wp.copy(state_0.body_qd, state_1.body_qd)
        wp.copy(state_0.joint_q, state_1.joint_q)
        wp.copy(state_0.joint_qd, state_1.joint_qd)

    one_step()  # warm-up, matches the reference run's first step
    with wp.ScopedCapture(device) as capture:
        one_step()  # capture records this step without executing it
    heights = []
    for _ in range(STEPS_free_root_predictor - 1):
        wp.capture_launch(capture.graph)
        heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
    np.testing.assert_allclose(heights, heights_ref[1:], atol=1e-6)


class TestSteelFreeRootPredictor(unittest.TestCase):
    pass


for _name, _func in (
    ("test_frictionless_slide_is_spin_invariant", test_frictionless_slide_is_spin_invariant),
    ("test_anchored_spinner_com_stays_put", test_anchored_spinner_com_stays_put),
    ("test_integrator_transport_identities", test_integrator_transport_identities),
    ("test_predictor_and_qdd_transport_identities", test_predictor_and_qdd_transport_identities),
    ("test_compact_root_metadata_includes_standalone_joint", test_compact_root_metadata_includes_standalone_joint),
    ("test_captured_step_matches_uncaptured", test_captured_step_matches_uncaptured),
):
    add_function_test(TestSteelFreeRootPredictor, _name, _func, devices=devices)


for _name, _func, _devices in (
    ("test_frictionless_slide_is_spin_invariant", test_frictionless_slide_is_spin_invariant, get_test_devices()),
    ("test_anchored_spinner_com_stays_put", test_anchored_spinner_com_stays_put, get_test_devices()),
    ("test_predictor_and_qdd_transport_identities", test_predictor_and_qdd_transport_identities, get_test_devices()),
    (
        "test_compact_root_metadata_includes_standalone_joint",
        test_compact_root_metadata_includes_standalone_joint,
        get_test_devices(),
    ),
    ("test_captured_step_matches_uncaptured", test_captured_step_matches_uncaptured, devices),
):
    add_function_test(TestSteelFreeRootPredictor, f"{_name}_split", _func, devices=_devices, solve_mode="split")


# The integrator kernel is mode-independent; also check it on CPU.
add_function_test(
    TestSteelFreeRootPredictor,
    "test_integrator_transport_identities",
    test_integrator_transport_identities,
    devices=[d for d in get_test_devices() if d.is_cpu],
)


# The propagation response solves the free sphere's contact as a body-space row.
add_function_test(
    TestSteelFreeRootPredictor,
    "test_frictionless_slide_is_spin_invariant_propagation",
    test_frictionless_slide_is_spin_invariant,
    devices=devices,
    response="propagation",
)


def _body_omega(state, body=0):
    q = wp.quat(*state.body_q.numpy()[body, 3:])
    omega = wp.vec3(*state.body_qd.numpy()[body, 3:])
    return np.asarray(wp.quat_rotate_inv(q, omega), dtype=np.float64)


def _undamped_builder():
    """Return a builder whose free bodies have no angular damping."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    newton.solvers.SolverSteel.register_custom_attributes(builder)
    return builder


def _finalize_undamped(builder, device):
    model = builder.finalize(device=device)
    model.steel.rigid_body_angular_damping.zero_()
    return model


def _model(device, inertia, omega, *, armature=0.0, mass=1.0):
    builder = _undamped_builder()
    builder.add_body(mass=mass, inertia=wp.mat33(inertia), com=wp.vec3(0.03, -0.02, 0.01))
    model = _finalize_undamped(builder, device)
    model.joint_armature.fill_(armature)
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.array([0.1, 0.2, 0.3, *omega], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, output


def test_free_spin_energy(test, device):
    """Conserve kinetic energy and angular momentum of a torque-free asymmetric spinning body."""
    inertia = np.diag([0.001, 0.02, 0.0205])
    model, state, output = _model(device, inertia, (100.0, 20.0, 50.0))
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
    control = model.control()
    omega = _body_omega(state)
    energy = 0.5 * omega @ inertia @ omega
    momentum = np.linalg.norm(inertia @ omega)
    for step in range(1000):
        solver.step(state, output, control, None, 1.0 / 240.0)
        state, output = output, state
        omega = _body_omega(state)
        observed = 0.5 * omega @ inertia @ omega
        test.assertTrue(np.isfinite(observed), f"nonfinite energy at step {step}")
        test.assertAlmostEqual(observed / energy, 1.0, delta=1.0e-3, msg=f"step {step}")
    test.assertAlmostEqual(np.linalg.norm(inertia @ omega) / momentum, 1.0, delta=1.0e-3)
    np.testing.assert_allclose(state.body_qd.numpy()[0, :3], [0.1, 0.2, 0.3], atol=2.0e-4)


def test_principal_axis_torque(test, device):
    """Accelerate a body about a principal axis by torque over inertia plus armature."""
    inertia = np.diag([0.04, 0.02, 0.05])
    for armature in (0.0, 0.03):
        model, state, output = _model(device, inertia, (0.0, 0.0, 2.0), armature=armature)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
        control = model.control()
        torque = 0.1
        dt = 1.0 / 240.0
        for _ in range(120):
            state.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], dtype=np.float32))
            solver.step(state, output, control, None, dt)
            state, output = output, state
        expected = 2.0 + 120 * dt * torque / (inertia[2, 2] + armature)
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 5]), expected, delta=1.0e-4)


def test_rotated_inertia_and_armature(test, device):
    """Conserve rotational energy with a rotated inertia and armature across mass scales."""
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
    basis = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
    local_basis = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
    for scale, armature in ((1.0e-5, 0.0), (1.0, 0.003), (1.0e5, 0.0)):
        inertia = scale * local_basis @ np.diag([0.001, 0.02, 0.0205]) @ local_basis.T
        omega = basis @ local_basis @ np.array([20.0, 4.0, 10.0])
        model, state, output = _model(device, inertia, omega, mass=scale)
        rotor = armature * np.array([1.0, 2.0, 3.0])
        model.joint_armature.assign(np.array([0.0, 0.0, 0.0, *rotor], dtype=np.float32))
        q = state.joint_q.numpy()
        q[3:7] = np.asarray(rotation)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
        control = model.control()
        w = _body_omega(state)
        initial = w @ inertia @ w + np.dot(rotor, omega * omega)
        for _ in range(200):
            solver.step(state, output, control, None, 1.0 / 240.0)
            state, output = output, state
        w = _body_omega(state)
        omega = state.body_qd.numpy()[0, 3:].astype(np.float64)
        final = w @ inertia @ w + np.dot(rotor, omega * omega)
        test.assertAlmostEqual(final / initial, 1.0, delta=1.0e-3, msg=f"scale={scale}, armature={armature}")


def test_gyro_precession_converges(test, device):
    """Match a double-precision Euler-equation reference for gyroscopic precession."""
    inertia = np.diag([0.001, 0.02, 0.0205])
    inverse = np.linalg.inv(inertia)
    initial = np.array([20.0, 4.0, 10.0])
    duration = 0.2

    def derivative(omega):
        return inverse @ np.cross(inertia @ omega, omega)

    # Independent double-precision Euler-equation reference in the body frame.
    reference = initial.copy()
    h = duration / 4096
    for _ in range(4096):
        k1 = derivative(reference)
        k2 = derivative(reference + 0.5 * h * k1)
        k3 = derivative(reference + 0.5 * h * k2)
        k4 = derivative(reference + h * k3)
        reference += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    errors = []
    for steps in (48, 96):
        model, state, output = _model(device, inertia, initial)
        solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free")
        control = model.control()
        for _ in range(steps):
            solver.step(state, output, control, None, duration / steps)
            state, output = output, state
        errors.append(np.linalg.norm(_body_omega(state) - reference))
    test.assertLess(errors[0], 0.1)
    test.assertLess(errors[1], 0.35 * errors[0])


def _mixed_scene(device, *, arm, totes):
    """Build two worlds with free bodies on both sides of a 19-DOF articulation."""
    template = _undamped_builder()
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.001, 0.02, 0.0205])),
            com=wp.vec3(0.03, -0.02, 0.01),
        )
    if arm:
        joints = []
        parent = -1
        for _ in range(19):
            body = template.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.02))
            joints.append(
                template.add_joint_revolute(
                    parent,
                    body,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                    target_ke=40.0,
                    target_kd=4.0,
                    armature=0.01,
                )
            )
            parent = body
        template.add_articulation(joints)
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.diag([0.04, 0.002, 0.041])),
            com=wp.vec3(-0.02, 0.01, 0.04),
        )
    builder = _undamped_builder()
    builder.replicate(template, 2, spacing=(3.0, 0.0, 0.0))
    model = _finalize_undamped(builder, device)
    state, output = model.state(), model.state()
    velocity = state.joint_qd.numpy()
    free_joints = np.flatnonzero(model.joint_type.numpy() == newton.JointType.FREE)
    for i, joint in enumerate(free_joints):
        start = int(model.joint_qd_start.numpy()[joint])
        velocity[start + 3 : start + 6] = (100.0, 20.0, 50.0) if i % 2 == 0 else (20.0, 100.0, 50.0)
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.fill_(0.1)
    # The 19 revolute joints keep two finite position-limit rows each.
    solver = newton.solvers.SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=64)
    return model, solver, control, [state, output]


def test_mixed_arm_and_spinning_bodies(test, device):
    """Preserve isolated arm and rotor dynamics with mixed response sizes and graph replay."""
    for execution in ("eager", "graph"):
        with test.subTest(execution=execution):
            scenes = [
                _mixed_scene(device, arm=True, totes=True),
                _mixed_scene(device, arm=True, totes=False),
                _mixed_scene(device, arm=False, totes=True),
            ]
            model, solver, _, states = scenes[0]
            free_bodies = (0, 20, 21, 41)
            inertias = model.body_inertia.numpy().astype(np.float64)

            def energies(state, free_bodies=free_bodies, inertias=inertias):
                """Measure each free body's rotational energy in its inertia frame."""
                values = []
                for body in free_bodies:
                    omega = _body_omega(state, body)
                    values.append(omega @ inertias[body] @ omega)
                return np.asarray(values)

            initial_energy = energies(states[0])
            test.assertIn(6, solver._L_by_size)
            test.assertIn(19, solver._L_by_size)

            def step(scenes=scenes):
                """Advance the mixed scene and its independent controls together."""
                for _, scene_solver, control, pair in scenes:
                    scene_solver.step(pair[0], pair[1], control, None, 1.0 / 240.0)
                    pair[0], pair[1] = pair[1], pair[0]

            # Compile and initialize the same layouts before capture.
            step()
            graph = None
            if execution == "graph":
                with wp.ScopedCapture(device=device) as capture:
                    step()
                    step()
                graph = capture.graph
            for _ in range(100):
                if graph is None:
                    step()
                    step()
                else:
                    wp.capture_launch(graph)
                current = scenes[0][3][0]
                np.testing.assert_allclose(energies(current) / initial_energy, 1.0, atol=1.0e-3, rtol=0.0)
                for field in ("body_q", "body_qd"):
                    width = 7 if field == "body_q" else 6
                    mixed = getattr(current, field).numpy().reshape(2, 21, width)
                    arm = getattr(scenes[1][3][0], field).numpy().reshape(2, 19, width)
                    totes = getattr(scenes[2][3][0], field).numpy().reshape(2, 2, width)
                    np.testing.assert_allclose(mixed[:, 1:20], arm, atol=3.0e-5, rtol=1.0e-6)
                    np.testing.assert_allclose(mixed[:, [0, 20]], totes, atol=3.0e-5, rtol=1.0e-6)
            test.assertGreater(float(np.abs(scenes[1][3][0].joint_q.numpy()).max()), 0.01)


class TestSteelGyroscopic(unittest.TestCase):
    pass


for _device in get_cuda_test_devices():
    add_function_test(
        TestSteelGyroscopic,
        "test_mixed_arm_and_spinning_bodies",
        test_mixed_arm_and_spinning_bodies,
        devices=[_device],
    )
    add_function_test(TestSteelGyroscopic, "test_free_spin_energy", test_free_spin_energy, devices=[_device])
    add_function_test(TestSteelGyroscopic, "test_principal_axis_torque", test_principal_axis_torque, devices=[_device])
    add_function_test(
        TestSteelGyroscopic,
        "test_rotated_inertia_and_armature",
        test_rotated_inertia_and_armature,
        devices=[_device],
    )
    add_function_test(
        TestSteelGyroscopic, "test_gyro_precession_converges", test_gyro_precession_converges, devices=[_device]
    )


# Off-origin center of mass, so the tests distinguish the body COM from the body origin.
COM_LOCAL = (0.05, -0.03, 0.02)


OMEGA_joint_qd_convention = (0.0, 0.0, 3.0)


V_COM = (0.1, -0.2, 0.05)


ORIGINS = ((0.0, 0.0, 0.6), (5.0, 0.0, 0.6), (20.0, 0.0, 0.6))


def _build_free_body(origin, device):
    """Build a single gravity-free, contact-free free-root body placed at ``origin``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(*COM_LOCAL),
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04),
    )
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    return builder.finalize(device=device)


def _step_once(model, solver):
    """Step ``solver`` once from the seeded public free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA_joint_qd_convention
    state_in.joint_qd.assign(joint_qd)
    solver.step(state_in, state_out, model.control(), None, DT)
    return state_out


def test_public_eval_fk_round_trips_solver_body_qd(test, device, solve_mode="matrix_free"):
    """Reproduce the solver's published body_qd with eval_fk on its joint state, far from the origin."""
    for origin in ORIGINS:
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            state_out = _step_once(model, SolverSteel(model, solve_mode=solve_mode))

            probe = model.state()
            probe.joint_q.assign(state_out.joint_q)
            probe.joint_qd.assign(state_out.joint_qd)
            newton.eval_fk(model, probe.joint_q, probe.joint_qd, probe)
            np.testing.assert_allclose(probe.body_qd.numpy(), state_out.body_qd.numpy(), rtol=1e-6, atol=1e-6)
            # body_qd's linear part is the COM velocity, independent of the distance from the origin.
            np.testing.assert_allclose(state_out.body_qd.numpy()[0][0:3], V_COM, rtol=0.0, atol=1e-3)


def test_joint_qd_matches_featherstone_far_from_origin(test, device, solve_mode="matrix_free"):
    """Match SolverFeatherstone's joint_qd and body_qd from the same public twist, far from the origin."""
    for origin in ORIGINS:
        with test.subTest(origin=origin):
            model = _build_free_body(origin, device)
            reference = _step_once(model, SolverFeatherstone(model))
            state_out = _step_once(model, SolverSteel(model, solve_mode=solve_mode))
            # A convention mismatch would differ by omega x x_com_world, about 60 m/s at 20 m.
            np.testing.assert_allclose(state_out.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=1e-3)
            np.testing.assert_allclose(state_out.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1e-3)


class TestSteelJointQdConvention(unittest.TestCase):
    pass


for _name, _func in (
    ("test_public_eval_fk_round_trips_solver_body_qd", test_public_eval_fk_round_trips_solver_body_qd),
    ("test_joint_qd_matches_featherstone_far_from_origin", test_joint_qd_matches_featherstone_far_from_origin),
):
    add_function_test(TestSteelJointQdConvention, _name, _func, devices=get_cuda_test_devices())
    add_function_test(
        TestSteelJointQdConvention, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split"
    )


def _build_branched_d6(device):
    """Revolute root with a three-angular and a two-angular D6 child, all with off-origin COMs."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    axis = newton.ModelBuilder.JointDofConfig.create_unlimited
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04)
    root = builder.add_link(mass=1.0, com=wp.vec3(*COM_LOCAL), inertia=inertia)
    ball_like = builder.add_link(mass=0.7, com=wp.vec3(0.1, 0.04, -0.05), inertia=inertia)
    universal = builder.add_link(mass=0.5, com=wp.vec3(-0.02, 0.08, 0.03), inertia=inertia)
    joints = [
        builder.add_joint_revolute(
            -1, root, axis=newton.Axis.Z, parent_xform=wp.transform((0.3, 0.1, 0.5), wp.quat_identity())
        ),
        builder.add_joint_d6(
            root,
            ball_like,
            angular_axes=[axis(axis=newton.Axis.X), axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
            parent_xform=wp.transform((0.2, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)),
        ),
        builder.add_joint_d6(
            root,
            universal,
            angular_axes=[axis(axis=newton.Axis.Y), axis(axis=newton.Axis.Z)],
            parent_xform=wp.transform((-0.2, 0.1, 0.0), wp.quat_identity()),
        ),
    ]
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _fk_state(model, joint_q, joint_qd):
    state = model.state()
    state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
    state.joint_qd.assign(np.asarray(joint_qd, dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _spatial_velocity_at(model, state, point):
    """Body twists from eval_fk, referenced at the world point ``point``."""
    body_q = state.body_q.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64)
    twists = []
    for pose, twist, com in zip(body_q, body_qd, model.body_com.numpy(), strict=True):
        com_world = np.array(wp.transform_point(wp.transform(*pose[:3], *pose[3:]), wp.vec3(*com)))
        twists.append(np.concatenate([twist[:3] + np.cross(twist[3:], point - com_world), twist[3:]]))
    return np.array(twists)


def _stage1_motion(solver, model, state):
    """Run the stage-1 forward pass; return the solve origin and each body's spatial velocity and bias."""
    solver._prepare_augmented_state(state)
    solver._stage1_fk_id(state, solver, model.state())
    origin = solver._articulation_origin.numpy()[0].astype(np.float64)
    return origin, solver._body_v_s.numpy(), solver._body_a_s.numpy(), solver._joint_S_s.numpy()


def test_multi_angular_d6_motion_matches_fk(test, device, **solver_kwargs):
    """Match eval_fk's body velocity and its time derivative through multi-angular D6 joints."""
    model = _build_branched_d6(device)
    q = np.array((0.25, 0.4, -0.5, 0.6, 0.7, -0.35))
    qd = np.array((0.9, 1.1, -0.9, 1.3, 1.4, -1.2))
    solver = SolverSteel(model, **solver_kwargs)
    test.assertEqual(solver._tree_plan is not None, solver_kwargs.get("parallel_tree", False))
    origin, v_s, a_s, _ = _stage1_motion(solver, model, _fk_state(model, q, qd))
    np.testing.assert_allclose(v_s, _spatial_velocity_at(model, _fk_state(model, q, qd), origin), rtol=0.0, atol=1e-5)
    # With zero joint accelerations the bias is the derivative of the twist along q + t qd.
    h = 1e-3
    forward = _spatial_velocity_at(model, _fk_state(model, q + h * qd, qd), origin)
    backward = _spatial_velocity_at(model, _fk_state(model, q - h * qd, qd), origin)
    np.testing.assert_allclose(a_s, (forward - backward) / (2.0 * h), rtol=0.0, atol=2e-3)


def test_free_and_distance_roots_use_com_twists(test, device, **solver_kwargs):
    """Reference FREE and DISTANCE root twists at the root COM, with an off-origin COM far from the world origin."""
    for kind in ("free", "distance"):
        with test.subTest(kind=kind):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            body = builder.add_link(
                xform=wp.transform(wp.vec3(*ORIGINS[-1]), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)),
                mass=1.0,
                com=wp.vec3(*COM_LOCAL),
                inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04),
            )
            builder.add_articulation([getattr(builder, f"add_joint_{kind}")(parent=-1, child=body)])
            model = builder.finalize(device=device)
            joint_qd = np.array((*V_COM, 0.4, -0.6, 1.1))
            state = _fk_state(model, model.joint_q.numpy(), joint_qd)
            origin, v_s, a_s, motion = _stage1_motion(SolverSteel(model, **solver_kwargs), model, state)

            com_world = wp.transform_point(wp.transform(*state.body_q.numpy()[0]), wp.vec3(*COM_LOCAL))
            np.testing.assert_allclose(origin, com_world, rtol=0.0, atol=1e-5)
            np.testing.assert_allclose(motion, np.eye(6), rtol=0.0, atol=0.0)
            np.testing.assert_allclose(v_s[0], joint_qd, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(a_s[0], 0.0, rtol=0.0, atol=1e-6)


class TestSteelJointMotion(unittest.TestCase):
    pass


for _name, _func in (
    ("test_multi_angular_d6_motion_matches_fk", test_multi_angular_d6_motion_matches_fk),
    ("test_free_and_distance_roots_use_com_twists", test_free_and_distance_roots_use_com_twists),
):
    add_function_test(TestSteelJointMotion, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split")
    add_function_test(TestSteelJointMotion, _name, _func, devices=get_cuda_test_devices(), solve_mode="matrix_free")
add_function_test(
    TestSteelJointMotion,
    "test_multi_angular_d6_motion_matches_fk_parallel_tree",
    test_multi_angular_d6_motion_matches_fk,
    devices=get_cuda_test_devices(),
    solve_mode="matrix_free",
    parallel_tree=True,
)


def _build_two_worlds_with_global_body(device):
    """Two worlds with one driven slider each, plus one global (world -1) free body."""
    template = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_prismatic(-1, body, axis=newton.Axis.X, target_ke=100.0, target_kd=1.0)
    template.add_articulation([joint])

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.add_world(template)
    builder.add_world(template)
    builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    np.testing.assert_array_equal(model.articulation_world.numpy(), [0, 1, -1])
    return model


def _mask(values, device):
    return wp.array(values, dtype=wp.bool, device=device)


def test_reset_validates_mask_length(test, device, solve_mode="matrix_free"):
    """Accept ``(world_count + 1,)`` masks and reject every other length, including the legacy ``(world_count,)``."""
    model = _build_two_worlds_with_global_body(device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    state = model.state()
    solver._impulses.fill_(17.0)
    before = solver._impulses.numpy().copy()

    # Two worlds: the valid mask has three entries, the last one selecting global entities.
    solver.reset(state, _mask([True, False, True], device))
    for length in (1, 2, 4):
        with test.subTest(length=length):
            with test.assertRaisesRegex(ValueError, r"model\.world_count \+ 1 \(3\)"):
                solver.reset(state, wp.ones(length, dtype=wp.bool, device=device))
    with test.assertRaisesRegex(TypeError, "dtype bool"):
        solver.reset(state, wp.ones(3, dtype=wp.int32, device=device))
    # The solver keeps no impulse history, so a reset never touches the impulse scratch.
    np.testing.assert_array_equal(solver._impulses.numpy(), before)


def test_reset_isolates_global_slot(test, device, solve_mode="matrix_free"):
    """Select local worlds and global entities independently through the final mask entry."""
    model = _build_two_worlds_with_global_body(device)
    solver = SolverSteel(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
    state, output = model.state(), model.state()
    control = model.control()
    solver.step(state, output, control, None, 0.01)

    # The status has the mask's layout, so every entry is cleared only by its own mask entry.
    cases = (
        ([False, False, True], [0, 0, 1], [True, True, False]),
        ([True, False, False], [1, 0, 0], [False, True, True]),
        ([False, True, False], [0, 1, 0], [True, False, True]),
        (None, [1, 1, 1], [False, False, False]),
    )
    for mask, expected_refresh, expected_overflow in cases:
        with test.subTest(mask=mask):
            solver.constraint_overflow.fill_(True)
            solver.reset(state, None if mask is None else _mask(mask, device))
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), expected_overflow)
            solver.step(state, output, control, None, 0.01)
            np.testing.assert_array_equal(solver._mass_update_mask.numpy(), expected_refresh)
            # The request is consumed by the first step after the reset.
            solver.step(state, output, control, None, 0.01)
            np.testing.assert_array_equal(solver._mass_update_mask.numpy(), [0, 0, 0])


class TestSteelReset(unittest.TestCase):
    pass


add_function_test(TestSteelReset, "test_reset_validates_mask_length", test_reset_validates_mask_length, devices=devices)


add_function_test(TestSteelReset, "test_reset_isolates_global_slot", test_reset_isolates_global_slot, devices=devices)


for _name, _func in (
    ("test_reset_validates_mask_length", test_reset_validates_mask_length),
    ("test_reset_isolates_global_slot", test_reset_isolates_global_slot),
):
    add_function_test(TestSteelReset, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split")


PLATFORM_TOP = 0.5


BOX_HALF = 0.1


def _stack_on_platform(device, num_worlds=2):
    """Two free boxes stacked on a position-driven platform slider, without a ground plane.

    The lower box couples the platform's dense rows to the box-box free-body rows, so the
    stack only rests if the two row families see each other's impulses.
    """
    world = newton.ModelBuilder()
    world.default_shape_cfg.mu = 0.5
    platform = world.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()))
    world.add_shape_box(platform, hx=0.4, hy=0.4, hz=0.05)
    slider = world.add_joint_prismatic(
        -1,
        platform,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()),
        target_ke=5.0e4,
        target_kd=5.0e2,
    )
    world.add_articulation([slider])
    for level in range(2):
        z = PLATFORM_TOP + BOX_HALF * (2 * level + 1)
        body = world.add_body(xform=wp.transform(wp.vec3(0.02 * level, 0.0, z), wp.quat_identity()))
        world.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
    builder = newton.ModelBuilder()
    builder.replicate(world, num_worlds, spacing=(2.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _run_split(model, solver, steps):
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def test_mixed_world_stack_rests(test, device):
    """Rest a box stack on an articulated platform: dense and free-body rows exchange impulses every sweep."""
    model = _stack_on_platform(device)
    solver = SolverSteel(model, solve_mode="split", iterations=16)
    test.assertTrue(solver._has_mixed_contacts)
    state = _run_split(model, solver, 240)
    solver.check_constraint_capacity()
    test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)
    test.assertGreater(int(solver._mf_constraint_count.numpy()[0]), 0)
    # The drive spring sags under the load; the boxes rest on the platform and on each other.
    body_q = state.body_q.numpy().reshape(model.world_count, 3, 7)
    platform_top = body_q[:, 0, 2] + 0.05
    for level in range(2):
        expected = platform_top + BOX_HALF * (2 * level + 1)
        np.testing.assert_allclose(body_q[:, level + 1, 2], expected, atol=2.0e-3, err_msg=f"box {level}")
    test.assertLess(float(np.abs(state.body_qd.numpy()).max()), 2.0e-2)


def test_mixed_world_matches_matrix_free_solve(test, device):
    """Converge to the matrix-free solve's resting state on the same mixed scene."""
    final = {}
    for solve_mode in ("split", "matrix_free"):
        model = _stack_on_platform(device)
        solver = SolverSteel(model, solve_mode=solve_mode, iterations=32)
        final[solve_mode] = _run_split(model, solver, 120).body_q.numpy()
    np.testing.assert_allclose(final["split"], final["matrix_free"], rtol=0.0, atol=2.0e-3)


def test_captured_mixed_step_matches_eager(test, device):
    """Replay captured collide + split step on a mixed scene like eager stepping."""
    trajectories = []
    for capture in (False, True):
        model = _stack_on_platform(device)
        solver = SolverSteel(model, solve_mode="split")
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        def substep(
            solver=solver, pipeline=pipeline, contacts=contacts, state_0=state_0, state_1=state_1, control=control
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))

        substep()
        if capture:
            with wp.ScopedCapture(device=device) as graph:
                substep()
            for _ in range(59):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(59):
                substep()
        trajectories.append(state_0.body_q.numpy())
    # The right-hand side accumulates the contributions of several articulations with atomics,
    # so two runs agree only to accumulated round-off (about 1e-5 here).
    np.testing.assert_allclose(trajectories[1][:, :3], trajectories[0][:, :3], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(trajectories[1][:, 3:], trajectories[0][:, 3:], rtol=0.0, atol=5.0e-4)


class TestSteelSplit(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires a CUDA device to compare against")
    def test_cpu_and_cuda_split_solves_agree(self):
        """Match the CPU split solve (scalar kernels) and the CUDA split solve (native kernels)."""
        final = {}
        for device in ("cpu", "cuda:0"):
            model = _stack_on_platform(device)
            final[device] = _run_split(model, SolverSteel(model, solve_mode="split"), 120).body_q.numpy()
        np.testing.assert_allclose(final["cpu"][:, :3], final["cuda:0"][:, :3], rtol=0.0, atol=1.0e-4)
        np.testing.assert_allclose(final["cpu"][:, 3:], final["cuda:0"][:, 3:], rtol=0.0, atol=5.0e-4)


for _name, _func, _devices in (
    ("test_mixed_world_stack_rests", test_mixed_world_stack_rests, get_test_devices()),
    ("test_mixed_world_matches_matrix_free_solve", test_mixed_world_matches_matrix_free_solve, get_cuda_test_devices()),
    ("test_captured_mixed_step_matches_eager", test_captured_mixed_step_matches_eager, get_cuda_test_devices()),
):
    add_function_test(TestSteelSplit, _name, _func, devices=_devices)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the xyzw quaternion ``q``."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _closure_frames(model, joint):
    """Return a loop-closing joint's parent, child, anchor transforms and hinge axis, read once from the model."""
    parent, child = int(model.joint_parent.numpy()[joint]), int(model.joint_child.numpy()[joint])
    x_p = model.joint_X_p.numpy()[joint].astype(np.float64)
    x_c = model.joint_X_c.numpy()[joint].astype(np.float64)
    axis = np.array([1.0, 0.0, 0.0])
    if int(model.joint_type.numpy()[joint]) == int(newton.JointType.REVOLUTE):
        axis = model.joint_axis.numpy()[int(model.joint_qd_start.numpy()[joint])].astype(np.float64)
    return parent, child, x_p, x_c, axis


def _closure_metrics(frames, body_q) -> tuple[float, float]:
    """Return a loop-closing joint's anchor gap [m] and hinge-axis misalignment [deg] for host body poses."""
    parent, child, x_p, x_c, axis = frames
    a_p, p_a = _rotate(x_p[3:], axis), x_p[:3]
    if parent >= 0:
        a_p = _rotate(body_q[parent, 3:], a_p)
        p_a = body_q[parent, :3] + _rotate(body_q[parent, 3:], x_p[:3])
    a_c = _rotate(body_q[child, 3:], _rotate(x_c[3:], axis))
    p_b = body_q[child, :3] + _rotate(body_q[child, 3:], x_c[:3])
    misalignment = np.degrees(np.arccos(np.clip(np.dot(a_p, a_c), -1.0, 1.0)))
    return float(np.linalg.norm(p_a - p_b)), float(misalignment)


def _build_world_hinged_swing(test, closure, start_angle=0.6):
    """Hang a body from the world by a hinge about x, with its center of mass offset along the axis.

    ``closure`` is ``"revolute"`` or ``"ball"`` for a loop joint beside a free joint, or ``"tree"``
    for the same hinge as an articulation joint. Gravity twists the body about the hinge's
    perpendicular, which only a REVOLUTE closure's hinge rows resist.
    """
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    start = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), start_angle)
    body = b.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), start),
        mass=1.0,
        com=wp.vec3(0.15, 0.0, -0.3),
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.01),
    )
    x_p = wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
    x_c = wp.transform_identity()
    if closure == "tree":
        joint = b.add_joint_revolute(
            parent=-1, child=body, axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=x_p, child_xform=x_c
        )
        b.joint_q[-1] = start_angle
        b.add_articulation([joint])
        return b, joint
    b.add_articulation([b.add_joint_free(child=body)])
    if closure == "revolute":
        add_loop_joint = functools.partial(
            b.add_joint_revolute, parent=-1, child=body, axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=x_p, child_xform=x_c
        )
    else:
        add_loop_joint = functools.partial(b.add_joint_ball, parent=-1, child=body, parent_xform=x_p, child_xform=x_c)
    joint = _expect_one_warning(
        test, newton.exceptions.NewtonWarning, "another joint already connects these bodies", add_loop_joint
    )
    return b, joint


def _build_square_loop(closure):
    """Build four revolute links around a horizontal unit square (axes z, x, y, z), closed at the base.

    The loop is spatial, so a BALL cut frees a rotation that gravity excites; a REVOLUTE cut about x
    keeps the square rigid.
    """
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
    base = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=0.5, inertia=inertia)
    joints = [
        b.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    ]
    corners = np.array([(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)])
    centers = np.array([(0.5, 0.0, 1.0), (1.0, 0.5, 1.0), (0.5, 1.0, 1.0), (0.0, 0.5, 1.0)])
    axes = [(0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    links, prev, prev_center = [], base, np.array((0.0, 0.0, 1.0))
    for corner, center, axis in zip(corners, centers, axes, strict=True):
        child = b.add_link(xform=wp.transform(wp.vec3(*center), wp.quat_identity()), mass=0.5, inertia=inertia)
        joints.append(
            b.add_joint_revolute(
                parent=prev,
                child=child,
                axis=wp.vec3(*axis),
                parent_xform=wp.transform(wp.vec3(*(corner - prev_center)), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(*(corner - center)), wp.quat_identity()),
            )
        )
        links.append(child)
        prev, prev_center = child, center
    b.add_articulation(joints)
    x_p = wp.transform_identity()
    x_c = wp.transform(wp.vec3(*(corners[0] - centers[3])), wp.quat_identity())
    if closure == "revolute":
        joint = b.add_joint_revolute(
            parent=base, child=links[3], axis=wp.vec3(1.0, 0.0, 0.0), parent_xform=x_p, child_xform=x_c
        )
    else:
        joint = b.add_joint_ball(parent=base, child=links[3], parent_xform=x_p, child_xform=x_c)
    return b, joint


def _run_closure(builder, joint, device, steps=500, **solver_options):
    """Step a model; return its worst closure gap [m], worst misalignment [deg], body poses per step and the solver."""
    model = builder.finalize(device=device)
    solver = SolverSteel(model, iterations=8, enable_joint_limits=True, **solver_options)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    poses = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, 1.0 / 500.0)
        state_0, state_1 = state_1, state_0
        poses.append(state_0.body_q.numpy().astype(np.float64))
    worst_gap, worst_misalignment = 0.0, 0.0
    if joint is not None:
        frames = _closure_frames(model, joint)
        for body_q in poses:
            gap, misalignment = _closure_metrics(frames, body_q)
            worst_gap, worst_misalignment = max(worst_gap, gap), max(worst_misalignment, misalignment)
    return worst_gap, worst_misalignment, np.array(poses), solver


def test_revolute_loop_world_hinge_holds_its_axis(test, device):
    """Swing a body hung from the world by a REVOLUTE loop joint like the same tree hinge; a BALL closure tumbles."""
    gap, misalignment, loop, _ = _run_closure(*_build_world_hinged_swing(test, "revolute"), device)
    test.assertLess(gap, 5.0e-4)
    test.assertLess(misalignment, 0.05)
    builder, _ = _build_world_hinged_swing(test, "tree")
    tree = _run_closure(builder, None, device)[2]
    loop, tree = loop[:, 0], tree[:, 0]
    swing = np.degrees(2.0 * np.arccos(np.clip(np.abs(tree[:, 6]), 0.0, 1.0)))
    test.assertGreater(swing.max() - swing.min(), 20.0, "the tree reference should swing")
    dots = np.clip(np.abs(np.sum(loop[:, 3:] * tree[:, 3:], axis=1)), 0.0, 1.0)
    test.assertLess(np.degrees(2.0 * np.arccos(dots)).max(), 1.0, "the loop hinge must swing like the tree hinge")
    test.assertLess(np.linalg.norm(loop[:, :3] - tree[:, :3], axis=1).max(), 2.0e-3)
    ball_misalignment = _run_closure(*_build_world_hinged_swing(test, "ball"), device)[1]
    test.assertGreater(ball_misalignment, 10.0, "a BALL closure should let the body tumble")


def test_revolute_loop_spatial_loop_stays_rigid(test, device):
    """Keep a spatial four-link loop closed by a REVOLUTE joint rigid under gravity; a BALL closure frees it."""
    gap, misalignment, _, _ = _run_closure(*_build_square_loop("revolute"), device)
    test.assertLess(gap, 5.0e-4)
    test.assertLess(misalignment, 0.05)
    test.assertGreater(_run_closure(*_build_square_loop("ball"), device)[1], 10.0)


def test_revolute_loop_pre_eliminates_five_rows(test, device):
    """Pre-eliminate all five rows of a REVOLUTE closure."""
    gap, misalignment, _, solver = _run_closure(
        *_build_square_loop("revolute"), device, enable_bilateral_preelimination=True
    )
    test.assertTrue(solver._preelim_active)
    test.assertEqual(int(solver._preelim_nB.numpy()[0]), 5)
    test.assertLess(gap, 1.0e-4)
    test.assertLess(misalignment, 0.01)


def _check_revolute_loop_axis_edit(test, device, captured):
    """Hold a loop hinge about x against a pull about y, then edit ``Model.joint_axis`` in place to y."""
    builder, joint = _build_world_hinged_swing(test, "revolute", start_angle=0.0)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, iterations=8, enable_joint_limits=True)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    start = state_0.body_q.numpy()[0, 3:].astype(np.float64)

    def step_pair():
        nonlocal state_0, state_1
        for _ in range(2):
            solver.step(state_0, state_1, control, None, 1.0 / 500.0)
            state_0, state_1 = state_1, state_0

    run = step_pair
    if captured:
        # Two steps per graph leave state_0 holding the newest state after every replay.
        step_pair()
        with wp.ScopedCapture(device=device) as capture:
            step_pair()

        def run():
            wp.capture_launch(capture.graph)

    def rotation_since_start():
        """Return the body's rotation vector [deg] relative to its start orientation."""
        x1, y1, z1, w1 = state_0.body_q.numpy()[0, 3:].astype(np.float64)
        x0, y0, z0, w0 = -start[0], -start[1], -start[2], start[3]
        rel = np.array(
            [
                w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0,
                w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0,
                w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0,
                w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0,
            ]
        )
        if rel[3] < 0.0:
            rel = -rel
        sin_half = np.linalg.norm(rel[:3])
        if sin_half == 0.0:
            return np.zeros(3)
        return np.degrees(rel[:3] / sin_half * 2.0 * np.arctan2(sin_half, rel[3]))

    # The center of mass is offset along x below the pivot, so gravity pulls about y.
    for _ in range(100):
        run()
    test.assertLess(np.linalg.norm(rotation_since_start()), 0.05, "a hinge about x must hold against a pull about y")
    axis = model.joint_axis.numpy()
    axis[int(model.joint_qd_start.numpy()[joint])] = (0.0, 1.0, 0.0)
    model.joint_axis.assign(axis)
    if not captured:
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    for _ in range(100):
        run()
    rotation = rotation_since_start()
    test.assertGreater(abs(rotation[1]), 5.0, "after the edit the body must swing about y")
    test.assertLess(np.hypot(rotation[0], rotation[2]), 0.05, "and only about y")
    gap, misalignment = _closure_metrics(_closure_frames(model, joint), state_0.body_q.numpy().astype(np.float64))
    test.assertLess(gap, 5.0e-4)
    test.assertLess(misalignment, 0.05)


def test_revolute_loop_axis_edit_applies_at_runtime(test, device):
    """Re-aim a loop hinge by editing ``Model.joint_axis`` and notifying ``JOINT_PROPERTIES``."""
    _check_revolute_loop_axis_edit(test, device, captured=False)


def test_revolute_loop_axis_edit_applies_between_replays(test, device):
    """Re-aim a loop hinge by editing ``Model.joint_axis`` in place between replays of one captured graph."""
    _check_revolute_loop_axis_edit(test, device, captured=True)


def test_revolute_loop_overflow_drops_all_five_rows(test, device):
    """Keep a REVOLUTE closure whole when its five rows fit, and drop it whole when one row is missing."""
    model = _build_world_hinged_swing(test, "revolute")[0].finalize(device=device)

    def row_count(capacity):
        solver = SolverSteel(
            model, solve_mode="matrix_free", articulation_constraint_max=capacity, warn_constraint_overflow=False
        )
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, 1.0 / 500.0)
        slot = int(solver._connect_slot.numpy()[0])
        return int(solver._constraint_count.numpy()[0]), slot, bool(solver.constraint_overflow.numpy()[0])

    rows, slot, overflow = row_count(32)
    test.assertGreaterEqual(rows, 5)
    test.assertGreaterEqual(slot, 0)
    test.assertFalse(overflow)
    test.assertEqual(row_count(rows), (rows, slot, False))
    test.assertEqual(row_count(rows - 1), (rows - 5, -1, True))


def test_revolute_loop_warns_about_ignored_limits_and_drives(test, device):
    """Warn at construction about limits, drive gains, damping, armature or a zero axis on a loop-closing REVOLUTE joint."""
    builder, joint = _build_world_hinged_swing(test, "revolute")
    dof = builder.joint_qd_start[joint]
    builder.joint_limit_lower[dof], builder.joint_limit_upper[dof] = -0.5, 0.5
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "loop-closing REVOLUTE joint .* limits"):
        SolverSteel(builder.finalize(device=device))
    builder, joint = _build_world_hinged_swing(test, "revolute")
    builder.joint_target_ke[builder.joint_qd_start[joint]] = 10.0
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "loop-closing REVOLUTE joint .* drive gains"):
        SolverSteel(builder.finalize(device=device))
    for name, attribute, value in (("damping", "joint_damping", 100.0), ("armature", "joint_armature", 0.1)):
        builder, joint = _build_world_hinged_swing(test, "revolute")
        getattr(builder, attribute)[builder.joint_qd_start[joint]] = value
        with test.assertWarnsRegex(newton.exceptions.NewtonWarning, f"loop-closing REVOLUTE joint .* {name}"):
            SolverSteel(builder.finalize(device=device))
    builder, joint = _build_world_hinged_swing(test, "revolute")
    builder.joint_axis[builder.joint_qd_start[joint]] = (0.0, 0.0, 0.0)
    with test.assertWarnsRegex(
        newton.exceptions.NewtonWarning, "loop-closing REVOLUTE joint .* zero or non-finite axis"
    ):
        SolverSteel(builder.finalize(device=device))
    with warnings.catch_warnings():
        warnings.simplefilter("error", newton.exceptions.NewtonWarning)
        SolverSteel(_build_world_hinged_swing(test, "revolute")[0].finalize(device=device))


def test_revolute_loop_warns_about_edits_it_ignores(test, device):
    """Warn on an eager notification after an edit to a zero or NaN axis or to damping of a loop-closing REVOLUTE joint."""
    model = _build_world_hinged_swing(test, "revolute")[0].finalize(device=device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", newton.exceptions.NewtonWarning)
        solver = SolverSteel(model)
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES | newton.ModelFlags.JOINT_DOF_PROPERTIES)
    dof = int(model.joint_qd_start.numpy()[1])
    axis = model.joint_axis.numpy()
    for bad_axis in ((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0)):
        axis[dof] = bad_axis
        model.joint_axis.assign(axis)
        with test.assertWarnsRegex(
            newton.exceptions.NewtonWarning, "loop-closing REVOLUTE joint 1 has a zero or non-finite axis"
        ):
            solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    damping = model.joint_damping.numpy()
    damping[dof] = 100.0
    model.joint_damping.assign(damping)
    with test.assertWarnsRegex(newton.exceptions.NewtonWarning, "loop-closing REVOLUTE joint 1 has damping"):
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_FORCE_PROPERTIES)


def test_revolute_loop_dof_gets_no_drive_or_velocity_limit_rows(test, device):
    """Give a loop-closing REVOLUTE joint's DOF no drive or velocity-limit row; the steps match a run without them."""

    def run(gain=False, **solver_options):
        builder, joint = _build_square_loop("revolute")
        if gain:
            builder.joint_target_ke[builder.joint_qd_start[joint]] = 50.0
            builder.joint_target_kd[builder.joint_qd_start[joint]] = 1.0
            builder.joint_target_q[builder.joint_q_start[joint]] = 0.3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", newton.exceptions.NewtonWarning)
            _, _, poses, solver = _run_closure(
                builder, joint, device, steps=100, solve_mode="matrix_free", **solver_options
            )
        return poses, int(solver._constraint_count.numpy()[0])

    reference, reference_rows = run(drive_mode="rows")
    with test.subTest("drive rows"):
        poses, rows = run(gain=True, drive_mode="rows")
        test.assertEqual(rows, reference_rows)
        np.testing.assert_array_equal(poses, reference)
    with test.subTest("velocity-limit rows"):
        poses, rows = run(enable_joint_velocity_limits=True)
        # The four tree hinges take two rows each, the loop hinge none.
        test.assertEqual(rows, reference_rows + 8)
        np.testing.assert_array_equal(poses, reference)


def test_revolute_loop_realigns_to_the_nearer_axis_direction(test, device):
    """Realign a REVOLUTE closure that starts 179 degrees off along the short way to the hinge line."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = b.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.01),
    )
    b.add_articulation([b.add_joint_free(child=body)])
    flipped = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.radians(179.0)))
    joint = _expect_one_warning(
        test,
        newton.exceptions.NewtonWarning,
        "another joint already connects these bodies",
        functools.partial(
            b.add_joint_revolute,
            parent=-1,
            child=body,
            axis=wp.vec3(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(), flipped),
        ),
    )
    _, _, poses, solver = _run_closure(b, joint, device, steps=200)
    frames = _closure_frames(solver.model, joint)
    line_error = [min(m, 180.0 - m) for m in (_closure_metrics(frames, body_q)[1] for body_q in poses)]
    test.assertLess(max(line_error), 1.01)
    test.assertLess(line_error[-1], 0.05)


class TestSteelRevoluteLoop(unittest.TestCase):
    pass


for _name in (
    "test_revolute_loop_world_hinge_holds_its_axis",
    "test_revolute_loop_spatial_loop_stays_rigid",
    "test_revolute_loop_pre_eliminates_five_rows",
    "test_revolute_loop_axis_edit_applies_at_runtime",
    "test_revolute_loop_axis_edit_applies_between_replays",
    "test_revolute_loop_overflow_drops_all_five_rows",
    "test_revolute_loop_warns_about_ignored_limits_and_drives",
    "test_revolute_loop_warns_about_edits_it_ignores",
    "test_revolute_loop_dof_gets_no_drive_or_velocity_limit_rows",
    "test_revolute_loop_realigns_to_the_nearer_axis_direction",
):
    add_function_test(TestSteelRevoluteLoop, _name, globals()[_name], devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
