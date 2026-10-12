# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Connect (loop-closure) rows of SolverFeatherPGS."""

import functools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_four_bar():
    """Build a planar four-bar: two grounded revolute chains closed by a BALL loop joint.

    Ground pivots at x=0 and x=0.4; crank and rocker hang down 0.2 m; the coupler
    connects the crank tip to the rocker tip through the loop closure. The crank is
    position-driven; the rocker is undriven, so any coherent motion it does comes from
    the closed loop.
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

    # Loop closure: coupler tip pinned to rocker tip (a trailing BALL loop joint,
    # matching how the MJCF importer closes `connect` equalities).
    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )

    # Drive the crank only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b


def _loop_anchor_gap(model, state) -> float:
    """World-space distance between the loop joint's parent and child anchors [m]."""
    bq = state.body_q.numpy().reshape(-1, 7).astype(np.float64)
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    j = int(np.nonzero(jt == int(newton.JointType.BALL))[0][-1])

    def anchor(body, X):
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        a = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
        w = wp.transform_multiply(t, a)
        return np.array([w.p[0], w.p[1], w.p[2]])

    return float(np.linalg.norm(anchor(int(jp[j]), Xp[j]) - anchor(int(jc[j]), Xc[j])))


def _run(device, steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, **solver_kwargs)
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


# -- Four-bar closure ------------------------------------------------------------------


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
    eager = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
    captured = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free")

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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", **solver_kwargs)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
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
    """Reject non-BALL loop joints and closures between two dynamic articulations."""
    b = _build_four_bar()
    _expect_one_warning(
        test, UserWarning, "another joint already connects these bodies", lambda: b.add_joint_fixed(parent=1, child=2)
    )
    with test.assertRaisesRegex(NotImplementedError, "only BALL loop-closing joints"):
        SolverFeatherPGS(b.finalize(device=device), pgs_mode="matrix_free")

    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    first = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    second = b.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 1.0), wp.quat_identity()))
    for body in (first, second):
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        b.add_joint_ball(parent=first, child=second)
    with test.assertRaisesRegex(NotImplementedError, "connects articulations"):
        SolverFeatherPGS(b.finalize(device=device), pgs_mode="matrix_free")


def test_closure_row_overflow_is_reported(test, device):
    """Drop a closure that does not fit whole, and flag the overflow."""
    model = _build_four_bar().finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        dense_max_constraints=2,
        joint_limit_activation_gap=0.0,
        warn_constraint_overflow=False,
    )
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertEqual(int(solver.connect_slot.numpy()[0]), -1)
    test.assertEqual(int(solver.constraint_count.numpy()[0]), 0)
    test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))


# ---------------------------------------------------------------------------
# Closures with a prescribed (kinematic or world) parent
# ---------------------------------------------------------------------------

_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))
_REL_P = np.array([0.0, 0.0, -0.2])


def _build_carried_load(*, enabled: bool = True, world_parent: bool = False):
    """A kinematic carrier holding a dynamic box through three BALL loop joints.

    Three non-collinear point closures pin all six relative degrees of freedom, so the
    load must follow the carrier as if welded at ``_REL_P`` below it. With
    ``world_parent`` the carrier is the world and the anchors are world points.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    carrier_p = np.array([0.0, 0.0, 1.0])
    carrier = -1
    if not world_parent:
        carrier = b.add_body(xform=wp.transform(wp.vec3(*carrier_p), wp.quat_identity()), is_kinematic=True)
        b.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
    load = b.add_body(xform=wp.transform(wp.vec3(*(carrier_p + _REL_P)), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
    joints = []
    # The closures intentionally parallel the load's free joint and each other.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*another joint already connects these bodies", category=UserWarning
        )
        for c in _CHILD_ANCHORS:
            p = (_REL_P if not world_parent else carrier_p + _REL_P) + np.array(c)
            joints.append(
                b.add_joint_ball(
                    parent=carrier,
                    child=load,
                    parent_xform=wp.transform(wp.vec3(*p), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                    enabled=enabled,
                )
            )
    return b, carrier, load, joints


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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
    articulation = wp.array([int(solver.body_to_articulation.numpy()[carrier])], dtype=wp.int32, device=device)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
    state_0, state_1 = model.state(), model.state()
    z0 = float(state_0.body_q.numpy()[load, 2])
    for _ in range(48):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertGreater(z0 - float(state_0.body_q.numpy()[load, 2]), 0.15)
    test.assertEqual(solver.connect_slot.numpy().tolist(), [-1, -1, -1])

    _set_joint_enabled(model, joints, True)
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertTrue(np.all(solver.connect_slot.numpy() >= 0))


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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", joint_limit_activation_gap=0.0)
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
        SolverFeatherPGS(model, pgs_mode="matrix_free", joint_limit_activation_gap=0.0)


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
                UserWarning,
                "another joint already connects these bodies",
                functools.partial(b.add_mjcf, mjcf, convert_mjc_equality_constraints=True),
            )
        else:
            b.add_mjcf(mjcf, convert_mjc_equality_constraints=False)
        model = b.finalize(device=device)
        test.assertEqual(model.mujoco.equality_constraint_count, 1)
        if not convert:
            with test.assertRaisesRegex(NotImplementedError, "equality"):
                SolverFeatherPGS(model, pgs_mode="matrix_free")
            continue
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16)
        test.assertEqual(solver._connect_count, 1)
        state_0, state_1 = model.state(), model.state()
        for _ in range(120):
            solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        # The closure at the free end holds the hinge against gravity.
        test.assertLess(abs(float(state_0.joint_q.numpy()[0])), 1.0e-2)


# ---------------------------------------------------------------------------
# Closure state read from the model every step
# ---------------------------------------------------------------------------

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
    options = {"pgs_mode": "matrix_free", "pgs_iterations": 16, "pgs_beta": 0.1, **solver_kwargs}
    solver = SolverFeatherPGS(model, **options)
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
    test.assertEqual(int(solver.connect_slot.numpy()[0]), -1)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
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
    slots = solver.connect_slot.numpy()
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
    for name in ("feather_pgs", "xpbd"):
        b, _, load, joints = _build_carried_load(world_parent=True)
        model = b.finalize(device=device)
        if name == "feather_pgs":
            solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
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
    np.testing.assert_allclose(trajectories["feather_pgs"], trajectories["xpbd"], atol=3.0e-3)


def test_closure_owner_stays_awake_and_releases(test, device):
    """Keep a closure's articulation awake under sleeping, so a released closure takes effect."""
    b = _build_four_bar()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    model = b.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=16,
        pgs_beta=0.1,
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
    test.assertEqual(solver.sleeping.art_awake.numpy().tolist(), [1, 0])
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
            SolverFeatherPGS(model, pgs_mode="matrix_free", articulated_contact_response="propagation")


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
        SolverFeatherPGS(model, pgs_mode="matrix_free")


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


class TestFeatherPGSConnect(unittest.TestCase):
    pass


class TestFeatherPGSPrescribedParentConnect(unittest.TestCase):
    pass


class TestFeatherPGSConnectPropagation(unittest.TestCase):
    pass


class TestFeatherPGSConnectModelEdits(unittest.TestCase):
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
):
    add_function_test(TestFeatherPGSConnect, _name, globals()[_name], devices=cuda_devices)
for _name in (
    "test_kinematic_parent_closure_carries_load",
    "test_world_parent_closure_holds_hanging_load",
    "test_model_edits_release_and_reanchor_closures",
    "test_disabled_loop_joint_starts_released",
    "test_kinematic_parent_turning_dynamic_is_rejected",
):
    add_function_test(TestFeatherPGSPrescribedParentConnect, _name, globals()[_name], devices=cuda_devices)
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
    add_function_test(TestFeatherPGSConnectModelEdits, _name, globals()[_name], devices=cuda_devices)
add_function_test(
    TestFeatherPGSConnectPropagation,
    "test_propagation_standalone_world_root_is_not_loop_joint",
    test_propagation_standalone_world_root_is_not_loop_joint,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
