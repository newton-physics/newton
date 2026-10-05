# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for connect (loop-closure) constraint rows in SolverFeatherPGS."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton


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


def _run(steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, **solver_kwargs
    )
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


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSConnect(unittest.TestCase):
    def test_loop_joint_presence_is_stable(self):
        """Verify a trailing BALL loop joint no longer NaNs the articulation solve."""
        _, _, state = _run(steps=120, crank_target=0.0)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state.joint_qd.numpy()).all())

    def test_loop_closure_is_enforced(self):
        """Verify the connect rows hold the four-bar's loop anchors together under drive."""
        _, model, state = _run(steps=720, crank_target=0.6)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        gap = _loop_anchor_gap(model, state)
        self.assertLess(gap, 2.0e-3, f"loop anchor gap {gap * 1e3:.2f} mm — four-bar not closed")

    def test_rocker_follows_crank(self):
        """Verify the undriven rocker moves coherently with the driven crank via the loop."""
        _, _model, state = _run(steps=720, crank_target=0.6)
        q = state.joint_q.numpy()
        # Parallel four-bar (equal crank/rocker lengths): rocker angle tracks crank angle.
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_connect_ownership_survives_interleaved_articulation(self):
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
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free")

        self.assertEqual(int(model.joint_articulation.numpy()[loop_joint]), -1)
        np.testing.assert_array_equal(
            solver._model_plan.loop_joint_articulation,
            np.array([-1, -1, -1, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(solver._connect_art.numpy(), np.array([0], dtype=np.int32))

    def test_standalone_world_root_is_not_loop_joint(self):
        """Do not treat an unrelated unowned world-root joint as a closure."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)

        articulated = b.add_link()
        b.add_shape_box(articulated, hx=0.05, hy=0.05, hz=0.1)
        tree_joint = b.add_joint_revolute(parent=-1, child=articulated, axis=newton.Axis.Y)
        b.add_articulation([tree_joint], label="articulated")

        standalone = b.add_link()
        b.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
        standalone_joint = b.add_joint_fixed(parent=-1, child=standalone)

        model = b.finalize()
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0, -1])
        self.assertEqual(standalone_joint, 1)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
        )
        self.assertFalse(solver._has_loop_joints)
        self.assertEqual(solver._connect_count, 0)
        np.testing.assert_array_equal(
            solver._model_plan.loop_joint_articulation,
            np.array([-1, -1], dtype=np.int32),
        )

    def test_connect_survives_foreign_joint_between_tree_and_loop(self):
        """Attribute loop joints by body ownership when another articulation interleaves.

        ``add_usd`` of a subtree containing a closed-loop robot AND a floating rigid
        body creates the body's FREE joint during traversal but appends the equality
        loop joints at the very end of the parse — so the closures no longer trail
        their articulation's joint-index range (the trailing sentinel range of the LAST
        articulation swallows them instead). Index-range attribution then silently
        produced ZERO connect rows and the closed linkage fell open (measured on the
        Robotiq 2F-85 through the IsaacLab scene path: 107 mm anchor gap, fingers
        collapsed). The closure must be attributed via its child body's articulation.
        This test reconstructs that joint ordering directly: four-bar tree, then a
        free-jointed object articulation, THEN the closure ball joint.
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

        # a second articulation's joint lands BETWEEN the tree and the loop joint
        body = b.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.05)
        j_free = b.add_joint_free(child=body)
        b.add_articulation([j_free], label="free_object")

        # the loop closure, appended last (the add_usd equality-parse ordering)
        b.add_joint_ball(
            parent=coupler,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        )
        b.joint_target_ke[0] = 50.0
        b.joint_target_kd[0] = 5.0
        b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
        model = b.finalize()

        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
        self.assertEqual(
            getattr(solver, "_connect_count", 0),
            1,
            "closure row lost — loop joint attributed to the wrong articulation",
        )

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
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(gap_max, 2.0e-3, f"loop closure not enforced (anchor gap {gap_max:.4f} m)")


# ---------------------------------------------------------------------------
# Closures with a prescribed (kinematic or world) parent
# ---------------------------------------------------------------------------

_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))
_REL_P = np.array([0.0, 0.0, -0.2])


def _pgs_mode():
    return "matrix_free" if wp.get_device().is_cuda else "split"


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


def _anchor_gaps(model, state, joints, solver):
    """World-space parent/child anchor distances [m] using the solver's live anchors."""
    bq = state.body_q.numpy().astype(np.float64)
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()

    def anchor(body, local):
        if body < 0:
            return np.asarray(local, dtype=np.float64)
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        w = wp.transform_point(t, wp.vec3(*local))
        return np.array([w[0], w[1], w[2]])

    gaps = []
    for j in joints:
        anchor_p, anchor_c = solver.loop_joint_anchors(j)
        gaps.append(float(np.linalg.norm(anchor(int(jp[j]), anchor_p) - anchor(int(jc[j]), anchor_c))))
    return gaps


def _prescribe_carrier(model, solver, state, articulation, t, v, omega):
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
    solver.notify_state_changed(articulation)


class TestFeatherPGSPrescribedParentConnect(unittest.TestCase):
    def test_kinematic_parent_closure_carries_load(self):
        """A load pinned to a moving kinematic carrier must ride along with it.

        Without prescribed-parent support the closure is ignored (it crosses
        articulations) and the load free-falls; with it, the three point closures hold
        the load at its relative pose while the carrier translates and spins.
        """
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=_pgs_mode(), pgs_iterations=16, pgs_beta=0.2)
        self.assertFalse([w for w in caught if "loop joint" in str(w.message)], [str(w.message) for w in caught])
        articulation = wp.array([int(solver.body_to_articulation.numpy()[carrier])], dtype=wp.int32)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0
        v = np.array([0.3, 0.0, 0.1])
        omega = np.array([0.0, 0.0, 1.5])
        for i in range(240):
            _prescribe_carrier(model, solver, state_0, articulation, i * dt, v, omega)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        _prescribe_carrier(model, solver, state_0, articulation, 240 * dt, v, omega)

        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        expected = wp.transform_point(carrier_t, wp.vec3(*_REL_P))
        error = np.linalg.norm(bq[load, :3] - np.array([expected[0], expected[1], expected[2]]))
        self.assertLess(error, 3.0e-3, f"load lagged the carrier by {error:.4f} m")
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)
        # The carrier actually moved, so the test exercised a non-trivial target velocity.
        self.assertGreater(np.linalg.norm(bq[carrier, :3] - np.array([0.0, 0.0, 1.0])), 0.25)

    def test_world_parent_closure_holds_hanging_load(self):
        b, _, load, joints = _build_carried_load(world_parent=True)
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=_pgs_mode(), pgs_iterations=16, pgs_beta=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(np.linalg.norm(bq[load, :3] - (np.array([0.0, 0.0, 1.0]) + _REL_P)), 3.0e-3)
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)

    def test_runtime_enable_and_anchor_update(self):
        """Closures can be released, re-targeted at the measured pose and re-engaged."""
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=_pgs_mode(), pgs_iterations=16, pgs_beta=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0

        def run(steps):
            nonlocal state_0, state_1
            for _ in range(steps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

        for j in joints:
            solver.set_loop_joint_enabled(j, False)
        z0 = float(state_0.body_q.numpy()[load, 2])
        run(48)
        bq = state_0.body_q.numpy().astype(np.float64)
        dropped = z0 - float(bq[load, 2])
        self.assertGreater(dropped, 0.15, "released load should free-fall")

        # Re-target the closures at the measured relative pose, then re-engage them.
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        load_t = wp.transform(wp.vec3(*bq[load, :3]), wp.quat(*bq[load, 3:]))
        rel = wp.transform_multiply(wp.transform_inverse(carrier_t), load_t)
        for j, c in zip(joints, _CHILD_ANCHORS, strict=True):
            p = wp.transform_point(rel, wp.vec3(*c))
            solver.set_loop_joint_anchors(j, (p[0], p[1], p[2]), c)
            solver.set_loop_joint_enabled(j, True)
        held_z = float(bq[load, 2])
        run(120)
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(abs(float(bq[load, 2]) - held_z), 3.0e-3, "re-engaged closure should hold the load")
        for gap in _anchor_gaps(model, state_0, joints, solver):
            self.assertLess(gap, 3.0e-3)

        for j in joints:
            solver.set_loop_joint_enabled(j, False)
        run(48)
        self.assertGreater(held_z - float(state_0.body_q.numpy()[load, 2]), 0.1, "released again: must fall")

    def test_unknown_joint_is_rejected(self):
        b, _, _, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=_pgs_mode())
        with self.assertRaises(ValueError):
            solver.set_loop_joint_enabled(joints[0] + 100, False)


if __name__ == "__main__":
    unittest.main()
