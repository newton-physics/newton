# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for identity-matched FeatherPGS contact warm start."""

import inspect
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    gather_dense_warmstart,
    gather_mf_warmstart,
    gather_propagation_warmstart,
    prepare_world_impulses,
)


def _build_press():
    """Build a 1-DOF prismatic press: an articulated box driven down onto the ground.

    The drive target sits below the contact height, so after touchdown the press
    stalls against the ground under a steady drive force — a persistent dense
    contact family (articulated vs. static routes to the dense path) whose
    converged impulse is constant per step. Exactly the regime warm starting
    carries impulses across, and the regime where the pre-fix carry diverged.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        target_ke=2.0e3,
        target_kd=50.0,
    )
    builder.add_articulation([joint], label="press")
    return builder.finalize()


def _run_press(
    steps: int,
    warm_kwargs: dict,
    contact_matching: str | None = "sticky",
    pgs_mode: str = "matrix_free",
    articulated_contact_response: str = "immediate",
    propagation_cached_response: bool = True,
):
    """Drive the press to a stall and record per-step contact-impulse sums and speeds.

    Returns (impulse_sum_per_step, |joint_qd|_per_step, final_state).
    """
    model = _build_press()
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode=pgs_mode,
        articulated_contact_response=articulated_contact_response,
        propagation_cached_response=propagation_cached_response,
        pgs_iterations=8,
        pgs_beta=0.1,
        **warm_kwargs,
    )
    pipeline_kwargs = {} if contact_matching is None else {"contact_matching": contact_matching}
    pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = -0.25  # 5 cm below touchdown: sustained press after the stall
    control.joint_target_q.assign(targets)

    impulse_sums = np.zeros(steps)
    speeds = np.zeros(steps)
    for i in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        if articulated_contact_response.startswith("propagation"):
            counts = solver.propagation_constraint_count.numpy()
            rows = solver.propagation_row_type.numpy()[0, : counts[0]]
            lam = solver.propagation_impulses.numpy()[0, : counts[0]]
        else:
            counts = solver.constraint_count.numpy()
            rows = solver.row_type.numpy()[0, : counts[0]]
            lam = solver.impulses.numpy()[0, : counts[0]]
        impulse_sums[i] = np.abs(lam[rows == PGS_CONSTRAINT_TYPE_CONTACT]).sum()
        speeds[i] = np.abs(state_0.joint_qd.numpy()[0])
    return impulse_sums, speeds, state_0


def _matched_history(device):
    """Gather arguments for generation 1 of stream 1, matched against the solved generation 0."""
    return (
        wp.zeros(1, dtype=wp.int32, device=device),  # match generation
        wp.ones(1, dtype=wp.int32, device=device),  # contact generation
        1,  # contact stream
        wp.zeros(1, dtype=wp.int32, device=device),  # solved generation
        wp.ones(1, dtype=wp.int32, device=device),  # solved stream
    )


def _gather_dense_rows(
    *,
    prev_impulses,
    prev_types,
    prev_parents,
    prev_slots,
    current_types,
    current_parents,
    current_slots,
    match_indices,
    count,
    decay=1.0,
    dt_scale=1.0,
):
    """Launch the dense identity gather on CPU and return its row buffer."""
    max_c = len(current_types)
    n = len(current_slots)
    impulses = wp.zeros((1, max_c), dtype=wp.float32, device="cpu")
    wp.launch(
        gather_dense_warmstart,
        dim=n,
        inputs=[
            wp.array([n], dtype=wp.int32, device="cpu"),
            wp.array([0] * n, dtype=wp.int32, device="cpu"),
            wp.array(current_slots, dtype=wp.int32, device="cpu"),
            wp.array([0] * n, dtype=wp.int32, device="cpu"),
            wp.array(match_indices, dtype=wp.int32, device="cpu"),
            *_matched_history("cpu"),
            wp.array(prev_slots, dtype=wp.int32, device="cpu"),
            wp.array([prev_impulses], dtype=wp.float32, device="cpu"),
            wp.array([prev_types], dtype=wp.int32, device="cpu"),
            wp.array([prev_parents], dtype=wp.int32, device="cpu"),
            wp.array([count], dtype=wp.int32, device="cpu"),
            wp.array([current_types], dtype=wp.int32, device="cpu"),
            wp.array([current_parents], dtype=wp.int32, device="cpu"),
            wp.array([[0.0, 0.0, 1.0]] * n, dtype=wp.vec3, device="cpu"),
            wp.array([[0.0, 0.0, 1.0]] * n, dtype=wp.vec3, device="cpu"),
            wp.full((1, max_c), 100.0, dtype=wp.float32, device="cpu"),
            decay,
            dt_scale,
            max_c,
        ],
        outputs=[impulses],
        device="cpu",
    )
    return impulses.numpy()[0]


class TestFeatherPGSIdentityWarmstartKernel(unittest.TestCase):
    def test_noncontact_dense_cache_is_cold_initialized(self):
        """The dense initializer preserves no row without an identity contract."""
        impulses = wp.array([[1.0, 2.0, 3.0, 4.0]], dtype=wp.float32, device="cpu")
        wp.launch(
            prepare_world_impulses,
            dim=1,
            inputs=[
                wp.array([4], dtype=wp.int32, device="cpu"),
                4,
                1,
            ],
            outputs=[impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.zeros(4, dtype=np.float32))

    def test_two_contact_friction_span_transitions_do_not_cross_seed(self):
        """A neighboring contact never owns the source or destination tangent row."""
        c, f, dead = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, -1

        # A: 3 -> 1 rows, B: 1 -> 3 rows. A's previous second tangent lands at
        # B's current first tangent by raw offset and must not be written there.
        got = _gather_dense_rows(
            prev_impulses=[11.0, 22.0, 33.0, 55.0],
            prev_types=[c, f, f, c],
            prev_parents=[dead, 0, 0, dead],
            prev_slots=[0, 3],
            current_types=[c, c, f, f],
            current_parents=[dead, dead, 1, 1],
            current_slots=[0, 1],
            match_indices=[0, 1],
            count=4,
        )
        np.testing.assert_array_equal(got, np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32))

        # A: 1 -> 3 rows, B: 3 -> 1 rows. A's second destination tangent
        # overlaps B's previous first tangent and must stay cold.
        got = _gather_dense_rows(
            prev_impulses=[11.0, 55.0, 66.0, 77.0],
            prev_types=[c, c, f, f],
            prev_parents=[dead, dead, 1, 1],
            prev_slots=[0, 1],
            current_types=[c, f, f, c],
            current_parents=[dead, 0, 0, dead],
            current_slots=[0, 3],
            match_indices=[0, 1],
            count=4,
        )
        np.testing.assert_array_equal(got, np.array([11.0, 0.0, 0.0, 55.0], dtype=np.float32))

    def test_slot_churn_uses_identity_and_scales_dt(self):
        """Contact-order and slot churn follows match identity, not row index."""
        c, dead = PGS_CONSTRAINT_TYPE_CONTACT, -1
        got = _gather_dense_rows(
            prev_impulses=[55.0, 0.0, 0.0, 0.0, 11.0],
            prev_types=[c, dead, dead, dead, c],
            prev_parents=[dead] * 5,
            prev_slots=[4, 0],  # previous sorted contacts A, B
            current_types=[dead, c, dead, dead, c],
            current_parents=[dead] * 5,
            current_slots=[1, 4],  # current sorted contacts B, A
            match_indices=[1, 0],
            count=5,
            decay=0.5,
            dt_scale=4.0,
        )
        np.testing.assert_array_equal(got, np.array([0.0, 110.0, 0.0, 0.0, 22.0], dtype=np.float32))

    def test_mf_and_propagation_share_friction_ownership_rule(self):
        """Every routed family rejects A's old tangent from B's new span."""
        c, f, dead = PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, -1
        prev_impulses = wp.array([[11.0, 22.0, 33.0, 55.0]], dtype=wp.float32, device="cpu")
        prev_types = wp.array([[c, f, f, c]], dtype=wp.int32, device="cpu")
        prev_parents = wp.array([[dead, 0, 0, dead]], dtype=wp.int32, device="cpu")
        current_types = wp.array([[c, c, f, f]], dtype=wp.int32, device="cpu")
        current_parents = wp.array([[dead, dead, 1, 1]], dtype=wp.int32, device="cpu")
        contact_count = wp.array([2], dtype=wp.int32, device="cpu")
        contact_slots = wp.array([0, 1], dtype=wp.int32, device="cpu")
        contact_world = wp.array([0, 0], dtype=wp.int32, device="cpu")
        match_indices = wp.array([0, 1], dtype=wp.int32, device="cpu")
        prev_slots = wp.array([0, 3], dtype=wp.int32, device="cpu")
        constraint_count = wp.array([4], dtype=wp.int32, device="cpu")

        mf_impulses = wp.zeros((1, 4), dtype=wp.float32, device="cpu")
        wp.launch(
            gather_mf_warmstart,
            dim=2,
            inputs=[
                contact_count,
                wp.array([1, 1], dtype=wp.int32, device="cpu"),
                contact_slots,
                contact_world,
                match_indices,
                *_matched_history("cpu"),
                prev_slots,
                prev_impulses,
                prev_types,
                prev_parents,
                constraint_count,
                current_types,
                current_parents,
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.full((1, 4), 100.0, dtype=wp.float32, device="cpu"),
                1.0,
                1.0,
                4,
            ],
            outputs=[mf_impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(mf_impulses.numpy()[0], np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32))

        propagation_impulses = wp.zeros((1, 4), dtype=wp.float32, device="cpu")
        wp.launch(
            gather_propagation_warmstart,
            dim=2,
            inputs=[
                contact_count,
                wp.array([2, 2], dtype=wp.int32, device="cpu"),
                contact_slots,
                contact_world,
                match_indices,
                *_matched_history("cpu"),
                prev_slots,
                prev_impulses,
                prev_types,
                prev_parents,
                constraint_count,
                current_types,
                current_parents,
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.array([[0.0, 0.0, 1.0]] * 2, dtype=wp.vec3, device="cpu"),
                wp.full((1, 4), 100.0, dtype=wp.float32, device="cpu"),
                1.0,
                1.0,
                4,
            ],
            outputs=[propagation_impulses],
            device="cpu",
        )
        np.testing.assert_array_equal(
            propagation_impulses.numpy()[0], np.array([11.0, 55.0, 0.0, 0.0], dtype=np.float32)
        )

    def test_current_slot_is_bounded_by_constraint_count(self):
        got = _gather_dense_rows(
            prev_impulses=[9.0, 0.0, 0.0, 0.0, 0.0],
            prev_types=[PGS_CONSTRAINT_TYPE_CONTACT, -1, -1, -1, -1],
            prev_parents=[-1] * 5,
            prev_slots=[0],
            current_types=[-1, -1, -1, -1, PGS_CONSTRAINT_TYPE_CONTACT],
            current_parents=[-1] * 5,
            current_slots=[4],
            match_indices=[0],
            count=4,
        )
        np.testing.assert_array_equal(got, np.zeros(5, dtype=np.float32))

    def test_constructor_layout_and_decay_validation(self):
        parameters = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
        start = parameters.index("pgs_warmstart")
        self.assertEqual(
            parameters[start : start + 4],
            ("pgs_warmstart", "mf_warmstart", "mf_warmstart_decay", "pgs_mode"),
        )
        self.assertGreater(
            parameters.index("pgs_warmstart_decay"), parameters.index("articulation_pair_contact_gap_gate")
        )

        model = newton.ModelBuilder().finalize(device="cpu")
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, pgs_warmstart_decay=value)

        legacy = newton.solvers.SolverFeatherPGS(
            model,
            mf_warmstart=True,
            mf_warmstart_decay=0.25,
        )
        self.assertTrue(legacy.pgs_warmstart)
        self.assertEqual(legacy.pgs_warmstart_decay, 0.25)
        self.assertEqual(legacy._mf_warmstart_decay, legacy.pgs_warmstart_decay)

    def test_contacts_none_is_valid(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split", pgs_warmstart=True)
        solver.step(model.state(), model.state(), model.control(), None, 1.0 / 60.0)

    def test_split_mode_holds_the_press(self):
        """The identity carry works on the CPU-supported dense-row route."""
        impulses, speeds, _ = _run_press(120, {"pgs_warmstart": True}, pgs_mode="split")
        self.assertGreater(float(impulses[-1]), 0.0)
        self.assertLess(float(np.ptp(impulses[-10:])), 1.0e-5)
        self.assertLess(float(speeds[-10:].max()), 1.0e-4)


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSIdentityWarmstart(unittest.TestCase):
    def test_real_contact_insertion_moves_slots_without_cross_seeding(self):
        """A newly inserted lower-key contact shifts a persistent contact's real MF slot."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
        shape_a = builder.add_shape_sphere(body_a, radius=0.1)
        body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.099), wp.quat_identity()))
        shape_b = builder.add_shape_sphere(body_b, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            pgs_warmstart=True,
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        pipeline.collide(state_0, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        old_slot = int(solver._ws_prev_slot_sorted.numpy()[0])
        old_impulse = float(solver._ws_prev_mf_impulses.numpy()[0, old_slot])
        self.assertGreater(old_impulse, 0.0)

        # Bring the lower shape-id sphere into contact. Sorting inserts it
        # before B, so B moves to a new solver slot while its match points to
        # the previous frame's sole contact.
        q = state_1.body_q.numpy()
        q[body_a][2] = 0.099
        state_1.body_q.assign(q)
        pipeline.collide(state_1, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(count, 2)
        match = contacts.rigid_contact_match_index.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        b_indices = np.flatnonzero((shape0 == shape_b) | (shape1 == shape_b))
        a_indices = np.flatnonzero((shape0 == shape_a) | (shape1 == shape_a))
        self.assertEqual(len(b_indices), 1)
        self.assertEqual(len(a_indices), 1)
        b_contact = int(b_indices[0])
        a_contact = int(a_indices[0])
        self.assertEqual(int(match[b_contact]), 0)
        self.assertLess(int(match[a_contact]), 0)

        solver.pgs_iterations = 0
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)
        slots = solver.contact_slot.numpy()[:count]
        b_slot = int(slots[b_contact])
        a_slot = int(slots[a_contact])
        self.assertNotEqual(b_slot, old_slot, "test did not produce actual solver-slot churn")
        impulses = solver.mf_impulses.numpy()[0]
        self.assertAlmostEqual(float(impulses[b_slot]), old_impulse, delta=1.0e-6)
        self.assertEqual(float(impulses[a_slot]), 0.0)

    def test_identity_warmstart_holds_static_press(self):
        """Identity warm start keeps a stalled press at the cold equilibrium.

        Regression net for the warm-start velocity accounting: carried impulses
        are installed into the starting velocity exactly once. A missing install
        accumulates the ledger (``lambda_{n+1} = lambda_n + lambda*``, the
        historical divergence); a duplicated install halves it (measured 0.5x
        cold when an earlier identity path added a second fold on top of
        ``_stage6_apply_impulses_world``). The identity-matched path must also
        stay quiet and match the cold ledger while seeding by contact identity
        rather than raw slot index.
        """
        steps = 240
        stall = slice(120, None)  # well past touchdown + transient

        lam_cold, _speed_cold, _ = _run_press(steps, {})
        lam_warm, speed_warm, state = _run_press(steps, {"pgs_warmstart": True})

        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

        cold_end = lam_cold[-10:].mean()
        warm_end = lam_warm[-10:].mean()

        # The solver installs carried impulses into the starting velocity
        # (_stage6_apply_impulses_world under pgs_warmstart), so warm and cold
        # must share an equilibrium. A duplicated velocity install shows
        # up here as a halved impulse ledger (measured 0.5x when the matched
        # path folded the seed a second time); a missing install shows up as an
        # accumulating ledger (lambda_{n+1} = lambda_n + lambda*).
        warm_growth = lam_warm[-10:].mean() / max(lam_warm[stall][:10].mean(), 1e-12)
        self.assertLess(warm_growth, 1.25, f"warm-start impulse grew x{warm_growth:.2f} at a stall")
        self.assertGreater(warm_end, 0.7 * cold_end, f"warm impulse ledger {warm_end:.3f} below cold {cold_end:.3f}")
        self.assertLess(warm_end, 1.4 * cold_end, f"warm impulse ledger {warm_end:.3f} above cold {cold_end:.3f}")
        self.assertLess(
            speed_warm[stall].max(),
            0.02,
            f"press not quiet under warm start (peak |qd| {speed_warm[stall].max():.3f} m/s)",
        )

    def test_identity_warmstart_matches_cold_equilibrium(self):
        """The matched warm start converges to the cold solve's stall pose, not a new one.

        Warm starting changes the sweep's starting point, never the physics: at
        the stall both runs must agree on the joint position to sub-millimetre.
        """
        _, _, state_cold = _run_press(240, {})
        _, _, state_warm = _run_press(240, {"pgs_warmstart": True})
        q_cold = float(state_cold.joint_q.numpy()[0])
        q_warm = float(state_warm.joint_q.numpy()[0])
        self.assertAlmostEqual(q_warm, q_cold, delta=1.0e-3)

    def test_identity_warmstart_requires_contact_matching(self):
        """Stepping warm start with unmatched contacts raises.

        The identity gather needs ``Contacts.rigid_contact_match_index``; a
        pipeline built without ``contact_matching`` must fail loudly rather than
        silently reusing impulses by raw slot index.
        """
        with self.assertRaises(NotImplementedError):
            _run_press(3, {"pgs_warmstart": True}, contact_matching=None)

    def test_single_flag_enables_dense_and_mf_carry(self):
        """pgs_warmstart=True is the single all-contact warm-start mode."""
        model = _build_press()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_warmstart=True, pgs_iterations=4)
        self.assertTrue(solver.pgs_warmstart)
        self.assertTrue(solver._mf_warmstart_enabled)

    def test_propagation_routes_hold_static_press(self):
        """Serial, colored, and cached propagation install carried impulses once."""
        cases = (
            ("propagation", False),
            ("propagation-colored", False),
            ("propagation-colored", True),
            ("propagation-fused", False),
        )
        for response, cached in cases:
            with self.subTest(response=response, cached=cached):
                cold, _, state_cold = _run_press(
                    160,
                    {},
                    articulated_contact_response=response,
                    propagation_cached_response=cached,
                )
                warm, speed, state_warm = _run_press(
                    160,
                    {"pgs_warmstart": True},
                    articulated_contact_response=response,
                    propagation_cached_response=cached,
                )
                self.assertGreater(float(warm[-1]), 0.0)
                self.assertLess(float(np.ptp(warm[-10:])), 1.0e-5)
                self.assertLess(float(speed[-10:].max()), 1.0e-4)
                self.assertAlmostEqual(float(warm[-10:].mean()), float(cold[-10:].mean()), delta=2.0e-3)
                self.assertAlmostEqual(
                    float(state_warm.joint_q.numpy()[0]),
                    float(state_cold.joint_q.numpy()[0]),
                    delta=1.0e-3,
                )


def _insertion_after_solved_single_contact(articulated: bool):
    """Solve sphere B alone on the ground, then lower sphere A so its contact sorts before B's.

    ``articulated`` mounts each sphere on a vertical prismatic joint, so the contacts are
    dense rows; otherwise the spheres are free bodies on the matrix-free rows.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    shapes = []
    for x, z in ((-0.5, 1.0), (0.5, 0.099)):
        xform = wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        shapes.append(builder.add_shape_sphere(body, radius=0.1))
    builder.add_ground_plane()
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_iterations=8, pgs_warmstart=True)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
    if articulated:
        prev_slot, prev_impulses = solver._ws_prev_dense_slot_sorted, solver._ws_prev_dense_impulses
    else:
        prev_slot, prev_impulses = solver._ws_prev_slot_sorted, solver._ws_prev_mf_impulses
    carried = float(prev_impulses.numpy()[0, int(prev_slot.numpy()[0])])
    if articulated:
        # Sphere A's joint coordinate is its height offset from the 1 m start.
        q = state_1.joint_q.numpy()
        q[0] = 0.099 - 1.0
        state_1.joint_q.assign(q)
        newton.eval_fk(model, state_1.joint_q, state_1.joint_qd, state_1)
    else:
        # Keep A's body pose and free-joint coordinates consistent.
        q = state_1.body_q.numpy()
        q[0][2] = 0.099
        state_1.body_q.assign(q)
        q = state_1.joint_q.numpy()
        q[2] = 0.099
        state_1.joint_q.assign(q)
    return model, pipeline, contacts, solver, state_1, state_0, tuple(shapes), carried


def _seeded_sphere_impulses(solver, contacts, shapes, model, articulated, state_in=None, state_out=None):
    """Return the seeded normal impulses of A's and B's contacts.

    With states, first step without a sweep, so the solved impulses are the seeds;
    otherwise the caller has already stepped (for example by a graph replay).
    """
    if state_in is not None:
        solver.pgs_iterations = 0
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    slots = solver.contact_slot.numpy()[:count]
    impulses = (solver.impulses if articulated else solver.mf_impulses).numpy()[0]
    seeds = []
    for shape in shapes:
        index = np.flatnonzero((shape0 == shape) | (shape1 == shape))
        seeds.append(float(impulses[int(slots[int(index[0])])]))
    return seeds


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSWarmstartProvenance(unittest.TestCase):
    """Warm start reads match indices only when they refer to the contact set it solved."""

    def test_one_collision_pass_carries_the_persistent_contact(self):
        """The reference sequence: B keeps its impulse at its new slot and new A starts cold."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                self.assertGreater(carried, 0.0)
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertAlmostEqual(seed_b, carried, delta=1.0e-6)

    def test_interleaved_contact_buffers_start_cold(self):
        """A pass into the solved buffer after a pass into another buffer starts cold.

        The pipeline matches against its last pass, whichever buffer it wrote, so the
        second pass's match indices refer to the other buffer's contacts. Read as indices
        into the solved set, they would seed the new contact A with B's impulse.
        """
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                self.assertGreater(carried, 0.0)
                pipeline.collide(state_in, pipeline.contacts())
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_interleaved_contact_buffers_start_cold_under_graph_replay(self):
        """Alternating captured collision passes keep the interleaved-buffer cold start."""
        device = wp.get_device()
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                other = pipeline.contacts()
                solver.pgs_iterations = 0
                graphs = {}
                with wp.ScopedCapture(device) as capture:
                    pipeline.collide(state_in, contacts)
                graphs["collide"] = capture.graph
                with wp.ScopedCapture(device) as capture:
                    pipeline.collide(state_in, other)
                graphs["collide_other"] = capture.graph
                with wp.ScopedCapture(device) as capture:
                    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
                graphs["step"] = capture.graph

                # One pass straight into the solved buffer carries B's impulse.
                wp.capture_launch(graphs["collide"])
                wp.capture_launch(graphs["step"])
                seed_a, seed_b = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertEqual(seed_a, 0.0)
                self.assertAlmostEqual(seed_b, carried, delta=1.0e-6)

                wp.capture_launch(graphs["collide_other"])
                wp.capture_launch(graphs["collide"])
                wp.capture_launch(graphs["step"])
                seed_a, seed_b = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_skipped_collision_pass_starts_cold(self):
        """Two collision passes between solves match against an unsolved contact set."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                pipeline.collide(state_in, contacts)
                pipeline.collide(state_in, contacts)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, contacts, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_replaced_contact_buffer_starts_cold(self):
        """A fresh buffer has no solved history of its own, even at the same generation."""
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, _contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                replacement = pipeline.contacts()
                pipeline.collide(state_in, replacement)
                seed_a, seed_b = _seeded_sphere_impulses(
                    solver, replacement, shapes, model, articulated, state_in, state_out
                )
                self.assertEqual(seed_a, 0.0)
                self.assertEqual(seed_b, 0.0)

    def test_substep_on_the_same_contacts_reuses_their_own_impulses(self):
        """A second step on an unchanged contact set seeds each contact from itself.

        Its match indices refer to the set before the last collision pass, where B had
        another index; reading them would seed B from A's slot.
        """
        for articulated in (False, True):
            with self.subTest(articulated=articulated):
                model, pipeline, contacts, solver, state_in, state_out, shapes, _carried = (
                    _insertion_after_solved_single_contact(articulated)
                )
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
                solved = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated)
                self.assertGreater(solved[1], 0.0)
                seeded = _seeded_sphere_impulses(solver, contacts, shapes, model, articulated, state_in, state_out)
                for got, want in zip(seeded, solved, strict=True):
                    self.assertAlmostEqual(got, want, delta=1.0e-6 * max(1.0, abs(want)))


if __name__ == "__main__":
    unittest.main()
