# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Propagation contact responses of SolverFeatherPGS.

``articulated_contact_response="propagation"`` and ``"propagation-fused"`` solve contacts of
articulated bodies as body-space rows whose response comes from the articulated-body
factorization of each tree. These tests check the row response against the dense
``J H^-1 J^T`` of the immediate response, the native tree kernels against the generic
ones, row capacity status, CUDA graph capture and prescribed global bodies.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DENSE_PATH = 0
PROPAGATION_PATH = 2
RESPONSES = ("propagation", "propagation-fused", "propagation-colored")


def _chain(builder, x, n_links, root_z, joint="revolute"):
    """Add a fixed-base chain whose links can touch the ground."""
    parent = -1
    joints = []
    for i in range(n_links):
        link = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, root_z - 0.25 * i), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.04, half_height=0.1)
        if parent < 0:
            parent_xform = wp.transform(wp.vec3(x, 0.0, root_z + 0.12), wp.quat_rpy(0.7, 0.3, 0.0))
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, -0.12), wp.quat_identity())
        child_xform = wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity())
        if joint == "ball":
            joints.append(builder.add_joint_ball(parent, link, parent_xform=parent_xform, child_xform=child_xform))
        else:
            joints.append(
                builder.add_joint_revolute(
                    parent, link, axis=wp.vec3(0.0, 1.0, 0.0), parent_xform=parent_xform, child_xform=child_xform
                )
            )
        parent = link
    builder.add_articulation(joints)


def _floating_base(builder, x):
    """Add a floating box base with four revolute legs resting near the ground."""
    root = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.22), wp.quat_rpy(0.1, 0.05, 0.0)))
    builder.add_shape_box(root, hx=0.2, hy=0.1, hz=0.05)
    joints = [builder.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = builder.add_link(xform=wp.transform(wp.vec3(x + 0.18 * sx, 0.1 * sy, 0.08), wp.quat_identity()))
            builder.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                builder.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)


def _scene(device, kind, worlds=2):
    """Build ``worlds`` copies of a scene whose articulated bodies start in ground contact."""
    world = newton.ModelBuilder()
    if kind == "revolute":
        _chain(world, 0.0, 3, 0.3)
    elif kind == "ball":
        _chain(world, 0.0, 3, 0.3, joint="ball")
    else:
        _floating_base(world, 0.0)
    box = world.add_body(xform=wp.transform(wp.vec3(0.6, 0.0, 0.09), wp.quat_identity()))
    world.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(world, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _step(model, solver, steps=1, dt=1.0 / 240.0):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    # Sorted contacts, so solvers compared on separate pipelines see the same rows in the same order.
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0, contacts


def test_contact_effective_mass_matches_dense_response(test, device):
    """Give each propagation row the effective mass ``J H^-1 J^T`` of its dense immediate row, mass-split.

    The response of a body is scaled by the row-bearing bodies of its coupling group. Covers the native 0/1-DOF and free-root tree kernels, the generic kernels and
    multi-DOF (ball) joints. The rows of contacts between an articulated body and the
    ground are compared at the same state, before any solve.
    """
    cases = (
        ("revolute", "auto"),
        ("revolute", "generic"),
        ("floating", "auto"),
        ("floating", "generic"),
        ("ball", "generic"),
    )
    for kind, tree_kernel in cases:
        with test.subTest(scene=kind, tree_kernel=tree_kernel):
            model = _scene(device, kind)
            # Point friction: the propagation rows have no friction patches.
            # An articulation-local solve writes its rows' diagonals when it runs, so iterate once.
            reference = SolverFeatherPGS(model, pgs_iterations=1, dense_max_constraints=96, friction_anchor_beta=0.0)
            _, contacts = _step(model, reference)
            count = int(contacts.rigid_contact_count.numpy()[0])
            SolverFeatherPGS._kernel_overrides = {"propagation_tree_kernel": tree_kernel}
            try:
                solver = SolverFeatherPGS(
                    model,
                    pgs_iterations=0,
                    dense_max_constraints=96,
                    friction_anchor_beta=0.0,
                    articulated_contact_response="propagation",
                )
            finally:
                SolverFeatherPGS._kernel_overrides = {}
            test.assertEqual(solver._propagation_native_tree[solver._propagation_tree_sizes[0]], tree_kernel == "auto")
            _step(model, solver)

            ref_path = reference.contact_path.numpy()[:count]
            ref_slot = reference.contact_slot.numpy()[:count]
            path = solver.contact_path.numpy()[:count]
            slot = solver.contact_slot.numpy()[:count]
            world = solver.contact_world.numpy()[:count]
            dense_diag = reference.diag.numpy()
            eff_mass_inv = solver.propagation_eff_mass_inv.numpy()
            # Each compared row has the ground on one side.
            row_body = np.maximum(solver.propagation_body_a.numpy(), solver.propagation_body_b.numpy())
            split = solver.propagation_coupling_group_body_count.numpy()[solver.propagation_body_coupling_group.numpy()]
            checked = 0
            # Contacts between links of one articulation stay dense in both responses.
            for c in np.flatnonzero((ref_path == DENSE_PATH) & (ref_slot >= 0) & (path == PROPAGATION_PATH)):
                for row in range(3):
                    expected = float(dense_diag[world[c], ref_slot[c] + row]) * split[row_body[world[c], slot[c] + row]]
                    got = 1.0 / float(eff_mass_inv[world[c], slot[c] + row])
                    test.assertAlmostEqual(got, expected, delta=2.0e-4 * expected)
                    checked += 1
            test.assertGreater(checked, 0, "no articulated contact rows were compared")


def test_native_tree_kernels_match_generic(test, device):
    """Step identically with the native one-warp tree kernels and the generic per-articulation kernels."""
    for kind in ("revolute", "floating"):
        for response in RESPONSES:
            with test.subTest(scene=kind, response=response):
                results = []
                for tree_kernel in ("auto", "generic"):
                    model = _scene(device, kind)
                    SolverFeatherPGS._kernel_overrides = {"propagation_tree_kernel": tree_kernel}
                    try:
                        solver = SolverFeatherPGS(
                            model, pgs_iterations=8, dense_max_constraints=64, articulated_contact_response=response
                        )
                    finally:
                        SolverFeatherPGS._kernel_overrides = {}
                    state, _ = _step(model, solver)
                    results.append(state.joint_qd.numpy())
                np.testing.assert_allclose(results[0], results[1], rtol=1.0e-4, atol=1.0e-5)


def test_propagation_row_overflow_is_flagged_and_reset(test, device):
    """Flag a world whose propagation rows exceed their capacity until that world is reset."""
    builder = newton.ModelBuilder()
    for root_z in (0.3, 3.0):
        # World 1's chain hangs far above the ground and makes no contact.
        world = newton.ModelBuilder()
        _chain(world, 0.0, 2, root_z)
        builder.add_world(world)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    for response, capacity in (
        # Dense rows hold the two rows of each joint limit; one contact's three rows fit.
        ("propagation", {"mf_max_constraints": 1, "dense_max_constraints": 4}),
        ("propagation-fused", {"dense_max_constraints": 4}),
    ):
        with test.subTest(response=response):
            solver = SolverFeatherPGS(
                model, articulated_contact_response=response, warn_constraint_overflow=False, **capacity
            )
            state, contacts = _step(model, solver)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, False, False])
            test.assertEqual(int(solver.propagation_constraint_count.numpy()[0]), 3)
            with test.assertRaisesRegex(RuntimeError, r"worlds \[0\]"):
                solver.check_constraint_capacity()
            solver.reset(state, wp.array([True, False, False], dtype=wp.bool, device=device))
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])


def test_captured_steps_match_eager(test, device):
    """Replay a captured collide + step, including the first step, exactly like eager stepping."""
    for response in RESPONSES:
        with test.subTest(response=response):
            model = _scene(device, "floating")
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            control = model.control()

            def run(capture, model=model, pipeline=pipeline, contacts=contacts, control=control, response=response):
                solver = SolverFeatherPGS(model, dense_max_constraints=64, articulated_contact_response=response)
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

                def substeps():
                    for _ in range(2):
                        pipeline.collide(state_0, contacts)
                        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                        state_0.assign(state_1)

                if capture:
                    with wp.ScopedCapture(device=device) as graph:
                        substeps()
                    for _ in range(30):
                        wp.capture_launch(graph.graph)
                else:
                    for _ in range(30):
                        substeps()
                return state_0.joint_q.numpy(), state_0.joint_qd.numpy()

            eager_q, eager_qd = run(False)
            captured_q, captured_qd = run(True)
            np.testing.assert_allclose(captured_q, eager_q, atol=1.0e-6)
            np.testing.assert_allclose(captured_qd, eager_qd, atol=1.0e-5)


def _chains_on_global_floor(device, floor):
    """Two worlds with one chain each over a floor box whose top is at z = 0; ``floor`` is static or kinematic."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for _ in range(2):
        # Worlds do not collide with each other, so both chains share one place on the floor.
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        _chain(world, 0.0, 2, 0.24)
        builder.add_world(world)
    floor_body = -1 if floor == "static" else builder.add_body(is_kinematic=True, mass=1.0)
    builder.add_shape_box(
        floor_body,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()) if floor == "static" else None,
        hx=5.0,
        hy=5.0,
        hz=0.5,
    )
    model = builder.finalize(device=device)
    if floor != "static":
        joint_q = model.joint_q.numpy()
        joint_q[-5] = -0.5
        model.joint_q.assign(joint_q)
    return model


def test_global_kinematic_floor_moves_articulated_bodies(test, device):
    """Solve articulated contacts against a global kinematic floor in every world through the row target."""
    for response in RESPONSES:
        with test.subTest(response=response):
            results = {}
            for floor, floor_velocity in (("static", 0.0), ("kinematic", 0.0), ("kinematic", 0.5)):
                model = _chains_on_global_floor(device, floor)
                solver = SolverFeatherPGS(
                    model,
                    pgs_iterations=50,
                    dense_max_constraints=64,
                    friction_anchor_beta=0.0,
                    articulated_contact_response=response,
                )
                state_0, state_1 = model.state(), model.state()
                joint_qd = state_0.joint_qd.numpy()
                joint_qd[:4] = -1.0
                if floor == "kinematic":
                    joint_qd[-4] = floor_velocity
                state_0.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, model.control(), contacts, 0.01)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertGreater(count, 0)
                np.testing.assert_array_equal(solver.contact_path.numpy()[:count], PROPAGATION_PATH)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])
                results[(floor, floor_velocity)] = state_1.body_qd.numpy()[:4]
            # Both worlds respond to the global kinematic floor like to world geometry. The kinematic
            # floor's contacts differ from the static floor's by float32 rounding of its body frame,
            # and the colored sweep also orders the two floors' contacts differently.
            atol = 5.0e-5 if response == "propagation-colored" else 1.0e-5
            np.testing.assert_allclose(results[("kinematic", 0.0)], results[("static", 0.0)], rtol=1.0e-4, atol=atol)
            np.testing.assert_allclose(results[("static", 0.0)][:2], results[("static", 0.0)][2:], atol=1.0e-5)
            # The floor's prescribed velocity enters every world's contact target.
            moving = results[("kinematic", 0.5)]
            test.assertGreater(float(np.sum(moving[:, 2] - results[("static", 0.0)][:, 2])), 0.05)
            np.testing.assert_allclose(moving[:2], moving[2:], atol=1.0e-5)


def test_phased_sweeps_touch_only_their_row_family(test, device):
    """Restrict each phased matrix-free sweep of the propagation schedule to its row family.

    Phase 3 solves the dense joint-limit rows, phase 4 the dense and free-body contact rows
    and phase 5 the velocity-limit rows. The scene has an active joint limit, a penetrating
    contact between two links of one articulation (a dense row under the default routing)
    and a falling free body above its velocity bound.
    """
    builder = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.5
    base = builder.add_link()
    builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    joints = [
        builder.add_joint_revolute(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            limit_lower=-0.01,
            limit_upper=0.01,
        )
    ]
    for side in (1.0, -1.0):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            builder.add_joint_revolute(
                base,
                link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
    builder.add_articulation(joints)
    builder.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 2.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 1.0e-2),
        custom_attributes={"rigid_body_max_linear_velocity": 0.001, "rigid_body_max_angular_velocity": 0.001},
    )
    model = builder.finalize(device=device)
    joint_qd = model.joint_qd.numpy()
    # Fast enough to close the 0.01 rad gap to the upper limit within one step.
    joint_qd[0] = 5.0
    model.joint_qd.assign(joint_qd)
    solver = SolverFeatherPGS(
        model, pgs_iterations=0, enable_joint_limits=True, articulated_contact_response="propagation"
    )
    _step(model, solver)

    dense_count = int(solver.constraint_count.numpy()[0])
    row_type = solver.row_type.numpy()[0, :dense_count]
    limit_rows = row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    contact_rows = (row_type == PGS_CONSTRAINT_TYPE_CONTACT) | (row_type == PGS_CONSTRAINT_TYPE_FRICTION)
    test.assertTrue(limit_rows.any() and contact_rows.any())
    mf_count = int(solver.mf_constraint_count.numpy()[0])
    mf_contact_end = int(solver.mf_contact_rows_end.numpy()[0])
    test.assertGreater(mf_count, mf_contact_end)

    touched = {}
    for row_phase in (3, 4, 5):
        solver.impulses.zero_()
        solver.mf_impulses.zero_()
        wp.copy(solver.v_out, solver.v_hat)
        solver._launch_mf_gs_phase(row_phase)
        dense = np.abs(solver.impulses.numpy()[0, :dense_count]) > 0.0
        velocity_limits = np.abs(solver.mf_impulses.numpy()[0, mf_contact_end:mf_count]) > 0.0
        touched[row_phase] = (
            bool(dense[limit_rows].any()),
            bool(dense[contact_rows].any()),
            bool(velocity_limits.any()),
        )
    test.assertEqual(touched[3], (True, False, False))
    test.assertEqual(touched[4], (False, True, False))
    test.assertEqual(touched[5], (False, False, True))


def _build_box_chain(device, links, boxes, articulations=1):
    """Build revolute chains in zero gravity with ``boxes`` free boxes pressed into their links."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    link_hx, link_hy, link_hz, cube_h = 0.15, 0.06, 0.045, 0.04
    slots = np.ceil(boxes / links)
    if slots > 2:
        link_hy = max(link_hy, 0.5 * (slots - 1) * 0.09 + cube_h + 0.01)
    for art in range(articulations):
        origin = wp.vec3(0.0, 0.55 * art, 0.5)
        parent = -1
        joints = []
        for _ in range(links):
            link = builder.add_link()
            builder.add_shape_box(link, hx=link_hx, hy=link_hy, hz=link_hz)
            parent_xform = wp.transform(origin if parent == -1 else wp.vec3(link_hx, 0.0, 0.0), wp.quat_identity())
            joints.append(
                builder.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=parent_xform,
                    child_xform=wp.transform(wp.vec3(-link_hx, 0.0, 0.0), wp.quat_identity()),
                )
            )
            parent = link
        builder.add_articulation(joints)
        # Spread the boxes over the links, side by side when a link carries several.
        per_link = [[] for _ in range(links)]
        if boxes <= links:
            chosen = sorted({int(round(v)) for v in np.linspace(0, links - 1, boxes)}) if boxes > 1 else [links - 1]
            for link_index in chosen:
                per_link[link_index].append(0)
        else:
            for box in range(boxes):
                per_link[box % links].append(box // links)
        for link_index, link_slots in enumerate(per_link):
            for slot in range(len(link_slots)):
                y = 0.0 if len(link_slots) == 1 else (slot - 0.5 * (len(link_slots) - 1)) * 0.09
                position = wp.vec3(
                    (2.0 * link_index + 1.0) * link_hx, float(origin[1]) + y, 0.5 + link_hz + cube_h - 0.012
                )
                cube = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
                builder.add_shape_box(cube, hx=cube_h, hy=cube_h, hz=cube_h)
    return builder.finalize(device=device)


def test_propagation_matches_immediate_when_converged(test, device):
    """Agree with the immediate response on articulated-free multi-box contacts at 32 iterations.

    The responses sweep rows in different orders, so after a few iterations they differ
    legitimately; converged, the step must agree.
    """
    cases = ((4, 1, 1), (4, 2, 1), (2, 2, 1), (4, 4, 1), (4, 1, 2))
    for links, boxes, articulations in cases:
        model = _build_box_chain(device, links, boxes, articulations)
        initial = model.state()
        newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(initial, contacts)
        capacity = 16 * articulations * boxes
        results = {}
        for response in ("immediate", *RESPONSES):
            solver = SolverFeatherPGS(
                model,
                articulated_contact_response=response,
                pgs_iterations=32,
                friction_anchor_beta=0.0,
                dense_max_constraints=capacity,
                mf_max_constraints=16,
            )
            state_out = model.state()
            solver.step(initial, state_out, model.control(), contacts, 1.0 / 240.0)
            solver.check_constraint_capacity()
            if response != "immediate":
                test.assertGreater(int(solver.propagation_constraint_count.numpy().sum()), 0)
            results[response] = {
                name: getattr(state_out, name).numpy().astype(np.float64)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd")
            }
        reference = results["immediate"]
        for response in RESPONSES:
            with test.subTest(links=links, boxes=boxes, articulations=articulations, response=response):
                result = results[response]
                rel_l2 = np.linalg.norm(result["joint_qd"] - reference["joint_qd"]) / max(
                    np.linalg.norm(reference["joint_qd"]), 1.0e-30
                )
                state_linf = max(np.max(np.abs(result[name] - reference[name])) for name in reference)
                test.assertLess(rel_l2, 1.0e-4)
                test.assertLess(state_linf, 1.0e-4)


class TestFeatherPGSPropagation(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_contact_effective_mass_matches_dense_response",
    "test_native_tree_kernels_match_generic",
    "test_propagation_row_overflow_is_flagged_and_reset",
    "test_captured_steps_match_eager",
    "test_global_kinematic_floor_moves_articulated_bodies",
    "test_phased_sweeps_touch_only_their_row_family",
    "test_propagation_matches_immediate_when_converged",
):
    add_function_test(TestFeatherPGSPropagation, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
