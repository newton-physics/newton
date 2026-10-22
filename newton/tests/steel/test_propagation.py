# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Propagation contact responses, colored sweeps, tree-parallel kernels and row schedules of SolverSteel."""

import inspect
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PROPAGATION_COLOR_TAIL,
    PROPAGATION_UNIT_META,
    PROPAGATION_UNIT_RING,
)
from newton._src.solvers.steel.solver_steel import _colored_staging_supported, _SteelTreePlan
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import (
    CONTACT_MAX,
    _build_branched_model,
    _build_contact_model,
    _build_model,
    _build_scissor_model,
    _dynamics_fields,
    _heap,
    _poison,
    _run_stage1,
    _solver,
    set_torsion_radius,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DENSE_PATH = 0


PROPAGATION_PATH = 2


RESPONSES = ("propagation", "propagation_colored")


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
    """Match each propagation row's exact response to its dense immediate row, and step it with the split response.

    Covers the native 0/1-DOF and free-root tree kernels, the generic kernels and
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
            reference = SolverSteel(model, iterations=1, articulation_constraint_max=96, friction_anchor_beta=0.0)
            _, contacts = _step(model, reference)
            count = int(contacts.rigid_contact_count.numpy()[0])
            with kernel_overrides({"propagation_tree_kernel": tree_kernel}):
                solver = SolverSteel(
                    model,
                    iterations=0,
                    articulation_constraint_max=96,
                    friction_anchor_beta=0.0,
                    articulated_contact_response="propagation",
                )
            test.assertEqual(solver._propagation_native_tree[solver._propagation_tree_sizes[0]], tree_kernel == "auto")
            _step(model, solver)

            ref_path = reference._contact_path.numpy()[:count]
            ref_slot = reference._contact_slot.numpy()[:count]
            path = solver._contact_path.numpy()[:count]
            slot = solver._contact_slot.numpy()[:count]
            world = solver._contact_world.numpy()[:count]
            dense_diag = reference._diag.numpy()
            eff_mass_inv = solver._propagation_eff_mass_inv.numpy()
            body_a = solver._propagation_body_a.numpy()
            J_a = solver._propagation_J_a.numpy()
            J_b = solver._propagation_J_b.numpy()
            # Each compared row has the ground on one side.
            row_body = np.maximum(body_a, solver._propagation_body_b.numpy())
            response = solver._propagation_body_response.numpy()
            response_split = solver._propagation_body_response_split.numpy()
            checked = 0
            # Contacts between links of one articulation stay dense in both responses.
            for c in np.flatnonzero((ref_path == DENSE_PATH) & (ref_slot >= 0) & (path == PROPAGATION_PATH)):
                for row in range(3):
                    w, s = world[c], slot[c] + row
                    body = row_body[w, s]
                    J = J_a[w, s] if body_a[w, s] >= 0 else J_b[w, s]
                    dense = float(dense_diag[w, ref_slot[c] + row])
                    test.assertAlmostEqual(float(J @ response[body] @ J) + solver.cfm, dense, delta=2.0e-4 * dense)
                    expected = float(J @ response_split[body] @ J) + solver.cfm
                    got = 1.0 / float(eff_mass_inv[w, s])
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
                    with kernel_overrides({"propagation_tree_kernel": tree_kernel}):
                        solver = SolverSteel(
                            model, iterations=8, articulation_constraint_max=64, articulated_contact_response=response
                        )
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
        ("propagation", {"free_body_constraint_max": 1, "articulation_constraint_max": 4}),
        ("propagation_colored", {"free_body_constraint_max": 1, "articulation_constraint_max": 4}),
    ):
        with test.subTest(response=response):
            solver = SolverSteel(
                model, articulated_contact_response=response, warn_constraint_overflow=False, **capacity
            )
            state, contacts = _step(model, solver)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)
            np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, False, False])
            test.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), 3)
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
                solver = SolverSteel(model, articulation_constraint_max=64, articulated_contact_response=response)
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
                solver = SolverSteel(
                    model,
                    iterations=50,
                    articulation_constraint_max=64,
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
                np.testing.assert_array_equal(solver._contact_path.numpy()[:count], PROPAGATION_PATH)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False, False])
                results[(floor, floor_velocity)] = state_1.body_qd.numpy()[:4]
            # Both worlds respond to the global kinematic floor like to world geometry. The kinematic
            # floor's contacts differ from the static floor's by float32 rounding of its body frame,
            # the colored sweep orders the two floors' contacts differently, and the 50 unconverged
            # iterations amplify both.
            np.testing.assert_allclose(results[("kinematic", 0.0)], results[("static", 0.0)], rtol=1.0e-4, atol=5.0e-5)
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
    SolverSteel.register_custom_attributes(builder)
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
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 0.001,
            "steel:rigid_body_max_angular_velocity": 0.001,
        },
    )
    model = builder.finalize(device=device)
    joint_qd = model.joint_qd.numpy()
    # Fast enough to close the 0.01 rad gap to the upper limit within one step.
    joint_qd[0] = 5.0
    model.joint_qd.assign(joint_qd)
    solver = SolverSteel(model, iterations=0, enable_joint_limits=True, articulated_contact_response="propagation")
    _step(model, solver)

    dense_count = int(solver._constraint_count.numpy()[0])
    row_type = solver._row_type.numpy()[0, :dense_count]
    limit_rows = row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    contact_rows = (row_type == PGS_CONSTRAINT_TYPE_CONTACT) | (row_type == PGS_CONSTRAINT_TYPE_FRICTION)
    test.assertTrue(limit_rows.any() and contact_rows.any())
    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    test.assertGreater(mf_count, mf_contact_end)

    touched = {}
    for row_phase in (3, 4, 5):
        solver._impulses.zero_()
        solver._mf_impulses.zero_()
        wp.copy(solver._v_out, solver._v_hat)
        solver._launch_mf_gs_phase(row_phase)
        dense = np.abs(solver._impulses.numpy()[0, :dense_count]) > 0.0
        velocity_limits = np.abs(solver._mf_impulses.numpy()[0, mf_contact_end:mf_count]) > 0.0
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
            solver = SolverSteel(
                model,
                articulated_contact_response=response,
                iterations=32,
                friction_anchor_beta=0.0,
                articulation_constraint_max=capacity,
                free_body_constraint_max=16,
            )
            state_out = model.state()
            solver.step(initial, state_out, model.control(), contacts, 1.0 / 240.0)
            solver.check_constraint_capacity()
            if response != "immediate":
                test.assertGreater(int(solver._propagation_constraint_count.numpy().sum()), 0)
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


class TestSteelPropagation(unittest.TestCase):
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
    add_function_test(TestSteelPropagation, _name, globals()[_name], devices=devices)


N_WORLDS = 4


DT = 1.0 / 240.0


def _add_floating_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A floating box base trailing ``n_links`` revolute links, with collision spheres on ``spheres``."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1, cfg=no_collision)
    if "base" in spheres:
        builder.add_shape_sphere(base, radius=0.1)
    joints = [builder.add_joint_free(base)]
    prev = base
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
        offset = 0.22 if i == 0 else 0.12
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(offset, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)


def _add_fixed_chain(
    builder: newton.ModelBuilder, *, n_links: int, base_z: float, base_y: float, spheres: tuple[str, ...]
) -> None:
    """A world-anchored all-revolute chain drooping onto the ground, with collision spheres on ``spheres``."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.12, hy=0.03, hz=0.03, cfg=no_collision)
        if "base" in spheres and i == 0:
            builder.add_shape_sphere(link, radius=0.1)
        if "deep" in spheres and i == n_links - 1:
            builder.add_shape_sphere(link, radius=0.1)
        if "mid" in spheres and i == n_links // 2:
            builder.add_shape_sphere(link, radius=0.1)
        parent_xform = (
            wp.transform(wp.vec3(0.0, base_y, base_z), wp.quat_identity())
            if prev == -1
            else wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity())
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)


def _build_chain_model(
    device: str, *, chains: tuple[int, ...], spheres: tuple[str, ...], fixed_base: bool = False
) -> newton.Model:
    """N_WORLDS identical worlds, each holding one chain per entry of ``chains``."""
    scene = newton.ModelBuilder()
    add_chain = _add_fixed_chain if fixed_base else _add_floating_chain
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        for k, n_links in enumerate(chains):
            add_chain(world, n_links=n_links, base_z=0.105, base_y=0.8 * k, spheres=spheres)
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _build_clutter_model(device: str, *, n_boxes: int) -> newton.Model:
    """A fixed-base arm plus free spheres resting on the ground.

    The spheres are frictionless so each has one disjoint row, which keeps the tree walk reproducible.
    """
    scene = newton.ModelBuilder()
    for _ in range(N_WORLDS):
        world = newton.ModelBuilder()
        world.default_shape_cfg.density = 1000.0
        world.default_shape_cfg.mu = 0.7
        # Every arm sphere starts pressed into the ground, so the active-body set is stable.
        _add_fixed_chain(world, n_links=6, base_z=0.095, base_y=0.0, spheres=("base", "mid", "deep"))
        clutter_cfg = world.default_shape_cfg.copy()
        clutter_cfg.mu = 0.0
        for b in range(n_boxes):
            ball = world.add_link(
                xform=wp.transform(wp.vec3(-0.4 - 0.25 * (b % 3), 0.6 + 0.25 * (b // 3), 0.0995), wp.quat_identity())
            )
            world.add_shape_sphere(ball, radius=0.1, cfg=clutter_cfg)
            world.add_articulation([world.add_joint_free(parent=-1, child=ball)])
        scene.add_world(world)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _tree_inputs(model, solver, art):
    return {
        "S": solver._propagation_joint_S_flat.numpy().astype(np.float64),
        "U": solver._propagation_tree_U.numpy().astype(np.float64),
        "Dinv": solver._propagation_tree_D_inv.numpy().astype(np.float64),
        "com": solver._propagation_body_com_rel.numpy().astype(np.float64),
        "jp": model.joint_parent.numpy(),
        "jc": model.joint_child.numpy(),
        "qd_start": model.joint_qd_start.numpy(),
        "j0": int(model.articulation_start.numpy()[art]),
        "j1": int(model.articulation_start.numpy()[art + 1]),
    }


def _xlt_wrench(w, e):
    return np.concatenate([w[:3], w[3:] + np.cross(e, w[:3])])


def _xlt_twist(v, e):
    return np.concatenate([v[:3] + np.cross(v[3:], e), v[3:]])


def _ref_propagate(ti, imp_by_body, v_out):
    """Mirror of kernels.propagate_tree_impulses_for_size (one articulation)."""
    S, U, Dinv, com = ti["S"], ti["U"], ti["Dinv"], ti["com"]
    jp, jc, qd_start = ti["jp"], ti["jc"], ti["qd_start"]
    j0, j1 = ti["j0"], ti["j1"]
    pA = {}
    u = np.zeros(S.shape[0])
    body_delta = {}
    for j in range(j0, j1):
        pA[int(jc[j])] = -np.asarray(imp_by_body.get(int(jc[j]), np.zeros(6)), dtype=np.float64)
        body_delta[int(jc[j])] = np.zeros(6)
    for j in range(j1 - 1, j0 - 1, -1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        for g in range(d0, d1):
            u[g] = -np.dot(S[g], pA[child])
        if parent >= 0:
            p = pA[child].copy()
            for a in range(d0, d1):
                coeff = sum(Dinv[j, a - d0, b - d0] * u[b] for b in range(d0, d1))
                p += U[a] * coeff
            pA[parent] += _xlt_wrench(p, com[child] - com[parent])
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        pd = np.zeros(6)
        if parent >= 0:
            pd = _xlt_twist(body_delta[parent], com[child] - com[parent])
        qdd = np.zeros(d1 - d0)
        for a in range(d0, d1):
            acc = 0.0
            for b in range(d0, d1):
                pt = np.dot(U[b], pd) if parent >= 0 else 0.0
                acc += Dinv[j, a - d0, b - d0] * (u[b] - pt)
            qdd[a - d0] = acc
            v_out[a] += acc
        body_delta[child] = pd + sum(S[a] * qdd[a - d0] for a in range(d0, d1))
    # Recompute body twists from the updated v_out.
    body_qd = {}
    twist = {}
    for j in range(j0, j1):
        child, parent = int(jc[j]), int(jp[j])
        d0, d1 = int(qd_start[j]), int(qd_start[j + 1])
        val = np.zeros(6)
        if parent >= 0:
            val = _xlt_twist(twist[parent], com[child] - com[parent])
        for a in range(d0, d1):
            val = val + S[a] * v_out[a]
        twist[child] = val
        body_qd[child] = val
    return v_out, body_qd


def _make_solver(
    model: newton.Model, response: str, *, cached: bool, cache_max_bodies: int = 8, **solver_kwargs
) -> SolverSteel:
    return SolverSteel(
        model,
        articulated_contact_response=response,
        iterations=8,
        friction_anchor_beta=0.0,
        propagation_cached_response=cached,
        propagation_cached_response_body_max=cache_max_bodies,
        **solver_kwargs,
    )


def _seed_joint_qd(model: newton.Model) -> np.ndarray:
    n = int(model.joint_dof_count)
    return (0.4 * np.sin(0.7 * np.arange(n, dtype=np.float64) + 0.3)).astype(np.float32)


def _active_body_qd(solver, body_map_solver) -> dict[int, np.ndarray]:
    """COM twists of the contact-active bodies in ``body_map_solver``'s body list."""
    qd = solver._propagation_body_qd.numpy()
    counts = body_map_solver._propagation_body_count.numpy()
    body_list = body_map_solver._propagation_body_list.numpy()
    out: dict[int, np.ndarray] = {}
    for world in range(body_list.shape[0]):
        n = min(int(counts[world]), body_list.shape[1])
        for slot in range(n):
            body = int(body_list[world, slot])
            if body >= 0:
                out[body] = qd[body].copy()
    return out


class _LockstepStats:
    def __init__(self):
        self.max_diffs = {"joint_q": 0.0, "joint_qd": 0.0, "v_out": 0.0, "active_body_qd": 0.0}
        self.max_active_bodies = 0
        self.total_rows = 0


def _run_lockstep(model, response, n_steps, *, cache_max_bodies: int = 8, **solver_kwargs):
    """Step the cached solver from each state of a tree-walk trajectory and record the output diffs.

    A second tree-walk solver must match the first bitwise, so the diffs measure the cache alone.
    """
    ref = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    ref2 = _make_solver(model, response, cached=False, cache_max_bodies=cache_max_bodies, **solver_kwargs)
    test = _make_solver(model, response, cached=True, cache_max_bodies=cache_max_bodies, **solver_kwargs)

    state_in = model.state()
    ref_out = model.state()
    ref2_out = model.state()
    test_out = model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision_pipeline.contacts()

    stats = _LockstepStats()
    for _ in range(n_steps):
        state_in.clear_forces()
        # A deterministic collide before each solver gives all three the same contacts.
        collision_pipeline.collide(state_in, contacts)
        test.step(state_in, test_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref2.step(state_in, ref2_out, control, contacts, DT)
        collision_pipeline.collide(state_in, contacts)
        ref.step(state_in, ref_out, control, contacts, DT)
        wp.synchronize()

        for key in ("joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(ref_out, key).numpy(),
                getattr(ref2_out, key).numpy(),
                err_msg=f"tree-walk solve not reproducible for {key}; scene invalidates the gate",
            )

        for key in ("joint_q", "joint_qd"):
            diff = float(np.max(np.abs(getattr(test_out, key).numpy() - getattr(ref_out, key).numpy())))
            stats.max_diffs[key] = max(stats.max_diffs[key], diff)
        stats.max_diffs["v_out"] = max(
            stats.max_diffs["v_out"], float(np.max(np.abs(test._v_out.numpy() - ref._v_out.numpy())))
        )
        # The tree walk builds no body list, so both read the cached solver's.
        ref_body_qd = _active_body_qd(ref, test)
        test_body_qd = _active_body_qd(test, test)
        for body, ref_qd in ref_body_qd.items():
            diff = float(np.max(np.abs(test_body_qd[body] - ref_qd)))
            stats.max_diffs["active_body_qd"] = max(stats.max_diffs["active_body_qd"], diff)

        counts = test._propagation_body_count.numpy()
        stats.max_active_bodies = max(stats.max_active_bodies, int(counts.max()))
        stats.total_rows += int(ref._propagation_constraint_count.numpy().sum())

        state_in, ref_out = ref_out, state_in
    return stats, ref, test


TOL = 1.0e-4


def _assert_stats(test, stats, *, expect_min_bodies):
    test.assertGreater(stats.total_rows, 0, "no propagation contact rows produced")
    test.assertGreaterEqual(stats.max_active_bodies, expect_min_bodies)
    for key, diff in stats.max_diffs.items():
        test.assertLessEqual(diff, TOL, f"{key} per-step max abs diff {diff:.3e} exceeds {TOL:g}")


def _step_cached(model, steps):
    solver = _make_solver(model, "propagation", cached=True)
    state_in, state_out = model.state(), model.state()
    control = model.control()
    state_in.joint_qd.assign(_seed_joint_qd(model))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    collision_pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = collision_pipeline.contacts()
    for _ in range(steps):
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, DT)
        state_in, state_out = state_out, state_in
    wp.synchronize()
    return solver


def test_cached_path_detected(test, device):
    """Build a cached kernel per size group and only with the option on."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    solver = _make_solver(model, "propagation", cached=True)
    test.assertIsNotNone(solver._propagation_cached_gemv_kernel)
    test.assertEqual(
        solver._propagation_cache_max_bodies,
        min(solver._max_propagation_bodies, solver.propagation_cached_response_body_max),
    )
    for size in solver._propagation_tree_sizes:
        test.assertIn(size, solver._propagation_cached_kernels)
    test.assertTrue(np.all(solver._propagation_cache_art_eligible.numpy() == 1))

    off = _make_solver(model, "propagation", cached=False)
    test.assertFalse(off._propagation_cached_kernels)
    test.assertIsNone(off._propagation_cached_gemv_kernel)


def test_lockstep_multi_contact_bodies(test, device):
    """Match the tree walk with three contact spheres per articulation, so cross blocks are live."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    test.assertTrue(np.all(cached._propagation_cache_world_flag.numpy() == 1))
    _assert_stats(test, stats, expect_min_bodies=2)


def test_lockstep_second_size_group(test, device):
    """Match the tree walk with two articulation sizes per world, both cached."""
    model = _build_chain_model(device, chains=(6, 3), spheres=("base", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    sizes = cached._propagation_tree_sizes
    test.assertEqual(len(sizes), 2, f"expected two size groups, got {sizes}")
    for size in sizes:
        test.assertIn(size, cached._propagation_cached_kernels)
    _assert_stats(test, stats, expect_min_bodies=2)


def test_lockstep_single_contact(test, device):
    """Match the tree walk with one contact sphere per world."""
    model = _build_chain_model(device, chains=(6,), spheres=("base",))
    stats, _, _ = _run_lockstep(model, "propagation", 40)
    _assert_stats(test, stats, expect_min_bodies=1)


def test_lockstep_colored(test, device):
    """Match the tree walk under the colored sweep."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation_colored", 40)
    test.assertTrue(cached._propagation_cached_kernels)
    test.assertTrue(np.all(cached._propagation_cache_world_flag.numpy() == 1))
    _assert_stats(test, stats, expect_min_bodies=2)


def test_lockstep_fixed_base_single_dof(test, device):
    """Match the tree walk with fixed-base all-revolute chains, the single-DOF extraction."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=True)
    stats, _, cached = _run_lockstep(model, "propagation", 40)
    for size in cached._propagation_tree_sizes:
        test.assertFalse(cached._propagation_free_root_tree[size])
        test.assertIn(size, cached._propagation_cached_kernels)
    test.assertTrue(np.all(cached._propagation_cache_world_flag.numpy() == 1))
    _assert_stats(test, stats, expect_min_bodies=2)


def test_overflow_falls_back_to_tree_walk(test, device):
    """Take the per-world tree walk when the active bodies exceed the cache capacity."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    stats, _, cached = _run_lockstep(model, "propagation", 40, cache_max_bodies=2)
    test.assertEqual(cached._propagation_cache_max_bodies, 2)
    counts = cached._propagation_body_count.numpy()
    flags = cached._propagation_cache_world_flag.numpy()
    test.assertTrue(np.all(counts > 2), f"scene must overflow the cap, got counts {counts}")
    test.assertTrue(np.all(flags == 0), f"overflowing worlds must clear the cache flag, got {flags}")
    _assert_stats(test, stats, expect_min_bodies=3)


def test_clutter_does_not_evict_cache(test, device):
    """Count only the articulation's bodies against the cache capacity, not free clutter."""
    model = _build_clutter_model(device, n_boxes=2)
    stats, _, cached = _run_lockstep(model, "propagation", 40, cache_max_bodies=4, articulation_constraint_max=128)
    counts = cached._propagation_body_count.numpy()
    test.assertTrue(np.all(counts > 4), f"scene must overflow the total active-body count, got {counts}")
    eligible_counts = cached._propagation_cache_body_count.numpy()
    test.assertTrue(np.all((eligible_counts >= 2) & (eligible_counts <= 4)), f"eligible counts {eligible_counts}")
    test.assertTrue(np.all(cached._propagation_cache_world_flag.numpy() == 1))
    # Eligible bodies occupy the prefix of each world's body list.
    art_eligible = cached._propagation_cache_art_eligible.numpy()
    body_to_art = cached._body_to_articulation.numpy()
    body_list = cached._propagation_body_list.numpy()
    for world in range(body_list.shape[0]):
        for slot in range(min(int(counts[world]), body_list.shape[1])):
            body = int(body_list[world, slot])
            test.assertGreaterEqual(body, 0)
            is_eligible = int(art_eligible[int(body_to_art[body])]) == 1
            test.assertEqual(is_eligible, slot < int(eligible_counts[world]), f"world {world} slot {slot}")
    _assert_stats(test, stats, expect_min_bodies=5)


def test_cached_matrices_match_numpy_reference(test, device, fixed_base=False):
    """Match R and B to a float64 mirror of the tree walk.

    R_b column j is the joint response to a unit wrench j at body b; B_ab column j is the twist of body a.
    """
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"), fixed_base=fixed_base)
    solver = _step_cached(model, 20)
    counts = solver._propagation_body_count.numpy()
    body_list = solver._propagation_body_list.numpy()
    body_to_art = solver._body_to_articulation.numpy()
    flags = solver._propagation_cache_world_flag.numpy()
    art_dof_start = model.joint_qd_start.numpy()[model.articulation_start.numpy()[:-1]]
    R = solver._propagation_cache_R.numpy()
    B = solver._propagation_cache_B.numpy()

    checked_pairs = 0
    for world in range(min(2, body_list.shape[0])):
        test.assertEqual(int(flags[world]), 1)
        n = int(counts[world])
        test.assertGreaterEqual(n, 2, "need at least two active bodies for cross blocks")
        for slot_b in range(n):
            body_b = int(body_list[world, slot_b])
            art = int(body_to_art[body_b])
            ti = _tree_inputs(model, solver, art)
            dof0 = int(art_dof_start[art])
            n_dofs = int(ti["qd_start"][ti["j1"]] - ti["qd_start"][ti["j0"]])
            for basis in range(6):
                imp = np.zeros(6)
                imp[basis] = 1.0
                v_ref, body_qd_ref = _ref_propagate(
                    ti, {body_b: imp}, np.zeros(solver._v_out.shape[0], dtype=np.float64)
                )
                got_R = R[world, slot_b, :n_dofs, basis]
                ref_R = v_ref[dof0 : dof0 + n_dofs]
                scale = max(1.0, float(np.max(np.abs(ref_R))))
                test.assertLess(float(np.max(np.abs(got_R - ref_R))), 1e-4 * scale, f"R {world} {slot_b} {basis}")
                for slot_a in range(n):
                    body_a = int(body_list[world, slot_a])
                    ref_col = body_qd_ref[body_a] if int(body_to_art[body_a]) == art else np.zeros(6)
                    got_col = B[world, slot_a, slot_b].reshape(6, 6)[:, basis]
                    scale = max(1.0, float(np.max(np.abs(ref_col))))
                    test.assertLess(
                        float(np.max(np.abs(got_col - ref_col))), 1e-4 * scale, f"B {world} {slot_a} {slot_b} {basis}"
                    )
                    checked_pairs += 1
    test.assertGreater(checked_pairs, 0)


def test_impulses_consumed(test, device):
    """Clear the deferred body impulses in the GEMV of every iteration."""
    model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
    solver = _step_cached(model, 10)
    test.assertEqual(float(np.max(np.abs(solver._propagation_body_impulses.numpy()))), 0.0)


def test_captured_cached_response_matches_eager(test, device):
    """Replay a captured step of the cached response, with a capacity overflow, with the eager result."""
    for cache_max_bodies in (8, 2):
        finals = []
        for capture in (False, True):
            model = _build_chain_model(device, chains=(6,), spheres=("base", "mid", "deep"))
            solver = _make_solver(model, "propagation", cached=True, cache_max_bodies=cache_max_bodies)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign(_seed_joint_qd(model))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            control = model.control()

            def substep(
                state_0=state_0, state_1=state_1, solver=solver, pipeline=pipeline, contacts=contacts, control=control
            ):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_0, name), getattr(state_1, name))

            substep()
            if capture:
                with wp.ScopedCapture(device) as scope:
                    substep()
                for _ in range(5):
                    wp.capture_launch(scope.graph)
            else:
                for _ in range(5):
                    substep()
            finals.append(state_0.joint_qd.numpy())
        np.testing.assert_allclose(finals[1], finals[0], rtol=0.0, atol=1.0e-5)


class TestSteelPropagationCachedResponse(unittest.TestCase):
    pass


for _fn in (
    test_cached_path_detected,
    test_lockstep_multi_contact_bodies,
    test_lockstep_second_size_group,
    test_lockstep_single_contact,
    test_lockstep_colored,
    test_lockstep_fixed_base_single_dof,
    test_overflow_falls_back_to_tree_walk,
    test_clutter_does_not_evict_cache,
    test_cached_matrices_match_numpy_reference,
    test_impulses_consumed,
    test_captured_cached_response_matches_eager,
):
    add_function_test(TestSteelPropagationCachedResponse, _fn.__name__, _fn, devices=devices)


add_function_test(
    TestSteelPropagationCachedResponse,
    "test_fixed_base_cached_matrices_match_numpy_reference",
    test_cached_matrices_match_numpy_reference,
    devices=devices,
    fixed_base=True,
)


def _combination_scene(device, restitution=0.0, mimic=False):
    """A driven, limited three-link chain over the ground with three boxes falling onto it, in two worlds."""
    template = newton.ModelBuilder()
    template.default_shape_cfg.mu = 0.6
    template.default_shape_cfg.restitution = restitution
    parent = -1
    joints = []
    for _ in range(3):
        link = template.add_link()
        template.add_shape_box(link, hx=0.15, hy=0.06, hz=0.045)
        joints.append(
            template.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, 0.3) if parent < 0 else wp.vec3(0.15, 0.0, 0.0), wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
                target_ke=40.0,
                target_kd=2.0,
                target_pos=0.2,
                limit_lower=-0.6,
                limit_upper=0.6,
            )
        )
        parent = link
    template.add_articulation(joints)
    if mimic:
        template.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    for x in (0.15, 0.45, 0.75):
        box = template.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.45), wp.quat_identity()))
        template.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    template.add_ground_plane()
    builder = newton.ModelBuilder()
    builder.replicate(template, 2)
    return builder.finalize(device=device)


def _combination_solver(model, **kwargs):
    options = {"articulation_constraint_max": 128, "free_body_constraint_max": 128, "friction_anchor_beta": 0.0}
    options.update(kwargs)
    return SolverSteel(model, **options)


def _pipeline(model, warmstart=False):
    return newton.CollisionPipeline(model, deterministic=True, contact_matching="latest" if warmstart else "disabled")


def _settled_state(model, steps):
    """Return a mid-trajectory state of the immediate response (boxes landing on the chain)."""
    solver = _combination_solver(model, iterations=32)
    pipeline = _pipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0


def _one_step(model, state, **kwargs):
    solver = _combination_solver(model, **kwargs)
    pipeline = _pipeline(model, kwargs.get("contact_warmstart", False))
    contacts = pipeline.contacts()
    state_out = model.state()
    pipeline.collide(state, contacts)
    solver.step(state, state_out, model.control(), contacts, 1.0 / 240.0)
    solver.check_constraint_capacity()
    return solver, contacts, state_out.body_qd.numpy()


CASES = {
    "point friction": {},
    "friction patches": {"friction_anchor_beta": 0.2},
    "regularization": {"contact_regularization": 0.05},
    "velocity iterations": {"velocity_iterations": 300},
    "drive rows": {"drive_mode": "rows"},
    "drive rows and velocity limits": {
        "drive_mode": "rows",
        "enable_joint_velocity_limits": True,
        "velocity_limit_activation_fraction": 0.5,
    },
    "joint limits": {"enable_joint_limits": True},
}


def test_options_match_the_immediate_response_at_convergence(test, device):
    """Converge each option to the immediate response's step from the same state, in both responses."""
    for name, options in CASES.items():
        model = _combination_scene(device)
        if "velocity_limit_activation_fraction" in options:
            model.joint_velocity_limit.fill_(0.5)
        state = _settled_state(model, 40)
        _, _, reference = _one_step(model, state, iterations=300, **options)
        for response in RESPONSES:
            with test.subTest(option=name, response=response):
                solver, contacts, body_qd = _one_step(
                    model, state, iterations=300, articulated_contact_response=response, **options
                )
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertTrue((solver._contact_path.numpy()[:count] == PROPAGATION_PATH).any())
                np.testing.assert_allclose(body_qd, reference, rtol=0.0, atol=2.0e-3)


def test_restitution_rebounds_on_propagation_rows(test, device):
    """Rebound a box falling onto the chain with its restitution through the propagation rows."""
    for response in RESPONSES:
        with test.subTest(response=response):
            model = _combination_scene(device, restitution=0.8)
            state = model.state()
            # The first box sits just above link 0 and moves down at 2 m/s.
            box_joint = 3
            q_start = int(model.joint_q_start.numpy()[box_joint])
            qd_start = int(model.joint_qd_start.numpy()[box_joint])
            joint_q = state.joint_q.numpy()
            joint_q[q_start + 2] = 0.3 + 0.045 + 0.05 + 0.002
            state.joint_q.assign(joint_q)
            joint_qd = state.joint_qd.numpy()
            joint_qd[qd_start + 2] = -2.0
            state.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver, contacts, after = _one_step(
                model, state, iterations=64, articulated_contact_response=response, enable_joint_limits=True
            )
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertTrue((solver._contact_path.numpy()[:count] == PROPAGATION_PATH).any())
            test.assertGreater(float(np.abs(solver._propagation_restitution_bias.numpy()).max()), 1.0)
            test.assertGreater(float(after[3, 2]), 0.5)
            _, _, reference = _one_step(model, state, iterations=64, enable_joint_limits=True)
            np.testing.assert_allclose(after[3], reference[3], rtol=0.0, atol=1.0e-2)


def test_regularization_weights_reach_the_propagation_rows(test, device):
    """Write the regularization weight of penetrating propagation contact rows.

    A row regularizes against its unsplit diagonal ``d``: ``w = 1 / (1 + g d / d_s)`` for the split diagonal ``d_s``.
    """
    model = _combination_scene(device)
    state = _settled_state(model, 60)
    for response in RESPONSES:
        with test.subTest(response=response):
            solver, _, _ = _one_step(
                model, state, iterations=8, articulated_contact_response=response, contact_regularization=1.0
            )
            count = int(solver._propagation_constraint_count.numpy()[0])
            rows = solver._propagation_row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
            penetrating = solver._propagation_phi.numpy()[0, :count] <= 0.0
            weights = solver._propagation_row_w.numpy()[0, :count]
            # The unsplit diagonal from the bodies' own responses.
            body_response = solver._propagation_body_response.numpy()
            d = np.full(count, solver.cfm)
            for body, J in (
                (solver._propagation_body_a.numpy()[0, :count], solver._propagation_J_a.numpy()[0, :count]),
                (solver._propagation_body_b.numpy()[0, :count], solver._propagation_J_b.numpy()[0, :count]),
            ):
                has_body = body >= 0
                d[has_body] += np.einsum("ri,rij,rj->r", J[has_body], body_response[body[has_body]], J[has_body])
            d_split = 1.0 / solver._propagation_eff_mass_inv.numpy()[0, :count]
            checked = rows & penetrating
            test.assertTrue((checked & (d_split > 1.01 * d)).any(), "no mass-split row was checked")
            np.testing.assert_allclose(weights[checked], (1.0 / (1.0 + d / d_split))[checked], rtol=1.0e-4)


def test_warm_start_seeds_the_propagation_rows(test, device):
    """Carry the propagation impulses of resting contacts into the next step."""
    model = _combination_scene(device)
    state = _settled_state(model, 150)
    for response in RESPONSES:
        with test.subTest(response=response):
            solver = _combination_solver(
                model, iterations=64, contact_warmstart=True, articulated_contact_response=response
            )
            pipeline = _pipeline(model, warmstart=True)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state, name))
            for _ in range(3):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            solved = solver._propagation_impulses.numpy().copy()
            # Without iterations the step publishes the seeded impulses unchanged.
            solver._iterations = 0
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
            seeded = solver._propagation_impulses.numpy()
            test.assertGreater(float(np.abs(solved).max()), 0.0)
            np.testing.assert_allclose(seeded.sum(axis=1), solved.sum(axis=1), rtol=0.1)


def test_mimic_rows_hold_with_propagation(test, device):
    """Keep a mimic follower on its leader with the propagation responses."""
    for response in RESPONSES:
        with test.subTest(response=response):
            model = _combination_scene(device, mimic=True)
            solver = _combination_solver(model, iterations=32, articulated_contact_response=response)
            pipeline = _pipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            for _ in range(60):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            joint_q = state_0.joint_q.numpy()
            test.assertGreater(solver._mimic_count, 0)
            test.assertLess(abs(float(joint_q[2] - joint_q[1])), 5.0e-3)


class TestSteelPropagationCombinations(unittest.TestCase):
    pass


for _fn in (
    test_options_match_the_immediate_response_at_convergence,
    test_restitution_rebounds_on_propagation_rows,
    test_regularization_weights_reach_the_propagation_rows,
    test_warm_start_seeds_the_propagation_rows,
    test_mimic_rows_hold_with_propagation,
):
    add_function_test(TestSteelPropagationCombinations, _fn.__name__, _fn, devices=devices)


PROPAGATION_RESPONSES = ("propagation", "propagation_colored")


REGULARIZED = {"baumgarte": 0.0, "cfm": 0.0, "contact_regularization": 3.0}


# (articulated_contact_response, propagation_cached_response)
PROPAGATION_ROUTES = (
    ("propagation", False),
    ("propagation", True),
    ("propagation_colored", False),
    ("propagation_colored", True),
)


def _quadruped_model(device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [builder.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = builder.add_link(xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, 0.08), wp.quat_identity()))
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
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _three_foot_slider_model(device):
    """One prismatic-Z DOF carrying three spheres on fixed-linked bodies above a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.0
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    joints = [builder.add_joint_prismatic(-1, slider, axis=wp.vec3(0.0, 0.0, 1.0))]
    for x in (-0.2, 0.0, 0.2):
        foot = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(foot, radius=0.05)
        joints.append(
            builder.add_joint_fixed(
                slider,
                foot,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _linkage_gripper_model(device):
    """Two pads on passive mimic linkages of one soft, effort-limited driver, under a stiff lift, around a box."""
    builder = newton.ModelBuilder()
    lift = builder.add_link(mass=5.0, inertia=wp.mat33(np.eye(3) * 0.05))
    joints = [
        builder.add_joint_prismatic(
            -1,
            lift,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform((0.0, 0.0, 0.175), wp.quat_identity()),
            target_ke=5.0e4,
            target_kd=500.0,
        )
    ]
    palm = builder.add_link(mass=0.7, inertia=wp.mat33(np.eye(3) * 1.0e-3))
    joints.append(builder.add_joint_revolute(lift, palm, axis=(0.0, 0.0, 1.0), target_ke=5.0e4, target_kd=50.0))
    pad_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=5.0, gap=0.005)
    knuckles, fingers = {}, {}
    for side in (-1.0, 1.0):
        driven = side < 0.0
        knuckle = builder.add_link(mass=0.013, inertia=wp.mat33(np.eye(3) * 2.0e-6))
        knuckles[side] = builder.add_joint_revolute(
            palm,
            knuckle,
            axis=(0.0, side, 0.0),
            parent_xform=wp.transform((side * 0.0532, 0.0, -0.03), wp.quat_identity()),
            target_ke=100.0 if driven else 0.0,
            target_kd=10.0 if driven else 0.0,
            effort_limit=10.0 if driven else 1.0e6,
            limit_lower=-0.1,
            limit_upper=0.9,
        )
        pad = builder.add_link(mass=0.04, inertia=wp.mat33(np.eye(3) * 1.0e-5))
        builder.add_shape_box(
            pad, xform=wp.transform((0.0, 0.0, -0.03), wp.quat_identity()), hx=0.006, hy=0.012, hz=0.03, cfg=pad_cfg
        )
        fingers[side] = builder.add_joint_revolute(
            knuckle,
            pad,
            axis=(0.0, side, 0.0),
            parent_xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()),
            limit_lower=-0.9,
            limit_upper=0.1,
        )
        joints += [knuckles[side], fingers[side]]
    builder.add_articulation(joints)
    # The left knuckle drives the right knuckle and both pad joints.
    builder.set_joint_mimic(knuckles[1.0], knuckles[-1.0], coeffs=(0.0, 1.0))
    for finger in fingers.values():
        builder.set_joint_mimic(finger, knuckles[-1.0], coeffs=(0.0, -1.0))
    box = builder.add_body(xform=wp.transform((0.0, 0.0, 0.0305), wp.quat_identity()))
    builder.add_shape_box(
        box, hx=0.03, hy=0.024, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=0.3 / 1.728e-4, mu=0.5, gap=0.005)
    )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    return builder.finalize(device=device), palm, box


def _coupled_solver(model, response, iterations, cached=False, **kwargs):
    options = {
        "articulated_contact_response": response,
        "propagation_cached_response": cached,
        "friction_anchor_beta": 0.0,
        "articulation_constraint_max": 96,
        "free_body_constraint_max": 96,
        "iterations": iterations,
    }
    options.update(kwargs)
    return SolverSteel(model, **options)


def _rollout(model, solver, joint_qd, steps):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        yield state_0.joint_qd.numpy()


def test_shared_dof_feet_reach_rest(test, device):
    """Stop three feet on one prismatic DOF at the plane for every sweep count."""
    model = _three_foot_slider_model(device)
    for response in PROPAGATION_RESPONSES:
        for iterations in (1, 2, 12, 13):
            with test.subTest(response=response, iterations=iterations):
                solver = _coupled_solver(model, response, iterations, baumgarte=0.0, cfm=0.0)
                (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                test.assertLess(abs(float(qd[0])), 1.0e-3)


def test_shared_dof_feet_split_their_whole_response(test, device):
    """Split all of the response of feet whose only mobility is one shared DOF, on the native and generic kernels."""
    model = _three_foot_slider_model(device)
    feet = [1, 2, 3]
    for tree_kernel in ("auto", "generic"):
        with test.subTest(tree_kernel=tree_kernel):
            with kernel_overrides({"propagation_tree_kernel": tree_kernel}):
                solver = _coupled_solver(model, "propagation", 1, baumgarte=0.0, cfm=0.0)
            test.assertEqual(solver._propagation_native_tree[solver._propagation_tree_sizes[0]], tree_kernel == "auto")
            list(_rollout(model, solver, np.array([-1.0], dtype=np.float32), 1))
            exact = solver._propagation_body_response.numpy()[feet]
            split = solver._propagation_body_response_split.numpy()[feet]
            test.assertGreater(float(np.abs(exact).max()), 0.0)
            np.testing.assert_allclose(split, 3.0 * exact, rtol=1.0e-5, atol=1.0e-7)


def test_regularized_feet_reach_immediate_equilibrium(test, device):
    """Keep the regularized contact fixed point of three coupled feet under the split."""
    model = _three_foot_slider_model(device)
    for iterations in (32, 128):
        solver = _coupled_solver(model, "immediate", iterations, **REGULARIZED)
        (expected,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
        for response, cached in PROPAGATION_ROUTES:
            with test.subTest(response=response, cached=cached, iterations=iterations):
                solver = _coupled_solver(model, response, iterations, cached=cached, **REGULARIZED)
                (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                test.assertAlmostEqual(float(qd[0]), float(expected[0]), delta=1.0e-4)


def test_graph_replay_matches_eager_steps(test, device):
    """Recount the per-joint split on every replay of a captured step pair."""
    model = _three_foot_slider_model(device)
    for response, cached in PROPAGATION_ROUTES:
        with test.subTest(response=response, cached=cached):
            eager = _coupled_solver(model, response, 12, cached=cached, **REGULARIZED)
            expected = list(_rollout(model, eager, np.array([-1.0], dtype=np.float32), 7))[-1]
            solver = _coupled_solver(model, response, 12, cached=cached, **REGULARIZED)
            state, output = model.state(), model.state()
            control = model.control()
            state.joint_qd.assign(np.array([-1.0], dtype=np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, DT)
            with wp.ScopedCapture(device=model.device) as capture:
                for source, target in ((output, state), (state, output)):
                    pipeline.collide(source, contacts)
                    solver.step(source, target, control, contacts, DT)
            for _ in range(3):
                wp.capture_launch(capture.graph)
            np.testing.assert_allclose(output.joint_qd.numpy(), expected, atol=1.0e-6)
            test.assertEqual(int(solver._propagation_joint_split.numpy().max()), 3)


def test_linkage_gripper_holds_lifted_box(test, device):
    """Hold a box through a lift with pads whose mobility is their own passive linkage."""
    model, palm, box = _linkage_gripper_model(device)

    def box_below_palm(state):
        body_q = state.body_q.numpy()
        return float(body_q[palm, 2] - body_q[box, 2])

    for response in PROPAGATION_RESPONSES:
        with test.subTest(response=response):
            solver = SolverSteel(
                model,
                articulated_contact_response=response,
                iterations=64,
                friction_anchor_beta=0.0,
                contact_friction_gap_max=0.001,
            )
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state, output, control = model.state(), model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
            with wp.ScopedCapture(device=model.device) as capture:
                pipeline.collide(state, contacts)
                for source, target in ((state, output), (output, state)) * 2:
                    source.clear_forces()
                    solver.step(source, target, control, contacts, 1.0 / 800.0)
            joint_target = model.joint_target_q.numpy()
            grasped = None
            for tick in range(260):
                # Close the driver, then lift by 0.2 m along a smoothstep.
                joint_target[2] = 0.8 if tick >= 40 else 0.0
                lift = min(max(tick - 120, 0) / 100.0, 1.0)
                joint_target[0] = 0.2 * lift * lift * (3.0 - 2.0 * lift)
                control.joint_target_q.assign(joint_target)
                wp.capture_launch(capture.graph)
                if tick == 119:
                    grasped = box_below_palm(state)
            test.assertLess(abs(grasped - box_below_palm(state)), 0.01)


def test_floating_quadruped_with_joint_velocities_stays_bounded(test, device):
    """Keep the roll rate of four feet coupled through a light free root physical."""
    model = _quadruped_model(device)
    joint_qd = np.linspace(-1.5, 1.5, model.joint_dof_count).astype(np.float32)
    for response in PROPAGATION_RESPONSES:
        with test.subTest(response=response):
            solver = _coupled_solver(model, response, 12)
            trajectory = list(_rollout(model, solver, joint_qd, 120))
            test.assertLess(float(np.abs(trajectory[0][3:6]).max()), 10.0)
            test.assertTrue(np.isfinite(trajectory[-1]).all())
            test.assertLess(float(np.abs(np.stack(trajectory)).max()), 50.0)


class TestSteelPropagationCoupledContacts(unittest.TestCase):
    pass


for _fn in (
    test_shared_dof_feet_reach_rest,
    test_shared_dof_feet_split_their_whole_response,
    test_regularized_feet_reach_immediate_equilibrium,
    test_graph_replay_matches_eager_steps,
    test_linkage_gripper_holds_lifted_box,
    test_floating_quadruped_with_joint_velocities_stays_bounded,
):
    add_function_test(TestSteelPropagationCoupledContacts, _fn.__name__, _fn, devices=devices)


PGS_CFM = 1.0e-6


def _step_once(model, solver):
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, control, contacts, 1.0 / 200.0)
    return state_in, int(contacts.rigid_contact_count.numpy()[0])


def _analytic_H_and_X(model, state):
    """Joint-space inertia H [D,D] and per-body kinematic maps X [bodies,6,D]
    for single-world revolute trees, world frame, velocities as
    [v_com_lin, omega] (the propagation body convention)."""
    body_q = state.body_q.numpy().astype(np.float64)
    body_com = model.body_com.numpy().astype(np.float64)
    body_mass = model.body_mass.numpy().astype(np.float64)
    body_inertia = model.body_inertia.numpy().astype(np.float64)
    joint_parent = model.joint_parent.numpy().astype(np.int32)
    joint_child = model.joint_child.numpy().astype(np.int32)
    joint_axis = model.joint_axis.numpy().astype(np.float64)
    joint_X_p = model.joint_X_p.numpy().astype(np.float64)
    joint_qd_start = model.joint_qd_start.numpy().astype(np.int32)
    armature = model.joint_armature.numpy().astype(np.float64)

    def rot(q):
        x, y, z, w = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    D = int(model.joint_dof_count)
    n_bodies = model.body_count
    com_w = np.zeros((n_bodies, 3))
    R_w = np.zeros((n_bodies, 3, 3))
    for b in range(n_bodies):
        p, q = body_q[b, :3], body_q[b, 3:]
        R_w[b] = rot(q)
        com_w[b] = p + R_w[b] @ body_com[b]

    # world-frame joint axis and anchor per joint
    n_joints = joint_parent.shape[0]
    axis_w = np.zeros((n_joints, 3))
    anchor_w = np.zeros((n_joints, 3))
    parent_joint_of_body = np.full(n_bodies, -1, dtype=np.int32)
    for j in range(n_joints):
        parent = int(joint_parent[j])
        if parent >= 0:
            Rp = R_w[parent]
            pp = body_q[parent, :3]
        else:
            Rp = np.eye(3)
            pp = np.zeros(3)
        Xp_p, Xp_q = joint_X_p[j, :3], joint_X_p[j, 3:]
        anchor_w[j] = pp + Rp @ Xp_p
        axis_w[j] = (
            Rp @ rot(Xp_q) @ joint_axis[joint_qd_start[j]] if joint_axis.ndim == 2 else Rp @ rot(Xp_q) @ joint_axis[j]
        )
        parent_joint_of_body[int(joint_child[j])] = j

    X = np.zeros((n_bodies, 6, D))
    for b in range(n_bodies):
        j = int(parent_joint_of_body[b])
        while j >= 0:
            dof = int(joint_qd_start[j])
            a = axis_w[j]
            X[b, 0:3, dof] = np.cross(a, com_w[b] - anchor_w[j])
            X[b, 3:6, dof] = a
            parent = int(joint_parent[j])
            j = int(parent_joint_of_body[parent]) if parent >= 0 else -1

    H = np.zeros((D, D))
    for b in range(n_bodies):
        M = np.zeros((6, 6))
        M[0:3, 0:3] = body_mass[b] * np.eye(3)
        M[3:6, 3:6] = R_w[b] @ body_inertia[b] @ R_w[b].T
        H += X[b].T @ M @ X[b]
    H += np.diag(armature[:D])
    return H, X


def test_scissor_scene_produces_same_articulation_contact(test, device, solve_mode="matrix_free"):
    """Route a contact between two links of one articulation to the dense articulated rows."""
    model = _build_scissor_model(device)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=0, cfm=PGS_CFM)
    _, contact_count = _step_once(model, solver)
    test.assertGreater(contact_count, 0, "scissor scene produced no link-link contact")
    paths = solver._contact_path.numpy()[:contact_count]
    slots = solver._contact_slot.numpy()[:contact_count]
    routed = (paths == DENSE_PATH) & (slots >= 0)
    test.assertTrue(routed.any(), "no same-articulation contact reached the dense rows")
    test.assertEqual(int(solver._mf_constraint_count.numpy()[0]), 0)


def test_scissor_dense_diagonal_matches_reference(test, device, solve_mode="matrix_free"):
    """Match each dense row's effective mass to ``J H^-1 J^T`` from an analytic joint-space inertia."""
    model = _build_scissor_model(device)
    solver = SolverSteel(model, solve_mode=solve_mode, iterations=0, cfm=PGS_CFM)
    state, _ = _step_once(model, solver)
    H, _ = _analytic_H_and_X(model, state)
    count = int(solver._constraint_count.numpy()[0])
    test.assertGreater(count, 0)
    D = H.shape[0]
    # One articulation: the world rows are that articulation's grouped rows.
    J = solver._J_by_size[D].numpy()[0, :count, :D].astype(np.float64)
    reference = np.einsum("rd,rd->r", J, np.linalg.solve(H, J.T).T) + PGS_CFM
    got = solver._diag.numpy()[0, :count].astype(np.float64)
    np.testing.assert_allclose(got, reference, rtol=1.0e-3, atol=1.0e-6)


def _routed_slots(solver, contact_count, path):
    """Map each contact routed to ``path`` in world 0 to its normal-row slot."""
    paths = solver._contact_path.numpy()[:contact_count]
    slots = solver._contact_slot.numpy()[:contact_count]
    worlds = solver._contact_world.numpy()[:contact_count]
    return {c: int(slots[c]) for c in range(contact_count) if paths[c] == path and slots[c] >= 0 and worlds[c] == 0}


def test_default_routing_keeps_same_articulation_rows_dense(test, device):
    """Keep same-articulation contacts on the dense rows under the default propagation response."""
    model = _build_scissor_model(device)
    solver = SolverSteel(model, iterations=0, cfm=PGS_CFM, articulated_contact_response="propagation")
    _, contact_count = _step_once(model, solver)
    test.assertGreater(len(_routed_slots(solver, contact_count, DENSE_PATH)), 0)
    test.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), 0)


def test_flag_requires_propagation_mode(test, device):
    """Reject same-articulation propagation rows without the response that solves them."""
    model = _build_scissor_model(device)
    for response in ("immediate", "propagation_colored"):
        with test.subTest(response=response):
            with test.assertRaisesRegex(ValueError, "propagation_same_articulation_rows"):
                SolverSteel(model, articulated_contact_response=response, propagation_same_articulation_rows=True)


def test_cross_response_effective_mass_matches_reference_operator(test, device):
    """Match each same-articulation propagation row's ``J M^-1 J^T`` to the analytic operator.

    The generalized row Jacobian is ``g = X_a^T j_a + X_b^T j_b`` from the solver's own
    body-space rows; the reference effective mass is ``g^T H^-1 g``. Per-link responses
    alone would miss the cross term between the two links.
    """
    model = _build_scissor_model(device)
    reference = SolverSteel(model, iterations=0, cfm=PGS_CFM)
    _, contact_count = _step_once(model, reference)
    dense_slots = _routed_slots(reference, contact_count, DENSE_PATH)
    test.assertGreater(len(dense_slots), 0)

    solver = SolverSteel(
        model,
        iterations=0,
        cfm=PGS_CFM,
        articulated_contact_response="propagation",
        propagation_same_articulation_rows=True,
    )
    state, routed_count = _step_once(model, solver)
    test.assertEqual(routed_count, contact_count)
    rows = _routed_slots(solver, contact_count, PROPAGATION_PATH)
    test.assertEqual(set(rows), set(dense_slots), "same-articulation contacts were not routed to propagation rows")

    count = int(solver._propagation_constraint_count.numpy()[0])
    J_a = solver._propagation_J_a.numpy()[0, :count].astype(np.float64)
    J_b = solver._propagation_J_b.numpy()[0, :count].astype(np.float64)
    MiJt_a = solver._propagation_MiJt_a.numpy()[0, :count].astype(np.float64)
    MiJt_b = solver._propagation_MiJt_b.numpy()[0, :count].astype(np.float64)
    body_a = solver._propagation_body_a.numpy()[0, :count]
    body_b = solver._propagation_body_b.numpy()[0, :count]
    eff_mass_inv = solver._propagation_eff_mass_inv.numpy()[0, :count].astype(np.float64)
    H, X = _analytic_H_and_X(model, state)
    width = int(reference._world_dof_count.numpy()[0])
    Y = reference._Y_world.numpy()[0].astype(np.float64)

    checked = 0
    for contact, row in sorted(rows.items()):
        test.assertGreaterEqual(int(body_a[row]), 0)
        test.assertGreaterEqual(int(body_b[row]), 0)
        g = X[body_a[row]].T @ J_a[row] + X[body_b[row]].T @ J_b[row]
        y_ref = np.linalg.solve(H, g)
        # Pin frames and ordering first: the dense row response is H^-1 g up to the row sign.
        y_got = Y[dense_slots[contact], :width]
        err = min(np.linalg.norm(y_got - y_ref), np.linalg.norm(y_got + y_ref))
        test.assertLess(err / max(np.linalg.norm(y_ref), 1.0e-12), 1.0e-3)

        expected = float(g @ y_ref)
        diagonal = float(J_a[row] @ MiJt_a[row] + J_b[row] @ MiJt_b[row])
        test.assertAlmostEqual(1.0 / eff_mass_inv[row], diagonal + PGS_CFM, delta=1.0e-6 * (diagonal + 1.0))
        if expected < 1.0e-6:
            # Kinematically locked direction: the true diagonal is zero.
            test.assertLess(abs(diagonal), 1.0e-6)
            continue
        checked += 1
        test.assertLess(abs(diagonal - expected) / expected, 1.0e-3, f"contact {contact}: cross response term missing")
    test.assertGreater(checked, 0, "no non-degenerate same-articulation rows were checked")


class TestPropagationSameArticulation(unittest.TestCase):
    pass


for _name in (
    "test_scissor_scene_produces_same_articulation_contact",
    "test_scissor_dense_diagonal_matches_reference",
    "test_default_routing_keeps_same_articulation_rows_dense",
    "test_flag_requires_propagation_mode",
    "test_cross_response_effective_mass_matches_reference_operator",
):
    add_function_test(TestPropagationSameArticulation, _name, globals()[_name], devices=devices)


# The dense reference rows also run in the split solve; the propagation responses are matrix-free only.
for _name in (
    "test_scissor_scene_produces_same_articulation_contact",
    "test_scissor_dense_diagonal_matches_reference",
):
    add_function_test(
        TestPropagationSameArticulation,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        solve_mode="split",
    )


def _colored_variant_solver(model, regularization, **overrides):
    """Build the colored solver with the given sweep-variant overrides."""
    with kernel_overrides(overrides):
        return SolverSteel(
            model,
            articulated_contact_response="propagation_colored",
            iterations=8,
            free_body_constraint_max=CONTACT_MAX,
            articulation_constraint_max=64,
            contact_regularization=regularization,
        )


def _settled_heap(device, regularization):
    """A heap settled into resting contact with patch history, and its contacts."""
    model = _heap(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    state_0, state_1 = model.state(), model.state()
    warm = _colored_variant_solver(model, regularization)
    for _ in range(8):
        pipeline.collide(state_0, contacts)
        warm.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    pipeline.collide(state_0, contacts)
    start = {name: getattr(state_0, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    return model, contacts, control, start


def _run(solver, model, contacts, control, start, steps):
    state_0, state_1 = model.state(), model.state()
    for name, value in start.items():
        getattr(state_0, name).assign(value)
    for _ in range(steps):
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy(), solver._propagation_impulses.numpy()


def test_staging_requires_cp_async_and_few_worlds(test, device):
    """Stage only with cp.async (compute capability 8.0), a few worlds and fitting shared memory."""
    test.assertFalse(_colored_staging_supported(75, 1, 60))
    test.assertTrue(_colored_staging_supported(80, 1, 60))
    test.assertTrue(_colored_staging_supported(120, 16, 60))
    test.assertFalse(_colored_staging_supported(120, 17, 60))
    test.assertFalse(_colored_staging_supported(120, 1, 400))


def test_sweep_variants_match_the_reference_bitwise(test, device):
    """Reproduce the reference sweep with the prefetched, packed and staged sweeps.

    Without regularization the row-weight buffer is a placeholder the sweeps must not read.
    """
    for regularization in (0.0, 0.01):
        model, contacts, control, start = _settled_heap(device, regularization)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 400)
        staged = _colored_staging_supported(
            model.device.arch, 1, _colored_variant_solver(model, regularization)._max_propagation_bodies
        )
        variants = {
            "reference": ({"colored_prefetch": False}, "_pf0_st0"),
            "prefetch": ({"colored_packed": False}, "_pf1_st0_pk0"),
            "packed": ({}, "_pf1_st0_pk1"),
            "staged": ({"colored_staged": True}, "_pf1_st1" if staged else "_pf1_st0_pk1"),
        }
        results = {}
        for name, (overrides, tag) in variants.items():
            solver = _colored_variant_solver(model, regularization, **overrides)
            test.assertIn(tag, solver._pgs_solve_propagation_colored_warp_kernel.key)
            results[name] = _run(solver, model, contacts, control, start, 5)
        reference = results["reference"]
        test.assertTrue(all(np.isfinite(x).all() for x in reference))
        test.assertGreater(np.abs(reference[2]).max(), 0.0)
        for name in ("prefetch", "packed", "staged"):
            for label, got, want in zip(("body_q", "body_qd", "impulses"), results[name], reference, strict=True):
                with test.subTest(regularization=regularization, variant=name, field=label):
                    np.testing.assert_array_equal(got, want)


def test_straight_line_units_match_the_row_loop(test, device):
    """Agree with the row loop to rounding when standard units are solved straight-line.

    One step is compared: friction-patch decisions taken from the impulses may relayout later rows.
    """
    for regularization in (0.0, 0.01):
        model, contacts, control, start = _settled_heap(device, regularization)
        results = {}
        for name, overrides in (("row loop", {"colored_straight_line": False}), ("straight line", {})):
            solver = _colored_variant_solver(model, regularization, **overrides)
            results[name] = _run(solver, model, contacts, control, start, 1)
            if name == "straight line":
                n = int(solver._color_world_offsets.numpy()[-1])
                records = solver._color_unit_meta.numpy().reshape(-1, PROPAGATION_UNIT_META)[:n]
                test.assertGreater(int(np.sum(records[:, 6 + PROPAGATION_UNIT_RING] == 1)), n // 2)
        for label, got, want in zip(
            ("body_q", "body_qd", "impulses"), results["straight line"], results["row loop"], strict=True
        ):
            with test.subTest(regularization=regularization, field=label):
                np.testing.assert_allclose(got, want, rtol=1.0e-4, atol=1.0e-6)


class TestSteelColoredPipeline(unittest.TestCase):
    pass


add_function_test(
    TestSteelColoredPipeline,
    "test_staging_requires_cp_async_and_few_worlds",
    test_staging_requires_cp_async_and_few_worlds,
    devices=get_test_devices(),
)


for _fn in (test_sweep_variants_match_the_reference_bitwise, test_straight_line_units_match_the_row_loop):
    add_function_test(TestSteelColoredPipeline, _fn.__name__, _fn, devices=get_cuda_test_devices())


def _colored_solver(model, **kwargs):
    options = {
        "articulated_contact_response": "propagation_colored",
        "iterations": 6,
        "free_body_constraint_max": CONTACT_MAX,
        "articulation_constraint_max": 64,
    }
    options.update(kwargs)
    return SolverSteel(model, **options)


def _settle(model, solver, steps=12):
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0, int(contacts.rigid_contact_count.numpy()[0])


def _color_counts(solver):
    """Units per color of world 0, and the units of the serial tail."""
    offsets = solver._color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2].astype(np.int64)
    counts = np.diff(offsets)
    return counts[:PROPAGATION_COLOR_TAIL], int(counts[PROPAGATION_COLOR_TAIL])


def _first_fit_reference(solver, world=0):
    """Color world 0's units on the host: sort by contact, first fit, stable counting sort."""
    entries = PROPAGATION_COLOR_TAIL + 2
    stride = solver._propagation_max_constraints
    n = min(int(solver._color_world_unit_cursor.numpy()[world]), stride)
    base = world * stride
    contact = solver._color_unit_contact.numpy()[base : base + n]
    body_a = solver._color_unit_body_a.numpy()[base : base + n]
    body_b = solver._color_unit_body_b.numpy()[base : base + n]
    length = solver._color_unit_len.numpy()[base : base + n]
    order = np.argsort(contact, kind="stable")
    used = {}
    color = np.zeros(n, dtype=np.int64)
    for u in order:
        c = 0
        while c < PROPAGATION_COLOR_TAIL and any(x >= 0 and c in used.get(x, ()) for x in (body_a[u], body_b[u])):
            c += 1
        color[u] = c
        if c < PROPAGATION_COLOR_TAIL:
            for x in (body_a[u], body_b[u]):
                if x >= 0:
                    used.setdefault(x, set()).add(c)
    offsets = np.concatenate([[0], np.cumsum(np.bincount(color, minlength=entries))])[:entries]
    cursor = offsets.copy()
    position_unit = np.zeros(n, dtype=np.int64)
    for u in order:
        position_unit[cursor[color[u]]] = u
        cursor[color[u]] += 1
    row_start = np.concatenate([[0], np.cumsum(length[position_unit])])[:n]
    return n, color, offsets, row_start, contact[position_unit]


def test_dense_heap_colors_without_a_serial_tail(test, device):
    """Color a dense heap into few, full colors with nothing left for the serial tail."""
    model = _heap(device)
    solver = _colored_solver(model)
    state, contact_count = _settle(model, solver)
    test.assertTrue(np.isfinite(state.body_q.numpy()).all())
    test.assertGreater(contact_count, 400)
    counts, tail = _color_counts(solver)
    used = int(np.sum(counts > 0))
    test.assertEqual(tail, 0)
    test.assertLess(used, PROPAGATION_COLOR_TAIL // 2)
    test.assertGreater(counts[counts > 0].mean(), 8.0)


def test_kinematic_hub_does_not_serialize_its_contacts(test, device):
    """Leave a kinematic body out of the conflicts, so a tray under the heap adds few colors."""
    colors = {}
    for tray in (False, True):
        model = _heap(device, kinematic_tray=tray)
        solver = _colored_solver(model)
        state, _ = _settle(model, solver)
        test.assertTrue(np.isfinite(state.body_q.numpy()).all())
        counts, tail = _color_counts(solver)
        test.assertEqual(tail, 0)
        colors[tray] = int(np.sum(counts > 0))
    test.assertLessEqual(colors[True], 1.5 * colors[False] + 8)


def test_kinematic_free_body_is_recorded_as_static(test, device):
    """Record a kinematic free body as -1 in the units, and keep dynamic bodies."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    kinematic = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(kinematic, hx=0.1, hy=0.1, hz=0.1)
    dynamic = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.298), wp.quat_identity()))
    builder.add_shape_box(dynamic, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    model.rigid_contact_max = CONTACT_MAX
    solver = _colored_solver(model, free_body_constraint_max=256)
    _settle(model, solver, steps=1)
    n = int(solver._color_world_unit_cursor.numpy()[0])
    bodies = np.concatenate([solver._color_unit_body_a.numpy()[:n], solver._color_unit_body_b.numpy()[:n]])
    test.assertGreater(n, 0)
    test.assertNotIn(kinematic, bodies)
    test.assertIn(dynamic, bodies)


def test_colored_result_matches_the_immediate_response(test, device):
    """Settle the heap like the immediate response."""
    finals = {}
    for response in ("propagation_colored", "immediate"):
        model = _heap(device, nx=4, ny=4, nz=2)
        solver = _colored_solver(model, articulated_contact_response=response, row_watermark=True)
        state, _ = _settle(model, solver, steps=24)
        test.assertEqual(solver.constraint_row_watermarks()["free_body_dropped_contact_rows_high_water"], 0)
        finals[response] = state.body_q.numpy()[:, :3]
    test.assertTrue(np.isfinite(finals["immediate"]).all())
    # A different sweep order, so the heights agree only to a tolerance.
    test.assertLess(float(np.max(np.abs(finals["propagation_colored"][:, 2] - finals["immediate"][:, 2]))), 0.01)


def test_coloring_is_deterministic_for_a_fixed_contact_set(test, device):
    """Give the same partition and velocities for the same state and contacts.

    The units are gathered with an atomic cursor, so the coloring sorts them by contact first.
    """
    model = _heap(device)
    state, _ = _settle(model, _colored_solver(model))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=CONTACT_MAX)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    control = model.control()
    results = []
    for _ in range(3):
        state_in, state_out = model.state(), model.state()
        for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
            wp.copy(getattr(state_in, name), getattr(state, name))
        solver = _colored_solver(model)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        offsets = solver._color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2].copy()
        order = solver._color_unit_sorted.numpy()[: offsets[-1]].copy()
        results.append((offsets, order, state_out.body_qd.numpy().copy()))
    offsets, order, body_qd = results[0]
    test.assertGreater(int(offsets[-1]), 400)
    for other_offsets, other_order, other_qd in results[1:]:
        np.testing.assert_array_equal(other_offsets, offsets)
        np.testing.assert_array_equal(other_order, order)
        np.testing.assert_array_equal(other_qd, body_qd)


def test_prebuild_matches_the_first_fit_reference(test, device):
    """Reproduce sort-by-contact first-fit exactly, with shared and with global used-color masks."""
    for shape in ((6, 6, 3), (7, 7, 6)):
        with test.subTest(shape=shape):
            model = _heap(device, *shape)
            # The large heap overflows the rows; the coloring covers the units that were allocated.
            solver = _colored_solver(model, warn_constraint_overflow=False)
            state, _ = _settle(model, solver, steps=4)
            test.assertTrue(np.isfinite(state.body_q.numpy()).all())
            n, color, offsets, row_start, sorted_contact = _first_fit_reference(solver)
            test.assertGreater(n, 400)
            np.testing.assert_array_equal(solver._color_unit_color.numpy()[:n], color)
            np.testing.assert_array_equal(solver._color_world_offsets.numpy()[: PROPAGATION_COLOR_TAIL + 2], offsets)
            np.testing.assert_array_equal(solver._color_world_row_order.numpy()[:n], row_start)
            np.testing.assert_array_equal(solver._color_unit_sorted.numpy()[:n], sorted_contact)


def test_colored_accepts_a_large_row_budget(test, device):
    """Keep the per-unit coloring state in global scratch, so the row budget is uncapped."""
    model = _heap(device, nx=2, ny=2, nz=1)
    model.rigid_contact_max = 65536
    solver = _colored_solver(model, free_body_constraint_max=32768, articulation_constraint_max=2048)
    test.assertEqual(solver._propagation_max_constraints, 32768 + 2048)


class TestSteelColoredScheduling(unittest.TestCase):
    pass


for _fn in (
    test_dense_heap_colors_without_a_serial_tail,
    test_kinematic_hub_does_not_serialize_its_contacts,
    test_kinematic_free_body_is_recorded_as_static,
    test_colored_result_matches_the_immediate_response,
    test_coloring_is_deterministic_for_a_fixed_contact_set,
    test_prebuild_matches_the_first_fit_reference,
    test_colored_accepts_a_large_row_budget,
):
    add_function_test(TestSteelColoredScheduling, _fn.__name__, _fn, devices=devices)


RADIUS = 0.05


ROW_CAPACITY = 6


CONTACT_COUNT = 4


def _mixed_gap_model(device):
    """Four disjoint spheres whose ground contacts take one or three rows."""
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    world = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6, margin=0.0, gap=0.015)
    for index, gap in enumerate((-0.001, 0.004, 0.007, 0.010)):
        body = world.add_link(
            xform=wp.transform(wp.vec3(0.25 * index, 0.0, RADIUS + gap), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001),
            lock_inertia=True,
        )
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene.add_world(world)
    scene.add_ground_plane(cfg=cfg)
    return scene.finalize(device=device)


def _unit_capacity_solver(model, response):
    """Six propagation rows; friction only on the penetrating contact."""
    return SolverSteel(
        model,
        articulated_contact_response=response,
        iterations=8,
        propagation_cached_response=False,
        friction_anchor_beta=0.0,
        articulation_constraint_max=3,
        free_body_constraint_max=3,
        contact_friction_gap_max=0.0,
        contact_speculative_scale=0.0,
    )


def _seed(model, state):
    """Give every sphere the same downward speed and a distinct tangential speed."""
    joint_qd = np.zeros(state.joint_qd.shape, dtype=np.float32)
    dof_starts = model.joint_qd_start.numpy()
    for index in range(CONTACT_COUNT):
        joint_qd[int(dof_starts[index])] = 0.15 + 0.03 * index
        joint_qd[int(dof_starts[index]) + 2] = -0.1
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _unit_impulses(model, solver, contacts):
    """Propagation impulses keyed by the dynamic body of each contact."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    shapes = (contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count])
    slots = solver._contact_slot.numpy()[:count]
    lengths = solver._contact_slots_needed.numpy()[:count]
    rows = solver._propagation_impulses.numpy()[0]
    result = {}
    for contact in np.flatnonzero(solver._contact_path.numpy()[:count] == 2):
        body = max(int(shape_body[shape[contact]]) if shape[contact] >= 0 else -1 for shape in shapes)
        result[body] = rows[slots[contact] : slots[contact] + lengths[contact]].copy()
    return result


def _assert_mixed_topology(test, solver, contacts):
    """Put all four disjoint units of one or three rows into the first color."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(count, CONTACT_COUNT)
    np.testing.assert_array_equal(solver._contact_path.numpy()[:count], 2)
    test.assertEqual(sorted(solver._contact_slots_needed.numpy()[:count].tolist()), [1, 1, 1, 3])
    test.assertEqual(int(solver._propagation_constraint_count.numpy()[0]), ROW_CAPACITY)
    offsets = solver._color_world_offsets.numpy().reshape(solver._world_count, -1)[0]
    counts = np.diff(offsets)
    test.assertEqual(int(offsets[-1]), CONTACT_COUNT)
    test.assertEqual(int(counts[0]), CONTACT_COUNT)
    test.assertFalse(counts[1:].any())
    test.assertEqual(sorted(solver._color_unit_sorted.numpy()[:CONTACT_COUNT].tolist()), list(range(CONTACT_COUNT)))


def test_mixed_units_match_serial_propagation(test, device):
    """Match the serial sweep's impulses, twists and trajectory with units of mixed row counts."""
    model = _mixed_gap_model(device)
    runs = {}
    for response in ("propagation_colored", "propagation"):
        solver = _unit_capacity_solver(model, response)
        state_in, state_out = model.state(), model.state()
        _seed(model, state_in)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        runs[response] = (solver, state_in, state_out, pipeline, pipeline.contacts())
    control = model.control()
    for step in range(4):
        for response, (solver, state_in, state_out, pipeline, contacts) in runs.items():
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DT)
            runs[response] = (solver, state_out, state_in, pipeline, contacts)
        colored, colored_state, _, _, colored_contacts = runs["propagation_colored"]
        serial, serial_state, _, _, serial_contacts = runs["propagation"]
        if step == 0:
            _assert_mixed_topology(test, colored, colored_contacts)
        colored_impulses = _unit_impulses(model, colored, colored_contacts)
        serial_impulses = _unit_impulses(model, serial, serial_contacts)
        test.assertEqual(sorted(colored_impulses), sorted(serial_impulses))
        for body, expected in serial_impulses.items():
            np.testing.assert_allclose(colored_impulses[body], expected, rtol=2.0e-5, atol=2.0e-6)
        for name in ("joint_q", "joint_qd"):
            np.testing.assert_allclose(
                getattr(colored_state, name).numpy(), getattr(serial_state, name).numpy(), rtol=2.0e-5, atol=2.0e-6
            )


def test_mixed_units_graph_matches_eager(test, device):
    """Replay a captured colored step with mixed units with the eager result."""
    model = _mixed_gap_model(device)
    finals = []
    for capture in (False, True):
        solver = _unit_capacity_solver(model, "propagation_colored")
        state, output = model.state(), model.state()
        _seed(model, state)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts = pipeline.contacts()
        control = model.control()

        def substep(state=state, output=output, solver=solver, pipeline=pipeline, contacts=contacts, control=control):
            pipeline.collide(state, contacts)
            solver.step(state, output, control, contacts, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state, name), getattr(output, name))

        substep()
        if capture:
            with wp.ScopedCapture(device) as scope:
                substep()
            wp.capture_launch(scope.graph)
            _assert_mixed_topology(test, solver, contacts)
        else:
            substep()
        finals.append((state.joint_qd.numpy(), solver._propagation_body_qd.numpy()))
    for got, want in zip(finals[1], finals[0], strict=True):
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1.0e-6)


class TestSteelColoredUnitCapacity(unittest.TestCase):
    pass


for _fn in (test_mixed_units_match_serial_propagation, test_mixed_units_graph_matches_eager):
    add_function_test(TestSteelColoredUnitCapacity, _fn.__name__, _fn, devices=devices)


def _check_graph_publication(test, model):
    """Publish body state eagerly and by CUDA graph replay into poisoned buffers."""
    device = model.device
    # Hold every finite limit row: these checks cover traversal, not row capacity.
    rows = 2 * model.joint_dof_count
    solvers = [_solver(model, enabled=value, articulation_constraint_max=rows) for value in (False, True)]
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
        after[-1]["tau"] = solver._joint_tau.numpy().copy()
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
                test.assertGreater(float(np.max(np.abs(solver._qd_work.numpy() - current.joint_qd.numpy()))), 0.01)
            states[index] = [following, current]
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(states[1][0], name).numpy(),
                getattr(states[0][0], name).numpy(),
                rtol=1e-5,
                atol=2e-6,
                err_msg=f"step {step} {name}",
            )
        np.testing.assert_allclose(solvers[1]._v_hat.numpy(), solvers[0]._v_hat.numpy(), rtol=1e-5, atol=2e-6)
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


for _name in (
    "test_cuda_actual_graph_publication",
    "test_interleaved_mixed_leaves_and_moving_parents",
    "test_current_frames_and_inertial_notifications",
    "test_full_steps_and_reset",
):
    add_function_test(TestLeafPublication, _name, globals()[_name], devices=devices)


def _build_fingers(device="cpu", *, articulations=3, include_chain=False, ragged=False):
    """Build four four-joint fingers per moving palm, plus an optional serial tree."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    inertia = wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5)
    for articulation in range(articulations + int(include_chain)):
        root = builder.add_link(mass=1.0, inertia=inertia, com=wp.vec3(0.03, -0.02, 0.01))
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=root,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(2.0 * articulation, -0.3, 0.8), wp.quat_rpy(0.2, -0.1, 0.3)),
            )
        ]
        fingers = 1 if articulation == articulations else 4
        for finger in range(fingers):
            parent = root
            length = finger + 1 if ragged and articulation == 1 else 4
            for level in range(length):
                child = builder.add_link(mass=0.5, inertia=inertia, com=wp.vec3(0.02, 0.01, -0.03))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y if level % 2 else newton.Axis.X,
                        parent_xform=wp.transform(wp.vec3(0.05, 0.04 * finger, 0.08), wp.quat_rpy(0.1, -0.2, 0.05)),
                        child_xform=wp.transform(wp.vec3(-0.01, 0.02, 0.03), wp.quat_identity()),
                    )
                )
                parent = child
            if ragged and articulation == 1 and finger == 1:
                for branch in range(2):
                    child = builder.add_link(mass=0.6, inertia=inertia)
                    joints.append(
                        builder.add_joint_revolute(
                            parent=parent,
                            child=child,
                            axis=newton.Axis.X,
                            parent_xform=wp.transform(wp.vec3(0.1, 0.1 * branch, 0.2), wp.quat_identity()),
                        )
                    )
        if ragged and articulation == 2:
            parent = -1
            for _level in range(3):
                child = builder.add_link(mass=0.8, inertia=inertia, com=wp.vec3(0.02, -0.03, 0.04))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y,
                        parent_xform=wp.transform(wp.vec3(0.2, -0.1, 0.3), wp.quat_identity()),
                    )
                )
                parent = child
        builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.3, 0.4, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


def _tree_ends(model):
    return model.articulation_start.numpy()[1:]


class TestSteelTreePlan(unittest.TestCase):
    def test_parallel_tree_is_keyword_only_and_defaults_to_serial(self):
        """Require an explicit keyword-only opt-in."""
        parameters = inspect.signature(SolverSteel).parameters
        self.assertIn("parallel_tree", parameters)
        self.assertIs(parameters["parallel_tree"].default, False)
        self.assertEqual(parameters["parallel_tree"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_complete_finger_levels_and_serial_group(self):
        """Schedule every finger segment, parents before children, and keep serial trees in one-lane groups."""
        model = _build_fingers(include_chain=True)
        plan = _SteelTreePlan.build(model, _tree_ends(model))
        self.assertIsNotNone(plan)
        self.assertTrue(any(group.lanes > 1 for group in plan.groups))
        self.assertTrue(any(group.lanes == 1 for group in plan.groups))
        seen = []
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        starts = model.articulation_start.numpy()
        for group in plan.groups:
            offsets = group.level_offsets.numpy()
            segment_offsets = group.segment_offsets.numpy()
            joints = group.segment_joints.numpy()
            child_offsets = group.child_offsets.numpy()
            child_segments = group.child_segments.numpy()
            for row, articulation in enumerate(group.articulations.numpy()):
                visited = set()
                for level in range(group.max_levels):
                    previous_levels = visited.copy()
                    for segment in range(offsets[row, level], offsets[row, level + 1]):
                        segment_joints = joints[segment_offsets[segment] : segment_offsets[segment + 1]]
                        first_parent = int(parents[segment_joints[0]])
                        if first_parent >= 0:
                            self.assertIn(first_parent, previous_levels)
                        for joint in segment_joints:
                            if parents[joint] >= 0:
                                self.assertIn(int(parents[joint]), visited)
                            visited.add(int(children[joint]))
                            seen.append(int(joint))
                        child_ids = child_segments[child_offsets[segment] : child_offsets[segment + 1]]
                        child_starts = [int(joints[segment_offsets[child]]) for child in child_ids]
                        self.assertEqual(child_starts, sorted(child_starts, reverse=True))
                        for child in child_starts:
                            self.assertEqual(int(parents[child]), int(children[segment_joints[-1]]))
                if articulation < 3:
                    self.assertEqual([int(offsets[row, level + 1] - offsets[row, level]) for level in range(2)], [1, 4])
                    first, last = offsets[row, 0], offsets[row, 2]
                    np.testing.assert_array_equal(np.diff(segment_offsets[first : last + 1]), [1, 4, 4, 4, 4])
                    self.assertEqual(len(visited), int(starts[articulation + 1] - starts[articulation]))
        self.assertEqual(sorted(seen), list(range(model.joint_count)))

    def test_comb_retains_joint_levels_when_chain_compression_adds_rounds(self):
        """Keep comb teeth parallel when compressing unary chains would lengthen the schedule."""
        builder = newton.ModelBuilder()
        joints = []
        parent = -1
        for _ in range(4):
            spine = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
            joints.append(builder.add_joint_revolute(parent=parent, child=spine, axis=newton.Axis.Z))
            tooth_parent = spine
            for _ in range(4):
                tooth = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)))
                joints.append(builder.add_joint_revolute(parent=tooth_parent, child=tooth, axis=newton.Axis.Y))
                tooth_parent = tooth
            parent = spine
        builder.add_articulation(joints)
        model = builder.finalize(device="cpu")

        # Compression delays the next fork until the whole preceding tooth ends.
        compressed_lengths = [[1], [4, 1], [4, 1], [4, 5]]
        joint_widths = [1, 2, 3, 4, 4, 3, 2, 1]
        compressed_rounds = sum(max(level) for level in compressed_lengths)
        joint_rounds = sum((width + 3) // 4 for width in joint_widths)
        self.assertEqual((compressed_rounds, joint_rounds), (14, 8))

        plan = _SteelTreePlan.build(model, _tree_ends(model))
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.groups), 1)
        group = plan.groups[0]
        self.assertEqual(group.lanes, 4)
        self.assertEqual(group.max_levels, len(joint_widths))
        np.testing.assert_array_equal(np.diff(group.level_offsets.numpy()[0]), joint_widths)
        np.testing.assert_array_equal(
            np.diff(group.segment_offsets.numpy()), np.ones(model.joint_count, dtype=np.int32)
        )
        self.assertEqual(sorted(group.segment_joints.numpy().tolist()), list(range(model.joint_count)))

    def test_serial_and_unsafe_topologies_fall_back(self):
        """Return no plan for wholly serial, aliased, cross-articulation and loop-trimmed topologies."""
        model, _, _ = _build_model(chain=True)
        self.assertIsNone(_SteelTreePlan.build(model, _tree_ends(model)))
        for unsafe in ("alias", "foreign_parent", "loop"):
            with self.subTest(unsafe=unsafe):
                model = _build_fingers(articulations=2)
                ends = _tree_ends(model).copy()
                if unsafe == "alias":
                    children = model.joint_child.numpy().copy()
                    children[2] = children[1]
                    model.joint_child.assign(children)
                elif unsafe == "foreign_parent":
                    parents = model.joint_parent.numpy().copy()
                    parents[1] = model.joint_child.numpy()[model.articulation_start.numpy()[1]]
                    model.joint_parent.assign(parents)
                else:
                    ends[0] -= 1
                self.assertIsNone(_SteelTreePlan.build(model, ends))


def test_serial_selection_skips_tree_setup(test, device):
    """Build no tree plan and no branch scratch under default or explicit serial execution."""
    model = _build_fingers(device, articulations=1)
    states = []
    for options in ({}, {"parallel_tree": False}):
        with test.subTest(options=options):
            with patch.object(_SteelTreePlan, "build", wraps=_SteelTreePlan.build) as build:
                solver = SolverSteel(model, solve_mode="matrix_free", **options)
            build.assert_not_called()
            test.assertIsNone(solver._tree_plan)
            test.assertEqual(solver._tree_net_wrenches, ())
            state = model.state()
            solver._prepare_augmented_state(state)
            solver._stage7_update_kinematics(state)
            states.append(state)
    for name in ("body_q", "body_qd"):
        np.testing.assert_array_equal(getattr(states[0], name).numpy(), getattr(states[1], name).numpy())


def test_explicit_parallel_preserves_eligibility_fallback(test, device):
    """Build a plan for branched trees and fall back to serial traversal for an unbranched chain."""
    model = _build_fingers(device, articulations=1)
    with patch.object(_SteelTreePlan, "build", wraps=_SteelTreePlan.build) as build:
        solver = SolverSteel(model, solve_mode="matrix_free", parallel_tree=True)
    build.assert_called_once()
    test.assertIsNotNone(solver._tree_plan)
    test.assertTrue(any(group.lanes > 1 for group in solver._tree_plan.groups))
    test.assertEqual(len(solver._tree_net_wrenches), len(solver._tree_plan.groups))
    model = _build_model(device, chain=True)[0]
    solver = SolverSteel(model, solve_mode="matrix_free", parallel_tree=True)
    test.assertIsNone(solver._tree_plan)
    test.assertEqual(solver._tree_net_wrenches, ())


def test_backward_torques_with_live_forces_and_passive_terms(test, device):
    """Match serial joint torques and subtree wrenches with external forces, joint forces, drives and damping."""
    for velocity_limits in (False, True):
        with test.subTest(velocity_limits=velocity_limits):
            model = _build_fingers(device, ragged=True)
            model.joint_velocity_limit.fill_(0.05)
            damping = np.linspace(0.1, 0.4, model.joint_dof_count, dtype=np.float32)
            model.joint_damping.assign(damping)
            model.joint_target_ke.assign(np.linspace(0.0, 4.0, model.joint_dof_count, dtype=np.float32))
            model.joint_target_kd.assign(np.linspace(0.5, 0.0, model.joint_dof_count, dtype=np.float32))
            # Hold every finite limit row: these checks cover traversal, not row capacity.
            options = {"velocity_limits": velocity_limits, "articulation_constraint_max": 2 * model.joint_dof_count}
            solvers = [_solver(model, enabled=value, **options) for value in (False, True)]
            states = [model.state() for _ in solvers]
            following = [model.state() for _ in solvers]
            controls = [model.control() for _ in solvers]
            seed = np.linspace(-0.4, 0.3, model.joint_dof_count, dtype=np.float32)
            previous_tau = None
            for force_phase in range(2):
                for index, solver in enumerate(solvers):
                    state = states[index]
                    solver._stage1_fk_id(state, solver, following[index])
                    wrench = np.arange(model.body_count * 6, dtype=np.float32).reshape(-1, 6)
                    wrench = (wrench % 11 - 5) * (0.03 + force_phase * 0.04)
                    state.body_f.assign(wrench)
                    controls[index].joint_f.assign(seed * (1.0 + force_phase))
                    solver._joint_tau.fill_(-123.25)
                    for scratch in solver._tree_net_wrenches:
                        scratch.fill_(-123.25)
                    solver._stage1_joint_tau(state, solver, following[index], controls[index], 1.0 / 240.0)
                for name in ("_body_ft_s", "_joint_tau", "_aug_row_counts", "_aug_row_dof_index", "_aug_row_K"):
                    np.testing.assert_allclose(
                        getattr(solvers[1], name).numpy(),
                        getattr(solvers[0], name).numpy(),
                        rtol=3e-6,
                        atol=3e-6,
                        err_msg=f"forces {force_phase}: {name}",
                    )
                tau = solvers[1]._joint_tau.numpy().copy()
                if previous_tau is not None:
                    test.assertGreater(float(np.max(np.abs(tau - previous_tau))), 0.01)
                previous_tau = tau


def test_stage1_matches_serial_with_velocity_prescale(test, device):
    """Match every forward-pass field across partial warps, a state edit and active velocity prescaling."""
    for velocity_limits in (False, True):
        with test.subTest(velocity_limits=velocity_limits):
            model = _build_fingers(device, ragged=True)
            model.joint_velocity_limit.fill_(0.05)
            # Hold every finite limit row: these checks cover traversal, not row capacity.
            options = {"velocity_limits": velocity_limits, "articulation_constraint_max": 2 * model.joint_dof_count}
            solvers = [_solver(model, enabled=value, **options) for value in (False, True)]
            states = [model.state() for _ in solvers]
            outputs = [model.state() for _ in solvers]
            for solver, state in zip(solvers, states, strict=True):
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                _poison(_dynamics_fields(solver, state), keep=("body_q",))
            original_qd = model.joint_qd.numpy().copy()
            for phase in ("first", "edited"):
                for solver, state, following in zip(solvers, states, outputs, strict=True):
                    if phase == "edited":
                        q = state.joint_q.numpy().copy()
                        start, end = model.articulation_start.numpy()[1:3]
                        q0, q1 = model.joint_q_start.numpy()[[start, end]]
                        q[q0:q1] += 0.1
                        state.joint_q.assign(q)
                    predictor_qd = _run_stage1(solver, state, following)
                    np.testing.assert_array_equal(state.joint_qd.numpy(), original_qd)
                    if velocity_limits:
                        test.assertGreater(float(np.max(np.abs(predictor_qd.numpy() - original_qd))), 0.01)
                fields = [_dynamics_fields(solver, state) for solver, state in zip(solvers, states, strict=True)]
                for name, actual in fields[1].items():
                    np.testing.assert_allclose(
                        actual.numpy(), fields[0][name].numpy(), rtol=3e-6, atol=3e-6, err_msg=f"{phase}: {name}"
                    )
                if velocity_limits:
                    np.testing.assert_array_equal(solvers[1]._qd_work.numpy(), solvers[0]._qd_work.numpy())


def test_publication_matches_public_fk(test, device):
    """Publish FREE and mixed D6 states with the public forward-kinematics convention."""
    model, _, _ = _build_branched_model(device, floating_root=True)
    solver = _solver(model, enabled=True)
    state = model.state()
    state.body_q.fill_(-123.25)
    state.body_qd.fill_(-123.25)
    solver._stage7_update_kinematics(state)
    reference = model.state()
    newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
    for name in ("body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6, err_msg=name
        )


def test_contact_trajectory_matches_serial(test, device):
    """Match serial traversal over a contact-loaded trajectory with a branched articulation."""
    model = _build_contact_model(device)
    model.rigid_contact_max = 32
    results = []
    for parallel in (False, True):
        solver = _solver(model, enabled=parallel)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        contacts = pipeline.contacts()
        state, following = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        for _ in range(40):
            pipeline.collide(state, contacts)
            solver.step(state, following, control, contacts, 1.0 / 240.0)
            state, following = following, state
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        results.append(state)
    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
        np.testing.assert_allclose(
            getattr(results[1], name).numpy(), getattr(results[0], name).numpy(), rtol=1e-5, atol=2e-6, err_msg=name
        )


def _run_split_fingers(device, parallel_tree):
    model = _build_fingers(device, articulations=2, ragged=True)
    solver = SolverSteel(model, solve_mode="split", parallel_tree=parallel_tree)
    state, next_state = model.state(), model.state()
    control = model.control()
    for _ in range(10):
        solver.step(state, next_state, control, None, 1.0 / 240.0)
        state, next_state = next_state, state
    solver.check_constraint_capacity()
    return solver, state.body_q.numpy(), state.body_qd.numpy()


class TestSteelSplitCpuTree(unittest.TestCase):
    def test_branched_split_cpu_keeps_serial_passes(self):
        """Run branched split trees on CPU with the serial traversal and scalar composite inertia.

        A requested parallel traversal falls back to the serial one bitwise, and the CPU trajectory
        matches the CUDA split solve, whose cooperative kernels do not run on CPU.
        """
        serial = _run_split_fingers("cpu", False)
        parallel = _run_split_fingers("cpu", True)
        for solver in (serial[0], parallel[0]):
            self.assertIsNone(solver._tree_plan)
            self.assertIsNone(solver._composite_inertia_warp_kernel)
            self.assertGreater(solver._composite_articulation_count, 0)
        for cpu, fallback in zip(serial[1:], parallel[1:], strict=True):
            self.assertTrue(np.isfinite(cpu).all())
            np.testing.assert_array_equal(cpu, fallback)
        if wp.is_cuda_available():
            cuda = _run_split_fingers("cuda:0", False)
            for cpu, gpu in zip(serial[1:], cuda[1:], strict=True):
                np.testing.assert_allclose(cpu, gpu, rtol=1.0e-4, atol=1.0e-4)


class TestSteelTreeExecution(unittest.TestCase):
    pass


for _name in (
    "test_serial_selection_skips_tree_setup",
    "test_explicit_parallel_preserves_eligibility_fallback",
    "test_backward_torques_with_live_forces_and_passive_terms",
    "test_stage1_matches_serial_with_velocity_prescale",
    "test_publication_matches_public_fk",
    "test_contact_trajectory_matches_serial",
):
    add_function_test(TestSteelTreeExecution, _name, globals()[_name], devices=devices)


SCHEDULES = ("interleaved", "contact_then_internal", "phased")


# Scissor link offset whose links start just apart, with speculative contact rows.
_SPECULATIVE_OFFSET = 0.031


def _schedule_scene(device, worlds=1, offset=0.028):
    """A limited, driven three-link scissor next to a free body above its speed bound.

    At the default ``offset`` the two links overlap slightly, so the first step has contact rows.
    """
    template = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(template)
    template.default_shape_cfg.mu = 0.5
    base = template.add_link()
    template.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    joints = [
        template.add_joint_revolute(
            -1,
            base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            limit_lower=-0.001,
            limit_upper=0.001,
        )
    ]
    for side in (1.0, -1.0):
        link = template.add_link()
        template.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04)
        joints.append(
            template.add_joint_revolute(
                base,
                link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.06, side * offset, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                target_ke=50.0 if side > 0 else 0.0,
                target_kd=2.0 if side > 0 else 0.0,
                target_pos=0.3,
            )
        )
    template.add_articulation(joints)
    template.add_body(
        xform=wp.transform(wp.vec3(2.0, 0.0, 2.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.eye(3) * 1.0e-2),
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 0.001,
            "steel:rigid_body_max_angular_velocity": 0.001,
        },
    )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    model = builder.finalize(device=device)
    joint_qd = model.joint_qd.numpy()
    # Fast enough to close the 0.001 rad gap to the upper limit within one step.
    joint_qd[0::4] = 5.0
    model.joint_qd.assign(joint_qd)
    return model


def _schedule_solver(model, **kwargs):
    options = {
        "enable_joint_limits": True,
        "friction_anchor_beta": 0.0,
        "drive_mode": "rows",
        "articulation_constraint_max": 64,
    }
    options.update(kwargs)
    return SolverSteel(model, **options)


def _schedule_step(model, solver, steps=1):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return state_0


def test_schedule_validation(test, device):
    """Accept the three schedules and reject unknown names and unsupported combinations."""
    model = _schedule_scene(device)
    for schedule in SCHEDULES:
        test.assertEqual(SolverSteel(model, row_schedule=schedule).row_schedule, schedule)
    with test.assertRaisesRegex(ValueError, "row_schedule"):
        SolverSteel(model, row_schedule="bad")
    for schedule in SCHEDULES[1:]:
        with test.subTest(schedule=schedule):
            set_torsion_radius(model, 0.01)
            with test.assertRaisesRegex(NotImplementedError, "contact torsion.*row_schedule='interleaved'"):
                SolverSteel(model, row_schedule=schedule)
            set_torsion_radius(model, 0.0)
            with test.assertRaisesRegex(NotImplementedError, "contact_compliance.*row_schedule='interleaved'"):
                SolverSteel(model, row_schedule=schedule, contact_compliance=True, friction_anchor_beta=0.0)
    for response in ("propagation", "propagation_colored"):
        for schedule in SCHEDULES[1:]:
            with test.subTest(response=response, schedule=schedule):
                solver = SolverSteel(model, row_schedule=schedule, articulated_contact_response=response)
                test.assertEqual(solver.row_schedule, schedule)


def test_split_rejects_schedules(test, device):
    """Reject every schedule but the interleaved one in the split solve."""
    model = _schedule_scene(device)
    for schedule in SCHEDULES[1:]:
        with test.subTest(schedule=schedule):
            with test.assertRaisesRegex(NotImplementedError, "row_schedule.*requires solve_mode='matrix_free'"):
                SolverSteel(model, solve_mode="split", row_schedule=schedule)


def test_row_phases_touch_only_their_row_families(test, device):
    """Restrict each row phase of the matrix-free kernel to its row families.

    The scene has a PGS drive row, an active joint limit, contacts between two links of one
    articulation (dense rows) and a free body above its velocity bound (velocity-limit rows).
    """
    model = _schedule_scene(device)
    solver = _schedule_solver(model, iterations=0, row_schedule="phased")
    _schedule_step(model, solver)
    dense_count = int(solver._constraint_count.numpy()[0])
    row_type = solver._row_type.numpy()[0, :dense_count]
    families = {
        "drive": row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET,
        "limit": row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
        "contact": (row_type == PGS_CONSTRAINT_TYPE_CONTACT) | (row_type == PGS_CONSTRAINT_TYPE_FRICTION),
    }
    for name, mask in families.items():
        test.assertTrue(mask.any(), name)
    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    test.assertGreater(mf_count, mf_contact_end)

    touched = {}
    for row_phase in (1, 2, 3, 4, 5):
        solver._impulses.zero_()
        solver._mf_impulses.zero_()
        wp.copy(solver._v_out, solver._v_hat)
        solver._launch_mf_gs_phase(row_phase)
        dense = np.abs(solver._impulses.numpy()[0, :dense_count]) > 0.0
        velocity_limits = np.abs(solver._mf_impulses.numpy()[0, mf_contact_end:mf_count]) > 0.0
        touched[row_phase] = (*(bool(dense[mask].any()) for mask in families.values()), bool(velocity_limits.any()))
    # (drive, limit, contact, velocity limit)
    test.assertEqual(touched[1], (False, False, True, False))
    test.assertEqual(touched[2], (True, True, False, True))
    test.assertEqual(touched[3], (True, True, False, False))
    test.assertEqual(touched[4], (False, False, True, False))
    test.assertEqual(touched[5], (False, False, False, True))


def test_phased_keeps_the_interleaved_row_order(test, device):
    """Match the interleaved sweep: the dense layout already puts internal rows before contacts."""
    for iterations in (1, 12):
        results = []
        for schedule in ("interleaved", "phased"):
            model = _schedule_scene(device, worlds=2, offset=_SPECULATIVE_OFFSET)
            results.append(
                _schedule_step(
                    model, _schedule_solver(model, iterations=iterations, row_schedule=schedule)
                ).joint_qd.numpy()
            )
        np.testing.assert_allclose(results[1], results[0], rtol=0.0, atol=1.0e-6)


def test_contact_then_internal_gives_internal_rows_the_last_word(test, device):
    """Leave the internal rows at their fixed point after the contact iterations, unlike one interleaved sweep."""
    changes = {}
    for schedule, iterations in (("contact_then_internal", 12), ("interleaved", 1)):
        model = _schedule_scene(device, offset=_SPECULATIVE_OFFSET)
        solver = _schedule_solver(model, iterations=iterations, row_schedule=schedule)
        _schedule_step(model, solver)
        dense_count = int(solver._constraint_count.numpy()[0])
        row_type = solver._row_type.numpy()[0, :dense_count]
        internal = (row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET) | (row_type == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        before = solver._impulses.numpy()[0, :dense_count][internal].copy()
        solver._launch_mf_gs_phase(3)
        after = solver._impulses.numpy()[0, :dense_count][internal]
        changes[schedule] = float(np.abs(after - before).max())
    test.assertLess(changes["contact_then_internal"], 1.0e-4)
    test.assertGreater(changes["interleaved"], 1.0e-3)


def test_propagation_schedules(test, device):
    """Run the propagation response with every schedule; phased is its interleaved order."""
    results = {}
    for schedule in SCHEDULES:
        model = _schedule_scene(device, offset=_SPECULATIVE_OFFSET)
        solver = _schedule_solver(
            model,
            drive_mode="augmented",
            iterations=200,
            row_schedule=schedule,
            articulated_contact_response="propagation",
        )
        results[schedule] = _schedule_step(model, solver, steps=3).joint_qd.numpy()
        test.assertTrue(np.isfinite(results[schedule]).all())
    np.testing.assert_array_equal(results["phased"], results["interleaved"])


def test_captured_schedules_match_eager(test, device):
    """Replay every schedule's step in a CUDA graph with the eager result."""
    for schedule in SCHEDULES:
        with test.subTest(schedule=schedule):
            finals = []
            for capture in (False, True):
                model = _schedule_scene(device, offset=_SPECULATIVE_OFFSET)
                solver = _schedule_solver(model, iterations=4, row_schedule=schedule)
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                pipeline = newton.CollisionPipeline(model, deterministic=True)
                contacts = pipeline.contacts()
                control = model.control()

                def substep(
                    state_0=state_0,
                    state_1=state_1,
                    contacts=contacts,
                    solver=solver,
                    pipeline=pipeline,
                    control=control,
                ):
                    pipeline.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                    for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                        wp.copy(getattr(state_0, name), getattr(state_1, name))

                substep()
                if capture:
                    with wp.ScopedCapture(device) as scope:
                        substep()
                    wp.capture_launch(scope.graph)
                else:
                    substep()
                finals.append(state_0.joint_qd.numpy())
            np.testing.assert_allclose(finals[1], finals[0], rtol=0.0, atol=1.0e-5)


class TestSteelSchedule(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _fn in (
    test_schedule_validation,
    test_row_phases_touch_only_their_row_families,
    test_phased_keeps_the_interleaved_row_order,
    test_contact_then_internal_gives_internal_rows_the_last_word,
    test_propagation_schedules,
    test_captured_schedules_match_eager,
):
    add_function_test(TestSteelSchedule, _fn.__name__, _fn, devices=cuda_devices)


add_function_test(
    TestSteelSchedule, "test_split_rejects_schedules", test_split_rejects_schedules, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main()
