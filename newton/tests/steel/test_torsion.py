# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact torsion: per-shape spin friction, host and device preparation of SolverSteel."""

import inspect
import itertools
import unittest
import warnings
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.steel import contact_torsion, solver_steel
from newton._src.solvers.steel import contact_torsion as host
from newton._src.solvers.steel.contact_torsion import _contact_groups, _group_budget
from newton._src.solvers.steel.contact_torsion_device import (
    DeviceTorsionPreparation,
    _host_dot3,
    _host_transform_point,
    enable_device_torsion,
)
from newton._src.solvers.steel.friction_patches import link_patch_rows
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import (
    PATCH_OPTIONS,
    _assert_budget,
    _build_four_bar,
    _loop_anchor_gap,
    _patch_fixture,
    fixture,
    set_torsion_radius,
)
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_cuda_test_devices, get_test_devices

# SAT box-box manifolds are a separate collision feature; their subcases run where it exists.
_HAS_BOX_BOX_SAT = "box_box_sat" in inspect.signature(newton.CollisionPipeline.__init__).parameters


def held_box(radius, torque, *, device="cuda:0", steps=20, dt=0.0025, mu=0.5, mass=0.3, half=0.005):
    """Rest a small articulated box on the ground under gravity and a yaw torque with patch friction.

    The square footprint keeps the two patch anchors within ``sqrt(2) * half`` of the
    center, so their own twist capacity stays far below ``mu * N * radius``.
    """
    b = newton.ModelBuilder()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu, restitution=0))
    pose = wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity())
    body = b.add_link(xform=pose, mass=mass, inertia=wp.mat33(np.diag([3e-4, 3e-4, 3e-4]).astype(np.float32)))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
    b.add_articulation([joint])
    b.add_shape_box(
        body, hx=half, hy=half, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=0)
    )
    model = b.finalize(device=device)
    set_torsion_radius(model, radius)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=64,
        velocity_iterations=0,
        baumgarte=0.05,
        cfm=0,
        contact_regularization=0,
        contact_warmstart=False,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        **PATCH_OPTIONS,
    )
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(
        model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
    )
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    force = np.zeros(model.joint_dof_count, np.float32)
    force[5] = torque
    control.joint_f.assign(force)
    for _ in range(steps):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    return float(s0.body_qd.numpy()[0, 5]), solver


def patch_rows(
    points, *, device, slots, slots_needed, row_type, parents, mu, patches_enabled=True, shape_type=GeoType.BOX
):
    """Link one patch region into dense rows and return the torsion host view of it.

    Rows follow the dense builder's layout: every contact keeps its normal, and only
    anchors receive an adjacent tangent pair. ``link_patch_rows`` then closes the
    normal-parent load ring and divides the anchors' friction coefficient.
    """
    model, state, contacts, patches = _patch_fixture(points, device=device)
    model.shape_type = wp.full(3, int(shape_type), dtype=int, device=device)
    contacts.rigid_contact_max = len(points)
    contacts.rigid_contact_stiffness = None
    n = len(points)
    solver = SimpleNamespace(
        model=model,
        _contact_path=wp.zeros(n, dtype=int, device=device),
        _contact_slot=wp.array(slots, dtype=int, device=device),
        _contact_world=wp.zeros(n, dtype=int, device=device),
        _contact_slots_needed=wp.array(slots_needed, dtype=int, device=device),
        _row_type=wp.array([row_type], dtype=int, device=device),
        _row_parent=wp.array([parents], dtype=int, device=device),
        _row_mu=wp.array([mu], dtype=float, device=device),
        _constraint_count=wp.array([len(row_type)], dtype=int, device=device),
        _friction_anchors_enabled=patches_enabled,
        _friction_patches=patches,
        _contact_torsion_shape_radius=wp.full(model.shape_body.shape[0], 0.01, dtype=float, device=device),
    )
    if patches_enabled:
        wp.launch(
            link_patch_rows,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                patches.view,
                solver._contact_world,
                solver._contact_slot,
                solver._contact_path,
                solver._contact_slots_needed,
                0,
                solver._row_parent,
                solver._row_mu,
            ],
            device=device,
        )
    return solver, state, contacts


def test_two_anchor_region_pools_the_undivided_coefficient(test, device):
    """Budget one region by the full coefficient on every ring normal, consuming only anchor rows."""
    solver, state, contacts = patch_rows(
        [[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]],
        device=device,
        slots=[0, 3, 4],
        slots_needed=[3, 1, 3],
        row_type=[0, 2, 2, 0, 0, 2, 2],
        parents=[-1, 0, 0, -1, -1, 4, 4],
        mu=[0.5] * 7,
    )
    np.testing.assert_array_equal(solver._row_parent.numpy()[0], [3, 0, 0, 4, 0, 4, 4])
    np.testing.assert_allclose(solver._row_mu.numpy()[0], [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25])
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    test.assertEqual(sorted(w.slot for w in groups[0]), [0, 3, 4])
    coefficient, anchors, touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
    test.assertAlmostEqual(coefficient, 0.5)
    test.assertEqual(sorted(w.slot for w in anchors), [0, 4])
    test.assertEqual(len(touching), 3)


def test_single_anchor_region_keeps_the_coefficient(test, device):
    """Pool coincident normals behind one anchor without scaling the coefficient."""
    solver, state, contacts = patch_rows(
        [[0, 0, 0], [0, 0, 0]],
        device=device,
        slots=[0, 3],
        slots_needed=[3, 1],
        row_type=[0, 2, 2, 0],
        parents=[-1, 0, 0, -1],
        mu=[0.5] * 4,
    )
    np.testing.assert_array_equal(solver._row_parent.numpy()[0], [3, 0, 0, 0])
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    test.assertEqual(sorted(w.slot for w in groups[0]), [0, 3])
    coefficient, anchors, _touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
    test.assertAlmostEqual(coefficient, 0.5)
    test.assertEqual([w.slot for w in anchors], [0])


def test_point_friction_grouping_is_unchanged(test, device):
    """Keep every point-friction witness on its own tangent pair with the row coefficient."""
    solver, state, contacts = patch_rows(
        [[0, 0, 0], [0, 0, 0]],
        device=device,
        slots=[0, 3],
        slots_needed=[3, 3],
        row_type=[0, 2, 2, 0, 2, 2],
        parents=[-1, 0, 0, -1, 3, 3],
        mu=[0.5] * 6,
        patches_enabled=False,
    )
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    coefficient, anchors, _touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
    test.assertAlmostEqual(coefficient, 0.5)
    test.assertEqual(sorted(w.slot for w in anchors), [0, 3])


def test_region_without_anchor_rows_carries_no_spin(test, device):
    """Skip a region whose members all lost their tangent rows to friction filters."""
    solver, state, contacts = patch_rows(
        [[-0.1, 0, 0], [0.1, 0, 0]],
        device=device,
        slots=[0, 1],
        slots_needed=[1, 1],
        row_type=[0, 0],
        parents=[-1, -1],
        mu=[0.5] * 2,
    )
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    test.assertIsNone(_group_budget(solver, groups[0], solver._row_mu.numpy()))


def test_region_with_inadmissible_member_is_skipped(test, device):
    """Skip the whole region rather than budget a partial ring."""
    solver, state, contacts = patch_rows(
        [[-0.1, 0, 0], [0.1, 0, 0]],
        device=device,
        slots=[0, 3],
        slots_needed=[3, 3],
        row_type=[0, 2, 2, 0, 2, 2],
        parents=[-1, 0, 0, -1, 3, 3],
        mu=[0.5] * 6,
        shape_type=GeoType.MESH,
    )
    test.assertEqual(_contact_groups(solver, state, contacts), [])


def test_broken_load_ring_is_rejected(test, device):
    """Refuse a region whose normal-parent ring does not close over its own rows."""
    solver, state, contacts = patch_rows(
        [[-0.1, 0, 0], [0.1, 0, 0]],
        device=device,
        slots=[0, 3],
        slots_needed=[3, 3],
        row_type=[0, 2, 2, 0, 2, 2],
        parents=[-1, 0, 0, -1, 3, 3],
        mu=[0.5] * 6,
    )
    parents = solver._row_parent.numpy()
    parents[0, 3] = -1
    solver._row_parent.assign(parents)
    with test.assertRaisesRegex(RuntimeError, "load ring"):
        _contact_groups(solver, state, contacts)


def test_default_off_equivalent(test, device):
    """Keep a model without torsion radii exactly equivalent to the zero-radius output."""
    reference, *_ = fixture(None, device=device)
    actual, *_ = fixture(0.0, device=device)
    for key in reference:
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)


def test_spin_stops_below_bound(test, device):
    """Remove spin below the prescribed disk's Coulomb torque capacity."""
    baseline, *_ = fixture(0.0, device=device, spin=1.0, center_only=True)
    test.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
    result, solver, *_ = fixture(0.01, device=device, spin=1.0, center_only=True)
    test.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
    test.assertGreater(solver._torsion_stats["rows"], 0)


def test_zero_friction_and_zero_load(test, device):
    """Apply no torsion when friction or compressive normal load vanishes."""
    for options in ({"mu": 0.0}, {"closing": 0.0}, {"separation": 0.004}):
        actual, solver, *_ = fixture(0.01, device=device, **options)
        if "closing" in options:
            test.assertGreater(solver._torsion_stats["rows"], 0)
        else:
            test.assertEqual(solver._torsion_stats["rows"], 0)
        active = actual["row_type"] == PGS_CONSTRAINT_TYPE_TORSION
        test.assertLess(np.abs(actual["impulses"][active]).max(initial=0), 1e-9)


def test_joint_response_and_coupled_budget(test, device):
    """Respect Coulomb sharing, articulated response and kinetic-energy bounds."""
    # One central witness, the native face manifold, and (where available) the SAT manifold.
    manifolds = [(False, True), (False, False)] + ([(True, False)] if _HAS_BOX_BOX_SAT else [])
    for sat, center_only in manifolds:
        for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0), (100.0, 10.0)):
            actual, solver, _model, initial, _ = fixture(
                0.01, device=device, spin=spin, sliding=sliding, sat=sat, center_only=center_only
            )
            # Evaluate contact response at its input pose, before D6
            # integration changes the motion-basis readback. The existing
            # predictor already changes lateral momentum at extreme mixed
            # velocity; this gate isolates the contact solve from that predictor.
            v0 = actual["v_hat"].reshape(2, 6)
            v1 = actual["v_out"].reshape(2, 6)
            inertia = np.array([0.00012, 0.00012, 0.00008])

            def energy(v, inertia=inertia):
                return float(0.5 * 0.3 * np.sum(v[:, :3] ** 2) + 0.5 * np.sum(v[:, 3:] ** 2 * inertia))

            test.assertLessEqual(energy(v1), energy(v0) + 2e-6)
            np.testing.assert_allclose(np.sum(v1[:, :3], axis=0), np.sum(v0[:, :3], axis=0), atol=1e-5)
            positions = initial.body_q.numpy()[:, :3]

            def angular(v, inertia=inertia, positions=positions):
                return np.sum(v[:, 3:] * inertia + np.cross(positions, 0.3 * v[:, :3]), axis=0)

            np.testing.assert_allclose(angular(v1), angular(v0), atol=2e-6)
            count = int(actual["count"][0])
            impulse = actual["impulses"][0, :count]
            response = actual["Y_world"][0, :count].T @ impulse
            np.testing.assert_allclose(actual["v_out"] - actual["v_hat"], response, atol=2e-5)
            for group in solver._torsion_stats["groups"]:
                rows = group["normal_rows"]
                budget = 0.5 * sum(max(float(impulse[r]), 0) for r in rows)
                used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in rows)
                used += abs(float(impulse[group["row"]])) / 0.01
                test.assertLessEqual(used, budget + 2e-6)


def test_witness_count_does_not_multiply_torque(test, device):
    """Share one footprint budget across repeated normal quadrature witnesses."""
    outputs = []
    for count in (1, 2, 4):
        result, solver, *_ = fixture(0.01, device=device, spin=100.0, center_only=True, center_count=count)
        test.assertEqual(solver._torsion_stats["rows"], 1)
        outputs.append(result["body_qd"])
    for output in outputs[1:]:
        np.testing.assert_allclose(output, outputs[0], atol=2e-5)


def test_release_and_reset_carry_no_torque(test, device):
    """Forget spin impulse on contact loss and reproduce cold state after reset."""
    expected, solver, model, initial, contacts = fixture(0.01, device=device, spin=100.0)
    output = model.state()
    count = contacts.rigid_contact_count.numpy()
    contacts.rigid_contact_count.zero_()
    solver.step(initial, output, model.control(), contacts, 0.0025)
    test.assertEqual(solver._torsion_stats["rows"], 0)
    contacts.rigid_contact_count.assign(count)
    solver.reset(initial)
    solver.step(initial, output, model.control(), contacts, 0.0025)
    np.testing.assert_allclose(output.body_qd.numpy(), expected["body_qd"], atol=2e-5)


def test_timestep_and_capacity(test, device):
    """Keep impulse-level overload behavior across dt and reject insufficient rows."""
    velocities = []
    for dt in (0.00125, 0.0025, 0.005):
        result, solver, *_ = fixture(0.01, device=device, spin=100.0, center_only=True, dt=dt)
        velocities.append(result["v_out"])
        test.assertFalse(np.any(solver.constraint_overflow.numpy()))
        test.assertLessEqual(int(solver._constraint_count.numpy().max()), solver.articulation_constraint_max)
    for velocity in velocities[1:]:
        np.testing.assert_allclose(velocity, velocities[0], atol=2e-5)
    baseline, *_ = fixture(0.0, device=device, center_only=True)
    with test.assertRaisesRegex(RuntimeError, "capacity exceeded"):
        fixture(0.01, device=device, center_only=True, row_limit=int(baseline["count"][0]))


def test_per_shape_radius_selects_contacts(test, device):
    """Apply torsion when either shape of the pair has a radius, using the larger one."""
    baseline, *_ = fixture(0.0, device=device, center_only=True)
    test.assertLess(0.01, float(np.max(np.abs(baseline["body_qd"][:, 5]))))
    for shapes in ((0,), (1,), (0, 1)):
        with test.subTest(shapes=shapes):
            result, solver, *_ = fixture(0.01, device=device, center_only=True, torsion_shapes=shapes)
            test.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
            test.assertEqual(solver._torsion_stats["rows"], 1)
            test.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.01))
    # A pair uses the larger radius, so a smaller second radius gives the same rows.
    reference, *_ = fixture(0.01, device=device, center_only=True)
    mixed, *_ = fixture(np.array([0.01, 1.0e-4], dtype=np.float32), device=device, center_only=True)
    for key in reference:
        np.testing.assert_array_equal(mixed[key], reference[key], err_msg=key)


def test_invalid_input_and_unsupported_modes(test, device):
    """Reject invalid radii and modes rather than silently ignoring spin friction."""
    for radius in (-1.0, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            fixture(radius, device=device)
    with test.assertRaises(ValueError):
        fixture(0.01, device=device, contact_warmstart=True)


def test_capture_is_explicitly_rejected(test, device):
    """Reject capture of the host preparation before host contact grouping is attempted."""
    _, solver, model, initial, contacts = fixture(0.01, device=device)
    output = model.state()
    with test.assertRaisesRegex(RuntimeError, "graph capture"):
        with wp.ScopedCapture(device=device):
            solver.step(initial, output, model.control(), contacts, 0.0025)


def test_hydro_combination_is_rejected(test, device):
    """Reject actual hydro contact stiffness before combining contact mechanisms."""
    _, solver, model, initial, contacts = fixture(0.01, device=device)
    contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
    with test.assertRaisesRegex(ValueError, "hydroelastic"):
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)


def test_constructor_accepts_torsion_with_patches(test, device):
    """Build torsion on top of default and explicit patch friction without friction warnings."""
    for options in ({"friction_anchor_beta": None}, {}):
        with test.subTest(options=options), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, solver, *_ = fixture(0.01, device=device, center_only=True, **{**PATCH_OPTIONS, **options})
        test.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
        test.assertTrue(solver._contact_torsion_enabled)
        test.assertTrue(solver._friction_anchors_enabled)
        test.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
        test.assertEqual(solver._torsion_stats["rows"], 1)


def test_patch_regions_pool_the_spin_budget(test, device):
    """Share mu times the region's pooled normal load between anchor sliding and spin."""
    cases = [(False, True, 1), (False, False, 2)] + ([(True, False, 2)] if _HAS_BOX_BOX_SAT else [])
    for sat, center_only, anchors in cases:
        with test.subTest(sat=sat, anchors=anchors):
            actual, solver, *_ = fixture(
                0.01,
                device=device,
                spin=100.0,
                sliding=10.0,
                sat=sat,
                center_only=center_only,
                **PATCH_OPTIONS,
            )
            test.assertEqual(solver._torsion_stats["rows"], 1)
            group = solver._torsion_stats["groups"][0]
            test.assertEqual(len(group["anchor_rows"]), anchors)
            test.assertGreaterEqual(len(group["normal_rows"]), anchors)
            test.assertAlmostEqual(group["mu"], 0.5, places=6)
            test.assertAlmostEqual(float(actual["row_mu"][0, group["row"]]), 0.5, places=6)
            for slot in group["anchor_rows"]:
                test.assertAlmostEqual(float(actual["row_mu"][0, slot + 1]), 0.5 / anchors, places=6)
                test.assertEqual(list(actual["row_type"][0, slot + 1 : slot + 3]), [2, 2])
            for slot in set(group["normal_rows"]) - set(group["anchor_rows"]):
                test.assertNotEqual(int(actual["row_type"][0, slot + 1]), 2)
            impulse = actual["impulses"][0, : int(actual["count"][0])]
            pooled = sum(max(float(impulse[r]), 0.0) for r in group["normal_rows"])
            used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in group["anchor_rows"])
            used += abs(float(impulse[group["row"]])) / 0.01
            test.assertGreater(pooled, 0.0)
            test.assertLessEqual(used, 0.5 * pooled + 2e-6)


def test_single_anchor_spin_stops_below_bound(test, device):
    """Stop a slow spin that a lone patch anchor cannot resist on its own."""
    baseline, solver, *_ = fixture(0.0, device=device, spin=1.0, center_only=True, **PATCH_OPTIONS)
    test.assertTrue(solver._friction_anchors_enabled)
    test.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
    result, solver, *_ = fixture(0.01, device=device, spin=1.0, center_only=True, **PATCH_OPTIONS)
    test.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
    test.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 1)


def test_held_box_yaw_torque_threshold(test, device):
    """Hold a resting box below mu * N * radius and let it spin above, with patch friction."""
    radius, mu, mass = 0.05, 0.5, 0.3
    bound = mu * mass * 9.81 * radius
    held, solver = held_box(radius, 0.5 * bound, device=device, mu=mu, mass=mass)
    test.assertEqual(solver._torsion_stats["rows"], 1)
    test.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 2)
    test.assertLess(abs(held), 1e-3)
    released, _ = held_box(radius, 2.0 * bound, device=device, mu=mu, mass=mass)
    test.assertGreater(released, 1.0)
    anchors_only, _ = held_box(0.0, 0.5 * bound, device=device, mu=mu, mass=mass)
    test.assertGreater(anchors_only, 1.0)


class TestContactTorsionPatchGrouping(unittest.TestCase):
    """Follow persistent patch regions on the host without a solver step."""


class TestContactTorsion(unittest.TestCase):
    """Exercise an explicitly assumed uniform-disk effective spin radius."""


class TestContactTorsionWithPatches(unittest.TestCase):
    """Bound one spin row per persistent friction patch region."""


for _fn in (
    test_two_anchor_region_pools_the_undivided_coefficient,
    test_single_anchor_region_keeps_the_coefficient,
    test_point_friction_grouping_is_unchanged,
    test_region_without_anchor_rows_carries_no_spin,
    test_region_with_inadmissible_member_is_skipped,
    test_broken_load_ring_is_rejected,
):
    add_function_test(TestContactTorsionPatchGrouping, _fn.__name__, _fn, devices=get_test_devices())


for _fn in (
    test_default_off_equivalent,
    test_spin_stops_below_bound,
    test_zero_friction_and_zero_load,
    test_joint_response_and_coupled_budget,
    test_witness_count_does_not_multiply_torque,
    test_release_and_reset_carry_no_torque,
    test_timestep_and_capacity,
    test_per_shape_radius_selects_contacts,
    test_invalid_input_and_unsupported_modes,
    test_capture_is_explicitly_rejected,
    test_hydro_combination_is_rejected,
):
    add_function_test(TestContactTorsion, _fn.__name__, _fn, devices=get_cuda_test_devices())


for _fn in (
    test_constructor_accepts_torsion_with_patches,
    test_patch_regions_pool_the_spin_budget,
    test_single_anchor_spin_stops_below_bound,
    test_held_box_yaw_torque_threshold,
):
    add_function_test(TestContactTorsionWithPatches, _fn.__name__, _fn, devices=get_cuda_test_devices())


def test_radius_edits_apply_while_enablement_is_fixed(test, device):
    """Read edited torsion radii every step, but enable torsion only when a radius was positive at construction."""
    _, solver, model, initial, contacts = fixture(0.01, device=device, center_only=True)
    test.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.01))
    set_torsion_radius(model, 0.02)
    solver.step(initial, model.state(), model.control(), contacts, 0.0025)
    test.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.02))
    _, solver, model, initial, contacts = fixture(0.0, device=device, center_only=True)
    set_torsion_radius(model, 0.01)
    solver.step(initial, model.state(), model.control(), contacts, 0.0025)
    test.assertFalse(solver._contact_torsion_enabled)
    count = int(solver._constraint_count.numpy()[0])
    test.assertFalse(np.any(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION))


def test_previous_friction_rows_are_not_current_rows(test, device):
    """Reject stale tangent metadata when a later step admits only a normal."""
    _, solver, model, initial, contacts = fixture(0.01, device=device, center_only=True, contact_friction_gap_max=1.0)
    test.assertEqual(solver._torsion_stats["rows"], 1)
    solver.contact_friction_gap_max = -1.0
    solver.step(initial, model.state(), model.control(), contacts, 0.0025)
    test.assertEqual(solver._torsion_stats["rows"], 0)


def test_both_tangent_rows_must_belong_to_current_normal(test, device):
    """Reject a missing or mismatched second tangent, not just the first."""
    for invalid in ("count", "type", "parent"):
        with test.subTest(invalid=invalid):
            _, solver, _model, initial, contacts = fixture(0.01, device=device, center_only=True)
            row = solver._torsion_stats["groups"][0]["normal_rows"][0]
            if invalid == "count":
                counts = solver._constraint_count.numpy()
                counts[0] = row + 2
                solver._constraint_count.assign(counts)
            else:
                array = solver._row_type if invalid == "type" else solver._row_parent
                values = array.numpy()
                values[0, row + 2] = -1
                array.assign(values)
            test.assertEqual(_contact_groups(solver, initial, contacts), [])


def test_normal_parent_metadata_is_preserved(test, device):
    """Keep CONTACT parents available for pooled friction-patch load rings."""
    result, _solver, *_ = fixture(0.01, device=device, center_only=True)
    normal = result["row_type"][0, : result["count"][0]] == 0
    test.assertGreater(np.count_nonzero(normal), 0)
    np.testing.assert_array_equal(result["row_parent"][0, : result["count"][0]][normal], -1)


def test_torsion_uses_cfm_floor(test, device):
    """Apply the same configured diagonal floor as other dense rows."""
    # The spin row's J Y is 2 / I_zz = 25000; use a floor that float32 can resolve next to it.
    cfm = 8.0
    result, solver, *_ = fixture(0.01, device=device, center_only=True, cfm=cfm)
    count = int(result["count"][0])
    types = result["row_type"][0, :count]
    test.assertEqual(np.count_nonzero(types == PGS_CONSTRAINT_TYPE_TORSION), 1)
    unregularized = np.einsum("rd,rd->r", result["J_world"][0, :count], result["Y_world"][0, :count])
    floor = solver._diag.numpy()[0, :count] - unregularized
    np.testing.assert_allclose(floor[types == PGS_CONSTRAINT_TYPE_TORSION], cfm, atol=1e-2)
    np.testing.assert_allclose(floor[types == 0], cfm, atol=1e-2)


def test_torsion_respects_relaxation(test, device):
    """Apply half the unconstrained spin correction at omega one half."""
    result, solver, *_ = fixture(100.0, device=device, center_only=True, iterations=1, relaxation=0.5)
    test.assertEqual(solver._torsion_stats["rows"], 1)
    spin = np.abs(result["v_out"].reshape(2, 6)[:, 5])
    np.testing.assert_allclose(spin, 0.5, atol=1e-4)


def test_torsion_keeps_joint_velocity_limits(test, device):
    """Let joint velocity limits have the last word over the spin rows of each sweep."""
    limit = 0.1
    passes = ((1, 0), (4, 0), (1, 2))
    preparations = ("host", "device", "graph")
    for (iterations, velocity_iterations), preparation, radius in itertools.product(passes, preparations, (0.0, 1.0)):
        with test.subTest(
            iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
        ):
            _, solver, model, initial, contacts = fixture(
                radius,
                device=device,
                center_only=True,
                spin=0.0,
                iterations=iterations,
                velocity_iterations=velocity_iterations,
                contact_torsion_device=preparation != "host",
            )
            limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
            limits[5] = limit
            model.joint_velocity_limit.assign(limits)
            # Only the second pad spins, so any DOF 5 motion is transferred through the contact.
            velocity = initial.joint_qd.numpy()
            velocity[5], velocity[11] = 0.0, 10.0
            initial.joint_qd.assign(velocity)
            newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
            output = model.state()
            if preparation == "graph":
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(initial, output, model.control(), contacts, 0.0025)
                wp.capture_launch(capture.graph)
                solver.check_constraint_capacity()
            else:
                solver.step(initial, output, model.control(), contacts, 0.0025)
            count = int(solver._constraint_count.numpy()[0])
            spin_rows = np.count_nonzero(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION)
            test.assertEqual(spin_rows, int(radius > 0.0))
            qd = output.joint_qd.numpy()
            test.assertLessEqual(abs(float(qd[5])), limit * (1.0 + 1e-5))
            if radius > 0.0:
                # The spin row still resists the relative spin.
                test.assertLess(float(qd[11]), 9.0)


def test_torsion_keeps_fused_drive_velocity_limits(test, device):
    """Let the fused drive-row velocity clamp have the last word over the spin rows of each sweep."""
    limit = 0.1
    passes = ((1, 0), (4, 0), (1, 2))
    preparations = ("host", "device", "graph")
    limits = np.full(12, np.inf, dtype=np.float32)
    limits[5] = limit
    damping = np.zeros(12, dtype=np.float32)
    damping[5] = 1.0e-3
    for (iterations, velocity_iterations), preparation, radius in itertools.product(passes, preparations, (0.0, 1.0)):
        with test.subTest(
            iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
        ):
            _, solver, model, initial, contacts = fixture(
                radius,
                device=device,
                center_only=True,
                spin=0.0,
                model_arrays={"joint_velocity_limit": limits, "joint_target_kd": damping},
                drive_mode="rows",
                fuse_joint_velocity_limits=True,
                iterations=iterations,
                velocity_iterations=velocity_iterations,
                contact_torsion_device=preparation != "host",
            )
            test.assertTrue(solver.velocity_limits_fused)
            # Only the second pad spins, so any DOF 5 motion is transferred through the contact.
            velocity = initial.joint_qd.numpy()
            velocity[5], velocity[11] = 0.0, 10.0
            initial.joint_qd.assign(velocity)
            newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
            output = model.state()
            if preparation == "graph":
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(initial, output, model.control(), contacts, 0.0025)
                wp.capture_launch(capture.graph)
                solver.check_constraint_capacity()
            else:
                solver.step(initial, output, model.control(), contacts, 0.0025)
            count = int(solver._constraint_count.numpy()[0])
            row_type = solver._row_type.numpy()[0, :count]
            test.assertEqual(np.count_nonzero(row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET), 1)
            test.assertEqual(np.count_nonzero(row_type == PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
            qd = output.joint_qd.numpy()
            test.assertLessEqual(abs(float(qd[5])), limit * (1.0 + 1e-5))
            if radius > 0.0:
                test.assertLess(float(qd[11]), 9.0)


def test_torsion_keeps_free_body_velocity_limits(test, device):
    """Let the free-body velocity-limit rows have the last word over a mixed contact's spin row."""
    limit = 0.1
    passes = ((1, 0), (4, 0), (1, 2))
    preparations = ("host", "device", "graph")
    for (iterations, velocity_iterations), preparation, radius in itertools.product(passes, preparations, (0.0, 1.0)):
        with test.subTest(
            iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
        ):
            # A spinning articulated pad presses on a resting free body whose spin is limited.
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            SolverSteel.register_custom_attributes(builder)
            inertia = wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
            shape = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=0.0)
            pose = wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
            pad = builder.add_link(xform=pose, mass=0.3, inertia=inertia)
            axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
            builder.add_articulation(
                [builder.add_joint_d6(-1, pad, parent_xform=pose, linear_axes=axes, angular_axes=axes)]
            )
            builder.add_shape_box(pad, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
            free = builder.add_body(
                xform=wp.transform(wp.vec3(0, 0, 0.025), wp.quat_identity()),
                mass=0.3,
                inertia=inertia,
                custom_attributes={"steel:rigid_body_max_angular_velocity": limit},
            )
            builder.add_shape_box(free, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
            model = builder.finalize(device=device)
            initial = model.state()
            initial.joint_qd.assign(np.array([0, 0, 0.1, 0, 0, 10.0, 0, 0, -0.1, 0, 0, 0], np.float32))
            newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
            pipeline = newton.CollisionPipeline(
                model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
            )
            contacts = pipeline.contacts()
            pipeline.collide(initial, contacts)
            # One central witness: any free-body spin comes from the spin row.
            poses = initial.body_q.numpy()
            for side in (0, 1):
                name = f"rigid_contact_point{side}"
                body = model.shape_body.numpy()[getattr(contacts, f"rigid_contact_shape{side}").numpy()[0]]
                points = getattr(contacts, name).numpy()
                points[0] = -poses[body, :3]
                getattr(contacts, name).assign(points)
            contacts.rigid_contact_count.assign(np.array([1], np.int32))
            set_torsion_radius(model, radius)
            solver = SolverSteel(
                model,
                solve_mode="matrix_free",
                friction_anchor_beta=0.0,
                iterations=iterations,
                velocity_iterations=velocity_iterations,
                baumgarte=0.05,
                cfm=0,
                contact_regularization=0,
                contact_warmstart=False,
                articulation_constraint_max=64,
                free_body_constraint_max=16,
                contact_torsion_device=preparation != "host",
            )
            solver._rigid_body_angular_damping.zero_()
            output = model.state()
            if preparation == "graph":
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(initial, output, model.control(), contacts, 0.0025)
                wp.capture_launch(capture.graph)
                solver.check_constraint_capacity()
            else:
                solver.step(initial, output, model.control(), contacts, 0.0025)
            count = int(solver._constraint_count.numpy()[0])
            spin_rows = np.count_nonzero(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION)
            test.assertEqual(spin_rows, int(radius > 0.0))
            test.assertGreater(int(solver._mf_constraint_count.numpy()[0]), 0)
            qd = output.joint_qd.numpy()
            test.assertLessEqual(abs(float(qd[11])), limit * (1.0 + 1e-5))
            if radius > 0.0:
                test.assertLess(float(qd[5]), 9.0)


def test_contact_row_loss_fails_without_diagnostics(test, device):
    """Detect rolled-back dropped contacts even with the overflow warning off."""
    baseline, *_ = fixture(0.0, device=device, center_only=True)
    limit = int(baseline["count"][0]) - 1
    with test.assertRaisesRegex(RuntimeError, "[Oo]verflow|[Dd]ropped|capacity"):
        fixture(0.01, device=device, center_only=True, row_limit=limit, warn_constraint_overflow=False)


def test_selected_unsupported_shape_fails_at_construction(test, device):
    """Reject unsupported geometry with a torsion radius instead of ignoring it."""
    _, _solver, model, *_ = fixture(0.0, device=device, center_only=True)
    types = model.shape_type.numpy()
    types[0] = int(GeoType.MESH)
    model.shape_type.assign(types)
    set_torsion_radius(model, 0.01, (0,))
    with test.assertRaisesRegex(ValueError, r"does not support the geometry of shapes \[0\]"):
        SolverSteel(model, solve_mode="matrix_free")


def test_hydro_contact_input_is_rejected(test, device):
    """Reject actual positive hydro contact stiffness, not a synthetic solver flag."""
    for device_preparation in (False, True):
        _, solver, model, initial, contacts = fixture(
            0.01, device=device, center_only=True, contact_torsion_device=device_preparation, step=False
        )
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with test.subTest(device_preparation=device_preparation), test.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


def test_contact_input_overflow_is_rejected(test, device):
    """Reject a contact count beyond the buffer instead of grouping a truncated input."""
    for device_preparation in (False, True):
        _, solver, model, initial, contacts = fixture(
            0.01, device=device, center_only=True, contact_torsion_device=device_preparation, step=False
        )
        contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
        with (
            test.subTest(device_preparation=device_preparation),
            test.assertRaisesRegex(RuntimeError, "Contact input overflow"),
        ):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


def test_patch_spanning_shape_pairs_takes_the_largest_radius(test, device):
    """Give a patch over two shape pairs the larger radius, or no row when one pair has none, on both responses.

    Covers either contact order and every preparation; propagation prepares on the device only.
    """
    radius_cases = {"positive": [0.0, 0.001, 0.04], "zero_member": [0.0, 0.0, 0.04]}
    cases = [("positive", preparation, "immediate") for preparation in ("host", "patch_segments", "world_scan")]
    cases += [("positive", "patch_segments", "propagation"), ("zero_member", "patch_segments", "immediate")]
    cases += [("zero_member", preparation, "propagation") for preparation in ("patch_segments", "world_scan")]
    results = {}
    for (radii, preparation, response), reverse in itertools.product(cases, (False, True)):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        pose = wp.transform(wp.vec3(0.0, 0.0, 0.0199), wp.quat_identity())
        inertia = wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        body = builder.add_link(xform=pose, mass=0.3, inertia=inertia)
        axes = [newton.ModelBuilder.JointDofConfig(axis=axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        builder.add_articulation(
            [builder.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)]
        )
        for x in (-0.02, 0.02):
            builder.add_shape_box(
                body,
                hx=0.02,
                hy=0.015,
                hz=0.02,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5),
            )
        model = builder.finalize(device=device)
        set_torsion_radius(model, np.array(radius_cases[radii], dtype=np.float32))
        solver = SolverSteel(
            model,
            articulated_contact_response=response,
            contact_torsion_device=preparation != "host",
            iterations=128,
            friction_anchor_beta=0.2,
            cfm=0.0,
            articulation_constraint_max=64,
            free_body_constraint_max=16,
        )
        solver._rigid_body_angular_damping.zero_()
        if preparation == "world_scan":
            solver._device_torsion = DeviceTorsionPreparation(solver, patch_segments=False)
            solver._device_torsion.allocate_snapshots(model)
        state, output = model.state(), model.state()
        state.joint_qd.assign(np.array([0.0, 0.0, -0.1, 0.0, 0.0, 100.0], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=64)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        # Put every witness at the pieces' shared boundary so only the spin radius differs between orders.
        count = int(contacts.rigid_contact_count.numpy()[0])
        point0, point1 = contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()
        point0[:count] = (0.0, 0.0, 0.0)
        point1[:count] = (0.0, 0.0, -0.02)
        contacts.rigid_contact_point0.assign(point0)
        contacts.rigid_contact_point1.assign(point1)
        if reverse:
            shape1 = contacts.rigid_contact_shape1.numpy()
            shape1[:count] = shape1[:count][::-1]
            contacts.rigid_contact_shape1.assign(shape1)
        test.assertEqual(sorted(set(contacts.rigid_contact_shape1.numpy()[:count].tolist())), [1, 2])
        with test.subTest(radii=radii, preparation=preparation, response=response, reverse=reverse):
            solver.step(state, output, model.control(), contacts, 0.0025)
            solver.check_constraint_capacity()
            test.assertTrue(np.all(solver._contact_path.numpy()[:count] == (2 if response == "propagation" else 0)))
            stats = solver._torsion_stats if preparation == "host" else solver._device_torsion.read_stats()
            if radii == "positive":
                test.assertEqual(stats["rows"], 1)
                test.assertAlmostEqual(stats["groups"][0]["effective_radius_m"], 0.04, places=6)
            else:
                # The zero-radius pair rejects the whole region.
                test.assertEqual(stats["rows"], 0)
            if response == "propagation":
                test.assertEqual(len(_propagation_torsion_rows(solver)), stats["rows"])
        results[radii, preparation, response, reverse] = output.joint_qd.numpy()
    for (radii, preparation, response, reverse), joint_qd in results.items():
        with test.subTest(radii=radii, preparation=preparation, response=response, reverse=reverse):
            reference = results[radii, "host" if radii == "positive" else "patch_segments", "immediate", False]
            np.testing.assert_allclose(joint_qd, reference, rtol=0.0, atol=1.0e-4)


class TestContactTorsionRegressions(unittest.TestCase):
    """Reject invalid storage/lifecycle paths rather than silently welding spin."""


for _fn in (
    test_radius_edits_apply_while_enablement_is_fixed,
    test_previous_friction_rows_are_not_current_rows,
    test_both_tangent_rows_must_belong_to_current_normal,
    test_normal_parent_metadata_is_preserved,
    test_torsion_uses_cfm_floor,
    test_torsion_respects_relaxation,
    test_torsion_keeps_joint_velocity_limits,
    test_torsion_keeps_fused_drive_velocity_limits,
    test_torsion_keeps_free_body_velocity_limits,
    test_contact_row_loss_fails_without_diagnostics,
    test_selected_unsupported_shape_fails_at_construction,
    test_hydro_contact_input_is_rejected,
    test_contact_input_overflow_is_rejected,
    test_patch_spanning_shape_pairs_takes_the_largest_radius,
):
    add_function_test(TestContactTorsionRegressions, _fn.__name__, _fn, devices=get_cuda_test_devices())


def _check_pair(test, device, regularization, velocity, **options):
    """Check split position integration and the accumulated velocity correction."""
    common = dict(
        iterations=64,
        contact_regularization=regularization,
        **options,
    )
    baseline, *_ = fixture(0.01, device=device, velocity_iterations=0, **common)
    result, solver, model, state, contacts = fixture(0.01, device=device, velocity_iterations=velocity, **common)
    np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
    count = int(result["count"][0])
    delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
    np.testing.assert_allclose(
        result["v_out"],
        baseline["v_out"].astype(np.float64) + result["Y_world"][0, :count].astype(np.float64).T @ delta,
        atol=2e-5,
        rtol=8 * np.finfo(np.float32).eps,
    )
    weights = solver._row_w.numpy()[0, :count]
    types = result["row_type"][0, :count]
    np.testing.assert_allclose(weights[types == 0], 1 / (1 + regularization), atol=1e-7)
    _assert_budget(test, result, solver)
    test.assertTrue(np.isfinite(result["body_qd"]).all())
    return baseline, result, solver, model, state, contacts


def test_analytic_normal_load_position_soft_velocity_rigid(test, device):
    """Undo normal softness only in velocity cleanup and budget spin from final load."""
    for regularization in (0.01, 0.02, 0.5):
        for velocity in (1, 4, 16):
            with test.subTest(reg=regularization, velocity=velocity):
                baseline, result, solver, *_ = _check_pair(
                    test, device, regularization, velocity, center_only=True, spin=100.0
                )
                group = solver._torsion_stats["groups"][0]
                normals = group["normal_rows"]
                test.assertAlmostEqual(
                    float(baseline["impulses"][0, normals].sum()), 0.03 / (1 + regularization), delta=2e-7
                )
                normal = float(result["impulses"][0, normals].sum())
                test.assertAlmostEqual(normal, 0.03, delta=2e-7)
                test.assertAlmostEqual(abs(float(result["impulses"][0, group["row"]])), 0.01 * 0.5 * normal, delta=2e-8)
                test.assertLess(abs(float(result["body_qd"][0, 5])), 100.0)


def test_distributed_budget_and_export(test, device):
    """Keep point and patch budgets coherent across internal timesteps."""
    for patch_options in ({}, PATCH_OPTIONS):
        for dt in (0.00125, 0.0025, 0.005):
            with test.subTest(patch=bool(patch_options), dt=dt):
                _, result, solver, _, _, contacts = _check_pair(
                    test, device, 0.01, 16, spin=10.0, sliding=1.0, dt=dt, **patch_options
                )
                count = int(contacts.rigid_contact_count.numpy()[0])
                forces = result["contact_f"][:count, :3]
                normals = contacts.rigid_contact_normal.numpy()[:count]
                for index, slot in enumerate(solver._contact_slot.numpy()[:count]):
                    if slot >= 0:
                        test.assertAlmostEqual(
                            abs(float(forces[index] @ normals[index])) * dt,
                            float(result["impulses"][0, slot]),
                            delta=2e-7,
                        )


def test_no_load_near_positive_gap_and_release(test, device):
    """Refund sliding and spin even for the narrow positive initial-gap tolerance."""
    for gap in (0.5e-6, 0.01):
        _, _, solver, *_ = _check_pair(test, device, 0.01, 16, center_only=True, spin=100.0, **PATCH_OPTIONS)
        group = solver._torsion_stats["groups"][0]
        n = int(solver._constraint_count.numpy()[0])
        before = solver._impulses.numpy()[0, :n].copy()
        phi = solver._phi.numpy()
        phi[0, group["normal_rows"]] = gap
        solver._phi.assign(phi)
        # Force a separating end-gap through the normal row's response vector.
        row = group["normal_rows"][0]
        separating = solver._Y_world.numpy()[0, row].copy()
        solver._v_out_snap.assign(separating)
        solver._v_hat.assign(separating)
        solver._v_out.assign(separating + solver._Y_world.numpy()[0, :n].T @ before)
        velocity_before = solver._v_out.numpy().copy()
        _conclude_position_problem(solver, 0.0025)
        _run_velocity_post_solve(solver)
        after = solver._impulses.numpy()[0, :n]
        test.assertLess(np.abs(after[group["normal_rows"]]).max(initial=0), 2e-7)
        test.assertLess(abs(float(after[group["row"]])), 2e-7)
        for anchor in group["anchor_rows"]:
            test.assertLess(np.linalg.norm(after[anchor + 1 : anchor + 3]), 2e-7)
        np.testing.assert_allclose(
            solver._v_out.numpy(), velocity_before + solver._Y_world.numpy()[0, :n].T @ (after - before), atol=2e-5
        )


class TestTorsionCombined(unittest.TestCase):
    pass


for _fn in (
    test_analytic_normal_load_position_soft_velocity_rigid,
    test_distributed_budget_and_export,
    test_no_load_near_positive_gap_and_release,
):
    add_function_test(TestTorsionCombined, _fn.__name__, _fn, devices=get_cuda_test_devices())


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _rounding_samples(
    poses: wp.array[wp.transform],
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    transformed: wp.array[wp.vec3],
    dotted: wp.array[float],
):
    i = wp.tid()
    transformed[i] = _host_transform_point(poses[i], points[i])
    dotted[i] = _host_dot3(points[i], directions[i])


def synthetic_fixture(*, worlds=3, witnesses=5, patches=True, seed=4, device="cpu", padding=7):
    """Create ordered/reordered contact worlds with unequal articulation DOFs."""
    rng = np.random.default_rng(seed)
    n = worlds * witnesses
    capacity = n + padding
    dense = witnesses * 3 + 8
    records = [(world, local) for world in range(worlds) for local in range(witnesses)]
    rng.shuffle(records)

    def arr(value, dtype=None):
        return wp.array(np.asarray(value), dtype=dtype, device=device)

    def zero(shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device=device)

    def full(shape, value, dtype=int):
        return wp.full(shape, value, dtype=dtype, device=device)

    count = np.full(worlds, witnesses * 3, np.int32)
    row_types = np.zeros((worlds, dense), np.int32)
    parents = np.full((worlds, dense), -1, np.int32)
    mu = np.zeros((worlds, dense), np.float32)
    world_ids = np.zeros(capacity, np.int32)
    slots = np.full(capacity, -1, np.int32)
    anchors = np.zeros(capacity, np.int32)
    sa = np.zeros(capacity, np.int32)
    sb = np.zeros(capacity, np.int32)
    owner = np.full(capacity, -1, np.int32)
    normals = np.zeros((capacity, 3), np.float32)
    points = np.zeros((capacity, 3), np.float32)
    region_seeds = {}
    for c, (world, local) in enumerate(records):
        slot = local * 3
        region = local % 2
        region_seeds.setdefault((world, region), c)
        world_ids[c], slots[c], anchors[c] = world, slot, 3
        sa[c], sb[c] = world * 4 + region, world * 4 + 2 + region
        owner[c] = region_seeds[world, region]
        normals[c] = (0.0, 0.0, -1.0)
        points[c] = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0)
        row_types[world, slot : slot + 3] = (
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        )
        parents[world, slot : slot + 3] = (slot, slot, slot)
        mu[world, slot + 1 : slot + 3] = 0.7
    if patches:
        for world in range(worlds):
            for region in range(2):
                members = [c for c, (w, local) in enumerate(records) if w == world and local % 2 == region]
                for i, c in enumerate(members):
                    parents[world, slots[c]] = slots[members[(i + 1) % len(members)]]
                    mu[world, slots[c] + 1 : slots[c] + 3] /= len(members)
    shape_body = np.repeat(np.arange(worlds * 2, dtype=np.int32), 2)
    response_sizes = np.tile([6, 3], worlds).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(response_sizes))).astype(np.int32)
    total_dofs = int(starts[-1])
    motion = rng.normal(size=(total_dofs, 6)).astype(np.float32)
    body_velocity = rng.normal(size=(worlds * 2, 6)).astype(np.float32)
    prescribed = np.zeros(worlds * 2, np.int32)
    if worlds > 1:
        prescribed[-1] = 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        shape_count=worlds * 4,
        shape_body=arr(shape_body, int),
        shape_type=full(worlds * 4, int(GeoType.BOX)),
        joint_ancestor=full(worlds * 2, -1),
        joint_qd_start=arr(starts, int),
    )
    state = SimpleNamespace(
        body_q=arr(np.tile([0, 0, 0, 0, 0, 0, 1], (worlds * 2, 1)).astype(np.float32), wp.transform)
    )
    augmented = SimpleNamespace(
        _joint_S_s=arr(motion, wp.spatial_vector), _body_v_s=arr(body_velocity, wp.spatial_vector)
    )
    contacts = SimpleNamespace(
        rigid_contact_count=arr([n], int),
        rigid_contact_max=capacity,
        rigid_contact_stiffness=None,
        rigid_contact_shape0=arr(sa, int),
        rigid_contact_shape1=arr(sb, int),
        rigid_contact_normal=arr(normals, wp.vec3),
        rigid_contact_point0=arr(points, wp.vec3),
        rigid_contact_point1=arr(points, wp.vec3),
        rigid_contact_margin0=zero(capacity),
        rigid_contact_margin1=zero(capacity),
    )
    solver = SimpleNamespace(
        model=model,
        _world_count=worlds,
        _articulation_constraint_max=dense,
        _max_contacts_alloc=capacity,
        _contact_torsion_shape_radius=full(worlds * 4, 0.01, float),
        _friction_anchors_enabled=patches,
        _friction_patches=SimpleNamespace(current=SimpleNamespace(owner=arr(owner, int))),
        _contact_path=zero(capacity, int),
        _contact_slot=arr(slots, int),
        _contact_world=arr(world_ids, int),
        _contact_slots_needed=arr(anchors, int),
        _constraint_count=arr(count, int),
        _slot_counter=arr(count, int),
        _row_type=arr(row_types, int),
        _row_parent=arr(parents, int),
        _row_mu=arr(mu, float),
        _row_dropped_dense=zero(worlds, int),
        _contact_torsion_group=full((worlds, dense), -1),
        _torsion_row_radius=zero((worlds, dense)),
        _body_to_joint=arr(np.arange(worlds * 2, dtype=np.int32), int),
        _body_to_articulation=arr(np.arange(worlds * 2, dtype=np.int32), int),
        _articulation_dof_start=arr(starts[:-1], int),
        _articulation_response_dof_count=arr(response_sizes, int),
        _art_group_idx=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        _art_to_world=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        _prescribed_articulation=arr(prescribed, int),
        _group_to_art={size: arr(np.flatnonzero(response_sizes == size).astype(np.int32), int) for size in (3, 6)},
        _J_by_size={size: zero((worlds, dense, size)) for size in (3, 6)},
        _propagation_torsion_rows=False,
    )
    for name in ("_phi", "_target_velocity", "_row_restitution"):
        setattr(solver, name, zero((worlds, dense)))
    return solver, state, augmented, contacts


def run_oracle(solver, state, augmented, contacts):
    """Invoke only the host preparation oracle, not a CPU solver backend."""
    with patch.object(host, "validate_torsion_step", return_value=None):
        host.prepare_torsion_rows(solver, state, augmented, contacts)


# --- TestDeviceTorsionPreparation ---
"""Exercise portable preparation kernels independently of a full CUDA solve."""


def test_public_device_option_is_default_off(test, device):
    """Keep public device preparation opt-in."""
    test.assertIs(signature(solver_steel.SolverSteel).parameters["contact_torsion_device"].default, False)


def compare(test, *, device, modify=None, **options):
    """Compare every row field, group membership and size-specific Jacobian."""
    expected, state, augmented, contacts = synthetic_fixture(device=device, **options)
    actual, state2, augmented2, contacts2 = synthetic_fixture(device=device, **options)
    if modify is not None:
        modify(expected, contacts)
        modify(actual, contacts2)
    run_oracle(expected, state, augmented, contacts)
    preparer = DeviceTorsionPreparation(actual)
    preparer.prepare(state2, augmented2, contacts2)
    for name in (
        "_constraint_count",
        "_slot_counter",
        "_row_type",
        "_row_parent",
        "_row_mu",
        "_phi",
        "_target_velocity",
        "_row_restitution",
        "_contact_torsion_group",
    ):
        np.testing.assert_allclose(
            getattr(actual, name).numpy(), getattr(expected, name).numpy(), rtol=2e-6, atol=2e-6, err_msg=name
        )
    for size in expected._J_by_size:
        np.testing.assert_allclose(
            actual._J_by_size[size].numpy(),
            expected._J_by_size[size].numpy(),
            rtol=2e-6,
            atol=2e-6,
            err_msg=f"J{size}",
        )
    stats = preparer.read_stats()
    test.assertEqual(stats["rows"], expected._torsion_stats["rows"])
    actual_groups = sorted(stats["groups"], key=lambda group: (group["world"], group["row"]))
    expected_groups = sorted(expected._torsion_stats["groups"], key=lambda group: (group["world"], group["row"]))
    for actual_group, expected_group in zip(actual_groups, expected_groups, strict=True):
        for name in expected_group:
            np.testing.assert_allclose(actual_group[name], expected_group[name], rtol=2e-6, atol=2e-6)
    return actual, preparer


def test_randomized_patch_and_point_oracle(test, device):
    """Preserve contact-order semantics across mixed shapes and unequal DOFs."""
    for patches in (False, True):
        for seed in range(5):
            with test.subTest(patches=patches, seed=seed):
                compare(test, device=device, patches=patches, seed=seed)


def test_host_math_rounding(test, device):
    """Match explicit dot and quaternion-transform rounding on CPU and CUDA."""
    rng = np.random.default_rng(84)
    poses = rng.normal(size=(1024, 7)).astype(np.float32)
    poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
    points = rng.normal(size=(1024, 3)).astype(np.float32)
    directions = rng.normal(size=points.shape).astype(np.float32)
    expected_points = np.array([host._transform_point(q, p) for q, p in zip(poses, points, strict=True)])
    # Independent scalar reference: float32 products, sequential float64
    # sum, one float32 output rounding. Do not consult platform BLAS.
    expected_dot = np.array(
        [sum(float(np.float32(a[i] * b[i])) for i in range(3)) for a, b in zip(points, directions, strict=True)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        np.array([host._dot3(a, b) for a, b in zip(points, directions, strict=True)]), expected_dot
    )
    transformed = wp.empty(1024, dtype=wp.vec3, device=device)
    dotted = wp.empty(1024, dtype=float, device=device)
    wp.launch(
        _rounding_samples,
        dim=1024,
        inputs=[
            wp.array(poses, dtype=wp.transform, device=device),
            wp.array(points, dtype=wp.vec3, device=device),
            wp.array(directions, dtype=wp.vec3, device=device),
            transformed,
            dotted,
        ],
        device=device,
    )
    np.testing.assert_array_equal(transformed.numpy(), expected_points)
    np.testing.assert_array_equal(dotted.numpy(), expected_dot)


def test_grouping_and_jacobians_do_not_depend_on_blas(test, device):
    """Build host groups and rows without platform-dependent NumPy dot."""
    for patches in (False, True):
        with test.subTest(patches=patches):
            with patch.object(host.np, "dot", side_effect=AssertionError("BLAS dot must not decide torsion rows")):
                compare(test, device=device, patches=patches)


def test_empty_and_larger_preallocated_buffer(test, device):
    """Keep zero contacts valid and distinguish configured from input capacity."""
    solver, state, augmented, contacts = synthetic_fixture(device=device)
    solver._max_contacts_alloc += 5
    preparer = DeviceTorsionPreparation(solver)
    preparer.prepare(state, augmented, contacts)
    test.assertEqual(int(np.count_nonzero(preparer.work.spin_row.numpy() >= 0)), 6)
    contacts.rigid_contact_count.zero_()
    preparer.prepare(state, augmented, contacts)
    test.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))


def test_ring_and_capacity_errors_are_latched(test, device):
    """Reject incomplete rings and capacity failures before row writes."""
    for failure in ("ring", "dense", "input", "spin"):
        solver, state, augmented, contacts = synthetic_fixture(device=device)
        before = solver._row_type.numpy().copy()
        if failure == "ring":
            parents = solver._row_parent.numpy()
            parents[0, 0] = solver._articulation_constraint_max - 1
            solver._row_parent.assign(parents)
        elif failure == "dense":
            solver._row_dropped_dense.fill_(1)
        elif failure == "input":
            contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
        else:
            solver._constraint_count.fill_(solver._articulation_constraint_max)
        preparer = DeviceTorsionPreparation(solver)
        with test.subTest(failure=failure), test.assertRaises(RuntimeError):
            preparer.prepare(state, augmented, contacts)
        np.testing.assert_array_equal(before, solver._row_type.numpy())
        test.assertNotEqual(int(preparer.work.status.numpy()[0]), 0)
        with test.assertRaises(RuntimeError):
            preparer.prepare(state, augmented, None)


def test_patch_segments_match_world_scan(test, device):
    """Match the per-world patch scan exactly under selection, rejection, negative owners and unequal radii."""
    for seed in range(8):
        results = []
        for patch_segments in (False, True):
            solver, state, augmented, contacts = synthetic_fixture(worlds=4, witnesses=12, seed=seed, device=device)
            rng = np.random.default_rng(seed)
            positive = rng.choice((0.01, 0.03), solver.model.shape_count)
            radii = np.where(rng.random(solver.model.shape_count) < 0.6, positive, 0.0).astype(np.float32)
            solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=device)
            anchors = solver._contact_slots_needed.numpy()
            anchors[rng.random(anchors.size) < 0.3] = 1
            solver._contact_slots_needed.assign(anchors)
            points = contacts.rigid_contact_point0.numpy()
            points[rng.random(len(points)) < 0.3, 2] = 0.002
            contacts.rigid_contact_point0.assign(points)
            if seed % 2:
                owner = solver._friction_patches.current.owner.numpy()
                owner[owner == owner.max()] = -1
                solver._friction_patches.current.owner.assign(owner)
            preparer = DeviceTorsionPreparation(solver, patch_segments=patch_segments)
            try:
                preparer.prepare(state, augmented, contacts)
                error = None
            except RuntimeError as exc:
                error = str(exc)
            names = ("_constraint_count", "_row_type", "_row_parent", "_row_mu", "_target_velocity")
            arrays = [
                getattr(solver, name).numpy() for name in (*names, "_contact_torsion_group", "_torsion_row_radius")
            ]
            arrays += [preparer.work.status.numpy(), preparer.work.spin_row.numpy()]
            results.append((error, arrays))
        with test.subTest(device=device, seed=seed):
            test.assertEqual(results[0][0], results[1][0])
            for legacy, segmented in zip(results[0][1], results[1][1], strict=True):
                np.testing.assert_array_equal(segmented, legacy)


def test_patch_eligibility_and_point_cluster_edges(test, device):
    """Match mixed-anchor patches, selector rejection and nontransitive clusters."""
    for scenario in ("nonanchor", "positive_gap", "selector", "nontransitive"):

        def modify(solver, contacts, scenario=scenario):
            if scenario == "nonanchor":
                anchors = solver._contact_slots_needed.numpy()
                anchors[::2] = 1
                solver._contact_slots_needed.assign(anchors)
            elif scenario == "positive_gap":
                points = contacts.rigid_contact_point0.numpy()
                points[:, 2] = 0.002
                contacts.rigid_contact_point0.assign(points)
            elif scenario == "selector":
                radii = np.zeros(solver.model.shape_count, dtype=np.float32)
                radii[0] = 0.01
                solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=device)
            else:
                # Adjacent normals match, but the endpoints do not. The
                # greedy oracle tests every member, not only its seed.
                angles = (np.arange(contacts.rigid_contact_max) % 3) * 0.035
                normals = np.column_stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)))
                contacts.rigid_contact_normal.assign(normals.astype(np.float32))
                contacts.rigid_contact_point0.zero_()
                contacts.rigid_contact_point1.zero_()

        for patches in (False, True):
            with test.subTest(scenario=scenario, patches=patches):
                compare(test, device=device, patches=patches, modify=modify)


def test_touching_threshold_preserves_host_rounding(test, device):
    """Keep admission identical when cancelling dot products straddle the gap gate."""

    def modify(_solver, contacts):
        rng = np.random.default_rng(18)
        shape = (contacts.rigid_contact_max, 3)
        normal = rng.normal(size=shape).astype(np.float32)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        point = rng.normal(size=shape).astype(np.float32) * 0.1
        point -= np.sum(normal * point, axis=1, keepdims=True) * normal
        point += np.float32(1e-5) * normal
        contacts.rigid_contact_normal.assign(-normal)
        contacts.rigid_contact_point0.assign(point)
        contacts.rigid_contact_point1.zero_()

    for patches in (False, True):
        with test.subTest(device=device, patches=patches):
            compare(test, device=device, worlds=512, witnesses=1, patches=patches, modify=modify)


def test_rejected_region_and_normal_clusters(test, device):
    """Preserve whole-region rejection and all-member coplanar clustering."""
    for patches in (False, True):
        pairs = [synthetic_fixture(device=device, patches=patches) for _ in range(2)]
        for solver, _state, _augmented, contacts in pairs:
            types = solver.model.shape_type.numpy()
            types[0] = int(GeoType.MESH)
            solver.model.shape_type.assign(types)
            normals = contacts.rigid_contact_normal.numpy()
            normals[3] = (1, 0, 0)
            contacts.rigid_contact_normal.assign(normals)
            points = contacts.rigid_contact_point0.numpy()
            points[6, 2] = 0.001
            contacts.rigid_contact_point0.assign(points)
        expected, state, augmented, contacts = pairs[0]
        actual, state2, augmented2, contacts2 = pairs[1]
        run_oracle(expected, state, augmented, contacts)
        preparer = DeviceTorsionPreparation(actual)
        preparer.prepare(state2, augmented2, contacts2)
        np.testing.assert_array_equal(actual._contact_torsion_group.numpy(), expected._contact_torsion_group.numpy())
        np.testing.assert_array_equal(actual._row_parent.numpy(), expected._row_parent.numpy())


def test_more_than_host_global_capacity(test, device):
    """Retain over 4096 witnesses against a decomposed host grouping oracle."""
    solver, state, augmented, contacts = synthetic_fixture(worlds=822, device=device, witnesses=5)
    with test.assertRaisesRegex(RuntimeError, "host grouping capacity"):
        host._contact_groups(solver, state, contacts)
    worlds = solver._contact_world.numpy()
    expected = set()
    for first, last in ((0, 411), (411, 822)):
        solver._contact_path.assign(np.where((worlds >= first) & (worlds < last), 0, -1).astype(np.int32))
        for group in host._contact_groups(solver, state, contacts):
            expected.add((group[0].world, tuple(sorted(w.slot for w in group))))
    solver._contact_path.zero_()
    preparer = DeviceTorsionPreparation(solver)
    preparer.prepare(state, augmented, contacts)
    membership = solver._contact_torsion_group.numpy()
    actual = set()
    for world in range(solver._world_count):
        for spin in np.unique(membership[world]):
            if spin >= 0:
                actual.add((world, tuple(np.flatnonzero(membership[world] == spin))))
    test.assertEqual(expected, actual)
    test.assertEqual(len(actual), 1644)


def test_deferred_failure_is_not_partial_success(test, device):
    """Latch failure, suppress row solves and restore public dynamic state."""
    solver, state, augmented, contacts = synthetic_fixture(device=device)
    out = SimpleNamespace(body_q=wp.clone(state.body_q))
    preparer = DeviceTorsionPreparation(solver, deferred_errors=True)
    preparer.begin_step(state, out)
    out.body_q.zero_()
    solver._row_dropped_dense.fill_(1)
    preparer.prepare(state, augmented, contacts)
    preparer.end_step(out)
    np.testing.assert_array_equal(out.body_q.numpy(), state.body_q.numpy())
    test.assertTrue(np.all(solver._constraint_count.numpy() == 0))
    test.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))
    with test.assertRaisesRegex(RuntimeError, "Dense contact row overflow"):
        preparer.validate()
    solver._row_dropped_dense.zero_()
    preparer.prepare(state, augmented, contacts)
    with test.assertRaises(RuntimeError):
        preparer.validate()


def test_deferred_preparation_has_no_host_array_transfers(test, device):
    """Reject accidental full or scalar host transfers in deferred preparation."""
    solver, state, augmented, contacts = synthetic_fixture(device=device)
    preparer = DeviceTorsionPreparation(solver, deferred_errors=True)
    with (
        patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected device-to-host transfer")),
        patch.object(wp.array, "assign", side_effect=AssertionError("Unexpected host assignment")),
    ):
        preparer.prepare(state, augmented, contacts)
    preparer.validate()


def test_cuda_cancelling_angular_projection_matches_host_rounding(test, device):
    """Preserve float32 host rounding for nearly cancelling angular rows."""
    pairs = [synthetic_fixture(device=device) for _ in range(2)]
    normal = np.array([0.8962938785552979, 0.44297733902931213, 0.020699456334114075], dtype=np.float32)
    angular = np.array(
        [
            [0.16465608775615692, -0.29055461287498474, -0.9425849914550781],
            [-0.411504328250885, 0.8481746912002563, -0.33356380462646484],
            [0.41150417923927307, -0.8481748104095459, 0.3335646986961365],
            [0.4004249572753906, 0.6479430198669434, -0.647942841053009],
        ],
        dtype=np.float32,
    )
    for _solver, _state, augmented, contacts in pairs:
        contacts.rigid_contact_normal.assign(np.tile(-normal, (contacts.rigid_contact_max, 1)))
        motion = augmented._joint_S_s.numpy()
        motion[:4, 3:] = angular
        augmented._joint_S_s.assign(motion)
    expected, state, augmented, contacts = pairs[0]
    actual, state2, augmented2, contacts2 = pairs[1]
    run_oracle(expected, state, augmented, contacts)
    DeviceTorsionPreparation(actual).prepare(state2, augmented2, contacts2)
    np.testing.assert_array_equal(actual._J_by_size[6].numpy()[0, :, :4], expected._J_by_size[6].numpy()[0, :, :4])


def test_cuda_loaded_closed_linkage(test, device):
    """Preserve loaded loop-closure behavior with active spin rows in both profiles."""
    for velocity, regularization in ((16, 0.0), (0, 0.01)):
        results = []
        for device_preparation in (False, True):
            builder = _build_four_bar()
            root_half = float(np.sqrt(0.5))
            builder.add_shape_plane(plane=(0.0, root_half, root_half, -0.362 * root_half))
            model = builder.finalize(device=device)
            set_torsion_radius(model, 0.01)
            solver = newton.solvers.SolverSteel(
                model,
                solve_mode="matrix_free",
                iterations=64,
                velocity_iterations=velocity,
                contact_regularization=regularization,
                articulation_constraint_max=128,
                free_body_constraint_max=64,
                enable_bilateral_preelimination=False,
                **PATCH_OPTIONS,
            )
            test.assertGreater(solver._connect_count, 0)
            if device_preparation:
                enable_device_torsion(solver)
            a, z = model.state(), model.state()
            newton.eval_fk(model, a.joint_q, a.joint_qd, a)
            control = model.control()
            target = model.joint_target_q.numpy()
            target[0] = 0.03
            control.joint_target_q.assign(target)
            pipeline = newton.CollisionPipeline(
                model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
            )
            contacts = pipeline.contacts()
            spins = 0
            load = 0.0
            for _ in range(32):
                a.clear_forces()
                pipeline.collide(a, contacts)
                solver.step(a, z, control, contacts, 0.0025)
                types = solver._row_type.numpy()
                spins = max(spins, int(np.count_nonzero(types == PGS_CONSTRAINT_TYPE_TORSION)))
                load = max(load, float(solver._impulses.numpy()[types == PGS_CONSTRAINT_TYPE_CONTACT].max(initial=0)))
                a, z = z, a
            test.assertGreater(spins, 0)
            test.assertGreater(load, 1e-8)
            test.assertTrue(np.isfinite(a.body_q.numpy()).all())
            results.append((a.body_q.numpy(), a.joint_q.numpy(), a.joint_qd.numpy(), _loop_anchor_gap(model, a)))
        for expected, actual in zip(*results, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


def test_cuda_solver_profiles(test, device):
    """Match the host's solved state and impulses for both production profiles."""

    def device_prepare(solver, state, augmented, contacts):
        enable_device_torsion(solver).prepare(state, augmented, contacts)

    for patches in (False, True):
        for velocity, regularization in ((16, 0.0), (0, 0.01)):
            options = {
                "iterations": 16,
                "velocity_iterations": velocity,
                "contact_regularization": regularization,
                "sliding": 0.03,
                "spin": 0.4,
            }
            if patches:
                options.update(PATCH_OPTIONS)
            expected, *_ = fixture(0.01, device=device, **options)
            with patch.object(solver_steel, "prepare_torsion_rows", device_prepare):
                actual, *_ = fixture(0.01, device=device, **options)
            for key in expected:
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=3e-5, atol=3e-6, err_msg=f"{patches=} {velocity=} {key=}"
                )


def test_cuda_preparation_graph_changes_and_failure(test, device):
    """Replay changing groups and fail-stop invalid rings without host reads."""
    solver, state, augmented, contacts = synthetic_fixture(device=device)
    preparer = DeviceTorsionPreparation(solver, deferred_errors=True)
    initial_count = wp.clone(solver._constraint_count)
    initial_type = wp.clone(solver._row_type)
    initial_parent = wp.clone(solver._row_parent)
    output = SimpleNamespace(body_q=wp.clone(state.body_q))
    preparer.begin_step(state, output)

    def operation():
        wp.copy(solver._constraint_count, initial_count)
        wp.copy(solver._row_type, initial_type)
        wp.copy(solver._row_parent, initial_parent)
        preparer.begin_step(state, output)
        output.body_q.zero_()
        preparer.prepare(state, augmented, contacts)
        preparer.end_step(output)

    operation()
    with wp.ScopedCapture(device=device) as capture:
        operation()
    for count in (0, 15, 0, 15):
        contacts.rigid_contact_count.fill_(count)
        wp.capture_launch(capture.graph)
        preparer.validate()
        test.assertEqual(np.count_nonzero(preparer.work.spin_row.numpy() >= 0), 0 if count == 0 else 6)
    bad = initial_parent.numpy()
    bad[0, 0] = solver._articulation_constraint_max - 1
    initial_parent.assign(bad)
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, "load ring"):
        preparer.validate()
    np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
    test.assertTrue(np.all(solver._constraint_count.numpy() == 0))


def test_cuda_captured_capacity_errors(test, device):
    """Report an input, dense or spin overflow from graph replay instead of turning it into success."""
    for failure in ("input", "dense", "spin"):
        solver, state, augmented, contacts = synthetic_fixture(device=device)
        preparer = DeviceTorsionPreparation(solver, deferred_errors=True)
        output = SimpleNamespace(body_q=wp.clone(state.body_q))
        initial_count = wp.clone(solver._constraint_count)
        preparer.begin_step(state, output)

        def operation(
            solver=solver,
            state=state,
            augmented=augmented,
            contacts=contacts,
            preparer=preparer,
            output=output,
            initial_count=initial_count,
        ):
            wp.copy(solver._constraint_count, initial_count)
            preparer.begin_step(state, output)
            output.body_q.zero_()
            preparer.prepare(state, augmented, contacts)
            preparer.end_step(output)

        operation()
        with wp.ScopedCapture(device=device) as capture:
            operation()
        if failure == "input":
            contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
        elif failure == "dense":
            solver._row_dropped_dense.fill_(1)
        else:
            initial_count.fill_(solver._articulation_constraint_max)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            with test.subTest(failure=failure), test.assertRaises(RuntimeError):
                preparer.validate()
            np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
            test.assertTrue(np.all(solver._constraint_count.numpy() == 0))
            test.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))


def test_public_api_noop_and_fail_stop(test, device):
    """Allocate no torsion state without a radius, refuse to capture host torsion, and raise latched errors eagerly."""
    _, plain, model, state, _ = fixture(0.0, device=device, contact_torsion_device=True, iterations=16)
    test.assertIsNone(plain._device_torsion)
    plain.check_constraint_capacity()

    _, reference, model, state, contacts = fixture(0.01, device=device, iterations=16)
    with test.assertRaisesRegex(RuntimeError, "requires contact_torsion_device"):
        with wp.ScopedCapture(device=model.device):
            reference.step(state, model.state(), model.control(), contacts, 0.0025)

    _, solver, model, state, contacts = fixture(0.01, device=device, contact_torsion_device=True, iterations=16)
    output = model.state()
    solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
    for _ in range(2):
        with test.assertRaisesRegex(RuntimeError, "load ring"):
            solver.step(state, output, model.control(), contacts, 0.0025)
        with test.assertRaisesRegex(RuntimeError, "contact torsion rejected its input: .*load ring"):
            solver.check_constraint_capacity()
    solver.reset(state)
    solver.check_constraint_capacity()
    solver.step(state, output, model.control(), contacts, 0.0025)
    test.assertFalse(np.array_equal(output.joint_q.numpy(), state.joint_q.numpy()))


def test_cuda_cold_capture_reports_errors_until_reset(test, device):
    """Capture the first step of a solver that never stepped, and report a replay's error until reset."""
    _, solver, model, state, contacts = fixture(
        0.01, device=device, contact_torsion_device=True, iterations=16, step=False
    )
    test.assertEqual(solver._step, 0)
    output = model.state()
    with wp.ScopedCapture(device=model.device) as capture:
        solver.step(state, output, model.control(), contacts, 0.0025)
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    count = int(solver._constraint_count.numpy()[0])
    test.assertGreater(np.count_nonzero(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION), 0)
    test.assertFalse(np.array_equal(output.joint_q.numpy(), state.joint_q.numpy()))

    solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
    for _ in range(2):
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
        np.testing.assert_array_equal(output.joint_qd.numpy(), state.joint_qd.numpy())
        with test.assertRaisesRegex(RuntimeError, "contact torsion rejected its input"):
            solver.check_constraint_capacity()
    solver.reset(state)
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    test.assertFalse(np.array_equal(output.joint_q.numpy(), state.joint_q.numpy()))


def test_cuda_full_solver_graph(test, device):
    """Match captured and eager solves through the default-off public device API."""

    for velocity, regularization in ((16, 0.0), (0, 0.01)):
        results = []
        for captured in (False, True):
            _, solver, model, a, contacts = fixture(
                0.01,
                device=device,
                contact_torsion_device=True,
                iterations=16,
                velocity_iterations=velocity,
                contact_regularization=regularization,
                sliding=0.01,
                spin=0.4,
                **PATCH_OPTIONS,
            )
            z = model.state()
            control = model.control()
            pipeline = newton.CollisionPipeline(
                model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
            )

            def pair(a=a, z=z, pipeline=pipeline, solver=solver, control=control, contacts=contacts):
                a.clear_forces()
                pipeline.collide(a, contacts)
                solver.step(a, z, control, contacts, 0.0025)
                z.clear_forces()
                pipeline.collide(z, contacts)
                solver.step(z, a, control, contacts, 0.0025)

            pair()
            if captured:
                with wp.ScopedCapture(device=model.device) as capture:
                    pair()
                for _ in range(5):
                    wp.capture_launch(capture.graph)
                    solver.check_constraint_capacity()
            else:
                for _ in range(5):
                    pair()
                    solver.check_constraint_capacity()
            results.append((a.joint_q.numpy(), a.joint_qd.numpy(), a.body_q.numpy()))
        for eager, graph in zip(*results, strict=True):
            np.testing.assert_allclose(graph, eager, rtol=3e-5, atol=3e-6)


class TestDeviceTorsionPreparation(unittest.TestCase):
    """Compare device preparation kernels and the solver's device path against the host oracle."""


# Preparation kernels run on any device; solver tests need CUDA.
for _fn in (
    test_public_device_option_is_default_off,
    test_randomized_patch_and_point_oracle,
    test_host_math_rounding,
    test_grouping_and_jacobians_do_not_depend_on_blas,
    test_empty_and_larger_preallocated_buffer,
    test_ring_and_capacity_errors_are_latched,
    test_patch_segments_match_world_scan,
    test_patch_eligibility_and_point_cluster_edges,
    test_touching_threshold_preserves_host_rounding,
    test_rejected_region_and_normal_clusters,
    test_more_than_host_global_capacity,
    test_deferred_failure_is_not_partial_success,
    test_deferred_preparation_has_no_host_array_transfers,
):
    add_function_test(TestDeviceTorsionPreparation, _fn.__name__, _fn, devices=get_test_devices())


for _fn in (
    test_cuda_cancelling_angular_projection_matches_host_rounding,
    test_cuda_loaded_closed_linkage,
    test_cuda_solver_profiles,
    test_cuda_preparation_graph_changes_and_failure,
    test_cuda_captured_capacity_errors,
    test_public_api_noop_and_fail_stop,
    test_cuda_cold_capture_reports_errors_until_reset,
    test_cuda_full_solver_graph,
):
    add_function_test(TestDeviceTorsionPreparation, _fn.__name__, _fn, devices=get_cuda_test_devices())


def _check_budget(test, result, solver, groups=None):
    """Bound final shared impulses and account for their entire velocity response."""
    groups = solver._torsion_stats["groups"] if groups is None else groups
    count = int(result["count"][0])
    impulses = result["impulses"][0, :count]
    # Compare full velocities in float64 rather than subtracting two large
    # float32 spins. Repeated in-place updates at 100rad/s accumulate several
    # velocity ULPs even though the impulse budget and linear response agree.
    response = result["Y_world"][0, :count].astype(np.float64).T @ impulses.astype(np.float64)
    np.testing.assert_allclose(
        result["v_out"],
        result["v_hat"].astype(np.float64) + response,
        rtol=8 * np.finfo(np.float32).eps,
        atol=2e-5,
    )
    test.assertTrue(groups)
    for group in groups:
        load = sum(max(float(impulses[n]), 0.0) for n in group["normal_rows"])
        sliding = sum(float(np.linalg.norm(impulses[n + 1 : n + 3])) for n in group["anchor_rows"])
        spin = abs(float(impulses[group["row"]])) / group["effective_radius_m"]
        test.assertLessEqual(sliding + spin, group["mu"] * load + 2e-6)
    test.assertEqual(bool(np.any(solver.constraint_overflow.numpy())), False)


def test_analytic_regularized_load_and_saturated_spin(test, device):
    """Use the solved normal impulse once, without double-regularizing spin capacity."""
    for regularization in (0.01, 0.02, 0.5):
        with test.subTest(regularization=regularization):
            result, solver, *_ = fixture(
                0.01,
                device=device,
                center_only=True,
                spin=100.0,
                iterations=16,
                contact_regularization=regularization,
            )
            group = solver._torsion_stats["groups"][0]
            impulse = result["impulses"][0]
            # Two equal 0.3kg bodies close at 0.1m/s each: rigid lambda=.03Ns.
            normal = sum(float(impulse[n]) for n in group["normal_rows"])
            test.assertAlmostEqual(normal, 0.03 / (1 + regularization), delta=2e-7)
            test.assertAlmostEqual(abs(float(impulse[group["row"]])), 0.01 * 0.5 * normal, delta=2e-8)
            before = result["v_hat"].reshape(2, 6)
            after = result["v_out"].reshape(2, 6)
            test.assertLess(abs(after[0, 5] - after[1, 5]), abs(before[0, 5] - before[1, 5]))
            inertia = np.array([0.3, 0.3, 0.3, 0.00012, 0.00012, 0.00008])
            test.assertLessEqual(float(np.sum(after**2 * inertia)), float(np.sum(before**2 * inertia)) + 2e-6)
            _check_budget(test, result, solver)


def test_final_load_budget_across_iterations(test, device):
    """Correct stale sliding and spin when distributed normal loads shrink."""
    for regularization in (0.01, 0.02, 0.5):
        for iterations in (1, 2, 4, 16, 64):
            worlds, solver = _solve_initial_states(
                _BUDGET_STATES,
                device=device,
                iterations=iterations,
                contact_regularization=regularization,
                **PATCH_OPTIONS,
            )
            for (spin, _), (result, groups) in zip(_BUDGET_STATES, worlds, strict=True):
                with test.subTest(reg=regularization, iterations=iterations, spin=spin):
                    _check_budget(test, result, solver, groups)
                    count = int(result["count"][0])
                    types = result["row_type"][0, :count]
                    weights = result["row_w"][0, :count]
                    normal_rows = types == 0
                    test.assertTrue(np.any(normal_rows))
                    np.testing.assert_allclose(weights[normal_rows], 1 / (1 + regularization), atol=1e-7)
                    # Torsion itself is a dry-friction row, not another soft normal.
                    np.testing.assert_array_equal(weights[types == PGS_CONSTRAINT_TYPE_TORSION], 1.0)


def test_zero_controls_and_no_load(test, device):
    """Preserve excluded/radius-zero controls and forbid spin without compressive load."""
    for regularization in (0.0, 0.01, 0.02):
        base, *_ = fixture(0.0, device=device, contact_regularization=regularization)
        excluded, *_ = fixture(0.01, device=device, contact_regularization=regularization, torsion_shapes=())
        for name in base:
            np.testing.assert_array_equal(base[name], excluded[name], err_msg=name)
        for options in ({"closing": 0.0}, {"mu": 0.0}):
            result, *_ = fixture(
                0.01, device=device, contact_regularization=regularization, center_only=True, **options
            )
            test.assertLess(
                np.abs(result["impulses"][result["row_type"] == PGS_CONSTRAINT_TYPE_TORSION]).max(initial=0), 1e-9
            )


def test_substep_force_export_and_release(test, device):
    """Export the actual solved normal load over dt and forget torque on release."""
    for substeps in (1, 2, 4):
        dt = 0.005 / substeps
        result, solver, model, initial, contacts = fixture(
            0.01,
            device=device,
            center_only=True,
            spin=100.0,
            dt=dt,
            contact_regularization=0.02,
            **PATCH_OPTIONS,
        )
        count = int(contacts.rigid_contact_count.numpy()[0])
        forces = result["contact_f"][:count, :3]
        normals = contacts.rigid_contact_normal.numpy()[:count]
        slots = solver._contact_slot.numpy()[:count]
        for contact, slot in enumerate(slots):
            if slot >= 0:
                test.assertAlmostEqual(
                    abs(float(np.dot(forces[contact], normals[contact]))) * dt,
                    float(result["impulses"][0, slot]),
                    delta=2e-7,
                )
        contacts.rigid_contact_count.zero_()
        solver.step(initial, model.state(), model.control(), contacts, dt)
        test.assertEqual(solver._torsion_stats["rows"], 0)


# (spin [rad/s], sliding [m/s]) initial states of the final-load budget cases, one world each.
_BUDGET_STATES = ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0))


def _solve_initial_states(states, *, device, radius=0.0075, closing=0.1, dt=0.0025, **solver_overrides):
    """Step ``fixture``'s pad pair once per ``(spin, sliding)`` state, each pair in its own world of one model.

    Returns one ``(result, torsion groups)`` per world, ``result`` laid out like a one-world ``fixture`` result.
    """
    pads = newton.ModelBuilder(gravity=(0, 0, 0))
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * 0.025), wp.quat_identity())
        body = pads.add_link(
            xform=pose, mass=0.3, inertia=wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        )
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        joint = pads.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        pads.add_articulation([joint])
        pads.add_shape_box(body, hx=0.02, hy=0.015, hz=0.025, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5))
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    builder.replicate(pads, len(states), spacing=(1.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    set_torsion_radius(model, radius)
    a, z = model.state(), model.state()
    a.joint_qd.assign(
        np.array(
            [[sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin] for spin, sliding in states],
            np.float32,
        ).ravel()
    )
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline = newton.CollisionPipeline(
        model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64 * len(states)
    )
    contacts = pipeline.contacts()
    pipeline.collide(a, contacts)
    # fixture's solver settings; the row capacities stay per world.
    options = {
        "friction_anchor_beta": 0.0,
        "iterations": 128,
        "velocity_iterations": 0,
        "baumgarte": 0.05,
        "cfm": 0,
        "contact_regularization": 0,
        "enable_joint_velocity_limits": True,
        "contact_warmstart": False,
        "articulation_constraint_max": 64,
        "free_body_constraint_max": 16,
        **solver_overrides,
    }
    solver = SolverSteel(model, solve_mode="matrix_free", **options)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
    solver.step(a, z, model.control(), contacts, dt, observables=observables)
    test_arrays = {name: getattr(solver, f"_{name}").numpy() for name in ("impulses", "row_type", "row_w", "Y_world")}
    count = solver._constraint_count.numpy()
    velocities = {name: getattr(solver, f"_{name}").numpy().reshape(len(states), -1) for name in ("v_hat", "v_out")}
    worlds = []
    for world in range(len(states)):
        result = {name: values[world : world + 1] for name, values in test_arrays.items()}
        result.update({name: values[world] for name, values in velocities.items()})
        result["count"] = count[world : world + 1]
        groups = [group for group in solver._torsion_stats["groups"] if group["world"] == world]
        worlds.append((result, groups))
    return worlds, solver


class TestTorsionRegularization(unittest.TestCase):
    pass


for _fn in (
    test_analytic_regularized_load_and_saturated_spin,
    test_final_load_budget_across_iterations,
    test_zero_controls_and_no_load,
    test_substep_force_export_and_release,
):
    add_function_test(TestTorsionRegularization, _fn.__name__, _fn, devices=get_cuda_test_devices())


def _conclude_position_problem(solver, dt):
    """Switch the solved position problem to the velocity-only right-hand sides, as a step does."""
    solver._compute_velocity_pass_rhs(dt)


def _run_velocity_post_solve(solver):
    """Run the velocity-only iterations of a step on the current row state."""
    solver._pack_mf_meta(solver._mf_rhs_unbiased)
    solver._launch_pgs_solve(
        solver._rhs_unbiased,
        solver.velocity_iterations,
        regularize=False,
        freeze_drive_rows=solver.velocity_drive_mode == "freeze",
    )


def test_end_gap_retires_spin_but_preserves_target_and_carried_impulse(test, device):
    """Retire separated spin groups without deleting their required velocity correction."""
    for initial_gap, end_gap in ((-0.001, -0.001), (-0.001, 0.0001), (0.0001, -0.0001), (0.0001, 0.0002)):
        with test.subTest(initial_gap=initial_gap, end_gap=end_gap):
            count = wp.array([2], dtype=int, device=device)
            types = wp.array([[0, PGS_CONSTRAINT_TYPE_TORSION]], dtype=int, device=device)
            groups = wp.array([[1, -1]], dtype=int, device=device)
            phi = wp.array([[initial_gap, 0.0]], dtype=float, device=device)
            target = wp.array([[0.0, 0.25]], dtype=float, device=device)
            velocity = wp.array([(end_gap - initial_gap) / 0.01], dtype=float, device=device)
            dofs = wp.array([1], dtype=int, device=device)
            indices = wp.array([[0]], dtype=int, device=device)
            jacobian = wp.array(np.array([[[1.0], [0.0]]], np.float32), device=device)
            rhs = wp.array([[3.0, 9.0]], dtype=float, device=device)
            wp.launch(
                contact_torsion._prepare_velocity_torsion_rows,
                dim=2,
                inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                outputs=[rhs],
                device=device,
            )
            retired = initial_gap > 1e-6 and end_gap > 1e-6
            test.assertEqual(int(groups.numpy()[0, 1]), contact_torsion._TORSION_SPIN_RETIRED if retired else -1)
            np.testing.assert_array_equal(rhs.numpy(), [[3.0, -0.25]])
            # Re-admission must clear a previous retirement, including when
            # these same buffers are reused by a captured graph replay.
            phi.assign(np.array([[-0.001, 0.0]], dtype=np.float32))
            wp.launch(
                contact_torsion._prepare_velocity_torsion_rows,
                dim=2,
                inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                outputs=[rhs],
                device=device,
            )
            np.testing.assert_array_equal(groups.numpy(), [[1, -1]])


def test_rebound_preserves_loaded_sliding_when_spin_retires(test, device):
    """Preserve ordinary sliding friction on an admitted rebounding contact."""
    for separation in (0.0, 5.0e-6):
        for restitution in (0.0, 0.5):
            with test.subTest(separation=separation, restitution=restitution):
                options = dict(
                    separation=separation,
                    restitution=restitution,
                    closing=1.0,
                    sliding=1.0,
                    spin=0.0,
                    center_only=True,
                    iterations=64,
                    velocity_iterations=16,
                    **PATCH_OPTIONS,
                )
                baseline, *_ = fixture(0.0, device=device, **options)
                actual, solver, *_ = fixture(0.01, device=device, **options)
                normals = actual["row_type"] == 0
                tangent = actual["row_type"] == 2
                np.testing.assert_allclose(actual["impulses"][normals], baseline["impulses"][normals], atol=2e-6)
                test.assertGreater(float(np.linalg.norm(baseline["impulses"][tangent])), 0.01)
                np.testing.assert_allclose(actual["impulses"][tangent], baseline["impulses"][tangent], atol=2e-6)
                np.testing.assert_allclose(actual["body_qd"], baseline["body_qd"], atol=3e-5)
                _assert_budget(test, actual, solver)


def test_spin_opposition_and_final_budget(test, device):
    """Dissipate angular slip and respect final normal load in each velocity count."""
    for velocity in (1, 4, 16):
        for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0)):
            with test.subTest(velocity=velocity, spin=spin, sliding=sliding):
                result, solver, *_ = fixture(
                    0.01,
                    device=device,
                    iterations=16,
                    velocity_iterations=velocity,
                    spin=spin,
                    sliding=sliding,
                    center_only=sliding == 0,
                    **PATCH_OPTIONS,
                )
                _assert_budget(test, result, solver)
                test.assertTrue(np.isfinite(result["body_qd"]).all())
                if sliding == 0:
                    test.assertLess(np.max(np.abs(result["body_qd"][:, 5])), spin)


def test_postpass_accumulated_impulse_response_and_position_split(test, device):
    """Apply delta impulses once while retaining the biased position trajectory."""
    original = fixture(
        0.01,
        device=device,
        spin=100.0,
        closing=0.1,
        center_only=True,
        iterations=16,
        **PATCH_OPTIONS,
    )
    result, solver, *_ = fixture(
        0.01,
        device=device,
        spin=100.0,
        closing=0.1,
        center_only=True,
        iterations=16,
        velocity_iterations=4,
        **PATCH_OPTIONS,
    )
    baseline = original[0]
    np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
    count = int(result["count"][0])
    delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
    np.testing.assert_allclose(result["v_out"] - baseline["v_out"], result["Y_world"][0, :count].T @ delta, atol=2e-5)
    _assert_budget(test, result, solver)


def test_zero_load_and_speculation(test, device):
    """Remove torque when normal load or current touching support disappears."""
    for options in ({"closing": 0.0}, {"closing": -0.1}, {"separation": 0.004}, {"mu": 0.0}):
        with test.subTest(options=options):
            result, solver, *_ = fixture(0.01, device=device, velocity_iterations=4, **PATCH_OPTIONS, **options)
            active = result["row_type"] == PGS_CONSTRAINT_TYPE_TORSION
            test.assertLess(np.abs(result["impulses"][active]).max(initial=0), 2e-7)
            _assert_budget(test, result, solver)


def test_carried_spin_is_refunded_when_postpass_retires_group(test, device):
    """Undo carried spin through its response column when final support is separated."""
    original, solver, model, state, contacts = fixture(
        0.01, device=device, spin=100.0, center_only=True, velocity_iterations=4, **PATCH_OPTIONS
    )
    group = solver._torsion_stats["groups"][0]
    count = int(solver._constraint_count.numpy()[0])
    before_lambda = solver._impulses.numpy()[0, :count].copy()
    test.assertGreater(abs(float(before_lambda[group["row"]])), 1e-8)
    before_velocity = solver._v_out.numpy().copy()
    phi = solver._phi.numpy()
    phi[0, group["normal_rows"]] = 0.01
    solver._phi.assign(phi)
    solver._v_out_snap.zero_()
    _conclude_position_problem(solver, 0.0025)
    test.assertEqual(float(solver._row_mu.numpy()[0, group["row"]]), group["mu"])
    test.assertEqual(int(solver._contact_torsion_group.numpy()[0, group["row"]]), contact_torsion._TORSION_SPIN_RETIRED)
    # Eligibility changes the admissible set, not the already-applied load.
    np.testing.assert_array_equal(solver._impulses.numpy()[0, :count], before_lambda)
    _run_velocity_post_solve(solver)
    after_lambda = solver._impulses.numpy()[0, :count]
    test.assertLess(abs(float(after_lambda[group["row"]])), 1e-9)
    np.testing.assert_allclose(
        solver._v_out.numpy() - before_velocity,
        solver._Y_world.numpy()[0, :count].T @ (after_lambda - before_lambda),
        atol=2e-5,
    )
    # A new position solve must restore admission rather than inherit the
    # prior velocity-pass retirement in a reused membership buffer.
    output = model.state()
    solver.step(state, output, model.control(), contacts, 0.0025)
    group = solver._torsion_stats["groups"][0]
    test.assertEqual(int(solver._contact_torsion_group.numpy()[0, group["row"]]), -1)
    test.assertGreater(abs(float(solver._impulses.numpy()[0, group["row"]])), 1e-8)
    np.testing.assert_allclose(output.body_qd.numpy(), original["body_qd"], atol=2e-5)


def test_internal_timestep_export_uses_final_linear_impulse(test, device):
    """Export final normal impulses in newtons at each internal substep duration."""
    for dt in (0.00125, 0.0025, 0.005):
        with test.subTest(dt=dt):
            result, solver, model, initial, contacts = fixture(
                0.01, device=device, spin=100.0, center_only=True, dt=dt, velocity_iterations=4, **PATCH_OPTIONS
            )
            observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
            for _ in range(4):
                output = model.state()
                solver.step(initial, output, model.control(), contacts, dt, observables=observables)
                row = int(solver._contact_slot.numpy()[0])
                normal = contacts.rigid_contact_normal.numpy()[0]
                force = observables.contact_f.numpy()[0]
                expected = max(float(solver._impulses.numpy()[0, row]), 0.0) / dt
                test.assertAlmostEqual(abs(float(np.dot(force[:3], normal))), expected, delta=1e-4)
                # Torque slots are deliberately placeholders, not solved spin wrench.
                np.testing.assert_array_equal(force[3:], 0.0)
                np.testing.assert_allclose(output.body_qd.numpy(), result["body_qd"], atol=2e-5)


def test_selected_out_torsion_matches_zero_radius(test, device):
    """Keep the torsion-zero velocity path identical when no shapes are selected."""
    for velocity in (0, 4):
        expected, *_ = fixture(0.0, device=device, velocity_iterations=velocity, **PATCH_OPTIONS)
        actual, *_ = fixture(0.01, device=device, velocity_iterations=velocity, torsion_shapes=(), **PATCH_OPTIONS)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


class TestTorsionVelocityAdmission(unittest.TestCase):
    pass


add_function_test(
    TestTorsionVelocityAdmission,
    "test_end_gap_retires_spin_but_preserves_target_and_carried_impulse",
    test_end_gap_retires_spin_but_preserves_target_and_carried_impulse,
    devices=get_test_devices(),
)


class TestTorsionVelocitySolve(unittest.TestCase):
    pass


for _fn in (
    test_rebound_preserves_loaded_sliding_when_spin_retires,
    test_spin_opposition_and_final_budget,
    test_postpass_accumulated_impulse_response_and_position_split,
    test_zero_load_and_speculation,
    test_carried_spin_is_refunded_when_postpass_retires_group,
    test_internal_timestep_export_uses_final_linear_impulse,
    test_selected_out_torsion_matches_zero_radius,
):
    add_function_test(TestTorsionVelocitySolve, _fn.__name__, _fn, devices=get_cuda_test_devices())


def test_final_budget_and_velocity_response(test, device):
    """Bound final traction plus spin and apply every correction to velocity."""
    for iterations in (1, 2, 4, 8, 64):
        worlds, _ = _solve_initial_states(
            _BUDGET_STATES, device=device, friction_anchor_beta=0.2, iterations=iterations
        )
        for (spin, sliding), (result, groups) in zip(_BUDGET_STATES, worlds, strict=True):
            with test.subTest(iterations=iterations, spin=spin, sliding=sliding):
                count = int(result["count"][0])
                response = result["Y_world"][0, :count].T @ result["impulses"][0, :count]
                np.testing.assert_allclose(result["v_out"] - result["v_hat"], response, atol=2e-5)
                test.assertTrue(groups, "The regression must exercise coupled patch/torsion rows")
                impulse = result["impulses"][0]
                for group in groups:
                    normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
                    traction = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
                    spin_used = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
                    test.assertLessEqual(traction + spin_used, group["mu"] * normal + 2e-6)


class TestFinalPatchBudget(unittest.TestCase):
    pass


add_function_test(
    TestFinalPatchBudget,
    "test_final_budget_and_velocity_response",
    test_final_budget_and_velocity_response,
    devices=get_cuda_test_devices(),
)


_PROPAGATION_TORSION = {"articulated_contact_response": "propagation", "contact_torsion_device": True}
# A central witness leaves the spin to torsion; spinning corner witnesses slide on the whole budget.
_CENTER_SPIN = {"center_only": True, "spin": 100.0}
_CORNER_SPIN = {"spin": 10.0, "sliding": 0.2}


def test_propagation_rows_host_groups_in_spares(test, device):
    """Host each group's row in its last member's spare, keep other spares inert and match immediate's groups."""
    for options in ({}, PATCH_OPTIONS):
        with test.subTest(patches=bool(options)):
            reference, *_ = fixture(0.01, device=device, spin=10.0, contact_torsion_device=True, **options)
            _, solver, _model, _state, contacts = fixture(
                0.01, device=device, spin=10.0, **options, **_PROPAGATION_TORSION
            )
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 1)
            test.assertTrue(np.all(solver._contact_path.numpy()[:count] == 2))
            rows = _propagation_torsion_rows(solver)
            dense_rows = reference["row_type"][0, : int(reference["count"][0])]
            test.assertEqual(len(rows), int(np.count_nonzero(dense_rows == PGS_CONSTRAINT_TYPE_TORSION)))
            test.assertEqual(len(rows), 1)
            spare_rows = _spare_rows(solver, count)
            row_type = solver._propagation_row_type.numpy()[0]
            parent = solver._propagation_row_parent.numpy()[0]
            body_a = solver._propagation_body_a.numpy()[0]
            body_b = solver._propagation_body_b.numpy()[0]
            J_a = solver._propagation_J_a.numpy()[0]
            J_b = solver._propagation_J_b.numpy()[0]
            normals = -contacts.rigid_contact_normal.numpy()[:count]
            slots = solver._contact_slot.numpy()[:count]
            contact_of = {int(slots[c]): c for c in range(count)}
            for spin, members in rows.items():
                test.assertEqual(spin, spare_rows[max(contact_of[n] for n in members)])
                lead_row = int(parent[spin])
                lead = contact_of[lead_row]
                test.assertEqual((body_a[spin], body_b[spin]), (body_a[lead_row], body_b[lead_row]))
                np.testing.assert_array_equal(J_a[spin, :3], 0.0)
                np.testing.assert_array_equal(J_b[spin, :3], 0.0)
                np.testing.assert_allclose(J_a[spin, 3:], normals[lead], atol=1e-7)
                np.testing.assert_allclose(J_b[spin, 3:], -normals[lead], atol=1e-7)
                test.assertEqual(sorted(members), sorted(int(slots[c]) for c in range(count)))
            for row in set(spare_rows.values()) - set(rows):
                test.assertEqual(row_type[row], PGS_CONSTRAINT_TYPE_TORSION)
                test.assertEqual((body_a[row], body_b[row]), (-1, -1))
                np.testing.assert_array_equal(J_a[row], 0.0)
            test.assertEqual(
                int(solver._propagation_constraint_count.numpy()[0]),
                int(solver._contact_slots_needed.numpy()[:count].sum()),
            )


def test_propagation_free_body_contacts_reserve_no_spare(test, device):
    """Give free-body contacts, which immediate solves without torsion, no spare and no torsion row."""
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0099), wp.quat_identity()))
    b.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
    model = b.finalize(device=device)
    set_torsion_radius(model, 0.01)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        friction_anchor_beta=0.0,
        velocity_iterations=0,
        iterations=8,
        **_PROPAGATION_TORSION,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    solver.step(state, model.state(), model.control(), contacts, 0.0025)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.all(solver._contact_path.numpy()[:count] == 2))
    test.assertTrue(np.all(solver._contact_slots_needed.numpy()[:count] == 3))
    test.assertEqual(int(solver._propagation_torsion_head.numpy()[0]), -1)


def test_propagation_budget_holds_after_every_sweep(test, device):
    """Keep each torsion impulse within its group's residual sliding budget after every plain sweep."""
    for patches in ({}, PATCH_OPTIONS):
        for case, options in (("center", _CENTER_SPIN), ("corners", _CORNER_SPIN)):
            with test.subTest(patches=bool(patches), case=case):
                sweeps = _record_propagation_sweeps(device, 0.01, **options, **patches)
                saturated, sliding_saturated = _assert_torsion_budgets(test, sweeps)
                if case == "center":
                    test.assertTrue(saturated, "No torsion row reached its residual budget")
                else:
                    test.assertTrue(sliding_saturated, "No group's sliding pairs used the whole budget")


def test_propagation_spin_decelerates_at_the_coulomb_rate_and_sticks(test, device):
    """Decelerate a sliding spin by r mu N / I and hold a slow spin, on both responses."""
    inertia_z = 0.00008
    for response in ("immediate", "propagation"):
        with test.subTest(response=response):
            options = {"articulated_contact_response": response, "contact_torsion_device": True}
            result, solver, *_ = fixture(0.01, device=device, spin=100.0, center_only=True, **options)
            if response == "immediate":
                count = int(result["count"][0])
                impulses, types = result["impulses"][0, :count], result["row_type"][0, :count]
            else:
                count = int(solver._propagation_constraint_count.numpy()[0])
                impulses = solver._propagation_impulses.numpy()[0, :count]
                types = solver._propagation_row_type.numpy()[0, :count]
            load = float(impulses[types == PGS_CONSTRAINT_TYPE_CONTACT].sum())
            test.assertGreater(load, 0.0)
            expected = 0.01 * 0.5 * load / inertia_z
            deceleration = 100.0 - float(result["joint_qd"][5])
            test.assertAlmostEqual(deceleration, expected, delta=0.02 * expected)
            held, *_ = fixture(0.01, device=device, spin=1.0, center_only=True, **options)
            test.assertLess(np.max(np.abs(held["body_qd"][:, 5])), 1e-4)


def test_propagation_friction_keeps_its_rows_with_a_spare(test, device):
    """Solve and report friction as immediate does when the propagation contacts reserve torsion spares."""
    for patches in ({}, PATCH_OPTIONS):
        with test.subTest(patches=bool(patches)):
            results = {}
            for response in ("immediate", "propagation"):
                options = {"articulated_contact_response": response, "contact_torsion_device": True}
                results[response], *_ = fixture(0.01, device=device, spin=10.0, sliding=0.2, **options, **patches)
            expected, actual = results["immediate"], results["propagation"]
            test.assertGreater(float(np.abs(expected["contact_f"][:, :2]).max()), 0.0)
            np.testing.assert_allclose(actual["contact_f"], expected["contact_f"], rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(actual["body_qd"], expected["body_qd"], rtol=1e-3, atol=1e-3)


def test_propagation_spin_effect_tracks_immediate(test, device):
    """Require torsion to change a spin far beyond the response gap, and within the shared non-no-op contract."""
    x = {}
    for response in ("immediate", "propagation"):
        for radius in (0.0, 0.01):
            x[response, radius], _ = _spinner(device, response, 32, radius, spin=4.0)
    gap_off = float(np.abs(x["propagation", 0.0] - x["immediate", 0.0]).max())
    gap_on = float(np.abs(x["propagation", 0.01] - x["immediate", 0.01]).max())
    effect_immediate = float(np.abs(x["immediate", 0.01] - x["immediate", 0.0]).max())
    effect = float(np.abs(x["propagation", 0.01] - x["propagation", 0.0]).max())
    test.assertGreaterEqual(effect, 0.5 * effect_immediate)
    test.assertGreaterEqual(effect, 10.0 * gap_off)
    test.assertLessEqual(gap_on, 2.0 * gap_off + 1.0e-6)


def test_propagation_converges_to_the_immediate_response(test, device):
    """Close the coupled three-foot gap to immediate as sweeps rise, with torsion deciding stick."""
    gaps = {}
    for iterations in (8, 32, 128):
        expected, _ = _spinner(device, "immediate", iterations, 0.01, spin=1.14)
        actual, solver = _spinner(device, "propagation", iterations, 0.01, spin=1.14)
        gaps[iterations] = float(np.abs(actual - expected).max())
        test.assertEqual(len(_propagation_torsion_rows(solver)), 3)
    released, _ = _spinner(device, "immediate", 128, 0.0, spin=1.14)
    held, _ = _spinner(device, "immediate", 128, 0.01, spin=1.14)
    test.assertGreater(abs(float(released[1])), 10.0 * abs(float(held[1])) + 1.0e-3)
    test.assertLessEqual(gaps[128], 1.0e-4, gaps)
    test.assertLessEqual(gaps[128], gaps[32] + 1.0e-6, gaps)


def test_propagation_worlds_solve_as_if_isolated(test, device):
    """Solve each world of an odd world count, packed two per block, as the same world alone."""
    spins = (4.0, 1.14, 0.3)
    together, solver = _spinner(device, "propagation", 32, 0.01, spin=spins, steps=2)
    test.assertEqual(solver_steel._PROPAGATION_WORLDS_PER_BLOCK, 2)
    for world, spin in enumerate(spins):
        test.assertEqual(len(_propagation_torsion_rows(solver, world)), 3)
        alone, _ = _spinner(device, "propagation", 32, 0.01, spin=spin, steps=2)
        np.testing.assert_allclose(together.reshape(-1, 2)[world], alone, atol=1.0e-6)


def test_propagation_contact_order_changes_the_result_no_more_than_on_immediate(test, device):
    """Reorder the contact buffer: propagation's change stays within immediate's order floor plus float noise."""
    results = {}
    for response in ("immediate", "propagation"):
        for reverse in (False, True):
            _, solver, model, state, contacts = fixture(
                0.01,
                device=device,
                spin=10.0,
                sliding=0.2,
                step=False,
                articulated_contact_response=response,
                contact_torsion_device=True,
            )
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 2)
            if reverse:
                _permute_contacts(contacts, np.arange(count)[::-1])
            output = model.state()
            solver.step(state, output, model.control(), contacts, 0.0025)
            results[response, reverse] = output.body_qd.numpy()
    floor = float(np.abs(results["immediate", True] - results["immediate", False]).max())
    change = float(np.abs(results["propagation", True] - results["propagation", False]).max())
    test.assertLessEqual(change, floor + 1.0e-6, (change, floor))


def test_propagation_dropped_members_leave_the_highest_kept_host(test, device):
    """Host a group whose members overflowed the row cap after its last kept member, budgeting kept members only."""
    _, solver, model, state, contacts = fixture(
        0.01,
        device=device,
        spin=10.0,
        step=False,
        enable_joint_velocity_limits=False,
        articulation_constraint_max=8,
        free_body_constraint_max=2,
        **_PROPAGATION_TORSION,
    )
    wp.synchronize_device(model.device)
    capture = StdOutCapture()
    capture.begin()
    try:
        solver.step(state, model.state(), model.control(), contacts, 0.0025)
    finally:
        try:
            wp.synchronize_device(model.device)
        finally:
            output = capture.end()
    lines = [line for line in output.splitlines() if line.strip() and not line.startswith("Module")]
    test.assertEqual(len(lines), 1, output)
    test.assertRegex(
        lines[0],
        r"Warning: Steel propagation constraint-row overflow in world 0: requested 16 rows, limit 10; "
        r"dropped 8 contact/friction rows\.",
    )
    count = int(contacts.rigid_contact_count.numpy()[0])
    slots = solver._contact_slot.numpy()[:count]
    kept = [c for c in range(count) if slots[c] >= 0]
    test.assertEqual((len(kept), count), (2, 4))
    rows = _propagation_torsion_rows(solver)
    test.assertEqual(len(rows), 1)
    ((spin, members),) = rows.items()
    test.assertEqual(sorted(members), sorted(int(slots[c]) for c in kept))
    test.assertEqual(spin, _spare_rows(solver, count)[max(kept)])
    test.assertGreater(int(solver._row_dropped_propagation.numpy()[0]), 0)


def test_propagation_rows_follow_the_recorded_reservation(test, device):
    """Write no tangent rows for a contact allocated without them, whatever the builder's own gap test says."""
    allocator = solver_steel.allocate_world_contact_slots
    gap_index = [arg.label for arg in allocator.adj.args].index("contact_friction_gap_limit")
    launch = wp.launch

    def frictionless_allocation(kernel, *args, **kwargs):
        # Only the allocator sees every contact past the friction gap.
        if kernel is allocator:
            kwargs["inputs"] = list(kwargs["inputs"])
            kwargs["inputs"][gap_index] = -float("inf")
        return launch(kernel, *args, **kwargs)

    _, solver, model, state, contacts = fixture(0.01, device=device, spin=10.0, step=False, **_PROPAGATION_TORSION)
    with patch.object(solver_steel.wp, "launch", frictionless_allocation):
        solver.step(state, model.state(), model.control(), contacts, 0.0025)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertTrue(np.all(solver._contact_slots_needed.numpy()[:count] == 2))
    rows = int(solver._propagation_constraint_count.numpy()[0])
    row_type = solver._propagation_row_type.numpy()[0, :rows]
    test.assertEqual(rows, 2 * count)
    np.testing.assert_array_equal(row_type[::2], PGS_CONSTRAINT_TYPE_CONTACT)
    np.testing.assert_array_equal(row_type[1::2], PGS_CONSTRAINT_TYPE_TORSION)


def test_propagation_prescribed_spin_turns_a_resting_sphere(test, device):
    """Spin up a sphere resting on a spinning kinematic platform through torsion alone, as on immediate."""
    rates = {response: _turntable(device, response) for response in ("immediate", "propagation")}
    test.assertGreater(rates["immediate"], 0.5)
    test.assertAlmostEqual(rates["propagation"], rates["immediate"], delta=1.0e-4)


def test_propagation_mixed_families_and_contact_lifecycle(test, device):
    """Solve dense and propagation torsion rows and free-body rows in one step, through contact loss and recovery."""
    model, spinner_q = _mixed_scene(device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=32,
        velocity_iterations=0,
        baumgarte=0.05,
        cfm=0.0,
        friction_anchor_beta=0.0,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        propagation_same_articulation_rows=False,
        **_PROPAGATION_TORSION,
    )
    pipeline = newton.CollisionPipeline(model, reduce_contacts=False, broad_phase="nxn", rigid_contact_max=256)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    velocity = np.zeros(model.joint_dof_count, np.float32)
    velocity[spinner_q + 5] = 20.0
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

    def step(lift=None):
        nonlocal state, output
        if lift is not None:
            q = state.joint_q.numpy()
            q[spinner_q + 2] = lift
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 0.0025)
        state, output = output, state
        solver.check_constraint_capacity()
        return _family_census(solver, contacts)

    for lift, groups in ((None, [1, 1]), (0.2, [0, 1]), (0.0, [1, 1])):
        with test.subTest(lift=lift):
            census = step(lift)
            test.assertEqual(census["propagation_groups"], groups)
            test.assertTrue(all(count > 0 for count in census["dense_torsion"]), census)
            test.assertTrue(census["free_body_rows"], census)
            test.assertTrue(np.isfinite(state.joint_qd.numpy()).all())
    solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
    with test.assertRaisesRegex(RuntimeError, "load ring"):
        step()
    solver.reset(state)
    census = step()
    test.assertEqual(census["propagation_groups"], [1, 1])


def test_propagation_patch_impulse_matches_immediate(test, device):
    """Give a patch region the same torsion row count and final impulse as immediate at 128 sweeps."""
    impulses = {}
    for response in ("immediate", "propagation"):
        options = {"articulated_contact_response": response, "contact_torsion_device": True, "iterations": 128}
        result, solver, *_ = fixture(
            0.01, device=device, spin=1.0, center_only=True, center_count=4, **options, **PATCH_OPTIONS
        )
        if response == "immediate":
            count = int(result["count"][0])
            torsion = result["impulses"][0, :count][result["row_type"][0, :count] == PGS_CONSTRAINT_TYPE_TORSION]
        else:
            rows = _propagation_torsion_rows(solver)
            test.assertEqual([len(members) for members in rows.values()], [4])
            torsion = solver._propagation_impulses.numpy()[0, sorted(rows)]
        impulses[response] = torsion
    test.assertEqual(len(impulses["immediate"]), 1)
    test.assertGreater(abs(float(impulses["immediate"][0])), 0.0)
    np.testing.assert_allclose(impulses["propagation"], impulses["immediate"], rtol=1.0e-4)


def test_propagation_captured_steps_match_eager_steps_and_report_errors(test, device):
    """Replay captured propagation steps as the eager steps do, and fail stop on a latched torsion error."""
    eager, _ = _spinner(device, "propagation", 16, 0.01, spin=4.0, steps=3)
    captured, _ = _spinner(device, "propagation", 16, 0.01, spin=4.0, steps=3, graph=True)
    np.testing.assert_allclose(captured, eager, atol=1.0e-6)
    _, solver, model, state, contacts = fixture(0.01, device=device, iterations=16, step=False, **_PROPAGATION_TORSION)
    output = model.state()
    with wp.ScopedCapture(device=model.device) as capture:
        solver.step(state, output, model.control(), contacts, 0.0025)
    wp.capture_launch(capture.graph)
    solver.check_constraint_capacity()
    test.assertEqual(len(_propagation_torsion_rows(solver)), 1)
    solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
    wp.capture_launch(capture.graph)
    with test.assertRaisesRegex(RuntimeError, "load ring"):
        solver.check_constraint_capacity()
    np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
    test.assertTrue(np.all(solver._propagation_constraint_count.numpy() == 0))


def test_propagation_unsupported_combinations_raise(test, device):
    """Keep torsion rejected where its rows are not solved."""
    for options, error, message in (
        (
            {"articulated_contact_response": "propagation_colored", "contact_torsion_device": True},
            NotImplementedError,
            "contact torsion is not supported",
        ),
        ({**_PROPAGATION_TORSION, "contact_warmstart": True}, ValueError, "contact_warmstart=False"),
        ({**_PROPAGATION_TORSION, "velocity_iterations": 1}, NotImplementedError, "velocity_iterations=0"),
        ({"articulated_contact_response": "propagation"}, NotImplementedError, "contact_torsion_device=True"),
    ):
        with test.subTest(**options), test.assertRaisesRegex(error, message):
            fixture(0.01, device=device, step=False, **options)


class TestPropagationTorsion(unittest.TestCase):
    """Contact torsion rows on the plain propagation response, against the immediate response."""


for _fn in (
    test_propagation_rows_host_groups_in_spares,
    test_propagation_free_body_contacts_reserve_no_spare,
    test_propagation_budget_holds_after_every_sweep,
    test_propagation_spin_decelerates_at_the_coulomb_rate_and_sticks,
    test_propagation_friction_keeps_its_rows_with_a_spare,
    test_propagation_spin_effect_tracks_immediate,
    test_propagation_converges_to_the_immediate_response,
    test_propagation_worlds_solve_as_if_isolated,
    test_propagation_contact_order_changes_the_result_no_more_than_on_immediate,
    test_propagation_rows_follow_the_recorded_reservation,
    test_propagation_prescribed_spin_turns_a_resting_sphere,
    test_propagation_mixed_families_and_contact_lifecycle,
    test_propagation_patch_impulse_matches_immediate,
    test_propagation_captured_steps_match_eager_steps_and_report_errors,
    test_propagation_unsupported_combinations_raise,
):
    add_function_test(TestPropagationTorsion, _fn.__name__, _fn, devices=get_cuda_test_devices())

# The test checks its own output: CheckOutput's capture would receive the device printf.
add_function_test(
    TestPropagationTorsion,
    "test_propagation_dropped_members_leave_the_highest_kept_host",
    test_propagation_dropped_members_leave_the_highest_kept_host,
    devices=get_cuda_test_devices(),
    check_output=False,
)


def _propagation_torsion_rows(solver, world=0):
    """Return world ``world``'s propagation torsion rows mapped to their member normal rows, from the sweep's lists."""
    head = solver._propagation_torsion_head.numpy()
    first = solver._propagation_torsion_first.numpy()[world]
    following = solver._propagation_torsion_next.numpy()[world]
    rows = {}
    spin = int(head[world])
    while spin >= 0:
        members, n = [], int(first[spin])
        while n >= 0:
            members.append(n)
            n = int(following[n])
        rows[spin] = members
        spin = int(following[spin])
    return rows


def _spare_rows(solver, count):
    """Return each spare-reserving propagation contact's spare row, the last of its reserved rows."""
    slots = solver._contact_slot.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    return {c: int(slots[c] + needed[c] - 1) for c in range(count) if slots[c] >= 0 and needed[c] in (2, 4)}


def _record_propagation_sweeps(device, radius, **options):
    """Record world 0's propagation impulses after every sweep of one pad-fixture step."""
    sweeps = []
    _, solver, model, state, contacts = fixture(radius, device=device, step=False, **options, **_PROPAGATION_TORSION)
    launch_tiled = wp.launch_tiled

    def record(kernel, *args, **kwargs):
        launch_tiled(kernel, *args, **kwargs)
        if kernel is not solver._pgs_solve_propagation_kernel:
            return
        rows = _propagation_torsion_rows(solver)
        row_type = solver._propagation_row_type.numpy()[0]
        parent = solver._propagation_row_parent.numpy()[0]
        count = int(solver._propagation_constraint_count.numpy()[0])
        pairs = {
            spin: [
                n
                for n in members
                if n + 2 < count
                and all(row_type[n + k] == PGS_CONSTRAINT_TYPE_FRICTION and parent[n + k] == n for k in (1, 2))
            ]
            for spin, members in rows.items()
        }
        sweeps.append(
            {
                "after": solver._propagation_impulses.numpy()[0].astype(np.float64),
                "rows": rows,
                "pairs": pairs,
                "mu": solver._propagation_row_mu.numpy()[0],
                "radius": solver._propagation_torsion_row_radius.numpy()[0],
            }
        )

    with patch.object(solver_steel.wp, "launch_tiled", record):
        solver.step(state, model.state(), model.control(), contacts, 0.0025)
    return sweeps


def _assert_torsion_budgets(test, sweeps):
    """Check every torsion impulse against its group's residual budget.

    Returns whether a torsion row and whether a group's sliding pairs reached their bounds.
    """
    test.assertTrue(sweeps)
    saturated = sliding_saturated = False
    for sweep in sweeps:
        impulses = sweep["after"]
        for spin, members in sweep["rows"].items():
            load = sum(max(float(impulses[n]), 0.0) for n in members)
            sliding = sum(float(np.hypot(impulses[n + 1], impulses[n + 2])) for n in sweep["pairs"][spin])
            budget = float(sweep["mu"][spin]) * load
            bound = float(sweep["radius"][spin]) * max(budget - sliding, 0.0)
            torsion = abs(float(impulses[spin]))
            # The sweep forms the residual budget in float32, where budget - sliding cancels.
            slack = 1.0e-5 * budget + 1.0e-12
            test.assertLessEqual(sliding, budget + slack, f"group {spin} slides past its load")
            test.assertLessEqual(
                torsion, bound + float(sweep["radius"][spin]) * slack, f"row {spin} exceeds its residual budget"
            )
            saturated |= torsion > 0.99 * bound > 0.0
            sliding_saturated |= sliding > 0.99 * budget > 0.0
    return saturated, sliding_saturated


def _turntable(device, response, steps=8):
    """Return the spin rate [rad/s] of a free-D6 sphere centered on a kinematic platform spinning at 5 rad/s."""
    b = newton.ModelBuilder()
    platform = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
    b.add_shape_box(platform, hx=0.1, hy=0.1, hz=0.01, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0))
    pose = wp.transform(wp.vec3(0.0, 0.0, 0.0299), wp.quat_identity())
    sphere = b.add_link(xform=pose, mass=0.3, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 3e-4))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    b.add_articulation([b.add_joint_d6(-1, sphere, parent_xform=pose, linear_axes=axes, angular_axes=axes)])
    b.add_shape_sphere(sphere, radius=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=0))
    model = b.finalize(device=device)
    flags = model.body_flags.numpy()
    flags[platform] |= int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    set_torsion_radius(model, 0.01)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        iterations=64,
        velocity_iterations=0,
        baumgarte=0.05,
        cfm=0.0,
        friction_anchor_beta=0.0,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        contact_torsion_device=True,
    )
    pipeline = newton.CollisionPipeline(model, reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64)
    contacts = pipeline.contacts()
    state, output = model.state(), model.state()
    velocity = np.zeros(model.joint_dof_count, np.float32)
    velocity[5] = 5.0
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 0.0025)
        state, output = output, state
        if response == "propagation":
            # The platform has no response DOFs, so no propagation row, its torsion row included, may name it.
            rows = int(solver._propagation_constraint_count.numpy()[0])
            for bodies in (solver._propagation_body_a, solver._propagation_body_b):
                assert platform not in bodies.numpy()[0, :rows], "a propagation row names the kinematic platform"
    return float(state.joint_qd.numpy()[11])


def _mixed_scene(device):
    """Return two worlds, each with a same-articulation link pair, a spinning D6 sphere and a free sphere on the ground.

    Also returns the D6 joint's first coordinate and DOF index in world 0, which coincide for D6 joints.
    The link pair's contact stays on dense rows; the D6 sphere's hosts a propagation torsion row; the free
    sphere's reserves no spare.
    """
    scene = newton.ModelBuilder()
    SolverSteel.register_custom_attributes(scene)
    for _ in range(2):
        world = newton.ModelBuilder()
        SolverSteel.register_custom_attributes(world)
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
        base = world.add_link()
        world.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        joints = [
            world.add_joint_revolute(
                parent=-1,
                child=base,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        ]
        for side in (1.0, -1.0):
            link = world.add_link()
            world.add_shape_box(link, hx=0.12, hy=0.03, hz=0.04, cfg=cfg)
            joints.append(
                world.add_joint_revolute(
                    parent=base,
                    child=link,
                    axis=wp.vec3(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(wp.vec3(0.06, side * 0.028, 0.0), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                )
            )
        world.add_articulation(joints)
        pose = wp.transform(wp.vec3(0.3, 0.0, 0.0199), wp.quat_identity())
        sphere = world.add_link(xform=pose, mass=0.3, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 3e-4))
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        world.add_articulation([world.add_joint_d6(-1, sphere, parent_xform=pose, linear_axes=axes, angular_axes=axes)])
        world.add_shape_sphere(sphere, radius=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5))
        free = world.add_body(xform=wp.transform(wp.vec3(-0.3, 0.0, 0.0199), wp.quat_identity()))
        world.add_shape_sphere(free, radius=0.02, cfg=cfg)
        scene.add_world(world)
    scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0))
    model = scene.finalize(device=device)
    set_torsion_radius(model, 0.01)
    d6 = int(np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.D6))[0])
    assert int(model.joint_q_start.numpy()[d6]) == int(model.joint_qd_start.numpy()[d6])
    return model, int(model.joint_qd_start.numpy()[d6])


def _family_census(solver, contacts):
    """Count each world's dense torsion rows and propagation torsion groups, and check the free-body contacts."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = solver.model.shape_body.numpy()
    body_art = solver._body_to_articulation.numpy()
    is_free = np.asarray(solver._model_plan.is_free_rigid)
    shapes = np.stack([contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]])
    path = solver._contact_path.numpy()[:count]
    slot = solver._contact_slot.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]

    def free_pair(c):
        bodies = [shape_body[s] if s >= 0 else -1 for s in shapes[:, c]]
        return all(b < 0 or is_free[body_art[b]] != 0 for b in bodies)

    free = [c for c in range(count) if slot[c] >= 0 and free_pair(c)]
    dense_count = solver._constraint_count.numpy()
    dense_type = solver._row_type.numpy()
    return {
        "propagation_groups": [len(_propagation_torsion_rows(solver, w)) for w in range(solver._world_count)],
        "dense_torsion": [
            int(np.count_nonzero(dense_type[w, : dense_count[w]] == PGS_CONSTRAINT_TYPE_TORSION))
            for w in range(solver._world_count)
        ],
        "free_body_rows": bool(free) and all(path[c] == 2 and needed[c] in (1, 3) for c in free),
    }


def _permute_contacts(contacts, order):
    """Reorder the first ``len(order)`` entries of every per-contact rigid array of ``contacts``."""
    for name, value in vars(contacts).items():
        if not name.startswith("rigid_contact_") or not isinstance(value, wp.array):
            continue
        if value.shape[0] != contacts.rigid_contact_max or name == "rigid_contact_count":
            continue
        values = value.numpy()
        values[: len(order)] = values[order]
        value.assign(values)


def _spinner(device, response, iterations, radius, *, spin, steps=1, graph=False):
    """Press three spheres fixed to one spinning slider onto the ground and return its joint velocity.

    The feet share the slider's two DOFs, so the propagation sweep splits their mass; the middle foot
    sits on the spin axis, where only its torsion row resists the spin. A sequence ``spin`` replicates
    one world per rate.
    """
    spins = np.atleast_1d(np.asarray(spin, dtype=np.float32))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.5
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    z = newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0))
    joints = [builder.add_joint_d6(-1, slider, linear_axes=[z], angular_axes=[z])]
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
    if len(spins) > 1:
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(builder, len(spins), spacing=(1.0, 0.0, 0.0))
        builder = scene
    model = builder.finalize(device=device)
    set_torsion_radius(model, radius)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        friction_anchor_beta=0.0,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        iterations=iterations,
        velocity_iterations=0,
        baumgarte=0.0,
        cfm=0.0,
        contact_regularization=3.0,
        contact_torsion_device=True,
    )
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.stack([np.full_like(spins, -1.0), spins], axis=1).reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    if graph:
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        state, output = output, state
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(state, contacts)
            solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
            pipeline.collide(output, contacts)
            solver.step(output, state, model.control(), contacts, 1.0 / 240.0)
        for _ in range((steps - 1) // 2):
            wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        return state.joint_qd.numpy(), solver
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        state, output = output, state
    return state.joint_qd.numpy(), solver


if __name__ == "__main__":
    unittest.main()
