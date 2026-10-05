# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate opt-in, load-bounded spin resistance on articulated contacts."""

import inspect
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.feather_pgs.contact_torsion import _contact_groups, _group_budget
from newton._src.solvers.feather_pgs.friction_patches import link_patch_rows
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_TORSION
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_friction_patches import _patch_fixture
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

# Persistent friction patches (the default friction law).
PATCH_OPTIONS = {"friction_anchor_beta": 0.2}
# SAT box-box manifolds are a separate collision feature; their subcases run where it exists.
_HAS_BOX_BOX_SAT = "box_box_sat" in inspect.signature(newton.CollisionPipeline.__init__).parameters


def fixture(
    radius=None,
    *,
    device="cuda:0",
    mu=0.5,
    closing=0.1,
    spin=1.0,
    sliding=0.0,
    sat=False,
    separation=0.0,
    center_only=False,
    center_count=1,
    restitution=0.0,
    dt=0.0025,
    row_limit=None,
    model_arrays=None,
    **solver_overrides,
):
    """Solve two equal articulated rectangular pads touching face to face.

    ``model_arrays`` maps model attribute names to values assigned before the solver is built.
    """
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * (0.025 + separation / 2)), wp.quat_identity())
        body = b.add_link(
            xform=pose, mass=0.3, inertia=wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        )
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        b.add_articulation([joint])
        b.add_shape_box(
            body,
            hx=0.02,
            hy=0.015,
            hz=0.025,
            cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=restitution),
        )
    model = b.finalize(device=device)
    for name, values in (model_arrays or {}).items():
        getattr(model, name).assign(values)
    a, z = model.state(), model.state()
    a.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline_options = {"box_box_sat": True} if sat else {}
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest",
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        **pipeline_options,
    )
    contacts = pipeline.contacts()
    pipeline.collide(a, contacts)
    if center_only and int(contacts.rigid_contact_count.numpy()[0]) > 0:
        # Deliberately represent the known planar disk by one central witness:
        # this isolates the new material spin law from native point lever arms.
        shapes = [contacts.rigid_contact_shape0.numpy()[0], contacts.rigid_contact_shape1.numpy()[0]]
        poses = a.body_q.numpy()
        for side, name in enumerate(("rigid_contact_point0", "rigid_contact_point1")):
            values = getattr(contacts, name).numpy()
            body = model.shape_body.numpy()[shapes[side]]
            values[:center_count] = -poses[body, :3]
            values[:center_count, 2] += np.sign(poses[body, 2]) * separation / 2
            getattr(contacts, name).assign(values)
        contacts.rigid_contact_count.assign(np.array([center_count], np.int32))
    kwargs = {
        # Keep this material-law control on the same point-friction baseline
        # whether the optional spin radius is zero or positive.
        "friction_anchor_beta": 0.0,
        "pgs_iterations": 128,
        "pgs_velocity_iterations": 0,
        "pgs_beta": 0.05,
        "pgs_cfm": 0,
        "pgs_contact_regularization": 0,
        "enable_joint_velocity_limits": True,
        "pgs_warmstart": False,
        "dense_max_constraints": 64,
        "mf_max_constraints": 16,
    }
    if radius is not None:
        kwargs["contact_torsion_radius"] = radius
    kwargs.update(solver_overrides)
    if kwargs["friction_anchor_beta"] is None:
        # Use the constructor default (persistent friction patches).
        del kwargs["friction_anchor_beta"]
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", **kwargs)
    solver.rigid_body_angular_damping.zero_()
    if row_limit is not None:
        solver.dense_max_constraints = row_limit
    solver.step(a, z, model.control(), contacts, dt)
    result = {name: getattr(z, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    result.update(
        {
            name: getattr(solver, name).numpy()
            for name in ("impulses", "row_type", "row_parent", "row_mu", "J_world", "Y_world", "v_hat", "v_out")
        }
    )
    result["count"] = solver.constraint_count.numpy()
    return result, solver, model, a, contacts


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
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        contact_torsion_radius=radius,
        pgs_iterations=64,
        pgs_velocity_iterations=0,
        pgs_beta=0.05,
        pgs_cfm=0,
        pgs_contact_regularization=0,
        pgs_warmstart=False,
        dense_max_constraints=64,
        mf_max_constraints=16,
        **PATCH_OPTIONS,
    )
    solver.rigid_body_angular_damping.zero_()
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
        contact_path=wp.zeros(n, dtype=int, device=device),
        contact_slot=wp.array(slots, dtype=int, device=device),
        contact_world=wp.zeros(n, dtype=int, device=device),
        contact_slots_needed=wp.array(slots_needed, dtype=int, device=device),
        row_type=wp.array([row_type], dtype=int, device=device),
        row_parent=wp.array([parents], dtype=int, device=device),
        row_mu=wp.array([mu], dtype=float, device=device),
        constraint_count=wp.array([len(row_type)], dtype=int, device=device),
        _friction_anchors_enabled=patches_enabled,
        _friction_patches=patches,
        _contact_torsion_shape_set=None,
    )
    if patches_enabled:
        wp.launch(
            link_patch_rows,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                patches.view,
                solver.contact_world,
                solver.contact_slot,
                solver.contact_path,
                solver.contact_slots_needed,
                0,
                solver.row_parent,
                solver.row_mu,
            ],
            device=device,
        )
    return solver, state, contacts


# ---------------------------------------------------------------------------
# Host grouping of persistent patch regions (no solver step)
# ---------------------------------------------------------------------------


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
    np.testing.assert_array_equal(solver.row_parent.numpy()[0], [3, 0, 0, 4, 0, 4, 4])
    np.testing.assert_allclose(solver.row_mu.numpy()[0], [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25])
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    test.assertEqual(sorted(w.slot for w in groups[0]), [0, 3, 4])
    coefficient, anchors, touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
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
    np.testing.assert_array_equal(solver.row_parent.numpy()[0], [3, 0, 0, 0])
    groups = _contact_groups(solver, state, contacts)
    test.assertEqual(len(groups), 1)
    test.assertEqual(sorted(w.slot for w in groups[0]), [0, 3])
    coefficient, anchors, _touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
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
    coefficient, anchors, _touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
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
    test.assertIsNone(_group_budget(solver, groups[0], solver.row_mu.numpy()))


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
    parents = solver.row_parent.numpy()
    parents[0, 3] = -1
    solver.row_parent.assign(parents)
    with test.assertRaisesRegex(RuntimeError, "load ring"):
        _contact_groups(solver, state, contacts)


# ---------------------------------------------------------------------------
# Solver behavior (CUDA)
# ---------------------------------------------------------------------------


def test_default_off_equivalent(test, device):
    """Keep an empty shape selection exactly equivalent to the zero-radius output."""
    reference, *_ = fixture(0.0, device=device)
    actual, *_ = fixture(0.01, device=device, contact_torsion_shape_indices=())
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
        test.assertLessEqual(int(solver.constraint_count.numpy().max()), solver.dense_max_constraints)
    for velocity in velocities[1:]:
        np.testing.assert_allclose(velocity, velocities[0], atol=2e-5)
    baseline, *_ = fixture(0.0, device=device, center_only=True)
    with test.assertRaisesRegex(RuntimeError, "capacity exceeded"):
        fixture(0.01, device=device, center_only=True, row_limit=int(baseline["count"][0]))


def test_public_shape_selection(test, device):
    """Resolve public index and regex scopes without private model patches."""
    baseline, *_ = fixture(0.0, device=device, center_only=True)
    excluded, *_ = fixture(0.01, device=device, center_only=True, contact_torsion_shape_indices=())
    np.testing.assert_array_equal(baseline["body_qd"], excluded["body_qd"])
    for selection in ({"contact_torsion_shape_indices": (0,)}, {"contact_torsion_shape_patterns": (".*",)}):
        result, solver, *_ = fixture(0.01, device=device, center_only=True, **selection)
        test.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        test.assertEqual(solver._torsion_stats["rows"], 1)


def test_invalid_input_and_unsupported_modes(test, device):
    """Reject invalid scopes and modes rather than silently ignoring spin friction."""
    for radius in (-1.0, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            fixture(radius, device=device)
    for options in (
        {"contact_torsion_shape_indices": (-1,)},
        {"contact_torsion_shape_indices": (True,)},
        {"contact_torsion_shape_patterns": ("[",)},
        {"contact_torsion_shape_patterns": ("missing-label",)},
        {"contact_torsion_shape_patterns": "box"},
        {"contact_torsion_shape_indices": (), "contact_torsion_shape_patterns": ()},
        {"pgs_warmstart": True},
    ):
        with test.subTest(options=options), test.assertRaises(ValueError):
            fixture(0.01, device=device, **options)


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


# ---------------------------------------------------------------------------
# Torsion with persistent friction patches (CUDA)
# ---------------------------------------------------------------------------


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
    test_public_shape_selection,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
