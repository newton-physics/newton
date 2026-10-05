# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate opt-in, load-bounded spin resistance on articulated contacts."""

import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.feather_pgs.contact_torsion import _contact_groups, _group_budget
from newton._src.solvers.feather_pgs.friction_patches import link_patch_rows
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_friction_patches import _patch_fixture

# Persistent patches select their own friction locations; shared-anchor flags only warn.
PATCH_OPTIONS = {"friction_anchor_beta": 0.2, "contact_shared_anchor": False, "contact_friction_shared_anchor": False}


def fixture(
    radius=None,
    *,
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
    model = b.finalize(device="cuda:0")
    for name, values in (model_arrays or {}).items():
        getattr(model, name).assign(values)
    a, z = model.state(), model.state()
    a.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest",
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        box_box_sat=sat,
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
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "pgs_iterations": 128,
        "pgs_velocity_iterations": 0,
        "pgs_beta": 0.05,
        "pgs_cfm": 0,
        "pgs_contact_regularization": 0,
        "contact_friction_position_iterations": -1,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "enable_joint_velocity_limits": True,
        "pgs_warmstart": False,
        "angular_damping": 0,
        "dense_max_constraints": 64,
        "mf_max_constraints": 16,
        "row_watermark": True,
    }
    if radius is not None:
        kwargs["contact_torsion_radius"] = radius
    kwargs.update(solver_overrides)
    solver = SolverFeatherPGS(model, **kwargs)
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


def held_box(radius, torque, *, steps=20, dt=0.0025, mu=0.5, mass=0.3, half=0.005):
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
    model = b.finalize(device="cuda:0")
    solver = SolverFeatherPGS(
        model,
        contact_torsion_radius=radius,
        pgs_mode="matrix_free",
        articulated_contact_response="immediate",
        pgs_iterations=64,
        pgs_velocity_iterations=0,
        pgs_beta=0.05,
        pgs_cfm=0,
        pgs_contact_regularization=0,
        pgs_warmstart=False,
        angular_damping=0,
        dense_max_constraints=64,
        mf_max_constraints=16,
        **PATCH_OPTIONS,
    )
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


def patch_rows(points, *, slots, slots_needed, row_type, parents, mu, patches_enabled=True, shape_type=GeoType.BOX):
    """Link one CPU patch region into dense rows and return the torsion host view of it.

    Rows follow the dense builder's layout: every contact keeps its normal, and only
    anchors receive an adjacent tangent pair. ``link_patch_rows`` then closes the
    normal-parent load ring and divides the anchors' friction coefficient.
    """
    model, state, contacts, patches = _patch_fixture(points)
    model.shape_type = wp.full(3, int(shape_type), dtype=int, device="cpu")
    contacts.rigid_contact_max = len(points)
    contacts.rigid_contact_stiffness = None
    n = len(points)
    solver = SimpleNamespace(
        model=model,
        contact_path=wp.zeros(n, dtype=int, device="cpu"),
        contact_slot=wp.array(slots, dtype=int, device="cpu"),
        contact_world=wp.zeros(n, dtype=int, device="cpu"),
        contact_slots_needed=wp.array(slots_needed, dtype=int, device="cpu"),
        row_type=wp.array([row_type], dtype=int, device="cpu"),
        row_parent=wp.array([parents], dtype=int, device="cpu"),
        row_mu=wp.array([mu], dtype=float, device="cpu"),
        constraint_count=wp.array([len(row_type)], dtype=int, device="cpu"),
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
            device="cpu",
        )
    return solver, state, contacts


class TestContactTorsionPatchGrouping(unittest.TestCase):
    """Follow persistent patch regions on the host without a CUDA solve."""

    def test_two_anchor_region_pools_the_undivided_coefficient(self):
        """Budget one region by the full coefficient on every ring normal, consuming only anchor rows."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]],
            slots=[0, 3, 4],
            slots_needed=[3, 1, 3],
            row_type=[0, 2, 2, 0, 0, 2, 2],
            parents=[-1, 0, 0, -1, -1, 4, 4],
            mu=[0.5] * 7,
        )
        np.testing.assert_array_equal(solver.row_parent.numpy()[0], [3, 0, 0, 4, 0, 4, 4])
        np.testing.assert_allclose(solver.row_mu.numpy()[0], [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25])
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(w.slot for w in groups[0]), [0, 3, 4])
        coefficient, anchors, touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual(sorted(w.slot for w in anchors), [0, 4])
        self.assertEqual(len(touching), 3)

    def test_single_anchor_region_keeps_the_coefficient(self):
        """Pool coincident normals behind one anchor without scaling the coefficient."""
        solver, state, contacts = patch_rows(
            [[0, 0, 0], [0, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 1],
            row_type=[0, 2, 2, 0],
            parents=[-1, 0, 0, -1],
            mu=[0.5] * 4,
        )
        np.testing.assert_array_equal(solver.row_parent.numpy()[0], [3, 0, 0, 0])
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(w.slot for w in groups[0]), [0, 3])
        coefficient, anchors, _touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual([w.slot for w in anchors], [0])

    def test_point_friction_grouping_is_unchanged(self):
        """Keep every point-friction witness on its own tangent pair with the row coefficient."""
        solver, state, contacts = patch_rows(
            [[0, 0, 0], [0, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
            patches_enabled=False,
        )
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        coefficient, anchors, _touching = _group_budget(solver, groups[0], solver.row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual(sorted(w.slot for w in anchors), [0, 3])

    def test_region_without_anchor_rows_carries_no_spin(self):
        """Skip a region whose members all lost their tangent rows to friction filters."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 1],
            slots_needed=[1, 1],
            row_type=[0, 0],
            parents=[-1, -1],
            mu=[0.5] * 2,
        )
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertIsNone(_group_budget(solver, groups[0], solver.row_mu.numpy()))

    def test_region_with_inadmissible_member_is_skipped(self):
        """Skip the whole region rather than budget a partial ring."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
            shape_type=GeoType.MESH,
        )
        self.assertEqual(_contact_groups(solver, state, contacts), [])

    def test_broken_load_ring_is_rejected(self):
        """Refuse a region whose normal-parent ring does not close over its own rows."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
        )
        parents = solver.row_parent.numpy()
        parents[0, 3] = -1
        solver.row_parent.assign(parents)
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            _contact_groups(solver, state, contacts)


@unittest.skipUnless(wp.get_device().is_cuda, "Contact torsion currently requires CUDA")
class TestContactTorsion(unittest.TestCase):
    """Exercise an explicitly assumed uniform-disk effective spin radius."""

    def test_default_off_equivalent(self):
        """Keep the zero-radius option exactly equivalent to prefeature output."""
        reference, *_ = fixture(0.0)
        actual, *_ = fixture(0.01, contact_torsion_shape_indices=())
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)

    def test_spin_stops_below_bound(self):
        """Remove spin below the prescribed disk's Coulomb torque capacity."""
        baseline, *_ = fixture(0.0, spin=1.0, center_only=True)
        self.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
        result, solver, *_ = fixture(0.01, spin=1.0, center_only=True)
        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        self.assertGreater(solver._torsion_stats["rows"], 0)

    def test_zero_friction_and_zero_load(self):
        """Apply no torsion when friction or compressive normal load vanishes."""
        for options in ({"mu": 0.0}, {"closing": 0.0}, {"separation": 0.004}):
            actual, solver, *_ = fixture(0.01, **options)
            if "closing" in options:
                self.assertGreater(solver._torsion_stats["rows"], 0)
            else:
                self.assertEqual(solver._torsion_stats["rows"], 0)
            active = actual["row_type"] == 7
            self.assertLess(np.abs(actual["impulses"][active]).max(initial=0), 1e-9)

    def test_joint_response_and_coupled_budget(self):
        """Respect Coulomb sharing, articulated response and kinetic-energy bounds."""
        for sat in (False, True):
            for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0), (100.0, 10.0)):
                actual, solver, _model, initial, _ = fixture(
                    0.01, spin=spin, sliding=sliding, sat=sat, center_only=not sat
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

                self.assertLessEqual(energy(v1), energy(v0) + 2e-6)
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
                    self.assertLessEqual(used, budget + 2e-6)

    def test_witness_count_does_not_multiply_torque(self):
        """Share one footprint budget across repeated normal quadrature witnesses."""
        outputs = []
        for count in (1, 2, 4):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, center_count=count)
            self.assertEqual(solver._torsion_stats["rows"], 1)
            outputs.append(result["body_qd"])
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], atol=2e-5)

    def test_release_and_reset_carry_no_torque(self):
        """Forget spin impulse on contact loss and reproduce cold state after reset."""
        expected, solver, model, initial, contacts = fixture(0.01, spin=100.0)
        output = model.state()
        count = contacts.rigid_contact_count.numpy()
        contacts.rigid_contact_count.zero_()
        solver.step(initial, output, model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)
        contacts.rigid_contact_count.assign(count)
        solver.reset(initial)
        solver.step(initial, output, model.control(), contacts, 0.0025)
        np.testing.assert_allclose(output.body_qd.numpy(), expected["body_qd"], atol=2e-5)

    def test_timestep_and_capacity(self):
        """Keep impulse-level overload behavior across dt and reject insufficient rows."""
        velocities = []
        for dt in (0.00125, 0.0025, 0.005):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, dt=dt)
            velocities.append(result["v_out"])
            self.assertEqual(int(solver._row_dropped_dense_high_water.numpy()[0]), 0)
            self.assertLessEqual(int(solver.constraint_count.numpy().max()), solver.dense_max_constraints)
        for velocity in velocities[1:]:
            np.testing.assert_allclose(velocity, velocities[0], atol=2e-5)
        baseline, *_ = fixture(0.0, center_only=True)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            fixture(0.01, center_only=True, row_limit=int(baseline["count"][0]))

    def test_public_shape_selection(self):
        """Resolve public index and regex scopes without private model patches."""
        baseline, *_ = fixture(0.0, center_only=True)
        excluded, *_ = fixture(0.01, center_only=True, contact_torsion_shape_indices=())
        np.testing.assert_array_equal(baseline["body_qd"], excluded["body_qd"])
        for selection in ({"contact_torsion_shape_indices": (0,)}, {"contact_torsion_shape_patterns": (".*",)}):
            result, solver, *_ = fixture(0.01, center_only=True, **selection)
            self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
            self.assertEqual(solver._torsion_stats["rows"], 1)

    def test_invalid_input_and_unsupported_modes(self):
        """Reject invalid scopes and modes rather than silently ignoring spin friction."""
        for radius in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fixture(radius)
        for options in (
            {"contact_torsion_shape_indices": (-1,)},
            {"contact_torsion_shape_indices": (True,)},
            {"contact_torsion_shape_patterns": ("[",)},
            {"contact_torsion_shape_patterns": ("missing-label",)},
            {"contact_torsion_shape_indices": (), "contact_torsion_shape_patterns": ()},
            {"pgs_warmstart": True},
            {"pgs_velocity_iterations": 1, "enable_bilateral_preelimination": True},
            {"articulated_contact_response": "propagation"},
            {"friction_mode": "bisection"},
            {"pgs_debug": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fixture(0.01, **options)
        with patch.dict("os.environ", {"IL_NEWTON_FPGS_MF_WARMSTART": "1"}):
            with self.assertRaises(ValueError):
                fixture(0.01)

    def test_compliance_combination_rejected_at_construction(self):
        """Fail before stepping when both experimental responses are requested."""
        with self.assertRaisesRegex(ValueError, "contact_torsion_radius.*contact_compliance"):
            fixture(
                0.01,
                contact_compliance=True,
                enable_restitution=False,
                contact_shared_anchor=False,
                contact_friction_shared_anchor=False,
            )

    def test_capture_is_explicitly_rejected(self):
        """Reject capture before host contact grouping is attempted."""
        _, solver, model, initial, contacts = fixture(0.01)
        output = model.state()
        with self.assertRaisesRegex(RuntimeError, "graph capture"):
            with wp.ScopedCapture():
                solver.step(initial, output, model.control(), contacts, 0.0025)

    def test_hydro_combination_is_rejected(self):
        """Reject actual hydro contact stiffness before combining contact mechanisms."""
        _, solver, model, initial, contacts = fixture(0.01)
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion currently requires CUDA")
class TestContactTorsionWithPatches(unittest.TestCase):
    """Bound one spin row per persistent friction patch region."""

    def test_constructor_accepts_torsion_with_patches(self):
        """Build torsion on top of default and explicit patch friction without friction warnings."""
        for options in ({"friction_anchor_beta": None}, {}):
            with self.subTest(options=options), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, solver, *_ = fixture(0.01, center_only=True, **{**PATCH_OPTIONS, **options})
            self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
            self.assertTrue(solver._contact_torsion_enabled)
            self.assertTrue(solver._friction_anchors_enabled)
            self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
            self.assertEqual(solver._torsion_stats["rows"], 1)

    def test_patch_regions_pool_the_spin_budget(self):
        """Share mu times the region's pooled normal load between anchor sliding and spin."""
        for center_only, anchors in ((True, 1), (False, 2)):
            with self.subTest(anchors=anchors):
                actual, solver, *_ = fixture(
                    0.01, spin=100.0, sliding=10.0, sat=not center_only, center_only=center_only, **PATCH_OPTIONS
                )
                self.assertEqual(solver._torsion_stats["rows"], 1)
                group = solver._torsion_stats["groups"][0]
                self.assertEqual(len(group["anchor_rows"]), anchors)
                self.assertGreaterEqual(len(group["normal_rows"]), anchors)
                self.assertAlmostEqual(group["mu"], 0.5, places=6)
                self.assertAlmostEqual(float(actual["row_mu"][0, group["row"]]), 0.5, places=6)
                for slot in group["anchor_rows"]:
                    self.assertAlmostEqual(float(actual["row_mu"][0, slot + 1]), 0.5 / anchors, places=6)
                    self.assertEqual(list(actual["row_type"][0, slot + 1 : slot + 3]), [2, 2])
                for slot in set(group["normal_rows"]) - set(group["anchor_rows"]):
                    self.assertNotEqual(int(actual["row_type"][0, slot + 1]), 2)
                impulse = actual["impulses"][0, : int(actual["count"][0])]
                pooled = sum(max(float(impulse[r]), 0.0) for r in group["normal_rows"])
                used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in group["anchor_rows"])
                used += abs(float(impulse[group["row"]])) / 0.01
                self.assertGreater(pooled, 0.0)
                self.assertLessEqual(used, 0.5 * pooled + 2e-6)

    def test_single_anchor_spin_stops_below_bound(self):
        """Stop a slow spin that a lone patch anchor cannot resist on its own."""
        baseline, solver, *_ = fixture(0.0, spin=1.0, center_only=True, **PATCH_OPTIONS)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
        result, solver, *_ = fixture(0.01, spin=1.0, center_only=True, **PATCH_OPTIONS)
        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        self.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 1)

    def test_held_box_yaw_torque_threshold(self):
        """Hold a resting box below mu * N * radius and let it spin above, with patch friction."""
        radius, mu, mass = 0.05, 0.5, 0.3
        bound = mu * mass * 9.81 * radius
        held, solver = held_box(radius, 0.5 * bound, mu=mu, mass=mass)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        self.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 2)
        self.assertLess(abs(held), 1e-3)
        released, _ = held_box(radius, 2.0 * bound, mu=mu, mass=mass)
        self.assertGreater(released, 1.0)
        anchors_only, _ = held_box(0.0, 0.5 * bound, mu=mu, mass=mass)
        self.assertGreater(anchors_only, 1.0)


if __name__ == "__main__":
    unittest.main()
