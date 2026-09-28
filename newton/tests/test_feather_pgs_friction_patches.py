# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for persistent patch friction in FeatherPGS."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import (
    _FrictionPatchState,
    _pose_motion,
    finish_patch_impulses,
    link_patch_rows,
    seed_patch_impulses,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    contact_friction_residuals,
    pgs_solve_loop,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel
from newton.tests.test_feather_pgs_friction_anchors import _SQUEEZE_SOLVER, _build_v_jaws, _run_squeeze


@wp.kernel(enable_backward=False)
def _friction_residual_probe(result: wp.array[wp.vec4]):
    result[0] = contact_friction_residuals(0.0, 4.0, 0.25, 0.0, wp.vec2(-1.0, 0.0), wp.vec2(1.0, 0.0))
    result[1] = contact_friction_residuals(0.0, 4.0, 0.25, 0.0, wp.vec2(-1.0, 0.0), wp.vec2(2.5, 0.0))
    result[2] = contact_friction_residuals(2.0, 2.0, 0.5, -0.1, wp.vec2(0.0), wp.vec2(0.0))


@wp.kernel(enable_backward=False)
def _pose_motion_probe(
    before: wp.array[wp.transform],
    after: wp.array[wp.transform],
    point: wp.array[wp.vec3],
    result: wp.array[wp.vec3],
):
    i = wp.tid()
    result[i] = _pose_motion(after, before, i, point[i])


def _ground_box(device, **solver_kwargs):
    """Return a resting 1 kg box on the ground with patch friction enabled."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5))
    model = builder.finalize(device=device)
    kwargs = {
        "pgs_iterations": 32,
        "pgs_mode": "matrix_free" if model.device.is_cuda else "split",
    }
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step():
        s0, s1 = states
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 0.005)
        states.reverse()

    return model, solver, contacts, step


def _patch_fixture(
    points, *, shape0=None, normals=None, materials=(0.5, 0.5, 0.5), shape_bodies=(0, 1, 0), device="cpu"
):
    """Construct patch geometry without a collision matcher or a solver step."""
    n = len(points)
    body_count = max(shape_bodies) + 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_count=body_count,
        body_world=wp.zeros(body_count, dtype=int, device=device),
        shape_body=wp.array(shape_bodies, dtype=int, device=device),
        shape_collision_radius=wp.array([0.2] * len(shape_bodies), dtype=float, device=device),
        shape_transform=wp.array([wp.transform_identity()] * len(shape_bodies), dtype=wp.transform, device=device),
        shape_scale=wp.ones(len(shape_bodies), dtype=wp.vec3, device=device),
        shape_type=wp.zeros(len(shape_bodies), dtype=int, device=device),
        shape_source_ptr=wp.zeros(len(shape_bodies), dtype=wp.uint64, device=device),
        shape_margin=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_gap=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_is_solid=wp.ones(len(shape_bodies), dtype=bool, device=device),
        shape_material_mu=wp.array(materials, dtype=float, device=device),
    )
    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device))
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([n], dtype=int, device=device),
        rigid_contact_shape0=wp.array(shape0 if shape0 is not None else [0] * n, dtype=int, device=device),
        rigid_contact_shape1=wp.ones(n, dtype=int, device=device),
        rigid_contact_point0=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_normal=wp.array(
            normals if normals is not None else [[0.0, 0.0, -1.0]] * n, dtype=wp.vec3, device=device
        ),
        rigid_contact_margin0=wp.zeros(n, dtype=float, device=device),
        rigid_contact_margin1=wp.zeros(n, dtype=float, device=device),
    )
    patches = _FrictionPatchState(model, max(n, 8), True, wp.zeros(max(n, 8), dtype=wp.vec2, device=device))
    patches.build(model, state, contacts)
    return model, state, contacts, patches


class TestFrictionPatchHistory(unittest.TestCase):
    def test_support_edge_centers_ignore_contact_order(self):
        """Keep identical edge centers when contact ordering changes at rest."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        points = np.array(
            [[x, y, 0.0] for x in np.linspace(-0.001, 0.001, 128) for y in (-0.05, 0.05)], dtype=np.float32
        )
        expected = np.array([[0.0, -0.05, 0.0], [0.0, 0.05, 0.0]], dtype=np.float32)
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                order = np.random.default_rng(seed).permutation(len(points))
                _, _, _, patches = _patch_fixture(points[order], device=device)
                locations = patches.view.point_a.numpy()[patches.view.weight.numpy() > 0.0]
                locations = locations[np.argsort(locations[:, 1])]
                np.testing.assert_array_equal(locations, expected)

    def test_narrow_patch_preserves_footprint_reflection_symmetry(self):
        """Avoid a diagonal friction couple on a symmetric narrow contact footprint."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for angle in (0.0, 0.37, 1.1):
            axis = np.array([np.sin(angle), np.cos(angle), 0.0])
            across = np.array([np.cos(angle), -np.sin(angle), 0.0])
            points = [x * across + y * axis for x in (-0.001, 0.001) for y in (-0.025, 0.025)]
            with self.subTest(angle=angle):
                _, _, _, patches = _patch_fixture(points, device=device)
                active = patches.view.weight.numpy() > 0.0
                locations = patches.view.point_a.numpy()[active]
                self.assertEqual(len(locations), 2)
                np.testing.assert_allclose(locations @ across, 0.0, atol=1.0e-7)
                np.testing.assert_allclose(np.sort(locations @ axis), [-0.025, 0.025], atol=1.0e-7)

    def test_pose_increment_preserves_fixed_pivots_and_no_slip_rolling(self):
        """Distinguish rigid rotation from slip without querying the collision shape."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        pivot = wp.vec3(0.3, -0.2, 0.5)
        origin = wp.vec3(-0.1, 0.2, 0.3)
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1, 2, 3)), 0.7)
        before = [wp.transform(origin, wp.quat_identity()), wp.transform_identity()]
        after = [
            wp.transform(pivot + wp.quat_rotate(rotation, origin - pivot), rotation),
            wp.transform(wp.vec3(0.002, 0, 0), wp.quat_identity()),
        ]
        points = [pivot, wp.vec3(0)]
        for rate in (60, 120, 240):
            advance = 1.0 / rate
            # The rigid integrator advances translation and normalizes its
            # first-order quaternion update. Its no-slip rolling pair has
            # this relative rotation; a material chord falsely reports slip.
            angle = float(2.0 * np.arctan(0.5 * 20.0 / rate))
            before.append(wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
            after.append(wp.transform(wp.vec3(advance, 0, 0.05), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle)))
            points.append(wp.vec3(advance, 0, 0))
        result = wp.zeros(len(points), dtype=wp.vec3, device=device)
        wp.launch(
            _pose_motion_probe,
            dim=len(points),
            inputs=[
                wp.array(before, dtype=wp.transform, device=device),
                wp.array(after, dtype=wp.transform, device=device),
                wp.array(points, dtype=wp.vec3, device=device),
                result,
            ],
            device=device,
        )
        values = result.numpy()
        np.testing.assert_allclose(values[0], 0.0, atol=1.0e-7)
        np.testing.assert_allclose(values[1], [0.002, 0, 0], atol=1.0e-7)
        np.testing.assert_allclose(values[2:, :2], 0.0, atol=1.0e-7)

    def test_supported_normal_turn_preserves_friction_history(self):
        """Retain supported history during small relative rotations on either body."""
        for body in (0, 1):
            for axis, retained in ((wp.vec3(0, 1, 0), True), (wp.vec3(0, 0, 1), True)):
                with self.subTest(body=body, axis=axis):
                    model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
                    types = model.shape_type.numpy()
                    types[body] = int(newton.GeoType.SPHERE)
                    model.shape_type.assign(types)
                    patches.build(model, state, contacts)
                    patches.store(state)
                    patches.build(model, state, contacts)
                    self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
                    transforms = [wp.transform_identity(), wp.transform_identity()]
                    transforms[body] = wp.transform(wp.vec3(0), wp.quat_from_axis_angle(axis, 0.01))
                    state.body_q.assign(transforms)
                    patches.build(model, state, contacts)
                    self.assertEqual(int(patches.current.source.numpy()[0]) >= 0, retained)

    def test_normal_turn_transports_tangent_error_without_decay(self):
        """Preserve spring length during supported rocking without additional slip."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        patches.store(state)
        pivot = wp.vec3(0.0005, 0, 0)
        local_point = wp.vec3(-0.0005, 0, 0)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([local_point])
        contacts.rigid_contact_point1.assign([pivot])
        patches.build(model, state, contacts)
        for step in range(100):
            patches.store(state)
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.025 if step % 2 else -0.025)
            state.body_q.assign(
                [wp.transform(pivot - wp.quat_rotate(rotation, local_point), rotation), wp.transform_identity()]
            )
            patches.build(model, state, contacts)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_shared_pad_randomization_preserves_material_regions(self):
        """Pool one sampled pad material while keeping genuinely different coefficients separate."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]], shape0=[0, 2])
        for coefficient in np.random.default_rng(42).uniform(0.1, 1.0, 3):
            model.shape_material_mu.assign([coefficient, 0.5, coefficient])
            patches.update_geometry(model)
            patches.build(model, state, contacts)
            self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
            np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])
            patches.store(state)
        model.shape_material_mu.assign([0.2, 0.5, 0.8])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_pose_history_covers_bodies_beyond_contact_capacity(self):
        """Capture every body pose even when the contact buffer is smaller."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(9, 10, 9))
        transforms = state.body_q.numpy()
        transforms[10, 0] = 0.002
        state.body_q.assign(transforms)
        patches.build(model, state, contacts)
        patches.store(state)
        transforms[10, 0] = 0.003
        state.body_q.assign(transforms)
        patches.build(model, state, contacts)
        self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
        self.assertAlmostEqual(float(np.linalg.norm(patches.view.phi.numpy()[0])), 0.001, delta=1.0e-7)

    def test_rocking_face_retires_the_lifted_anchor(self):
        """Keep the supported edge's history and replace the opposite anchor after rocking."""
        points = [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]]
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        pivot = wp.vec3(-0.1, 0, 0)
        state.body_q.assign([wp.transform(pivot - wp.quat_rotate(rotation, pivot), rotation), wp.transform_identity()])
        contacts.rigid_contact_count.assign([2])
        edge = [points[0], points[2], points[0], points[2]]
        contacts.rigid_contact_point0.assign(edge)
        contacts.rigid_contact_point1.assign(edge)
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-7)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy()[active] >= 0), 1)

    def test_unloading_penetration_keeps_supported_anchors(self):
        """Retain history while decompression leaves the actual surfaces in contact."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_point0.assign([[-0.1, 0, -0.005], [0.1, 0, -0.005]])
        patches.build(model, state, contacts)
        patches.store(state)
        state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.003), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2)

    def test_support_witnesses_use_existing_contact_gap_limits(self):
        """Keep supported history within the contact envelope and respect tighter solver gates."""
        for limits, carried in (({}, True), ({"contact_gap_gate": 0.0001}, False), ({"friction_gap": 0.0001}, True)):
            with self.subTest(limits=limits):
                model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
                model.shape_gap.fill_(0.0005)
                patches.store(state)
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0002), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0004], [0.1, 0, -0.0004]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2 if carried else 0)
                # A fresh sample on the supported region cannot retain history
                # whose old footprint has lifted beyond the shape gap envelope.
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0012), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0014], [0.1, 0, -0.0014]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 0)

    def test_geometry_edits_retire_only_affected_history(self):
        """Invalidate edited surfaces across carrier seams while preserving unrelated patches."""
        for field in ("shape_transform", "shape_scale", "shape_margin", "shape_source_ptr"):
            for static in (False, True):
                with self.subTest(field=field, static=static):
                    model, state, contacts, patches = _patch_fixture(
                        [[0, 0, 0], [0, 0, 0]],
                        shape0=[0, 3],
                        shape_bodies=(0, -1 if static else 1, 0, 2, 3),
                        materials=(0.5,) * 5,
                    )
                    contacts.rigid_contact_shape1.assign([1, 4])
                    patches.build(model, state, contacts)
                    patches.store(state)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [1, 1])
                    # Static geometry is identified by shape. A dynamic edit on
                    # shape 2 must also retire shape 0's anchor on the same body.
                    edited = 1 if static else 2
                    values = getattr(model, field).numpy()
                    if field == "shape_transform":
                        values[edited, 0] += 1
                    else:
                        values[edited] += 1
                    getattr(model, field).assign(values)
                    patches.update_geometry(model)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [0, 1])

    def test_geometry_updates_allow_contacts_without_a_shape(self):
        """Preserve implicit static contacts when an unrelated shape changes."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(0, 1, 1))
        contacts.rigid_contact_shape1.assign([-1])
        patches.build(model, state, contacts)
        patches.store(state)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)
        # Editing the last shape must not alias the contact's -1 sentinel.
        transforms = model.shape_transform.numpy()
        transforms[-1, 0] = 1.0
        model.shape_transform.assign(transforms)
        patches.update_geometry(model)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)

    def test_reused_contacts_keep_anchors_across_substeps(self):
        """Accumulate slip across substeps regardless of stale collision match indices."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_match_index = wp.array([1, 0], dtype=int, device=model.device)
        active = patches.current.valid.numpy() != 0
        patches.store(state)
        for substep in range(1, 4):
            displacement = substep * 0.001
            state.body_q.assign(
                [wp.transform(wp.vec3(displacement, 0, 0), wp.quat_identity()), wp.transform_identity()]
            )
            patches.build(model, state, contacts)
            active = patches.current.valid.numpy() != 0
            np.testing.assert_allclose(
                np.linalg.norm(patches.view.phi.numpy()[active], axis=1), displacement, atol=1.0e-7
            )
            self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))
            patches.store(state)

    def test_connected_convex_chain_forms_one_region(self):
        """Join a chain into one friction region even when the two end shapes do not overlap."""
        model, state, contacts, patches = _patch_fixture(
            [[-0.15, 0, 0], [0.15, 0, 0], [-0.05, 0, 0], [0.05, 0, 0]],
            shape0=[0, 4, 2, 3],
            shape_bodies=(0, 1, 0, 0, 0),
            materials=(0.5,) * 5,
        )
        model.shape_transform.assign(
            [wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()) for x in (-0.15, 0, -0.05, 0.05, 0.15)]
        )
        model.shape_collision_radius.assign([0.06, 0.3, 0.06, 0.06, 0.06])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:4], [0.5, 0.5, 0, 0])

    def test_disconnected_convex_shapes_form_separate_regions(self):
        """Keep two separated pads on one body from pooling friction through empty space."""
        model, state, contacts, patches = _patch_fixture([[-0.15, 0, 0], [0.15, 0, 0]], shape0=[0, 2])
        model.shape_transform.assign(
            [
                wp.transform(wp.vec3(-0.15, 0, 0), wp.quat_identity()),
                wp.transform_identity(),
                wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity()),
            ]
        )
        model.shape_collision_radius.assign([0.03, 0.2, 0.03])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_residuals_use_patch_support_and_keep_normal_terms_local(self):
        """Distinguish supported friction, a cone violation, and normal error in the residual diagnostics."""
        residuals = wp.zeros(3, dtype=wp.vec4, device="cpu")
        wp.launch(_friction_residual_probe, dim=1, inputs=[residuals], device="cpu")
        np.testing.assert_allclose(residuals.numpy(), [[0, 0, 0, 0], [1.5, 1.5, 0, 1.5], [0, 0.2, 0.1, 0]], atol=1e-7)

    def test_incomplete_frame_cannot_create_anchor_history(self):
        """Reject an overflowing contact frame before selecting or carrying anchors."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_count.assign([3])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.valid.numpy()), 0)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 0)

    def test_reversed_contact_orientation_uses_same_patch(self):
        """Give a body pair one identity regardless of collision shape ordering."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 0])
        contacts.rigid_contact_normal.assign([[0, 0, -1], [0, 0, 1]])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])

    def test_saturation_only_releases_an_anchor_with_sliding_motion(self):
        """Keep history under impending slip, and release it under motion against saturated friction."""
        _, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        zeros = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[0, 0, 0]], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.5, 0.5]], dtype=float, device="cpu")
        impulses = wp.array([[2, 1, 0]], dtype=float, device="cpu")
        velocity = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        args = [
            contacts.rigid_contact_count,
            patches.current,
            state.body_q,
            state.body_q,
            velocity,
            wp.zeros(2, dtype=wp.vec3, device="cpu"),
            zeros,
            zeros,
            zeros,
            lengths,
            0,
            parents,
            mu,
            impulses,
            0.005,
        ]
        impulses.zero_()
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 0, "unloaded speculative contacts must not accrue stiction")
        patches.current.valid.fill_(1)
        impulses.assign([[2, 1, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 1)
        velocity.assign([[0, -0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 0)

    def test_convex_shapes_share_one_body_patch(self):
        """Pool friction load across adjacent convex pieces with compatible materials."""
        _, _, _, patches = _patch_fixture(
            [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], shape0=[0, 2, 0, 2]
        )
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 2)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        self.assertAlmostEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_material_normal_and_disconnected_regions_stay_separate(self):
        """Keep friction from pooling across incompatible or spatially separate regions."""
        cases = (
            {"points": [[0, 0, 0], [0.01, 0, 0]], "shape0": [0, 2], "materials": (0.5, 0.5, 0.8)},
            {"points": [[0, 0, 0], [0.01, 0, 0]], "normals": [[0, 0, -1], [0, -1, 0]]},
            {"points": [[0, 0, 0], [0, 0, 0.05]]},
            {"points": [[0, 0, 0], [1, 0, 0]]},
        )
        for case in cases:
            with self.subTest(case=case):
                _, _, _, patches = _patch_fixture(**case)
                self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
                np.testing.assert_array_equal(patches.view.weight.numpy()[:2], [1, 1])

    def test_contact_churn_preserves_error_at_current_points(self):
        """Preserve a sticking region's history even when every contact sample changes."""
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        points = points[[3, 1, 0, 2]] * 0.98
        contacts.rigid_contact_point0.assign(points)
        contacts.rigid_contact_point1.assign(points)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        np.testing.assert_allclose(patches.view.point_a.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(patches.view.point_b.numpy()[active], patches.current.center.numpy()[active])
        np.testing.assert_allclose(np.linalg.norm(patches.view.phi.numpy()[active], axis=1), 0.001, atol=1.0e-7)
        self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))

    def test_contact_migration_transports_twist_to_new_lever_arms(self):
        """Preserve a region's twist rather than copying error from old lever arms."""
        points = np.array([[-0.1, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.01)
        state.body_q.assign([wp.transform(wp.vec3(0), rotation), wp.transform_identity()])
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        patches.build(model, state, contacts)
        before = patches.view.phi.numpy()[:2].copy()
        self.assertGreater(np.linalg.norm(before), 0.001)
        patches.store(state)
        points *= 0.8
        contacts.rigid_contact_point0.assign([wp.quat_rotate(wp.quat_inverse(rotation), wp.vec3(*p)) for p in points])
        contacts.rigid_contact_point1.assign(points)
        patches.build(model, state, contacts)
        np.testing.assert_allclose(patches.view.phi.numpy()[:2], 0.8 * before, atol=1.0e-7)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())

    def test_coincident_contacts_use_one_anchor(self):
        """Reject duplicate witnesses that would create two coincident friction constraints."""
        _, _, _, patches = _patch_fixture([[0, 0, 0]] * 4)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 1)
        self.assertEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_warmstart_uses_patch_load_when_anchor_normal_is_unloaded(self):
        """Keep a cached tangent supported when other normals carry the load."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
        patches.current.tangent_impulse.assign([[0.8, 0, 0], [0, 0, 0], [0.8, 0, 0]] + [[0, 0, 0]] * 5)
        patches.store(state)
        patches.build(model, state, contacts)
        slots = wp.array([0, 3, 4], dtype=int, device="cpu")
        worlds = wp.zeros(3, dtype=int, device="cpu")
        paths = wp.zeros(3, dtype=int, device="cpu")
        lengths = wp.array([3, 1, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, -1, 4, 4]], dtype=int, device="cpu")
        mu = wp.array([[0.5] * 7], dtype=float, device="cpu")
        impulses = wp.array([[0, 0, 0, 4, 0, 0, 0]], dtype=float, device="cpu")
        wp.launch(
            link_patch_rows,
            dim=3,
            inputs=[contacts.rigid_contact_count, patches.view, worlds, slots, paths, lengths, 0, parents, mu],
            device="cpu",
        )
        wp.launch(
            seed_patch_impulses,
            dim=3,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                worlds,
                slots,
                paths,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
            ],
            device="cpu",
        )
        result = impulses.numpy()[0]
        self.assertAlmostEqual(float(np.linalg.norm(result[1:3])), 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(result[5:7])), 0.8)

    def test_seed_keeps_matched_warmstart_without_patch_history(self):
        """Keep the contact-matched friction warm start on an anchor that has no patch history yet."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        self.assertEqual(int(patches.current.source.numpy()[0]), -1)
        slots = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25] + [0.0] * 5], dtype=float, device="cpu")
        seeded = [1.0, 0.3, -0.2] + [0.0] * 5
        impulses = wp.array([seeded], dtype=float, device="cpu")
        wp.launch(
            seed_patch_impulses,
            dim=1,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                slots,
                slots,
                slots,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
            ],
            device="cpu",
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.array(seeded, dtype=np.float32))

    def test_anchors_without_rows_keep_history(self):
        """Keep a valid anchor and its cached impulse when its region gets no friction rows this step."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        cached = [[0.1, 0.2, 0.0]] + [[0.0, 0.0, 0.0]] * 7
        zeros = wp.zeros(1, dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.zeros((1, 8), dtype=float, device="cpu")
        impulses = wp.zeros((1, 8), dtype=float, device="cpu")
        qd = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        com = wp.zeros(2, dtype=wp.vec3, device="cpu")
        for slot, length, load, retained in ((-1, 3, 0.0, 1), (0, 1, 2.0, 1), (0, 1, 0.0, 0)):
            with self.subTest(slot=slot, slots_needed=length):
                impulses.assign([[load] + [0.0] * 7])
                patches.current.valid.assign([1] + [0] * 7)
                patches.current.tangent_impulse.assign(cached)
                wp.launch(
                    finish_patch_impulses,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        patches.current,
                        state.body_q,
                        state.body_q,
                        qd,
                        com,
                        zeros,
                        wp.array([slot], dtype=int, device="cpu"),
                        zeros,
                        wp.array([length], dtype=int, device="cpu"),
                        0,
                        parents,
                        mu,
                        impulses,
                        0.005,
                    ],
                    device="cpu",
                )
                self.assertEqual(int(patches.current.valid.numpy()[0]), retained)
                np.testing.assert_array_equal(
                    patches.current.tangent_impulse.numpy()[0], np.asarray(cached[0], dtype=np.float32)
                )

    def test_stationary_history_does_not_accumulate_roundoff(self):
        """Avoid accumulating world round-trip error while a transformed contact remains stationary."""
        model, state, contacts, patches = _patch_fixture([[0.02, -0.03, 0.0]])
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.5, 0.8)), 1.1)
        state.body_q.assign([wp.transform(wp.vec3(1.3, -0.7, 2.1), rotation), wp.transform_identity()])
        patches.build(model, state, contacts)
        for _ in range(100):
            patches.store(state)
            patches.build(model, state, contacts)
            self.assertGreaterEqual(int(patches.current.source.numpy()[0]), 0)
            np.testing.assert_array_equal(patches.view.phi.numpy()[0], [0.0, 0.0])

    def test_filtered_members_carry_history_without_starting_it(self):
        """Carry existing anchors through a friction-filtered step with zero weight, and never create new ones."""
        model, state, contacts, patches = _patch_fixture([[-0.05, 0, 0], [0.05, 0, 0]])
        patches.store(state)
        patches.build(model, state, contacts, friction_gap=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 2)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
        patches.previous.valid.zero_()
        patches.build(model, state, contacts, friction_gap=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 0)
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)


def _run_rolling(geometry, device, *, friction_anchor_beta=None, segments=64, hz=240, direction=1, deterministic=False):
    """Roll a generated wheel for one second and average speed over its final quarter."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
    if geometry == "sphere":
        builder.add_shape_sphere(body, radius=0.05)
    elif geometry in ("mesh", "convex_hull"):
        mesh = newton.Mesh.create_cylinder(radius=0.05, half_height=0.025, segments=segments)
        getattr(builder, "add_shape_" + geometry)(body, mesh=mesh)
    else:
        rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 2)
        add_shape = getattr(builder, "add_shape_" + geometry)
        add_shape(body, radius=0.05, half_height=0.025, xform=wp.transform(wp.vec3(0), rotation))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=friction_anchor_beta)
    pipeline = newton.CollisionPipeline(model, deterministic=deterministic)
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    s0.joint_qd.assign([direction, 0, 0, 0, 20 * direction, 0])
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    control = model.control()
    final_velocity = []
    for step in range(hz):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 1.0 / hz)
        s0, s1 = s1, s0
        if step >= 3 * hz // 4:
            final_velocity.append(s0.body_qd.numpy()[0])
    pose, velocity = s0.body_q.numpy()[0], s0.body_qd.numpy()[0]
    return pose, velocity, np.mean(final_velocity, axis=0)


class TestFeatherPGSFrictionPatches(unittest.TestCase):
    def test_default_friction_preserves_free_rolling(self):
        """Match velocity-only free rolling without adding a persistent rearward friction force."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for geometry in ("sphere", "capsule", "cylinder", "mesh", "convex_hull"):
            results = []
            # Isolate displacement persistence from row reduction on faceted
            # wheels: a negligible positive gain keeps the same patch rows with
            # effectively no positional correction. The separate tessellation
            # matrix compares default patches against full point friction.
            reference_beta = 1.0e-8 if geometry in ("mesh", "convex_hull") else 0.0
            # At 240 Hz the 64-facet wheel advances almost one facet per step.
            # Resolve facet crossings for this strict persistence comparison;
            # the tessellation matrix below retains the coarse-timestep cases.
            hz = 1920 if geometry in ("mesh", "convex_hull") else 240
            for kwargs in ({"friction_anchor_beta": reference_beta}, {}):
                result = _run_rolling(geometry, device, hz=hz, deterministic=True, **kwargs)
                self.assertTrue(all(np.isfinite(value).all() for value in result))
                results.append(result)
            with self.subTest(geometry=geometry):
                reference, actual = results
                self.assertAlmostEqual(float(actual[0][0]), float(reference[0][0]), delta=0.01)
                self.assertLess(abs(float(actual[0][1] - reference[0][1])), 0.01)
                # Facet impacts lose energy even without positional correction;
                # compare averaged speed rather than individual impact phases.
                self.assertAlmostEqual(float(actual[2][0]), float(reference[2][0]), delta=0.02)
                self.assertAlmostEqual(float(actual[2][4]), float(reference[2][4]), delta=0.4)
                self.assertLess(abs(float(actual[2][1])), 0.01)
                if geometry not in ("mesh", "convex_hull"):
                    self.assertAlmostEqual(float(actual[1][0]), float(reference[1][0]), delta=0.01)
                    self.assertAlmostEqual(float(actual[1][4]), float(reference[1][4]), delta=0.2)
                    self.assertGreater(float(actual[1][0]), 0.9)
                    self.assertLess(abs(float(actual[1][0] - 0.05 * actual[1][4])), 0.01)

    def test_faceted_rolling_across_tessellation_direction_and_timestep(self):
        """Bound patch approximation error across coarse and fine faceted wheels."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for geometry in ("mesh", "convex_hull"):
            for segments in (32, 64, 128):
                for hz in (120, 240):
                    for direction in (-1, 1):
                        case = {"segments": segments, "hz": hz, "direction": direction}
                        reference = _run_rolling(geometry, device, friction_anchor_beta=0.0, **case)
                        actual = _run_rolling(geometry, device, **case)
                        with self.subTest(geometry=geometry, **case):
                            self.assertTrue(all(np.isfinite(value).all() for value in actual))
                            # A two-location friction wrench approximates the
                            # per-contact reference. Bound its one-second travel
                            # error to 3% and mean speed error to 6% of launch speed;
                            # include lateral motion so steering errors cannot hide.
                            # The 128-segment convex hull at 120 Hz lands within a
                            # few percent of these bounds and differs slightly across
                            # GPU architectures (5.1% speed error on an RTX 5080).
                            self.assertLess(np.max(np.abs(actual[0][:2] - reference[0][:2])), 0.03)
                            self.assertLess(abs(float(actual[2][0] - reference[2][0])), 0.06)
                            # Facet rocking can change the phase of lateral
                            # oscillations; bound their absolute speed as well
                            # as the lateral displacement checked above.
                            self.assertLess(abs(float(actual[2][1])), 0.05)
                            # Same 6% margin on the 20 rad/s launch spin.
                            self.assertLess(abs(float(actual[2][4] - reference[2][4])), 1.2)

    def test_curved_grasp_preserves_history_under_small_disturbances(self):
        """Bound held sphere/capsule drift despite repeated small relative rotations."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            for geometry in ("sphere", "capsule"):
                drifts = []
                for enabled in (False, True):
                    model, jaws, obj = _build_v_jaws(5.0, geometry)
                    kwargs = dict(_SQUEEZE_SOLVER)
                    kwargs.pop("contact_shared_anchor")
                    kwargs.pop("contact_friction_shared_anchor")
                    if not model.device.is_cuda:
                        kwargs.update(pgs_mode="split", enable_bilateral_preelimination=False)
                    if not enabled:
                        kwargs["friction_anchor_beta"] = 0.0
                    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
                    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, broad_phase="nxn")
                    contacts = pipeline.contacts()
                    s0, s1 = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
                    relative_height = []
                    retained = []
                    for step in range(600):
                        if step >= 100:
                            # External pose disturbances exercise history through
                            # normal changes, even when stiction resists the torque.
                            q = s0.joint_q.numpy()
                            angle = 2.0e-4 * (np.sin(step * 0.07) - np.sin((step - 1) * 0.07))
                            q[-4:] = np.asarray(
                                wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(angle)) * wp.quat(*q[-4:])
                            )
                            s0.joint_q.assign(q)
                            newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
                        pipeline.collide(s0, contacts)
                        s0.clear_forces()
                        force = s0.body_f.numpy()
                        force[obj, 4] = 2.0e-4 * np.sin(step * 0.07)
                        s0.body_f.assign(force)
                        solver.step(s0, s1, control, contacts, 0.005)
                        s0, s1 = s1, s0
                        pose = s0.body_q.numpy()
                        self.assertTrue(np.isfinite(pose).all())
                        relative_height.append(pose[obj, 2] - pose[jaws[0], 2])
                        if enabled and step >= 100:
                            retained.append(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0) >= 2)
                    drifts.append(float(np.ptp(relative_height[100:])))
                    if enabled:
                        with self.subTest(geometry=geometry):
                            self.assertGreater(np.mean(retained), 0.8)
                            self.assertLess(drifts[-1], 1.0e-4)
                with self.subTest(geometry=geometry):
                    self.assertGreater(drifts[0], 5 * drifts[1] + 5.0e-5)

    def test_default_friction_keeps_a_box_stack_at_rest(self):
        """Settle an ordinary stack without grasp-specific settings or anchor opt-in."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for height in (0.05, 0.151, 0.252):
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, height), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0
        poses, velocities = s0.body_q.numpy(), s0.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all() and np.isfinite(velocities).all())
        np.testing.assert_allclose(poses[:, 2], [0.05, 0.15, 0.25], atol=0.005)
        self.assertLess(float(np.max(np.abs(poses[:, :2]))), 0.005)
        self.assertLess(float(np.max(np.linalg.norm(velocities[:, :3], axis=1))), 0.05)
        self.assertGreater(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 0)

    def test_default_patch_friction_and_explicit_opt_out(self):
        """Build persistent patches by default while retaining an explicit velocity-only opt-out."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for kwargs, enabled in (({}, True), ({"friction_anchor_beta": 0.0}, False)):
            with self.subTest(enabled=enabled):
                _, solver, _, step = _ground_box(device, **kwargs)
                for _ in range(3):
                    step()
                self.assertEqual(solver._friction_anchors_enabled, enabled)
                if enabled:
                    self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
                    self.assertEqual(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 2)
                else:
                    self.assertFalse(hasattr(solver._friction_patches, "current"))

    def test_rocking_cube_assigns_friction_only_to_supported_edge(self):
        """Retire the lifted face anchor before building rows for a cube resting on one edge."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.rigid_gap = 1.0e-4
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=1.0e-4)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        self.assertEqual(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 2)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        translation = wp.vec3(-0.1, 0, 0) - wp.quat_rotate(rotation, wp.vec3(-0.1, 0, -0.1))
        transform = np.array([*translation, *rotation], dtype=np.float32)
        s1.joint_q.assign(transform)
        newton.eval_fk(model, s1.joint_q, s1.joint_qd, s1)
        pipeline.collide(s1, contacts)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        active = patches.view.weight.numpy() > 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-6)

    def test_contact_churn_reports_linear_force_but_not_a_wrench(self):
        """Compare the linear export with solved rows under an off-centre load after churn."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        ground = builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, pgs_iterations=64)
        contacts = newton.Contacts(4, 0, requested_attributes=["force"], device=device)
        contacts.rigid_contact_count.assign([4])
        contacts.rigid_contact_shape0.assign([box] * 4)
        contacts.rigid_contact_shape1.assign([ground] * 4)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 4)
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        s0, s1 = model.state(), model.state()
        dt = 0.005
        solver.step(s0, s1, model.control(), contacts, dt)
        points[:, :2] *= 0.8
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        # The x force and z torque correspond to an off-centre tangential load.
        s1.body_f.assign([[1, 0, 0, 0, 0, 0.05]])
        solver.step(s1, s0, model.control(), contacts, dt)
        self.assertGreater(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 0)
        count = int(solver.mf_constraint_count.numpy()[0])
        self.assertTrue(np.all(solver.contact_path.numpy()[:4] == 1))
        solved = np.einsum("ij,i->j", solver.mf_J_a.numpy()[0, :count], solver.mf_impulses.numpy()[0, :count]) / dt
        self.assertGreater(np.linalg.norm(solved[3:]), 0.01, "fixture must exercise a nonzero contact moment")
        solver.update_contacts(contacts)
        reported = contacts.force.numpy().sum(axis=0)
        np.testing.assert_allclose(reported[:3], solved[:3], atol=1.0e-4)
        # Torque is explicitly unsupported by this export, including on the base
        # branch. Preserve that limitation visibly instead of asserting a wrench.
        np.testing.assert_array_equal(reported[3:], [0, 0, 0])

    def test_shape_translation_retires_anchors_outside_new_geometry(self):
        """Retire the old contact footprint after changing a shape's local transform."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

        transforms = model.shape_transform.numpy()
        transforms[box, 0] = 1.0
        model.shape_transform.assign(transforms)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        pipeline.collide(s1, contacts)
        patches.build(model, s1, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_array_equal(patches.current.source.numpy()[active], [-1, -1])
        local_x = patches.current.anchor_a.numpy()[active, 0]
        self.assertTrue(np.all(local_x >= 0.89), "friction still acts on the removed x=-0.1..0.1 footprint")

    def test_rejected_contact_history_keeps_correct_reset_world(self):
        """Keep world ownership valid when contact order changes and normal gates reject rows."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        template = newton.ModelBuilder()
        template.add_ground_plane()
        body = template.add_body()
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=0.001)
        contacts = newton.CollisionPipeline(model, rigid_contact_max=2).contacts()
        shape_body = model.shape_body.numpy()
        boxes = np.flatnonzero(shape_body >= 0)
        grounds = np.flatnonzero(shape_body < 0)
        contacts.rigid_contact_count.assign([2])
        contacts.rigid_contact_shape0.assign(boxes)
        contacts.rigid_contact_shape1.assign(grounds)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 2)
        contacts.rigid_contact_point0.zero_()
        contacts.rigid_contact_point1.zero_()
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertEqual(int(patches.previous.valid.numpy().sum()), 2)

        contacts.rigid_contact_shape0.assign(boxes[::-1].copy())
        contacts.rigid_contact_shape1.assign(grounds[::-1].copy())
        contacts.rigid_contact_point0.assign([[0, 0, 0.002]] * 2)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        self.assertTrue((solver.contact_slot.numpy()[:2] == -1).all())
        bodies = patches.current.body_a.numpy()[:2]
        worlds = model.body_world.numpy()[bodies]
        np.testing.assert_array_equal(patches.previous_world.numpy()[:2], worlds)
        solver.reset(s0, wp.array([True, False], dtype=bool, device=device))
        valid = patches.previous.valid.numpy()[:2]
        np.testing.assert_array_equal(valid, (worlds != 0).astype(np.int32))

    @unittest.skipUnless(wp.is_cuda_available(), "explicit point kernels require CUDA")
    def test_explicit_point_solver_defaults_remain_compatible(self):
        """Honor existing point-solver selections without requiring a new opt-out argument."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cuda:0")
        for kwargs in (
            {"pgs_mode": "matrix_free", "friction_mode": "bisection"},
            {"pgs_mode": "matrix_free", "pgs_kernel": "tiled_contact"},
            {"pgs_mode": "matrix_free", "pgs_kernel": "streaming"},
        ):
            with self.subTest(kwargs=kwargs), self.assertWarnsRegex(UserWarning, "point-contact"):
                solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
                self.assertFalse(solver._friction_anchors_enabled)
                self.assertEqual(solver.friction_anchor_beta, 0.0)

    def test_shared_point_flags_warn_with_patch_friction(self):
        """Explain when patch anchors override an explicitly requested shared friction point."""
        model = newton.ModelBuilder().finalize(device="cpu")
        for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
            with self.subTest(flag=flag), self.assertWarnsRegex(UserWarning, "friction_anchor_beta=0"):
                newton.solvers.SolverFeatherPGS(model, **{flag: True})

    def test_incompatible_point_solvers_are_rejected(self):
        """Reject coupled point solves that would silently consume a patch's shared normal load."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "Patch friction requires"):
            newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, friction_mode="bisection")
        # CPU resolves every native kernel selector to the scalar loop, so only CUDA rejects them.
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, pgs_kernel="tiled_contact")
        self.assertEqual(solver.pgs_kernel, "loop")
        if wp.is_cuda_available():
            cuda_model = builder.finalize(device="cuda:0")
            for kernel in ("tiled_contact", "streaming"):
                with self.subTest(pgs_kernel=kernel), self.assertRaisesRegex(ValueError, "Patch friction requires"):
                    newton.solvers.SolverFeatherPGS(cuda_model, friction_anchor_beta=0.2, pgs_kernel=kernel)

    def test_deprecated_anchor_limit_warns_instead_of_raising(self):
        """Ignore the deprecated anchor limit without overriding the default or explicit opt-out."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
        if wp.is_cuda_available():
            # Coupled friction modes need the CUDA matrix-free route; the shim must warn and stay off there.
            cuda_model = builder.finalize(device="cuda:0")
            with self.assertWarns(DeprecationWarning):
                solver = newton.solvers.SolverFeatherPGS(
                    cuda_model,
                    contact_friction_anchor_limit=2,
                    friction_mode="bisection",
                    pgs_mode="matrix_free",
                    friction_anchor_beta=0.0,
                )
            self.assertFalse(solver._friction_anchors_enabled)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, pgs_kernel="tiled_contact")
        self.assertTrue(solver._friction_anchors_enabled)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, friction_anchor_beta=0.3)
        self.assertAlmostEqual(solver.friction_anchor_beta, 0.3)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, friction_anchor_beta=0.0)
        self.assertFalse(solver._friction_anchors_enabled)

    def test_anchor_selection_respects_contact_gap_filters(self):
        """Keep filtered extreme points from removing friction from a loaded middle contact."""
        for gate in ("contact_friction_gap_threshold", "contact_gap_gate"):
            with self.subTest(gate=gate), wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                ground = builder.add_ground_plane()
                body = builder.add_body()
                box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, **{gate: 0.001})
                contacts = newton.CollisionPipeline(model, rigid_contact_max=3).contacts()
                contacts.rigid_contact_count.assign([3])
                contacts.rigid_contact_shape0.assign([box] * 3)
                contacts.rigid_contact_shape1.assign([ground] * 3)
                contacts.rigid_contact_point0.assign([[-0.08, 0, 0.002], [0, 0, 0], [0.08, 0, 0.002]])
                contacts.rigid_contact_point1.assign([[-0.08, 0, 0], [0, 0, 0], [0.08, 0, 0]])
                contacts.rigid_contact_normal.assign([[0, 0, -1]] * 3)
                solver.step(model.state(), model.state(), model.control(), contacts, 0.005)
                rows = solver.mf_row_type.numpy()[0, : solver.mf_constraint_count.numpy()[0]]
                self.assertEqual(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION), 2)
                np.testing.assert_allclose(solver._friction_patches.view.weight.numpy()[:3], [0, 1, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_propagation_variants_preserve_a_warmstarted_patch(self):
        """Retain a sticking articulated grasp across cached, fused, and colored propagation."""
        for response in ("propagation", "propagation-fused", "propagation-colored"):
            with self.subTest(response=response), self.assertWarnsRegex(UserWarning, "Patch friction selects"):
                drift, solver, state = _run_squeeze(
                    5.0,
                    220,
                    friction_anchor_beta=0.2,
                    pgs_warmstart=True,
                    contact_friction_position_iterations=-1,
                    pgs_contact_regularization=0.0,
                    articulated_contact_response=response,
                )
                self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                self.assertLess(abs(drift), 2.0e-4)
                self.assertGreater(np.count_nonzero(solver._friction_patches.current.valid.numpy()), 0)

    def test_shape_updates_refresh_geometry_and_keep_history(self):
        """Refresh broadphase bounds without discarding anchors on unchanged geometry."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        radius = patches.body_radius.numpy().copy()
        patches.previous.valid.fill_(1)
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(patches.body_radius.numpy(), 2 * radius)
        self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), patches.previous.valid.shape[0])

    def test_patch_resists_twist_and_releases_above_its_limit(self):
        """Resist a static yaw torque with separated anchors, but permit a larger torque to spin."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for torque in (0.2, 1.0):
            with self.subTest(torque=torque), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
                builder.add_shape_box(
                    body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5)
                )
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    friction_anchor_beta=0.2,
                    pgs_iterations=64,
                    pgs_mode="matrix_free" if wp.is_cuda_available() else "split",
                )
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
                contacts = pipeline.contacts()
                s0, s1 = model.state(), model.state()
                control = model.control()
                for _ in range(100):
                    s0.body_f.assign([[0, 0, 0, 0, 0, torque]])
                    pipeline.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, 0.005)
                    s0, s1 = s1, s0
                angular_speed = abs(float(s0.body_qd.numpy()[0, 5]))
                self.assertTrue(np.isfinite(s0.body_q.numpy()).all())
                if torque == 0.2:
                    self.assertLess(angular_speed, 0.01)
                    self.assertLess(abs(float(s0.body_q.numpy()[0, 5])), 0.01)
                else:
                    self.assertGreater(angular_speed, 1.0)

    def test_pooled_coulomb_budget(self):
        """Share the loaded middle normal's budget across two unloaded anchor normals."""
        self._check_pooled_projection("cpu", native=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_native_pooled_coulomb_budget(self):
        """Use the complete patch's normal load in native CUDA projection."""
        self._check_pooled_projection("cuda:0", native=True)

    def _check_pooled_projection(self, device, native):
        capacity = 32  # Native row kernels use a full warp of storage.
        count = wp.array([7], dtype=int, device=device)
        diag = wp.ones((1, capacity), dtype=float, device=device)
        matrix = wp.array(np.eye(capacity, dtype=np.float32)[None], dtype=float, device=device)
        rhs = wp.array([[0, -100, 0, -4, 0, -100, 0] + [0] * 25], dtype=float, device=device)
        impulses = wp.zeros((1, capacity), dtype=float, device=device)
        rows = wp.array([[0, 2, 2, 0, 0, 2, 2] + [-1] * 25], dtype=int, device=device)
        parents = wp.array([[3, 0, 0, 4, 0, 4, 4] + [-1] * 25], dtype=int, device=device)
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25] + [0] * 25], dtype=float, device=device)
        args = [count, diag, matrix, rhs, impulses, 2, 1.0, rows, parents, mu, 0, 0]
        if native:
            kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(wp.get_device(device).arch))
            wp.launch_tiled(kernel, dim=[1], inputs=args, block_dim=32, device=device)
        else:
            wp.launch(pgs_solve_loop, dim=1, inputs=[count, capacity, *args[1:]], device=device)
        np.testing.assert_allclose(impulses.numpy()[0], [0, 1, 0, 4, 0, 1, 0] + [0] * 25, atol=1.0e-6)

    def test_planar_patch_reduces_friction_rows(self):
        """Retain a box face's four normals while using only two friction anchors."""
        self._check_planar_patch("cpu", "split")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_planar_patch_reduces_friction_rows(self):
        """Solve the same patch layout with native split and fused kernels."""
        for mode in ("split", "matrix_free"):
            with self.subTest(mode=mode):
                self._check_planar_patch("cuda:0", mode)

    def _check_planar_patch(self, device, mode):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize()
            solver = newton.solvers.SolverFeatherPGS(
                model, friction_anchor_beta=0.2, dense_max_constraints=64, pgs_mode=mode
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="latest")
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, model.state(), model.control(), contacts, 0.005)
            rows = np.concatenate(
                [
                    solver.row_type.numpy()[0, : solver.constraint_count.numpy()[0]],
                    solver.mf_row_type.numpy()[0, : solver.mf_constraint_count.numpy()[0]],
                ]
            )
            normals = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_CONTACT))
            friction = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION))
            self.assertEqual(normals, 4)
            self.assertEqual(friction, 4)

    def test_contact_forces_report_per_row_normal_load(self):
        """Report each contact's own normal force so a resting box's contact forces sum to its weight."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            model, solver, contacts, step = _ground_box(device)
            for _ in range(100):
                step()
            solver.update_contacts(contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            forces = contacts.rigid_contact_force.numpy()[:count]
        weight = float(model.body_mass.numpy()[0]) * 9.81
        self.assertGreaterEqual(count, 3)
        self.assertAlmostEqual(float(abs(forces[:, 2].sum())), weight, delta=0.05 * weight)
        self.assertLess(float(np.abs(forces[:, 2]).max()), weight)

    def test_gap_filtered_step_keeps_anchor_history(self):
        """Carry a region's anchors through a step in which the gap filter removes every friction row."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            _model, solver, _contacts, step = _ground_box(device, contact_friction_gap_threshold=0.001)
            patches = solver._friction_patches
            for _ in range(40):
                step()
            self.assertGreater(int((patches.current.source.numpy() >= 0).sum()), 0)
            solver.contact_friction_gap_threshold = -1.0
            step()
            self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
            self.assertGreater(int(patches.previous.valid.numpy().sum()), 0)
            solver.contact_friction_gap_threshold = 0.001
            step()
            weights = patches.view.weight.numpy()
            sources = patches.current.source.numpy()
            self.assertGreater(int(((weights > 0) & (sources >= 0)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
