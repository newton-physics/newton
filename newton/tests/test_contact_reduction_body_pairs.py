# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction ("contact compiler").

Multi-shape bodies multiply narrow-phase candidates: a foot approximated by 7
cylinders emits up to 28 points against the ground and up to 49 against another
such foot. The body-pair reduction pass approximates each body-pair/normal-bin
patch with one depth slot and six sampled footprint-support slots, so the
registered contact count reflects the physical patch more closely than the
collider decomposition. Directional sampling has no shape-independent support
bound, and nonzero hysteresis permits an incumbent within the configured score
margin of the instantaneous winner.

The tests here assert both halves of the contract:

* the registered count drops on multi-shape pairs (and why: interior points of
  a single flat patch are the ones discarded), and
* supported FeatherPGS paths consume a strictly smaller buffer while retaining
  bounded rest-height, angular-residual, multi-patch, and touchdown behavior.
"""

import gc
import os
import unittest
import unittest.mock
import weakref

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_body_pairs import (
    _BP_FACE_NORMALS_DATA,
    CELL_COORD_MAX,
    BodyPairContactReducer,
    _up_axis_rotation,
)
from newton._src.sim.contacts import Contacts
from newton.geometry import HydroelasticSDF

DT = 1.0 / 240.0
FPGS_PROPAGATION_PATH = 2


def _cylinder_foot(builder, pos, mass=1.0, num_cyl=7, radius=0.02, half_height=0.015):
    """One free body whose collision is ``num_cyl`` small cylinders in a row.

    Mimics the URDF-exported G1 foot: several primitive colliders on one link,
    all reaching the ground at nearly the same height, so a plane contact
    produces a multiple of ``num_cyl`` candidate contacts for a single patch.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            body,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    return body


def _free_jointed_foot(builder, pos, num_cyl=7, radius=0.02, half_height=0.015):
    """A cylinder-row foot with its free articulation written explicitly.

    ``ModelBuilder.add_body`` creates the same free articulation as a
    convenience. This expanded form is useful in tests that need to contrast
    free-root routing with non-free propagation paths.
    """
    link = builder.add_link(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            link,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    joint = builder.add_joint_free(parent=-1, child=link)
    builder.add_articulation([joint])
    return link


def _prismatic_jointed_foot(builder, pos, num_cyl=7, radius=0.02, half_height=0.015):
    """A cylinder-row foot on a world-rooted prismatic joint.

    Unlike a single-link free articulation, this is a non-free articulation.
    Contacts on it must therefore take the articulated propagation-row path in
    every propagation response mode, including ``propagation-fused``.
    """
    link = builder.add_link()
    for i in range(num_cyl):
        x = (i - (num_cyl - 1) / 2.0) * (2.2 * radius)
        builder.add_shape_cylinder(
            link,
            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
            radius=radius,
            half_height=half_height,
        )
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        limit_lower=-1.0,
        limit_upper=1.0,
    )
    builder.add_articulation([joint])
    return link


def _same_articulation_scissor(builder):
    """Build two overlapping sibling links in one non-free articulation."""
    no_collision = builder.default_shape_cfg.copy()
    no_collision.has_shape_collision = False
    no_collision.collision_group = 0

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.04, hy=0.04, hz=0.04, cfg=no_collision)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=base,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform_identity(),
        )
    ]
    for side in (-1.0, 1.0):
        link = builder.add_link()
        # The siblings overlap. Three slightly overlapping boxes on each link
        # make several box-box manifolds for one body pair, guaranteeing that
        # body-pair reduction has more than one manifold's worth of real work.
        for x in (-0.06, 0.0, 0.06):
            builder.add_shape_box(
                link,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                hx=0.05,
                hy=0.04,
                hz=0.04,
            )
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.06, side * 0.02, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.12, 0.0, 0.0), wp.quat_identity()),
                collision_filter_parent=False,
            )
        )
    builder.add_articulation(joints)
    return base


def _sphere_grid_body(builder, pos, n=5, spacing=0.05, radius=0.01, mass=1.0):
    """One free body with an ``n x n`` grid of sphere colliders on its underside.

    Resting on a plane this emits ``n * n`` same-normal contacts -- one flat
    patch whose only irredundant members are the grid's outer ring (hull) and
    the deepest point.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    for i in range(n):
        for j in range(n):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(
                    wp.vec3((i - (n - 1) / 2.0) * spacing, (j - (n - 1) / 2.0) * spacing, 0.0),
                    wp.quat_identity(),
                ),
                radius=radius,
            )
    return body


def _speculative_ring_body(
    builder,
    pos,
    *,
    base_half=0.015,
    base_hz=0.0015,
    ring_radius=0.028,
    ring_n=8,
    ring_lift=0.002,
    ring_half=0.003,
    ring_hz=0.01,
    mass=0.04,
):
    """Thin base slab ringed by boxes raised ``ring_lift`` above it: a tumbler.

    The ring emits speculative candidates further out than every touching base
    contact.  The touching face is at ``pos.z - base_hz``.
    """
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), mass=mass)
    builder.add_shape_box(body, hx=base_half, hy=base_half, hz=base_hz)
    for k in range(ring_n):
        a = 2.0 * np.pi * (k + 0.5) / ring_n
        z = -base_hz + ring_lift + ring_hz
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(ring_radius * np.cos(a), ring_radius * np.sin(a), z), wp.quat_identity()),
            hx=ring_half,
            hy=ring_half,
            hz=ring_hz,
        )
    return body


def _world_gaps(model, state, contacts):
    """Canonical signed separation of every live contact (positive = speculative)."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    nrm = contacts.rigid_contact_normal.numpy()[:n]
    m0 = contacts.rigid_contact_margin0.numpy()[:n]
    m1 = contacts.rigid_contact_margin1.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()

    def to_world(shape, p):
        b = shape_body[shape]
        if b < 0:
            return p
        pos, quat = body_q[b][:3], body_q[b][3:7]
        u = np.array(quat[:3])
        return pos + p + 2.0 * np.cross(u, np.cross(u, p) + quat[3] * p)

    w0 = np.array([to_world(s0[k], p0[k]) for k in range(n)]).reshape(n, 3)
    w1 = np.array([to_world(s1[k], p1[k]) for k in range(n)]).reshape(n, 3)
    gaps = np.einsum("ij,ij->i", nrm, w1 - w0) - m0 - m1
    return gaps, 0.5 * (w0 + w1)


def _max_angular_hole(points_xy, center_xy):
    """Largest angular gap [rad] between consecutive points seen from ``center_xy``."""
    if len(points_xy) < 2:
        return 2.0 * np.pi
    ang = np.sort(np.arctan2(points_xy[:, 1] - center_xy[1], points_xy[:, 0] - center_xy[0]))
    holes = np.diff(np.concatenate([ang, [ang[0] + 2.0 * np.pi]]))
    return float(holes.max())


def _make_pipeline(model, reduce_body_pairs, *, reduce_mesh_contacts=True, **kwargs):
    config_fields = {
        "body_pair_cell_size",
        "body_pair_verify",
        "body_pair_hysteresis",
        "body_pair_hashtable_headroom",
    }
    config_kwargs = {name: kwargs.pop(name) for name in tuple(kwargs) if name in config_fields}
    deterministic = kwargs.pop("deterministic", True)
    return newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        deterministic=deterministic,
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
            mesh=reduce_mesh_contacts,
            body_pairs=reduce_body_pairs,
            **config_kwargs,
        ),
        **kwargs,
    )


def _collide_once(model, state, reduce_body_pairs, **kwargs):
    pipeline = _make_pipeline(model, reduce_body_pairs, **kwargs)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return contacts


def _contact_snapshot(contacts):
    """Return (count, shape0, shape1, normal, point0) copies of the active contacts."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    return (
        n,
        contacts.rigid_contact_shape0.numpy()[:n].copy(),
        contacts.rigid_contact_shape1.numpy()[:n].copy(),
        contacts.rigid_contact_normal.numpy()[:n].copy(),
        contacts.rigid_contact_point0.numpy()[:n].copy(),
    )


def _full_contact_rows(contacts, *, canonical=False):
    """Return rows over every solver-visible rigid-contact input field.

    The materialized length is clamped to capacity because Newton deliberately
    preserves an over-capacity raw counter for diagnostics.  Buffer order is
    retained by default because sequential PGS can observe it.
    """
    raw_count = int(contacts.rigid_contact_count.numpy()[0])
    n = min(raw_count, contacts.rigid_contact_max)
    cols = [
        contacts.rigid_contact_shape0.numpy()[:n],
        contacts.rigid_contact_shape1.numpy()[:n],
        contacts.rigid_contact_point0.numpy()[:n],
        contacts.rigid_contact_point1.numpy()[:n],
        contacts.rigid_contact_offset0.numpy()[:n],
        contacts.rigid_contact_offset1.numpy()[:n],
        contacts.rigid_contact_normal.numpy()[:n],
        contacts.rigid_contact_margin0.numpy()[:n],
        contacts.rigid_contact_margin1.numpy()[:n],
        contacts.rigid_contact_point_id.numpy()[:n],
    ]
    for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
        arr = getattr(contacts, name, None)
        if arr is not None:
            cols.append(arr.numpy()[:n])
    rows = []
    for k in range(n):
        row = []
        for col in cols:
            row.extend(round(float(x), 6) for x in np.atleast_1d(col[k]))
        rows.append(tuple(row))
    return sorted(rows) if canonical else rows


def _world_points0(model, state, contacts):
    """World positions of contact witness points on shape0's body."""
    n = int(contacts.rigid_contact_count.numpy()[0])
    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    shape_body = model.shape_body.numpy()
    body_q = state.body_q.numpy()
    out = np.zeros((n, 3))
    for k in range(n):
        b = shape_body[s0[k]]
        if b < 0:
            out[k] = p0[k]
            continue
        q = body_q[b]
        pos, quat = q[:3], q[3:7]
        x, y, z, w = quat
        v = p0[k]
        # quaternion rotate (xyzw)
        u = np.array([x, y, z])
        out[k] = pos + v + 2.0 * np.cross(u, np.cross(u, v) + w * v)
    return out


class TestBodyPairReductionCounts(unittest.TestCase):
    """Registered contacts drop to patch-descriptive sets on multi-shape pairs."""

    def test_stats_doc_tracks_packed_cell_range(self):
        """Keep the public clamp telemetry contract aligned with its bit layout."""
        self.assertIn(f"+/-{CELL_COORD_MAX} range", BodyPairContactReducer.stats.__doc__)

    def _grid_on_plane(self, reduce_body_pairs):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        # off-origin placement is deliberate: cells are measured from the pair's
        # own reference body, so distance from the world origin must not matter
        _sphere_grid_body(builder, (5.13, 5.07, 0.0095))  # spheres just touching the plane
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        # one spatial cell: this test asserts the tight single-patch slot bound
        return model, state, _collide_once(model, state, reduce_body_pairs, body_pair_cell_size=10.0)

    def test_flat_patch_keeps_extremes_and_deepest(self):
        """Reduce a 25-point flat patch to at most 7 while keeping its footprint.

        The 5x5 sphere grid on a plane is one patch with one normal. The pass
        must keep no more than the per-bin slot count (6 spatial extremes + 1
        deepest), and the kept points must still span the patch: the bounding
        box of the kept footprint must cover most of the original one, and the
        deepest contact of the pair must survive.
        """
        model, state, base = self._grid_on_plane(reduce_body_pairs=False)
        n_base, *_ = _contact_snapshot(base)
        self.assertGreaterEqual(n_base, 25)

        model2, state2, red = self._grid_on_plane(reduce_body_pairs=True)
        n_red, *_ = _contact_snapshot(red)
        self.assertLessEqual(n_red, 7)
        self.assertGreaterEqual(n_red, 3)

        pts_base = _world_points0(model, state, base)
        pts_red = _world_points0(model2, state2, red)
        for axis in (0, 1):
            span_base = pts_base[:, axis].max() - pts_base[:, axis].min()
            span_red = pts_red[:, axis].max() - pts_red[:, axis].min()
            self.assertGreaterEqual(span_red, 0.7 * span_base)

    def test_two_normal_patches_never_merge(self):
        """Keep both patches when one body touches ground and wall simultaneously.

        A body with sphere colliders near a floor and near a vertical wall has
        two normal clusters in the same body pair... reduction must keep
        representatives of BOTH normals -- collapsing them into one patch would
        delete a support direction entirely.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()), mass=1.0)
        for i in range(4):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(0.02 * i, 0.0, -0.0405), wp.quat_identity()),
                radius=0.01,
            )
        # second cluster of shapes near a box acting as a wall
        for i in range(4):
            builder.add_shape_sphere(
                body,
                xform=wp.transform(wp.vec3(-0.0405, 0.0, -0.02 * i), wp.quat_identity()),
                radius=0.01,
            )
        # static wall box to the -x side, floor below
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.11, 0.0, 0.0), wp.quat_identity()),
            hx=0.06,
            hy=0.5,
            hz=0.5,
        )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        red = _collide_once(model, state, reduce_body_pairs=True)
        n, _s0, _s1, normals, _ = _contact_snapshot(red)
        self.assertGreater(n, 0)
        # classify kept normals: floor-ish (+z) vs wall-ish (+/-x)
        up = np.abs(normals @ np.array([0.0, 0.0, 1.0])) > 0.7
        side = np.abs(normals @ np.array([1.0, 0.0, 0.0])) > 0.7
        self.assertTrue(up.any(), "floor patch was dropped entirely")
        self.assertTrue(side.any(), "wall patch was dropped entirely")

    def test_single_contact_untouched(self):
        """Pass a lone sphere-on-plane contact through unchanged.

        One sphere emits one contact; there is nothing to reduce and the pass
        must not perturb the contact's data.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = _collide_once(model, state, reduce_body_pairs=False)
        red = _collide_once(model, state, reduce_body_pairs=True)
        nb, _s0b, _s1b, nrmb, p0b = _contact_snapshot(base)
        nr, _s0r, _s1r, nrmr, p0r = _contact_snapshot(red)
        self.assertEqual(nb, nr)
        np.testing.assert_allclose(np.sort(p0b, axis=0), np.sort(p0r, axis=0), atol=1e-7)
        np.testing.assert_allclose(np.sort(nrmb, axis=0), np.sort(nrmr, axis=0), atol=1e-7)

    def test_seven_cylinder_foot_on_plane(self):
        """Collapse the 7-cylinder foot's plane manifold to a patch-descriptive set.

        This is the measured G1 pathology in miniature: ~4 points per cylinder
        against the plane for one flat patch. After reduction the pair must
        register at most 7 contacts, and the deepest candidate must be kept.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.0149))  # slightly penetrating, inside one cell
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = _collide_once(model, state, reduce_body_pairs=False)
        # one spatial cell: the foot spans two default cells, this asserts the per-cell bound
        red = _collide_once(model, state, reduce_body_pairs=True, body_pair_cell_size=10.0)
        n_base, *_ = _contact_snapshot(base)
        n_red, *_ = _contact_snapshot(red)
        self.assertGreaterEqual(n_base, 7)
        self.assertLessEqual(n_red, 7)
        self.assertLess(n_red, n_base)

    def test_deterministic_output(self):
        """Produce an identical kept set on repeated collides of the same state."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0149))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        snap1 = _contact_snapshot(contacts)
        pipeline.collide(state, contacts)
        snap2 = _contact_snapshot(contacts)
        self.assertEqual(snap1[0], snap2[0])
        np.testing.assert_array_equal(snap1[1], snap2[1])
        np.testing.assert_array_equal(snap1[2], snap2[2])
        np.testing.assert_allclose(snap1[3], snap2[3], atol=0.0)
        np.testing.assert_allclose(snap1[4], snap2[4], atol=0.0)


class TestBodyPairReductionGuarantees(unittest.TestCase):
    """Unsupported configurations are rejected at construction or solver start."""

    def _mesh_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.6), wp.quat_identity()))
        mesh = newton.Mesh.create_box(0.25, 0.25, 0.25, duplicate_vertices=False, compute_inertia=False)
        builder.add_shape_mesh(body, mesh=mesh)
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def _foot_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.020))
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def test_unified_reduce_contacts_bool_compatibility(self):
        """Keep the released bool spelling limited to producer-side reduction."""
        model = self._mesh_model()
        for enabled_value, disabled_value in ((True, False), (np.bool_(True), np.bool_(False))):
            with self.subTest(bool_type=type(enabled_value).__name__):
                enabled = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=enabled_value)
                disabled = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=disabled_value)
                self.assertTrue(enabled.reduce_contacts)
                self.assertTrue(enabled.contact_reduction_config.mesh)
                self.assertTrue(enabled.mesh_contact_reduction_enabled)
                self.assertFalse(enabled.contact_reduction_config.body_pairs)
                self.assertFalse(disabled.reduce_contacts)
                self.assertFalse(disabled.contact_reduction_config.mesh)
                self.assertFalse(disabled.mesh_contact_reduction_enabled)
                self.assertFalse(disabled.contact_reduction_config.body_pairs)

    def test_unified_config_enables_body_pair_stage(self):
        """Select body-pair reduction through the existing reduce_contacts entry."""
        model = self._foot_model()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
        )
        # The requested producer policy remains visible, while the effective
        # stage is inactive because this model has no mesh path.
        self.assertTrue(pipeline.reduce_contacts)
        self.assertTrue(pipeline.contact_reduction_config.mesh)
        self.assertFalse(pipeline.mesh_contact_reduction_enabled)
        self.assertTrue(pipeline.contact_reduction_config.body_pairs)
        self.assertIsNotNone(pipeline._body_pair_reducer)

    def test_body_pair_stage_requires_mesh_stage_for_mesh_scenes(self):
        """Reject a postpass that cannot protect raw mesh output from overflow."""
        model = self._mesh_model()
        config = newton.CollisionPipeline.ContactReductionConfig(mesh=False, body_pairs=True)
        with self.assertRaisesRegex(ValueError, "producer reduction to be active"):
            newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=config)

    def test_mesh_and_body_pair_stages_compose(self):
        """Run producer and body-pair reduction in order on one compound mesh patch."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.0005), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
        )
        # Twelve-by-twelve is deliberately denser than the producer's per-pair
        # voxel and normal-slot budget, so the first reduction is observable.
        grid_size = 12
        grid = np.linspace(-0.035, 0.035, grid_size, dtype=np.float32)
        vertices = np.array([(x, y, 0.0) for x in grid for y in grid], dtype=np.float32)
        indices = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                a = i * grid_size + j
                b = (i + 1) * grid_size + j
                c = (i + 1) * grid_size + j + 1
                d = i * grid_size + j + 1
                indices.extend((a, b, c, a, c, d))
        tile = newton.Mesh(vertices, np.asarray(indices, dtype=np.int32), compute_inertia=False, is_solid=False)
        for x in (-0.04, 0.04):
            for y in (-0.04, 0.04):
                builder.add_shape_mesh(
                    body,
                    mesh=tile,
                    xform=wp.transform(wp.vec3(x, y, 0.0), wp.quat_identity()),
                )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        rigid_contact_max = 4096

        def contact_count(config):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                deterministic=True,
                rigid_contact_max=rigid_contact_max,
                reduce_contacts=config,
            )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            return pipeline, int(contacts.rigid_contact_count.numpy()[0])

        raw, raw_count = contact_count(newton.CollisionPipeline.ContactReductionConfig(mesh=False))
        mesh_only, mesh_count = contact_count(newton.CollisionPipeline.ContactReductionConfig(mesh=True))
        both, both_count = contact_count(
            newton.CollisionPipeline.ContactReductionConfig(
                mesh=True,
                body_pairs=True,
                body_pair_cell_size=1.0,
            )
        )
        stats = both.body_pair_reduction_stats()
        self.assertFalse(raw.mesh_contact_reduction_enabled)
        self.assertTrue(mesh_only.mesh_contact_reduction_enabled, "mesh producer reduction was not active")
        self.assertTrue(both.mesh_contact_reduction_enabled)
        self.assertGreater(raw_count, mesh_count, "producer stage did not reduce raw mesh-plane vertex contacts")
        self.assertEqual(stats["sum_contacts_in"], mesh_count, "body-pair stage did not receive producer output")
        self.assertEqual(stats["sum_contacts_kept"], both_count)
        self.assertLess(both_count, mesh_count, "body-pair stage did not further compact the compound patch")
        self.assertEqual(stats["input_overflow_frames"], 0)

    def test_unified_reduce_contacts_rejects_unknown_policy(self):
        """Reject strings and other values that are not reduction policies."""
        model = self._foot_model()
        with self.assertRaisesRegex(TypeError, "reduce_contacts must be bool"):
            newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts="body_pairs")

    def test_unused_hydroelastic_config_is_allowed(self):
        """Allow a hydroelastic config when the model has no active hydro pair.

        Merely supplying the independent hydro configuration must not disable
        body-pair reduction for an ordinary-contact-only model.
        """
        model = self._foot_model()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
            sdf_hydroelastic_config=HydroelasticSDF.Config(),
        )
        self.assertIsNone(pipeline.narrow_phase.hydroelastic_sdf)

    def test_active_hydroelastic_contacts_rejected_at_construction(self):
        """Reject body-pair reduction when hydroelastic contacts are active."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("hydroelastic contacts require a CUDA device")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            is_hydroelastic=True,
            sdf_max_resolution=16,
            sdf_narrow_band_range=(-0.02, 0.02),
            gap=0.01,
        )
        body0 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()))
        body1 = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()))
        builder.add_shape_box(body0, hx=0.05, hy=0.05, hz=0.05)
        builder.add_shape_box(body1, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
                sdf_hydroelastic_config=HydroelasticSDF.Config(),
            )

    def test_set_deterministic_without_sort(self):
        """Produce the same kept set with the deterministic sorter disabled.

        Winner selection packs score plus quantized pair-relative position,
        and contacts self-identify as winners, so the kept SET is a pure
        function of the physical state -- no sorting required. Compare kept sets as
        order-independent multisets across repeated collides of one state.
        """
        model = self._foot_model()
        state = model.state()

        def kept_set(contacts):
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            p0 = np.round(contacts.rigid_contact_point0.numpy()[:n], 6)
            rows = sorted(map(tuple, np.column_stack([s0, s1, p0[:, 0], p0[:, 1], p0[:, 2]]).tolist()))
            return rows

        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=False,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        first = kept_set(contacts)
        for _ in range(3):
            pipeline.collide(state, contacts)
            self.assertEqual(kept_set(contacts), first)
        # and it matches the sorted pipeline's kept set
        sorted_pipe = _make_pipeline(model, True)
        sorted_contacts = sorted_pipe.contacts()
        sorted_pipe.collide(state, sorted_contacts)
        self.assertEqual(kept_set(sorted_contacts), first)

    def test_unvalidated_solver_rejected_at_step(self):
        """A solver without body-pair reduction support refuses a reduced buffer.

        Reduced buffers are stamped; every in-repo solver that has not been
        conformance-tested calls ``_require_unreduced_contacts`` at the top of
        ``step`` and must raise rather than consume contacts whose depth
        convention it might disagree with.
        """
        model = self._foot_model()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        for solver in (
            newton.solvers.SolverSemiImplicit(model),
            # XPBD decides whether a contact is active after predicting body
            # poses.  Pre-prediction spatial winners are therefore not a sound
            # class-wide contract, even if selected settling examples match.
            newton.solvers.SolverXPBD(model),
        ):
            with self.subTest(solver=type(solver).__name__):
                with self.assertRaisesRegex(ValueError, "supports_body_pair_reduced_contacts"):
                    solver.step(state_0, state_1, model.control(), contacts, DT)
        self.assertFalse(newton.solvers.SolverXPBD.supports_body_pair_reduced_contacts)
        self.assertTrue(newton.solvers.SolverFeatherPGS.supports_body_pair_reduced_contacts)

    def test_feather_pgs_warmstart_rejected_at_step(self):
        """Reject reduced contacts in FeatherPGS warm-start configurations.

        Every contact warm-start route requires contact-match identity. Body-pair
        compaction has not been validated against that contract, so the
        otherwise-supported solver must reject both entry points before
        recording launches or mutating step state.
        """
        model = self._foot_model()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        cases = (
            ("pgs_warmstart=True", {"pgs_mode": "split", "pgs_warmstart": True}),
            ("pgs_warmstart=True", {"pgs_mode": "split", "mf_warmstart": True}),
        )
        for label, kwargs in cases:
            with self.subTest(configuration=label):
                solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
                with self.assertRaisesRegex(NotImplementedError, "contact_matching"):
                    solver.step(state_0, state_1, model.control(), contacts, DT)

    def test_feather_pgs_environment_mf_warmstart_rejected_at_step(self):
        """Reject the environment-enabled MF warm-start route on reduced contacts."""
        model = self._foot_model()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        with unittest.mock.patch.dict(os.environ, {"IL_NEWTON_FPGS_MF_WARMSTART": "1"}):
            solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split")
        self.assertTrue(solver._mf_warmstart_enabled)
        with self.assertRaisesRegex(NotImplementedError, "contact_matching"):
            solver.step(state_0, state_1, model.control(), contacts, DT)


class TestBodyPairReductionMultiPatch(unittest.TestCase):
    """Same-normal patches far apart on ONE shape pair each keep full representation."""

    def _two_cluster_body(self, builder, spread=1.0):
        """One rigid body with two 9-sphere feet ``spread`` apart on the same plane.

        Nine contacts per cluster exceed the 7-slot budget, so each cluster is
        both fully represented (deepest + extremes) AND strictly reduced --
        the plank settle test asserts the latter to stay non-vacuous.
        """
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=2.0)
        for end in (-spread / 2.0, spread / 2.0):
            for i, j in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (0, 2), (1, 2), (2, 2)):
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3(end + 0.03 * i, 0.03 * j, 0.0), wp.quat_identity()),
                    radius=0.01,
                )
        return body

    def test_both_clusters_fully_kept(self):
        """Keep deepest + extremes for each cluster, not one shared slot set.

        Both clusters contact the SAME ground shape with the SAME normal --
        without spatial cells they would compete for a single (pair, bin)
        entry and one cluster could lose all representation. With cells, each
        cluster (1 m apart >> 0.25 m cell) must keep at least 3 contacts.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        red = _collide_once(model, state, reduce_body_pairs=True)
        n, _s0, _s1, _, _p0 = _contact_snapshot(red)
        self.assertGreater(n, 0)
        pts = _world_points0(model, state, red)
        left = (pts[:, 0] < -0.2).sum()
        right = (pts[:, 0] > 0.2).sum()
        self.assertGreaterEqual(int(left), 3, "left cluster under-represented")
        self.assertGreaterEqual(int(right), 3, "right cluster under-represented")

    def test_both_clusters_fully_kept_far_from_origin(self):
        """Keep both clusters for a body 200 m from the origin, not just near it.

        The spatial cell is packed as two signed 6-bit values. Measured from the
        world origin those saturate past ~8 m, so both clusters of a distant
        body land in the same border cell, compete for one slot set, and one
        loses its support points -- while the identical scene at the origin
        passes. Anchoring the cell grid at the pair's reference body makes the
        coordinates relative, so distance from the origin cannot matter.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        q = state.body_q.numpy()
        q[body][0] += 200.0
        q[body][1] += 200.0
        state.body_q.assign(q)
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pts = _world_points0(model, state, contacts)
        left = int((pts[:, 0] < 199.8).sum())
        right = int((pts[:, 0] > 200.2).sum())
        self.assertGreaterEqual(left, 3, "left cluster under-represented far from origin")
        self.assertGreaterEqual(right, 3, "right cluster under-represented far from origin")
        self.assertEqual(
            pipeline._body_pair_reducer.stats()["cell_clamp_events"],
            0,
            "cell coordinates should be relative to the body pair, so nothing clamps",
        )

    def _settle_plank(self, reduce_on):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = self._two_cluster_body(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        stats = pipeline.body_pair_reduction_stats() if reduce_on else None
        return state_0.body_q.numpy()[body], stats

    def test_plank_settles_level(self):
        """Settle the two-cluster plank level, at the unreduced rest height.

        If either cluster lost its support points the plank would tilt about
        the other end. The invariant is against the unreduced pipeline: same
        rest height (within solver noise) and no pitch, on the same scene.
        """
        q_off, _ = self._settle_plank(False)
        q_on, stats = self._settle_plank(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "plank contacts were never reduced")
        self.assertLess(abs(float(q_on[4])), 0.02, "plank tilted -- a cluster lost support")
        self.assertLess(abs(float(q_on[2]) - float(q_off[2])), 1.5e-3)


class TestBodyPairReductionSolverConformance(unittest.TestCase):
    """Every solver declaring body-pair reduction support settles identically on/off."""

    def _settle(self, build_fn, make_solver, reduce_on, steps=240, *, deterministic=True):
        """Drop, settle, and return dynamics plus solver-path watermarks.

        The fell/contact guards make the comparison non-vacuous: a body that
        is accidentally kinematic, misconfigured, or not routed through the
        intended solver path neither falls nor proves anything by matching
        heights.
        The angular residual is the mean |omega| over the final quarter of the
        run -- ``body_qd`` is (linear, angular), so omega is components 3:6.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = build_fn(builder)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, reduce_on, deterministic=deterministic)
        contacts = pipeline.contacts()
        solver = make_solver(model)
        z0 = float(state_0.body_q.numpy()[body][2])
        peak_contacts = 0
        w_tail = []
        for k in range(steps):
            pipeline.collide(state_0, contacts)
            if k % 20 == 0:
                peak_contacts = max(peak_contacts, int(contacts.rigid_contact_count.numpy()[0]))
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            if k >= steps * 3 // 4 and k % 10 == 0:
                w_tail.append(float(np.abs(state_0.body_qd.numpy()[body][3:6]).max()))
        z_end = float(state_0.body_q.numpy()[body][2])
        watermarks = solver.constraint_row_watermarks() if hasattr(solver, "constraint_row_watermarks") else {}
        return z_end, z_end < z0 - 0.005, peak_contacts, float(np.mean(w_tail)), watermarks

    def _compare(
        self,
        build_fn,
        make_solver,
        tol=5e-4,
        w_tol=0.05,
        expected_row_path=None,
        *,
        deterministic=True,
    ):
        z_off, fell_off, contacts_off, w_off, rows_off = self._settle(
            build_fn,
            make_solver,
            False,
            deterministic=deterministic,
        )
        z_on, fell_on, contacts_on, w_on, rows_on = self._settle(
            build_fn,
            make_solver,
            True,
            deterministic=deterministic,
        )
        self.assertTrue(fell_off and fell_on, "body did not fall: the solver is not simulating it")
        self.assertGreater(contacts_off, 0)
        self.assertGreater(contacts_on, 0)
        self.assertLess(contacts_on, contacts_off, "the reduced run never actually compacted a contact manifold")
        if expected_row_path is not None:
            key = f"{expected_row_path}_high_water"
            self.assertGreater(rows_off[key], 0, f"unreduced run never exercised {expected_row_path} rows")
            self.assertGreater(rows_on[key], 0, f"reduced run never exercised {expected_row_path} rows")
            self.assertGreater(rows_on["contact_high_water"], 0, "solver saw no reduced rigid-contact rows")
        self.assertLess(abs(z_off - z_on), tol)
        # angular residual (true omega: body_qd components 3:6, tail-window
        # mean): reduced must settle as calm as unreduced up to a per-scene
        # floor. Individual supported scenes may pass an explicit measured
        # bound instead of the calm default.
        self.assertLess(w_on, max(2.0 * w_off, w_tol), f"reduced settle is rocking: |w| {w_on} vs {w_off}")

    def test_row_watermark_reports_raw_dense_overflow(self):
        """Report attempted dense rows separately from the clamped count."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        no_collision = builder.default_shape_cfg.copy()
        no_collision.has_shape_collision = False
        joints = []
        parent = -1
        for index in range(3):
            child = builder.add_link()
            builder.add_shape_box(child, hx=0.05, hy=0.05, hz=0.05, cfg=no_collision)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=newton.Axis.Z,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2 * index), wp.quat_identity()),
                    child_xform=wp.transform_identity(),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = child
        builder.add_articulation(joints)
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pipeline = _make_pipeline(model, False)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            angular_damping=0.0,
            dense_max_constraints=4,
            enable_joint_limits=True,
            joint_limit_activation_gap=float("inf"),
            row_watermark=True,
        )
        solver.step(state_0, state_1, model.control(), contacts, DT)
        rows = solver.constraint_row_watermarks()

        self.assertGreater(rows["dense_high_water"], 0)
        self.assertLessEqual(rows["dense_high_water"], 4)
        self.assertGreater(rows["dense_raw_high_water"], rows["dense_high_water"])
        self.assertEqual(rows["dense_overflow_world_steps"], 1)
        self.assertGreater(rows["dense_overflow_excess_high_water"], 0)
        self.assertEqual(int(solver._row_overflow_warning_emitted.numpy()[0]), 1)

    def test_row_watermark_includes_rolled_back_contact_rows(self):
        """Count contact bundles rejected and rolled back by the allocator."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _prismatic_jointed_foot(builder, (0.0, 0.0, 0.02))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pipeline = _make_pipeline(model, False)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 1)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            angular_damping=0.0,
            dense_max_constraints=4,
            row_watermark=True,
        )
        solver.step(state_0, state_1, model.control(), contacts, DT)
        rows = solver.constraint_row_watermarks()

        self.assertGreater(rows["dense_raw_high_water"], rows["dense_high_water"])
        self.assertGreater(rows["dense_dropped_contact_rows_high_water"], 0)
        self.assertEqual(
            rows["dense_raw_high_water"],
            rows["dense_high_water"] + rows["dense_dropped_contact_rows_high_water"],
        )
        self.assertEqual(rows["dense_overflow_world_steps"], 1)
        self.assertGreater(rows["dense_overflow_excess_high_water"], 0)

    def test_constraint_overflow_warning_does_not_require_row_watermark(self):
        """Keep the default warning active without full high-water telemetry."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _prismatic_jointed_foot(builder, (0.0, 0.0, 0.02))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pipeline = _make_pipeline(model, False)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            angular_damping=0.0,
            dense_max_constraints=4,
        )
        solver.step(state_0, state_1, model.control(), contacts, DT)

        self.assertFalse(solver._row_watermark)
        self.assertTrue(solver._track_row_capacity)
        self.assertGreater(int(solver._row_dropped_dense.numpy().max()), 0)
        self.assertEqual(int(solver._row_overflow_warning_emitted.numpy()[0]), 1)

    def test_constraint_overflow_warning_covers_matrix_free_and_propagation_rows(self):
        """Identify overflows on both non-dense contact row families."""
        if not wp.get_device().is_cuda:
            self.skipTest("matrix-free FeatherPGS overflow coverage is CUDA-only")

        cases = (
            ("immediate", _free_jointed_foot, 1, "_row_dropped_mf"),
            ("propagation", _prismatic_jointed_foot, 2, "_row_dropped_propagation"),
        )
        for response, build_foot, warning_index, dropped_attribute in cases:
            with self.subTest(response=response):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                build_foot(builder, (0.0, 0.0, 0.02))
                builder.add_ground_plane()
                model = builder.finalize(device=wp.get_device())
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                pipeline = _make_pipeline(model, False)
                contacts = pipeline.contacts()
                pipeline.collide(state_0, contacts)

                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    angular_damping=0.0,
                    pgs_mode="matrix_free",
                    articulated_contact_response=response,
                    dense_max_constraints=4,
                    mf_max_constraints=4,
                )
                solver.step(state_0, state_1, model.control(), contacts, DT)

                dropped_rows = getattr(solver, dropped_attribute)
                self.assertGreater(int(dropped_rows.numpy().max()), 0)
                self.assertEqual(int(solver._row_overflow_warning_emitted.numpy()[warning_index]), 1)

    def test_feather_pgs_conformance(self):
        """SolverFeatherPGS rests a free-jointed foot at the same height on/off.

        This is the conformance requirement for
        supports_body_pair_reduced_contacts:
        the solver's contact-depth convention must agree with the ranking's
        canonical contact_surface_separation, or the kept set starves the
        solver of its load-bearing contacts. The foot uses the explicit form
        of the free articulation so this scene's routing is unambiguous.
        """
        self._compare(
            lambda b: _free_jointed_foot(b, (0.0, 0.0, 0.05)),
            lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0),
        )

    def test_feather_pgs_nondeterministic_conformance(self):
        """Conform on the production-default unsorted collision pipeline.

        Sequential PGS observes row order, so the sorted conformance cases do
        not establish support for ``CollisionPipeline(deterministic=False)``.
        This exercises the default split solver with strict contact compaction.
        """
        self._compare(
            lambda b: _free_jointed_foot(b, (0.0, 0.0, 0.05)),
            lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0),
            deterministic=False,
        )

    def test_feather_pgs_dynamic_dynamic_conformance(self):
        """Settle a free-jointed foot dropped onto another free-jointed foot.

        Covers articulated dynamic-dynamic contact: both the falling body and
        its support are simulated bodies, so the reduced set must preserve the
        body-body patch as well as the body-ground patches.
        """

        def build(b):
            _free_jointed_foot(b, (0.0, 0.0, 0.0175))
            return _free_jointed_foot(b, (0.0, 0.005, 0.09))

        self._compare(build, lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0), tol=2e-3, w_tol=0.15)

    def test_feather_pgs_contact_mode_conformance(self):
        """Settle identically on/off in every FeatherPGS contact mode.

        supports_body_pair_reduced_contacts is class-wide, so the evidence must cover
        the dense and matrix-free contact paths, not only the default split
        mode the other tests exercise (matrix-free additionally runs generated
        native CUDA).
        """
        if not wp.get_device().is_cuda:
            self.skipTest("FeatherPGS contact-mode conformance is a CUDA-only merge gate")

        configs = [
            ("split", "immediate", 16, _prismatic_jointed_foot, "dense"),
            ("split", "immediate", 16, _free_jointed_foot, "mf"),
        ]
        # Matrix-free (and its propagation response family) is CUDA-only. Use
        # a non-free prismatic articulation for every propagation variant. A
        # free-only scene disables propagation-fused and would let that
        # subtest pass through the ordinary matrix-free solver.
        configs += [
            ("matrix_free", "immediate", None, _free_jointed_foot, "mf"),
            ("matrix_free", "propagation", 16, _prismatic_jointed_foot, "propagation"),
            ("matrix_free", "propagation-fused", None, _prismatic_jointed_foot, "propagation"),
            ("matrix_free", "propagation-colored", 16, _prismatic_jointed_foot, "propagation"),
        ]
        # Dense, propagation, and propagation-colored run 16 iterations. The
        # two ``None`` entries deliberately omit pgs_iterations and assert the
        # production default (12), rather than testing a stale assumed default.
        # At 8 iterations the UNREDUCED redundant set does not converge (dense
        # settles 5 mm high, plain/colored propagation 4.8 mm high at 0.01980;
        # the REDUCED set and split land at the true 0.01500 at every
        # iteration count) -- redundant near-parallel rows hurt PGS
        # conditioning, the same effect as the stack-collapse case, and
        # reduction itself is what removes it.  The comparison must be against
        # a converged baseline.
        for mode, response, iters, build_foot, row_path in configs:
            with self.subTest(pgs_mode=mode, response=response):

                def make_solver(model, mode=mode, response=response, iters=iters, row_path=row_path):
                    kwargs = {}
                    if iters is not None:
                        kwargs["pgs_iterations"] = iters
                    solver = newton.solvers.SolverFeatherPGS(
                        model,
                        angular_damping=0.0,
                        pgs_mode=mode,
                        articulated_contact_response=response,
                        row_watermark=True,
                        **kwargs,
                    )
                    if iters is None:
                        self.assertEqual(solver.pgs_iterations, 12)
                    if row_path == "propagation":
                        self.assertTrue(
                            solver._propagation_contacts_enabled(),
                            f"{response} test scene did not enable propagation contacts",
                        )
                    if response == "propagation-fused":
                        self.assertTrue(solver.propagation_full_fused_iterations)
                    return solver

                self._compare(
                    lambda b, build_foot=build_foot: build_foot(b, (0.0, 0.0, 0.05)),
                    make_solver,
                    tol=2e-3 if row_path == "propagation" else 5e-4,
                    expected_row_path=row_path,
                )

    def test_feather_pgs_same_articulation_rows_are_nonvacuous(self):
        """Route a strictly reduced sibling-link manifold to propagation rows.

        The general propagation scenes are link-to-ground contacts. This pins
        the opt-in cross-response path for contacts whose two bodies belong to
        the same articulation, and proves both that reduction did work and
        that the resulting rows reached the intended CUDA kernel family.
        """
        if not wp.get_device().is_cuda:
            self.skipTest("matrix-free propagation requires CUDA")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _same_articulation_scissor(builder)
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        raw_pipeline = _make_pipeline(model, False)
        raw_contacts = raw_pipeline.contacts()
        raw_pipeline.collide(state_0, raw_contacts)
        raw_count = int(raw_contacts.rigid_contact_count.numpy()[0])

        reduced_pipeline = _make_pipeline(model, True, body_pair_cell_size=10.0)
        reduced_contacts = reduced_pipeline.contacts()
        reduced_pipeline.collide(state_0, reduced_contacts)
        reduced_count = int(reduced_contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(raw_count, 0, "scissor scene emitted no sibling-link contacts")
        self.assertLess(reduced_count, raw_count, "same-articulation manifold was not reduced")

        solver = newton.solvers.SolverFeatherPGS(
            model,
            angular_damping=0.0,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            propagation_same_articulation_rows=True,
            row_watermark=True,
        )
        self.assertEqual(solver.pgs_iterations, 12)
        self.assertTrue(solver._propagation_contacts_enabled())
        solver.step(state_0, state_1, model.control(), reduced_contacts, DT)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_1.body_qd.numpy()).all())
        rows = solver.constraint_row_watermarks()
        self.assertGreater(rows["propagation_high_water"], 0)
        active_paths = solver.contact_path.numpy()[:reduced_count]
        routed = np.flatnonzero(active_paths == FPGS_PROPAGATION_PATH)
        self.assertGreater(routed.size, 0, "no reduced contact was routed to the propagation path")
        shape_body = model.shape_body.numpy()
        body_art = solver.body_to_articulation.numpy()
        shape0 = reduced_contacts.rigid_contact_shape0.numpy()[:reduced_count]
        shape1 = reduced_contacts.rigid_contact_shape1.numpy()[:reduced_count]
        for contact in routed:
            body0 = int(shape_body[shape0[contact]])
            body1 = int(shape_body[shape1[contact]])
            self.assertGreaterEqual(body0, 0)
            self.assertGreaterEqual(body1, 0)
            self.assertEqual(
                int(body_art[body0]),
                int(body_art[body1]),
                "the path assertion did not cover a same-articulation contact",
            )

    def test_mixed_separation_tilted_patch(self):
        """Settle a tilted patch whose contacts mix penetrating and speculative.

        This is also the adversarial topology that prevents an XPBD-wide
        compatibility claim: footprint slots may be won by speculative
        endpoints while other contacts become active after XPBD predicts its
        body poses. The probe characterizes that reducer input, while dynamics
        are checked only with the solver that declares compatibility.
        """
        pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.06)

        def build(b):
            body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.026), pitch), mass=1.0)
            for i in range(7):
                b.add_shape_cylinder(
                    body,
                    xform=wp.transform(wp.vec3((i - 3.0) * 0.044, 0.0, 0.0), wp.quat_identity()),
                    radius=0.02,
                    half_height=0.015,
                )
            return body

        def build_jointed(b):
            link = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.026), pitch))
            for i in range(7):
                b.add_shape_cylinder(
                    link,
                    xform=wp.transform(wp.vec3((i - 3.0) * 0.044, 0.0, 0.0), wp.quat_identity()),
                    radius=0.02,
                    half_height=0.015,
                )
            b.add_articulation([b.add_joint_free(parent=-1, child=link)])
            return link

        # prove the scene actually generates the adversarial topology at some
        # point of the settle, in the reducer's OWN competition terms: within
        # a single reduction group (one contact_entry), several LOAD-BEARING
        # candidates (within 1 mm of the group's deepest) coexist with a far
        # SHALLOWER candidate (> 5 mm above the deepest) that WINS one of the
        # six real scan-direction slots (0/60/../300 degrees on the face
        # plane) and SURVIVES a strictly reducing compaction, while the
        # group's deepest survives too.  Note the canonical surface separation
        # of cylinder-plane witnesses saturates at >= 0 under penetration (the
        # solver derives depth from body poses at solve time), so
        # deep/shallow is a gap-band split, not a signed-depth split; the
        # dynamic equivalence itself is pinned by the settle comparisons
        # below.  Hysteresis is disabled on the probe pipeline so slot winners
        # are exact projections, not incumbency-biased ones.
        probe_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        probe_body = build(probe_builder)
        probe_builder.add_ground_plane()
        probe_model = probe_builder.finalize(device=wp.get_device())
        ps0, ps1 = probe_model.state(), probe_model.state()
        probe_control = probe_model.control()
        probe_pipe = _make_pipeline(probe_model, True, body_pair_hysteresis=0.0)
        probe_contacts = probe_pipe.contacts()
        probe_solver = newton.solvers.SolverXPBD(probe_model, iterations=8)
        red = probe_pipe._body_pair_reducer
        # Match the reduced pipeline's deterministic producer and capacity so
        # its pre-compaction scratch rows correspond one-for-one to this raw
        # buffer when the topology is inspected below.
        raw_pipe = _make_pipeline(probe_model, False)
        raw_contacts = raw_pipe.contacts()
        scan_dirs = [
            (float(np.cos(k * np.pi / 3.0)), float(np.sin(k * np.pi / 3.0)))
            for k in range(6)  # BODY_PAIR_NUM_DIRECTIONS
        ]
        topology_seen = False
        _ = probe_body
        for _k in range(60):
            raw_pipe.collide(ps0, raw_contacts)
            n_raw = int(raw_contacts.rigid_contact_count.numpy()[0])
            probe_pipe.collide(ps0, probe_contacts)
            n_red = int(probe_contacts.rigid_contact_count.numpy()[0])
            if n_raw > 2 and n_red < n_raw and not topology_seen:
                entries = red.contact_entry.numpy()[:n_raw]
                gaps = red.contact_gap.numpy()[:n_raw]
                pos = red.contact_pos2d.numpy()[:n_raw]
                kept = red.keep_flags.numpy()[:n_raw]
                for e in np.unique(entries[entries >= 0]):
                    members = np.where(entries == e)[0]
                    if members.size < 3:
                        continue
                    g0 = float(gaps[members].min())
                    deep = members[gaps[members] < g0 + 0.001]
                    shallow = members[gaps[members] > g0 + 0.005]
                    if deep.size < 2 or shallow.size == 0:
                        continue
                    deepest = members[int(np.argmin(gaps[members]))]
                    for dx, dy in scan_dirs:
                        proj = pos[members, 0] * dx + pos[members, 1] * dy
                        winner = members[int(np.argmax(proj))]
                        if winner in shallow and kept[winner] == 1 and kept[deepest] == 1:
                            topology_seen = True
                            break
                    if topology_seen:
                        break
            ps0.clear_forces()
            # Advance using the unreduced buffer. XPBD's activity decision is
            # made after this prediction, later than the reducer competition.
            probe_solver.step(ps0, ps1, probe_control, raw_contacts, DT)
            ps0, ps1 = ps1, ps0
        self.assertTrue(
            topology_seen,
            "settle never produced a strictly reduced group whose kept set holds both the deepest "
            "candidate and a shallow scan-direction slot winner",
        )

        self._compare(
            build_jointed, lambda m: newton.solvers.SolverFeatherPGS(m, angular_damping=0.0), tol=1e-3, w_tol=0.15
        )


class TestBodyPairReductionTableSizing(unittest.TestCase):
    """The group table is sized from scene topology, not from contact capacity."""

    def test_replicated_compound_feet_do_not_exhaust_default_table(self):
        """Replication must retain the patch budget without tuning headroom."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 512, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        stats = pipeline.body_pair_reduction_stats()
        self.assertEqual(stats["fallback_frames"], 0)
        self.assertEqual(stats["probe_failures"], 0)
        self.assertLess(stats["max_contacts_kept"], stats["max_contacts_in"])

    def _scene(self, n_bodies):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for k in range(n_bodies):
            body = builder.add_body(xform=wp.transform(wp.vec3(k * 0.5, 0.0, 0.02), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def test_capacity_tracks_group_pairs_not_contact_capacity(self):
        """Derive the table capacity from the model's reachable group pairs.

        The entry count is a function of scene topology, so a table anchored to
        ``rigid_contact_max`` scales with contact density instead -- the very
        quantity the reduction varies. Doubling the contact buffer at fixed
        topology must not change the capacity; adding bodies must.
        """
        model = self._scene(8)
        pipeline = _make_pipeline(model, True)
        desc = pipeline.body_pair_reduction_description()
        self.assertGreater(desc["group_pair_bound"], 0, "topology anchor was not derived")
        # Capped at rigid_contact_max (a group needs a contact to create it),
        # except that a minimum viable table always wins.
        self.assertGreaterEqual(desc["hashtable_capacity_request"], 1024)
        self.assertLessEqual(desc["hashtable_capacity_request"], max(1024, desc["rigid_contact_max"]))
        # Same topology, 4x the contact buffer -> identical request.
        wide = _make_pipeline(model, True, rigid_contact_max=model.rigid_contact_max * 4)
        wide_desc = wide.body_pair_reduction_description()
        self.assertEqual(wide_desc["group_pair_bound"], desc["group_pair_bound"])
        self.assertEqual(wide_desc["hashtable_capacity_request"], desc["hashtable_capacity_request"])
        # More bodies -> more reachable group pairs -> a larger anchor.
        bigger = _make_pipeline(self._scene(24), True)
        self.assertGreater(bigger.body_pair_reduction_description()["group_pair_bound"], desc["group_pair_bound"])

    def test_headroom_scales_the_derived_capacity(self):
        """Scale the derived capacity by ``body_pair_hashtable_headroom``."""
        model = self._scene(24)
        small = _make_pipeline(model, True, body_pair_hashtable_headroom=1.0)
        large = _make_pipeline(model, True, body_pair_hashtable_headroom=8.0)
        self.assertGreater(
            large.body_pair_reduction_description()["hashtable_capacity_request"],
            small.body_pair_reduction_description()["hashtable_capacity_request"],
        )


class TestBodyPairReductionGrouping(unittest.TestCase):
    """Contacts may only compete when merging them cannot change the physics."""

    def _pad_body(self, mu_pad, mu_ring):
        """One body: a 5x5 low-mu sphere grid with one center pad sphere of mu_pad.

        The pad sits at the patch's interior, the exact position body-level
        grouping discards.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        ring_cfg = newton.ModelBuilder.ShapeConfig(mu=mu_ring)
        pad_cfg = newton.ModelBuilder.ShapeConfig(mu=mu_pad)
        for i in range(5):
            for j in range(5):
                if i == 2 and j == 2:
                    continue
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3((i - 2.0) * 0.04, (j - 2.0) * 0.04, 0.0), wp.quat_identity()),
                    radius=0.01,
                    cfg=ring_cfg,
                )
        builder.add_shape_sphere(
            body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()), radius=0.01, cfg=pad_cfg
        )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        contacts = _collide_once(model, state, True, body_pair_cell_size=10.0)
        n, s0, s1, _n2, _p0 = _contact_snapshot(contacts)
        mu_arr = model.shape_material_mu.numpy()
        sb = model.shape_body.numpy()
        kept_mus = {round(float(mu_arr[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        return n, kept_mus

    def test_heterogeneous_materials_never_compete(self):
        """Keep a high-friction center pad that body-level grouping would delete.

        Solvers read material laws from the surviving shape ids. A center pad
        is an interior point of the merged patch -- exactly what the reduction
        discards -- so merging it with the surrounding low-friction colliders
        deletes the high-friction law entirely. With material-equivalence
        classes the pad is its own group and must survive. Identical materials
        must still merge into one group (the point of body-level grouping).
        """
        _n_hetero, mus_hetero = self._pad_body(1.0, 0.2)
        self.assertIn(1.0, mus_hetero, "the high-friction pad's contact was deleted")
        n_homo, _mus_homo = self._pad_body(0.2, 0.2)
        self.assertLessEqual(n_homo, 7, "identical materials must merge into one group")

    def test_material_mutation_requires_refresh_and_then_splits(self):
        """Split reduction groups after a runtime material mutation + refresh.

        Group ids are construction-time snapshots; mutating one collider's
        friction afterwards must not silently keep it competing in its old
        class once the caller refreshes the grouping.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.2)
        for i in range(5):
            for j in range(5):
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3((i - 2.0) * 0.04, (j - 2.0) * 0.04, 0.0), wp.quat_identity()),
                    radius=0.01,
                    cfg=cfg,
                )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True, body_pair_cell_size=10.0)
        contacts = pipeline.contacts()

        # mutate the CENTER collider (index 12: interior point of the patch)
        mu = model.shape_material_mu.numpy()
        mu[12] = 1.0
        model.shape_material_mu.assign(mu)

        pipeline.collide(state, contacts)  # stale classes: center still merged
        _n, s0, s1, _n1, _p0 = _contact_snapshot(contacts)
        sb = model.shape_body.numpy()
        kept_mus_stale = {round(float(mu[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        self.assertNotIn(1.0, kept_mus_stale, "scene no longer discards the interior point: rebuild it")

        pipeline.refresh_body_pair_reduction_groups()
        pipeline.collide(state, contacts)
        _n2b, s0, s1, _n2, _p0 = _contact_snapshot(contacts)
        kept_mus = {round(float(mu[a if sb[a] >= 0 else b]), 3) for a, b in zip(s0, s1, strict=True)}
        self.assertIn(1.0, kept_mus, "refreshed classes did not separate the mutated collider")

    def test_swapped_endpoints_share_one_group(self):
        """Merge one physical patch whose contacts arrive with both endpoint orders.

        The narrow phase orders each shape pair by geometry type, so a flat
        interface between two bodies with interleaved sphere/box colliders
        produces contacts in BOTH directions for the same body pair. Without a
        canonical group normal they bin as two opposite-normal groups and keep
        two slot sets for one patch.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        base = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=0.0)
        builder.add_shape_box(base, hx=0.3, hy=0.1, hz=0.1)
        top = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.2195), wp.quat_identity()), mass=1.0)
        for i in range(4):
            x = (i - 1.5) * 0.12
            if i % 2 == 0:
                builder.add_shape_sphere(top, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), radius=0.02)
            else:
                builder.add_shape_box(
                    top, xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()), hx=0.02, hy=0.02, hz=0.02
                )
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True, body_pair_cell_size=10.0)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n, _s0, _s1, _nrm, _p0 = _contact_snapshot(contacts)
        self.assertGreater(n, 0)
        # the raw narrow-phase output must actually contain both orientations,
        # or this test is vacuous (geometry-type dispatch produces the swap)
        base_pipe = newton.CollisionPipeline(model, broad_phase="nxn")
        base_contacts = base_pipe.contacts()
        base_pipe.collide(state, base_contacts)
        nb = int(base_contacts.rigid_contact_count.numpy()[0])
        nz = base_contacts.rigid_contact_normal.numpy()[:nb, 2]
        self.assertGreater(int((nz > 0).sum()), 0)
        self.assertGreater(int((nz < 0).sum()), 0, "scene no longer produces swapped endpoint orders")
        # one flat patch between one body pair must occupy exactly ONE group;
        # without the canonical normal the two orientations bin separately
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["max_hashtable_entries"], 1, "endpoint order leaked into grouping")
        self.assertLessEqual(n, 7)

    def test_up_axis_rotation_protects_ground_normals(self):
        """Give the ground direction its wide bin margin in X-, Y-, and Z-up models.

        The bin table is oriented for +Z; other up axes must be rotated into
        that frame or their ground normals land near the table's narrow
        horizontal margins and curved-terrain grouping churns again.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        for up_axis in (0, 1, 2):
            rot = np.array(_up_axis_rotation(up_axis), dtype=np.float64)
            self.assertTrue(np.allclose(rot @ rot.T, np.eye(3)), "rotation must be orthonormal")
            self.assertAlmostEqual(float(np.linalg.det(rot)), 1.0, places=12)
            up = np.zeros(3)
            up[up_axis] = 1.0
            for sign in (1.0, -1.0):
                mapped = rot @ (sign * up)
                dots = np.sort(faces @ mapped)[::-1]
                self.assertGreater(dots[0] - dots[1], 0.2, f"ground margin lost for up_axis={up_axis}")


class TestBodyPairReductionSafety(unittest.TestCase):
    """Resource exhaustion must fail open deterministically, never corrupt memory."""

    def test_narrowphase_overflow_fails_open(self):
        """Deliver the raw over-capacity counter untouched when the input overflows.

        Newton's narrow phase reserves contact indices before checking capacity,
        so the raw counter legitimately exceeds ``rigid_contact_max`` on
        overflow and is NOT a safe array bound. The reduction must detect that
        on device, do nothing that frame, and leave the counter and contact
        prefix exactly as an unreduced pipeline would deliver them.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _sphere_grid_body(builder, (0.0, 0.0, 0.0095))  # 25 same-plane contacts
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        plain = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            rigid_contact_max=8,  # deliberately far below the ~25 candidates
        )
        reduced = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            rigid_contact_max=8,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
        )
        contacts = reduced.contacts()
        plain.collide(state, contacts)
        raw_count = int(contacts.rigid_contact_count.numpy()[0])
        rows_before = _full_contact_rows(contacts)
        self.assertGreater(raw_count, 8, "scene must actually overflow the buffer")

        # Isolate the postpass from a second nondeterministic narrow-phase run:
        # applying it to this exact materialized prefix must be an identity.
        reduced._body_pair_reducer.reduce(model, state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), raw_count)
        self.assertEqual(
            _full_contact_rows(contacts),
            rows_before,
            "overflow fail-open changed the materialized contact prefix",
        )
        overflow_stats = reduced.body_pair_reduction_stats()
        self.assertEqual(overflow_stats["total_frames"], 1)
        self.assertEqual(overflow_stats["input_overflow_frames"], 1)
        self.assertEqual(overflow_stats["sum_contacts_in"], 0)
        self.assertEqual(overflow_stats["sum_contacts_kept"], 0)

    def test_hashtable_saturation_keeps_whole_frame(self):
        """Keep the entire unreduced set when the group table budget is exceeded.

        Per-contact fail-open would let CUDA scheduling decide which groups get
        the last table entries -- a scheduling-dependent kept set. Exceeding
        the budget must instead fall back for the whole frame: every contact
        kept, which is deterministic, and counted in telemetry.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rng = np.random.default_rng(9)
        n_bodies = 1200  # > the 1024 minimum table capacity, one group each
        for k in range(n_bodies):
            x, y = (k % 40) * 0.5, (k // 40) * 0.5
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, y, 0.0095 + float(rng.uniform(0, 0.001))), wp.quat_identity()),
                mass=1.0,
            )
            builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base = int(contacts_base.rigid_contact_count.numpy()[0])
        self.assertGreater(n_base, 1024)

        # a headroom small enough to hit the 1024-entry floor: 1200 groups cannot fit
        pipeline = _make_pipeline(model, True, body_pair_hashtable_headroom=1e-6)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red = int(contacts.rigid_contact_count.numpy()[0])
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(n_red, n_base, "saturated frame must keep the whole unreduced set")
        self.assertEqual(
            _full_contact_rows(contacts),
            _full_contact_rows(contacts_base),
            "whole-frame fallback changed a solver-visible contact field or row order",
        )
        self.assertEqual(stats["fallback_frames"], 1)
        self.assertGreater(stats["probe_failures"], 0)
        self.assertEqual(stats["failed_insertions"], stats["probe_failures"])
        self.assertEqual(stats["total_frames"], 1)
        self.assertEqual(stats["sum_contacts_in"], n_base)
        self.assertEqual(stats["sum_contacts_kept"], n_base)

    def test_mismatched_contacts_buffer_rejected(self):
        """Refuse an external Contacts buffer whose capacity differs from the pipeline's.

        The reducer's caches and launch bounds are sized to the pipeline's
        capacity; a larger external buffer would let the narrow phase write
        beyond them.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        oversized = Contacts(
            rigid_contact_max=pipeline.rigid_contact_max * 2, soft_contact_max=0, device=wp.get_device()
        )
        with self.assertRaisesRegex(ValueError, "rigid_contact_max"):
            pipeline.collide(state, oversized)

    def test_invalid_configuration_rejected(self):
        """Reject non-finite or non-positive reduction configuration at construction."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        for kwargs in (
            {"body_pair_cell_size": 0.0},
            {"body_pair_cell_size": float("inf")},
            {"body_pair_cell_size": float("nan")},
            {"body_pair_cell_size": 1.0e-50},
            {"body_pair_hysteresis": float("inf")},
            {"body_pair_hysteresis": -1.0},
            {"body_pair_hysteresis": -1.0e-50},
            {"body_pair_hysteresis": 1.0e-50},
            {"body_pair_hashtable_headroom": 0.0},
        ):
            with self.assertRaises(ValueError, msg=f"accepted invalid {kwargs}"):
                _make_pipeline(model, True, **kwargs)

    def test_identity_frames_leave_buffer_untouched(self):
        """Detect no-benefit frames and skip compaction without changing results.

        Single-collider bodies produce one contact per group; reduction cannot
        remove anything, so the fast path must deliver the identical buffer an
        unreduced pipeline would, and count the frame in telemetry.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        rng = np.random.default_rng(4)
        for k in range(24):
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3((k % 6) * 0.5, (k // 6) * 0.5, 0.0095 + float(rng.uniform(0, 0.001))),
                    wp.quat_identity(),
                ),
                mass=1.0,
            )
            builder.add_shape_sphere(body, radius=0.01)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base = int(contacts_base.rigid_contact_count.numpy()[0])

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red = int(contacts.rigid_contact_count.numpy()[0])

        self.assertEqual(n_red, n_base, "identity frame must keep every contact")
        self.assertEqual(
            _full_contact_rows(contacts),
            _full_contact_rows(contacts_base),
            "identity fast path changed a solver-visible contact field or row order",
        )
        stats = pipeline.body_pair_reduction_stats()
        self.assertEqual(stats["identity_frames"], 1)
        self.assertEqual(stats["total_frames"], 1)
        self.assertEqual(stats["sum_contacts_in"], n_base)
        self.assertEqual(stats["sum_contacts_kept"], n_base)

    def test_noop_frames_skip_compaction_for_small_groups(self):
        """Detect no-op frames whose groups are larger than one contact.

        A plain box on the ground is ONE group of four contacts that all fit
        in the slots: nothing is discarded, so the copy-back must be skipped
        (tier-2 detection after the keep scan) and the buffer delivered
        exactly as the unreduced pipeline would.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0495), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        base = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        contacts_base = base.contacts()
        base.collide(state, contacts_base)
        n_base = int(contacts_base.rigid_contact_count.numpy()[0])
        self.assertGreater(n_base, 1, "a box should rest on several contacts")

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n_red = int(contacts.rigid_contact_count.numpy()[0])
        self.assertEqual(n_red, n_base)
        self.assertEqual(
            _full_contact_rows(contacts),
            _full_contact_rows(contacts_base),
            "tier-two no-op changed a solver-visible contact field or row order",
        )
        self.assertGreaterEqual(pipeline.body_pair_reduction_stats()["identity_frames"], 1)

    def test_stats_totals_give_achieved_ratio(self):
        """Accumulate paired frame totals that yield the achieved reduction ratio.

        The max watermarks are independent and cannot form a ratio; the sums
        are paired per frame, so on a statically reducing scene the achieved
        ratio is exact: total_frames counts every collide, sum_contacts_in is
        frame-count times the (deterministic) input count, and kept stays
        strictly below in.  Clearing resets every accumulator, including the
        64-bit ones.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        raw = newton.CollisionPipeline(model, broad_phase="nxn", deterministic=True)
        raw_contacts = raw.contacts()
        observed_in = 0
        observed_kept = 0
        observed_max_in = 0
        observed_max_kept = 0
        for _ in range(4):
            raw.collide(state, raw_contacts)
            frame_in = int(raw_contacts.rigid_contact_count.numpy()[0])
            pipeline.collide(state, contacts)
            frame_kept = int(contacts.rigid_contact_count.numpy()[0])
            observed_in += frame_in
            observed_kept += frame_kept
            observed_max_in = max(observed_max_in, frame_in)
            observed_max_kept = max(observed_max_kept, frame_kept)
        stats = pipeline.body_pair_reduction_stats()
        self.assertEqual(stats["total_frames"], 4)
        self.assertEqual(stats["sum_contacts_in"], observed_in)
        self.assertEqual(stats["sum_contacts_kept"], observed_kept)
        self.assertEqual(stats["max_contacts_in"], observed_max_in)
        self.assertEqual(stats["max_contacts_kept"], observed_max_kept)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "cylinder foot must reduce")
        pipeline.clear_body_pair_reduction_stats()
        cleared = pipeline.body_pair_reduction_stats()
        for key in (
            "total_frames",
            "sum_contacts_in",
            "sum_contacts_kept",
            "max_contacts_in",
            "max_contacts_kept",
            "probe_failures",
            "failed_insertions",
            "cell_clamp_events",
            "invariant_violations",
            "outranked_discards",
            "input_overflow_frames",
            "fallback_frames",
            "identity_frames",
        ):
            self.assertEqual(cleared[key], 0, f"{key} survived clear")

    def test_telemetry_rejected_during_capture(self):
        """Reject stats reads and clears while a CUDA graph capture is active.

        stats() would die on its device sync anyway, but cryptically;
        clear_stats() would silently record its zeroing into the graph and
        wipe telemetry on every replay.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # warm-up
        with wp.ScopedCapture(device) as capture:
            # Record real reducer work first so the guard is exercised during
            # a non-empty capture, not only in an otherwise empty graph.
            pipeline.collide(state, contacts)
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                pipeline.body_pair_reduction_stats()
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                pipeline.clear_body_pair_reduction_stats()
        self.assertIsNotNone(capture.graph)

    def test_property_schema_switch_is_safe(self):
        """Reduce a property-enabled buffer after a property-less one safely.

        The first reduce installs zero-length material placeholders in the
        gather scratch; a later same-capacity buffer WITH per-contact material
        arrays must re-provision them or the gather writes out of bounds.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        plain = pipeline.contacts()
        pipeline.collide(state, plain)  # installs the property-less scratch

        rich = Contacts(
            rigid_contact_max=pipeline.rigid_contact_max,
            soft_contact_max=0,
            device=wp.get_device(),
            per_contact_shape_properties=True,
        )
        pipeline.collide(state, rich)  # must not write into zero-length arrays
        n = int(rich.rigid_contact_count.numpy()[0])
        self.assertGreater(n, 0)
        self.assertTrue(np.isfinite(rich.rigid_contact_friction.numpy()[:n]).all())

    def test_deterministic_sort_keeps_rich_properties_aligned(self):
        """Permute an external rich buffer's material triple with its geometry.

        Reducer pipelines normally allocate property-less Contacts, but their
        deterministic sorter must still provision material scratch for a rich
        external buffer. Otherwise the subsequent reducer receives stiffness,
        damping, and friction values attached to the wrong contact rows.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for k in range(4):
            builder.add_shape_sphere(-1, xform=wp.transform(wp.vec3(float(k), 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize(device=wp.get_device())
        pipeline = _make_pipeline(model, True, rigid_contact_max=4)
        contacts = Contacts(
            rigid_contact_max=4,
            soft_contact_max=0,
            device=wp.get_device(),
            per_contact_shape_properties=True,
        )
        contacts.rigid_contact_count.assign(np.array([4], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.arange(4, dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.full(4, -1, dtype=np.int32))
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        zeros3 = np.zeros((4, 3), dtype=np.float32)
        contacts.rigid_contact_point0.assign(points)
        contacts.rigid_contact_point1.assign(points + 0.5)
        contacts.rigid_contact_offset0.assign(zeros3)
        contacts.rigid_contact_offset1.assign(zeros3)
        contacts.rigid_contact_normal.assign(np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1)))
        contacts.rigid_contact_margin0.assign(np.zeros(4, dtype=np.float32))
        contacts.rigid_contact_margin1.assign(np.zeros(4, dtype=np.float32))
        contacts.rigid_contact_tids.assign(np.arange(4, dtype=np.int32))
        contacts.rigid_contact_stiffness.assign(np.arange(100.0, 104.0, dtype=np.float32))
        contacts.rigid_contact_damping.assign(np.arange(200.0, 204.0, dtype=np.float32))
        contacts.rigid_contact_friction.assign(np.arange(300.0, 304.0, dtype=np.float32))
        pipeline._sort_key_array.assign(np.array([40, 30, 20, 10], dtype=np.int64))

        pipeline._contact_sorter.sort_full(
            pipeline._sort_key_array,
            contacts.rigid_contact_count,
            shape0=contacts.rigid_contact_shape0,
            shape1=contacts.rigid_contact_shape1,
            point0=contacts.rigid_contact_point0,
            point1=contacts.rigid_contact_point1,
            offset0=contacts.rigid_contact_offset0,
            offset1=contacts.rigid_contact_offset1,
            normal=contacts.rigid_contact_normal,
            margin0=contacts.rigid_contact_margin0,
            margin1=contacts.rigid_contact_margin1,
            tids=contacts.rigid_contact_tids,
            stiffness=contacts.rigid_contact_stiffness,
            damping=contacts.rigid_contact_damping,
            friction=contacts.rigid_contact_friction,
            device=wp.get_device(),
        )
        np.testing.assert_array_equal(contacts.rigid_contact_shape0.numpy(), [3, 2, 1, 0])
        np.testing.assert_allclose(contacts.rigid_contact_stiffness.numpy(), [103.0, 102.0, 101.0, 100.0])
        np.testing.assert_allclose(contacts.rigid_contact_damping.numpy(), [203.0, 202.0, 201.0, 200.0])
        np.testing.assert_allclose(contacts.rigid_contact_friction.numpy(), [303.0, 302.0, 301.0, 300.0])

    def test_clear_resets_reduced_provenance(self):
        """Reset the reduced marker when the buffer is cleared.

        A cleared buffer holds no contacts at all; keeping the stale marker
        would make unsupported solvers reject an empty buffer.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
        contacts.clear()
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)

    def test_malformed_reset_masks_rejected(self):
        """Reject reset masks with the wrong length, rank, or dtype.

        A short mask would silently reset only a prefix of worlds.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for _w in range(2):
            builder.begin_world()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        pipeline = _make_pipeline(model, True)
        for bad in (
            np.array([1], dtype=np.int32),  # short
            np.array([1, 0, 1], dtype=np.int32),  # long
            np.array([[1], [0]], dtype=np.int32),  # wrong rank
            np.array([1.0, 0.0]),  # float: int conversion would truncate silently
        ):
            with self.assertRaises(ValueError, msg=f"accepted malformed mask {bad!r}"):
                pipeline.reset_body_pair_reduction_history(bad)

    def test_reduced_marker_tracks_pipeline_mode(self):
        """Assign buffer provenance from the pipeline mode on every collide.

        A sticky marker would make an unsupported solver reject a buffer that
        was later refilled by an ordinary pipeline.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        reduced = _make_pipeline(model, True)
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = reduced.contacts()
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
        reduced.collide(state, contacts)
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
        plain.collide(state, contacts)
        self.assertFalse(
            contacts.rigid_contacts_body_pair_reduced,
            "marker must clear when an ordinary pipeline refills",
        )


class TestBodyPairReductionRobustness(unittest.TestCase):
    """Remaining contract edges: matching, capacity, graph capture, grad, worlds."""

    def _foot_scene(self, pos=(5.13, 5.07, 0.0149)):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, pos)
        builder.add_ground_plane()
        return builder.finalize(device=wp.get_device())

    def test_sticky_matching_rejected_at_construction(self):
        """Reject replay that can change geometry after selecting support extremes."""
        model = self._foot_scene()
        with self.assertRaisesRegex(ValueError, "contact_matching"):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="sticky",
                reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
            )

    def test_group_id_capacity_rejected_at_construction(self):
        """Reject scenes whose reduction-group count exceeds the exact key budget.

        Group ids pack exactly into the reduction key; overflow would alias
        two groups and could evict a patch's deepest contact, so the pipeline
        must refuse at construction rather than mask bits at runtime. Groups
        are material-equivalence classes: five bodies with distinct friction
        coefficients plus the ground are six groups against a patched budget
        of five (ids are 0-based, so a patched MAX_GROUP_ID of 4 admits five).
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for k in range(5):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.2 * k, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01, cfg=newton.ModelBuilder.ShapeConfig(mu=0.1 * (k + 1)))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        with unittest.mock.patch("newton._src.sim.collide.MAX_GROUP_ID", 4):
            with self.assertRaisesRegex(ValueError, "at most 5 reduction groups"):
                _make_pipeline(model, True)

    def test_cuda_graph_capture(self):
        """Replay a captured collide() on changing states against an uncaptured reference.

        All reduction launches are fixed-size, so capture must succeed, and a
        replay must equal an ordinary collide that saw the same state history
        (hysteresis makes the kept set history-dependent, so the reference
        pipeline is stepped through the identical sequence).  Every
        solver-visible contact field is compared, on the settled pose and
        again after pitching the foot in place.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = Contacts(
            rigid_contact_max=pipeline.rigid_contact_max,
            soft_contact_max=pipeline.soft_contact_max,
            device=device,
            per_contact_shape_properties=True,
        )
        pipeline.collide(state, contacts)  # warm-up: lazy allocations + kernel loads
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)

        # reference: same model, fresh pipeline, identical executed history
        # (capture itself records without executing: warm-up + two replays)
        state_ref = model.state()
        ref = _make_pipeline(model, True)
        contacts_ref = Contacts(
            rigid_contact_max=ref.rigid_contact_max,
            soft_contact_max=ref.soft_contact_max,
            device=device,
            per_contact_shape_properties=True,
        )
        for _ in range(3):
            ref.collide(state_ref, contacts_ref)
        self.assertEqual(_full_contact_rows(contacts, canonical=True), _full_contact_rows(contacts_ref, canonical=True))
        self.assertEqual(_full_contact_rows(contacts), _full_contact_rows(contacts_ref), "captured row order changed")

        # mutate the captured state IN PLACE (the graph holds the pointers):
        # pitch shifts depth ownership across the foot, changing the kept set
        q = state.body_q.numpy()
        pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.05)
        q[0][3:7] = [pitch[0], pitch[1], pitch[2], pitch[3]]
        state.body_q.assign(q)
        state_ref.body_q.assign(q)
        wp.capture_launch(capture.graph)
        rows_before = _full_contact_rows(contacts_ref, canonical=True)
        ref.collide(state_ref, contacts_ref)
        rows_after = _full_contact_rows(contacts_ref, canonical=True)
        self.assertNotEqual(
            rows_before, rows_after, "pitch did not change the reference output; the changing-state leg is vacuous"
        )
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertEqual(_full_contact_rows(contacts, canonical=True), rows_after)
        self.assertEqual(_full_contact_rows(contacts), _full_contact_rows(contacts_ref), "captured row order changed")

    def test_second_reducer_graph_and_buffer_rejected(self):
        """Refuse a second live reducer-writer graph or a second buffer.

        Graph replay repeats neither the buffer-switch history reset nor the
        provenance assignment, so two captured buffers would silently share
        hysteresis history; collide() must raise at capture time instead --
        BEFORE any state mutation: the rejected buffer and the reducer's
        history must come back untouched.  Even the same buffer cannot be
        recaptured while its first graph is live: independent replays would
        race the reducer history, telemetry, and output rows.  Skipped on CPU
        devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts_a = pipeline.contacts()
        contacts_b = pipeline.contacts()
        pipeline.collide(state, contacts_a)  # warm-up
        with wp.ScopedCapture(device) as capture_a:
            pipeline.collide(state, contacts_a)
        wp.capture_launch(capture_a.graph)
        red = pipeline._body_pair_reducer
        prev_keys_before = red.prev_keys.numpy().copy()
        generations_before = red.history_generation.numpy().copy()
        # A second graph for the same buffer is unsafe even though its pointers
        # match: either graph executable could be launched independently.
        capture_a2 = wp.ScopedCapture(device)
        capture_a2.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "second body-pair reducer graph"):
                pipeline.collide(state, contacts_a)
        finally:
            capture_a2.__exit__(None, None, None)
        # different buffer: must raise while capture is active; end the
        # capture outside the exception path so the stream is left clean
        capture_b = wp.ScopedCapture(device)
        capture_b.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "per captured buffer"):
                pipeline.collide(state, contacts_b)
        finally:
            capture_b.__exit__(None, None, None)
        self.assertEqual(int(contacts_b.rigid_contact_count.numpy()[0]), 0, "rejected buffer was written")
        self.assertFalse(contacts_b.rigid_contacts_body_pair_reduced, "rejected buffer was stamped")
        np.testing.assert_array_equal(red.prev_keys.numpy(), prev_keys_before)
        np.testing.assert_array_equal(red.history_generation.numpy(), generations_before)
        # Dropping the exclusive graph clears capture provenance and unlocks
        # the pipeline, while preserving conservative ordinary provenance for
        # rows left by the final replay.
        del capture_a
        gc.collect()
        self.assertFalse(contacts_a.rigid_contacts_body_pair_reduced_capture)
        self.assertTrue(contacts_a.rigid_contacts_body_pair_reduced)
        pipeline.release_body_pair_reduction_capture()

    def test_ordinary_other_buffer_rejected_while_graph_live(self):
        """Refuse ANY other buffer while a captured graph's binding is live.

        An ordinary (uncaptured) collide with buffer B after capturing buffer
        A passes a capture-time-only check, then resets and repopulates the
        shared hysteresis state -- every subsequent replay of A's graph runs
        against B's history.  The binding must reject other buffers whether
        or not the current call is being captured.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts_a = pipeline.contacts()
        contacts_b = pipeline.contacts()
        pipeline.collide(state, contacts_a)  # warm-up
        with wp.ScopedCapture(device) as capture_a:
            pipeline.collide(state, contacts_a)
        wp.capture_launch(capture_a.graph)
        with self.assertRaisesRegex(RuntimeError, "per captured buffer"):
            pipeline.collide(state, contacts_b)
        # the captured buffer itself stays usable outside capture
        pipeline.collide(state, contacts_a)

    def test_capture_requires_warmed_buffer(self):
        """Refuse to capture a Contacts buffer that never collided outside capture.

        Capturing cold reaches the buffer-switch history reset inside the
        capture, recording its fills and launches into the graph -- every
        replay then erases hysteresis before reducing, silently restoring the
        memoryless behavior the graph's owner did not ask for.  Skipped on
        CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        capture = wp.ScopedCapture(device)
        capture.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "outside capture before capturing"):
                pipeline.collide(state, contacts)
        finally:
            capture.__exit__(None, None, None)

    def test_capture_release_allows_new_buffer(self):
        """Destroy every graph lease before rebinding a new buffer.

        Explicit release must refuse to detach arrays from a live graph.  Once
        the graph is destroyed its lease releases automatically, but ordinary
        reduced provenance stays conservative until a new collide or clear.
        Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts_a = pipeline.contacts()
        contacts_b = pipeline.contacts()
        pipeline.collide(state, contacts_a)
        with wp.ScopedCapture(device) as capture_a:
            pipeline.collide(state, contacts_a)
        wp.capture_launch(capture_a.graph)
        with self.assertRaisesRegex(RuntimeError, "graph is still live"):
            pipeline.release_body_pair_reduction_capture()
        del capture_a
        gc.collect()
        pipeline.release_body_pair_reduction_capture()
        self.assertFalse(contacts_a.rigid_contacts_body_pair_reduced_capture)
        self.assertTrue(
            contacts_a.rigid_contacts_body_pair_reduced,
            "destroying the last graph must conservatively describe rows left by its final replay",
        )
        pipeline.collide(state, contacts_b)  # warm-up now legal
        with wp.ScopedCapture(device) as capture_b:
            pipeline.collide(state, contacts_b)
        wp.capture_launch(capture_b.graph)

    def test_captured_buffer_keeps_conservative_provenance(self):
        """Keep rejecting unsupported solvers after a non-reducing refill of a captured buffer.

        Replay is invisible to host-side provenance: a plain pipeline can
        refill a captured buffer and stamp it unreduced, then a replay of the
        reducer graph compacts the device records again -- an unsupported
        solver reading the plain marker would consume reduced contacts
        without raising.  A buffer with any live reducer-writer graph lease
        carries a may-contain-reduced marker until the last graph is
        destroyed.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state_0, contacts)
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state_0, contacts)
        wp.capture_launch(capture.graph)
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        plain.collide(state_0, contacts)
        self.assertFalse(
            contacts.rigid_contacts_body_pair_reduced,
            "plain refill should clear the per-collide marker",
        )
        # Recreate the exact dangerous state: replay compacts the device rows,
        # but no Python collide runs to restamp ordinary provenance.
        wp.capture_launch(capture.graph)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced_capture)
        solver = newton.solvers.SolverSemiImplicit(model)
        with self.assertRaisesRegex(ValueError, "supports_body_pair_reduced_contacts"):
            solver.step(state_0, state_1, model.control(), contacts, DT)

    def test_graph_lease_retains_pipeline_and_contacts(self):
        """The graph itself owns every reducer/contact array it recorded."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline_ref = weakref.ref(pipeline)
        contacts_ref = weakref.ref(contacts)
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state, contacts)
        graph = capture.graph
        del capture, contacts, pipeline
        gc.collect()
        self.assertIsNotNone(pipeline_ref(), "graph lease did not retain reducer arrays")
        self.assertIsNotNone(contacts_ref(), "graph lease did not retain Contacts arrays")
        del graph
        gc.collect()
        self.assertIsNone(pipeline_ref())
        self.assertIsNone(contacts_ref())

    def test_reducer_writer_graph_is_exclusive_across_pipelines(self):
        """Two pipelines cannot capture writers for the same contact buffer."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline_a = _make_pipeline(model, True)
        pipeline_b = _make_pipeline(model, True)
        contacts = pipeline_a.contacts()
        pipeline_a.collide(state, contacts)
        pipeline_b.collide(state, contacts)
        with wp.ScopedCapture(device) as capture_a:
            pipeline_a.collide(state, contacts)
        capture_b_rejected = wp.ScopedCapture(device)
        capture_b_rejected.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "second body-pair reducer graph"):
                pipeline_b.collide(state, contacts)
        finally:
            capture_b_rejected.__exit__(None, None, None)

        del capture_a
        gc.collect()
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
        pipeline_a.release_body_pair_reduction_capture()
        # Once the first graph is truly gone, the already-warmed second
        # pipeline may acquire the sole writer lease.
        with wp.ScopedCapture(device) as capture_b:
            pipeline_b.collide(state, contacts)
        wp.capture_launch(capture_b.graph)
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced_capture)
        del capture_b
        gc.collect()
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
        pipeline_b.release_body_pair_reduction_capture()

    def test_precaptured_unsupported_solver_blocks_reducer_writer(self):
        """A raw-contact solver graph cannot later replay over reduced rows."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state_0, state_1 = model.state(), model.state()
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = plain.contacts()
        plain.collide(state_0, contacts)
        solver = newton.solvers.SolverSemiImplicit(model)
        control = model.control()
        solver.step(state_0, state_1, control, contacts, DT)  # warm-up
        with wp.ScopedCapture(device) as solver_capture:
            solver.step(state_0, state_1, control, contacts, DT)

        reducer = _make_pipeline(model, True)
        with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
            reducer.collide(state_0, contacts)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
        del solver_capture
        gc.collect()
        reducer.collide(state_0, contacts)

    def test_precaptured_feather_pgs_warmstart_blocks_reducer_writer(self):
        """Protect a warm-start FPGS graph from later body-pair compaction.

        FeatherPGS supports reduced contacts in its default configuration, but
        identity warm start has not been validated with compaction. Its per-call
        override must therefore acquire the same unreduced-reader lease as an
        unsupported solver before graph replay becomes possible.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state_0, state_1 = model.state(), model.state()
        plain = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        contacts = plain.contacts()
        plain.collide(state_0, contacts)
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="split", pgs_warmstart=True)
        control = model.control()
        solver.step(state_0, state_1, control, contacts, DT)
        with wp.ScopedCapture(device) as solver_capture:
            solver.step(state_0, state_1, control, contacts, DT)

        reducer = _make_pipeline(model, True)
        with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
            reducer.collide(state_0, contacts)
        del solver_capture
        gc.collect()
        reducer.collide(state_0, contacts)

    def test_capture_lifecycle_mutations_are_guarded(self):
        """Host topology/reset/release mutations never enter a CUDA graph."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        reset_mask = wp.zeros(max(int(model.world_count), 1), dtype=wp.int32, device=device)
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state, contacts)
            with self.assertRaisesRegex(RuntimeError, "no CUDA graph capture"):
                pipeline.refresh_body_pair_reduction_groups()
            with self.assertRaisesRegex(RuntimeError, "host mask"):
                pipeline.reset_body_pair_reduction_history(None)
            with self.assertRaisesRegex(RuntimeError, "host mask"):
                pipeline.reset_body_pair_reduction_history(np.ones(reset_mask.shape[0], dtype=np.int32))
            # Device masks are the intentionally capture-safe reset path.
            pipeline.reset_body_pair_reduction_history(reset_mask)
            with self.assertRaisesRegex(RuntimeError, "lifecycle mutation"):
                pipeline.release_body_pair_reduction_capture()
        self.assertIsNotNone(capture.graph)

    def test_requires_grad_diff_augmentation(self):
        """Evaluate finite differentiable contact distances on the compacted set."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.0149))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device(), requires_grad=True)
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model,
            broad_phase="nxn",
            deterministic=True,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
            requires_grad=True,
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        n = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(n, 0)
        distance = wp.zeros(contacts.rigid_contact_max, dtype=float, device=model.device, requires_grad=True)
        newton.eval_rigid_contact_kinematics(model, state, contacts, out_distance=distance)
        d = distance.numpy()[:n]
        self.assertTrue(np.isfinite(d).all())

    def test_multi_world_independence(self):
        """Reduce two worlds' identical feet to identical per-world kept sets.

        Worlds must not share groups: each world's foot keeps its own deepest
        and extremes, and the kept count is the same for identical scenes.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for _ in range(2):
            builder.begin_world()
            _cylinder_foot(builder, (5.13, 5.07, 0.0149))
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        red = _collide_once(model, state, reduce_body_pairs=True, body_pair_cell_size=10.0)
        n, s0, s1, _, _ = _contact_snapshot(red)
        shape_body = model.shape_body.numpy()
        per_world = [0, 0]
        for k in range(n):
            b = shape_body[s0[k]] if shape_body[s0[k]] >= 0 else shape_body[s1[k]]
            per_world[0 if b < model.body_count // 2 else 1] += 1
        self.assertEqual(per_world[0], per_world[1])
        self.assertGreaterEqual(per_world[0], 3)
        self.assertLessEqual(per_world[0], 7)


class TestBodyPairReductionFPGSImpact(unittest.TestCase):
    """FPGS-specific touchdown conformance (the G1 failure mode that unit XPBD tests missed)."""

    def test_fpgs_touchdown_capture(self):
        """Land a falling foot on FPGS at the same rest height with and without reduction.

        This covers the exact defect class found on the walking humanoid: a
        misranked kept set leaves no load-bearing contact at touchdown and
        the body free-falls before a violent late landing.
        """

        def run(reduce_on):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = _free_jointed_foot(builder, (5.13, 5.07, 0.06))  # 3 cm drop
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipeline = _make_pipeline(model, reduce_on)
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
            zs = []
            for _ in range(240):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, DT)
                state_0, state_1 = state_1, state_0
                zs.append(float(state_0.body_q.numpy()[body][2]))
            stats = pipeline.body_pair_reduction_stats() if reduce_on else None
            return np.array(zs), stats

        z_off, _ = run(False)
        z_on, stats = run(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "touchdown contacts were never reduced")
        self.assertLess(float(z_off.min()), 0.04, "foot never fell: the solver is not simulating it")
        self.assertLess(abs(float(z_off[-30:].mean()) - float(z_on[-30:].mean())), 5e-4)
        # bounded touchdown transient: never punches through the resting height by > 2.5 mm
        self.assertGreater(float(z_on.min()), 0.015 - 2.5e-3)


class TestBodyPairReductionGroupAssignment(unittest.TestCase):
    """Group assignment must be stable at world axes and under rigid translation."""

    def test_grouping_uses_effective_surface_midpoint(self):
        """Keep unequal-radius contacts together by their physical surface center.

        The two synthetic records have different raw support-point midpoints
        but exactly the same midpoint after applying their directed surface
        margins. The tilted normal gives the raw displacement a component on
        the selected bin plane, large enough to cross the 2 cm cell grid. A
        raw-midpoint implementation therefore creates two table entries; the
        effective-surface implementation creates one.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(xform=wp.transform_identity(), mass=1.0)
        shape_a = builder.add_shape_sphere(body, radius=0.01)
        shape_b = builder.add_shape_sphere(body, radius=0.02)
        ground = builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(
            model,
            True,
            rigid_contact_max=2,
            body_pair_cell_size=0.02,
            body_pair_hysteresis=0.0,
        )
        contacts = pipeline.contacts()

        normal = np.array([0.3, 0.0, np.sqrt(1.0 - 0.3**2)], dtype=np.float32)
        margins0 = np.array([0.0, 0.2], dtype=np.float32)
        raw_centers = np.stack([np.zeros(3, dtype=np.float32), -0.5 * margins0[1] * normal])
        contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
        contacts.rigid_contact_shape0.assign(np.array([shape_a, shape_b], dtype=np.int32))
        contacts.rigid_contact_shape1.assign(np.array([ground, ground], dtype=np.int32))
        contacts.rigid_contact_point0.assign(raw_centers)
        contacts.rigid_contact_point1.assign(raw_centers)
        contacts.rigid_contact_normal.assign(np.stack([normal, normal]))
        contacts.rigid_contact_margin0.assign(margins0)
        contacts.rigid_contact_margin1.assign(np.zeros(2, dtype=np.float32))

        reducer = pipeline._body_pair_reducer
        reducer.reduce(model, state, contacts)
        stats = reducer.stats()
        self.assertEqual(stats["max_hashtable_entries"], 1)
        np.testing.assert_allclose(reducer.contact_pos2d.numpy()[0], reducer.contact_pos2d.numpy()[1], atol=1e-6)

    def test_bin_margins_at_world_axes(self):
        """Keep every world axis well inside a bin, especially the ground normal.

        The shared icosahedron table stores the solid Y-up, which places +Z on a
        face boundary to within 7e-8 of dot product: any curved or mesh surface
        in a Z-up world then flickers its patch normals between two bins step to
        step, and the kept set churns (measured as a permanent rocking limit
        cycle). The body-pair table is rotated so a face CENTER points at +/-Z;
        this pins that property plus a sane margin at the horizontal axes.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(faces, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-6), "face normals must be unit length")

        def margin(d):
            dots = np.sort(faces @ np.asarray(d, dtype=np.float64))[::-1]
            return dots[0] - dots[1]

        for axis in ((0, 0, 1), (0, 0, -1)):
            self.assertGreater(margin(axis), 0.2, f"ground-normal margin too small at {axis}")
        for axis in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)):
            self.assertGreater(margin(axis), 0.05, f"horizontal margin too small at {axis}")

    def test_bin_stable_under_ground_normal_wobble(self):
        """Assign every normal within 15 degrees of +Z to the same single bin.

        Emulates curved/mesh terrain: patch normals wobble around world-up. If
        any of them crosses a bin boundary, contacts of one patch regroup
        between steps and the kept set flickers.
        """
        faces = np.array(_BP_FACE_NORMALS_DATA, dtype=np.float64).reshape(-1, 3)
        rng = np.random.default_rng(5)
        tilt = np.radians(rng.uniform(0.0, 15.0, size=512))
        yaw = rng.uniform(0.0, 2.0 * np.pi, size=512)
        normals = np.stack(
            [np.sin(tilt) * np.cos(yaw), np.sin(tilt) * np.sin(yaw), np.cos(tilt)],
            axis=1,
        )
        bins = np.argmax(normals @ faces.T, axis=1)
        self.assertEqual(len(np.unique(bins)), 1, "normals near +Z must all share one bin")

    def test_kept_set_invariant_under_translation(self):
        """Keep the identical relative contact set after translating the scene.

        Symmetric collider layouts tie in the scan directions constantly (a row
        of foot cylinders shares a coordinate exactly). The old tie-break hashed
        contact content, and a static shape's witness point is world-space, so
        pure translation flipped winners: a foot sliding in a straight line
        churned its kept set. The geometric tie-break is measured in the
        pair-anchored face frame and cannot see translation.
        """
        offsets = ((0.0, 0.0), (0.37, 0.53), (1.24, -0.86))
        kept_sets = []
        for ox, oy in offsets:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            body = _cylinder_foot(builder, (ox, oy, 0.0175))
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state = model.state()
            contacts = _collide_once(model, state, reduce_body_pairs=True)
            n, s0, s1, _nrm, _p0 = _contact_snapshot(contacts)
            self.assertGreater(n, 0)
            # canonicalize in the BODY frame: witness points on the static plane
            # are stored world-space and legitimately move with the scene, so
            # compare positions relative to the foot instead.
            pts = _world_points0(model, state, contacts) - state.body_q.numpy()[body][:3]
            rows = sorted(
                (int(a), int(b), *(round(float(v), 5) for v in p)) for a, b, p in zip(s0, s1, pts, strict=True)
            )
            kept_sets.append(rows)
        self.assertEqual(kept_sets[0], kept_sets[1], "kept set changed under translation")
        self.assertEqual(kept_sets[0], kept_sets[2], "kept set changed under translation")


class TestBodyPairReductionHysteresis(unittest.TestCase):
    """Temporal hysteresis: kept-set continuity without sticky wrong winners."""

    def _kept_set_handoffs(self, hysteresis, steps=120):
        """Winner handoffs of a micro-rocked plate on curved support, solver-free.

        A sphere-grid plate on a 20 m dome has near-degenerate winner scores
        (the gap spread across the plate is ~0.14 mm).  The plate is rocked
        KINEMATICALLY with a 5 mrad tilt oscillation -- contact points move
        ~0.4 mm per cycle, under the 0.5 mm identity quantum so incumbency can
        attach, and the score shifts stay well inside the 1 mm margin.  The
        kept set is identified by WHICH plate spheres survive; a handoff is
        any step whose kept multiset differs from the previous step's.  No
        solver runs, so this measures the reducer's own churn -- the quantity
        hysteresis exists to remove -- independent of any solver's support
        status.
        """
        r_ground = 20.0
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_shape_sphere(
            -1, xform=wp.transform(wp.vec3(0.0, 0.0, -r_ground), wp.quat_identity()), radius=r_ground
        )
        body = _sphere_grid_body(builder, (0.0, 0.0, 0.0095), n=4, spacing=0.05)
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True, body_pair_hysteresis=hysteresis)
        contacts = pipeline.contacts()
        q0 = state.body_q.numpy().copy()
        prev, handoffs = None, 0
        for k in range(steps):
            theta = 0.005 * np.sin(2.0 * np.pi * k / 40.0)
            rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(theta))
            q = q0.copy()
            q[body][3:7] = [rot[0], rot[1], rot[2], rot[3]]
            state.body_q.assign(q)
            pipeline.collide(state, contacts)
            n = int(contacts.rigid_contact_count.numpy()[0])
            rows = tuple(sorted(contacts.rigid_contact_shape1.numpy()[:n].tolist()))
            if prev is not None and rows != prev:
                handoffs += 1
            prev = rows
        return handoffs

    def test_hysteresis_removes_winner_churn_on_curved_support(self):
        """Stop near-degenerate winner handoffs on curved support.

        Memoryless selection hands the kept set off repeatedly as the rock
        sweeps the near-tied scores (measured 18 handoffs in 119 steps); the
        1 mm incumbency margin removes ALL of them.  This is the reducer-level
        mechanism behind the settle-churn the hysteresis feature was built
        for, pinned without a solver in the loop.
        """
        churn_off = self._kept_set_handoffs(hysteresis=0.0)
        churn_on = self._kept_set_handoffs(hysteresis=0.001)
        self.assertGreaterEqual(churn_off, 10, "scene stopped churning; the probe is vacuous")
        self.assertEqual(churn_on, 0, f"hysteresis left {churn_on} winner handoffs")

    def test_challenger_beyond_margin_wins_immediately(self):
        """Dethrone an incumbent depth winner as soon as a clearly deeper one exists.

        Hysteresis must only stop near-tie handoffs. A three-cylinder foot
        whose MIDDLE cylinder sits 2 mm lower is kept only through the depth
        slot (both end cylinders own every spatial extreme). Establish
        incumbency, then pitch the body so an end cylinder becomes deeper by
        several times the 1 mm margin while every contact's in-plane position
        moves less than the 0.5 mm identity quantum -- incumbency stays attached
        AND must lose: the middle contact has to vanish from the kept set on
        the very next collide.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0195), wp.quat_identity()), mass=1.0)
        for i in range(3):
            z_local = -0.002 if i == 1 else 0.0
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3((i - 1.0) * 0.05, 0.0, z_local), wp.quat_identity()),
                radius=0.02,
                half_height=0.015,
            )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # establish winners
        pipeline.collide(state, contacts)  # winners now incumbents
        pts = _world_points0(model, state, contacts)
        mid = np.abs(pts[:, 0]) < 0.02
        self.assertTrue(bool(mid.any()), "middle cylinder should be kept as the depth winner")

        # pitch about y: the -x end cylinder gains ~5 mm of depth over the
        # middle (>> 1 mm hysteresis), while in-plane positions shift by
        # 0.05 * (1 - cos 0.1) ~ 0.25 mm (< the 0.5 mm identity quantum)
        q = state.body_q.numpy()
        pitch = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.1)
        q[0][3:7] = [pitch[0], pitch[1], pitch[2], pitch[3]]
        state.body_q.assign(q)
        pipeline.collide(state, contacts)
        pts = _world_points0(model, state, contacts)
        mid = np.abs(pts[:, 0]) < 0.02
        self.assertFalse(bool(mid.any()), "beaten incumbent depth winner was kept anyway")

    def test_reset_history_matches_fresh_pipeline(self):
        """Produce a fresh pipeline's kept set on the first collide after a reset.

        Hysteresis history is trajectory state; episode resets and teleports
        must be able to sever it so no incumbency bonus crosses the boundary.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # establish history
        pipeline.reset_body_pair_reduction_history()
        pipeline.collide(state, contacts)
        n_reset, s0_r, s1_r, _n1, p0_r = _contact_snapshot(contacts)

        fresh = _make_pipeline(model, True)
        contacts_f = fresh.contacts()
        fresh.collide(state, contacts_f)
        n_f, s0_f, s1_f, _n2, p0_f = _contact_snapshot(contacts_f)

        self.assertEqual(n_reset, n_f)
        rows_r = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_r, s1_r, p0_r, strict=True)
        )
        rows_f = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_f, s1_f, p0_f, strict=True)
        )
        self.assertEqual(rows_r, rows_f, "reset did not sever hysteresis history")

    def test_new_contacts_buffer_resets_history(self):
        """Sever hysteresis history automatically when a different buffer is supplied.

        A different Contacts instance means a different stream of states;
        winners recorded for the previous buffer must not bias the new one.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts_a = pipeline.contacts()
        pipeline.collide(state, contacts_a)
        contacts_b = pipeline.contacts()
        pipeline.collide(state, contacts_b)  # must not inherit buffer A's winners
        red = pipeline._body_pair_reducer
        masks = red.contact_incumbent.numpy()[: int(contacts_b.rigid_contact_count.numpy()[0])]
        self.assertTrue((masks == 0).all(), "incumbency leaked across Contacts buffers")

    def test_reducer_launches_stay_off_the_tape(self):
        """Record no reducer bookkeeping launches on an active autodiff tape.

        Every reducer kernel has backward disabled; recording them would only
        bloat the tape. This guards the full reduce() path including the
        hashtable maintenance kernels.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)  # warm up allocations outside the tape
        red = pipeline._body_pair_reducer
        tape = wp.Tape()
        with tape:
            red.reduce(model, state, contacts)
        self.assertEqual(len(tape.launches), 0, "reducer bookkeeping was recorded on the tape")

    def test_per_world_reset_severs_only_masked_worlds(self):
        """Reset one world's hysteresis history and leave the other's intact.

        Vectorized RL teleports individual environments mid-rollout; a global
        reset would sever every environment's history for one env's reset.
        The masked form advances only the selected worlds' generations: their
        contacts stop matching the snapshot while other worlds keep their
        incumbency bits.
        """
        # one sphere per world: identity frames skip compaction, so the
        # incumbency-mask cache (pre-compaction indexed) aligns with the
        # delivered contacts and worlds can be attributed exactly
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for w in range(2):
            builder.begin_world()
            body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * w, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        self.assertGreaterEqual(model.world_count, 2)
        state = model.state()

        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)  # winners are now incumbents in both worlds

        def masks_by_world():
            red = pipeline._body_pair_reducer
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            sw = model.shape_world.numpy()
            masks = red.contact_incumbent.numpy()[:n]
            out = {0: [], 1: []}
            for k in range(n):
                w = max(int(sw[s0[k]]), int(sw[s1[k]]))
                out[w].append(int(masks[k]))
            return out

        before = masks_by_world()
        self.assertTrue(any(m != 0 for m in before[0]) and any(m != 0 for m in before[1]))

        mask = wp.array(np.array([1, 0], dtype=np.int32), dtype=wp.int32, device=wp.get_device())
        pipeline.reset_body_pair_reduction_history(mask)
        pipeline.collide(state, contacts)
        after = masks_by_world()
        self.assertTrue(all(m == 0 for m in after[0]), "reset world kept incumbency")
        self.assertTrue(any(m != 0 for m in after[1]), "unreset world lost incumbency")

    def test_wide_integer_reset_mask_is_normalized(self):
        """Reset a world selected by a mask value that would truncate to int32 zero.

        ``uint64(2**32)`` is nonzero but its low 32 bits are zero; a raw int32
        conversion would silently turn the reset into a no-op for that world.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        for w in range(2):
            builder.begin_world()
            body = builder.add_body(xform=wp.transform(wp.vec3(2.0 * w, 0.0, 0.0095), wp.quat_identity()), mass=1.0)
            builder.add_shape_sphere(body, radius=0.01)
            builder.end_world()
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)  # winners are now incumbents

        def masks_by_world():
            red = pipeline._body_pair_reducer
            n = int(contacts.rigid_contact_count.numpy()[0])
            s0 = contacts.rigid_contact_shape0.numpy()[:n]
            s1 = contacts.rigid_contact_shape1.numpy()[:n]
            sw = model.shape_world.numpy()
            masks = red.contact_incumbent.numpy()[:n]
            out = {0: [], 1: []}
            for k in range(n):
                w = max(int(sw[s0[k]]), int(sw[s1[k]]))
                out[w].append(int(masks[k]))
            return out

        before = masks_by_world()
        self.assertTrue(any(m != 0 for m in before[0]) and any(m != 0 for m in before[1]))
        pipeline.reset_body_pair_reduction_history(np.array([2**32, 0], dtype=np.uint64))
        pipeline.collide(state, contacts)
        after = masks_by_world()
        self.assertTrue(all(m == 0 for m in after[0]), "wide nonzero mask truncated to a no-op")
        self.assertTrue(any(m != 0 for m in after[1]), "unmasked world lost incumbency")

    def test_disabled_hysteresis_is_history_independent(self):
        """Reduce identically with hysteresis=0 regardless of what ran before.

        The zero setting must restore the exact memoryless behavior: the same
        state reduced by a pipeline that saw a different previous step and by a
        fresh pipeline must produce identical kept sets.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_a = model.state()
        state_b = model.state()
        q = state_b.body_q.numpy()
        q[body][0] += 0.01  # a slightly different "previous" step
        state_b.body_q.assign(q)

        with_history = _make_pipeline(model, True, body_pair_hysteresis=0.0)
        contacts_h = with_history.contacts()
        with_history.collide(state_b, contacts_h)
        with_history.collide(state_a, contacts_h)

        fresh = _make_pipeline(model, True, body_pair_hysteresis=0.0)
        contacts_f = fresh.contacts()
        fresh.collide(state_a, contacts_f)

        n_h, s0_h, s1_h, _n1, p0_h = _contact_snapshot(contacts_h)
        n_f, s0_f, s1_f, _n2, p0_f = _contact_snapshot(contacts_f)
        self.assertEqual(n_h, n_f)
        rows_h = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_h, s1_h, p0_h, strict=True)
        )
        rows_f = sorted(
            (int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0_f, s1_f, p0_f, strict=True)
        )
        self.assertEqual(rows_h, rows_f)


class TestBodyPairReductionVerifier(unittest.TestCase):
    """The verify mode re-derives every keep/discard decision and finds zero disagreements."""

    def test_verifier_zero_violations_settling(self):
        """Verify the implementation invariant over a full settling trajectory.

        Every collide re-checks: no discarded contact out-ranks a slot winner
        it was eligible for, and every kept registered contact matches a slot
        it won. Any nonzero counter is a slot race, clearing bug, or ranking
        regression.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (5.13, 5.07, 0.020))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()
        pipeline = _make_pipeline(model, True, body_pair_verify=True)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        for _ in range(240):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["invariant_violations"], 0)
        self.assertEqual(stats["failed_insertions"], 0)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "verifier saw no discard")

    def test_kept_set_independent_of_previous_step(self):
        """Reduce a state to the same kept set whether or not a busier step preceded it.

        The pass carries capacity-sized scratch across steps -- keep flags, the
        per-contact hashtable-entry cache, and the per-entry slot values -- and
        refreshes only the live range each step, so nothing may depend on how
        busy the previous step was. Drive one pipeline through a three-foot step
        and then a one-foot step, and require the one-foot result to match a
        pipeline that only ever saw that state.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        feet = [_cylinder_foot(builder, (1.5 * k, 0.0, 0.020)) for k in range(3)]
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())

        state_busy = model.state()
        state_quiet = model.state()
        q = state_quiet.body_q.numpy()
        for b in feet[1:]:
            q[b][2] += 5.0  # lift two feet clear of the ground
        state_quiet.body_q.assign(q)

        with_history = _make_pipeline(model, True, body_pair_verify=True)
        contacts_h = with_history.contacts()
        with_history.collide(state_busy, contacts_h)
        n_busy = int(contacts_h.rigid_contact_count.numpy()[0])
        with_history.collide(state_quiet, contacts_h)

        fresh = _make_pipeline(model, True, body_pair_verify=True)
        contacts_f = fresh.contacts()
        fresh.collide(state_quiet, contacts_f)

        n_h, s0_h, s1_h, _nrm_h, p0_h = _contact_snapshot(contacts_h)
        n_f, s0_f, s1_f, _nrm_f, p0_f = _contact_snapshot(contacts_f)
        self.assertGreater(n_busy, n_f, "the busy step must really register more contacts")
        self.assertEqual(n_h, n_f, "kept count depends on the previous step")

        def canonical(s0, s1, p0):
            rows = [(int(a), int(b), *(round(float(v), 6) for v in p)) for a, b, p in zip(s0, s1, p0, strict=True)]
            return sorted(rows)

        self.assertEqual(canonical(s0_h, s1_h, p0_h), canonical(s0_f, s1_f, p0_f))
        for pipe in (with_history, fresh):
            stats = pipe._body_pair_reducer.stats()
            self.assertEqual(stats["invariant_violations"], 0)
            self.assertEqual(stats["outranked_discards"], 0)

    def test_property_random_piles(self):
        """Fuzz the invariant and stability on randomized multi-collider piles.

        Seeded random mixes of spheres, boxes, capsules, and cylinders are
        dropped into a pile -- geometry nobody hand-picked. Every free-jointed
        body has seven offset colliders, so the test cannot pass solely through
        the reducer's one-collider identity path. At least one frame must
        strictly reduce; the invariant verifier must stay clean; and the
        supported FeatherPGS consumer must remain finite and above the ground.
        """
        rng = np.random.default_rng(1234)
        for trial in range(3):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
            bodies = []
            for k in range(10):
                # spawn strictly non-overlapping (max shape extent is 0.16 m):
                # interpenetrating spawns make BOTH pipelines ballistic and the
                # comparison chaotic rather than physical
                pos = (float(rng.uniform(-0.15, 0.15)), float(rng.uniform(-0.15, 0.15)), 0.10 + 0.18 * k)
                body = builder.add_link(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
                bodies.append(body)
                kind = int(rng.integers(0, 4))
                radius = float(rng.uniform(0.012, 0.02))
                half_height = float(rng.uniform(0.012, 0.025))
                for j in range(7):
                    offset = wp.transform(wp.vec3((j - 3) * 2.2 * radius, 0.0, 0.0), wp.quat_identity())
                    if kind == 0:
                        builder.add_shape_sphere(body, xform=offset, radius=radius)
                    elif kind == 1:
                        builder.add_shape_box(body, xform=offset, hx=radius, hy=radius, hz=radius)
                    elif kind == 2:
                        builder.add_shape_capsule(body, xform=offset, radius=radius, half_height=half_height)
                    else:
                        builder.add_shape_cylinder(body, xform=offset, radius=radius, half_height=half_height)
                joint = builder.add_joint_free(parent=-1, child=body)
                builder.add_articulation([joint])
            builder.add_ground_plane()
            model = builder.finalize(device=wp.get_device())
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipe_red = _make_pipeline(model, True, body_pair_verify=True)
            pipe_raw = newton.CollisionPipeline(model, broad_phase="nxn")
            c_red, c_raw = pipe_red.contacts(), pipe_raw.contacts()
            # Seven colliders per body generate thousands of speculative rows.
            # Both consumers must retain them for this to test reduction stability.
            mf_capacity = 3 * max(c_red.rigid_contact_max, c_raw.rigid_contact_max)
            solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0, mf_max_constraints=mf_capacity)
            raw_not_less, strict_reduction_frames, peak_red = 0, 0, 0.0
            for _ in range(150):
                pipe_raw.collide(state_0, c_raw)
                pipe_red.collide(state_0, c_red)
                raw_count = int(c_raw.rigid_contact_count.numpy()[0])
                reduced_count = int(c_red.rigid_contact_count.numpy()[0])
                raw_not_less += int(raw_count >= reduced_count)
                strict_reduction_frames += int(raw_count > reduced_count)
                state_0.clear_forces()
                solver.step(state_0, state_1, control, c_red, DT)
                state_0, state_1 = state_1, state_0
                peak_red = max(peak_red, float(np.abs(state_0.body_qd.numpy()).max()))

            # Reference peak from the SAME pile and same solver family, driven
            # by unreduced contacts. An absolute bound flakes on a random pile,
            # whose peak speed can legitimately be large in either pipeline.
            raw_state_0, raw_state_1 = model.state(), model.state()
            newton.eval_fk(model, raw_state_0.joint_q, raw_state_0.joint_qd, raw_state_0)
            raw_control = model.control()
            raw_dynamics_pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            raw_dynamics_contacts = raw_dynamics_pipeline.contacts()
            raw_solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0, mf_max_constraints=mf_capacity)
            peak_raw = 0.0
            for _ in range(150):
                raw_dynamics_pipeline.collide(raw_state_0, raw_dynamics_contacts)
                raw_state_0.clear_forces()
                raw_solver.step(raw_state_0, raw_state_1, raw_control, raw_dynamics_contacts, DT)
                raw_state_0, raw_state_1 = raw_state_1, raw_state_0
                peak_raw = max(peak_raw, float(np.abs(raw_state_0.body_qd.numpy()).max()))

            solver.check_constraint_capacity()
            raw_solver.check_constraint_capacity()
            stats = pipe_red._body_pair_reducer.stats()
            self.assertEqual(stats["invariant_violations"], 0, f"trial {trial}")
            self.assertEqual(raw_not_less, 150, f"trial {trial}: reduction increased a count")
            self.assertGreater(strict_reduction_frames, 0, f"trial {trial}: reducer never removed a contact")
            body_q = state_0.body_q.numpy()
            qd = state_0.body_qd.numpy()
            self.assertTrue(np.isfinite(body_q).all() and np.isfinite(qd).all())
            self.assertLess(peak_red, max(3.0 * peak_raw, 10.0), f"trial {trial}: pile blew up")
            self.assertGreater(float(body_q[bodies, 2].min()), -0.06, f"trial {trial}: body tunneled")


class TestBodyPairReductionTouchingFootprint(unittest.TestCase):
    """Speculative candidates must not displace the touching support polygon."""

    GAP = 0.005

    def _model(self, *, device, penetration=0.0005):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = self.GAP
        body = _speculative_ring_body(builder, (0.0, 0.0, 0.0015 - penetration))
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        return model, body

    def test_touching_footprint_survives_speculative_ring(self):
        """Keep at least three spanning touching contacts next to the speculative ring."""
        model, body = self._model(device=wp.get_device())
        state = model.state()
        raw = _collide_once(model, state, False)
        red = _collide_once(model, state, True)
        n_raw = int(raw.rigid_contact_count.numpy()[0])
        n_red = int(red.rigid_contact_count.numpy()[0])
        center = state.body_q.numpy()[body][:2]

        raw_gaps, raw_mid = _world_gaps(model, state, raw)
        raw_touch = raw_mid[raw_gaps <= 0.0]
        raw_spec = raw_mid[raw_gaps > 0.0]
        self.assertGreaterEqual(len(raw_touch), 3, "scene has no touching footprint; the probe is vacuous")
        self.assertGreaterEqual(len(raw_spec), 6, "scene has no speculative ring; the probe is vacuous")
        self.assertLess(_max_angular_hole(raw_touch[:, :2], center), np.pi)
        self.assertGreater(
            float(np.linalg.norm(raw_spec[:, :2] - center, axis=1).min()),
            float(np.linalg.norm(raw_touch[:, :2] - center, axis=1).max()),
            "speculative ring must lie outside the touching footprint for this probe to bite",
        )

        self.assertLess(n_red, n_raw, "reducer removed nothing")
        red_gaps, red_mid = _world_gaps(model, state, red)
        red_touch = red_mid[red_gaps <= 0.0]
        red_spec = red_mid[red_gaps > 0.0]
        self.assertGreaterEqual(len(red_touch), 3, f"touching footprint collapsed to {len(red_touch)} contact(s)")
        self.assertLess(
            _max_angular_hole(red_touch[:, :2], center),
            np.pi,
            "kept touching contacts do not surround the body: it would tip",
        )
        self.assertGreaterEqual(len(red_spec), 1, "speculative envelope must stay represented")

    def test_touching_footprint_kept_set_is_stable_at_rest(self):
        """Repeat collides of the resting body keep the same touching representatives."""
        model, _ = self._model(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        previous = None
        for _ in range(5):
            pipeline.collide(state, contacts)
            gaps, mid = _world_gaps(model, state, contacts)
            touch = np.round(mid[gaps <= 0.0][:, :2] * 1.0e4).astype(int)
            rows = sorted(map(tuple, touch.tolist()))
            if previous is not None:
                self.assertEqual(rows, previous, "touching representatives changed on an unchanged state")
            previous = rows

    @unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free solve requires CUDA")
    def test_speculative_ring_body_rests_still_with_feather_pgs(self):
        """Rest the ring body under 4 x 16 FeatherPGS: reduced must be as still as unreduced."""
        device = wp.get_cuda_device()
        dt, substeps, steps = 0.05, 4, 60

        def settle(reduce_on):
            model, body = self._model(device=device, penetration=-0.0005)
            state_0, state_1 = model.state(), model.state()
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            control = model.control()
            pipeline = _make_pipeline(model, reduce_on, contact_matching="latest")
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=16,
                pgs_beta=0.2,
                pgs_contact_regularization=0.01,
                pgs_warmstart=True,
                friction_anchor_beta=0.2,
                contact_friction_gap_threshold=0.001,
            )
            speeds, rots = [], []
            q_ref = None
            for k in range(steps):
                for _ in range(substeps):
                    pipeline.collide(state_0, contacts)
                    solver.step(state_0, state_1, control, contacts, dt / substeps)
                    state_0, state_1 = state_1, state_0
                if k == steps // 3:
                    q_ref = state_0.body_q.numpy()[body][3:7].copy()
                if k > steps // 3:
                    qd = state_0.body_qd.numpy()[body]
                    speeds.append(float(np.linalg.norm(qd[:3])))
                    q = state_0.body_q.numpy()[body][3:7]
                    cos_half = min(1.0, abs(float(np.dot(q, q_ref))))
                    rots.append(float(np.degrees(2.0 * np.arccos(cos_half))))
            stats = pipeline.body_pair_reduction_stats() if reduce_on else None
            return float(np.mean(speeds)), float(np.max(rots)), stats

        v_off, rot_off, _ = settle(False)
        v_on, rot_on, stats = settle(True)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "ring body contacts were never reduced")
        self.assertLess(v_off, 0.002, f"unreduced reference body is not at rest ({v_off * 1000:.1f} mm/s)")
        self.assertLess(v_on, 0.002, f"reduced body rocks at {v_on * 1000:.1f} mm/s (unreduced {v_off * 1000:.2f})")
        self.assertLess(rot_on, 0.5, f"reduced body drifted {rot_on:.2f} deg (unreduced {rot_off:.3f})")


if __name__ == "__main__":
    unittest.main()
