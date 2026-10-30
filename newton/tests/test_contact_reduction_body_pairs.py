# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction ("contact compiler").

Multi-shape bodies multiply narrow-phase candidates: a foot approximated by 7
cylinders emits up to 28 points against the ground and up to 49 against another
such foot. The body-pair reduction pass approximates each body-pair/normal-bin
patch with one depth slot and six sampled footprint-support slots, so the
registered contact count reflects the physical patch more closely than the
collider decomposition. Directional sampling has no shape-independent support
bound.

The tests here assert both halves of the contract:

* the registered count drops on multi-shape pairs (and why: interior points of
  a single flat patch are the ones discarded), and
* the allocation stamp and solver gate keep reduced rows away from solvers
  that do not declare support. SolverSteel's settling conformance lives in
  ``newton/tests/steel/test_contact_reduction.py``.
"""

import unittest
import unittest.mock

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
    every propagation response mode.
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


def _make_pipeline(model, reduce_body_pairs, *, reduce_mesh_contacts=True, body_pair_verify=False, **kwargs):
    config_kwargs = {
        name: kwargs.pop(f"body_pair_{name}")
        for name in ("cell_size", "hashtable_headroom", "material_attributes")
        if f"body_pair_{name}" in kwargs
    }
    deterministic = kwargs.pop("deterministic", True)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        deterministic=deterministic,
        reduce_contacts=reduce_mesh_contacts,
        body_pair_reduction=(
            newton.CollisionPipeline.BodyPairReductionConfig(**config_kwargs) if reduce_body_pairs else None
        ),
        **kwargs,
    )
    if body_pair_verify:
        pipeline._body_pair_reducer.verify = True
    return pipeline


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


def _two_cluster_body(builder, spread=1.0):
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


def _undamped(solver):
    """Turn off the default angular damping of the free roots."""
    solver._rigid_body_angular_damping.zero_()
    return solver


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

    def test_config_enables_body_pair_stage(self):
        """Enable body-pair reduction through its own argument, leaving reduce_contacts alone."""
        model = self._foot_model()
        config = newton.CollisionPipeline.BodyPairReductionConfig()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", body_pair_reduction=config)
        self.assertTrue(pipeline.reduce_contacts)
        self.assertIs(pipeline.body_pair_reduction, config)
        self.assertIsNotNone(pipeline._body_pair_reducer)
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        self.assertIsNone(plain.body_pair_reduction)
        self.assertIsNone(plain._body_pair_reducer)

    def test_body_pair_stage_requires_mesh_stage_for_mesh_scenes(self):
        """Reject a postpass that cannot protect raw mesh output from overflow."""
        model = self._mesh_model()
        with self.assertRaisesRegex(ValueError, "reduce_contacts=True"):
            newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                reduce_contacts=False,
                body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
            )

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

        def contact_count(reduce_contacts, body_pair_reduction=None):
            pipeline = newton.CollisionPipeline(
                model,
                broad_phase="nxn",
                deterministic=True,
                rigid_contact_max=rigid_contact_max,
                reduce_contacts=reduce_contacts,
                body_pair_reduction=body_pair_reduction,
            )
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            return pipeline, int(contacts.rigid_contact_count.numpy()[0])

        raw, raw_count = contact_count(False)
        mesh_only, mesh_count = contact_count(True)
        both, both_count = contact_count(True, newton.CollisionPipeline.BodyPairReductionConfig(cell_size=1.0))
        stats = both._body_pair_reducer.stats()
        self.assertFalse(raw.narrow_phase.reduce_contacts)
        self.assertTrue(mesh_only.narrow_phase.reduce_contacts, "mesh producer reduction was not active")
        self.assertTrue(both.narrow_phase.reduce_contacts)
        self.assertGreater(raw_count, mesh_count, "producer stage did not reduce raw mesh-plane vertex contacts")
        self.assertEqual(stats["sum_contacts_in"], mesh_count, "body-pair stage did not receive producer output")
        self.assertEqual(stats["sum_contacts_kept"], both_count)
        self.assertLess(both_count, mesh_count, "body-pair stage did not further compact the compound patch")
        self.assertEqual(stats["input_overflow_frames"], 0)

    def test_body_pair_reduction_rejects_other_types(self):
        """Reject values that are not a BodyPairReductionConfig."""
        model = self._foot_model()
        with self.assertRaisesRegex(TypeError, "BodyPairReductionConfig"):
            newton.CollisionPipeline(model, broad_phase="nxn", body_pair_reduction=True)

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
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
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
                body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
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
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
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
        self.assertTrue(newton.solvers.SolverSteel.supports_body_pair_reduced_contacts)

    def test_unsupported_solver_cannot_own_a_reducing_pipeline(self):
        """Reject an owned reducing pipeline at construction, before the first step detects into it."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.020))
        builder.add_ground_plane()
        builder.color()
        model = builder.finalize(device=wp.get_device())
        with self.assertRaisesRegex(ValueError, "body-pair contact reduction"):
            newton.solvers.SolverVBD(model, collision_pipeline=_make_pipeline(model, True))


class TestBodyPairReductionMultiPatch(unittest.TestCase):
    """Same-normal patches far apart on ONE shape pair each keep full representation."""

    def test_both_clusters_fully_kept(self):
        """Keep deepest + extremes for each cluster, not one shared slot set.

        Both clusters contact the SAME ground shape with the SAME normal --
        without spatial cells they would compete for a single (pair, bin)
        entry and one cluster could lose all representation. With cells, each
        cluster (1 m apart >> 0.25 m cell) must keep at least 3 contacts.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _two_cluster_body(builder)
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
        body = _two_cluster_body(builder)
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
        stats = pipeline._body_pair_reducer.stats()
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
        desc = pipeline._body_pair_reducer.describe()
        self.assertGreater(desc["group_pair_bound"], 0, "topology anchor was not derived")
        # Capped at rigid_contact_max (a group needs a contact to create it),
        # except that a minimum viable table always wins.
        self.assertGreaterEqual(desc["hashtable_capacity_request"], 1024)
        self.assertLessEqual(desc["hashtable_capacity_request"], max(1024, desc["rigid_contact_max"]))
        # Same topology, 4x the contact buffer -> identical request.
        wide = _make_pipeline(model, True, rigid_contact_max=model.rigid_contact_max * 4)
        wide_desc = wide._body_pair_reducer.describe()
        self.assertEqual(wide_desc["group_pair_bound"], desc["group_pair_bound"])
        self.assertEqual(wide_desc["hashtable_capacity_request"], desc["hashtable_capacity_request"])
        # More bodies -> more reachable group pairs -> a larger anchor.
        bigger = _make_pipeline(self._scene(24), True)
        self.assertGreater(bigger._body_pair_reducer.describe()["group_pair_bound"], desc["group_pair_bound"])

    def test_bound_counts_the_pipeline_candidate_pairs(self):
        """Count group pairs from the broad phase's own candidates without building the model pair array."""
        model = self._scene(8)
        nxn = _make_pipeline(model, True)
        self.assertIsNone(model._shape_contact_pairs, "NXN sizing built the model's device pair array")
        self.assertEqual(nxn._body_pair_reducer.describe()["group_pair_bound"], model.shape_contact_pair_count)
        explicit = newton.CollisionPipeline(
            model,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array(model.shape_contact_pairs.numpy()[:3], dtype=wp.vec2i, device=model.device),
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
        )
        self.assertEqual(explicit._body_pair_reducer.describe()["group_pair_bound"], 3)

    def test_headroom_scales_the_derived_capacity(self):
        """Scale the derived capacity by ``hashtable_headroom``."""
        model = self._scene(24)
        small = _make_pipeline(model, True, body_pair_hashtable_headroom=1.0)
        large = _make_pipeline(model, True, body_pair_hashtable_headroom=8.0)
        self.assertGreater(
            large._body_pair_reducer.describe()["hashtable_capacity_request"],
            small._body_pair_reducer.describe()["hashtable_capacity_request"],
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

    def _torsion_pad_model(self, pad_radius):
        """One articulated body: a 5x5 sphere grid whose center sphere has ``pad_radius`` torsion."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverSteel.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0095), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        for i in range(5):
            for j in range(5):
                if i == 2 and j == 2:
                    continue
                builder.add_shape_sphere(
                    body,
                    xform=wp.transform(wp.vec3((i - 2.0) * 0.04, (j - 2.0) * 0.04, 0.0), wp.quat_identity()),
                    radius=0.01,
                )
        pad = builder.add_shape_sphere(
            body, radius=0.01, custom_attributes={"steel:min_torsional_patch_radius": pad_radius}
        )
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        return model, state, pad

    @staticmethod
    def _pad_contact_count(contacts, pad):
        _count, s0, s1, _n, _p0 = _contact_snapshot(contacts)
        return int(np.sum((s0 == pad) | (s1 == pad)))

    def test_torsion_pad_is_its_own_group(self):
        """Keep the contact of an interior collider whose Steel torsional radius differs."""
        model, state, pad = self._torsion_pad_model(0.05)
        contacts = _collide_once(model, state, True, body_pair_cell_size=10.0)
        self.assertGreater(self._pad_contact_count(contacts, pad), 0, "the torsion pad's contact was deleted")

    def test_torsion_radius_mutation_splits_after_refresh(self):
        """Split groups after a runtime torsional-radius change and a refresh."""
        model, state, pad = self._torsion_pad_model(0.0)
        pipeline = _make_pipeline(model, True, body_pair_cell_size=10.0)
        contacts = pipeline.contacts()
        radius = model.steel.min_torsional_patch_radius.numpy()
        radius[pad] = 0.05
        model.steel.min_torsional_patch_radius.assign(radius)

        pipeline.collide(state, contacts)
        self.assertEqual(self._pad_contact_count(contacts, pad), 0, "scene no longer discards the interior pad")
        pipeline.refresh_body_pair_reduction_groups()
        pipeline.collide(state, contacts)
        self.assertGreater(self._pad_contact_count(contacts, pad), 0, "refresh did not separate the torsion pad")

    def test_material_attributes_select_the_split(self):
        """Split groups on exactly the per-shape attributes the config names.

        ``None`` uses every per-shape custom attribute the model registers, so
        the Steel torsion pad stays separate; an empty tuple merges it back into
        the body's group; an unknown name is rejected.
        """
        model, state, pad = self._torsion_pad_model(0.05)
        self.assertEqual(
            newton.CollisionPipeline.BodyPairReductionConfig().material_attributes,
            None,
        )
        named = _collide_once(
            model,
            state,
            True,
            body_pair_cell_size=10.0,
            body_pair_material_attributes=["steel:min_torsional_patch_radius"],
        )
        self.assertGreater(self._pad_contact_count(named, pad), 0)
        merged = _collide_once(model, state, True, body_pair_cell_size=10.0, body_pair_material_attributes=())
        self.assertEqual(self._pad_contact_count(merged, pad), 0, "an unnamed attribute still split the group")
        with self.assertRaisesRegex(ValueError, "steel:missing"):
            _make_pipeline(model, True, body_pair_material_attributes=("steel:missing",))

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
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
        )
        contacts = plain.contacts()
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
        overflow_stats = reduced._body_pair_reducer.stats()
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
        self.assertEqual(stats["total_frames"], 1)
        self.assertEqual(stats["sum_contacts_in"], n_base)
        self.assertEqual(stats["sum_contacts_kept"], n_base)

    def test_buffer_stamp_must_match_pipeline(self):
        """Pair stamped buffers with reducing pipelines only, at the reducer's own capacity.

        The stamp is fixed at allocation, so it holds through clear() and graph
        replay. An unstamped buffer cannot be reduced, a stamped one cannot be
        refilled unreduced, and a stamped buffer of another capacity would let
        the narrow phase write past the reducer's scratch.
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        _cylinder_foot(builder, (0.0, 0.0, 0.0175))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_device())
        state = model.state()
        pipeline = _make_pipeline(model, True)
        plain = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
        contacts.clear()
        self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
        self.assertFalse(plain.contacts().rigid_contacts_body_pair_reduced)
        with self.assertRaisesRegex(ValueError, "contacts\\(\\)"):
            plain.collide(state, contacts)
        for unstamped in (
            plain.contacts(),
            Contacts(rigid_contact_max=pipeline.rigid_contact_max, soft_contact_max=0, device=wp.get_device()),
        ):
            with self.assertRaisesRegex(ValueError, "contacts\\(\\)"):
                pipeline.collide(state, unstamped)
        oversized = _make_pipeline(model, True, rigid_contact_max=pipeline.rigid_contact_max * 2).contacts()
        with self.assertRaisesRegex(ValueError, "contacts\\(\\)"):
            pipeline.collide(state, oversized)
        self.assertEqual(int(oversized.rigid_contact_count.numpy()[0]), 0, "rejected buffer was written")

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
            {"body_pair_hashtable_headroom": 0.0},
        ):
            with self.assertRaises(ValueError, msg=f"accepted invalid {kwargs}"):
                _make_pipeline(model, True, **kwargs)
        with self.assertRaises(TypeError):
            _make_pipeline(model, True, body_pair_material_attributes="steel:min_torsional_patch_radius")

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
        stats = pipeline._body_pair_reducer.stats()
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
        self.assertGreaterEqual(pipeline._body_pair_reducer.stats()["identity_frames"], 1)

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
        stats = pipeline._body_pair_reducer.stats()
        self.assertEqual(stats["total_frames"], 4)
        self.assertEqual(stats["sum_contacts_in"], observed_in)
        self.assertEqual(stats["sum_contacts_kept"], observed_kept)
        self.assertEqual(stats["max_contacts_in"], observed_max_in)
        self.assertEqual(stats["max_contacts_kept"], observed_max_kept)
        self.assertLess(stats["sum_contacts_kept"], stats["sum_contacts_in"], "cylinder foot must reduce")
        pipeline._body_pair_reducer.clear_stats()
        cleared = pipeline._body_pair_reducer.stats()
        for key in (
            "total_frames",
            "sum_contacts_in",
            "sum_contacts_kept",
            "max_contacts_in",
            "max_contacts_kept",
            "probe_failures",
            "cell_clamp_events",
            "invariant_violations",
            "outranked_discards",
            "input_overflow_frames",
            "fallback_frames",
            "identity_frames",
        ):
            self.assertEqual(cleared[key], 0, f"{key} survived clear")

    def test_telemetry_rejected_during_capture(self):
        """Reject stats reads, clears and group refreshes while a CUDA graph capture is active.

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
        with wp.ScopedCapture(device) as capture:
            # Record real reducer work first so the guard is exercised during
            # a non-empty capture, not only in an otherwise empty graph.
            pipeline.collide(state, contacts)
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                pipeline._body_pair_reducer.stats()
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                pipeline._body_pair_reducer.clear_stats()
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                pipeline.refresh_body_pair_reduction_groups()
        self.assertIsNotNone(capture.graph)


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
                body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
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
            with self.assertRaisesRegex(ValueError, "at most 5 groups"):
                _make_pipeline(model, True)

    def test_cuda_graph_capture(self):
        """Capture collide() cold and replay it on changing states against an uncaptured reference.

        All reducer buffers are allocated at construction, so the very first
        collide on a fresh buffer can be captured.  Every solver-visible contact
        field is compared, on the settled pose and again after pitching the
        foot in place.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        with wp.ScopedCapture(device) as capture:
            pipeline.collide(state, contacts)
        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)

        state_ref = model.state()
        ref = _make_pipeline(model, True)
        contacts_ref = ref.contacts()
        ref.collide(state_ref, contacts_ref)
        self.assertEqual(_full_contact_rows(contacts), _full_contact_rows(contacts_ref), "captured rows changed")

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
        self.assertEqual(_full_contact_rows(contacts), _full_contact_rows(contacts_ref), "captured rows changed")

    def test_graphs_share_buffers_and_pipeline(self):
        """Capture several graphs, and buffers, on one reducing pipeline and replay them in any order.

        The reducer keeps no state across frames, so separate substep and
        reset graphs into one buffer, or graphs into two double-buffered
        Contacts, replay to the eager result.  Skipped on CPU devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True, contact_matching="latest")
        contacts_a = pipeline.contacts()
        contacts_b = pipeline.contacts()
        with wp.ScopedCapture(device) as step_a:
            pipeline.collide(state, contacts_a)
        with wp.ScopedCapture(device) as reset_a:
            pipeline.reset_contact_matching()
            pipeline.collide(state, contacts_a)
        with wp.ScopedCapture(device) as step_b:
            pipeline.collide(state, contacts_b)
        ref = _make_pipeline(model, True)
        contacts_ref = ref.contacts()
        ref.collide(state, contacts_ref)
        expected = _full_contact_rows(contacts_ref)
        for graph in (step_a, step_b, reset_a, step_b, step_a, step_a):
            wp.capture_launch(graph.graph)
        contacts_a.clear()
        wp.capture_launch(step_a.graph)
        self.assertEqual(_full_contact_rows(contacts_a), expected)
        self.assertEqual(_full_contact_rows(contacts_b), expected)
        self.assertTrue(contacts_a.rigid_contacts_body_pair_reduced)

    def test_unsupported_solver_rejects_stamped_buffer_in_capture(self):
        """Reject a stamped buffer when an unsupported solver step is captured, as when it runs eagerly."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        model = self._foot_scene()
        state_0, state_1 = model.state(), model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverSemiImplicit(model)
        capture = wp.ScopedCapture(device)
        capture.__enter__()
        try:
            pipeline.collide(state_0, contacts)
            with self.assertRaisesRegex(ValueError, "supports_body_pair_reduced_contacts"):
                solver.step(state_0, state_1, model.control(), contacts, DT)
        finally:
            capture.__exit__(None, None, None)

    def test_reducer_launches_stay_off_the_tape(self):
        """Record no reducer bookkeeping launches on an active autodiff tape.

        Every reducer kernel has backward disabled; recording them would only
        bloat the tape. This guards the full reduce() path including the
        hashtable maintenance kernels.
        """
        model = self._foot_scene()
        state = model.state()
        pipeline = _make_pipeline(model, True)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        red = pipeline._body_pair_reducer
        tape = wp.Tape()
        with tape:
            red.reduce(model, state, contacts)
        self.assertEqual(len(tape.launches), 0, "reducer bookkeeping was recorded on the tape")

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
            body_pair_reduction=newton.CollisionPipeline.BodyPairReductionConfig(),
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


class TestBodyPairReductionVerifier(unittest.TestCase):
    """The verify mode re-derives every keep/discard decision and finds zero disagreements."""

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


if __name__ == "__main__":
    unittest.main()
