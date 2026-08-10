# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Half-space AABBs for infinite planes: surface clamping vs. residual tilt.

The broad phase clamps an infinite plane's AABB at the surface along the world
axes the plane normal aligns with, so shapes far above an axis-aligned ground
stop being narrow-phase candidates. A normal that is only NEARLY axis-aligned
rises by ``(|n_j| + |n_k|) * d / |n_i|`` at lateral offset ``d`` from the plane
anchor; clamping such a plane at the anchor height silently drops resting
contacts far from the anchor -- a sphere resting 600 m out on a floor tilted
0.05 degrees loses its ground contact and falls through. The clamp therefore
carries the rise over the reach the AABB itself admits, which keeps it
conservative for every shape the AABB does not already prune laterally.
"""

from __future__ import annotations

import math
import unittest

import warp as wp

import newton

TILT = math.radians(0.05)  # cos = 0.9999996: inside a naive 0.999999 epsilon
FAR_X = -600.0  # the side the tilted surface RISES toward

# Tilt table spanning both sides of the old 0.999999 axis-align epsilon
# (cos(0.081 deg) = 0.9999990: right at the boundary) plus a genuinely
# tilted 1-degree ramp, crossed with lateral offsets far from the anchor.
TILT_TABLE_DEG = (0.0, 0.05, 0.081, 0.1, 1.0)
LATERAL_OFFSETS = (-100.0, -600.0)  # the side the tilted surface RISES toward

# Plane offsets along the normal, exercising a translated plane alongside the
# one through the origin.
PLANE_OFFSETS = (0.0, -25.0)


def _surface_z(normal, plane_d: float, x: float) -> float:
    """Return the plane surface height above ``x`` for ``n . p + plane_d = 0``."""
    return (-plane_d - normal[0] * x) / normal[2]


def _build(device, *, tilted: bool, shape_z_offset: float = 0.0):
    """One infinite plane through the origin plus a unit-diameter sphere.

    The sphere rests on (1 mm into) the plane surface at ``FAR_X``, or hovers
    ``shape_z_offset`` above that resting point.
    """
    builder = newton.ModelBuilder()
    if tilted:
        normal = (math.sin(TILT), 0.0, math.cos(TILT))
    else:
        normal = (0.0, 0.0, 1.0)
    builder.add_shape_plane(plane=(*normal, 0.0), width=0.0, length=0.0)

    # Resting-point height of a radius-0.5 sphere centered above FAR_X, with
    # 1 mm of penetration so the contact is unambiguous.
    surface_z = _surface_z(normal, 0.001, FAR_X)
    center_z = surface_z + 0.5 / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(FAR_X, 0.0, center_z + shape_z_offset)))
    builder.add_shape_sphere(body, radius=0.5)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return model, pipeline, contacts


def _build_resting_box(device, *, tilt_deg: float, lateral_offset: float, plane_d: float = 0.0):
    """One infinite plane plus a unit box resting on it.

    The axis-aligned box rests on (1 mm into) the plane surface at
    ``lateral_offset`` along the rising direction of the tilt. ``plane_d`` is
    the plane equation's offset term, so a nonzero value translates the plane
    away from the world origin.
    """
    builder = newton.ModelBuilder()
    tilt = math.radians(tilt_deg)
    normal = (math.sin(tilt), 0.0, math.cos(tilt))
    builder.add_shape_plane(plane=(*normal, plane_d), width=0.0, length=0.0)

    # Center height with 1 mm penetration: the box support along the plane
    # normal is hx*|nx| + hz*|nz|, so solve normal . center = -plane_d + support - 1 mm.
    half = 0.5
    support = half * (abs(normal[0]) + abs(normal[2]))
    center_z = (-plane_d + support - 0.001 - normal[0] * lateral_offset) / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(lateral_offset, 0.0, center_z)))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return model, pipeline, contacts


def _plane_aabb_top(pipeline) -> float:
    """Return the upper Z bound of shape 0, the infinite plane in every build."""
    return float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][2])


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneHalfSpaceAABB(unittest.TestCase):
    DEVICE = "cuda:0"

    def test_resting_contact_far_from_anchor_on_slightly_tilted_plane(self):
        """Retain a resting contact 600 m from the anchor of a 0.05-degree floor.

        The tilted surface is ~0.52 m above the anchor height at ``FAR_X``; an
        anchor-height AABB clamp prunes the pair in the broad phase and the
        sphere falls through the floor.
        """
        _, _, contacts = _build(self.DEVICE, tilted=True)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_tilted_plane_bound_clears_the_true_surface(self):
        """Keep the tilted plane's clamped bound above the surface it can reach.

        The bound carries the rise over the reach the AABB admits, so it must
        sit above the true surface height at the far test offset rather than at
        the anchor height.
        """
        normal = (math.sin(TILT), 0.0, math.cos(TILT))
        _, pipeline, _ = _build(self.DEVICE, tilted=True)
        self.assertGreater(_plane_aabb_top(pipeline), _surface_z(normal, 0.0, FAR_X))

    def test_axis_aligned_plane_keeps_halfspace_pruning(self):
        """Clamp the exactly axis-aligned ground at its surface.

        This is the point of the half-space optimization, and it must hold
        without losing resting contacts far from the anchor.
        """
        _, pipeline, contacts = _build(self.DEVICE, tilted=False)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(_plane_aabb_top(pipeline), 1.0)

    def test_axis_aligned_plane_prunes_hovering_shape(self):
        """Produce no contact for a sphere 50 m above the flat ground."""
        _, pipeline, contacts = _build(self.DEVICE, tilted=False, shape_z_offset=50.0)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(_plane_aabb_top(pipeline), 1.0)


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneTiltTableRestingBox(unittest.TestCase):
    DEVICE = "cuda:0"

    def test_resting_box_retained_across_tilt_table(self):
        """Keep a far-offset resting box in contact for every tilt and offset.

        Each case also checks the plane's clamped bound: an exactly aligned
        plane clamps at its surface, and every tilted one clamps above the true
        surface height at the box's offset so the pair survives the broad
        phase. Translated planes are covered alongside the one through the
        origin.
        """
        for plane_d in PLANE_OFFSETS:
            for tilt_deg in TILT_TABLE_DEG:
                for offset in LATERAL_OFFSETS:
                    with self.subTest(plane_d=plane_d, tilt_deg=tilt_deg, offset_m=offset):
                        _, pipeline, contacts = _build_resting_box(
                            self.DEVICE, tilt_deg=tilt_deg, lateral_offset=offset, plane_d=plane_d
                        )
                        count = int(contacts.rigid_contact_count.numpy()[0])
                        self.assertGreater(count, 0, "resting box lost its ground contact (fall-through)")

                        tilt = math.radians(tilt_deg)
                        normal = (math.sin(tilt), 0.0, math.cos(tilt))
                        anchor_z = _surface_z(normal, plane_d, 0.0)
                        top = _plane_aabb_top(pipeline)
                        if tilt_deg == 0.0:
                            self.assertLess(top, anchor_z + 1.0, "aligned plane must keep half-space pruning")
                        else:
                            self.assertGreater(
                                top,
                                _surface_z(normal, plane_d, offset),
                                "tilted plane bound must clear the surface at the box offset",
                            )


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneNearlyAlignedNormal(unittest.TestCase):
    DEVICE = "cuda:0"

    def test_non_z_axis_plane_still_clamps(self):
        """Clamp a plane whose normal carries quat_rotate rounding noise.

        ``quat_between_vectors`` reproduces a +Z normal exactly but leaves
        about one ulp of lateral residue on the other world axes. The rise that
        residue implies is centimeters over the supported reach, so the clamp
        must still engage rather than fall back to the unbounded extent.
        """
        builder = newton.ModelBuilder()
        builder.add_shape_plane(plane=(1.0, 0.0, 0.0, 0.0), width=0.0, length=0.0)
        model = builder.finalize(device=self.DEVICE)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        self.assertLess(float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][0]), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
