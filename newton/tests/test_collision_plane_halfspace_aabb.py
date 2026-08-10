# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Half-space AABBs for infinite planes: tilt slack vs. axis-align clamping.

The broad phase clamps an infinite plane's AABB at the surface along world
axes the plane normal aligns with, so shapes far above an axis-aligned ground
stop being narrow-phase candidates. A plane whose normal is only NEARLY
axis-aligned rises by ``lat * r`` at lateral distance ``r`` from its anchor
(``lat`` = norm of the two non-aligned normal components); clamping such a
plane at the anchor height silently drops resting contacts far from the
anchor — a sphere resting 600 m out on a floor tilted 0.05 degrees loses its
ground contact and falls through. The clamp must therefore only engage when
the worst-case rise over the supported reach stays within the AABB's contact
gap enlargement, and carry that rise as slack.
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
    surface_z = (-0.001 - normal[0] * FAR_X) / normal[2]
    center_z = surface_z + 0.5 / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(FAR_X, 0.0, center_z + shape_z_offset)))
    builder.add_shape_sphere(body, radius=0.5)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    wp.synchronize_device(device)
    return model, pipeline, contacts


def _build_resting_box(device, *, tilt_deg: float, lateral_offset: float):
    """One infinite plane through the origin plus a unit box resting on it.

    The axis-aligned box rests on (1 mm into) the plane surface at
    ``lateral_offset`` along the rising direction of the tilt.
    """
    builder = newton.ModelBuilder()
    tilt = math.radians(tilt_deg)
    normal = (math.sin(tilt), 0.0, math.cos(tilt))
    builder.add_shape_plane(plane=(*normal, 0.0), width=0.0, length=0.0)

    # Center height with 1 mm penetration: the box support along the plane
    # normal is hx*|nx| + hz*|nz|, so solve normal . center = support - 1 mm.
    half = 0.5
    support = half * (abs(normal[0]) + abs(normal[2]))
    center_z = (support - 0.001 - normal[0] * lateral_offset) / normal[2]
    body = builder.add_body(xform=wp.transform(wp.vec3(lateral_offset, 0.0, center_z)))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)

    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    return model, pipeline, contacts


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneHalfSpaceAABB(unittest.TestCase):
    DEVICE = "cuda:0"

    def _plane_aabb_top(self, pipeline) -> float:
        # Shape 0 is the plane in _build.
        return float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][2])

    def test_resting_contact_far_from_anchor_on_slightly_tilted_plane(self):
        """A 0.05-degree floor tilt must not drop resting contacts 600 m out.

        The tilted surface is ~0.52 m above the anchor height at FAR_X; an
        anchor-height AABB clamp prunes the pair in the broad phase and the
        sphere falls through the floor.
        """
        _, _, contacts = _build(self.DEVICE, tilted=True)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

    def test_tilted_plane_keeps_unbounded_aabb(self):
        """With the rise over the supported reach far beyond the contact gap,
        the tilted plane must keep the unbounded (always-candidate) extent."""
        _, pipeline, _ = _build(self.DEVICE, tilted=True)
        self.assertGreater(self._plane_aabb_top(pipeline), 1.0e9)

    def test_axis_aligned_plane_keeps_halfspace_pruning(self):
        """The exact axis-aligned ground keeps the clamped AABB (the point of
        the half-space optimization) without losing far-away resting contacts."""
        _, pipeline, contacts = _build(self.DEVICE, tilted=False)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(self._plane_aabb_top(pipeline), 1.0)

    def test_axis_aligned_plane_prunes_hovering_shape(self):
        """A sphere 50 m above the flat ground produces no contact (and with
        the clamped plane AABB, no broad-phase candidate pair either)."""
        _, pipeline, contacts = _build(self.DEVICE, tilted=False, shape_z_offset=50.0)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertLess(self._plane_aabb_top(pipeline), 1.0)


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPlaneTiltTableRestingBox(unittest.TestCase):
    DEVICE = "cuda:0"

    def test_resting_box_retained_across_tilt_table(self):
        """Verify a far-offset resting box keeps ground contact for every tilt.

        For every tilt in the table and both lateral offsets, the resting box
        must produce a contact (no fall-through). An exactly aligned plane must
        additionally keep the clamped half-space AABB, while every nonzero tilt
        in the table rises over the supported reach by far more than the
        contact gap and must keep the unbounded always-candidate extent.
        """
        for tilt_deg in TILT_TABLE_DEG:
            for offset in LATERAL_OFFSETS:
                with self.subTest(tilt_deg=tilt_deg, offset_m=offset):
                    _, pipeline, contacts = _build_resting_box(self.DEVICE, tilt_deg=tilt_deg, lateral_offset=offset)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0, "resting box lost its ground contact (fall-through)")
                    plane_aabb_top = float(pipeline.narrow_phase.shape_aabb_upper.numpy()[0][2])
                    if tilt_deg == 0.0:
                        self.assertLess(plane_aabb_top, 1.0, "aligned plane must keep half-space pruning")
                    else:
                        self.assertGreater(plane_aabb_top, 1.0e9, "tilted plane must stay conservative")


if __name__ == "__main__":
    unittest.main(verbosity=2)
