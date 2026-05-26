# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression test for off-origin convex hull contact reporting.

The MPR/GJK initial interior point ``v0`` of the Minkowski difference is
computed by :func:`geometric_center` in ``newton/_src/geometry/mpr.py``.
For ``CONVEX_MESH`` shapes whose authoring origin lies outside the hull
(common for collision hulls imported from authoring tools), using the
shape's local origin as ``v0`` can lead to a degenerate initial portal
where all triangle supports collapse onto a single vertex of the
partner.  MPR then converges on a wrong normal/penetration even when the
two shapes are clearly separated.

This test pins one such configuration: a small triangle from a
non-convex mesh against a thin convex hull whose local-frame origin sits
~17 cm outside its AABB.  In world space the two shapes are separated
by ~1.42 mm along Z.

Without the AABB-center initialization in ``geometric_center``, MPR
reports ~7.1 mm penetration along +Y; with it, MPR reports the correct
+1.42 mm separation along -Z.
"""

from __future__ import annotations

import unittest

import numpy as np

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices

# ---------------------------------------------------------------------------
# Geometry data (extracted from a real-world collision asset where this bug
# was first observed).  All quantities in metres.
# ---------------------------------------------------------------------------

# Triangle from a non-convex mesh; vertices are very small (sub-cm).
TRIANGLE_VERTICES_LOCAL = np.array(
    [
        [-0.0022889524698257446, -0.0025248676538467407, -0.07601077854633331],
        [-0.0016302913427352905, -0.002526000142097473, -0.07601077854633331],
        [-0.0022889524698257446, -0.003824576735496521, -0.07669369876384735],
    ],
    dtype=np.float32,
)

# Convex hull (8-vertex slab).  AABB center is at approximately
# (0.227, 0.038, -0.122) m in the hull's local frame, ~26 cm from the
# local origin (0, 0, 0).  The local origin is OUTSIDE the AABB on both
# X and Z, so MPR/GJK cannot use it as a sensible interior point.
CONVEX_HULL_VERTICES_LOCAL = np.array(
    [
        [0.14592893421649933, -0.17731179296970367, -0.15459845960140228],
        [0.30726319551467896, -0.17731179296970367, -0.15459845960140228],
        [0.14592893421649933, -0.17731179296970367, -0.08912084996700287],
        [0.30726319551467896, -0.17731179296970367, -0.08912084996700287],
        [0.14592893421649933, 0.25302475690841675, -0.08912088721990585],
        [0.30726319551467896, 0.25302475690841675, -0.08912088721990585],
        [0.14592893421649933, 0.25302475690841675, -0.15459850430488586],
        [0.30726319551467896, 0.25302475690841675, -0.15459850430488586],
    ],
    dtype=np.float32,
)
CONVEX_HULL_INDICES = np.array(
    [
        [7, 1, 0],
        [7, 0, 6],
        [7, 4, 5],
        [7, 6, 4],
        [2, 4, 6],
        [2, 6, 0],
        [3, 7, 5],
        [3, 1, 7],
        [3, 0, 1],
        [3, 2, 0],
        [3, 5, 4],
        [3, 4, 2],
    ],
    dtype=np.int32,
).reshape(-1)

# Body world transforms ((x, y, z), (qx, qy, qz, qw)) — both bodies are
# kinematic and oriented identity; only the translation matters for this
# regression case.
TRIANGLE_BODY_XFORM: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.18269123136997223, -0.1676918864250183, 0.18899677693843842),
    (0.0, 0.0, 0.0, 1.0),
)
CONVEX_HULL_BODY_XFORM: tuple[tuple[float, float, float], tuple[float, float, float, float]] = (
    (0.0, 0.0, 0.20000000298023224),
    (0.0, 0.0, 0.0, 1.0),
)

PER_SHAPE_GAP = 0.001  # m
EXPECTED_SIGNED_GAP = 0.0014239252  # m, derived from the AABB Z-separation
EXPECTED_NORMAL_Z = -1.0  # contact normal should point along -Z

# Tolerances are loose enough to absorb fp noise from MPR/GJK iteration but
# tight enough to distinguish the correct +1.42 mm separation from the
# bug's -7.1 mm penetration.
GAP_TOL = 5.0e-5
NORMAL_TOL = 1.0e-3


def _build_isolated_pair_model(
    device,
) -> tuple[newton.Model, newton.State, dict[str, int]]:
    """Build a two-shape model with a single triangle vs a convex hull."""
    triangle_mesh = newton.Mesh(
        TRIANGLE_VERTICES_LOCAL.copy(),
        np.array([0, 1, 2], dtype=np.int32),
        compute_inertia=False,
        is_solid=False,
    )
    hull_mesh = newton.Mesh(
        CONVEX_HULL_VERTICES_LOCAL.copy(),
        CONVEX_HULL_INDICES.copy(),
        compute_inertia=False,
        is_solid=False,
    )

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = PER_SHAPE_GAP
    builder.default_shape_cfg.is_hydroelastic = False

    triangle_body = builder.add_body(
        xform=TRIANGLE_BODY_XFORM,
        is_kinematic=True,
        label="triangle_body",
    )
    hull_body = builder.add_body(
        xform=CONVEX_HULL_BODY_XFORM,
        is_kinematic=True,
        label="hull_body",
    )

    triangle_shape = builder.add_shape_mesh(
        triangle_body,
        mesh=triangle_mesh,
        label="triangle",
    )
    hull_shape = builder.add_shape_convex_hull(
        hull_body,
        mesh=hull_mesh,
        label="off_origin_hull",
    )

    visible = int(newton.ShapeFlags.VISIBLE)
    collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
    for shape_idx in (triangle_shape, hull_shape):
        builder.shape_flags[shape_idx] = int(builder.shape_flags[shape_idx]) | visible | collide_shapes

    model = builder.finalize(device=device)
    state = model.state()
    return model, state, {"triangle": triangle_shape, "hull": hull_shape}


def _collect_pair_signed_gaps(
    contacts: newton.Contacts,
    model: newton.Model,
    state: newton.State,
    shape_a: int,
    shape_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (signed_gaps, normals) for contacts between the two given shapes."""
    reported = int(contacts.rigid_contact_count.numpy()[0])
    n = min(reported, contacts.rigid_contact_max)
    if n == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    s0 = contacts.rigid_contact_shape0.numpy()[:n]
    s1 = contacts.rigid_contact_shape1.numpy()[:n]
    p0 = contacts.rigid_contact_point0.numpy()[:n]
    p1 = contacts.rigid_contact_point1.numpy()[:n]
    normals = contacts.rigid_contact_normal.numpy()[:n]
    margin0 = contacts.rigid_contact_margin0.numpy()[:n]
    margin1 = contacts.rigid_contact_margin1.numpy()[:n]
    body_q = state.body_q.numpy()
    shape_body = model.shape_body.numpy()

    def _to_world(shape_idx: int, local: np.ndarray) -> np.ndarray:
        body_idx = int(shape_body[shape_idx])
        if body_idx < 0:
            return local
        bx = body_q[body_idx]
        pos = bx[:3]
        q = bx[3:]
        # Rotate v by quaternion (xyzw): v + 2*q_xyz x (q_xyz x v + q_w v)
        qv = q[:3]
        t = 2.0 * np.cross(qv, local)
        rotated = local + q[3] * t + np.cross(qv, t)
        return pos + rotated

    gaps: list[float] = []
    pair_normals: list[np.ndarray] = []
    for i in range(n):
        a, b = int(s0[i]), int(s1[i])
        if not ((a == shape_a and b == shape_b) or (a == shape_b and b == shape_a)):
            continue
        p0_world = _to_world(a, p0[i])
        p1_world = _to_world(b, p1[i])
        gap = float(np.dot(normals[i], p1_world - p0_world) - margin0[i] - margin1[i])
        gaps.append(gap)
        pair_normals.append(normals[i])

    return np.asarray(gaps, dtype=np.float64), np.asarray(pair_normals, dtype=np.float64)


def test_off_origin_convex_hull_reports_correct_separation(test, device):
    """MPR/GJK must report the true separation for an off-origin hull."""
    with test.subTest(device=str(device)):
        model, state, shapes = _build_isolated_pair_model(device)

        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=False,
            broad_phase="sap",
        )
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        gaps, normals = _collect_pair_signed_gaps(contacts, model, state, shapes["triangle"], shapes["hull"])

        test.assertGreater(
            gaps.size,
            0,
            "Expected at least one speculative contact within the configured gap.",
        )

        signed_gap = float(gaps.min())
        test.assertAlmostEqual(
            signed_gap,
            EXPECTED_SIGNED_GAP,
            delta=GAP_TOL,
            msg=(
                "Expected positive separation matching AABB Z-gap, "
                f"got {signed_gap:.10g}.  A negative value indicates the "
                "off-origin hull MPR bug has resurfaced."
            ),
        )

        mean_normal = normals.mean(axis=0)
        test.assertAlmostEqual(
            float(mean_normal[2]),
            EXPECTED_NORMAL_Z,
            delta=NORMAL_TOL,
            msg=(f"Expected contact normal along -Z, got {mean_normal.tolist()}."),
        )


class TestOffOriginConvexHullContacts(unittest.TestCase):
    pass


add_function_test(
    TestOffOriginConvexHullContacts,
    "test_off_origin_convex_hull_reports_correct_separation",
    test_off_origin_convex_hull_reports_correct_separation,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
