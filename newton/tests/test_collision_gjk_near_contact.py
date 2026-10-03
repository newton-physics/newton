# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test convex contact normals near the GJK convergence tolerance."""

import itertools
import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_transformed_convex_contacts(test, device):
    """Return unit, correctly oriented contact normals across the near-contact boundary."""
    mesh = newton.Mesh.create_box(0.01, 0.02, 0.03, duplicate_vertices=False, compute_inertia=False)
    for matching, swap, angle, translation, gap in itertools.product(
        ("disabled", "sticky"),
        (False, True),
        (0.0, 0.63),
        ((0.0, 0.0, 0.0), (100.0, -80.0, 3.0)),
        (-0.001, 0.0, 0.00005, 0.000095, 0.001),
    ):
        with test.subTest(matching=matching, swap=swap, angle=angle, translation=translation, gap=gap):
            builder = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig(margin=0.001, gap=0.005)
            rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
            direction = np.array(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
            poses = [np.array(translation), np.array(translation) + direction * (0.02 + gap)]
            if swap:
                poses.reverse()
            for pose in poses:
                body = builder.add_body(xform=wp.transform(wp.vec3(*pose), rotation))
                builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg)
            model = builder.finalize(device=device)
            pipeline = newton.CollisionPipeline(
                model, reduce_contacts=False, deterministic=True, contact_matching=matching, rigid_contact_max=32
            )
            state = model.state()
            contacts = pipeline.contacts()
            for _ in range(3):
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertGreater(count, 0)
                n = contacts.rigid_contact_normal.numpy()[:count]
                np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
                ids = contacts.rigid_contact_shape0.numpy()[:count]
                expected = direction * (-1.0 if swap else 1.0)
                expected_rows = np.where((ids == 0)[:, None], expected, -expected)
                test.assertTrue(np.all(np.sum(n * expected_rows, axis=1) > 0.98))


class TestCollisionGJKNearContact(unittest.TestCase):
    """Check convex contact normals through the collision pipeline without assets or solvers."""


add_function_test(
    TestCollisionGJKNearContact,
    "test_transformed_convex_contacts",
    test_transformed_convex_contacts,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
