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
    cases = list(
        itertools.product(
            (False, True),
            (0.0, 0.63),
            ((0.0, 0.0, 0.0), (100.0, -80.0, 3.0)),
            (-0.001, 0.0, 0.00005, 0.000095, 0.001),
        )
    )
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.001, gap=0.005)
    shape_pairs = []
    expected_normals = []
    for swap, angle, translation, gap in cases:
        builder.begin_world()
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)
        direction = np.array(wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0)))
        poses = [np.array(translation), np.array(translation) + direction * (0.02 + gap)]
        if swap:
            poses.reverse()
        pair = []
        for pose in poses:
            body = builder.add_body(xform=wp.transform(wp.vec3(*pose), rotation))
            pair.append(builder.add_shape_convex_hull(body, mesh=mesh, cfg=cfg))
        builder.end_world()
        shape_pairs.append(pair)
        expected_normals.append(direction * (-1.0 if swap else 1.0))
    model = builder.finalize(device=device)
    shape_world = model.shape_world.numpy()

    for matching in ("disabled", "sticky"):
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=False,
            deterministic=True,
            contact_matching=matching,
            rigid_contact_max=32 * len(cases),
        )
        state = model.state()
        contacts = pipeline.contacts()
        for collision_pass in range(3):
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            normals = contacts.rigid_contact_normal.numpy()[:count]
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            with test.subTest(matching=matching, collision_pass=collision_pass):
                np.testing.assert_array_equal(shape_world[shape0], shape_world[shape1])
                test.assertTrue(np.all(shape0 != shape1))
            for (swap, angle, translation, gap), (first, second), expected in zip(
                cases, shape_pairs, expected_normals, strict=True
            ):
                with test.subTest(
                    matching=matching,
                    collision_pass=collision_pass,
                    swap=swap,
                    angle=angle,
                    translation=translation,
                    gap=gap,
                ):
                    mask = ((shape0 == first) & (shape1 == second)) | ((shape0 == second) & (shape1 == first))
                    test.assertGreater(np.count_nonzero(mask), 0)
                    n = normals[mask]
                    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-5)
                    expected_rows = np.where((shape0[mask] == first)[:, None], expected, -expected)
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
