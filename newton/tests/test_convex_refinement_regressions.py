# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check signed near contacts and convex collisions with non-hull source triangles."""

import unittest
from itertools import product
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.tests.test_convex_contact_witness import (
    _create_witness_kernel,
    _failed_distance_query,
    _unconverged_distance_query,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _contacts(test, device, half, vertices, indices, pose, scale, *, swap=False, split=False, margin=0.0):
    """Return physical witnesses and box-to-mesh normals through the public pipeline."""
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    cfg = builder.ShapeConfig(margin=margin, gap=1.0e-3, density=0.0)
    mesh = newton.Mesh(vertices, np.asarray(indices, dtype=np.int32).ravel(), compute_inertia=False)
    box_index = -1
    for box in (False, True) if swap else (True, False):
        if box:
            box_index = builder.add_shape_box(-1, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
        else:
            builder.add_shape_convex_hull(body, mesh=mesh, scale=scale, cfg=cfg)
    model = builder.finalize(device)
    threshold = 0 if split else 10**9
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
    contacts = pipeline.contacts()
    pipeline.collide(model.state(), contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    points0 = contacts.rigid_contact_point0.numpy()[:count]
    points1 = contacts.rigid_contact_point1.numpy()[:count]
    box_points, mesh_points, directions = [], [], []
    for i in range(count):
        box_first = shape0[i] == box_index
        box_points.append(points0[i] if box_first else points1[i])
        local = points1[i] if box_first else points0[i]
        mesh_points.append(np.asarray(wp.transform_point(pose, wp.vec3(local))))
        directions.append(normals[i] if box_first else -normals[i])
    return np.asarray(box_points), np.asarray(mesh_points), np.asarray(directions)


def test_refinement_signed_separation(test, device):
    """Preserve physical gaps and normal direction inside the inflated contact envelope."""
    from scipy.spatial import ConvexHull

    half = np.array([0.000514290446477979, 0.0020610823604064504, 0.0014347418727108845])
    scale = np.array([0.0003197908223699861, 0.002374501846244661, 0.002116659093001135])
    position = np.array([3.049481617882759e-06, 0.0035338771424738916, 0.0023541224223130363])
    rotation = wp.quat(0.040956851094961166, 0.37573423981666565, -0.8881269693374634, 0.2614895701408386)
    vertices = np.random.default_rng(31415).normal(size=(18, 3)).astype(np.float32)
    hull = ConvexHull(vertices)
    corners = np.array(list(product(*[(-value, value) for value in half])))
    partner = np.array([wp.quat_rotate(rotation, wp.vec3(vertex * scale)) for vertex in vertices]) + position
    planes = ConvexHull((corners[:, None] - partner[None, :]).reshape(-1, 3)).equations
    plane = planes[np.argmax(planes[:, 3])]
    expected_normal, gap = plane[:3], plane[3]
    for split, swap, shift in product(
        (False, True) if device.is_cuda else (False,), (False, True), (0.0, -gap, -2.0 * gap)
    ):
        with test.subTest(split=split, swap=swap, shift=shift):
            pose = wp.transform(wp.vec3(position + shift * expected_normal), rotation)
            pa, pb, normals = _contacts(
                test,
                device,
                half,
                vertices,
                hull.simplices,
                pose,
                scale,
                swap=swap,
                split=split,
                margin=1.0e-5,
            )
            test.assertTrue(np.all(np.isfinite(normals)))
            # Check actual shape surfaces independently of the reported normal/depth.
            # Ordinary portal controls use the default 10 micrometre MPR tolerance.
            surface_tolerance = 2.0e-6 if shift == 0.0 else 1.0e-5
            np.testing.assert_allclose(np.max(np.abs(pa) - half, axis=1), 0.0, atol=surface_tolerance)
            local = np.array([wp.transform_point(wp.transform_inverse(pose), wp.vec3(point)) for point in pb])
            scaled_normals = hull.equations[:, :3] / scale
            lengths = np.linalg.norm(scaled_normals, axis=1)
            surface = (local @ scaled_normals.T + hull.equations[:, 3]) / lengths
            np.testing.assert_allclose(np.max(surface, axis=1), 0.0, atol=surface_tolerance)
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1.0e-5)
            test.assertGreater(float(np.min(normals @ expected_normal)), 0.99)
            depths = np.sum((pa - pb) * normals, axis=1)
            np.testing.assert_allclose(depths, -(gap + shift), atol=1.0e-5)
            if shift == 0.0:
                test.assertTrue(np.all(depths < 0.0))
            elif shift == -2.0 * gap:
                test.assertTrue(np.all(depths > 0.0))


def test_refinement_source_topology(test, device):
    """Keep correct contacts when the same support vertices have concave or incomplete triangles."""
    from scipy.spatial import ConvexHull

    vertices = np.array(
        [
            [-0.4000000059604645, -0.30000001192092896, -0.20000000298023224],
            [0.6000000238418579, -0.20000000298023224, -0.25],
            [0.15000000596046448, 0.699999988079071, -0.30000001192092896],
            [0.05000000074505806, 0.10000000149011612, 0.699999988079071],
            [0.10916666686534882, 0.07041666656732559, -0.1431249976158142],
            [0.09083333611488342, -0.03958333283662796, 0.04020833224058151],
            [0.008333333767950535, 0.12541666626930237, 0.031041666865348816],
            [0.19166666269302368, 0.14374999701976776, 0.02187499962747097],
        ],
        dtype=np.float32,
    )
    hull = ConvexHull(vertices)
    half = np.array([0.45, 0.35, 0.3])
    pose = wp.transform(
        wp.vec3([-0.5678544640541077, -0.4294249415397644, -0.5421493649482727]),
        wp.quat([-0.7031238079071045, 0.21531128883361816, 0.2050347626209259, -0.6459246873855591]),
    )
    partner = np.array([wp.transform_point(pose, wp.vec3(vertex)) for vertex in vertices])
    corners = np.array(list(product(*[(-value, value) for value in half])))
    planes = ConvexHull((corners[:, None] - partner[None, :]).reshape(-1, 3)).equations
    plane = planes[np.argmax(planes[:, 3])]
    topologies = (
        hull.simplices,
        np.array(
            [
                [2, 1, 4],
                [1, 0, 4],
                [0, 2, 4],
                [3, 1, 5],
                [1, 0, 5],
                [0, 3, 5],
                [3, 2, 6],
                [2, 0, 6],
                [0, 3, 6],
                [3, 2, 7],
                [2, 1, 7],
                [1, 3, 7],
            ]
        ),
        hull.simplices[:1],
    )
    for split, swap, index in product(
        (False, True) if device.is_cuda else (False,), (False, True), range(len(topologies))
    ):
        with test.subTest(split=split, swap=swap, topology=index):
            pa, pb, normals = _contacts(
                test,
                device,
                half,
                vertices,
                topologies[index],
                pose,
                np.ones(3),
                swap=swap,
                split=split,
            )
            test.assertGreater(float(np.min(normals @ plane[:3])), 0.99)
            test.assertAlmostEqual(float(np.max(np.sum((pa - pb) * normals, axis=1))), -plane[3], delta=1.0e-5)


def test_refinement_failed_query_separation(test, device, *, unconverged=False):
    """Leave a separated pair unresolved when every refinement distance query fails."""
    from scipy.spatial import ConvexHull

    half = np.array([0.000514290446477979, 0.0020610823604064504, 0.0014347418727108845])
    scale = np.array([0.0003197908223699861, 0.002374501846244661, 0.002116659093001135])
    vertices = [
        np.array(list(product(*[(-value, value) for value in half])), dtype=np.float32),
        (np.random.default_rng(31415).normal(size=(18, 3)).astype(np.float32) * scale).astype(np.float32),
    ]
    meshes = [
        wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=device),
            indices=wp.array(ConvexHull(points).simplices.ravel(), dtype=int, device=device),
        )
        for points in vertices
    ]
    kernel = _create_witness_kernel(_unconverged_distance_query if unconverged else _failed_distance_query)
    hit = wp.zeros(1, dtype=int, device=device)
    points = wp.zeros(3, dtype=wp.vec3, device=device)
    depth = wp.zeros(1, dtype=float, device=device)
    for budget in (0, 1, 2, 30):
        with test.subTest(iterations=budget):
            wp.launch(
                kernel,
                dim=1,
                inputs=[
                    meshes[0].id,
                    meshes[1].id,
                    wp.quat(0.040956851094961166, 0.37573423981666565, -0.8881269693374634, 0.2614895701408386),
                    wp.vec3(3.049481617882759e-06, 0.0035338771424738916, 0.0023541224223130363),
                    budget,
                    0.0002,
                ],
                outputs=[hit, points, depth],
                device=device,
            )
            test.assertEqual(int(hit.numpy()[0]), 0)


def test_refinement_unconverged_query_separation(test, device):
    """Reject uncertified distance witnesses for physically separated inflated shapes."""
    test_refinement_failed_query_separation(test, device, unconverged=True)


def test_refinement_resting_box_stack(test, device):
    """Report face contacts for centered, nearly aligned resting cubes.

    Centered support ties make MPR's region tests evaluate to float32 noise and
    accept a sliver portal that misses the origin ray, which used to report a
    single tilted contact 0.155 m deep.
    """
    lower = wp.transform(
        wp.vec3(-1.2568341e-03, -4.8650557e-04, 1.2503407e00),
        wp.quat(1.9415199e-04, -4.9863686e-04, 2.3975765e-05, 9.9999988e-01),
    )
    upper = wp.transform(
        wp.vec3(-1.7543929e-03, -6.8072625e-04, 1.7503396e00),
        wp.quat(1.9580426e-04, -4.9785688e-04, 2.5186999e-05, 9.9999988e-01),
    )
    for split, swap in product((False, True) if device.is_cuda else (False,), (False, True)):
        with test.subTest(split=split, swap=swap):
            builder = newton.ModelBuilder()
            for pose in (upper, lower) if swap else (lower, upper):
                body = builder.add_body(xform=pose)
                builder.add_shape_box(body, hx=0.25, hy=0.25, hz=0.25)
            model = builder.finalize(device)
            threshold = 0 if split else 10**9
            with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", threshold):
                pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreaterEqual(count, 4)
            body_q = state.body_q.numpy()
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            points0 = contacts.rigid_contact_point0.numpy()[:count]
            points1 = contacts.rigid_contact_point1.numpy()[:count]
            normals = contacts.rigid_contact_normal.numpy()[:count]
            for i in range(count):
                pa = np.asarray(wp.transform_point(wp.transform(*body_q[shape0[i]]), wp.vec3(points0[i])))
                pb = np.asarray(wp.transform_point(wp.transform(*body_q[shape1[i]]), wp.vec3(points1[i])))
                # Normals point from shape 0 to shape 1, so the lower cube decides the sign.
                up = 1.0 if body_q[shape0[i]][2] < body_q[shape1[i]][2] else -1.0
                test.assertGreater(float(normals[i][2] * up), 0.99)
                test.assertAlmostEqual(float(np.dot(pb - pa, normals[i])), 0.0, delta=1.0e-4)


class TestConvexRefinementRegressions(unittest.TestCase):
    pass


for function in (
    test_refinement_signed_separation,
    test_refinement_source_topology,
    test_refinement_failed_query_separation,
    test_refinement_unconverged_query_separation,
    test_refinement_resting_box_stack,
):
    add_function_test(TestConvexRefinementRegressions, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
