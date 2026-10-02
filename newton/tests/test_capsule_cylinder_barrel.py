# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check specialized capsule-cylinder point, line, and rim contacts."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.geometry.collision_primitive import (
    _capsule_cylinder_rim_normal,
    _collide_capsule_cylinder_barrel,
    collide_capsule_cylinder,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def query_barrel(
    positions: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    handled: wp.array[int],
    distances: wp.array[float],
    centers: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    ok, distance, center, normal = _collide_capsule_cylinder_barrel(
        positions[i], axes[i], 0.25, 0.5, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 3.75, 1.5
    )
    handled[i] = int(ok)
    distances[i] = distance
    centers[i] = center
    normals[i] = normal


def test_barrel_guards(test, device):
    """Limit the barrel fast path to exterior witnesses on the finite barrel."""
    positions = [
        (3.95, 0.0, 0.3),  # interior barrel witness
        (4.45, 0.0, 0.3),  # core endpoint is the closest witness
        (4.1, 0.0, 0.3),  # separated but analytically handled
        (3.95, 0.0, 0.3),  # parallel: use the line manifold
        (3.95, 0.0, 1.5),  # exact rim: use the finite-cylinder solver
        (3.95, 0.0, 1.6),  # outside barrel: use the finite-cylinder solver
        (3.7, 0.0, 0.3),  # core intersection: use the finite-cylinder solver
    ]
    axes = [
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    n = len(positions)
    with wp.ScopedDevice(device):
        handled = wp.zeros(n, dtype=int)
        distances = wp.zeros(n, dtype=float)
        centers = wp.zeros(n, dtype=wp.vec3)
        normals = wp.zeros(n, dtype=wp.vec3)
        wp.launch(
            query_barrel,
            n,
            [wp.array(positions, dtype=wp.vec3), wp.array(axes, dtype=wp.vec3), handled, distances, centers, normals],
        )
    np.testing.assert_array_equal(handled.numpy(), [1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(distances.numpy()[:3], [-0.05, -0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(normals.numpy()[:3], [[-1.0, 0.0, 0.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(centers.numpy()[:3], [[3.725, 0.0, 0.3], [3.725, 0.0, 0.3], [3.8, 0.0, 0.3]], atol=1e-6)
    test.assertTrue(np.all(distances.numpy()[3:] >= MAXVAL * 0.99))


@wp.kernel
def query_capsule_cylinder(
    positions: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    cylinder_axes: wp.array[wp.vec3],
    dimensions: wp.array[wp.vec4],
    distances: wp.array[wp.vec2],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
):
    i = wp.tid()
    dims = dimensions[i]
    d0, p0, d1, _p1, normal = collide_capsule_cylinder(
        positions[i], axes[i], dims[0], dims[1], wp.vec3(0.0), cylinder_axes[i], dims[2], dims[3]
    )
    distances[i] = wp.vec2(d0, d1)
    points[i] = p0
    normals[i] = normal


def test_support_witnesses(test, device):
    """Certify separated and penetrating witnesses across poses and scales."""
    rng = np.random.default_rng(42)
    count = 288
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    positions = rng.uniform(-2.0, 2.0, size=(count, 3))
    dimensions = rng.uniform(0.1, 1.5, size=(count, 4))
    # Include full core intersections, zero-length capsules, and all scales.
    positions[:24] *= 0.1
    positions[:6] = 0.0
    axes[:6] = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0], [1e-5, 0, 1], [1, 0, 1e-5]]
    axes[:6] /= np.linalg.norm(axes[:6], axis=1)[:, None]
    dimensions[:6] = [0.25, 3.0, 1.0, 1.0]
    dimensions[24:32, 1] = 0.0
    # Almost horizontal overhangs and almost circular projected rims also need
    # valid surface witnesses, not just an accurate scalar gap.
    index = 32
    for c in (1e-5, 0.001, 0.01, 1.0 - 1e-5):
        sine = np.sqrt(1.0 - c * c)
        for a in (-0.1, -1e-4, 1e-4, 0.1):
            for b in (0.3, 1.0):
                axes[index] = [sine, 0.0, c]
                positions[index] = (a + sine) * np.array([-c, 0.0, sine]) + [0.0, b, 0.0]
                dimensions[index] = [0.1, 2.0, 1.0, 1.0]
                index += 1
    scales = np.tile([0.001, 1.0, 1000.0], count // 3)
    rotations, _ = np.linalg.qr(rng.normal(size=(count, 3, 3)))
    with wp.ScopedDevice(device):
        distances = wp.zeros(count, dtype=wp.vec2)
        points = wp.zeros(count, dtype=wp.vec3)
        normals = wp.zeros(count, dtype=wp.vec3)
        wp.launch(
            query_capsule_cylinder,
            count,
            [
                wp.array(np.einsum("nij,nj->ni", rotations, positions) * scales[:, None], dtype=wp.vec3),
                wp.array(np.einsum("nij,nj->ni", rotations, axes), dtype=wp.vec3),
                wp.array(rotations[:, :, 2], dtype=wp.vec3),
                wp.array(dimensions * scales[:, None], dtype=wp.vec4),
                distances,
                points,
                normals,
            ],
        )
    gaps = distances.numpy()[:, 0] / scales
    manifolds = distances.numpy()[:, 1] < MAXVAL
    centers = np.einsum("nji,nj->ni", rotations, points.numpy()) / scales[:, None]
    outward = np.einsum("nji,nj->ni", rotations, -normals.numpy())
    # An independent collection of separating planes detects missed features.
    sampled = rng.normal(size=(16384, 3))
    sampled /= np.linalg.norm(sampled, axis=1)[:, None]
    sampled = np.concatenate((sampled, np.eye(3), -np.eye(3)))
    for i in range(count):
        with test.subTest(case=i, scale=scales[i]):
            radius, length, cylinder_radius, height = dimensions[i]
            # The deliberately approximate near-parallel manifold bounds core
            # drift by .01*R, so its tangent-plane witness error is O(.0001*R).
            witness_tolerance = 5e-5 * cylinder_radius + 2e-6 if manifolds[i] else 2e-5
            normal = outward[i]
            np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=2e-6)
            plane_gaps = (
                sampled @ positions[i]
                - length * np.abs(sampled @ axes[i])
                - height * np.abs(sampled[:, 2])
                - cylinder_radius * np.linalg.norm(sampled[:, :2], axis=1)
                - radius
            )
            test.assertGreaterEqual(gaps[i] + 2e-5, np.max(plane_gaps))
            capsule_surface = centers[i] + 0.5 * gaps[i] * normal
            cylinder_surface = centers[i] - 0.5 * gaps[i] * normal
            along = np.clip((capsule_surface - positions[i]) @ axes[i], -length, length)
            core = positions[i] + along * axes[i]
            test.assertAlmostEqual(np.linalg.norm(capsule_surface - core), radius, delta=witness_tolerance)
            radial = np.linalg.norm(cylinder_surface[:2])
            axial = abs(cylinder_surface[2])
            test.assertLessEqual(radial, cylinder_radius + witness_tolerance)
            test.assertLessEqual(axial, height + witness_tolerance)
            test.assertLessEqual(min(abs(radial - cylinder_radius), abs(axial - height)), witness_tolerance)


@wp.kernel
def query_rim_normals(parameters: wp.array[wp.vec3], represented: wp.array[wp.vec3], normals: wp.array[wp.vec3]):
    i = wp.tid()
    c = parameters[i][0]
    sine = wp.sqrt(1.0 - c * c)
    axis = wp.vec3(sine, 0.0, c)
    e0 = wp.vec3(-c, 0.0, sine)
    relative = (parameters[i][1] + sine) * e0 + wp.vec3(0.0, parameters[i][2], 0.0)
    represented[i] = wp.vec3(c, wp.dot(relative, e0) - sine, parameters[i][2])
    normals[i] = _capsule_cylinder_rim_normal(relative, axis, 1.0, 1.0, 1.0)


def test_rim_conditioning(test, device):
    """Check thin ellipses, nearly circular rims, and repeated-root limits."""
    parameters = []
    for c in (1e-5, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0 - 1e-5):
        for a in (-2.0, -0.1, -1e-4, 0.0, 1e-4, 0.1, 2.0):
            for b in (0.0, 1e-4, 0.01, 0.3, 0.9, 1.0, 1.1, 2.0, c * abs(a)):
                parameters.append((c, a, b))
    parameters = np.asarray(parameters, dtype=np.float32)
    with wp.ScopedDevice(device):
        normals = wp.zeros(len(parameters), dtype=wp.vec3)
        represented = wp.zeros(len(parameters), dtype=wp.vec3)
        wp.launch(query_rim_normals, len(parameters), [wp.array(parameters, dtype=wp.vec3), represented, normals])
    angles = np.linspace(0.0, 0.5 * np.pi, 16385)
    cs, sn = np.cos(angles), np.sin(angles)
    for (c, a, b), normal in zip(represented.numpy().astype(np.float64), normals.numpy(), strict=True):
        with test.subTest(c=c, a=a, b=b):
            x = normal @ np.array([-c, 0.0, np.sqrt(1.0 - c * c)])
            y = normal[1]
            radius = np.sqrt(c * c * x * x + y * y)
            gap = a * x + b * y - radius
            sampled = a * cs + b * sn - np.sqrt(c * c * cs * cs + sn * sn)
            test.assertGreaterEqual(gap + 2e-5, sampled.max())
            if x > 1e-5 and y > 1e-5:
                derivative = -a * y + b * x - (1.0 - c * c) * x * y / radius
                test.assertLessEqual(abs(derivative), 2e-5 * max(1.0, abs(a), b))


def test_pipeline_dispatch(test, device):
    """Handle sharp cylinders without GJK/MPR and preserve rounded-cylinder dispatch."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in (
                "barrel",
                "parallel",
                "parallel_clipped",
                "near_parallel",
                "cap_face",
                "cap_clipped",
                "cap_clipped_negative",
                "cap_bridge",
                "cap_outside",
                "cap_tilted",
                "cap_penetrating",
                "cap_deep",
                "cap_bottom",
                "cap_axial",
                "cap_axial_bottom",
                "rim",
                "core",
                "rounded",
            ):
                with test.subTest(reversed_order=reversed_order, kind=kind):
                    builder = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.00025)
                    cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    cylinder_kwargs = {"body": cylinder_body, "radius": 0.00375, "half_height": 0.0015, "cfg": cfg}
                    if kind == "rounded":
                        cylinder_kwargs["barrel_radius"] = 0.02
                    capsule_kwargs = {
                        "body": capsule_body,
                        "radius": 0.00025,
                        "half_height": 0.005 if kind == "cap_bridge" else 0.0005,
                        "cfg": cfg,
                    }
                    if reversed_order:
                        builder.add_shape_capsule(**capsule_kwargs)
                        builder.add_shape_cylinder(**cylinder_kwargs)
                    else:
                        builder.add_shape_cylinder(**cylinder_kwargs)
                        builder.add_shape_capsule(**capsule_kwargs)
                    model = builder.finalize()
                    state = model.state()
                    poses = model.body_q.numpy()
                    poses[capsule_body, :3] = (0.00395, 0.0, 0.0003)
                    axis = wp.vec3(0.0, 1.0, 0.0)
                    if kind in ("parallel", "parallel_clipped", "cap_axial", "cap_axial_bottom"):
                        axis = wp.vec3(0.0, 0.0, 1.0)
                    elif kind == "near_parallel":
                        axis = wp.normalize(wp.vec3(0.05, 0.0, 1.0))
                    elif kind == "cap_tilted":
                        axis = wp.normalize(wp.vec3(1.0, 0.0, 0.01))
                    elif kind in (
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_outside",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_bottom",
                    ):
                        axis = wp.vec3(1.0, 0.0, 0.0)
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    if kind in (
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_outside",
                        "cap_tilted",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_axial",
                        "cap_axial_bottom",
                    ):
                        poses[capsule_body, :3] = (0.0, 0.0, 0.0017)
                    if kind == "cap_clipped":
                        poses[capsule_body, 0] = 0.0037
                    elif kind == "cap_clipped_negative":
                        poses[capsule_body, 0] = -0.0037
                    elif kind == "cap_outside":
                        poses[capsule_body, 0] = 0.0045
                    elif kind == "cap_penetrating":
                        poses[capsule_body, 2] = 0.00145
                    elif kind == "cap_deep":
                        poses[capsule_body, 2] = 0.001
                    elif kind == "cap_bottom":
                        poses[capsule_body, :3] = (0.0, 0.0, -0.0017)
                    elif kind == "cap_axial_bottom":
                        poses[capsule_body, :3] = (0.0, 0.0, -0.0017)
                    elif kind == "parallel_clipped":
                        poses[capsule_body, 2] = 0.0013
                    elif kind == "rim":
                        poses[capsule_body, 2] = 0.0016
                    elif kind == "core":
                        poses[capsule_body, 0] = 0.0037
                    state.body_q.assign(poses)
                    pipeline = newton.CollisionPipeline(model)
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts)
                    count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
                    manifold_kinds = (
                        "barrel",
                        "parallel",
                        "parallel_clipped",
                        "near_parallel",
                        "cap_face",
                        "cap_clipped",
                        "cap_clipped_negative",
                        "cap_bridge",
                        "cap_tilted",
                        "cap_penetrating",
                        "cap_deep",
                        "cap_bottom",
                        "cap_axial",
                        "cap_axial_bottom",
                    )
                    test.assertEqual(count, 1 if kind == "rounded" else 0)
                    if kind in ("cap_outside", "rim", "core"):
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    if kind in manifold_kinds and kind != "near_parallel":
                        expected_count = 1 if kind in ("barrel", "cap_axial", "cap_axial_bottom") else 2
                        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), expected_count)
                        bodies = model.shape_body.numpy()
                        core_points = []
                        for i in range(expected_count):
                            a = int(contacts.rigid_contact_shape0.numpy()[i])
                            b = int(contacts.rigid_contact_shape1.numpy()[i])
                            x0 = np.asarray(
                                wp.transform_point(
                                    wp.transform(*poses[bodies[a]]), wp.vec3(*contacts.rigid_contact_point0.numpy()[i])
                                )
                            )
                            x1 = np.asarray(
                                wp.transform_point(
                                    wp.transform(*poses[bodies[b]]), wp.vec3(*contacts.rigid_contact_point1.numpy()[i])
                                )
                            )
                            normal = contacts.rigid_contact_normal.numpy()[i]
                            gap = float(
                                (x1 - x0) @ normal
                                - contacts.rigid_contact_margin0.numpy()[i]
                                - contacts.rigid_contact_margin1.numpy()[i]
                            )
                            expected_gap = -0.00005
                            if kind == "cap_tilted":
                                capsule_core = x0 if bodies[a] == capsule_body else x1
                                expected_gap = capsule_core[2] - 0.0015 - 0.00025
                            elif kind == "cap_penetrating":
                                expected_gap = -0.0003
                            elif kind == "cap_deep":
                                expected_gap = -0.00075
                            elif kind in ("cap_axial", "cap_axial_bottom"):
                                expected_gap = -0.00055
                            test.assertAlmostEqual(gap, expected_gap, delta=1e-8)
                            if kind in ("cap_bottom", "cap_axial_bottom"):
                                expected = np.array([0.0, 0.0, -1.0])
                            elif kind in (
                                "cap_face",
                                "cap_clipped",
                                "cap_clipped_negative",
                                "cap_bridge",
                                "cap_tilted",
                                "cap_penetrating",
                                "cap_deep",
                                "cap_axial",
                            ):
                                expected = np.array([0.0, 0.0, 1.0])
                            else:
                                expected = np.array([1.0, 0.0, 0.0])
                            if bodies[a] == capsule_body:
                                expected = -expected
                            np.testing.assert_allclose(normal, expected, atol=1e-6)
                            core_points.append(x0 if bodies[a] == capsule_body else x1)
                        if expected_count == 2:
                            expected_length = 0.001
                            if kind == "parallel_clipped":
                                expected_length = 0.0007
                            elif kind in ("cap_clipped", "cap_clipped_negative"):
                                expected_length = 0.00055
                            elif kind == "cap_bridge":
                                expected_length = 0.0075
                            test.assertAlmostEqual(
                                np.linalg.norm(core_points[1] - core_points[0]), expected_length, delta=1e-8
                            )
                            expected_cap_x = {
                                "cap_face": [-0.0005, 0.0005],
                                "cap_clipped": [0.0032, 0.00375],
                                "cap_clipped_negative": [-0.00375, -0.0032],
                                "cap_bridge": [-0.00375, 0.00375],
                            }
                            if kind in expected_cap_x:
                                np.testing.assert_allclose(
                                    sorted(point[0] for point in core_points), expected_cap_x[kind], atol=1e-8
                                )


def test_core_crosses_both_caps(test, device):
    """Resolve a core crossing both caps with the whole capsule's support gap."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.01)
        cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
        capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
        builder.add_shape_cylinder(body=cylinder_body, radius=10.0, half_height=1.0, cfg=cfg)
        builder.add_shape_capsule(body=capsule_body, radius=0.25, half_height=3.0, cfg=cfg)
        model = builder.finalize()
        state = model.state()
        poses = model.body_q.numpy()
        poses[capsule_body, :3] = (0.0, 0.0, 0.1)
        axis = wp.normalize(wp.vec3(1.0, 0.0, 1.0))
        poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
        state.body_q.assign(poses)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 0)
        bodies = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()
        point0 = contacts.rigid_contact_point0.numpy()
        point1 = contacts.rigid_contact_point1.numpy()
        for i in range(count):
            if bodies[shape0[i]] == capsule_body:
                capsule_local, cylinder_local = point0[i], point1[i]
            else:
                capsule_local, cylinder_local = point1[i], point0[i]
            capsule_world = np.asarray(wp.transform_point(wp.transform(*poses[capsule_body]), wp.vec3(*capsule_local)))
            cylinder_world = np.asarray(
                wp.transform_point(wp.transform(*poses[cylinder_body]), wp.vec3(*cylinder_local))
            )
            normal = contacts.rigid_contact_normal.numpy()[i]
            if bodies[shape0[i]] == capsule_body:
                np.testing.assert_allclose(normal, [0.0, 0.0, -1.0], atol=1e-6)
            else:
                np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-6)
            gap = (
                capsule_world[2]
                - cylinder_world[2]
                - contacts.rigid_contact_margin0.numpy()[i]
                - contacts.rigid_contact_margin1.numpy()[i]
            )
            # Moving upward must lift the bottommost core endpoint past the
            # top cap. Clipping the core to z=0 understates this penetration.
            test.assertAlmostEqual(gap, 0.1 - 3.0 / np.sqrt(2.0) - 1.0 - 0.25, delta=2e-6)


def test_rim_overhang_and_near_parallel(test, device):
    """Preserve rim gaps and two barrel witnesses when the capsule tilts."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in ("rim", "barrel"):
                angles = (30.0, 60.0, 80.0) if kind == "rim" else (0.0, 0.001, -0.001)
                for angle in angles:
                    for direction in (-1.0, 1.0):
                        for drop in (0.0, 0.05):
                            with test.subTest(order=reversed_order, kind=kind, angle=angle, cap=direction, drop=drop):
                                builder = newton.ModelBuilder()
                                cfg = newton.ModelBuilder.ShapeConfig(gap=0.2)
                                body = builder.add_body()
                                cylinder = {"body": -1, "radius": 1.0, "half_height": 1.0, "cfg": cfg}
                                capsule = {
                                    "body": body,
                                    "radius": 0.1,
                                    "half_height": 2.0 if kind == "rim" else 0.5,
                                    "cfg": cfg,
                                }
                                if reversed_order:
                                    builder.add_shape_capsule(**capsule)
                                    builder.add_shape_cylinder(**cylinder)
                                else:
                                    builder.add_shape_cylinder(**cylinder)
                                    builder.add_shape_capsule(**capsule)
                                if kind == "rim":
                                    theta = np.radians(angle)
                                    axis = np.array([np.cos(theta), 0.0, -direction * np.sin(theta)])
                                    outward = np.array([np.sin(theta), 0.0, direction * np.cos(theta)])
                                    center = np.array([1.0, 0.0, direction]) + 0.1 * outward - 0.5 * axis
                                    center[2] -= direction * drop
                                else:
                                    axis = direction * np.array([np.sin(angle), 0.0, np.cos(angle)])
                                    center = [1.1 + 0.5 * abs(np.sin(angle)) - drop, 0.0, 0.0]
                                model = builder.finalize()
                                state = model.state()
                                rotation = wp.quat_from_axis_angle(
                                    wp.vec3(0.0, 1.0, 0.0), float(np.arctan2(axis[0], axis[2]))
                                )
                                pose = wp.transform(wp.vec3(*center), rotation)
                                state.body_q.assign([pose])
                                pipeline = newton.CollisionPipeline(model)
                                contacts = pipeline.contacts()
                                pipeline.collide(state, contacts)
                                count = int(contacts.rigid_contact_count.numpy()[0])
                                test.assertGreater(count, 0)
                                test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 0)
                                gaps = []
                                core_points = []
                                for i in range(count):
                                    points = []
                                    for shape, point in (
                                        (
                                            contacts.rigid_contact_shape0.numpy()[i],
                                            contacts.rigid_contact_point0.numpy()[i],
                                        ),
                                        (
                                            contacts.rigid_contact_shape1.numpy()[i],
                                            contacts.rigid_contact_point1.numpy()[i],
                                        ),
                                    ):
                                        world = np.asarray(point)
                                        if model.shape_body.numpy()[shape] == body:
                                            world = np.asarray(wp.transform_point(pose, wp.vec3(*point)))
                                            core_points.append(world)
                                        points.append(world)
                                    normal = contacts.rigid_contact_normal.numpy()[i]
                                    gaps.append(
                                        (points[1] - points[0]) @ normal
                                        - contacts.rigid_contact_margin0.numpy()[i]
                                        - contacts.rigid_contact_margin1.numpy()[i]
                                    )
                                    if kind == "rim":
                                        test.assertGreater(abs(normal[0]), 0.1)
                                if kind == "rim":
                                    test.assertAlmostEqual(min(gaps), -drop * np.cos(theta), delta=5e-4)
                                else:
                                    test.assertEqual(count, 2)
                                    np.testing.assert_allclose(
                                        sorted(p[2] for p in core_points),
                                        [-0.5 * np.cos(angle), 0.5 * np.cos(angle)],
                                        atol=1e-6,
                                    )
                                    np.testing.assert_allclose(
                                        sorted(gaps), [-drop, abs(np.sin(angle)) - drop], atol=1e-6
                                    )


def test_gap_admission(test, device):
    """Generate separated barrel and cap contacts inside the combined shape gap."""
    with wp.ScopedDevice(device):
        for kind, positions in (
            ("barrel", (0.0043, 0.0046)),
            ("cap", (0.00205, 0.00235)),
        ):
            for position, expected_count in zip(positions, (1 if kind == "barrel" else 2, 0), strict=True):
                with test.subTest(kind=kind, position=position):
                    builder = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.00025)
                    cylinder_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    capsule_body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
                    builder.add_shape_cylinder(body=cylinder_body, radius=0.00375, half_height=0.0015, cfg=cfg)
                    builder.add_shape_capsule(body=capsule_body, radius=0.00025, half_height=0.0005, cfg=cfg)
                    model = builder.finalize()
                    state = model.state()
                    poses = model.body_q.numpy()
                    if kind == "barrel":
                        poses[capsule_body, :3] = (position, 0.0, 0.0)
                        axis = wp.vec3(0.0, 1.0, 0.0)
                    else:
                        poses[capsule_body, :3] = (0.0, 0.0, position)
                        axis = wp.vec3(1.0, 0.0, 0.0)
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    state.body_q.assign(poses)
                    for speculative in (False, True):
                        with test.subTest(speculative=speculative):
                            pipeline = newton.CollisionPipeline(
                                model,
                                speculative_contact_gap_max=0.001 if speculative else None,
                            )
                            contacts = pipeline.contacts()
                            if speculative:
                                pipeline.collide(state, contacts, dt=0.01)
                            else:
                                pipeline.collide(state, contacts)
                            test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), expected_count)


def test_generic_dispatch_bounds(test, device):
    """Handle sharp pairs without GJK and keep live rounded-cylinder routing."""
    with wp.ScopedDevice(device):
        for rounded in (False, True):
            for reversed_order in (False, True):
                builder = newton.ModelBuilder()
                cylinder = {"body": -1, "radius": 1.0, "half_height": 1.0}
                if rounded:
                    cylinder["barrel_radius"] = 2.0
                body = builder.add_body(xform=wp.transform(wp.vec3(1.09, 0.0, 0.0), wp.quat_identity()))
                capsule = {"body": body, "radius": 0.1, "half_height": 0.5}
                if reversed_order:
                    builder.add_shape_capsule(**capsule)
                    builder.add_shape_cylinder(**cylinder)
                else:
                    builder.add_shape_cylinder(**cylinder)
                    builder.add_shape_capsule(**capsule)
                model = builder.finalize()
                state = model.state()
                for mode in ("nxn", "sap", "explicit"):
                    with test.subTest(rounded=rounded, order=reversed_order, broad_phase=mode):
                        pipeline = newton.CollisionPipeline(
                            model,
                            broad_phase=mode,
                            shape_pairs_filtered=wp.array([[0, 1]], dtype=wp.vec2i) if mode == "explicit" else None,
                        )
                        contacts = pipeline.contacts()
                        pipeline.collide(state, contacts)
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                        test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), int(rounded))
                        if not rounded:
                            scale = model.shape_scale.numpy()
                            cylinder_index = int(reversed_order)
                            scale[cylinder_index, 2] = 2.0
                            model.shape_scale.assign(scale)
                            pipeline.collide(state, contacts)
                            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
                            test.assertEqual(int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]), 1)
                            scale[cylinder_index, 2] = 0.0
                            model.shape_scale.assign(scale)


class TestCapsuleCylinderBarrel(unittest.TestCase):
    """Check point, line, rim, and penetrating contacts."""


add_function_test(TestCapsuleCylinderBarrel, "test_barrel_guards", test_barrel_guards, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_support_witnesses", test_support_witnesses, devices=get_test_devices()
)
add_function_test(TestCapsuleCylinderBarrel, "test_rim_conditioning", test_rim_conditioning, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_generic_dispatch_bounds", test_generic_dispatch_bounds, devices=get_test_devices()
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_core_crosses_both_caps",
    test_core_crosses_both_caps,
    devices=get_test_devices(),
)
add_function_test(TestCapsuleCylinderBarrel, "test_gap_admission", test_gap_admission, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_rim_overhang_and_near_parallel",
    test_rim_overhang_and_near_parallel,
    devices=get_test_devices(),
)
add_function_test(
    TestCapsuleCylinderBarrel, "test_pipeline_dispatch", test_pipeline_dispatch, devices=get_test_devices()
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
