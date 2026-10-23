# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the bounded analytic barrel witness and its general-query fallbacks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.geometry.collision_primitive import _collide_capsule_cylinder_barrel
from newton._src.geometry.narrow_phase import NarrowPhase
from newton._src.sim.collide import write_contact, write_contact_speculative
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
    """Evaluate barrel witnesses and fallback guards in a batch."""
    i = wp.tid()
    ok, distance, center, normal = _collide_capsule_cylinder_barrel(
        positions[i], axes[i], 0.25, 0.5, wp.vec3(0.0), wp.vec3(0.0, 0.0, 1.0), 3.75, 1.5
    )
    handled[i] = int(ok)
    distances[i] = distance
    centers[i] = center
    normals[i] = normal


def test_barrel_guards(test, device):
    """Preserve parallel, cap/rim and core-intersection fallback; handle endpoints."""
    positions = [
        (3.95, 0.0, 0.3),  # interior barrel witness
        (4.45, 0.0, 0.3),  # core endpoint is the closest witness
        (4.1, 0.0, 0.3),  # separated but analytically handled
        (3.95, 0.0, 0.3),  # parallel: preserve the general manifold
        (3.95, 0.0, 1.5),  # exact rim: fallback
        (3.95, 0.0, 1.6),  # outside barrel: fallback
        (3.7, 0.0, 0.3),  # core intersects cylinder: fallback
        (3.95, 0.0, 0.3),  # near-parallel: preserve the general manifold
        (3.75, 0.0, 0.3),  # core exactly on barrel: fallback
        (3.95, 0.0, -1.5),  # lower rim: fallback
        (0.0, 0.0, 1.6),  # cap interior: fallback
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
    axes.extend([(0.0001, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
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
    np.testing.assert_array_equal(handled.numpy(), [1, 1, 1] + [0] * (n - 3))
    np.testing.assert_allclose(distances.numpy()[:3], [-0.05, -0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(normals.numpy()[:3], [[-1.0, 0.0, 0.0]] * 3, atol=1e-6)
    np.testing.assert_allclose(centers.numpy()[:3], [[3.725, 0.0, 0.3], [3.725, 0.0, 0.3], [3.8, 0.0, 0.3]], atol=1e-6)
    test.assertTrue(np.all(distances.numpy()[3:] >= MAXVAL * 0.99))


def test_pipeline_dispatch(test, device, speculative=False, sparse_split=False, transformed=False):
    """Use analytic barrel contacts in either shape order; keep general fallbacks."""
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for kind in (
                "barrel",
                "endpoint",
                "separated",
                "parallel",
                "near_parallel",
                "rim",
                "cap",
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
                    capsule_kwargs = {"body": capsule_body, "radius": 0.00025, "half_height": 0.0005, "cfg": cfg}
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
                    axis = wp.vec3(0.0, 0.0, 1.0) if kind == "parallel" else wp.vec3(0.0, 1.0, 0.0)
                    if kind == "endpoint":
                        poses[capsule_body, 0] = 0.00445
                        axis = wp.vec3(1.0, 0.0, 0.0)
                    elif kind == "separated":
                        poses[capsule_body, 0] = 0.005
                    elif kind == "near_parallel":
                        axis = wp.normalize(wp.vec3(0.0001, 0.0, 1.0))
                    poses[capsule_body, 3:] = np.asarray(wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), axis))
                    if kind == "rim":
                        poses[capsule_body, 2] = 0.0016
                    elif kind == "core":
                        poses[capsule_body, 0] = 0.0037
                    elif kind == "cap":
                        poses[capsule_body, :3] = (0.0, 0.0, 0.0016)
                    world_rotation = wp.quat_identity()
                    if transformed:
                        world_rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
                        world_transform = wp.transform(wp.vec3(0.01, -0.02, 0.03), world_rotation)
                        for body in (capsule_body, cylinder_body):
                            poses[body] = np.asarray(world_transform * wp.transform(*poses[body]))
                    state.body_q.assign(poses)
                    pipeline = newton.CollisionPipeline(
                        model, speculative_contact_gap_max=0.001 if speculative else None
                    )
                    if sparse_split:
                        pipeline.narrow_phase = NarrowPhase(
                            max_candidate_pairs=pipeline.shape_pairs_max,
                            max_triangle_pairs=1,
                            device=device,
                            shape_aabb_lower=pipeline.narrow_phase.shape_aabb_lower,
                            shape_aabb_upper=pipeline.narrow_phase.shape_aabb_upper,
                            contact_writer_warp_func=write_contact_speculative if speculative else write_contact,
                            has_meshes=False,
                            sparse_gjk_pairs=True,
                            split_gjk_mpr=True,
                            contact_max=pipeline.rigid_contact_max,
                            speculative=speculative,
                            contact_writer_supports_speculative=speculative,
                        )
                    contacts = pipeline.contacts()
                    pipeline.collide(state, contacts, dt=1.0 / 60.0 if speculative else None)
                    count = int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0])
                    test.assertEqual(count, 0 if kind in ("barrel", "endpoint", "separated") else 1)
                    if kind == "separated":
                        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
                    elif kind in ("barrel", "endpoint"):
                        test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
                        a = int(contacts.rigid_contact_shape0.numpy()[0])
                        b = int(contacts.rigid_contact_shape1.numpy()[0])
                        bodies = model.shape_body.numpy()
                        x0 = np.asarray(
                            wp.transform_point(
                                wp.transform(*poses[bodies[a]]), wp.vec3(*contacts.rigid_contact_point0.numpy()[0])
                            )
                        )
                        x1 = np.asarray(
                            wp.transform_point(
                                wp.transform(*poses[bodies[b]]), wp.vec3(*contacts.rigid_contact_point1.numpy()[0])
                            )
                        )
                        normal = contacts.rigid_contact_normal.numpy()[0]
                        gap = float(
                            (x1 - x0) @ normal
                            - contacts.rigid_contact_margin0.numpy()[0]
                            - contacts.rigid_contact_margin1.numpy()[0]
                        )
                        test.assertAlmostEqual(gap, -0.00005, delta=1e-8)
                        expected = (
                            np.array([1.0, 0.0, 0.0]) if bodies[a] == cylinder_body else np.array([-1.0, 0.0, 0.0])
                        )
                        expected = np.asarray(wp.quat_rotate(world_rotation, wp.vec3(*expected)))
                        np.testing.assert_allclose(normal, expected, atol=1e-6)
                    else:
                        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)


def _review_case_contacts(
    center,
    rotation,
    capsule_radius,
    capsule_half_length,
    cylinder_radius,
    cylinder_half_height,
    reversed_order,
    speculative,
):
    """Return physical surface witnesses for a reviewer-supplied configuration."""
    builder = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, gap=0.2)
    body = builder.add_body(mass=1.0, inertia=wp.mat33(*np.eye(3).ravel()))
    capsule = {"body": body, "radius": capsule_radius, "half_height": capsule_half_length, "cfg": cfg}
    cylinder = {"body": -1, "radius": cylinder_radius, "half_height": cylinder_half_height, "cfg": cfg}
    if reversed_order:
        capsule_shape = builder.add_shape_capsule(**capsule)
        builder.add_shape_cylinder(**cylinder)
    else:
        builder.add_shape_cylinder(**cylinder)
        capsule_shape = builder.add_shape_capsule(**capsule)
    model = builder.finalize()
    state = model.state()
    pose = wp.transform(wp.vec3(*center), rotation)
    state.body_q.assign([pose])
    axis = np.asarray(wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0)), dtype=float)
    axis /= np.linalg.norm(axis)
    pipeline = newton.CollisionPipeline(model, speculative_contact_gap_max=0.2 if speculative else None)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts, dt=1.0 / 60.0 if speculative else None)
    count = int(contacts.rigid_contact_count.numpy()[0])
    shapes = contacts.rigid_contact_shape0.numpy()[:count]
    points0 = contacts.rigid_contact_point0.numpy()[:count]
    points1 = contacts.rigid_contact_point1.numpy()[:count]
    margins0 = contacts.rigid_contact_margin0.numpy()[:count]
    margins1 = contacts.rigid_contact_margin1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    gaps, capsule_errors, cylinder_errors, outward_normals = [], [], [], []
    for shape, local0, local1, margin0, margin1, normal in zip(
        shapes, points0, points1, margins0, margins1, normals, strict=True
    ):
        point0 = (
            np.asarray(wp.transform_point(pose, wp.vec3(*local0)), dtype=float) if shape == capsule_shape else local0
        )
        point1 = (
            np.asarray(wp.transform_point(pose, wp.vec3(*local1)), dtype=float) if shape != capsule_shape else local1
        )
        gaps.append((point1 - point0) @ normal - margin0 - margin1)
        surface0, surface1 = point0 + margin0 * normal, point1 - margin1 * normal
        if shape == capsule_shape:
            capsule_surface, cylinder_surface = surface0, surface1
            outward_normals.append(-normal)
        else:
            capsule_surface, cylinder_surface = surface1, surface0
            outward_normals.append(normal)
        along = np.clip((capsule_surface - center) @ axis, -capsule_half_length, capsule_half_length)
        capsule_errors.append(abs(np.linalg.norm(capsule_surface - center - along * axis) - capsule_radius))
        radial = np.linalg.norm(cylinder_surface[:2]) - cylinder_radius
        axial = abs(cylinder_surface[2]) - cylinder_half_height
        # Exact distance to the sharp cylinder surface, including cap/rim points.
        cylinder_errors.append(abs(min(max(radial, axial), 0.0) + np.hypot(max(radial, 0.0), max(axial, 0.0))))
    return (
        int(pipeline.narrow_phase.gjk_candidate_pairs_count.numpy()[0]),
        np.asarray(gaps),
        np.asarray(outward_normals),
        np.asarray(capsule_errors),
        np.asarray(cylinder_errors),
    )


def test_review_rim_overhang(test, device):
    """Preserve rim contacts and normals for the reviewer's tilted-overhang repro."""
    # Main retains a 0.1 mm support radius; allow that bias without accepting
    # the 15-100 mm gap errors or missing contacts from the earlier cap solver.
    tolerance = 1.2e-4
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for speculative in (False, True):
                for degrees in (30.0, 60.0, 80.0):
                    theta = np.radians(degrees)
                    axis = np.array([np.cos(theta), 0.0, -np.sin(theta)])
                    outward = np.array([np.sin(theta), 0.0, np.cos(theta)])
                    rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*axis))
                    for drop in (0.0, 0.05):
                        with test.subTest(order=reversed_order, speculative=speculative, tilt=degrees, drop=drop):
                            center = np.array([1.0, 0.0, 1.0]) + 0.1 * outward - 0.5 * axis - [0.0, 0.0, drop]
                            routed, gaps, normals, capsule_errors, cylinder_errors = _review_case_contacts(
                                center, rotation, 0.1, 2.0, 1.0, 1.0, reversed_order, speculative
                            )
                            test.assertEqual(routed, 1)
                            test.assertGreater(len(gaps), 0)
                            np.testing.assert_allclose(gaps, -drop * np.cos(theta), atol=tolerance)
                            np.testing.assert_allclose(normals, np.tile(outward, (len(gaps), 1)), atol=2e-4)
                            test.assertLess(np.max(capsule_errors), tolerance)
                            test.assertLess(np.max(cylinder_errors), 2e-6)


def test_review_near_horizontal_witnesses(test, device):
    """Keep reconstructed witnesses on the shapes in the attached cap/rim repro."""
    rotation = wp.quat(0.0, -np.sqrt(0.5), 0.0, np.sqrt(0.5))
    with wp.ScopedDevice(device):
        for reversed_order in (False, True):
            for speculative in (False, True):
                for z, radius in ((-0.54, 0.75), (-0.719, 0.01), (-0.72, 0.01), (-0.725, 0.01), (-0.73, 0.01)):
                    with test.subTest(order=reversed_order, speculative=speculative, z=z, radius=radius):
                        routed, gaps, normals, capsule_errors, cylinder_errors = _review_case_contacts(
                            np.array([0.8, 0.85, z]), rotation, radius, 0.94, 1.38, 0.72, reversed_order, speculative
                        )
                        test.assertEqual(routed, 1)
                        test.assertEqual(len(gaps), 2)
                        expected_gap = -abs(z + 0.72) - radius if z >= -0.72 else abs(z + 0.72) - radius
                        np.testing.assert_allclose(gaps, expected_gap, atol=1.2e-4)
                        np.testing.assert_allclose(normals, np.tile([0.0, 0.0, -1.0], (len(gaps), 1)), atol=2e-4)
                        test.assertLess(np.max(capsule_errors), 1.2e-4)
                        test.assertLess(np.max(cylinder_errors), 2e-6)


class TestCapsuleCylinderBarrel(unittest.TestCase):
    """The private shortcut must never claim the general cap/manifold cases."""


add_function_test(TestCapsuleCylinderBarrel, "test_barrel_guards", test_barrel_guards, devices=get_test_devices())
add_function_test(
    TestCapsuleCylinderBarrel, "test_pipeline_dispatch", test_pipeline_dispatch, devices=get_test_devices()
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_pipeline_dispatch_speculative",
    test_pipeline_dispatch,
    devices=get_test_devices(),
    speculative=True,
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_pipeline_dispatch_sparse_split",
    test_pipeline_dispatch,
    devices=get_test_devices(),
    sparse_split=True,
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_pipeline_dispatch_transformed",
    test_pipeline_dispatch,
    devices=get_test_devices(),
    transformed=True,
)
add_function_test(
    TestCapsuleCylinderBarrel, "test_review_rim_overhang", test_review_rim_overhang, devices=get_test_devices()
)
add_function_test(
    TestCapsuleCylinderBarrel,
    "test_review_near_horizontal_witnesses",
    test_review_near_horizontal_witnesses,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
