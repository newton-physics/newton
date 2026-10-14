# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cover MPR portals whose normal projection lies outside the portal triangle."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.mpr import create_solve_mpr
from newton._src.geometry.support_function import (
    GenericShapeData,
    SupportMapDataProvider,
    pack_mesh_ptr,
    support_map,
    support_map_lean,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _surface_error(vertices, indices, point):
    """Measure signed distance to the most violated convex face plane."""
    triangles = vertices[indices.reshape(-1, 3)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    inward = np.sum(normals * (vertices.mean(axis=0) - triangles[:, 0]), axis=1) > 0.0
    normals[inward] *= -1.0
    return float(np.max(np.sum(normals * (point - triangles[:, 0]), axis=1)))


@wp.func
def _failed_distance_query(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    orientation_b: wp.quat,
    position_b: wp.vec3,
    extend: float,
    data_provider: SupportMapDataProvider,
    MAX_ITER: int,
    COLLIDE_EPSILON: float,
) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float]:
    """Simulate a distance query that cannot obtain separated witnesses."""
    return False, wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0), 0.0


@wp.func
def _unconverged_distance_query(
    geom_a: GenericShapeData,
    geom_b: GenericShapeData,
    orientation_b: wp.quat,
    position_b: wp.vec3,
    extend: float,
    data_provider: SupportMapDataProvider,
    MAX_ITER: int,
    COLLIDE_EPSILON: float,
) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float]:
    """Simulate an early GJK exit with interior witnesses."""
    return True, wp.vec3(0.0), position_b, wp.normalize(position_b), wp.length(position_b)


def _create_witness_kernel(distance_query=None):
    if distance_query is not None:
        with patch("newton._src.geometry.simplex_solver.create_solve_closest_distance") as factory:
            factory.return_value.core = distance_query
            solve = create_solve_mpr(support_map_lean).core
    else:
        solve = create_solve_mpr(support_map_lean).core

    @wp.kernel
    def kernel(
        mesh_a: wp.uint64,
        mesh_b: wp.uint64,
        rotation: wp.quat,
        position: wp.vec3,
        max_iter: int,
        extend: float,
        hit: wp.array[int],
        points: wp.array[wp.vec3],
        depth: wp.array[float],
    ):
        shape_a = GenericShapeData()
        shape_a.shape_type = int(newton.GeoType.CONVEX_MESH)
        shape_a.shape_index = -1
        shape_a.scale = wp.vec3(1.0)
        shape_a.auxiliary = pack_mesh_ptr(mesh_a)
        shape_b = shape_a
        shape_b.auxiliary = pack_mesh_ptr(mesh_b)
        collision, pa, pb, normal, penetration = solve(
            shape_a, shape_b, rotation, position, extend, SupportMapDataProvider(), max_iter
        )
        hit[0] = int(collision)
        points[0] = pa - normal * (0.5 * extend)
        points[1] = pb + normal * (0.5 * extend)
        points[2] = normal
        depth[0] = penetration - extend

    return kernel


def test_convex_contact_witness(test, device, *, swap=False, margin=0.0, capture=False):
    """Keep zero-margin witnesses on both hulls for the saved pose and nearby poses."""
    # Public two-hull fixture from https://github.com/newton-physics/newton/issues/4414.
    with np.load(Path(__file__).parent / "assets" / "convex_contact_4414.npz", allow_pickle=False) as fixture:
        vertices = [fixture[key + "_vertices"].copy() for key in ("torso", "elbow")]
        indices = [fixture[key + "_indices"].astype(np.int32) for key in ("torso", "elbow")]
        position = fixture["position"].copy()
        rotation = wp.quat(*fixture["rotation"])
    if swap:
        vertices.reverse()
        indices.reverse()

    poses = [(position, rotation)]
    for axis in range(3):
        direction = np.eye(3, dtype=np.float32)[axis]
        for sign in (-1.0, 1.0):
            poses.append((position + sign * 0.0001 * direction, rotation))
            poses.append((position, wp.quat_from_axis_angle(wp.vec3(direction), sign * 0.001) * rotation))

    # These wider perturbations require multiple retries; shrinking the seed
    # into each new tetrahedron makes the final portal numerically degenerate.
    wider_perturbations = (
        ((1.5673162869, -0.1565069156, 0.7797705622), (0.0035926571, 0.0044446284, 0.0084114846), 0.0910780397),
        ((0.1183510903, -0.5681262162, 0.9866573833), (0.0007874704, 0.0086510388, -0.0045985128), -0.0468000330),
    )
    for axis, delta_position, angle in wider_perturbations:
        direction = np.asarray(axis)
        direction /= np.linalg.norm(direction)
        delta_rotation = wp.quat_from_axis_angle(wp.vec3(direction), angle)
        poses.append((position + delta_position, delta_rotation * rotation))

    raycast_pose_index = len(poses)
    # This pose reaches the raycast fallback. Its support points lie on the
    # hulls but lose surface membership if the writer projects them onto a
    # normal that is not aligned with their difference.
    poses.append(
        (
            np.array([0.030662711709737778, 0.03921249508857727, 0.15601256489753723]),
            wp.quat(0.4862031936645508, 0.7860594987869263, -0.33473095297813416, 0.18349947035312653),
        )
    )

    offset = wp.vec3(0.1, -0.03, 0.02)
    builder = newton.ModelBuilder()
    for side in range(2):
        body = builder.add_body()
        mesh = newton.Mesh(vertices[side], indices[side])
        builder.add_shape_convex_hull(
            body,
            mesh=mesh,
            xform=wp.transform(offset, wp.quat_identity()),
            cfg=newton.ModelBuilder.ShapeConfig(gap=0.01, margin=margin),
        )
    model = builder.finalize(device=device)
    state = model.state()
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    contacts = pipeline.contacts()
    graph = None
    if capture:
        test.assertTrue(pipeline.narrow_phase.split_gjk_mpr)
    for pose_index, (shape_position, shape_rotation) in enumerate(poses):
        with test.subTest(pose=pose_index):
            body_transforms = [
                wp.transform(-offset, wp.quat_identity()),
                wp.transform(wp.vec3(shape_position) - wp.quat_rotate(shape_rotation, offset), shape_rotation),
            ]
            if swap:
                body_transforms.reverse()
            state.body_q.assign(np.asarray(body_transforms, dtype=np.float32))
            if capture:
                if graph is None:
                    # Load kernels before capturing the reusable work queues.
                    pipeline.collide(state, contacts)
                    with wp.ScopedCapture(device=device) as captured:
                        pipeline.collide(state, contacts)
                    graph = captured.graph
                wp.capture_launch(graph)
            else:
                pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            shapes = [contacts.rigid_contact_shape0.numpy(), contacts.rigid_contact_shape1.numpy()]
            points = [contacts.rigid_contact_point0.numpy(), contacts.rigid_contact_point1.numpy()]
            normals = contacts.rigid_contact_normal.numpy()
            if capture and pose_index == 0:
                original_b = 0 if swap else 1
                direction = normals[0] if int(shapes[1][0]) == original_b else -normals[0]
                # Reuse a slot for penetration, then ordinary separated GJK,
                # then penetration again without rebuilding the captured graph.
                poses.extend(((position + direction * 0.05354986867157328, rotation), (position, rotation)))
            for contact in range(count):
                world_points = []
                for side in range(2):
                    shape = int(shapes[side][contact])
                    local_point = points[side][contact].astype(np.float64) - np.asarray(offset, dtype=np.float64)
                    error = _surface_error(vertices[shape].astype(np.float64), indices[shape], local_point)
                    test.assertLessEqual(abs(error), 1.0e-5, f"shape {shape}: surface error {error} m")
                    world_points.append(
                        np.asarray(wp.transform_point(body_transforms[shape], wp.vec3(points[side][contact])))
                    )
                test.assertAlmostEqual(float(np.linalg.norm(normals[contact])), 1.0, delta=1.0e-6)
                depth = -float(np.dot(world_points[1] - world_points[0], normals[contact]))
                if capture and pose_index == len(poses) - 2:
                    test.assertAlmostEqual(depth, -0.001, delta=1.0e-5)
                else:
                    test.assertGreater(depth, 0.0)
                if pose_index == raycast_pose_index:
                    # Preserve the original ray's 65.76 mm depth bound;
                    # aligning distant support vertices would give 176 mm.
                    test.assertLessEqual(depth, 0.06576)
                if pose_index == 0 or (capture and pose_index == len(poses) - 1):
                    # Independent FP64 convex Minkowski-difference boundary calculation.
                    test.assertAlmostEqual(depth, 0.05254986867157328, delta=1.0e-5)


class TestConvexContactWitness(unittest.TestCase):
    pass


def test_convex_contact_iteration_limit(test, device, *, distance_query=None):
    """Report hull witnesses, a supported normal, and its exact depth when refinement cannot converge."""
    with np.load(Path(__file__).parent / "assets" / "convex_contact_4414.npz", allow_pickle=False) as fixture:
        vertices = [fixture[key + "_vertices"].copy() for key in ("torso", "elbow")]
        indices = [fixture[key + "_indices"].astype(np.int32) for key in ("torso", "elbow")]
        position = wp.vec3(fixture["position"])
        rotation = wp.quat(*fixture["rotation"])
    meshes = [
        wp.Mesh(points=wp.array(v, dtype=wp.vec3, device=device), indices=wp.array(i, dtype=int, device=device))
        for v, i in zip(vertices, indices, strict=True)
    ]
    kernel = _create_witness_kernel(distance_query)
    hit = wp.zeros(1, dtype=int, device=device)
    points = wp.zeros(3, dtype=wp.vec3, device=device)
    penetration = wp.zeros(1, dtype=float, device=device)
    world_b = np.asarray([wp.quat_rotate(rotation, wp.vec3(v)) + position for v in vertices[1]], dtype=np.float64)
    for budget, extend in ((1, 0.0), (2, 0.0), (3, 0.0), (2, 0.0002)):
        with test.subTest(budget=budget, extend=extend):
            wp.launch(
                kernel,
                dim=1,
                inputs=[meshes[0].id, meshes[1].id, rotation, position, budget, extend],
                outputs=[hit, points, penetration],
                device=device,
            )
            test.assertEqual(int(hit.numpy()[0]), 1)
            witnesses = points.numpy().astype(np.float64)
            test.assertTrue(np.all(np.isfinite(witnesses)))
            local_b = np.asarray(wp.quat_rotate_inv(rotation, wp.vec3(witnesses[1]) - position))
            for side, point in enumerate((witnesses[0], local_b)):
                test.assertLessEqual(abs(_surface_error(vertices[side], indices[side], point)), 1.0e-5)
            normal = witnesses[2]
            test.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, delta=1.0e-6)
            # Witnesses lie on both supporting planes of the reported normal,
            # so the depth is the exact overlap along it. Without distance
            # witnesses, this is the support plane along the refinement ray.
            test.assertAlmostEqual(
                float(np.dot(witnesses[0], normal)), float(np.max(vertices[0] @ normal)), delta=1.0e-5
            )
            test.assertAlmostEqual(float(np.dot(witnesses[1], normal)), float(np.min(world_b @ normal)), delta=1.0e-5)
            depth = float(penetration.numpy()[0])
            test.assertTrue(np.isfinite(depth))
            test.assertGreaterEqual(depth, 0.05254986867157328 - 1.0e-5)
            test.assertAlmostEqual(float(np.dot(witnesses[0] - witnesses[1], normal)), depth, delta=1.0e-6)
            if distance_query is None:
                # Certified distance witnesses also survive the writer's
                # midpoint and normal/depth reconstruction.
                center = 0.5 * (witnesses[0] + witnesses[1])
                np.testing.assert_allclose(center + 0.5 * depth * normal, witnesses[0], atol=1.0e-6)
                np.testing.assert_allclose(center - 0.5 * depth * normal, witnesses[1], atol=1.0e-6)


def test_convex_contact_failed_distance_query(test, device):
    """Fall back to the refinement ray's support plane when the raycast encounters overlap."""
    test_convex_contact_iteration_limit(test, device, distance_query=_failed_distance_query)


def test_convex_contact_unconverged_distance_query(test, device):
    """Reject interior witnesses even when the distance query reports separation."""
    test_convex_contact_iteration_limit(test, device, distance_query=_unconverged_distance_query)


def test_convex_contact_witness_margin(test, device):
    """Keep witnesses on physical hull surfaces after removing contact margins."""
    test_convex_contact_witness(test, device, margin=0.0001)


def test_convex_contact_split_witness(test, device):
    """Refine penetrating portals through captured split queues and reuse their slots."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_convex_contact_witness(test, device, capture=True)
        test_convex_contact_witness(test, device, swap=True, capture=True)


add_function_test(
    TestConvexContactWitness, "test_convex_contact_witness", test_convex_contact_witness, devices=get_test_devices()
)

for function in (
    test_convex_contact_iteration_limit,
    test_convex_contact_failed_distance_query,
    test_convex_contact_unconverged_distance_query,
    test_convex_contact_witness_margin,
    test_convex_contact_split_witness,
):
    add_function_test(TestConvexContactWitness, function.__name__, function, devices=get_test_devices())


@wp.kernel
def _resting_box_kernel(
    half_a: wp.vec3,
    half_b: wp.vec3,
    rotation: wp.quat,
    position: wp.vec3,
    normal: wp.array[wp.vec3],
    depth: wp.array[float],
):
    shape_a = GenericShapeData()
    shape_a.shape_type = int(newton.GeoType.BOX)
    shape_a.scale = half_a
    shape_a.shape_index = -1
    shape_b = shape_a
    shape_b.scale = half_b
    _collision, _pa, _pb, n, penetration = wp.static(create_solve_mpr(support_map).core)(
        shape_a, shape_b, rotation, position, 1.0e-4, SupportMapDataProvider()
    )
    normal[0] = n
    depth[0] = penetration - 1.0e-4


def test_convex_contact_resting_box(test, device):
    """Keep a centered face contact on its face instead of a retry's distant side face."""
    # Pose captured from a box resting on a long slab in the friction-ramp test.
    normal = wp.zeros(1, dtype=wp.vec3, device=device)
    depth = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        _resting_box_kernel,
        dim=1,
        inputs=[
            wp.vec3(0.5, 2.5, 0.05),
            wp.vec3(0.2, 0.2, 0.05),
            wp.quat(3.069639205932617e-06, 5.119977219969485e-10, -4.247564105996915e-12, 1.0),
            wp.vec3(0.0, -0.00018496252596378326, 0.09996524453163147),
        ],
        outputs=[normal, depth],
        device=device,
    )
    np.testing.assert_allclose(normal.numpy()[0], (0.0, 0.0, 1.0), atol=1.0e-4)
    test.assertAlmostEqual(float(depth.numpy()[0]), 0.1 - 0.09996524453163147, delta=1.0e-5)


add_function_test(
    TestConvexContactWitness,
    "test_convex_contact_resting_box",
    test_convex_contact_resting_box,
    devices=get_test_devices(),
)


def _box_minimum_depth(half, rotation, position):
    """Return the FP64 separating-axis depth of two equal boxes, B posed in A's frame."""
    axes_b = np.array([wp.quat_rotate(rotation, wp.vec3(*axis)) for axis in np.eye(3)], dtype=np.float64)
    axes = [*np.eye(3), *axes_b, *(np.cross(a, b) for a in np.eye(3) for b in axes_b)]
    depth = np.inf
    for axis in axes:
        length = np.linalg.norm(axis)
        if length > 1.0e-12:
            unit = axis / length
            radius = half * (np.sum(np.abs(unit)) + np.sum(np.abs(axes_b @ unit)))
            depth = min(depth, radius - abs(float(np.dot(unit, position))))
    return depth


def test_convex_contact_shallow_cubes(test, device, *, split=False):
    """Report no contact deeper than the minimum for shallow cube contacts that need refinement."""
    # Relative poses captured from a collapsed cube pile where refinement once
    # reported a far Minkowski corner, or a normal tilted by the ray shift, and
    # launched cubes. The portal misses the minimum certificate by its
    # convergence tolerance, so refinement runs.
    poses = (
        (
            wp.quat(-0.0011170628713443875, 1.7762567949830554e-05, -0.004657822661101818, 0.9999886751174927),
            wp.vec3(0.4209620952606201, -6.29425048828125e-05, 0.8006205558776855),
        ),
        (
            wp.quat(-0.0001483643427491188, -0.006294019520282745, 1.2986361980438232e-05, 0.9999802112579346),
            wp.vec3(0.8049442768096924, -6.67572021484375e-06, 0.005084991455078125),
        ),
    )
    half = 0.4
    for case, (rotation, position) in enumerate(poses):
        with test.subTest(case=case):
            builder = newton.ModelBuilder()
            builder.add_shape_box(-1, hx=half, hy=half, hz=half)
            body = builder.add_body(xform=wp.transform(position, rotation))
            builder.add_shape_box(body, hx=half, hy=half, hz=half)
            model = builder.finalize(device)
            pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
            test.assertEqual(pipeline.narrow_phase.split_gjk_mpr, split)
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            pose = wp.transform(*state.body_q.numpy()[body])
            depths = []
            for contact in range(count):
                points = [
                    contacts.rigid_contact_point0.numpy()[contact],
                    contacts.rigid_contact_point1.numpy()[contact],
                ]
                if int(contacts.rigid_contact_shape0.numpy()[contact]) == 1:
                    points.reverse()
                points[1] = np.asarray(wp.transform_point(pose, wp.vec3(points[1])))
                sign = 1.0 if int(contacts.rigid_contact_shape0.numpy()[contact]) == 0 else -1.0
                normal = sign * contacts.rigid_contact_normal.numpy()[contact]
                depths.append(-float(np.dot(points[1] - points[0], normal)))
            # The deepest contact must match the minimum penetration depth.
            test.assertAlmostEqual(max(depths), _box_minimum_depth(half, rotation, position), delta=1.0e-5)


def test_convex_contact_shallow_cubes_split(test, device):
    """Report no contact deeper than the minimum through the split CUDA collision kernels."""
    if not device.is_cuda:
        test.skipTest("Split collision kernels run only on CUDA")
    with patch("newton._src.sim.collide._SPLIT_GJK_MPR_LEAN_PAIR_COUNT_THRESHOLD", 0):
        test_convex_contact_shallow_cubes(test, device, split=True)


for function in (test_convex_contact_shallow_cubes, test_convex_contact_shallow_cubes_split):
    add_function_test(TestConvexContactWitness, function.__name__, function, devices=get_test_devices())


def test_convex_contact_witness_swapped(test, device):
    """Preserve surface membership when the two convex hulls are inserted in reverse order."""
    test_convex_contact_witness(test, device, swap=True)
    # Margins remove the inflation, so the raycast must certify its first
    # candidate even when it ties the fallback ray's depth.
    test_convex_contact_witness(test, device, swap=True, margin=0.0001)


add_function_test(
    TestConvexContactWitness,
    "test_convex_contact_witness_swapped",
    test_convex_contact_witness_swapped,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
