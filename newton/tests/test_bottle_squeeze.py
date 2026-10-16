# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check bottle boundaries, robot motion, and the fluid shown by the example."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.utils
from newton.examples.multiphysics._bottle import BottleFluid, BottleSurface, _density_multipliers, _shell_contacts
from newton.examples.multiphysics.example_franka_bottle_squeeze import (
    _BOTTLE_CENTER,
    _BOTTLE_HEIGHT,
    _CYCLE_DURATION,
    _add_robot,
    _bottle_mesh,
    _bottle_radius,
    _pet_bending_properties,
    _robot_motion,
    _squeeze_keyframes,
    _water_samples,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_gripper_plates_follow_fingers(test, device):
    """Match the visible and physical plate gaps as the fingers close."""
    asset = newton.utils.download_asset("franka_emika_panda")
    for scale in (1.0, 100.0):
        builder = newton.ModelBuilder()
        _add_robot(builder, asset, scale=scale)
        plates = [i for i, label in enumerate(builder.shape_label) if label.endswith("_squeeze_plate")]
        test.assertEqual(len(plates), 2)
        model = builder.finalize(device=device)
        state = model.state()
        finger_coords = [builder.joint_q_start[builder.joint_label.index(f"fr3/fr3_finger_joint{i}")] for i in (1, 2)]
        for opening, expected_gap in ((0.04, 0.080), (0.018, 0.036)):
            q = model.joint_q.numpy()
            q[finger_coords] = opening * scale
            newton.eval_fk(model, wp.array(q, device=device), model.joint_qd, state)
            poses = state.body_q.numpy()
            faces = []
            normals = []
            for shape in plates:
                body = builder.shape_body[shape]
                test.assertIn("finger", builder.body_label[body])
                test.assertTrue(builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_PARTICLES))
                test.assertTrue(builder.shape_flags[shape] & int(newton.ShapeFlags.VISIBLE))
                frame = wp.transform(wp.vec3(*poses[body, :3]), wp.quat(*poses[body, 3:]))
                local = wp.transform_point(builder.shape_transform[shape], wp.vec3(0.0, -0.0015 * scale, 0.0))
                faces.append(np.asarray(wp.transform_point(frame, local)))
                normals.append(np.asarray(wp.transform_vector(frame, wp.vec3(0.0, -1.0, 0.0))))
            test.assertAlmostEqual(float(np.linalg.norm(faces[0] - faces[1])) / scale, expected_gap, places=5)
            test.assertAlmostEqual(float(np.dot(normals[0], normals[1])), -1.0, places=5)


def test_water_xpbd_pressure_history(test, device):
    """Converge compliant pressure and release it without allowing tension."""
    points = wp.array([[-16.0, -16.0, -10.0], [16.0, -16.0, -10.0], [0.0, 16.0, -10.0]], dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=points, indices=wp.array([0, 1, 2], dtype=int, device=device))
    positions = wp.zeros(1, dtype=wp.vec3, device=device)
    grid = wp.HashGrid(8, 8, 8, device=device)
    grid.build(positions, 1.0)
    multipliers = wp.zeros(1, dtype=float, device=device)
    deltas = wp.empty_like(multipliers)
    density = wp.empty_like(multipliers)
    gradients = wp.empty(1, dtype=wp.vec3, device=device)
    # An isolated over-dense sample has no positional gradient. Finite
    # compliance must still admit a well-defined pressure equilibrium.
    for _ in range(5):
        wp.launch(
            _density_multipliers,
            1,
            inputs=[grid.id, positions, mesh.id, 1.0, 1.0, 0.5, multipliers, deltas, density, gradients],
            device=device,
        )
    test.assertLess(float(multipliers.numpy()[0]), 0.0)
    np.testing.assert_allclose(density.numpy() - 1.0 + 0.5 * multipliers.numpy(), 0.0, atol=1.0e-6)
    test.assertLess(float(abs(deltas.numpy()[0])), 1.0e-6)
    for _ in range(5):
        wp.launch(
            _density_multipliers,
            1,
            inputs=[grid.id, positions, mesh.id, 1.0, 0.01, 0.5, multipliers, deltas, density, gradients],
            device=device,
        )
    np.testing.assert_allclose(multipliers.numpy(), [0.0], atol=1.0e-6)
    test.assertGreater(float(deltas.numpy()[0]), 0.0)


def test_water_xpbd_resting_bottle(test, device):
    """Settle incompressible samples without spurious wall escape."""
    vertices, triangles = _bottle_mesh(32, 28)
    shell = wp.array(100.0 * (vertices + _BOTTLE_CENTER), dtype=wp.vec3, device=device)
    positions, spacing = _water_samples(4096, seed=42)
    fluid = BottleFluid(
        positions,
        spacing,
        shell,
        triangles,
        floor=76.0,
        mouth_height=100.0 * (_BOTTLE_CENTER[2] + _BOTTLE_HEIGHT),
        iterations=8,
        device=device,
    )
    test.assertEqual(fluid.compliance, 0.0)
    for _ in range(120):
        fluid.step(shell, 1.0 / 480.0)
    settled = fluid.positions.numpy() * 0.01
    heights = settled[:, 2] - _BOTTLE_CENTER[2]
    radii = np.linalg.norm(settled[:, :2] - _BOTTLE_CENTER[:2], axis=1)
    test.assertTrue(np.isfinite(settled).all())
    test.assertLessEqual(float(np.max(radii - _bottle_radius(heights))), 0.001)
    test.assertLess(float(heights.max()), _BOTTLE_HEIGHT)
    np.testing.assert_array_equal(fluid.inside_bottle.numpy(), np.ones(len(positions), dtype=np.int32))


def test_water_xpbd_substep_reset(test, device):
    """Discard previous substep multipliers while preserving fluid state."""
    points = wp.array([[-16.0, -16.0, -10.0], [16.0, -16.0, -10.0], [0.0, 16.0, -10.0]], dtype=wp.vec3, device=device)
    positions = np.stack(np.meshgrid(*([np.arange(4) * 0.35] * 3), indexing="ij"), axis=-1).reshape(-1, 3)
    options = {"floor": -20.0, "mouth_height": 20.0, "iterations": 4, "device": device, "bulk_modulus": 100.0}
    fresh = BottleFluid(positions, 1.0, points, [0, 1, 2], **options)
    reused = BottleFluid(positions, 1.0, points, [0, 1, 2], **options)
    reused.multipliers.fill_(-100.0)
    fresh.step(points, 0.01)
    reused.step(points, 0.01)
    np.testing.assert_allclose(reused.positions.numpy(), fresh.positions.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(reused.velocities.numpy(), fresh.velocities.numpy(), atol=1.0e-6)
    np.testing.assert_allclose(reused.multipliers.numpy(), fresh.multipliers.numpy(), atol=1.0e-6)
    test.assertTrue(np.any(fresh.multipliers.numpy() < 0.0))


def test_robot_repeated_squeezes(test, device):
    """Vary grip height and close once per angle with fingers open during repositioning."""
    grips = np.array([[0.0, 11.2, 4.0, 4.0], [1.0, 13.0, 4.0, 4.0], [-1.0, 9.4, 4.0, 4.0]], dtype=np.float32)
    retreats = grips.copy()
    retreats[:, 1] = 27.0
    times, poses = _squeeze_keyframes(grips, retreats, (2, 3), 1.2)
    clock = wp.zeros(1, dtype=float, device=device)
    times_wp = wp.array(times, dtype=float, device=device)
    poses_wp = wp.array(poses, dtype=float, device=device)
    q, qd = wp.empty(4, dtype=float, device=device), wp.empty(4, dtype=float, device=device)
    dt = 1.0e-4

    def sample(time):
        clock.fill_(time - dt)
        wp.launch(_robot_motion, 4, inputs=[clock, dt, times_wp, poses_wp, q, qd], device=device)
        return q.numpy(), qd.numpy()

    for cycle, grip in enumerate(grips):
        base = cycle * _CYCLE_DURATION
        open_pose, _ = sample(base + 0.3)
        np.testing.assert_allclose(open_pose, grip, atol=1.0e-5)
        closed_pose, velocity = sample(base + 2.0)
        np.testing.assert_allclose(closed_pose, [*grip[:2], 1.2, 1.2], atol=1.0e-5)
        np.testing.assert_allclose(velocity, 0.0, atol=1.0e-5)
        for phase in (3.5, 4.0, 4.8, 5.6):
            reposition, _ = sample(base + phase)
            np.testing.assert_allclose(reposition[2:], 4.0, atol=1.0e-5)
    final_pose, velocity = sample(3.0 * _CYCLE_DURATION + 1.0)
    np.testing.assert_allclose(final_pose, retreats[-1], atol=1.0e-5)
    np.testing.assert_allclose(velocity, 0.0, atol=1.0e-5)
    # A smooth trajectory has no position or velocity jump at a waypoint.
    for time in times[1:-1]:
        before, velocity_before = sample(float(time) - dt)
        after, velocity_after = sample(float(time) + dt)
        np.testing.assert_allclose(before, after, atol=1.0e-5)
        test.assertLess(float(np.max(np.abs(velocity_before))), 0.02)
        test.assertLess(float(np.max(np.abs(velocity_after))), 0.02)


def test_water_shell_contact(test, device):
    """Block slow and fast wall crossings on both sides, but leave openings clear."""
    points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=int, device=device)
    mesh = wp.Mesh(points=points, indices=indices)
    # Two wall sides, crossings beyond the proximity search, an open triangle
    # boundary, and proximity-only contact without crossing the wall.
    start = [[0.2, 0.2, z] for z in (0.3, -0.3, 0.3, -0.3)]
    start += [[0.8, 0.8, 0.3], [0.8, 0.8, -0.3], [0.2, 0.2, 0.3], [0.2, 0.2, -0.3]]
    target = np.array(
        [[*p[:2], z] for p, z in zip(start, (-0.05, 0.05, -2.0, 2.0, -0.05, 0.05, 0.02, -0.02), strict=True)],
        dtype=np.float32,
    )
    positions = wp.array(target, dtype=wp.vec3, device=device)
    wp.launch(
        _shell_contacts,
        len(start),
        inputs=[
            mesh.id,
            indices,
            points,
            wp.array(start, dtype=wp.vec3, device=device),
            positions,
            wp.zeros_like(positions),
            wp.zeros(len(start), dtype=int, device=device),
            10.0,
            0.1,
            1.0,
            -10.0,
        ],
        device=device,
    )
    expected = target.copy()
    expected[[0, 1, 2, 3, 6, 7], 2] = [0.1, -0.1, 0.1, -0.1, 0.1, -0.1]
    np.testing.assert_allclose(positions.numpy(), expected, atol=1.0e-6)


def test_water_boundary_density(test, device):
    """Complete missing neighbor density with the integrated wall half-space."""
    points = wp.array([[-16.0, -16.0, 0.0], [16.0, -16.0, 0.0], [0.0, 16.0, 0.0]], dtype=wp.vec3, device=device)
    mesh = wp.Mesh(points=points, indices=wp.array([0, 1, 2], dtype=int, device=device))
    distances = np.array([0.05, 0.25, 0.9], dtype=np.float32)
    positions = wp.array(np.column_stack(([-4.0, 0.0, 4.0], np.zeros(3), distances)), dtype=wp.vec3, device=device)
    grid = wp.HashGrid(16, 16, 16, device=device)
    grid.build(positions, 1.0)
    multipliers = wp.zeros(3, dtype=float, device=device)
    deltas = wp.empty_like(multipliers)
    densities = wp.empty_like(multipliers)
    gradients = wp.empty(3, dtype=wp.vec3, device=device)
    volume = 0.001
    wp.launch(
        _density_multipliers,
        3,
        inputs=[grid.id, positions, mesh.id, 1.0, volume, 0.0, multipliers, deltas, densities, gradients],
        device=device,
    )
    # Independently integrate spherical caps of Poly6 with Gaussian quadrature.
    nodes, weights = np.polynomial.legendre.leggauss(12)
    expected = []
    for distance in distances:
        z = 0.5 * (1.0 - distance) * nodes + 0.5 * (1.0 + distance)
        cap = 315.0 / 256.0 * (1.0 - z * z) ** 4
        expected.append(0.5 * (1.0 - distance) * np.dot(weights, cap) + volume * 315.0 / (64.0 * np.pi))
    np.testing.assert_allclose(densities.numpy(), expected, atol=2.0e-7)
    np.testing.assert_allclose(gradients.numpy()[:, :2], 0.0, atol=1.0e-6)
    test.assertTrue(np.all(gradients.numpy()[:, 2] < 0.0))
    # An underfilled free surface must never attract particles through tension.
    np.testing.assert_array_equal(multipliers.numpy(), np.zeros(3))


@wp.kernel
def _surface_signs(mesh: wp.uint64, vertices: wp.array[wp.vec3], signs: wp.array[float]):
    i = wp.tid()
    # Independent ray-based classification checks the rendered mesh itself.
    query = wp.mesh_query_point(mesh, vertices[i], 1.0)
    signs[i] = query.sign


def _make_water_surface(positions, spacing, device, *, deformed=False, neck_lift=0.0):
    vertices, triangles = _bottle_mesh(32, 28)
    if deformed:
        zs = np.unique(vertices[:, 2])
        factors = np.interp(zs, [0.0, 0.086, 0.11, 0.116, 0.127, 0.14, 0.21], [1.0, 1.0, 0.35, 0.7, 0.4, 1.0, 1.0])
        vertices[:, 0] *= np.interp(vertices[:, 2], zs, factors)
        positions = positions.copy()
        positions[:, 0] = _BOTTLE_CENTER[0] + (positions[:, 0] - _BOTTLE_CENTER[0]) * np.interp(
            positions[:, 2] - _BOTTLE_CENTER[2], zs, factors
        )
    if neck_lift:
        vertices[:, 2] += neck_lift * np.clip((vertices[:, 2] - 0.16) / 0.05, 0.0, 1.0)
        positions = positions.copy()
        positions[:, 2] += neck_lift * np.clip((positions[:, 2] - _BOTTLE_CENTER[2] - 0.16) / 0.05, 0.0, 1.0)
    shell = wp.array(vertices + _BOTTLE_CENTER, dtype=wp.vec3, device=device)
    surface = BottleSurface(
        shell,
        triangles,
        segments=32,
        spacing=spacing,
        voxel_size=0.0015,
        margin=0.0005,
        floor=0.76,
        max_grid_cells=1_000_000,
        device=device,
    )
    particles = wp.array(positions, dtype=wp.vec3, device=device)
    radii = wp.full(len(positions), (3.0 * spacing**3 / (4.0 * np.pi)) ** (1.0 / 3.0), device=device)
    vertices, indices, normals = surface.extract(particles, radii)
    return vertices.numpy(), indices.numpy(), normals.numpy(), surface


def test_water_surface_containment(test, device):
    """Keep reconstructed liquid within the resting wall, sharp folds, and a raised neck."""
    positions, spacing = _water_samples(4096, seed=42)
    xy = np.arange(-0.007, 0.008, 0.0025)
    z = np.arange(0.190, 0.201, 0.0025)
    neck_water = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3) + _BOTTLE_CENTER
    for deformed, neck_lift in ((False, 0.0), (True, 0.0), (False, 0.03)):
        with test.subTest(deformed=deformed, neck_lift=neck_lift):
            water = np.concatenate((0.01 * positions, neck_water)) if neck_lift else 0.01 * positions
            vertices, indices, normals, surface = _make_water_surface(
                water, 0.01 * spacing, device, deformed=deformed, neck_lift=neck_lift
            )
            signs = wp.empty(len(vertices), dtype=float, device=device)
            wp.launch(
                _surface_signs,
                len(vertices),
                inputs=[surface.mesh.id, wp.array(vertices, dtype=wp.vec3, device=device), signs],
                device=device,
            )
            test.assertGreater(len(indices), 0)
            test.assertTrue(np.all(signs.numpy() < 0.0))
            test.assertTrue(np.isfinite(normals).all())
            test.assertGreaterEqual(float(vertices[:, 2].min()), 0.76 - 1.0e-5)
            if neck_lift:
                test.assertGreater(float(vertices[:, 2].max()), _BOTTLE_CENTER[2] + _BOTTLE_HEIGHT)
            elif not deformed:
                heights = vertices[:, 2] - _BOTTLE_CENTER[2]
                body = (heights > 0.025) & (heights < 0.15)
                gap = _bottle_radius(heights[body]) - np.linalg.norm(vertices[body, :2] - _BOTTLE_CENTER[:2], axis=1)
                test.assertGreater(np.count_nonzero(body), 0)
                test.assertGreaterEqual(float(gap.min()), 0.0002)


def test_water_surface_open_mouth_and_spill(test, device):
    """Preserve a jet above the open neck and real exterior droplets."""
    positions, spacing = _water_samples(4096, seed=42)
    spacing *= 0.01
    xy = np.arange(-0.0045, 0.005, 0.003)
    z = np.arange(0.194, 0.223, 0.003)
    jet = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3) + _BOTTLE_CENTER
    cloud = np.stack(np.meshgrid(xy, xy, xy, indexing="ij"), axis=-1).reshape(-1, 3)
    cloud += _BOTTLE_CENTER + np.array([0.049, 0.0, 0.025])
    particles = np.concatenate((0.01 * positions, jet, cloud))
    vertices, _, _, _ = _make_water_surface(particles, spacing, device)
    test.assertGreater(float(vertices[:, 2].max()), _BOTTLE_CENTER[2] + _BOTTLE_HEIGHT + 0.005)
    exterior = (vertices[:, 0] > _BOTTLE_CENTER[0] + 0.041) & (vertices[:, 2] < _BOTTLE_CENTER[2] + _BOTTLE_HEIGHT)
    test.assertGreater(np.count_nonzero(exterior), 10)


class TestBottleSqueeze(unittest.TestCase):
    def test_bottle_open_neck(self):
        """Make the neck the only boundary and orient the bottom outwards."""
        vertices, triangles = _bottle_mesh(32, 28)
        edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
        boundary_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        self.assertEqual(np.count_nonzero(counts == 1), 32)
        self.assertTrue(np.all(counts <= 2))
        self.assertTrue(np.all(vertices[boundary_edges[counts == 1], 2] == vertices[:, 2].max()))
        bottom = triangles[np.all(vertices[triangles, 2] == 0.0, axis=1)]
        normals = np.cross(
            vertices[bottom[:, 1]] - vertices[bottom[:, 0]], vertices[bottom[:, 2]] - vertices[bottom[:, 0]]
        )
        self.assertTrue(np.all(normals[:, 2] < 0.0))

    def test_pet_units_and_refinement(self):
        """Scale hinge stiffness and yield angles with the rest dual width."""
        values = []
        for width in (1.0, 0.5):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for p in [[0.0, width, 0.0], [0.0, -width, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]:
                builder.add_particle(wp.vec3(p), wp.vec3(0.0), 1.0)
            builder.add_edge(0, 1, 2, 3)
            yields = _pet_bending_properties(builder)
            stiffness = builder.edge_bending_properties[0][0]
            values.append((stiffness, yields[0]))
        rigidity_cm = 3.0e9 * 0.0003**3 / (12.0 * (1.0 - 0.38**2)) * 100.0**2
        self.assertAlmostEqual(values[0][0], rigidity_cm / (2.0 / 3.0))
        self.assertAlmostEqual(values[0][1], 2.0 * 55.0e6 / (3.0e9 * 0.0003) / 100.0 * (2.0 / 3.0), places=6)
        np.testing.assert_allclose(values[1], [2.0 * values[0][0], 0.5 * values[0][1]])


for function in (
    test_gripper_plates_follow_fingers,
    test_water_xpbd_pressure_history,
    test_water_xpbd_substep_reset,
    test_robot_repeated_squeezes,
    test_water_shell_contact,
    test_water_boundary_density,
):
    add_function_test(TestBottleSqueeze, function.__name__, function, devices=get_cuda_test_devices())

for function in (
    test_water_xpbd_resting_bottle,
    test_water_surface_containment,
    test_water_surface_open_mouth_and_spill,
):
    add_function_test(TestBottleSqueeze, function.__name__, function, devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
