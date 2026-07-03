# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for SensorLidar."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorLidar
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestSensorLidar(unittest.TestCase):
    """Test SensorLidar validation and configuration."""

    def _make_simple_model(self):
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body, label="lidar")
        builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize()
        return model, site

    def test_sensor_creation_defaults(self):
        """Test default scan pattern: 36 azimuth x 4 elevation = 144 rays."""
        model, site = self._make_simple_model()
        sensor = SensorLidar(model, sites=[site])

        self.assertEqual(sensor.n_sensors, 1)
        self.assertEqual(sensor.n_rays, 144)
        self.assertEqual(sensor.distances.shape, (1, 144))
        self.assertEqual(sensor.ray_directions.shape, (144,))
        np.testing.assert_allclose(np.linalg.norm(sensor.ray_directions.numpy(), axis=1), 1.0, atol=1e-6)

    def test_sensor_string_pattern(self):
        """Test SensorLidar accepts a string pattern for sites."""
        model, _site = self._make_simple_model()
        sensor = SensorLidar(model, sites="lidar*")
        self.assertEqual(sensor.n_sensors, 1)

    def test_sensor_validation_empty_sites(self):
        """Test error when sites is empty."""
        model, _site = self._make_simple_model()
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[])

    def test_sensor_validation_no_match(self):
        """Test error when no labels match."""
        model, _site = self._make_simple_model()
        with self.assertRaises(ValueError):
            SensorLidar(model, sites="nonexistent_*")

    def test_sensor_validation_not_a_site(self):
        """Test error when index is not a site."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        shape = builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[shape])

    def test_sensor_validation_scan_pattern(self):
        """Test error on invalid scan pattern or range parameters."""
        model, site = self._make_simple_model()

        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], azimuth_count=0)
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], elevation_count=0)
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], azimuth_min=1.0, azimuth_max=0.0)
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], elevation_min=1.0, elevation_max=0.0)
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], min_range=-1.0)
        with self.assertRaises(ValueError):
            SensorLidar(model, sites=[site], max_range=0.5, min_range=1.0)


def _add_lidar_body(builder: newton.ModelBuilder, label: str, site_xform: wp.transform | None = None) -> int:
    """Add a body with a lidar site and return the site index."""
    body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
    return builder.add_site(body, xform=site_xform, label=label)


def _cardinal_lidar(model, sites, **kwargs) -> SensorLidar:
    """Lidar with a single ring of four rays: -X, -Y, +X, +Y (in site frame)."""
    return SensorLidar(
        model,
        sites,
        azimuth_count=4,
        elevation_count=1,
        elevation_min=0.0,
        elevation_max=0.0,
        **kwargs,
    )


def test_lidar_analytic_scene(test: TestSensorLidar, device: str):
    """Rays against a box and a sphere at known distances yield exact values."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    site = _add_lidar_body(builder, "lidar")
    # Box with its near face at x = 2.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(2.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
    )
    # Sphere with its near surface at x = -2.5.
    builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(-3.0, 0.0, 0.0), wp.quat_identity()), radius=0.5)
    builder.end_world()
    model = builder.finalize(device=device)

    sensor = _cardinal_lidar(model, [site])
    state = model.state()
    sensor.update(state)

    # Azimuth samples (endpoint-exclusive from -pi): -pi, -pi/2, 0, pi/2 -> -X, -Y, +X, +Y.
    np.testing.assert_allclose(sensor.distances.numpy()[0], [2.5, -1.0, 2.0, -1.0], atol=1e-5)


def test_lidar_elevation_rings(test: TestSensorLidar, device: str):
    """Elevation rings hit a ground plane at the expected slant distance."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    body = builder.add_body(
        mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity())
    )
    site = builder.add_site(body, label="lidar")
    builder.add_ground_plane()
    builder.end_world()
    model = builder.finalize(device=device)

    elevation = -math.pi / 6.0  # 30 degrees downward
    sensor = SensorLidar(
        model,
        [site],
        azimuth_count=8,
        elevation_count=2,
        elevation_min=elevation,
        elevation_max=0.0,
        max_range=10.0,
    )
    state = model.state()
    sensor.update(state)

    distances = sensor.distances.numpy().reshape(2, 8)
    # Ring 0 (down 30 degrees) from height 1: slant distance = 1 / sin(30 deg) = 2.
    np.testing.assert_allclose(distances[0], 1.0 / math.sin(-elevation), atol=1e-5)
    # Ring 1 (horizontal) never reaches the ground.
    np.testing.assert_allclose(distances[1], -1.0, atol=1e-5)


def test_lidar_batched_worlds(test: TestSensorLidar, device: str):
    """Sensors in different worlds see their own world's shapes plus global shapes."""
    builder = newton.ModelBuilder()

    # Global-world box visible from every world: near face at y = 4.5.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(0.0, 5.0, 0.0), wp.quat_identity()), hx=2.0, hy=0.5, hz=2.0
    )

    builder.begin_world()
    site0 = _add_lidar_body(builder, "lidar_w0")
    # World 0 wall: near face at x = 2.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(2.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
    )
    builder.end_world()

    builder.begin_world()
    site1 = _add_lidar_body(builder, "lidar_w1")
    # World 1 wall: near face at x = 4.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(4.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
    )
    # World 1 only: wall behind, near face at x = -3.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(-3.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
    )
    builder.end_world()

    model = builder.finalize(device=device)

    sensor = _cardinal_lidar(model, [site0, site1])
    test.assertEqual(sensor.distances.shape, (2, 4))

    state = model.state()
    sensor.update(state)
    distances = sensor.distances.numpy()

    # Ray order: -X, -Y, +X, +Y.
    # World 0: its own wall at 2 m; world 1's walls must not leak into -X.
    np.testing.assert_allclose(distances[0], [-1.0, -1.0, 2.0, 4.5], atol=1e-5)
    # World 1: own walls at 4 m (+X) and 3 m (-X), shared global box at 4.5 m (+Y).
    np.testing.assert_allclose(distances[1], [3.0, -1.0, 4.0, 4.5], atol=1e-5)


def test_lidar_miss_and_range_limits(test: TestSensorLidar, device: str):
    """Misses return -1; max_range converts far hits to misses; min_range skips near hits."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    site = _add_lidar_body(builder, "lidar")
    # Near box: front face at x = 0.5.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=1.0, hz=1.0
    )
    # Far wall: front face at x = 4.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(4.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=1.0, hz=1.0
    )
    builder.end_world()
    model = builder.finalize(device=device)
    state = model.state()

    with test.subTest("default_range_hits_near_box"):
        sensor = _cardinal_lidar(model, [site])
        sensor.update(state)
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, 0.5, -1.0], atol=1e-5)

    with test.subTest("min_range_skips_near_box"):
        sensor = _cardinal_lidar(model, [site], min_range=2.0)
        sensor.update(state)
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, 4.0, -1.0], atol=1e-5)

    with test.subTest("max_range_turns_far_hits_into_misses"):
        sensor = _cardinal_lidar(model, [site], min_range=2.0, max_range=3.0)
        sensor.update(state)
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, -1.0, -1.0], atol=1e-5)


def test_lidar_attached_frame_moves_with_body(test: TestSensorLidar, device: str):
    """The scan pattern follows the parent body's transform and the site's local offset."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    # Site is offset half a meter forward of the body origin.
    site = _add_lidar_body(builder, "lidar", site_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()))
    # Static wall: near face at x = 4.
    builder.add_shape_box(
        body=-1, xform=wp.transform(wp.vec3(4.5, 0.0, 0.0), wp.quat_identity()), hx=0.5, hy=2.0, hz=2.0
    )
    builder.end_world()
    model = builder.finalize(device=device)

    sensor = _cardinal_lidar(model, [site])
    state = model.state()

    with test.subTest("initial_pose"):
        sensor.update(state)
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, 3.5, -1.0], atol=1e-5)

    with test.subTest("translated_body"):
        state.body_q.assign([wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity())])
        sensor.update(state)
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, 1.5, -1.0], atol=1e-5)

    with test.subTest("rotated_body"):
        # Rotate the body 90 degrees about +Z: the site's local -Y ray now points along world +X.
        rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.pi / 2.0)
        state.body_q.assign([wp.transform(wp.vec3(2.0, 0.0, 0.0), rot)])
        sensor.update(state)
        # Site origin is now at (2, 0.5, 0); the local -Y ray travels 2 m to the wall.
        np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, 2.0, -1.0, -1.0], atol=1e-5)


def test_lidar_mesh_scene(test: TestSensorLidar, device: str):
    """Lidar rays hit triangle-mesh shapes through the shape BVH."""
    builder = newton.ModelBuilder()
    builder.begin_world()
    site = _add_lidar_body(builder, "lidar")

    # Quad wall in the plane x = 3 facing the sensor.
    vertices = np.array(
        [
            [3.0, -2.0, -2.0],
            [3.0, 2.0, -2.0],
            [3.0, 2.0, 2.0],
            [3.0, -2.0, 2.0],
        ],
        dtype=np.float32,
    )
    indices = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32).flatten()
    with wp.ScopedDevice(device):
        mesh = newton.Mesh(vertices, indices, compute_inertia=False)
    builder.add_shape_mesh(body=-1, mesh=mesh)
    builder.end_world()
    model = builder.finalize(device=device)

    sensor = _cardinal_lidar(model, [site])
    state = model.state()
    sensor.update(state)

    np.testing.assert_allclose(sensor.distances.numpy()[0], [-1.0, -1.0, 3.0, -1.0], atol=1e-5)


devices = get_test_devices()

add_function_test(TestSensorLidar, "test_lidar_analytic_scene", test_lidar_analytic_scene, devices=devices)
add_function_test(TestSensorLidar, "test_lidar_elevation_rings", test_lidar_elevation_rings, devices=devices)
add_function_test(TestSensorLidar, "test_lidar_batched_worlds", test_lidar_batched_worlds, devices=devices)
add_function_test(
    TestSensorLidar, "test_lidar_miss_and_range_limits", test_lidar_miss_and_range_limits, devices=devices
)
add_function_test(
    TestSensorLidar,
    "test_lidar_attached_frame_moves_with_body",
    test_lidar_attached_frame_moves_with_body,
    devices=devices,
)
add_function_test(TestSensorLidar, "test_lidar_mesh_scene", test_lidar_mesh_scene, devices=devices)


if __name__ == "__main__":
    unittest.main()
