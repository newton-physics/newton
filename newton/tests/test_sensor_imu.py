# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for SensorIMU."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.articulation import eval_fk
from newton.sensors import SensorIMU


class TestSensorIMU(unittest.TestCase):
    """Test SensorIMU functionality."""

    def test_sensor_creation(self):
        """Test basic sensor creation."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body, label="imu_site")
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[site])

        self.assertEqual(sensor.n_sensors, 1)
        self.assertEqual(sensor.accelerometer.shape[0], 1)
        self.assertEqual(sensor.gyroscope.shape[0], 1)

    def test_sensor_multiple_sites(self):
        """Test sensor with multiple sites."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site1 = builder.add_site(body, label="site1")
        site2 = builder.add_site(body, label="site2")
        site3 = builder.add_site(body, label="site3")
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[site1, site2, site3])

        self.assertEqual(sensor.n_sensors, 3)
        self.assertEqual(sensor.accelerometer.shape[0], 3)
        self.assertEqual(sensor.gyroscope.shape[0], 3)

    def test_sensor_validation_empty_sites(self):
        """Test error when sites is empty."""
        builder = newton.ModelBuilder()
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorIMU(model, sites=[])

    def test_sensor_validation_invalid_site_index(self):
        """Test error when site index is out of bounds."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorIMU(model, sites=[9999])

    def test_sensor_validation_not_a_site(self):
        """Test error when index is not a site."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        shape = builder.add_shape_sphere(body, radius=0.1)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorIMU(model, sites=[shape])

    def test_sensor_update_without_body_qdd(self):
        """Test error when updating without body_qdd."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        model = builder.finalize()

        state = model.state()
        sensor = SensorIMU(model, sites=[site])

        with self.assertRaises(ValueError):
            sensor.update(state)

    def test_sensor_update_with_body_qdd(self):
        """Test sensor update with body_qdd allocated."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(2, 4, 6)), 4.0)
        site = builder.add_site(body, label="imu", xform=wp.transform(wp.vec3(0, 0, 0), rot))
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[site])
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        state.body_qdd.zero_()
        sensor.update(state)

        acc = sensor.accelerometer.numpy()
        gyro = sensor.gyroscope.numpy()
        self.assertEqual(acc.shape, (1, 3))
        self.assertEqual(gyro.shape, (1, 3))
        np.testing.assert_allclose(acc, [wp.quat_rotate_inv(rot, -wp.vec3(model.gravity.numpy()[0]))], atol=1e-8)
        np.testing.assert_allclose(gyro, [[0.0, 0.0, 0.0]], atol=1e-8)

    def test_sensor_static_body_gravity(self):
        """Test IMU on static body measures gravity."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body, label="imu")
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[site])
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        state.body_qdd.zero_()
        sensor.update(state)

        acc = sensor.accelerometer.numpy()[0]
        gyro = sensor.gyroscope.numpy()[0]
        gravity = model.gravity.numpy()[0]

        np.testing.assert_allclose(acc, -gravity, atol=1e-5)
        np.testing.assert_allclose(gyro, [0, 0, 0], atol=1e-5)

    def test_sensor_world_frame_site(self):
        """Test IMU on site attached to world frame (body=-1)."""
        builder = newton.ModelBuilder()
        world_site = builder.add_site(-1, label="world_imu")
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[world_site])
        state = model.state()

        state.body_qdd.zero_()
        sensor.update(state)

        acc = sensor.accelerometer.numpy()[0]
        gyro = sensor.gyroscope.numpy()[0]
        gravity = model.gravity.numpy()[0]

        np.testing.assert_allclose(acc, -gravity, atol=1e-5)
        np.testing.assert_allclose(gyro, [0, 0, 0], atol=1e-5)

    def test_sensor_rotated_site(self):
        """Test IMU with rotated site frame."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        rot_90_z = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)
        site = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), rot_90_z), label="imu")
        model = builder.finalize()

        sensor = SensorIMU(model, sites=[site])
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        state.body_qdd.zero_()
        sensor.update(state)

        acc = sensor.accelerometer.numpy()[0]

        gravity = model.gravity.numpy()[0]
        expected_acc = wp.quat_rotate_inv(rot_90_z, wp.vec3(-gravity[0], -gravity[1], -gravity[2]))
        np.testing.assert_allclose(acc, [expected_acc[0], expected_acc[1], expected_acc[2]], atol=1e-5)

    def test_sensor_string_pattern(self):
        """Test SensorIMU accepts a string pattern for sites."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, label="imu_site")
        model = builder.finalize()

        sensor = SensorIMU(model, sites="imu_site")
        self.assertEqual(sensor.n_sensors, 1)

    def test_sensor_wildcard_pattern(self):
        """Test SensorIMU with wildcard pattern."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, label="imu_a")
        builder.add_site(body, label="imu_b")
        builder.add_site(body, label="other")
        model = builder.finalize()

        sensor = SensorIMU(model, sites="imu_*")
        self.assertEqual(sensor.n_sensors, 2)

    def test_sensor_no_match_raises(self):
        """Test SensorIMU raises when no labels match."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, label="site")
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorIMU(model, sites="nonexistent_*")


if __name__ == "__main__":
    unittest.main()
