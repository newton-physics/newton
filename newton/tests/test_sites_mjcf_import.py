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

"""Tests for site parsing from MJCF files."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, ShapeFlags


class TestMJCFSiteImport(unittest.TestCase):
    """Test parsing sites from MJCF XML."""

    def test_parse_simple_site(self):
        """Test parsing a simple site from MJCF."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link1" pos="0 0 0">
                    <site name="sensor_imu" pos="0.1 0 0" quat="1 0 0 0" type="sphere" size="0.02"/>
                    <geom name="visual" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Find site
        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        shape_types = model.shape_type.numpy()

        site_idx = None
        for i in range(model.shape_count):
            if "sensor_imu" in shape_keys[i] and (shape_flags[i] & ShapeFlags.SITE):
                site_idx = i
                break

        self.assertIsNotNone(site_idx, "Site not found")
        self.assertEqual(shape_types[site_idx], GeoType.SPHERE)

        # Check transform (approximate due to floating point)
        xform = wp.transform(*model.shape_transform.numpy()[site_idx])
        pos = wp.transform_get_translation(xform)
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0.1, 0, 0], atol=1e-5)

    def test_parse_multiple_sites(self):
        """Test parsing multiple sites from MJCF."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="torso">
                    <site name="site1" pos="0.1 0 0"/>
                    <site name="site2" pos="0 0.1 0"/>
                    <site name="site3" pos="0 0 0.1"/>
                    <geom name="torso_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        site_names = ["site1", "site2", "site3"]
        found_sites = []

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key

        for i in range(model.shape_count):
            if (shape_flags[i] & ShapeFlags.SITE) and shape_keys[i] in site_names:
                found_sites.append(shape_keys[i])

        self.assertEqual(len(found_sites), 3)
        self.assertEqual(set(found_sites), set(site_names))

    def test_parse_site_types(self):
        """Test parsing sites with different types."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link">
                    <site name="sphere_site" type="sphere" size="0.01"/>
                    <site name="box_site" type="box" size="0.01 0.02 0.03"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        expected_types = {
            "sphere_site": GeoType.SPHERE,
            "box_site": GeoType.BOX,
        }

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        shape_types = model.shape_type.numpy()

        for i in range(model.shape_count):
            key = shape_keys[i]
            if key in expected_types and (shape_flags[i] & ShapeFlags.SITE):
                self.assertEqual(shape_types[i], expected_types[key], f"Type mismatch for {key}")

    def test_parse_site_no_type(self):
        """Test site without explicit type (should default to sphere)."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link">
                    <site name="default_site" pos="0 0 0"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        shape_types = model.shape_type.numpy()

        for i in range(model.shape_count):
            if shape_keys[i] == "default_site" and (shape_flags[i] & ShapeFlags.SITE):
                self.assertEqual(shape_types[i], GeoType.SPHERE)
                return

        self.fail("Site not found")

    def test_parse_site_no_name(self):
        """Test site without name (should get auto-generated name)."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link">
                    <site pos="0 0 0"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Should have at least one site
        shape_flags = model.shape_flags.numpy()
        site_count = sum(1 for i in range(model.shape_count) if shape_flags[i] & ShapeFlags.SITE)
        self.assertGreaterEqual(site_count, 1)

    def test_parse_worldbody_sites(self):
        """Test parsing sites in worldbody (not in bodies)."""
        mjcf = """
        <mujoco>
            <worldbody>
                <site name="world_site" pos="1 2 3" type="sphere" size="0.05"/>
                <body name="link" pos="0 0 1">
                    <geom name="link_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Find world site
        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        shape_bodies = model.shape_body.numpy()

        world_site_idx = None
        for i in range(model.shape_count):
            if "world_site" in shape_keys[i] and (shape_flags[i] & ShapeFlags.SITE):
                world_site_idx = i
                break

        self.assertIsNotNone(world_site_idx, "World site not found")
        # World sites should be attached to body -1
        self.assertEqual(shape_bodies[world_site_idx], -1)

    def test_parse_site_orientations(self):
        """Test parsing sites with different orientation specifications."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link">
                    <site name="quat_site" quat="1 0 0 0"/>
                    <site name="euler_site" euler="0 0 90"/>
                    <site name="axisangle_site" axisangle="0 0 1 90"/>
                </body>
            </worldbody>
        </mujoco>
        """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Verify sites were created and check their orientations
        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_key
        shape_transforms = model.shape_transform.numpy()

        # Find each site and check orientation
        found_sites = {}
        for i in range(model.shape_count):
            if shape_flags[i] & ShapeFlags.SITE:
                key = shape_keys[i]
                if key in ["quat_site", "euler_site", "axisangle_site"]:
                    xform = wp.transform(*shape_transforms[i])
                    quat = wp.transform_get_rotation(xform)
                    found_sites[key] = [quat.w, quat.x, quat.y, quat.z]

        # Check all three sites were found
        self.assertEqual(len(found_sites), 3, "Expected to find 3 sites with orientations")

        # quat="1 0 0 0" should be identity quaternion
        np.testing.assert_allclose(
            found_sites["quat_site"], [1, 0, 0, 0], atol=1e-5, err_msg="Identity quaternion mismatch"
        )

        # euler="0 0 90" is 90 degrees around Z axis
        # Quaternion for 90° around Z: [cos(45°), 0, 0, sin(45°)] = [0.7071, 0, 0, 0.7071]
        np.testing.assert_allclose(
            found_sites["euler_site"], [0.7071068, 0, 0, 0.7071068], atol=1e-5, err_msg="Euler 90° Z rotation mismatch"
        )

        # axisangle="0 0 1 90" is also 90 degrees around Z axis (angle in degrees)
        np.testing.assert_allclose(
            found_sites["axisangle_site"],
            [0.7071068, 0, 0, 0.7071068],
            atol=1e-5,
            err_msg="Axis-angle 90° Z rotation mismatch",
        )


if __name__ == "__main__":
    unittest.main()
