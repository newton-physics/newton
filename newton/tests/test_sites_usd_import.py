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

"""Tests for site parsing from USD files."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType, ShapeFlags


class TestUSDSiteImport(unittest.TestCase):
    """Test parsing sites from USD files."""

    def _create_usd_stage(self, usd_content: str):
        """Create a USD stage in memory from the given content.

        Uses ImportFromString() instead of programmatic stage construction to allow
        applying unregistered API schemas (like MjcSiteAPI) without requiring schema plugins.
        """
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString(usd_content)
        return stage

    def test_parse_simple_site(self):
        """Test parsing a simple site from USD."""
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "link1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)

        def Sphere "sensor_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.02
            double3 xformOp:translate = (0.1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Cube "visual" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }
}
"""
        stage = self._create_usd_stage(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        # Find site
        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_label
        shape_types = model.shape_type.numpy()

        site_idx = None
        for i in range(model.shape_count):
            if "sensor_site" in shape_keys[i] and (shape_flags[i] & ShapeFlags.SITE):
                site_idx = i
                break

        self.assertIsNotNone(site_idx, "Site not found")
        self.assertEqual(shape_types[site_idx], GeoType.SPHERE)

        # Check transform
        xform = wp.transform(*model.shape_transform.numpy()[site_idx])
        pos = wp.transform_get_translation(xform)
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0.1, 0, 0], atol=1e-5)

    def test_parse_multiple_sites(self):
        """Test parsing multiple sites from USD."""
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "torso" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)
        def Sphere "site1" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            double3 xformOp:translate = (0.1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Sphere "site2" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            double3 xformOp:translate = (0, 0.1, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Sphere "site3" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            double3 xformOp:translate = (0, 0, 0.1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }

        def Cube "torso_geom" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double size = 0.2
        }
    }
}
"""
        stage = self._create_usd_stage(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        site_names = ["site1", "site2", "site3"]
        found_sites = []

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_label

        for i in range(model.shape_count):
            if shape_flags[i] & ShapeFlags.SITE:
                key = shape_keys[i]
                # Check if any of the expected site names is in the key
                for name in site_names:
                    if name in key and key not in [s for s, _ in found_sites]:
                        found_sites.append((key, name))
                        break

        self.assertEqual(len(found_sites), 3)
        # Verify we found all expected site names
        found_names = {name for _, name in found_sites}
        self.assertEqual(found_names, set(site_names))

    def test_parse_site_types(self):
        """Test parsing sites with different types."""
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)

        def Sphere "sphere_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
        }

        def Cube "box_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double size = 0.02
        }

        def Capsule "capsule_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            double height = 0.05
        }

        def Cylinder "cylinder_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            double height = 0.05
        }
    }
}
"""
        stage = self._create_usd_stage(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        expected_types = {
            "sphere_site": GeoType.SPHERE,
            "box_site": GeoType.BOX,
            "capsule_site": GeoType.CAPSULE,
            "cylinder_site": GeoType.CYLINDER,
        }

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_label
        shape_types = model.shape_type.numpy()

        for i in range(model.shape_count):
            key = shape_keys[i]
            if key in expected_types and (shape_flags[i] & ShapeFlags.SITE):
                self.assertEqual(shape_types[i], expected_types[key], f"Type mismatch for {key}")

    def test_parse_site_orientations(self):
        """Test parsing sites with different orientations."""
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)

        def Sphere "identity_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
        }

        def Sphere "rotated_site" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
            quatd xformOp:orient = (0.7071068, 0, 0, 0.7071068)
            uniform token[] xformOpOrder = ["xformOp:orient"]
        }
    }
}
"""
        stage = self._create_usd_stage(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_label
        shape_transforms = model.shape_transform.numpy()

        # Find sites and check orientations
        found_sites = {}
        for i in range(model.shape_count):
            if shape_flags[i] & ShapeFlags.SITE:
                key = shape_keys[i]
                # Extract the site name from the full path
                for site_name in ["identity_site", "rotated_site"]:
                    if site_name in key:
                        xform = wp.transform(*shape_transforms[i])
                        quat = wp.transform_get_rotation(xform)
                        found_sites[site_name] = [quat.w, quat.x, quat.y, quat.z]
                        break

        # Check both sites were found
        self.assertEqual(len(found_sites), 2, "Expected to find 2 sites with orientations")

        # identity_site should have identity quaternion
        np.testing.assert_allclose(
            found_sites["identity_site"], [1, 0, 0, 0], atol=1e-5, err_msg="Identity quaternion mismatch"
        )

        # rotated_site should have 90° Z rotation
        np.testing.assert_allclose(
            found_sites["rotated_site"],
            [0.7071068, 0, 0, 0.7071068],
            atol=1e-5,
            err_msg="Rotated quaternion mismatch",
        )

    def test_site_without_mjcsite_api(self):
        """Test that shapes without MjcSiteAPI are not treated as sites."""
        usd_content = """#usda 1.0
(
    upAxis = "Z"
)

def Xform "World"
{
    def Xform "link" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.0
        float3 physics:diagonalInertia = (0.1, 0.1, 0.1)

        def Sphere "regular_sphere"
        {
            double radius = 0.01
        }

        def Sphere "site_sphere" (
            prepend apiSchemas = ["MjcSiteAPI"]
        )
        {
            double radius = 0.01
        }
    }
}
"""
        stage = self._create_usd_stage(usd_content)

        builder = newton.ModelBuilder()
        builder.add_usd(stage)
        model = builder.finalize()

        shape_flags = model.shape_flags.numpy()
        shape_keys = model.shape_label

        # Count sites
        site_count = 0
        regular_is_site = False
        site_is_site = False

        for i in range(model.shape_count):
            if shape_flags[i] & ShapeFlags.SITE:
                site_count += 1
                if "regular_sphere" in shape_keys[i]:
                    regular_is_site = True
                if "site_sphere" in shape_keys[i]:
                    site_is_site = True

        # Should have exactly 1 site
        self.assertEqual(site_count, 1, "Should have exactly 1 site")
        self.assertFalse(regular_is_site, "regular_sphere should not be a site")
        self.assertTrue(site_is_site, "site_sphere should be a site")


if __name__ == "__main__":
    unittest.main()
