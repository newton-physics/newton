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

"""Tests for site export to MuJoCo."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton.solvers import SolverMuJoCo


class TestMuJoCoSiteExport(unittest.TestCase):
    """Test exporting sites to MuJoCo models."""

    def test_export_single_site(self):
        """Test that a site is exported to both MuJoCo Warp and regular MuJoCo models."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))
        builder.add_site(
            body, type=GeoType.SPHERE, key="test_site", xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity())
        )
        builder.add_joint_free(child=body)

        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model
        mj_model = solver.mj_model

        # Verify site exists in MuJoCo Warp model
        self.assertGreater(mjw_model.nsite, 0, "Site should be exported to MuJoCo Warp model")

        # Verify site exists in regular MuJoCo model
        self.assertGreater(mj_model.nsite, 0, "Site should be exported to regular MuJoCo model")

        # Both should have the same site count
        self.assertEqual(mjw_model.nsite, mj_model.nsite, "Both models should have the same site count")

    def test_export_multiple_sites(self):
        """Test exporting multiple sites."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, key="site_1")
        builder.add_site(body, type=GeoType.SPHERE, key="site_2")
        builder.add_site(body, type=GeoType.SPHERE, key="site_3")

        builder.add_joint_free(child=body)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        self.assertEqual(mjw_model.nsite, 3, "Should have exactly 3 sites")

    def test_site_not_exported_as_geom(self):
        """Test that sites are NOT exported as collision geoms."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))

        # Add site
        builder.add_site(body, type=GeoType.SPHERE, key="my_site")
        # Add regular collision shape
        builder.add_shape_sphere(body, radius=0.1)

        builder.add_joint_free(child=body)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Should have 1 site and 1 geom
        self.assertEqual(mjw_model.nsite, 1, "Should have exactly 1 site")
        self.assertEqual(mjw_model.ngeom, 1, "Should have exactly 1 geom (not counting site)")

    def test_export_site_transforms(self):
        """Test that site transforms are correctly exported."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))

        site_xform = wp.transform(wp.vec3(0.5, 0.3, 0.1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.57))
        builder.add_site(body, type=GeoType.SPHERE, xform=site_xform, key="positioned_site")

        builder.add_joint_free(child=body)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify site exists and check position
        self.assertGreater(mjw_model.nsite, 0, "Site should exist")
        site_pos = mjw_model.site_pos.numpy()[0, 0]  # First world, first site
        np.testing.assert_allclose(site_pos[:3], [0.5, 0.3, 0.1], atol=1e-5)

    def test_export_site_types(self):
        """Test that site types are exported correctly."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, scale=(0.05, 0.05, 0.05), key="sphere")
        builder.add_site(body, type=GeoType.BOX, scale=(0.1, 0.2, 0.3), key="box")

        builder.add_joint_free(child=body)
        model = builder.finalize()

        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify we have 2 sites
        self.assertEqual(mjw_model.nsite, 2, "Should have 2 sites")

        # Verify types - first should be sphere (2), second should be box (6)
        site_types = mjw_model.site_type.numpy()
        if len(site_types.shape) == 2:  # [nworld, nsite]
            site_types = site_types[0]  # First world
        self.assertEqual(site_types[0], 2, "First site should be sphere (type 2)")
        self.assertEqual(site_types[1], 6, "Second site should be box (type 6)")

    def test_export_include_sites_parameter(self):
        """Test that sites are exported by default."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, I_m=wp.mat33(np.eye(3)))

        builder.add_site(body, type=GeoType.SPHERE, key="my_site")
        builder.add_joint_free(child=body)

        model = builder.finalize()

        # Create solver - sites should be exported by default
        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model
        self.assertEqual(mjw_model.nsite, 1, "Should have 1 site exported by default")

    def test_mjcf_roundtrip(self):
        """Test MJCF → Newton → MuJoCo round-trip preserves sites."""
        mjcf = """
        <mujoco>
            <worldbody>
                <body name="link" pos="0 0 1">
                    <joint type="free"/>
                    <site name="sensor_site" pos="0.1 0.05 0" type="sphere" size="0.02"/>
                    <geom name="body_geom" type="box" size="0.1 0.1 0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        # Import to Newton
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()

        # Export to MuJoCo
        solver = SolverMuJoCo(model)
        mjw_model = solver.mjw_model

        # Verify site exists (should have at least 1 site)
        self.assertGreater(mjw_model.nsite, 0, "Should have at least 1 site after round-trip")

        # Verify we have 1 geom (the box)
        self.assertGreater(mjw_model.ngeom, 0, "Should have at least 1 geom (the box)")


if __name__ == "__main__":
    unittest.main()
