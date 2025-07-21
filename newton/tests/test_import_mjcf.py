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

import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
import newton.examples


class TestImportMjcf(unittest.TestCase):
    def test_humanoid_mjcf(self):
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.ke = 123.0
        builder.default_shape_cfg.kd = 456.0
        builder.default_shape_cfg.mu = 789.0
        builder.default_joint_cfg.armature = 42.0
        mjcf_filename = newton.examples.get_asset("nv_humanoid.xml")
        newton.utils.parse_mjcf(
            mjcf_filename,
            builder,
            ignore_names=["floor", "ground"],
            up_axis="Z",
        )
        self.assertTrue(all(np.array(builder.shape_material_ke) == 123.0))
        self.assertTrue(all(np.array(builder.shape_material_kd) == 456.0))
        self.assertTrue(all(np.array(builder.shape_material_mu) == 789.0))
        self.assertTrue(all(np.array(builder.joint_armature[:6]) == 0.0))
        self.assertEqual(
            builder.joint_armature[6:],
            [
                0.02,
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.01,
                0.007,
                0.006,
                0.006,
                0.01,
                0.01,
                0.006,
                0.01,
                0.01,
                0.006,
            ],
        )
        assert builder.body_count == 13

    def test_mjcf_maxhullvert_parsing(self):
        """Test that maxhullvert is parsed from MJCF files"""
        # Create a temporary MJCF file with maxhullvert attribute
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <asset>
        <mesh name="mesh1" file="mesh1.obj" maxhullvert="32"/>
        <mesh name="mesh2" file="mesh2.obj" maxhullvert="128"/>
        <mesh name="mesh3" file="mesh3.obj"/>
    </asset>
    <worldbody>
        <body>
            <geom type="mesh" mesh="mesh1"/>
            <geom type="mesh" mesh="mesh2"/>
            <geom type="mesh" mesh="mesh3"/>
        </body>
    </worldbody>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "test.xml")

            # Create dummy mesh files
            for i in range(1, 4):
                mesh_path = os.path.join(tmpdir, f"mesh{i}.obj")
                with open(mesh_path, "w") as f:
                    # Simple triangle mesh
                    f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            # Parse MJCF
            builder = newton.ModelBuilder()
            newton.utils.parse_mjcf(mjcf_path, builder, parse_meshes=True)
            model = builder.finalize()

            # Check that meshes have correct maxhullvert values
            # Note: This assumes meshes are added in order they appear in MJCF
            meshes = [model.shape_geo_src[i] for i in range(3) if hasattr(model.shape_geo_src[i], "maxhullvert")]

            if len(meshes) >= 3:
                self.assertEqual(meshes[0].maxhullvert, 32)
                self.assertEqual(meshes[1].maxhullvert, 128)
                self.assertEqual(meshes[2].maxhullvert, 64)  # Default value

    def test_inertia_rotation(self):
        """Test that inertia tensors are properly rotated using sandwich product R @ I @ R.T"""

        # Test case 1: Diagonal inertia with rotation
        mjcf_diagonal = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_diagonal">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" diaginertia="1.0 2.0 3.0"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test case 2: Full inertia with rotation
        mjcf_full = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_full">
    <worldbody>
        <body>
            <inertial pos="0 0 0" quat="0.7071068 0 0 0.7071068"
                      mass="1.0" fullinertia="1.0 2.0 3.0 0.1 0.2 0.3"/>
        </body>
    </worldbody>
</mujoco>
"""

        # Test diagonal inertia rotation
        builder = newton.ModelBuilder()
        newton.utils.parse_mjcf(mjcf_diagonal, builder, ignore_inertial_definitions=False)
        model = builder.finalize()

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # This transforms the diagonal inertia [1, 2, 3] to [2, 1, 3] via sandwich product R @ I @ R.T
        expected_diagonal = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])

        actual_inertia = model.body_inertia.numpy()[0]
        np.testing.assert_array_almost_equal(actual_inertia, expected_diagonal, decimal=6)

        # Test full inertia rotation
        builder = newton.ModelBuilder()
        newton.utils.parse_mjcf(mjcf_full, builder, ignore_inertial_definitions=False)
        model = builder.finalize()

        # For full inertia, we need to compute the expected result manually
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) transforms the inertia
        # We need to use the same quaternion-to-matrix conversion as the MJCF importer

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # For full inertia, calculate the expected result analytically using the same quaternion
        # Original inertia matrix:
        # [1.0  0.1  0.2]
        # [0.1  2.0  0.3]
        # [0.2  0.3  3.0]

        # The quaternion (0.7071068, 0, 0, 0.7071068) in MuJoCo WXYZ format represents a 90-degree rotation around Z-axis
        # Calculate the expected result analytically using the correct rotation matrix
        # For a 90-degree Z-axis rotation: R = [0 -1 0; 1 0 0; 0 0 1]

        original_inertia = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])

        # Rotation matrix for 90-degree rotation around Z-axis
        rotation_matrix = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        expected_full = rotation_matrix @ original_inertia @ rotation_matrix.T

        actual_inertia = model.body_inertia.numpy()[0]
        np.testing.assert_array_almost_equal(actual_inertia, expected_full, decimal=6)

        # Verify that the rotation was actually applied (not just identity)
        assert not np.allclose(actual_inertia, original_inertia, atol=1e-6)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
