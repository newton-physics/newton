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
from newton._src.sim.joints import JOINT_LIMIT_UNLIMITED
from newton._src.utils.import_urdf import parse_urdf


class TestJointLimits(unittest.TestCase):
    """Test joint limit handling with sentinel values for unlimited joints."""

    def test_unlimited_joint_defaults(self):
        """Test that joints have unlimited limits by default."""
        builder = newton.ModelBuilder()

        # Add a body
        body = builder.add_body()

        # Add a revolute joint with default limits
        builder.add_joint_revolute(parent=-1, child=body)

        # Build model
        model = builder.finalize()

        # Check that default limits are unlimited
        lower_limits = model.joint_limit_lower.numpy()
        upper_limits = model.joint_limit_upper.numpy()
        self.assertEqual(lower_limits[0], -JOINT_LIMIT_UNLIMITED)
        self.assertEqual(upper_limits[0], JOINT_LIMIT_UNLIMITED)

    def test_limited_joint(self):
        """Test that limited joints work correctly."""
        builder = newton.ModelBuilder()

        # Add a body
        body = builder.add_body()

        # Add a revolute joint with specific limits
        builder.add_joint_revolute(parent=-1, child=body, limit_lower=-1.0, limit_upper=1.0)

        # Build model
        model = builder.finalize()

        # Check that limits are set correctly
        lower_limits = model.joint_limit_lower.numpy()
        upper_limits = model.joint_limit_upper.numpy()
        self.assertAlmostEqual(lower_limits[0], -1.0)
        self.assertAlmostEqual(upper_limits[0], 1.0)

    def test_partially_limited_joint(self):
        """Test joints with only one limit being unlimited."""
        builder = newton.ModelBuilder()

        # Add a body
        body = builder.add_body()

        # Add a revolute joint with only upper limit
        builder.add_joint_revolute(parent=-1, child=body, limit_lower=-JOINT_LIMIT_UNLIMITED, limit_upper=2.0)

        # Build model
        model = builder.finalize()

        # Check lower is unlimited, upper is limited
        lower_limits = model.joint_limit_lower.numpy()
        upper_limits = model.joint_limit_upper.numpy()
        self.assertEqual(lower_limits[0], -JOINT_LIMIT_UNLIMITED)
        self.assertAlmostEqual(upper_limits[0], 2.0)

        # Test the other way - only lower limit
        builder2 = newton.ModelBuilder()
        body2 = builder2.add_body()
        builder2.add_joint_revolute(parent=-1, child=body2, limit_lower=-1.5, limit_upper=JOINT_LIMIT_UNLIMITED)
        model2 = builder2.finalize()

        lower_limits2 = model2.joint_limit_lower.numpy()
        upper_limits2 = model2.joint_limit_upper.numpy()
        self.assertAlmostEqual(lower_limits2[0], -1.5)
        self.assertEqual(upper_limits2[0], JOINT_LIMIT_UNLIMITED)

    def test_continuous_joint_from_urdf(self):
        """Test that continuous joints from URDF are unlimited."""
        urdf_content = """<?xml version="1.0"?>
        <robot name="test_robot">
            <link name="base_link">
                <inertial>
                    <mass value="1.0"/>
                    <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0.0" ixz="0.0" iyz="0.0"/>
                </inertial>
            </link>
            <link name="rotating_link">
                <inertial>
                    <mass value="0.5"/>
                    <inertia ixx="0.05" iyy="0.05" izz="0.05" ixy="0.0" ixz="0.0" iyz="0.0"/>
                </inertial>
            </link>
            <joint name="continuous_joint" type="continuous">
                <parent link="base_link"/>
                <child link="rotating_link"/>
                <axis xyz="0 0 1"/>
            </joint>
        </robot>
        """

        # Write URDF to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as f:
            f.write(urdf_content)
            urdf_path = f.name

        try:
            # Import URDF
            builder = newton.ModelBuilder()
            parse_urdf(builder, urdf_path)
            model = builder.finalize()

            # Find the continuous joint (should be the first joint)
            lower_limits = model.joint_limit_lower.numpy()
            upper_limits = model.joint_limit_upper.numpy()
            self.assertEqual(lower_limits[0], -JOINT_LIMIT_UNLIMITED)
            self.assertEqual(upper_limits[0], JOINT_LIMIT_UNLIMITED)

        finally:
            # Clean up temp file
            os.unlink(urdf_path)

    def test_joint_d6_with_mixed_limits(self):
        """Test D6 joint with mixed limited and unlimited axes."""
        builder = newton.ModelBuilder()

        # Add a body
        body = builder.add_body()

        # Create a D6 joint with:
        # - X translation: limited
        # - Y translation: unlimited
        # - Z translation: partially limited (only lower)
        # - X rotation: unlimited
        # - Y rotation: limited
        # - Z rotation: partially limited (only upper)
        builder.add_joint_d6(
            parent=-1,
            child=body,
            linear_axes=[
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, limit_lower=-1.0, limit_upper=1.0),
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Y, limit_lower=-JOINT_LIMIT_UNLIMITED, limit_upper=JOINT_LIMIT_UNLIMITED
                ),
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Z, limit_lower=-0.5, limit_upper=JOINT_LIMIT_UNLIMITED
                ),
            ],
            angular_axes=[
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.X, limit_lower=-JOINT_LIMIT_UNLIMITED, limit_upper=JOINT_LIMIT_UNLIMITED
                ),
                newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, limit_lower=-np.pi / 4, limit_upper=np.pi / 4),
                newton.ModelBuilder.JointDofConfig(
                    axis=newton.Axis.Z, limit_lower=-JOINT_LIMIT_UNLIMITED, limit_upper=np.pi / 2
                ),
            ],
        )

        model = builder.finalize()

        # Get numpy arrays for testing
        lower_limits = model.joint_limit_lower.numpy()
        upper_limits = model.joint_limit_upper.numpy()

        # Check linear axes
        self.assertAlmostEqual(lower_limits[0], -1.0)  # X limited
        self.assertAlmostEqual(upper_limits[0], 1.0)

        self.assertEqual(lower_limits[1], -JOINT_LIMIT_UNLIMITED)  # Y unlimited
        self.assertEqual(upper_limits[1], JOINT_LIMIT_UNLIMITED)

        self.assertAlmostEqual(lower_limits[2], -0.5)  # Z partially limited
        self.assertEqual(upper_limits[2], JOINT_LIMIT_UNLIMITED)

        # Check angular axes
        self.assertEqual(lower_limits[3], -JOINT_LIMIT_UNLIMITED)  # X rot unlimited
        self.assertEqual(upper_limits[3], JOINT_LIMIT_UNLIMITED)

        self.assertAlmostEqual(lower_limits[4], -np.pi / 4)  # Y rot limited
        self.assertAlmostEqual(upper_limits[4], np.pi / 4)

        self.assertEqual(lower_limits[5], -JOINT_LIMIT_UNLIMITED)  # Z rot partially limited
        self.assertAlmostEqual(upper_limits[5], np.pi / 2)

    def test_create_unlimited_joint_config(self):
        """Test the create_unlimited helper method."""
        # Create unlimited config
        config = newton.ModelBuilder.JointDofConfig.create_unlimited(newton.Axis.X)

        # Check limits are unlimited
        self.assertEqual(config.limit_lower, -JOINT_LIMIT_UNLIMITED)
        self.assertEqual(config.limit_upper, JOINT_LIMIT_UNLIMITED)

        # Check other properties
        self.assertEqual(config.limit_ke, 0.0)
        self.assertEqual(config.limit_kd, 0.0)
        self.assertEqual(config.armature, 0.0)
        self.assertEqual(config.mode, newton.JointMode.NONE)

    def test_robustness_of_limit_comparisons(self):
        """Test that limit comparisons work robustly with >= and <= operators."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        # Add joint with unlimited limits
        builder.add_joint_revolute(
            parent=-1, child=body, limit_lower=-JOINT_LIMIT_UNLIMITED, limit_upper=JOINT_LIMIT_UNLIMITED
        )

        model = builder.finalize()

        # Test robust comparisons
        # These should work even if JOINT_LIMIT_UNLIMITED changes from wp.inf to a large finite value
        lower_limits = model.joint_limit_lower.numpy()
        upper_limits = model.joint_limit_upper.numpy()
        self.assertTrue(lower_limits[0] <= -JOINT_LIMIT_UNLIMITED)
        self.assertTrue(upper_limits[0] >= JOINT_LIMIT_UNLIMITED)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
