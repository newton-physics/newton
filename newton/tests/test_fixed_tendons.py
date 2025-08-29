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

"""Tests for fixed tendon support in Newton."""

import unittest

import numpy as np
import warp as wp

import newton


class TestFixedTendons(unittest.TestCase):
    """Test cases for fixed tendon functionality."""

    def test_add_tendon_basic(self):
        """Test basic tendon creation and validation."""
        builder = newton.ModelBuilder()

        # Create simple chain
        ground = builder.add_body(mass=0.0)
        bodies = []
        joints = []

        for _i in range(3):
            body = builder.add_body(mass=1.0)
            joint = builder.add_joint_revolute(
                parent=bodies[-1] if bodies else ground,
                child=body,
                parent_xform=wp.transform([0.0, 0.0, 1.0], wp.quat_identity()),
                child_xform=wp.transform_identity(),
                axis=[1.0, 0.0, 0.0],
            )
            bodies.append(body)
            joints.append(joint)

        # Add tendon
        tendon_id = builder.add_tendon(
            name="test_tendon",
            joint_ids=[joints[0], joints[2]],
            gearings=[1.0, -1.0],
            stiffness=100.0,
            damping=10.0,
            rest_length=0.0,
            lower_limit=-1.0,
            upper_limit=1.0,
        )

        self.assertEqual(tendon_id, 0)
        self.assertEqual(len(builder.tendon_start), 1)
        self.assertEqual(len(builder.tendon_key), 1)
        self.assertEqual(builder.tendon_key[0], "test_tendon")

    def test_add_tendon_validation(self):
        """Test tendon creation validation."""
        builder = newton.ModelBuilder()

        # Create minimal setup
        body1 = builder.add_body(mass=1.0)
        body2 = builder.add_body(mass=1.0)
        joint1 = builder.add_joint_revolute(
            parent=-1,
            child=body1,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=[1.0, 0.0, 0.0],
        )
        joint2 = builder.add_joint_revolute(
            parent=body1,
            child=body2,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            axis=[1.0, 0.0, 0.0],
        )

        # Test single joint error
        with self.assertRaises(ValueError):
            builder.add_tendon(name="bad_tendon", joint_ids=[joint1], gearings=[1.0])

        # Test mismatched gearings error
        with self.assertRaises(ValueError):
            builder.add_tendon(
                name="bad_tendon",
                joint_ids=[joint1, joint2],
                gearings=[1.0],  # Should have 2 gearings
            )

    def test_tendon_finalization(self):
        """Test tendon data transfer to Model."""
        builder = newton.ModelBuilder()

        # Create joints
        body = builder.add_body(mass=1.0, key="body1")
        joint1 = builder.add_joint_revolute(
            parent=-1,
            child=body,
            parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[1.0, 0.0, 0.0],
        )
        body2 = builder.add_body(mass=1.0, key="body2")
        joint2 = builder.add_joint_revolute(
            parent=body,
            child=body2,
            parent_xform=wp.transform([0.0, 0.0, 1.0], wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=[1.0, 0.0, 0.0],
        )

        # Add tendons
        builder.add_tendon(
            name="tendon1",
            joint_ids=[joint1, joint2],
            gearings=[1.0, -1.0],
            stiffness=50.0,
            damping=5.0,
            rest_length=0.1,
        )

        builder.add_tendon(
            name="tendon2",
            joint_ids=[joint1, joint2],
            gearings=[2.0, -2.0],
            stiffness=100.0,
            damping=10.0,
            rest_length=0.0,
            lower_limit=-0.5,
            upper_limit=0.5,
        )

        # Finalize
        model = builder.finalize()

        # Check counts
        self.assertEqual(model.tendon_count, 2)

        # Check arrays exist
        self.assertIsNotNone(model.tendon_start)
        self.assertIsNotNone(model.tendon_params)
        self.assertIsNotNone(model.tendon_joints)
        self.assertIsNotNone(model.tendon_gearings)
        self.assertEqual(len(model.tendon_key), 2)

        # Check data
        tendon_start = model.tendon_start.numpy()
        tendon_params = model.tendon_params.numpy()

        # First tendon
        self.assertEqual(tendon_start[0], 0)
        self.assertEqual(tendon_start[1], 2)
        np.testing.assert_array_almost_equal(tendon_params[0], [50.0, 5.0, 0.1, float("-inf"), float("inf")], decimal=5)

        # Second tendon
        self.assertEqual(tendon_start[2], 4)
        np.testing.assert_array_almost_equal(tendon_params[1], [100.0, 10.0, 0.0, -0.5, 0.5], decimal=5)


if __name__ == "__main__":
    unittest.main()
