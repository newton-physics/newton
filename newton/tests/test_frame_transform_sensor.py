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

"""Tests for FrameTransformSensor."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.articulation import eval_fk
from newton.sensors import FrameTransformSensor


class TestFrameTransformSensor(unittest.TestCase):
    """Test FrameTransformSensor functionality."""

    def test_sensor_creation(self):
        """Test basic sensor creation."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site1 = builder.add_site(body, key="site1")
        site2 = builder.add_site(body, key="site2")

        model = builder.finalize()

        # Create sensor
        sensor = FrameTransformSensor(model, shape_indices=[site1], reference_site_indices=[site2])

        self.assertEqual(len(sensor.shape_indices), 1)
        self.assertEqual(len(sensor.reference_site_indices), 1)

        # Both sites are at the same location (identity transform), verify they remain so
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)
        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # Should be identity transform (same location)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        quat = wp.transform_get_rotation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose([quat.w, quat.x, quat.y, quat.z], [1, 0, 0, 0], atol=1e-5)

    def test_sensor_single_reference_for_multiple_shapes(self):
        """Test single reference site for multiple shapes."""
        builder = newton.ModelBuilder()
        body = builder.add_body()

        site1 = builder.add_site(body, key="site1")
        site2 = builder.add_site(body, key="site2")
        site3 = builder.add_site(body, key="site3")
        ref_site = builder.add_site(body, key="ref")

        model = builder.finalize()

        # Create sensor with one reference for multiple shapes
        sensor = FrameTransformSensor(
            model,
            shape_indices=[site1, site2, site3],
            reference_site_indices=[ref_site],  # Single reference
        )

        # Should expand to match all shapes
        self.assertEqual(len(sensor.reference_site_indices), 3)
        self.assertEqual(sensor.reference_site_indices, [ref_site, ref_site, ref_site])

    def test_sensor_validation_empty_shapes(self):
        """Test error when shape_indices is empty."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[], reference_site_indices=[site])

    def test_sensor_validation_empty_references(self):
        """Test error when reference_site_indices is empty."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[site], reference_site_indices=[])

    def test_sensor_validation_invalid_shape_index(self):
        """Test error when shape index is out of bounds."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[9999], reference_site_indices=[site])

    def test_sensor_validation_invalid_reference_index(self):
        """Test error when reference index is out of bounds."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[site], reference_site_indices=[9999])

    def test_sensor_validation_reference_not_site(self):
        """Test error when reference index is not a site."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site = builder.add_site(body)
        shape = builder.add_shape_sphere(body, radius=0.1)  # Regular shape, not a site
        model = builder.finalize()

        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[site], reference_site_indices=[shape])

    def test_sensor_validation_mismatched_lengths(self):
        """Test error when reference indices don't match shape indices."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        site1 = builder.add_site(body)
        site2 = builder.add_site(body)
        site3 = builder.add_site(body)
        model = builder.finalize()

        # 2 shapes but 2 references (not 1 or 2)
        with self.assertRaises(ValueError):
            FrameTransformSensor(model, shape_indices=[site1, site2], reference_site_indices=[site3, site3, site3])

    def test_sensor_site_to_site_same_body(self):
        """Test measuring site relative to another site on same body."""
        builder = newton.ModelBuilder()

        # Body rotated 45° around Z
        body = builder.add_body(
            xform=wp.transform(wp.vec3(5, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4))
        )

        # Reference site at body origin, rotated 30° around Y
        ref_site = builder.add_site(
            body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 6)), key="ref"
        )

        # Target site offset and rotated 60° around X
        target_site = builder.add_site(
            body,
            xform=wp.transform(wp.vec3(0.5, 0.3, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 3)),
            key="target",
        )

        builder.add_joint_free(body)
        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[target_site], reference_site_indices=[ref_site])

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # Relative transform should still be local offset (both on same body)
        # The position in the reference frame is affected by the reference frame's rotation
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        quat = wp.transform_get_rotation(wp.transform(*transforms[0]))

        # Position: target is at (0.5, 0.3, 0) in body frame
        # When expressed in reference frame (rotated 30° around Y), this becomes:
        # Rotating by -30° around Y: x' = x*cos(30°) - z*sin(30°), y' = y, z' = x*sin(30°) + z*cos(30°)
        expected_x = 0.5 * np.cos(np.pi / 6)  # ≈ 0.433
        expected_y = 0.3
        expected_z = 0.5 * np.sin(np.pi / 6)  # ≈ 0.25

        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [expected_x, expected_y, expected_z], atol=1e-3)

        # Verify rotation is not identity
        self.assertGreater(abs(quat.x) + abs(quat.y) + abs(quat.z), 0.1)

    def test_sensor_site_to_site_different_bodies(self):
        """Test measuring site relative to site on different body."""
        builder = newton.ModelBuilder()

        # Reference body at origin, rotated 45° around Z
        body1 = builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4))
        )
        # Reference site offset by (0.2, 0.1, 0), rotated 30° around X
        ref_site = builder.add_site(
            body1,
            xform=wp.transform(wp.vec3(0.2, 0.1, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 6)),
            key="ref",
        )
        builder.add_joint_free(body1)

        # Target body at (1, 2, 3), rotated 60° around Y
        body2 = builder.add_body(
            xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 3))
        )
        # Target site offset by (0.3, 0, 0.2), rotated 90° around Z
        target_site = builder.add_site(
            body2,
            xform=wp.transform(wp.vec3(0.3, 0, 0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
            key="target",
        )
        builder.add_joint_free(body2)

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[target_site], reference_site_indices=[ref_site])

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        quat = wp.transform_get_rotation(wp.transform(*transforms[0]))

        # Compute expected transform using same operations as the sensor
        # Reference site world transform: body1_xform * site1_xform
        body1_xform = wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4))
        site1_local = wp.transform(wp.vec3(0.2, 0.1, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 6))
        ref_world_xform = wp.transform_multiply(body1_xform, site1_local)

        # Target site world transform: body2_xform * site2_xform
        body2_xform = wp.transform(wp.vec3(1, 2, 3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 3))
        site2_local = wp.transform(wp.vec3(0.3, 0, 0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2))
        target_world_xform = wp.transform_multiply(body2_xform, site2_local)

        # Relative transform: inverse(ref) * target
        expected_xform = wp.transform_multiply(wp.transform_inverse(ref_world_xform), target_world_xform)

        expected_pos = wp.transform_get_translation(expected_xform)
        expected_quat = wp.transform_get_rotation(expected_xform)

        # Test position
        np.testing.assert_allclose(
            [pos[0], pos[1], pos[2]], [expected_pos[0], expected_pos[1], expected_pos[2]], atol=1e-5
        )

        # Test rotation
        np.testing.assert_allclose(
            [quat.w, quat.x, quat.y, quat.z],
            [expected_quat.w, expected_quat.x, expected_quat.y, expected_quat.z],
            atol=1e-5,
        )

    def test_sensor_shape_to_site(self):
        """Test measuring regular shape relative to site."""
        builder = newton.ModelBuilder()

        body1 = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()))
        ref_site = builder.add_site(body1, key="ref")
        builder.add_joint_free(body1)

        body2 = builder.add_body(xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()))
        geom = builder.add_shape_sphere(body2, radius=0.1, xform=wp.transform(wp.vec3(0.5, 0, 0), wp.quat_identity()))
        builder.add_joint_free(body2)

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[geom], reference_site_indices=[ref_site])

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [1.5, 0, 0], atol=1e-5)

    def test_sensor_multiple_shapes_single_reference(self):
        """Test multiple shapes measured relative to single reference."""
        builder = newton.ModelBuilder()

        body = builder.add_body(xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_identity()))
        ref_site = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="ref")

        site_a = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="site_a")
        site_b = builder.add_site(body, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()), key="site_b")
        site_c = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()), key="site_c")

        builder.add_joint_free(body)
        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(
            model,
            shape_indices=[site_a, site_b, site_c],
            reference_site_indices=[ref_site],  # Single reference for all
        )

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        self.assertEqual(transforms.shape[0], 3)

        # Check each transform
        pos_a = wp.transform_get_translation(wp.transform(*transforms[0]))
        pos_b = wp.transform_get_translation(wp.transform(*transforms[1]))
        pos_c = wp.transform_get_translation(wp.transform(*transforms[2]))

        np.testing.assert_allclose([pos_a[0], pos_a[1], pos_a[2]], [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose([pos_b[0], pos_b[1], pos_b[2]], [0, 1, 0], atol=1e-5)
        np.testing.assert_allclose([pos_c[0], pos_c[1], pos_c[2]], [0, 0, 1], atol=1e-5)

    def test_sensor_world_frame_site(self):
        """Test site attached to world frame (body=-1)."""
        builder = newton.ModelBuilder()

        # World site at (5, 6, 7)
        world_site = builder.add_site(-1, xform=wp.transform(wp.vec3(5, 6, 7), wp.quat_identity()), key="world")

        # Moving site
        body = builder.add_body(xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_identity()))
        moving_site = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="moving")
        builder.add_joint_free(body)

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[moving_site], reference_site_indices=[world_site])

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # Moving site should be at (1,2,3) relative to world site at (5,6,7)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [-4, -4, -4], atol=1e-5)

    def test_sensor_with_rotation(self):
        """Test sensor with rotated reference frame."""
        builder = newton.ModelBuilder()

        # Reference frame rotated 90 degrees around Z
        body1 = builder.add_body(
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2))
        )
        ref_site = builder.add_site(body1, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="ref")
        builder.add_joint_free(body1)

        # Target at (1, 0, 0) in world frame
        body2 = builder.add_body(xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()))
        target_site = builder.add_site(body2, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="target")
        builder.add_joint_free(body2)

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[target_site], reference_site_indices=[ref_site])

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # In reference frame rotated 90° around Z, point (1,0,0) should appear as (0,1,0)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0, -1, 0], atol=1e-5)

    def test_sensor_with_site_rotations(self):
        """Test sensor with sites that have non-identity rotations."""
        builder = newton.ModelBuilder()

        body = builder.add_body()

        # Reference site rotated 45° around Z
        ref_site = builder.add_site(
            body, xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4)), key="ref"
        )

        # Target site at (2, 0, 0), rotated 90° around Y
        target_site = builder.add_site(
            body,
            xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2)),
            key="target",
        )

        model = builder.finalize()
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[target_site], reference_site_indices=[ref_site])
        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # Target is 1 unit away in X direction (in ref frame coords)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))

        # Relative position: rotating -45° around Z transforms (1,0,0) to (0.707,-0.707,0)
        np.testing.assert_allclose([pos[0], pos[1]], [0.707, -0.707], atol=1e-3)

        # Check rotation is preserved
        quat = wp.transform_get_rotation(wp.transform(*transforms[0]))
        # Should not be identity
        self.assertGreater(abs(quat.x) + abs(quat.y) + abs(quat.z), 0.1)

    def test_sensor_articulation_chain(self):
        """Test sensor with sites on different links of an articulation chain."""
        builder = newton.ModelBuilder()

        # Root body at origin
        root = builder.add_body()
        ref_site = builder.add_site(root, key="ref")

        # Link 1: connected by revolute joint, extends 1m in +X from joint
        link1 = builder.add_body()
        site1 = builder.add_site(link1, key="site1")
        builder.add_joint_revolute(
            parent=root,
            child=link1,
            axis=wp.vec3(0, 0, 1),
            child_xform=wp.transform(wp.vec3(-1, 0, 0), wp.quat_identity()),  # Joint is 1m from link1 origin
        )

        # Link 2: connected to link1, extends another 1m in +X
        link2 = builder.add_body()
        site2 = builder.add_site(link2, key="site2")
        builder.add_joint_revolute(
            parent=link1,
            child=link2,
            axis=wp.vec3(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),  # Joint is 1m from link1 origin
            child_xform=wp.transform(wp.vec3(-1, 0, 0), wp.quat_identity()),  # Joint is 1m from link2 origin
        )

        model = builder.finalize()
        state = model.state()

        # Test with joints at zero position
        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = FrameTransformSensor(model, shape_indices=[site1, site2], reference_site_indices=[ref_site])
        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        # At zero joint angles, site1 should be at (1, 0, 0) and site2 at (3, 0, 0)
        # (link1 extends 1m from root, link2 extends 2m from link1)
        pos1 = wp.transform_get_translation(wp.transform(*transforms[0]))
        pos2 = wp.transform_get_translation(wp.transform(*transforms[1]))

        np.testing.assert_allclose([pos1[0], pos1[1], pos1[2]], [1, 0, 0], atol=1e-5)
        np.testing.assert_allclose([pos2[0], pos2[1], pos2[2]], [3, 0, 0], atol=1e-5)

        # Now rotate first joint by 90 degrees
        q_np = state.joint_q.numpy()
        q_np[0] = np.pi / 2
        state.joint_q.assign(q_np)
        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor.update(model, state)
        transforms = sensor.transforms.numpy()

        pos1 = wp.transform_get_translation(wp.transform(*transforms[0]))
        pos2 = wp.transform_get_translation(wp.transform(*transforms[1]))

        # After 90° rotation: site1 at (0, 1, 0), site2 at (0, 3, 0)
        np.testing.assert_allclose([pos1[0], pos1[1], pos1[2]], [0, 1, 0], atol=1e-5)
        np.testing.assert_allclose([pos2[0], pos2[1], pos2[2]], [0, 3, 0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
