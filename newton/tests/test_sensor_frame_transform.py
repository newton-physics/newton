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

"""Tests for SensorFrameTransform."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.articulation import eval_fk
from newton.sensors import SensorFrameTransform


class TestSensorFrameTransform(unittest.TestCase):
    """Test SensorFrameTransform functionality."""

    def test_sensor_creation(self):
        """Test basic sensor creation."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        site1 = builder.add_site(body, key="site1")
        site2 = builder.add_site(body, key="site2")

        model = builder.finalize()

        # Create sensor
        sensor = SensorFrameTransform(model, shapes=[site1], reference_sites=[site2])

        # Both sites are at the same location (identity transform), verify they remain so
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)
        sensor.update(state)
        transforms = sensor.transforms.numpy()

        # Should be identity transform (same location)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        quat = wp.transform_get_rotation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0, 0, 0], atol=1e-5)
        np.testing.assert_allclose([quat.w, quat.x, quat.y, quat.z], [1, 0, 0, 0], atol=1e-5)

    def test_sensor_single_reference_for_multiple_shapes(self):
        """Test single reference site for multiple shapes."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

        site1 = builder.add_site(body, key="site1")
        site2 = builder.add_site(body, key="site2")
        site3 = builder.add_site(body, key="site3")
        ref_site = builder.add_site(body, key="ref")

        model = builder.finalize()

        # Create sensor with one reference for multiple shapes
        sensor = SensorFrameTransform(
            model,
            shapes=[site1, site2, site3],
            reference_sites=[ref_site],  # Single reference
        )

        # Verify it creates successfully (reference is broadcasted internally)
        self.assertEqual(len(sensor.transforms), 3)

    def test_sensor_validation_empty_shapes(self):
        """Test error when shapes is empty."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[], reference_sites=[site])

    def test_sensor_validation_empty_references(self):
        """Test error when reference_sites is empty."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[site], reference_sites=[])

    def test_sensor_validation_invalid_shape_index(self):
        """Test error when shape index is out of bounds."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[9999], reference_sites=[site])

    def test_sensor_validation_invalid_reference_index(self):
        """Test error when reference index is out of bounds."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[site], reference_sites=[9999])

    def test_sensor_validation_reference_not_site(self):
        """Test error when reference index is not a site."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site = builder.add_site(body)
        shape = builder.add_shape_sphere(body, radius=0.1)  # Regular shape, not a site
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[site], reference_sites=[shape])

    def test_sensor_validation_mismatched_lengths(self):
        """Test error when reference indices don't match shape indices."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site1 = builder.add_site(body)
        site2 = builder.add_site(body)
        site3 = builder.add_site(body)
        model = builder.finalize()

        # 2 shapes but 3 references (not 1 or 2)
        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes=[site1, site2], reference_sites=[site3, site3, site3])

    def test_sensor_site_to_site_same_body(self):
        """Test measuring site relative to another site on same body."""
        builder = newton.ModelBuilder()

        # Body rotated 45° around Z
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(5, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4)),
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

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[target_site], reference_sites=[ref_site])

        sensor.update(state)
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
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4)),
        )
        # Reference site offset by (0.2, 0.1, 0), rotated 30° around X
        ref_site = builder.add_site(
            body1,
            xform=wp.transform(wp.vec3(0.2, 0.1, 0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi / 6)),
            key="ref",
        )

        # Target body at (1, 2, 3), rotated 60° around Y
        body2 = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 3)),
        )
        # Target site offset by (0.3, 0, 0.2), rotated 90° around Z
        target_site = builder.add_site(
            body2,
            xform=wp.transform(wp.vec3(0.3, 0, 0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
            key="target",
        )

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[target_site], reference_sites=[ref_site])

        sensor.update(state)
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

        body1 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity())
        )
        ref_site = builder.add_site(body1, key="ref")

        body2 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity())
        )
        geom = builder.add_shape_sphere(body2, radius=0.1, xform=wp.transform(wp.vec3(0.5, 0, 0), wp.quat_identity()))

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[geom], reference_sites=[ref_site])

        sensor.update(state)
        transforms = sensor.transforms.numpy()

        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [1.5, 0, 0], atol=1e-5)

    def test_sensor_multiple_shapes_single_reference(self):
        """Test multiple shapes measured relative to single reference."""
        builder = newton.ModelBuilder()

        body = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_identity())
        )
        ref_site = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="ref")

        site_a = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="site_a")
        site_b = builder.add_site(body, xform=wp.transform(wp.vec3(0, 1, 0), wp.quat_identity()), key="site_b")
        site_c = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 1), wp.quat_identity()), key="site_c")

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(
            model,
            shapes=[site_a, site_b, site_c],
            reference_sites=[ref_site],  # Single reference for all
        )

        sensor.update(state)
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
        body = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(1, 2, 3), wp.quat_identity())
        )
        moving_site = builder.add_site(body, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="moving")

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[moving_site], reference_sites=[world_site])

        sensor.update(state)
        transforms = sensor.transforms.numpy()

        # Moving site should be at (1,2,3) relative to world site at (5,6,7)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [-4, -4, -4], atol=1e-5)

    def test_sensor_with_rotation(self):
        """Test sensor with rotated reference frame."""
        builder = newton.ModelBuilder()

        # Reference frame rotated 90 degrees around Z
        body1 = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2)),
        )
        ref_site = builder.add_site(body1, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="ref")

        # Target at (1, 0, 0) in world frame
        body2 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity())
        )
        target_site = builder.add_site(body2, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="target")

        model = builder.finalize()
        state = model.state()

        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[target_site], reference_sites=[ref_site])

        sensor.update(state)
        transforms = sensor.transforms.numpy()

        # In reference frame rotated 90° around Z, point (1,0,0) should appear as (0,1,0)
        pos = wp.transform_get_translation(wp.transform(*transforms[0]))
        np.testing.assert_allclose([pos[0], pos[1], pos[2]], [0, -1, 0], atol=1e-5)

    def test_sensor_with_site_rotations(self):
        """Test sensor with sites that have non-identity rotations."""
        builder = newton.ModelBuilder()

        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))

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

        sensor = SensorFrameTransform(model, shapes=[target_site], reference_sites=[ref_site])
        sensor.update(state)
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
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        ref_site = builder.add_site(root, key="ref")

        # Link 1: connected by revolute joint, extends 1m in +X from joint
        link1 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site1 = builder.add_site(link1, key="site1")
        joint1 = builder.add_joint_revolute(
            parent=root,
            child=link1,
            axis=wp.vec3(0, 0, 1),
            child_xform=wp.transform(wp.vec3(-1, 0, 0), wp.quat_identity()),  # Joint is 1m from link1 origin
        )

        # Link 2: connected to link1, extends another 1m in +X
        link2 = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        site2 = builder.add_site(link2, key="site2")
        joint2 = builder.add_joint_revolute(
            parent=link1,
            child=link2,
            axis=wp.vec3(0, 0, 1),
            parent_xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),  # Joint is 1m from link1 origin
            child_xform=wp.transform(wp.vec3(-1, 0, 0), wp.quat_identity()),  # Joint is 1m from link2 origin
        )

        builder.add_articulation([joint1, joint2])

        model = builder.finalize()
        state = model.state()

        # Test with joints at zero position
        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor = SensorFrameTransform(model, shapes=[site1, site2], reference_sites=[ref_site])
        sensor.update(state)
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

        sensor.update(state)
        transforms = sensor.transforms.numpy()

        pos1 = wp.transform_get_translation(wp.transform(*transforms[0]))
        pos2 = wp.transform_get_translation(wp.transform(*transforms[1]))

        # After 90° rotation: site1 at (0, 1, 0), site2 at (0, 3, 0)
        np.testing.assert_allclose([pos1[0], pos1[1], pos1[2]], [0, 1, 0], atol=1e-5)
        np.testing.assert_allclose([pos2[0], pos2[1], pos2[2]], [0, 3, 0], atol=1e-5)

    def test_sensor_sparse_non_sorted_indices(self):
        """Test sensor with non-continuous, non-sorted shape indices."""
        builder = newton.ModelBuilder()

        # Create reference site
        base = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        ref_site = builder.add_site(base, key="ref")

        # Create multiple bodies with multiple sites to get many shapes
        all_sites = []

        # Body 1 with 3 sites at different positions
        body1 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity())
        )
        all_sites.append(builder.add_site(body1, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="b1_s0"))
        all_sites.append(
            builder.add_site(body1, xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity()), key="b1_s1")
        )
        all_sites.append(
            builder.add_site(body1, xform=wp.transform(wp.vec3(0.2, 0, 0), wp.quat_identity()), key="b1_s2")
        )

        # Body 2 with 3 sites at different positions
        body2 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(2, 0, 0), wp.quat_identity())
        )
        all_sites.append(builder.add_site(body2, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="b2_s0"))
        all_sites.append(
            builder.add_site(body2, xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity()), key="b2_s1")
        )
        all_sites.append(
            builder.add_site(body2, xform=wp.transform(wp.vec3(0.2, 0, 0), wp.quat_identity()), key="b2_s2")
        )

        # Body 3 with 3 sites at different positions
        body3 = builder.add_body(
            mass=1.0, inertia=wp.mat33(np.eye(3)), xform=wp.transform(wp.vec3(3, 0, 0), wp.quat_identity())
        )
        all_sites.append(builder.add_site(body3, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), key="b3_s0"))
        all_sites.append(
            builder.add_site(body3, xform=wp.transform(wp.vec3(0.1, 0, 0), wp.quat_identity()), key="b3_s1")
        )
        all_sites.append(
            builder.add_site(body3, xform=wp.transform(wp.vec3(0.2, 0, 0), wp.quat_identity()), key="b3_s2")
        )

        model = builder.finalize()
        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        # Shape indices accounting for ref_site being index 0:
        # ref_site = 0
        # all_sites[0] = 1 (b1_s0) at (1, 0, 0)
        # all_sites[1] = 2 (b1_s1) at (1.1, 0, 0)
        # all_sites[2] = 3 (b1_s2) at (1.2, 0, 0)
        # all_sites[3] = 4 (b2_s0) at (2, 0, 0)
        # all_sites[4] = 5 (b2_s1) at (2.1, 0, 0)
        # all_sites[5] = 6 (b2_s2) at (2.2, 0, 0)
        # all_sites[6] = 7 (b3_s0) at (3, 0, 0)
        # all_sites[7] = 8 (b3_s1) at (3.1, 0, 0)
        # all_sites[8] = 9 (b3_s2) at (3.2, 0, 0)

        # Select sparse, non-sorted subset: indices 8, 3, 9, 2, 6
        # This ensures: gaps (missing 0, 1, 4, 5, 7), non-sorted order
        sparse_indices = [8, 3, 9, 2, 6]

        expected_positions = [
            (3.1, 0, 0),  # index 8: b3_s1
            (1.2, 0, 0),  # index 3: b1_s2
            (3.2, 0, 0),  # index 9: b3_s2
            (1.1, 0, 0),  # index 2: b1_s1
            (2.2, 0, 0),  # index 6: b2_s2
        ]

        sensor = SensorFrameTransform(model, shapes=sparse_indices, reference_sites=[ref_site])

        sensor.update(state)
        transforms = sensor.transforms.numpy()

        # Verify each transform corresponds to the correct site
        self.assertEqual(len(transforms), len(sparse_indices))

        for i, expected_pos in enumerate(expected_positions):
            pos = wp.transform_get_translation(wp.transform(*transforms[i]))
            np.testing.assert_allclose(
                [pos[0], pos[1], pos[2]],
                expected_pos,
                atol=1e-5,
                err_msg=f"Site {sparse_indices[i]} at index {i} has incorrect position",
            )

    def test_sensor_string_patterns(self):
        """Test SensorFrameTransform accepts string patterns."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, key="target_a")
        builder.add_site(body, key="target_b")
        builder.add_site(body, key="ref")
        model = builder.finalize()

        sensor = SensorFrameTransform(model, shapes="target_*", reference_sites="ref")
        self.assertEqual(len(sensor.transforms), 2)

    def test_sensor_no_match_raises(self):
        """Test SensorFrameTransform raises when no labels match."""
        builder = newton.ModelBuilder()
        body = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_site(body, key="site")
        ref = builder.add_site(body, key="ref")
        model = builder.finalize()

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes="nonexistent_*", reference_sites=[ref])

        with self.assertRaises(ValueError):
            SensorFrameTransform(model, shapes="site", reference_sites="nonexistent_*")

    def test_sensor_string_matches_int_indices(self):
        """Test that string-resolved indices produce same results as int indices."""
        builder = newton.ModelBuilder()
        body = builder.add_body(
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            xform=wp.transform(wp.vec3(1, 0, 0), wp.quat_identity()),
        )
        target = builder.add_site(body, key="target")
        ref = builder.add_site(body, key="ref")
        model = builder.finalize()

        state = model.state()
        eval_fk(model, state.joint_q, state.joint_qd, state)

        sensor_int = SensorFrameTransform(model, shapes=[target], reference_sites=[ref])
        sensor_int.update(state)

        sensor_str = SensorFrameTransform(model, shapes="target", reference_sites="ref")
        sensor_str.update(state)

        np.testing.assert_array_equal(
            sensor_int.transforms.numpy(),
            sensor_str.transforms.numpy(),
        )


if __name__ == "__main__":
    unittest.main()
