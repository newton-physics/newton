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

"""
Custom attributes tests for ModelBuilder kwargs functionality.

Tests the ability to add custom attributes via **kwargs to ModelBuilder
add_* functions (add_body, add_shape, add_joint, etc.).
"""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import Model, ModelBuilder
from newton._src.usd import utils as usd_utils
from newton._src.utils.import_utils import parse_custom_attributes
from newton._src.utils.selection import ArticulationView

AttributeAssignment = Model.AttributeAssignment
AttributeFrequency = Model.AttributeFrequency


class TestCustomAttributes(unittest.TestCase):
    """Test custom attributes functionality via ModelBuilder kwargs."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = wp.get_device()

    def _add_test_robot(self, builder: ModelBuilder) -> dict[str, int]:
        """Build a simple 2-bar linkage robot without custom attributes."""
        base = builder.add_link(xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()), mass=1.0)
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)

        link1 = builder.add_link(xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()), mass=0.5)
        builder.add_shape_capsule(link1, radius=0.05, half_height=0.2)

        joint1 = builder.add_joint_revolute(
            parent=base,
            child=link1,
            parent_xform=wp.transform([0.0, 0.0, 0.1], wp.quat_identity()),
            child_xform=wp.transform([0.0, 0.0, -0.2], wp.quat_identity()),
            axis=[0.0, 1.0, 0.0],
        )

        link2 = builder.add_link(xform=wp.transform([0.0, 0.0, 0.9], wp.quat_identity()), mass=0.3)
        builder.add_shape_capsule(link2, radius=0.03, half_height=0.15)

        joint2 = builder.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform([0.0, 0.0, 0.2], wp.quat_identity()),
            child_xform=wp.transform([0.0, 0.0, -0.15], wp.quat_identity()),
            axis=[0.0, 1.0, 0.0],
        )

        # Add articulation for the joints
        builder.add_articulation([joint1, joint2])

        return {"base": base, "link1": link1, "link2": link2, "joint1": joint1, "joint2": joint2}

    def test_body_custom_attributes(self):
        """Test BODY frequency custom attributes with multiple data types and assignments."""
        builder = ModelBuilder()

        # Declare MODEL assignment attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                "custom_float",
                wp.float32,
                AttributeFrequency.BODY,
                AttributeAssignment.MODEL,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int",
                frequency=AttributeFrequency.BODY,
                dtype=wp.int32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                "custom_bool",
                frequency=AttributeFrequency.BODY,
                dtype=wp.bool,
                assignment=AttributeAssignment.MODEL,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                "custom_vec3",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec3,
                assignment=AttributeAssignment.MODEL,
            )
        )

        # Declare STATE assignment attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="velocity_limit",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec3,
                assignment=AttributeAssignment.STATE,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="is_active",
                frequency=AttributeFrequency.BODY,
                dtype=wp.bool,
                assignment=AttributeAssignment.STATE,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="energy",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.STATE,
            )
        )

        # Declare CONTROL assignment attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="gain",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.CONTROL,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="mode",
                frequency=AttributeFrequency.BODY,
                dtype=wp.int32,
                assignment=AttributeAssignment.CONTROL,
            )
        )

        robot_entities = self._add_test_robot(builder)

        body1 = builder.add_body(
            mass=1.0,
            custom_attributes={
                "custom_float": 25.5,
                "custom_int": 42,
                "custom_bool": True,
                "custom_vec3": [1.0, 0.5, 0.0],
                "velocity_limit": [2.0, 2.0, 2.0],
                "is_active": True,
                "energy": 100.5,
                "gain": 1.5,
                "mode": 3,
            },
        )

        body2 = builder.add_body(
            mass=2.0,
            custom_attributes={
                "custom_float": 30.0,
                "custom_int": 7,
                "custom_bool": False,
                "custom_vec3": [0.0, 1.0, 0.5],
                "velocity_limit": [3.0, 3.0, 3.0],
                "is_active": False,
                "energy": 200.0,
                "gain": 2.0,
                "mode": 5,
            },
        )

        model = builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        # Verify MODEL attributes
        float_numpy = model.custom_float.numpy()
        self.assertAlmostEqual(float_numpy[body1], 25.5, places=5)
        self.assertAlmostEqual(float_numpy[body2], 30.0, places=5)

        int_numpy = model.custom_int.numpy()
        self.assertEqual(int_numpy[body1], 42)
        self.assertEqual(int_numpy[body2], 7)

        bool_numpy = model.custom_bool.numpy()
        self.assertEqual(bool_numpy[body1], 1)
        self.assertEqual(bool_numpy[body2], 0)

        vec3_numpy = model.custom_vec3.numpy()
        np.testing.assert_array_almost_equal(vec3_numpy[body1], [1.0, 0.5, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(vec3_numpy[body2], [0.0, 1.0, 0.5], decimal=5)

        # Verify STATE attributes
        velocity_limit_numpy = state.velocity_limit.numpy()
        np.testing.assert_array_almost_equal(velocity_limit_numpy[body1], [2.0, 2.0, 2.0], decimal=5)
        np.testing.assert_array_almost_equal(velocity_limit_numpy[body2], [3.0, 3.0, 3.0], decimal=5)

        is_active_numpy = state.is_active.numpy()
        self.assertEqual(is_active_numpy[body1], 1)
        self.assertEqual(is_active_numpy[body2], 0)

        energy_numpy = state.energy.numpy()
        self.assertAlmostEqual(energy_numpy[body1], 100.5, places=5)
        self.assertAlmostEqual(energy_numpy[body2], 200.0, places=5)

        # Verify CONTROL attributes
        gain_numpy = control.gain.numpy()
        self.assertAlmostEqual(gain_numpy[body1], 1.5, places=5)
        self.assertAlmostEqual(gain_numpy[body2], 2.0, places=5)

        mode_numpy = control.mode.numpy()
        self.assertEqual(mode_numpy[body1], 3)
        self.assertEqual(mode_numpy[body2], 5)

        # Verify default values on robot entities (should be zeros for all assignments)
        self.assertAlmostEqual(float_numpy[robot_entities["base"]], 0.0, places=5)
        self.assertEqual(int_numpy[robot_entities["link1"]], 0)
        self.assertEqual(bool_numpy[robot_entities["link2"]], 0)
        np.testing.assert_array_almost_equal(velocity_limit_numpy[robot_entities["base"]], [0.0, 0.0, 0.0], decimal=5)
        self.assertEqual(is_active_numpy[robot_entities["link1"]], 0)
        self.assertAlmostEqual(energy_numpy[robot_entities["link2"]], 0.0, places=5)
        self.assertAlmostEqual(gain_numpy[robot_entities["base"]], 0.0, places=5)
        self.assertEqual(mode_numpy[robot_entities["link1"]], 0)

    def test_shape_custom_attributes(self):
        """Test SHAPE frequency custom attributes with multiple data types."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.int32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_bool",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.bool,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec2",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.vec2,
            )
        )

        robot_entities = self._add_test_robot(builder)

        shape1 = builder.add_shape_box(
            body=robot_entities["base"],
            hx=0.05,
            hy=0.05,
            hz=0.05,
            custom_attributes={
                "custom_float": 0.8,
                "custom_int": 3,
                "custom_bool": False,
                "custom_vec2": [0.2, 0.4],
            },
        )

        shape2 = builder.add_shape_sphere(
            body=robot_entities["link1"],
            radius=0.02,
            custom_attributes={
                "custom_float": 0.3,
                "custom_int": 1,
                "custom_bool": True,
                "custom_vec2": [0.8, 0.6],
            },
        )

        model = builder.finalize(device=self.device)

        # Verify authored values
        float_numpy = model.custom_float.numpy()
        self.assertAlmostEqual(float_numpy[shape1], 0.8, places=5)
        self.assertAlmostEqual(float_numpy[shape2], 0.3, places=5)

        int_numpy = model.custom_int.numpy()
        self.assertEqual(int_numpy[shape1], 3)
        self.assertEqual(int_numpy[shape2], 1)

        # Verify default values on robot shapes
        self.assertAlmostEqual(float_numpy[0], 0.0, places=5)
        self.assertEqual(int_numpy[1], 0)

    def test_joint_dof_coord_attributes(self):
        """Test JOINT_DOF and JOINT_COORD frequency attributes with list requirements."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.int32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_coord",
                frequency=AttributeFrequency.JOINT_COORD,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int_coord",
                frequency=AttributeFrequency.JOINT_COORD,
                dtype=wp.int32,
            )
        )

        robot_entities = self._add_test_robot(builder)

        body = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_revolute(
            parent=robot_entities["link2"],
            child=body,
            axis=[0.0, 0.0, 1.0],
            custom_attributes={
                "custom_float_dof": [0.05],
                "custom_int_dof": [15],
                "custom_float_coord": [0.5],
                "custom_int_coord": [12],
            },
        )
        builder.add_articulation([joint3])

        model = builder.finalize(device=self.device)

        # Verify DOF attributes
        dof_float_numpy = model.custom_float_dof.numpy()
        self.assertAlmostEqual(dof_float_numpy[2], 0.05, places=5)
        self.assertAlmostEqual(dof_float_numpy[0], 0.0, places=5)

        dof_int_numpy = model.custom_int_dof.numpy()
        self.assertEqual(dof_int_numpy[2], 15)
        self.assertEqual(dof_int_numpy[1], 0)

        # Verify coordinate attributes
        coord_float_numpy = model.custom_float_coord.numpy()
        self.assertAlmostEqual(coord_float_numpy[2], 0.5, places=5)
        self.assertAlmostEqual(coord_float_numpy[0], 0.0, places=5)

        coord_int_numpy = model.custom_int_coord.numpy()
        self.assertEqual(coord_int_numpy[2], 12)
        self.assertEqual(coord_int_numpy[1], 0)

    def test_joint_constraint_attributes(self):
        """Test JOINT_CONSTRAINT frequency attributes with list requirements."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.int32,
            )
        )

        robot_entities = self._add_test_robot(builder)

        body = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_revolute(
            parent=robot_entities["link2"],
            child=body,
            axis=[0.0, 0.0, 1.0],
            custom_attributes={
                "custom_float_cts": [0.01, 0.02, 0.03, 0.04, 0.05],
                "custom_int_cts": [1, 2, 3, 4, 5],
            },
        )
        builder.add_articulation([joint3])

        model = builder.finalize(device=self.device)

        # Verify constraint attributes
        cts_float_numpy = model.custom_float_cts.numpy()
        self.assertEqual(len(cts_float_numpy), 15)  # 10 from previous joints + 5 from this joint
        np.testing.assert_allclose(cts_float_numpy[0:10], np.zeros(10, dtype=np.float32))
        np.testing.assert_allclose(cts_float_numpy[10:15], np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32))

        cts_int_numpy = model.custom_int_cts.numpy()
        self.assertEqual(len(cts_int_numpy), 15)  # 10 from previous joints + 5 from this joint
        np.testing.assert_allclose(cts_int_numpy[0:10], np.zeros(10, dtype=np.int32))
        np.testing.assert_allclose(cts_int_numpy[10:15], np.array([1, 2, 3, 4, 5], dtype=np.int32))

    def test_multi_dof_joint_individual_values(self):
        """Test D6 joint with individual values per DOF and coordinate."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int_coord",
                frequency=AttributeFrequency.JOINT_COORD,
                dtype=wp.int32,
            )
        )

        robot_entities = self._add_test_robot(builder)
        cfg = ModelBuilder.JointDofConfig

        body = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                "custom_float_dof": [0.1, 0.2, 0.3],
                "custom_int_coord": [100, 200, 300],
            },
        )
        builder.add_articulation([joint3])

        model = builder.finalize(device=self.device)

        # Verify individual DOF values
        dof_float_numpy = model.custom_float_dof.numpy()
        self.assertAlmostEqual(dof_float_numpy[2], 0.1, places=5)
        self.assertAlmostEqual(dof_float_numpy[3], 0.2, places=5)
        self.assertAlmostEqual(dof_float_numpy[4], 0.3, places=5)
        self.assertAlmostEqual(dof_float_numpy[0], 0.0, places=5)

        # Verify individual coordinate values
        coord_int_numpy = model.custom_int_coord.numpy()
        self.assertEqual(coord_int_numpy[2], 100)
        self.assertEqual(coord_int_numpy[3], 200)
        self.assertEqual(coord_int_numpy[4], 300)
        self.assertEqual(coord_int_numpy[1], 0)

    def test_multi_dof_joint_constraint_individual_values(self):
        """Test D6 joint with individual values per constraint."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_int_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.int32,
            )
        )

        robot_entities = self._add_test_robot(builder)
        cfg = ModelBuilder.JointDofConfig

        body = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                "custom_float_cts": [0.01, 0.02, 0.03],
                "custom_int_cts": [1, 2, 3],
            },
        )
        builder.add_articulation([joint3])

        model = builder.finalize(device=self.device)

        # Verify constraint attributes
        cts_float_numpy = model.custom_float_cts.numpy()
        self.assertEqual(len(cts_float_numpy), 13)  # 10 from previous joints + 3 from this joint
        np.testing.assert_allclose(cts_float_numpy[0:10], np.zeros(10, dtype=np.float32))
        np.testing.assert_allclose(cts_float_numpy[10:13], np.array([0.01, 0.02, 0.03], dtype=np.float32))

        cts_int_numpy = model.custom_int_cts.numpy()
        self.assertEqual(len(cts_int_numpy), 13)  # 10 from previous joints + 3 from this joint
        np.testing.assert_allclose(cts_int_numpy[0:10], np.zeros(10, dtype=np.int32))
        np.testing.assert_allclose(cts_int_numpy[10:13], np.array([1, 2, 3], dtype=np.int32))

    def test_multi_dof_joint_vector_attributes(self):
        """Test D6 joint with vector attributes (list of lists)."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec2_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.vec2,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec3_coord",
                frequency=AttributeFrequency.JOINT_COORD,
                dtype=wp.vec3,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec3_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.vec3,
            )
        )

        robot_entities = self._add_test_robot(builder)
        cfg = ModelBuilder.JointDofConfig

        body = builder.add_link(mass=1.0)
        joint3 = builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                "custom_vec2_dof": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "custom_vec3_coord": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                "custom_vec3_cts": [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]],
            },
        )
        builder.add_articulation([joint3])

        model = builder.finalize(device=self.device)

        # Verify DOF vector values
        dof_vec2_numpy = model.custom_vec2_dof.numpy()
        np.testing.assert_array_almost_equal(dof_vec2_numpy[2], [1.0, 2.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[3], [3.0, 4.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[4], [5.0, 6.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[0], [0.0, 0.0], decimal=5)

        # Verify coordinate vector values
        coord_vec3_numpy = model.custom_vec3_coord.numpy()
        np.testing.assert_array_almost_equal(coord_vec3_numpy[2], [0.1, 0.2, 0.3], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[3], [0.4, 0.5, 0.6], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[4], [0.7, 0.8, 0.9], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[1], [0.0, 0.0, 0.0], decimal=5)

        # Verify constraint vector values
        cts_vec3_numpy = model.custom_vec3_cts.numpy()
        self.assertEqual(len(cts_vec3_numpy), 13)  # 10 from previous joints + 3 from this joint
        np.testing.assert_allclose(cts_vec3_numpy[0:10], np.zeros((10, 3), dtype=np.float32))
        np.testing.assert_array_almost_equal(cts_vec3_numpy[10], [0.01, 0.02, 0.03], decimal=5)
        np.testing.assert_array_almost_equal(cts_vec3_numpy[11], [0.04, 0.05, 0.06], decimal=5)
        np.testing.assert_array_almost_equal(cts_vec3_numpy[12], [0.07, 0.08, 0.09], decimal=5)

    def test_dof_coord_cts_list_requirements(self):
        """Test that DOF and coordinate attributes must be lists with correct lengths."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_coord",
                frequency=AttributeFrequency.JOINT_COORD,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float_cts",
                frequency=AttributeFrequency.JOINT_CONSTRAINT,
                dtype=wp.float32,
            )
        )

        robot_entities = self._add_test_robot(builder)
        cfg = ModelBuilder.JointDofConfig

        # Test wrong DOF list length (value error)
        body2 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_d6(
                parent=robot_entities["link2"],
                child=body2,
                linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
                angular_axes=[cfg(axis=[0, 0, 1])],
                custom_attributes={"custom_float_dof": [0.1, 0.2]},  # 2 values for 3-DOF joint
            )

        # Test wrong coordinate list length (value error) - wrong number of values
        body3 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_d6(
                parent=robot_entities["link2"],
                child=body3,
                linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
                angular_axes=[cfg(axis=[0, 0, 1])],
                custom_attributes={"custom_float_coord": [0.1, 0.2]},  # 2 values for 3-coord joint
            )

        # Test scalar broadcast for multi-coord joint (should succeed, not raise)
        body3b = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body3b,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={"custom_float_coord": 0.5},  # Scalar broadcast to all coords
        )

        # Test wrong constraint list length (value error)
        body4 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_d6(
                parent=robot_entities["link2"],
                child=body4,
                linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
                angular_axes=[cfg(axis=[0, 0, 1])],
                custom_attributes={"custom_float_cts": [0.1, 0.2]},  # 2 values for 3-constraint joint
            )

        # Test scalar broadcast for multi-constraint joint (should succeed, not raise)
        body5 = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body5,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={"custom_float_cts": 0.5},  # Scalar broadcast to all constraints
        )

    def test_vector_type_inference(self):
        """Test automatic dtype inference for vector types."""
        builder = ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec2",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec2,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec3",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec3,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_vec4",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec4,
            )
        )

        body = builder.add_body(
            mass=1.0,
            custom_attributes={
                "custom_vec2": [1.0, 2.0],
                "custom_vec3": [1.0, 2.0, 3.0],
                "custom_vec4": [1.0, 2.0, 3.0, 4.0],
            },
        )

        custom_attrs = builder.custom_attributes
        self.assertEqual(custom_attrs["custom_vec2"].dtype, wp.vec2)
        self.assertEqual(custom_attrs["custom_vec3"].dtype, wp.vec3)
        self.assertEqual(custom_attrs["custom_vec4"].dtype, wp.vec4)

        model = builder.finalize(device=self.device)

        vec2_numpy = model.custom_vec2.numpy()
        np.testing.assert_array_almost_equal(vec2_numpy[body], [1.0, 2.0])

        vec3_numpy = model.custom_vec3.numpy()
        np.testing.assert_array_almost_equal(vec3_numpy[body], [1.0, 2.0, 3.0])

    def test_string_attributes_handling(self):
        """Test that undeclared attributes and incorrect frequency/assignment are rejected."""
        builder = ModelBuilder()
        robot_entities = self._add_test_robot(builder)

        # Test 1: Undeclared string attribute should raise AttributeError
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
            )
        )

        with self.assertRaises(AttributeError):
            builder.add_body(
                mass=1.0,
                custom_attributes={"custom_string": "test_body", "custom_float": 25.0},
            )

        # But using only declared attribute should work
        builder.add_body(mass=1.0, custom_attributes={"custom_float": 25.0})

        custom_attrs = builder.custom_attributes
        self.assertIn("custom_float", custom_attrs)
        self.assertNotIn("custom_string", custom_attrs)

        # Test 2: Attribute with wrong frequency should raise ValueError
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_only_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
            )
        )

        # Trying to use BODY frequency attribute on a shape should fail
        with self.assertRaises(ValueError) as context:
            builder.add_shape_box(
                body=robot_entities["base"],
                hx=0.1,
                hy=0.1,
                hz=0.1,
                custom_attributes={"body_only_attr": 1.0},
            )
        self.assertIn("frequency", str(context.exception).lower())

        # Test 3: Using SHAPE frequency attribute on a body should fail
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="shape_only_attr",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.float32,
            )
        )

        with self.assertRaises(ValueError) as context:
            builder.add_body(mass=1.0, custom_attributes={"shape_only_attr": 2.0})
        self.assertIn("frequency", str(context.exception).lower())

        # Test 4: Using attributes with correct frequency should work
        builder.add_body(mass=1.0, custom_attributes={"body_only_attr": 1.5})
        builder.add_shape_box(
            body=robot_entities["base"],
            hx=0.1,
            hy=0.1,
            hz=0.1,
            custom_attributes={"shape_only_attr": 2.5},
        )

        # Verify attributes were created with correct assignments
        self.assertEqual(custom_attrs["custom_float"].assignment, AttributeAssignment.MODEL)
        self.assertEqual(custom_attrs["body_only_attr"].assignment, AttributeAssignment.MODEL)

        model = builder.finalize(device=self.device)
        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(model, "custom_string"))

    def test_assignment_types(self):
        """Test custom attribute assignment to MODEL objects."""
        builder = ModelBuilder()

        # Declare custom attribute before use
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_float",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
            )
        )

        builder.add_body(mass=1.0, custom_attributes={"custom_float": 25.0})

        custom_attrs = builder.custom_attributes
        self.assertEqual(custom_attrs["custom_float"].assignment, AttributeAssignment.MODEL)

        model = builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(state, "custom_float"))
        self.assertFalse(hasattr(control, "custom_float"))

    def test_value_dtype_compatibility(self):
        """Test that values work correctly with declared dtypes."""
        builder = ModelBuilder()

        # Declare attributes with different dtypes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="scalar_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="vec3_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.vec3,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="int_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.int32,
            )
        )

        # Create bodies with appropriate values
        body = builder.add_body(
            mass=1.0,
            custom_attributes={
                "scalar_attr": 42.5,
                "vec3_attr": [1.0, 2.0, 3.0],
                "int_attr": 7,
            },
        )

        # Verify values are stored and converted correctly by Warp
        model = builder.finalize(device=self.device)
        scalar_val = model.scalar_attr.numpy()
        vec3_val = model.vec3_attr.numpy()
        int_val = model.int_attr.numpy()

        self.assertAlmostEqual(scalar_val[body], 42.5, places=5)
        np.testing.assert_array_almost_equal(vec3_val[body], [1.0, 2.0, 3.0], decimal=5)
        self.assertEqual(int_val[body], 7)

    def test_custom_attributes_with_multi_builders(self):
        """Test that custom attributes are preserved when using add_world()."""
        # Create a sub-builder with custom attributes
        sub_builder = ModelBuilder()

        # Declare attributes with different frequencies and assignments
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="robot_id",
                frequency=AttributeFrequency.BODY,
                dtype=wp.int32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="temperature",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.STATE,
            )
        )
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="shape_color",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.vec3,
                assignment=AttributeAssignment.MODEL,
            )
        )
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="gain_dof",
                frequency=AttributeFrequency.JOINT_DOF,
                dtype=wp.float32,
                assignment=AttributeAssignment.CONTROL,
            )
        )

        # Create a simple robot in sub-builder
        body1 = sub_builder.add_link(
            mass=1.0,
            custom_attributes={"robot_id": 100, "temperature": 37.5},
        )
        sub_builder.add_shape_sphere(body1, radius=0.1, custom_attributes={"shape_color": [1.0, 0.0, 0.0]})

        body2 = sub_builder.add_link(
            mass=0.5,
            custom_attributes={"robot_id": 200, "temperature": 38.0},
        )
        sub_builder.add_shape_box(
            body2,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            custom_attributes={"shape_color": [0.0, 1.0, 0.0]},
        )

        sub_joint = sub_builder.add_joint_revolute(
            parent=body1,
            child=body2,
            axis=[0, 0, 1],
            custom_attributes={"gain_dof": [1.5]},
        )
        sub_builder.add_articulation([sub_joint])

        # Create main builder and add sub-builder multiple times
        main_builder = ModelBuilder()

        # Add some entities to the main builder, so the custom attribute
        # values added through the sub builder will need to be merged
        # and their indices need to be adjusted.
        body3 = main_builder.add_link(mass=1.0)
        body4 = main_builder.add_link(mass=1.0)
        main_builder.add_shape_sphere(body3, radius=0.1)
        main_builder.add_shape_sphere(body4, radius=0.1)
        main_joint = main_builder.add_joint_revolute(parent=body3, child=body4, axis=[0, 0, 1])
        main_builder.add_articulation([main_joint])

        # Add first instance
        main_builder.add_world(sub_builder)  # World 0
        # Add second instance
        main_builder.add_world(sub_builder)  # World 1

        # Verify custom attributes were merged
        self.assertIn("robot_id", main_builder.custom_attributes)
        self.assertIn("temperature", main_builder.custom_attributes)
        self.assertIn("shape_color", main_builder.custom_attributes)
        self.assertIn("gain_dof", main_builder.custom_attributes)

        # Verify frequencies and assignments
        self.assertEqual(main_builder.custom_attributes["robot_id"].frequency, AttributeFrequency.BODY)
        self.assertEqual(main_builder.custom_attributes["robot_id"].assignment, AttributeAssignment.MODEL)
        self.assertEqual(main_builder.custom_attributes["temperature"].assignment, AttributeAssignment.STATE)
        self.assertEqual(main_builder.custom_attributes["shape_color"].frequency, AttributeFrequency.SHAPE)
        self.assertEqual(main_builder.custom_attributes["gain_dof"].frequency, AttributeFrequency.JOINT_DOF)

        # Build model and verify values
        model = main_builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        # Verify BODY attributes (2 bodies per instance, 2 instances = 4 bodies total)
        robot_ids = model.robot_id.numpy()
        temperatures = state.temperature.numpy()

        # Verify BODY attributes
        np.testing.assert_array_almost_equal(robot_ids, [0, 0, 100, 200, 100, 200], decimal=5)
        np.testing.assert_array_almost_equal(temperatures, [0.0, 0.0, 37.5, 38.0, 37.5, 38.0], decimal=5)

        # Verify SHAPE attributes
        shape_colors = model.shape_color.numpy()

        np.testing.assert_array_almost_equal(shape_colors[0], [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[1], [0.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[2], [1.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[3], [0.0, 1.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[4], [1.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[5], [0.0, 1.0, 0.0], decimal=5)

        # Verify JOINT_DOF attributes
        dof_gains = control.gain_dof.numpy()

        np.testing.assert_array_almost_equal(dof_gains, [0.0, 1.5, 1.5], decimal=5)

    def test_namespaced_attributes(self):
        """Test namespaced custom attributes with hierarchical organization."""
        builder = ModelBuilder()

        # Declare attributes in different namespaces
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="damping",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="enable_ccd",
                frequency=AttributeFrequency.BODY,
                dtype=wp.bool,
                assignment=AttributeAssignment.STATE,
                namespace="physx",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="custom_id",
                frequency=AttributeFrequency.SHAPE,
                dtype=wp.int32,
                assignment=AttributeAssignment.MODEL,
                namespace="mujoco",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="temperature",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )

        robot_entities = self._add_test_robot(builder)

        # Create bodies with namespaced attributes
        body1 = builder.add_body(
            mass=1.0,
            custom_attributes={
                "mujoco:damping": 0.1,
                "physx:enable_ccd": True,
                "temperature": 37.5,
            },
        )

        body2 = builder.add_body(
            mass=2.0,
            custom_attributes={
                "mujoco:damping": 0.2,
                "physx:enable_ccd": False,
                "temperature": 40.0,
            },
        )

        # Create shapes with namespaced attributes
        shape1 = builder.add_shape_box(
            body=body1,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            custom_attributes={"mujoco:custom_id": 100},
        )

        shape2 = builder.add_shape_sphere(
            body=body2,
            radius=0.05,
            custom_attributes={"mujoco:custom_id": 200},
        )

        model = builder.finalize(device=self.device)
        state = model.state()

        # Verify namespaced attributes exist on correct objects
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(state, "physx"))
        self.assertTrue(hasattr(model, "temperature"))  # default namespace

        # Verify mujoco namespace attributes
        mujoco_damping = model.mujoco.damping.numpy()
        self.assertAlmostEqual(mujoco_damping[body1], 0.1, places=5)
        self.assertAlmostEqual(mujoco_damping[body2], 0.2, places=5)
        self.assertAlmostEqual(mujoco_damping[robot_entities["base"]], 0.0, places=5)  # default value

        mujoco_custom_id = model.mujoco.custom_id.numpy()
        self.assertEqual(mujoco_custom_id[shape1], 100)
        self.assertEqual(mujoco_custom_id[shape2], 200)

        # Verify physx namespace attributes
        physx_enable_ccd = state.physx.enable_ccd.numpy()
        self.assertEqual(physx_enable_ccd[body1], 1)  # True
        self.assertEqual(physx_enable_ccd[body2], 0)  # False
        self.assertEqual(physx_enable_ccd[robot_entities["link1"]], 0)  # default False

        # Verify default namespace attribute
        temperatures = model.temperature.numpy()
        self.assertAlmostEqual(temperatures[body1], 37.5, places=5)
        self.assertAlmostEqual(temperatures[body2], 40.0, places=5)

    def test_attribute_uniqueness_constraints(self):
        """Test uniqueness constraints for custom attributes based on full identifier (namespace:name)."""

        # Test 1: Same name in different namespaces with different assignments - SHOULD WORK
        # Key "float_attr" vs "namespace_a:float_attr" are different
        builder1 = ModelBuilder()
        builder1.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        builder1.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.STATE,
                namespace="namespace_a",
            )
        )
        # Should work - different full keys
        body = builder1.add_body(
            mass=1.0,
            custom_attributes={
                "float_attr": 1.0,  # MODEL
                "namespace_a:float_attr": 2.0,  # STATE, namespaced
            },
        )
        model1 = builder1.finalize(device=self.device)
        state1 = model1.state()

        self.assertAlmostEqual(model1.float_attr.numpy()[body], 1.0, places=5)
        self.assertAlmostEqual(state1.namespace_a.float_attr.numpy()[body], 2.0, places=5)

        # Test 2: Same name (no namespace) with different assignments - SHOULD FAIL
        # Both use key "float_attr"
        builder2 = ModelBuilder()
        builder2.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        with self.assertRaises(ValueError) as context:
            builder2.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="float_attr",
                    frequency=AttributeFrequency.BODY,
                    dtype=wp.float32,
                    assignment=AttributeAssignment.STATE,
                )
            )
        self.assertIn("already exists", str(context.exception))
        self.assertIn("incompatible spec", str(context.exception))

        # Test 3: Same namespace:name with different assignments - SHOULD FAIL
        # Both use key "namespace_a:float_attr"
        builder3 = ModelBuilder()
        builder3.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                namespace="namespace_a",
            )
        )
        with self.assertRaises(ValueError) as context:
            builder3.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="float_attr",
                    frequency=AttributeFrequency.BODY,
                    dtype=wp.float32,
                    assignment=AttributeAssignment.STATE,
                    namespace="namespace_a",
                )
            )
        self.assertIn("already exists", str(context.exception))

        # Test 4: Same name in different namespaces with same assignment - SHOULD WORK
        # Keys "namespace_a:float_attr" and "namespace_b:float_attr" are different
        builder4 = ModelBuilder()
        builder4.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                "float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                namespace="namespace_a",
            )
        )
        builder4.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                "float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                namespace="namespace_b",
            )
        )
        # Should work - different namespaces create different keys
        body = builder4.add_body(
            mass=1.0,
            custom_attributes={
                "namespace_a:float_attr": 10.0,
                "namespace_b:float_attr": 20.0,
            },
        )
        model4 = builder4.finalize(device=self.device)

        self.assertAlmostEqual(model4.namespace_a.float_attr.numpy()[body], 10.0, places=5)
        self.assertAlmostEqual(model4.namespace_b.float_attr.numpy()[body], 20.0, places=5)

        # Test 5: Idempotent declaration - declaring same attribute twice with identical params - SHOULD WORK
        builder5 = ModelBuilder()
        builder5.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        # Declaring again with same parameters should be allowed
        builder5.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        # Should still work
        self.assertEqual(len(builder5.custom_attributes), 1)

        # Test 6: Same key with different frequency - SHOULD FAIL
        builder6 = ModelBuilder()
        builder6.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="float_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
            )
        )
        with self.assertRaises(ValueError) as context:
            builder6.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="float_attr",
                    frequency=AttributeFrequency.SHAPE,
                    dtype=wp.float32,
                    assignment=AttributeAssignment.MODEL,
                )
            )
        self.assertIn("already exists", str(context.exception))
        self.assertIn("incompatible spec", str(context.exception))

        # Test 7: Same key with different references - SHOULD FAIL
        builder7 = ModelBuilder()
        # Register custom frequency before adding attributes
        builder7.add_custom_frequency(ModelBuilder.CustomFrequency(name="item", namespace="test"))
        builder7.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ref_attr",
                frequency="test:item",
                dtype=wp.int32,
                namespace="test",
                references="body",
            )
        )
        with self.assertRaises(ValueError) as context:
            builder7.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="ref_attr",
                    frequency="test:item",
                    dtype=wp.int32,
                    namespace="test",
                    references="shape",  # Different references
                )
            )
        self.assertIn("already exists", str(context.exception))
        self.assertIn("incompatible spec", str(context.exception))

    def test_mixed_free_and_articulated_bodies(self):
        """Test BODY and ARTICULATION frequency custom attributes with mixed free and articulated bodies."""
        builder = ModelBuilder()

        # Declare custom attributes
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="temperature",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                default=20.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="density",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=AttributeAssignment.STATE,
                default=1.0,
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="articulation_stiffness",
                frequency=AttributeFrequency.ARTICULATION,
                dtype=wp.float32,
                assignment=AttributeAssignment.MODEL,
                default=100.0,
            )
        )

        # Create free bodies (no articulation)
        free_body_ids = []
        for i in range(3):
            body = builder.add_link(
                xform=wp.transform([float(i), 0.0, 0.0], wp.quat_identity()),
                mass=1.0,
                custom_attributes={
                    "temperature": 25.0 + float(i) * 5.0,
                    "density": 0.5 + float(i) * 0.1,
                }
                if i > 0
                else None,
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            free_body_ids.append(body)

        # Create articulations with bodies and joints
        arctic_body_ids = []
        for i in range(2):
            # Create 2-link articulation
            # Temperature NOT assigned to articulated bodies (use defaults)
            # Density assigned with different values than free bodies
            base = builder.add_link(
                xform=wp.transform([3.0 + float(i), 0.0, 0.0], wp.quat_identity()),
                mass=1.0,
                custom_attributes={"density": 2.0 + float(i) * 0.5},
            )
            builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)

            link = builder.add_link(
                xform=wp.transform([3.0 + float(i), 0.0, 0.5], wp.quat_identity()),
                mass=0.5,
                custom_attributes={"density": 3.0 + float(i) * 0.5},
            )
            builder.add_shape_capsule(link, radius=0.05, half_height=0.2)

            # Connect base to world with a free joint
            j_base = builder.add_joint_free(child=base)
            j_revolute = builder.add_joint_revolute(
                parent=base,
                child=link,
                parent_xform=wp.transform([0.0, 0.0, 0.1], wp.quat_identity()),
                child_xform=wp.transform([0.0, 0.0, -0.2], wp.quat_identity()),
                axis=[0.0, 1.0, 0.0],
            )

            # Create articulation from joints
            builder.add_articulation(
                [j_base, j_revolute],
                custom_attributes={
                    "articulation_stiffness": 100.0 + float(i) * 50.0,
                },
            )
            arctic_body_ids.extend([base, link])

        # Finalize and verify
        model = builder.finalize(device=self.device)
        state = model.state()

        # Check temperature attribute (MODEL assignment)
        temps = model.temperature.numpy()

        # Free bodies: first uses default, rest use custom values
        self.assertAlmostEqual(temps[free_body_ids[0]], 20.0, places=5)  # Default
        self.assertAlmostEqual(temps[free_body_ids[1]], 30.0, places=5)  # Custom
        self.assertAlmostEqual(temps[free_body_ids[2]], 35.0, places=5)  # Custom

        # Articulated bodies: all use default (temperature not assigned)
        self.assertAlmostEqual(temps[arctic_body_ids[0]], 20.0, places=5)  # arctic1 base - default
        self.assertAlmostEqual(temps[arctic_body_ids[1]], 20.0, places=5)  # arctic1 link - default
        self.assertAlmostEqual(temps[arctic_body_ids[2]], 20.0, places=5)  # arctic2 base - default
        self.assertAlmostEqual(temps[arctic_body_ids[3]], 20.0, places=5)  # arctic2 link - default

        # Check density attribute (STATE assignment)
        densities = state.density.numpy()

        # Free bodies: first uses default, rest use custom values (different from articulated)
        self.assertAlmostEqual(densities[free_body_ids[0]], 1.0, places=5)  # Default
        self.assertAlmostEqual(densities[free_body_ids[1]], 0.6, places=5)  # Custom (0.5 + 1*0.1)
        self.assertAlmostEqual(densities[free_body_ids[2]], 0.7, places=5)  # Custom (0.5 + 2*0.1)

        # Articulated bodies: all use custom values (different range from free bodies)
        self.assertAlmostEqual(densities[arctic_body_ids[0]], 2.0, places=5)  # arctic1 base
        self.assertAlmostEqual(densities[arctic_body_ids[1]], 3.0, places=5)  # arctic1 link
        self.assertAlmostEqual(densities[arctic_body_ids[2]], 2.5, places=5)  # arctic2 base
        self.assertAlmostEqual(densities[arctic_body_ids[3]], 3.5, places=5)  # arctic2 link

        # Check ARTICULATION attributes
        arctic_stiff = model.articulation_stiffness.numpy()
        self.assertEqual(len(arctic_stiff), 2)
        self.assertAlmostEqual(arctic_stiff[0], 100.0, places=5)
        self.assertAlmostEqual(arctic_stiff[1], 150.0, places=5)

    def test_usd_value_transformer_none_uses_default(self):
        """Test that USD transformers returning None leave attributes undefined."""
        builder = ModelBuilder()

        custom_attr = ModelBuilder.CustomAttribute(
            name="usd_default",
            frequency=AttributeFrequency.BODY,
            dtype=wp.float32,
            default=7.0,
            usd_value_transformer=lambda _value, _context: None,
        )
        builder.add_custom_attribute(custom_attr)

        class DummyUsdAttr:
            def __init__(self, value):
                self._value = value

            def HasAuthoredValue(self):
                return True

            def Get(self):
                return self._value

        class DummyPrim:
            def __init__(self, attributes):
                self._attributes = attributes

            def GetAttribute(self, name):
                return self._attributes.get(name)

        prim = DummyPrim({custom_attr.usd_attribute_name: DummyUsdAttr(123.0)})
        custom_attrs = usd_utils.get_custom_attribute_values(prim, [custom_attr])
        self.assertEqual(custom_attrs, {})

        body = builder.add_body(mass=1.0, custom_attributes=custom_attrs)
        model = builder.finalize(device=self.device)
        values = model.usd_default.numpy()
        self.assertAlmostEqual(values[body], 7.0, places=5)

    def test_mjcf_and_urdf_value_transformer_none_uses_default(self):
        """Test that MJCF/URDF transformers returning None leave attributes undefined."""
        builder = ModelBuilder()

        mjcf_attr = ModelBuilder.CustomAttribute(
            name="mjcf_default",
            frequency=AttributeFrequency.BODY,
            dtype=wp.float32,
            default=3.0,
            mjcf_value_transformer=lambda _value, _context: None,
        )
        urdf_attr = ModelBuilder.CustomAttribute(
            name="urdf_default",
            frequency=AttributeFrequency.BODY,
            dtype=wp.float32,
            default=5.0,
            urdf_value_transformer=lambda _value, _context: None,
        )
        builder.add_custom_attribute(mjcf_attr)
        builder.add_custom_attribute(urdf_attr)

        mjcf_values = parse_custom_attributes({mjcf_attr.mjcf_attribute_name: "1.23"}, [mjcf_attr], "mjcf")
        urdf_values = parse_custom_attributes({urdf_attr.urdf_attribute_name: "4.56"}, [urdf_attr], "urdf")
        self.assertEqual(mjcf_values, {})
        self.assertEqual(urdf_values, {})

        body = builder.add_body(
            mass=1.0,
            custom_attributes={**mjcf_values, **urdf_values},
        )
        model = builder.finalize(device=self.device)
        self.assertAlmostEqual(model.mjcf_default.numpy()[body], 3.0, places=5)
        self.assertAlmostEqual(model.urdf_default.numpy()[body], 5.0, places=5)


class TestCustomFrequencyAttributes(unittest.TestCase):
    """Test custom attributes with custom frequencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = wp.get_device()

    def test_custom_frequency_basic(self):
        """Test basic custom frequency attributes with add_custom_values()."""
        builder = ModelBuilder()

        # Register custom frequency before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="pair", namespace="test"))

        # Declare attributes with custom frequency
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_world",
                frequency="test:pair",
                dtype=wp.int32,
                default=0,
                namespace="test",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_value",
                frequency="test:pair",
                dtype=wp.float32,
                default=1.0,
                namespace="test",
            )
        )

        # Add values using add_custom_values()
        indices = builder.add_custom_values(
            **{
                "test:pair_world": 0,
                "test:pair_value": 10.5,
            }
        )
        self.assertEqual(indices["test:pair_world"], 0)
        self.assertEqual(indices["test:pair_value"], 0)

        indices = builder.add_custom_values(
            **{
                "test:pair_world": 0,
                "test:pair_value": 20.5,
            }
        )
        self.assertEqual(indices["test:pair_world"], 1)
        self.assertEqual(indices["test:pair_value"], 1)

        model = builder.finalize(device=self.device)

        # Verify values
        world_arr = model.test.pair_world.numpy()
        value_arr = model.test.pair_value.numpy()

        self.assertEqual(len(world_arr), 2)
        self.assertEqual(len(value_arr), 2)
        self.assertEqual(world_arr[0], 0)
        self.assertEqual(world_arr[1], 0)
        self.assertAlmostEqual(value_arr[0], 10.5, places=5)
        self.assertAlmostEqual(value_arr[1], 20.5, places=5)

        # Verify custom frequency count is stored
        self.assertEqual(model.get_custom_frequency_count("test:pair"), 2)

    def test_custom_frequency_requires_registration(self):
        """Test that using an unregistered custom frequency raises ValueError."""
        builder = ModelBuilder()

        # Try to add attribute with unregistered custom frequency - should fail
        with self.assertRaises(ValueError) as context:
            builder.add_custom_attribute(
                ModelBuilder.CustomAttribute(
                    name="unregistered_attr",
                    frequency="test:unregistered",
                    dtype=wp.int32,
                    namespace="test",
                )
            )
        self.assertIn("not registered", str(context.exception))
        self.assertIn("test:unregistered", str(context.exception))

    def test_custom_frequency_add_custom_values_batch(self):
        """Test batched custom frequency row insertion."""
        builder = ModelBuilder()
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="row", namespace="test"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="row_id",
                frequency="test:row",
                dtype=wp.int32,
                default=0,
                namespace="test",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="row_value",
                frequency="test:row",
                dtype=wp.float32,
                default=0.0,
                namespace="test",
            )
        )

        indices = builder.add_custom_values_batch(
            [
                {"test:row_id": 10, "test:row_value": 1.5},
                {"test:row_id": 11, "test:row_value": 2.5},
            ]
        )
        self.assertEqual(indices[0]["test:row_id"], 0)
        self.assertEqual(indices[1]["test:row_id"], 1)

        model = builder.finalize(device=self.device)
        np.testing.assert_array_equal(model.test.row_id.numpy(), [10, 11])
        np.testing.assert_array_almost_equal(model.test.row_value.numpy(), [1.5, 2.5], decimal=6)

    def test_custom_frequency_registration_methods(self):
        """Test different ways to register custom frequencies."""
        builder = ModelBuilder()

        # Test 1: Register with CustomFrequency (namespace + name)
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="freq1", namespace="ns"))
        self.assertIn("ns:freq1", builder.custom_frequencies)

        # Test 2: Register with CustomFrequency object
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="freq2", namespace="ns"))
        self.assertIn("ns:freq2", builder.custom_frequencies)

        # Test 3: Register without namespace
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="global_freq"))
        self.assertIn("global_freq", builder.custom_frequencies)

        # Test 4: Duplicate registration should be silently ignored (idempotent)
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="freq1", namespace="ns"))  # Should not raise
        self.assertEqual(len(builder.custom_frequencies), 3)  # Still 3 frequencies

    def test_custom_frequency_validation_inconsistent_counts(self):
        """Test that inconsistent counts for same custom frequency are handled gracefully with warnings."""
        builder = ModelBuilder()

        # Register custom frequency before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="pair", namespace="test"))

        # Declare attributes with same custom frequency
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_a",
                frequency="test:pair",
                dtype=wp.int32,
                namespace="test",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_b",
                frequency="test:pair",
                dtype=wp.int32,
                namespace="test",
            )
        )

        # Add different counts - pair_a has 2 values, pair_b has 1 value
        builder.add_custom_values(**{"test:pair_a": 1})
        builder.add_custom_values(**{"test:pair_a": 2})
        builder.add_custom_values(**{"test:pair_b": 10})  # Only 1 value for pair_b

        # This should now succeed with warnings and pad missing values with defaults
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = builder.finalize(device=self.device)

            # Should have warned about pair_a having fewer values (since pair_b expanded the frequency count)
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(any("pair_a" in msg and "missing values" in msg.lower() for msg in warning_messages))

        # Verify that arrays were created with correct counts (authoritative count expanded to 3 by pair_b)
        self.assertEqual(len(model.test.pair_a.numpy()), 3)
        self.assertEqual(len(model.test.pair_b.numpy()), 3)

        # Verify values: pair_a should have [1, 2, 0] (padded), pair_b should have [0, 0, 10]
        np.testing.assert_array_equal(model.test.pair_a.numpy(), [1, 2, 0])  # 0 is default for int32
        np.testing.assert_array_equal(model.test.pair_b.numpy(), [0, 0, 10])  # None values replaced with defaults

    def test_custom_frequency_add_custom_values_rejects_enum_frequency(self):
        """Test that add_custom_values() rejects enum frequency attributes."""
        builder = ModelBuilder()

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="body_attr",
                frequency=AttributeFrequency.BODY,
                dtype=wp.float32,
            )
        )

        with self.assertRaises(TypeError) as context:
            builder.add_custom_values(**{"body_attr": 1.0})
        self.assertIn("custom frequency", str(context.exception).lower())

    def test_custom_frequency_multi_world_merging(self):
        """Test custom frequency attributes are correctly offset during add_world() merging."""
        # Create sub-builder with custom frequency attributes
        sub_builder = ModelBuilder()

        # Register custom frequency before adding attributes
        sub_builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="item", namespace="test"))

        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="item_id",
                frequency="test:item",
                dtype=wp.int32,
                namespace="test",
            )
        )
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="item_value",
                frequency="test:item",
                dtype=wp.float32,
                namespace="test",
            )
        )

        # Add items to sub-builder
        sub_builder.add_custom_values(
            **{
                "test:item_id": 100,
                "test:item_value": 1.0,
            }
        )
        sub_builder.add_custom_values(
            **{
                "test:item_id": 200,
                "test:item_value": 2.0,
            }
        )

        # Create main builder and merge sub-builder twice
        main_builder = ModelBuilder()
        main_builder.add_world(sub_builder)  # World 0: items 0, 1
        main_builder.add_world(sub_builder)  # World 1: items 2, 3

        model = main_builder.finalize(device=self.device)

        # Verify merged values
        item_ids = model.test.item_id.numpy()
        item_values = model.test.item_value.numpy()

        self.assertEqual(len(item_ids), 4)
        # Values should be replicated (not offset, since item_id doesn't have references)
        np.testing.assert_array_equal(item_ids, [100, 200, 100, 200])
        np.testing.assert_array_almost_equal(item_values, [1.0, 2.0, 1.0, 2.0], decimal=5)

        # Verify custom frequency count
        self.assertEqual(model.get_custom_frequency_count("test:item"), 4)

    def test_custom_frequency_references_offset(self):
        """Test that custom frequency can be used as references for offsetting."""
        # Create sub-builder
        sub_builder = ModelBuilder()

        # Register custom frequencies before adding attributes
        sub_builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="entity", namespace="test"))
        sub_builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="ref", namespace="test"))

        # Entity attributes
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="entity_data",
                frequency="test:entity",
                dtype=wp.int32,
                namespace="test",
            )
        )

        # Reference attribute that references the entity frequency
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="ref_to_entity",
                frequency="test:ref",
                dtype=wp.int32,
                namespace="test",
                references="test:entity",  # Reference to custom frequency
            )
        )

        # Add entities
        sub_builder.add_custom_values(**{"test:entity_data": 100})
        sub_builder.add_custom_values(**{"test:entity_data": 200})

        # Add references (index into entity array)
        sub_builder.add_custom_values(**{"test:ref_to_entity": 0})  # References entity 0
        sub_builder.add_custom_values(**{"test:ref_to_entity": 1})  # References entity 1

        # Merge twice
        main_builder = ModelBuilder()
        main_builder.add_world(sub_builder)  # World 0
        main_builder.add_world(sub_builder)  # World 1

        model = main_builder.finalize(device=self.device)

        # Verify entity data is replicated
        entity_data = model.test.entity_data.numpy()
        np.testing.assert_array_equal(entity_data, [100, 200, 100, 200])

        # Verify references are offset by entity count
        refs = model.test.ref_to_entity.numpy()
        # World 0: refs point to 0, 1
        # World 1: refs should be offset by 2 (entity count from world 0), so 2, 3
        np.testing.assert_array_equal(refs, [0, 1, 2, 3])

    def test_custom_frequency_unknown_references_raises_error(self):
        """Test that unknown references value raises ValueError during add_world."""
        sub_builder = ModelBuilder()
        # Register custom frequency before adding attributes
        sub_builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="item", namespace="test"))
        sub_builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="bad_ref",
                frequency="test:item",
                dtype=wp.int32,
                namespace="test",
                references="shapes",  # Typo: should be "shape"
            )
        )
        sub_builder.add_custom_values(**{"test:bad_ref": 0})

        main_builder = ModelBuilder()
        with self.assertRaisesRegex(ValueError, "Unknown references value 'shapes'"):
            main_builder.add_world(sub_builder)

    def test_custom_frequency_different_frequencies_independent(self):
        """Test that different custom frequencies are independent."""
        builder = ModelBuilder()

        # Register custom frequencies before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="type_a", namespace="test"))
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="type_b", namespace="test"))

        # Two different custom frequencies
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="type_a_data",
                frequency="test:type_a",
                dtype=wp.int32,
                namespace="test",
            )
        )
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="type_b_data",
                frequency="test:type_b",
                dtype=wp.int32,
                namespace="test",
            )
        )

        # Add different counts for each frequency
        builder.add_custom_values(**{"test:type_a_data": 1})
        builder.add_custom_values(**{"test:type_a_data": 2})
        builder.add_custom_values(**{"test:type_a_data": 3})

        builder.add_custom_values(**{"test:type_b_data": 10})

        model = builder.finalize(device=self.device)

        # Verify independent counts
        type_a = model.test.type_a_data.numpy()
        type_b = model.test.type_b_data.numpy()

        self.assertEqual(len(type_a), 3)
        self.assertEqual(len(type_b), 1)

        self.assertEqual(model.get_custom_frequency_count("test:type_a"), 3)
        self.assertEqual(model.get_custom_frequency_count("test:type_b"), 1)

    def test_custom_frequency_empty(self):
        """Test that empty custom frequency attributes don't create arrays."""
        builder = ModelBuilder()

        # Register custom frequency before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="empty", namespace="test"))

        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="empty_attr",
                frequency="test:empty",
                dtype=wp.int32,
                namespace="test",
            )
        )

        model = builder.finalize(device=self.device)

        # Empty frequency shouldn't create a namespace or attribute
        self.assertFalse(hasattr(model, "test"))
        self.assertEqual(model.get_custom_frequency_count("test:empty"), 0)

    def test_custom_frequency_unknown_raises_keyerror(self):
        """Test that get_custom_frequency_count raises KeyError for unknown frequencies."""
        builder = ModelBuilder()
        # Register custom frequency before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="known", namespace="test"))
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="item",
                frequency="test:known",
                dtype=wp.int32,
                namespace="test",
            )
        )
        model = builder.finalize(device=self.device)

        # Known frequency works
        self.assertEqual(model.get_custom_frequency_count("test:known"), 0)

        # Unknown frequency raises KeyError
        with self.assertRaisesRegex(KeyError, "unknown"):
            model.get_custom_frequency_count("test:unknown")

    def test_custom_frequency_articulation_view_rejection(self):
        """Test that ArticulationView raises error for custom string frequency attributes."""

        builder = ModelBuilder()

        # Create an articulation
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], key="robot")

        # Register custom frequency before adding attributes
        builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="item"))

        # Add a custom string frequency attribute (no namespace for simpler access)
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="item_data",
                frequency="item",  # Custom string frequency
                dtype=wp.int32,
            )
        )
        builder.add_custom_values(**{"item_data": 42})

        model = builder.finalize(device=self.device)

        # Create ArticulationView
        view = ArticulationView(model, "robot")

        # Accessing a custom string frequency attribute should raise AttributeError
        with self.assertRaises(AttributeError) as context:
            view._get_attribute_array("item_data", model)

        self.assertIn("custom frequency", str(context.exception).lower())
        self.assertIn("item", str(context.exception))

    def test_world_frequency_merge_add_world(self):
        """Test that WORLD-frequency attributes are correctly indexed when using add_world()."""
        sub = ModelBuilder()
        sub.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="world_data",
                dtype=wp.int32,
                frequency=AttributeFrequency.WORLD,
                namespace="test",
                default=-999,
            )
        )
        # Manually set value at index 0 for the sub-builder's world
        sub.custom_attributes["test:world_data"].values = {0: 42}

        main = ModelBuilder()
        main.add_world(sub)
        main.add_world(sub)

        model = main.finalize(device=self.device)
        arr = model.test.world_data.numpy()

        self.assertEqual(model.num_worlds, 2)
        self.assertEqual(len(arr), 2)
        self.assertEqual(arr[0], 42)
        self.assertEqual(arr[1], 42)

    def test_transform_value_list_and_sentinel_shape_refs(self):
        """Test that transform_value handles lists with negative sentinel values correctly."""
        main = ModelBuilder()

        # Register custom frequency before adding attributes
        main.add_custom_frequency(ModelBuilder.CustomFrequency(name="pair", namespace="test"))

        # Declare a custom frequency attribute with shape references
        main.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geoms",
                dtype=wp.vec2i,
                frequency="test:pair",
                namespace="test",
                references="shape",
            )
        )

        # Create sub-builder with a shape and pair data
        sub = ModelBuilder()
        # Register custom frequency before adding attributes
        sub.add_custom_frequency(ModelBuilder.CustomFrequency(name="pair", namespace="test"))
        sub.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="pair_geoms",
                dtype=wp.vec2i,
                frequency="test:pair",
                namespace="test",
                references="shape",
            )
        )
        body = sub.add_body(mass=1.0)
        sub.add_shape_sphere(body, radius=0.1)  # shape 0
        # Add pair with value [0, -1] where -1 is sentinel for "no geom"
        sub.add_custom_values(**{"test:pair_geoms": [0, -1]})

        # Add main's own shape first
        main_body = main.add_body(mass=1.0)
        main.add_shape_sphere(main_body, radius=0.1)  # shape 0 in main

        # Merge sub as new world - shape offset should be 1
        main.add_world(sub)

        model = main.finalize(device=self.device)
        arr = model.test.pair_geoms.numpy()

        # Should have 1 pair entry
        self.assertEqual(len(arr), 1)
        # First element should be offset by 1 (shape_offset), second (-1) preserved
        self.assertEqual(arr[0][0], 1)  # 0 + 1 = 1
        self.assertEqual(arr[0][1], -1)  # sentinel preserved

    def test_merge_custom_attribute_default_only_no_crash(self):
        """Test add_builder does not crash when sub-builder has default-only attribute (no overrides)."""
        # Sub-builder declares a BODY-frequency custom attribute with default but no overrides
        sub = ModelBuilder()
        sub.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name="foo",
                frequency=AttributeFrequency.BODY,
                dtype=wp.int32,
                default=7,
                namespace="ns",
            )
        )
        sub.add_body()  # no custom override for 'ns:foo'

        # Main builder merges sub-builder as a new world
        main = ModelBuilder()
        main.add_world(sub)

        # Should not raise; should build an array of size == body_count with default 7
        model = main.finalize(device=self.device)
        arr = model.ns.foo.numpy().tolist()
        self.assertEqual(len(arr), model.body_count)
        self.assertTrue(all(v == 7 for v in arr))


if __name__ == "__main__":
    unittest.main(verbosity=2)
