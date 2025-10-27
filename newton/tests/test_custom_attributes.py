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

import numpy as np
import warp as wp

import newton
from newton._src.sim.model import ModelAttributeAssignment


class TestCustomAttributes(unittest.TestCase):
    """Test custom attributes functionality via ModelBuilder kwargs."""

    def setUp(self):
        """Set up test fixtures."""
        wp.init()
        self.device = wp.get_device()

    def _add_test_robot(self, builder: newton.ModelBuilder) -> dict[str, int]:
        """Build a simple 2-bar linkage robot without custom attributes."""
        base = builder.add_body(xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()), mass=1.0)
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)

        link1 = builder.add_body(xform=wp.transform([0.0, 0.0, 0.5], wp.quat_identity()), mass=0.5)
        builder.add_shape_capsule(link1, radius=0.05, half_height=0.2)

        joint1 = builder.add_joint_revolute(
            parent=base,
            child=link1,
            parent_xform=wp.transform([0.0, 0.0, 0.1], wp.quat_identity()),
            child_xform=wp.transform([0.0, 0.0, -0.2], wp.quat_identity()),
            axis=[0.0, 1.0, 0.0],
        )

        link2 = builder.add_body(xform=wp.transform([0.0, 0.0, 0.9], wp.quat_identity()), mass=0.3)
        builder.add_shape_capsule(link2, radius=0.03, half_height=0.15)

        joint2 = builder.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform([0.0, 0.0, 0.2], wp.quat_identity()),
            child_xform=wp.transform([0.0, 0.0, -0.15], wp.quat_identity()),
            axis=[0.0, 1.0, 0.0],
        )

        return {"base": base, "link1": link1, "link2": link2, "joint1": joint1, "joint2": joint2}

    def test_body_custom_attributes(self):
        """Test BODY frequency custom attributes with multiple data types and assignments."""
        builder = newton.ModelBuilder()

        # Declare MODEL assignment attributes
        builder.add_custom_attribute(
            "custom_float",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        builder.add_custom_attribute(
            "custom_int", newton.ModelAttributeFrequency.BODY, dtype=wp.int32, assignment=ModelAttributeAssignment.MODEL
        )
        builder.add_custom_attribute(
            "custom_bool", newton.ModelAttributeFrequency.BODY, dtype=wp.bool, assignment=ModelAttributeAssignment.MODEL
        )
        builder.add_custom_attribute(
            "custom_vec3", newton.ModelAttributeFrequency.BODY, dtype=wp.vec3, assignment=ModelAttributeAssignment.MODEL
        )

        # Declare STATE assignment attributes
        builder.add_custom_attribute(
            "velocity_limit",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.vec3,
            assignment=ModelAttributeAssignment.STATE,
        )
        builder.add_custom_attribute(
            "is_active", newton.ModelAttributeFrequency.BODY, dtype=wp.bool, assignment=ModelAttributeAssignment.STATE
        )
        builder.add_custom_attribute(
            "energy", newton.ModelAttributeFrequency.BODY, dtype=wp.float32, assignment=ModelAttributeAssignment.STATE
        )

        # Declare CONTROL assignment attributes
        builder.add_custom_attribute(
            "gain", newton.ModelAttributeFrequency.BODY, dtype=wp.float32, assignment=ModelAttributeAssignment.CONTROL
        )
        builder.add_custom_attribute(
            "mode", newton.ModelAttributeFrequency.BODY, dtype=wp.int32, assignment=ModelAttributeAssignment.CONTROL
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
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_float", newton.ModelAttributeFrequency.SHAPE, dtype=wp.float32)
        builder.add_custom_attribute("custom_int", newton.ModelAttributeFrequency.SHAPE, dtype=wp.int32)
        builder.add_custom_attribute("custom_bool", newton.ModelAttributeFrequency.SHAPE, dtype=wp.bool)
        builder.add_custom_attribute("custom_vec2", newton.ModelAttributeFrequency.SHAPE, dtype=wp.vec2)

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
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_float_dof", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("custom_int_dof", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.int32)
        builder.add_custom_attribute("custom_float_coord", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.float32)
        builder.add_custom_attribute("custom_int_coord", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.int32)

        robot_entities = self._add_test_robot(builder)

        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(
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

    def test_multi_dof_joint_individual_values(self):
        """Test D6 joint with individual values per DOF and coordinate."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_float_dof", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("custom_int_coord", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.int32)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                "custom_float_dof": [0.1, 0.2, 0.3],
                "custom_int_coord": [100, 200, 300],
            },
        )

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

    def test_multi_dof_joint_vector_attributes(self):
        """Test D6 joint with vector attributes (list of lists)."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_vec2_dof", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.vec2)
        builder.add_custom_attribute("custom_vec3_coord", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.vec3)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                "custom_vec2_dof": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "custom_vec3_coord": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            },
        )

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

    def test_dof_coord_list_requirements(self):
        """Test that DOF and coordinate attributes must be lists with correct lengths."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_float_dof", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("custom_float_coord", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.float32)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        # Test DOF attribute must be a list (type error)
        body1 = builder.add_body(mass=1.0)
        with self.assertRaises(TypeError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"],
                child=body1,
                axis=[0, 0, 1],
                custom_attributes={"custom_float_dof": 0.1},
            )

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

        # Test coordinate attribute must be a list (type error)
        body3 = builder.add_body(mass=1.0)
        with self.assertRaises(TypeError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"],
                child=body3,
                axis=[1, 0, 0],
                custom_attributes={"custom_float_coord": 0.5},
            )

    def test_vector_type_inference(self):
        """Test automatic dtype inference for vector types."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("custom_vec2", newton.ModelAttributeFrequency.BODY, dtype=wp.vec2)
        builder.add_custom_attribute("custom_vec3", newton.ModelAttributeFrequency.BODY, dtype=wp.vec3)
        builder.add_custom_attribute("custom_vec4", newton.ModelAttributeFrequency.BODY, dtype=wp.vec4)

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
        builder = newton.ModelBuilder()
        robot_entities = self._add_test_robot(builder)

        # Test 1: Undeclared string attribute should raise AttributeError
        builder.add_custom_attribute("custom_float", newton.ModelAttributeFrequency.BODY, dtype=wp.float32)

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
        builder.add_custom_attribute("body_only_attr", newton.ModelAttributeFrequency.BODY, dtype=wp.float32)

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
        builder.add_custom_attribute("shape_only_attr", newton.ModelAttributeFrequency.SHAPE, dtype=wp.float32)

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
        self.assertEqual(custom_attrs["custom_float"].assignment, ModelAttributeAssignment.MODEL)
        self.assertEqual(custom_attrs["body_only_attr"].assignment, ModelAttributeAssignment.MODEL)

        model = builder.finalize(device=self.device)
        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(model, "custom_string"))

    def test_assignment_types(self):
        """Test custom attribute assignment to MODEL objects."""
        builder = newton.ModelBuilder()

        # Declare custom attribute before use
        builder.add_custom_attribute("custom_float", newton.ModelAttributeFrequency.BODY, dtype=wp.float32)

        builder.add_body(mass=1.0, custom_attributes={"custom_float": 25.0})

        custom_attrs = builder.custom_attributes
        self.assertEqual(custom_attrs["custom_float"].assignment, ModelAttributeAssignment.MODEL)

        model = builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(state, "custom_float"))
        self.assertFalse(hasattr(control, "custom_float"))

    def test_value_dtype_compatibility(self):
        """Test that values work correctly with declared dtypes."""
        builder = newton.ModelBuilder()

        # Declare attributes with different dtypes
        builder.add_custom_attribute("scalar_attr", newton.ModelAttributeFrequency.BODY, dtype=wp.float32)
        builder.add_custom_attribute("vec3_attr", newton.ModelAttributeFrequency.BODY, dtype=wp.vec3)
        builder.add_custom_attribute("int_attr", newton.ModelAttributeFrequency.BODY, dtype=wp.int32)

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
        """Test that custom attributes are preserved when using add_builder()."""
        # Create a sub-builder with custom attributes
        sub_builder = newton.ModelBuilder()

        # Declare attributes with different frequencies and assignments
        sub_builder.add_custom_attribute(
            "robot_id", newton.ModelAttributeFrequency.BODY, dtype=wp.int32, assignment=ModelAttributeAssignment.MODEL
        )
        sub_builder.add_custom_attribute(
            "temperature",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.STATE,
        )
        sub_builder.add_custom_attribute(
            "shape_color",
            newton.ModelAttributeFrequency.SHAPE,
            dtype=wp.vec3,
            assignment=ModelAttributeAssignment.MODEL,
        )
        sub_builder.add_custom_attribute(
            "gain_dof",
            newton.ModelAttributeFrequency.JOINT_DOF,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.CONTROL,
        )

        # Create a simple robot in sub-builder
        body1 = sub_builder.add_body(
            mass=1.0,
            custom_attributes={"robot_id": 100, "temperature": 37.5},
        )
        sub_builder.add_shape_sphere(body1, radius=0.1, custom_attributes={"shape_color": [1.0, 0.0, 0.0]})

        body2 = sub_builder.add_body(
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

        sub_builder.add_joint_revolute(
            parent=body1,
            child=body2,
            axis=[0, 0, 1],
            custom_attributes={"gain_dof": [1.5]},
        )

        # Create main builder and add sub-builder multiple times
        main_builder = newton.ModelBuilder()

        # Add first instance
        main_builder.add_builder(sub_builder, world=0)
        # Add second instance
        main_builder.add_builder(sub_builder, world=1)

        # Verify custom attributes were merged
        self.assertIn("robot_id", main_builder.custom_attributes)
        self.assertIn("temperature", main_builder.custom_attributes)
        self.assertIn("shape_color", main_builder.custom_attributes)
        self.assertIn("gain_dof", main_builder.custom_attributes)

        # Verify frequencies and assignments
        self.assertEqual(main_builder.custom_attributes["robot_id"].frequency, newton.ModelAttributeFrequency.BODY)
        self.assertEqual(main_builder.custom_attributes["robot_id"].assignment, ModelAttributeAssignment.MODEL)
        self.assertEqual(main_builder.custom_attributes["temperature"].assignment, ModelAttributeAssignment.STATE)
        self.assertEqual(main_builder.custom_attributes["shape_color"].frequency, newton.ModelAttributeFrequency.SHAPE)
        self.assertEqual(main_builder.custom_attributes["gain_dof"].frequency, newton.ModelAttributeFrequency.JOINT_DOF)

        # Build model and verify values
        model = main_builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        # Verify BODY attributes (2 bodies per instance, 2 instances = 4 bodies total)
        robot_ids = model.robot_id.numpy()
        temperatures = state.temperature.numpy()

        # First instance (bodies 0, 1)
        self.assertEqual(robot_ids[0], 100)
        self.assertEqual(robot_ids[1], 200)
        self.assertAlmostEqual(temperatures[0], 37.5, places=5)
        self.assertAlmostEqual(temperatures[1], 38.0, places=5)

        # Second instance (bodies 2, 3)
        self.assertEqual(robot_ids[2], 100)
        self.assertEqual(robot_ids[3], 200)
        self.assertAlmostEqual(temperatures[2], 37.5, places=5)
        self.assertAlmostEqual(temperatures[3], 38.0, places=5)

        # Verify SHAPE attributes (2 shapes per instance, 2 instances = 4 shapes total)
        shape_colors = model.shape_color.numpy()

        # First instance (shapes 0, 1)
        np.testing.assert_array_almost_equal(shape_colors[0], [1.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[1], [0.0, 1.0, 0.0], decimal=5)

        # Second instance (shapes 2, 3)
        np.testing.assert_array_almost_equal(shape_colors[2], [1.0, 0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(shape_colors[3], [0.0, 1.0, 0.0], decimal=5)

        # Verify JOINT_DOF attributes (1 DOF per instance, 2 instances = 2 DOFs total)
        dof_gains = control.gain_dof.numpy()

        # First instance (DOF 0)
        self.assertAlmostEqual(dof_gains[0], 1.5, places=5)

        # Second instance (DOF 1)
        self.assertAlmostEqual(dof_gains[1], 1.5, places=5)

    def test_namespaced_attributes(self):
        """Test namespaced custom attributes with hierarchical organization."""
        builder = newton.ModelBuilder()

        # Declare attributes in different namespaces
        builder.add_custom_attribute(
            "damping",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            namespace="mujoco",
        )
        builder.add_custom_attribute(
            "enable_ccd",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.bool,
            assignment=ModelAttributeAssignment.STATE,
            namespace="physx",
        )
        builder.add_custom_attribute(
            "custom_id",
            newton.ModelAttributeFrequency.SHAPE,
            dtype=wp.int32,
            assignment=ModelAttributeAssignment.MODEL,
            namespace="mujoco",
        )
        # Declare a default namespace attribute (no namespace)
        builder.add_custom_attribute(
            "temperature",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
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
        builder1 = newton.ModelBuilder()
        builder1.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        builder1.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.STATE,
            namespace="namespace_a",
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
        builder2 = newton.ModelBuilder()
        builder2.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        with self.assertRaises(ValueError) as context:
            builder2.add_custom_attribute(
                "float_attr",
                newton.ModelAttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=ModelAttributeAssignment.STATE,  # Different assignment, same key
            )
        self.assertIn("already exists", str(context.exception))
        self.assertIn("incompatible spec", str(context.exception))

        # Test 3: Same namespace:name with different assignments - SHOULD FAIL
        # Both use key "namespace_a:float_attr"
        builder3 = newton.ModelBuilder()
        builder3.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            namespace="namespace_a",
        )
        with self.assertRaises(ValueError) as context:
            builder3.add_custom_attribute(
                "float_attr",
                newton.ModelAttributeFrequency.BODY,
                dtype=wp.float32,
                assignment=ModelAttributeAssignment.STATE,  # Different assignment, same namespace:name
                namespace="namespace_a",
            )
        self.assertIn("already exists", str(context.exception))

        # Test 4: Same name in different namespaces with same assignment - SHOULD WORK
        # Keys "namespace_a:float_attr" and "namespace_b:float_attr" are different
        builder4 = newton.ModelBuilder()
        builder4.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            namespace="namespace_a",
        )
        builder4.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            namespace="namespace_b",
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
        builder5 = newton.ModelBuilder()
        builder5.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        # Declaring again with same parameters should be allowed
        builder5.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        # Should still work
        self.assertEqual(len(builder5.custom_attributes), 1)

        # Test 6: Same key with different frequency - SHOULD FAIL
        builder6 = newton.ModelBuilder()
        builder6.add_custom_attribute(
            "float_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
        )
        with self.assertRaises(ValueError) as context:
            builder6.add_custom_attribute(
                "float_attr",
                newton.ModelAttributeFrequency.SHAPE,  # Different frequency
                dtype=wp.float32,
                assignment=ModelAttributeAssignment.MODEL,
            )
        self.assertIn("already exists", str(context.exception))
        self.assertIn("incompatible spec", str(context.exception))

    def test_mixed_free_and_articulated_bodies(self):
        """Test BODY and ARTICULATION frequency custom attributes with mixed free and articulated bodies."""
        builder = newton.ModelBuilder()

        # Declare custom attributes
        builder.add_custom_attribute(
            "temperature",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            default=20.0,
        )
        builder.add_custom_attribute(
            "density",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.STATE,
            default=1.0,
        )
        builder.add_custom_attribute(
            "articulation_stiffness",
            newton.ModelAttributeFrequency.ARTICULATION,
            dtype=wp.float32,
            assignment=ModelAttributeAssignment.MODEL,
            default=100.0,
        )

        # Create free bodies (no articulation)
        free_body_ids = []
        for i in range(3):
            body = builder.add_body(
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
            builder.add_articulation(
                custom_attributes={
                    "articulation_stiffness": 100.0 + float(i) * 50.0,
                }
            )

            # Create 2-link articulation
            # Temperature NOT assigned to articulated bodies (use defaults)
            # Density assigned with different values than free bodies
            base = builder.add_body(
                xform=wp.transform([3.0 + float(i), 0.0, 0.0], wp.quat_identity()),
                mass=1.0,
                custom_attributes={"density": 2.0 + float(i) * 0.5},
            )
            builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)

            link = builder.add_body(
                xform=wp.transform([3.0 + float(i), 0.0, 0.5], wp.quat_identity()),
                mass=0.5,
                custom_attributes={"density": 3.0 + float(i) * 0.5},
            )
            builder.add_shape_capsule(link, radius=0.05, half_height=0.2)

            builder.add_joint_revolute(
                parent=base,
                child=link,
                parent_xform=wp.transform([0.0, 0.0, 0.1], wp.quat_identity()),
                child_xform=wp.transform([0.0, 0.0, -0.2], wp.quat_identity()),
                axis=[0.0, 1.0, 0.0],
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


def run_tests():
    """Run all custom attributes tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestCustomAttributes))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running Custom Attributes Tests")
    print("=" * 60)
    print("Testing ModelBuilder kwargs functionality for custom attributes")
    print("=" * 60)

    success = run_tests()

    if success:
        print("\nAll custom attributes tests passed!")
    else:
        print("\nSome custom attributes tests failed!")
