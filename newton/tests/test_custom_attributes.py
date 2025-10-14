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
                ModelAttributeAssignment.MODEL: {
                    "custom_float": 25.5,
                    "custom_int": 42,
                    "custom_bool": True,
                    "custom_vec3": [1.0, 0.5, 0.0],
                },
                ModelAttributeAssignment.STATE: {
                    "velocity_limit": [2.0, 2.0, 2.0],
                    "is_active": True,
                    "energy": 100.5,
                },
                ModelAttributeAssignment.CONTROL: {
                    "gain": 1.5,
                    "mode": 3,
                },
            },
        )

        body2 = builder.add_body(
            mass=2.0,
            custom_attributes={
                ModelAttributeAssignment.MODEL: {
                    "custom_float": 30.0,
                    "custom_int": 7,
                    "custom_bool": False,
                    "custom_vec3": [0.0, 1.0, 0.5],
                },
                ModelAttributeAssignment.STATE: {
                    "velocity_limit": [3.0, 3.0, 3.0],
                    "is_active": False,
                    "energy": 200.0,
                },
                ModelAttributeAssignment.CONTROL: {
                    "gain": 2.0,
                    "mode": 5,
                },
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
                ModelAttributeAssignment.MODEL: {
                    "custom_float": 0.8,
                    "custom_int": 3,
                    "custom_bool": False,
                    "custom_vec2": [0.2, 0.4],
                }
            },
        )

        shape2 = builder.add_shape_sphere(
            body=robot_entities["link1"],
            radius=0.02,
            custom_attributes={
                ModelAttributeAssignment.MODEL: {
                    "custom_float": 0.3,
                    "custom_int": 1,
                    "custom_bool": True,
                    "custom_vec2": [0.8, 0.6],
                }
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
        builder.add_custom_attribute("dof_custom_float", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("dof_custom_int", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.int32)
        builder.add_custom_attribute("coord_custom_float", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.float32)
        builder.add_custom_attribute("coord_custom_int", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.int32)

        robot_entities = self._add_test_robot(builder)

        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(
            parent=robot_entities["link2"],
            child=body,
            axis=[0.0, 0.0, 1.0],
            custom_attributes={
                ModelAttributeAssignment.MODEL: {
                    "dof_custom_float": [0.05],
                    "dof_custom_int": [15],
                    "coord_custom_float": [0.5],
                    "coord_custom_int": [12],
                }
            },
        )

        model = builder.finalize(device=self.device)

        # Verify DOF attributes
        dof_float_numpy = model.dof_custom_float.numpy()
        self.assertAlmostEqual(dof_float_numpy[2], 0.05, places=5)
        self.assertAlmostEqual(dof_float_numpy[0], 0.0, places=5)

        dof_int_numpy = model.dof_custom_int.numpy()
        self.assertEqual(dof_int_numpy[2], 15)
        self.assertEqual(dof_int_numpy[1], 0)

        # Verify coordinate attributes
        coord_float_numpy = model.coord_custom_float.numpy()
        self.assertAlmostEqual(coord_float_numpy[2], 0.5, places=5)
        self.assertAlmostEqual(coord_float_numpy[0], 0.0, places=5)

        coord_int_numpy = model.coord_custom_int.numpy()
        self.assertEqual(coord_int_numpy[2], 12)
        self.assertEqual(coord_int_numpy[1], 0)

    def test_multi_dof_joint_individual_values(self):
        """Test D6 joint with individual values per DOF and coordinate."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("dof_custom_float", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("coord_custom_int", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.int32)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                ModelAttributeAssignment.MODEL: {
                    "dof_custom_float": [0.1, 0.2, 0.3],
                    "coord_custom_int": [100, 200, 300],
                }
            },
        )

        model = builder.finalize(device=self.device)

        # Verify individual DOF values
        dof_float_numpy = model.dof_custom_float.numpy()
        self.assertAlmostEqual(dof_float_numpy[2], 0.1, places=5)
        self.assertAlmostEqual(dof_float_numpy[3], 0.2, places=5)
        self.assertAlmostEqual(dof_float_numpy[4], 0.3, places=5)
        self.assertAlmostEqual(dof_float_numpy[0], 0.0, places=5)

        # Verify individual coordinate values
        coord_int_numpy = model.coord_custom_int.numpy()
        self.assertEqual(coord_int_numpy[2], 100)
        self.assertEqual(coord_int_numpy[3], 200)
        self.assertEqual(coord_int_numpy[4], 300)
        self.assertEqual(coord_int_numpy[1], 0)

    def test_multi_dof_joint_vector_attributes(self):
        """Test D6 joint with vector attributes (list of lists)."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("dof_custom_vec2", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.vec2)
        builder.add_custom_attribute("coord_custom_vec3", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.vec3)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            custom_attributes={
                ModelAttributeAssignment.MODEL: {
                    "dof_custom_vec2": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    "coord_custom_vec3": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                }
            },
        )

        model = builder.finalize(device=self.device)

        # Verify DOF vector values
        dof_vec2_numpy = model.dof_custom_vec2.numpy()
        np.testing.assert_array_almost_equal(dof_vec2_numpy[2], [1.0, 2.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[3], [3.0, 4.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[4], [5.0, 6.0], decimal=5)
        np.testing.assert_array_almost_equal(dof_vec2_numpy[0], [0.0, 0.0], decimal=5)

        # Verify coordinate vector values
        coord_vec3_numpy = model.coord_custom_vec3.numpy()
        np.testing.assert_array_almost_equal(coord_vec3_numpy[2], [0.1, 0.2, 0.3], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[3], [0.4, 0.5, 0.6], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[4], [0.7, 0.8, 0.9], decimal=5)
        np.testing.assert_array_almost_equal(coord_vec3_numpy[1], [0.0, 0.0, 0.0], decimal=5)

    def test_dof_coord_list_requirements(self):
        """Test that DOF and coordinate attributes must be lists with correct lengths."""
        builder = newton.ModelBuilder()

        # Declare custom attributes before use
        builder.add_custom_attribute("dof_custom_float", newton.ModelAttributeFrequency.JOINT_DOF, dtype=wp.float32)
        builder.add_custom_attribute("coord_custom_float", newton.ModelAttributeFrequency.JOINT_COORD, dtype=wp.float32)

        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        # Test DOF attribute must be a list
        body1 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"],
                child=body1,
                axis=[0, 0, 1],
                custom_attributes={ModelAttributeAssignment.MODEL: {"dof_custom_float": 0.1}},
            )

        # Test wrong DOF list length
        body2 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_d6(
                parent=robot_entities["link2"],
                child=body2,
                linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
                angular_axes=[cfg(axis=[0, 0, 1])],
                custom_attributes={
                    ModelAttributeAssignment.MODEL: {"dof_custom_float": [0.1, 0.2]}
                },  # 2 values for 3-DOF joint
            )

        # Test coordinate attribute must be a list
        body3 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"],
                child=body3,
                axis=[1, 0, 0],
                custom_attributes={ModelAttributeAssignment.MODEL: {"coord_custom_float": 0.5}},
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
                ModelAttributeAssignment.MODEL: {
                    "custom_vec2": [1.0, 2.0],
                    "custom_vec3": [1.0, 2.0, 3.0],
                    "custom_vec4": [1.0, 2.0, 3.0, 4.0],
                }
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
                custom_attributes={
                    ModelAttributeAssignment.MODEL: {"custom_string": "test_body", "custom_float": 25.0}
                },
            )

        # But using only declared attribute should work
        builder.add_body(mass=1.0, custom_attributes={ModelAttributeAssignment.MODEL: {"custom_float": 25.0}})

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
                custom_attributes={ModelAttributeAssignment.MODEL: {"body_only_attr": 1.0}},
            )
        self.assertIn("frequency", str(context.exception).lower())

        # Test 3: Using SHAPE frequency attribute on a body should fail
        builder.add_custom_attribute("shape_only_attr", newton.ModelAttributeFrequency.SHAPE, dtype=wp.float32)

        with self.assertRaises(ValueError) as context:
            builder.add_body(mass=1.0, custom_attributes={ModelAttributeAssignment.MODEL: {"shape_only_attr": 2.0}})
        self.assertIn("frequency", str(context.exception).lower())

        # Test 4: Using attributes with correct frequency should work
        builder.add_body(mass=1.0, custom_attributes={ModelAttributeAssignment.MODEL: {"body_only_attr": 1.5}})
        builder.add_shape_box(
            body=robot_entities["base"],
            hx=0.1,
            hy=0.1,
            hz=0.1,
            custom_attributes={ModelAttributeAssignment.MODEL: {"shape_only_attr": 2.5}},
        )

        # Test 5: Attribute with wrong assignment should raise ValueError
        # Declare an attribute with STATE assignment
        builder.add_custom_attribute(
            "state_attr",
            newton.ModelAttributeFrequency.BODY,
            dtype=wp.float32,
            assignment=newton.ModelAttributeAssignment.STATE,
        )

        # This should fail because the attribute expects STATE assignment but we're using MODEL
        with self.assertRaises(ValueError) as context:
            builder.add_body(mass=1.0, custom_attributes={ModelAttributeAssignment.MODEL: {"state_attr": 3.0}})
        self.assertIn("assignment", str(context.exception).lower())

        # Verify attributes were created with correct assignments
        self.assertEqual(custom_attrs["custom_float"].assignment, ModelAttributeAssignment.MODEL)
        self.assertEqual(custom_attrs["body_only_attr"].assignment, ModelAttributeAssignment.MODEL)
        self.assertEqual(builder.custom_attributes["state_attr"].assignment, ModelAttributeAssignment.STATE)

        model = builder.finalize(device=self.device)
        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(model, "custom_string"))

    def test_assignment_types(self):
        """Test custom attribute assignment to MODEL objects."""
        builder = newton.ModelBuilder()

        # Declare custom attribute before use
        builder.add_custom_attribute("custom_float", newton.ModelAttributeFrequency.BODY, dtype=wp.float32)

        builder.add_body(mass=1.0, custom_attributes={ModelAttributeAssignment.MODEL: {"custom_float": 25.0}})

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
                ModelAttributeAssignment.MODEL: {
                    "scalar_attr": 42.5,
                    "vec3_attr": [1.0, 2.0, 3.0],
                    "int_attr": 7,
                }
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
