#!/usr/bin/env python3
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
        """Test BODY frequency custom attributes with multiple data types."""
        builder = newton.ModelBuilder()
        robot_entities = self._add_test_robot(builder)

        body1 = builder.add_body(
            mass=1.0, custom_float=25.5, custom_int=42, custom_bool=True, custom_vec3=[1.0, 0.5, 0.0]
        )

        body2 = builder.add_body(
            mass=2.0, custom_float=30.0, custom_int=7, custom_bool=False, custom_vec3=[0.0, 1.0, 0.5]
        )

        model = builder.finalize(device=self.device)

        # Verify authored values
        float_numpy = model.custom_float.numpy()
        self.assertAlmostEqual(float_numpy[body1], 25.5, places=5)
        self.assertAlmostEqual(float_numpy[body2], 30.0, places=5)

        int_numpy = model.custom_int.numpy()
        self.assertEqual(int_numpy[body1], 42)
        self.assertEqual(int_numpy[body2], 7)

        bool_numpy = model.custom_bool.numpy()
        self.assertEqual(bool_numpy[body1], 1)
        self.assertEqual(bool_numpy[body2], 0)

        # Verify default values on robot entities
        self.assertAlmostEqual(float_numpy[robot_entities["base"]], 0.0, places=5)
        self.assertEqual(int_numpy[robot_entities["link1"]], 0)
        self.assertEqual(bool_numpy[robot_entities["link2"]], 0)

    def test_shape_custom_attributes(self):
        """Test SHAPE frequency custom attributes with multiple data types."""
        builder = newton.ModelBuilder()
        robot_entities = self._add_test_robot(builder)

        shape1 = builder.add_shape_box(
            body=robot_entities["base"],
            hx=0.05,
            hy=0.05,
            hz=0.05,
            custom_float=0.8,
            custom_int=3,
            custom_bool=False,
            custom_vec2=[0.2, 0.4],
        )

        shape2 = builder.add_shape_sphere(
            body=robot_entities["link1"],
            radius=0.02,
            custom_float=0.3,
            custom_int=1,
            custom_bool=True,
            custom_vec2=[0.8, 0.6],
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
        robot_entities = self._add_test_robot(builder)

        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(
            parent=robot_entities["link2"],
            child=body,
            dof_custom_float=[0.05],
            dof_custom_int=[15],
            coord_custom_float=[0.5],
            coord_custom_int=[12],
            axis=[0.0, 0.0, 1.0],
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
        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            dof_custom_float=[0.1, 0.2, 0.3],
            coord_custom_int=[100, 200, 300],
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
        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        body = builder.add_body(mass=1.0)
        builder.add_joint_d6(
            parent=robot_entities["link2"],
            child=body,
            linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
            angular_axes=[cfg(axis=[0, 0, 1])],
            dof_custom_vec2=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            coord_custom_vec3=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
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
        robot_entities = self._add_test_robot(builder)
        cfg = newton.ModelBuilder.JointDofConfig

        # Test DOF attribute must be a list
        body1 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"], child=body1, dof_custom_float=0.1, axis=[0, 0, 1]
            )

        # Test wrong DOF list length
        body2 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_d6(
                parent=robot_entities["link2"],
                child=body2,
                linear_axes=[cfg(axis=newton.Axis.X), cfg(axis=newton.Axis.Y)],
                angular_axes=[cfg(axis=[0, 0, 1])],
                dof_custom_float=[0.1, 0.2],  # 2 values for 3-DOF joint
            )

        # Test coordinate attribute must be a list
        body3 = builder.add_body(mass=1.0)
        with self.assertRaises(ValueError):
            builder.add_joint_revolute(
                parent=robot_entities["link2"], child=body3, coord_custom_float=0.5, axis=[1, 0, 0]
            )

    def test_vector_type_inference(self):
        """Test automatic dtype inference for vector types."""
        builder = newton.ModelBuilder()

        body = builder.add_body(
            mass=1.0, custom_vec2=[1.0, 2.0], custom_vec3=[1.0, 2.0, 3.0], custom_vec4=[1.0, 2.0, 3.0, 4.0]
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
        """Test that string attributes are skipped with warnings."""
        builder = newton.ModelBuilder()

        builder.add_body(mass=1.0, custom_string="test_body", custom_float=25.0)

        custom_attrs = builder.custom_attributes
        self.assertEqual(set(custom_attrs.keys()), {"custom_float"})
        self.assertNotIn("custom_string", custom_attrs)

        model = builder.finalize(device=self.device)
        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(model, "custom_string"))

    def test_assignment_types(self):
        """Test custom attribute assignment to MODEL objects."""
        builder = newton.ModelBuilder()

        builder.add_body(mass=1.0, custom_float=25.0)

        custom_attrs = builder.custom_attributes
        self.assertEqual(custom_attrs["custom_float"].assignment, ModelAttributeAssignment.MODEL)

        model = builder.finalize(device=self.device)
        state = model.state()
        control = model.control()

        self.assertTrue(hasattr(model, "custom_float"))
        self.assertFalse(hasattr(state, "custom_float"))
        self.assertFalse(hasattr(control, "custom_float"))


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
