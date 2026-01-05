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

"""Tests for actuator integration with ModelBuilder."""

import math
import unittest

import numpy as np

import newton
from control import PDActuator, PIDActuator


class TestActuatorIntegration(unittest.TestCase):
    """Tests for ModelBuilder.add_actuator and Model.actuators."""

    def test_add_single_pd_actuator(self):
        """Test adding a single PD actuator to one DOF."""
        builder = newton.ModelBuilder()

        # Create simple pendulum
        body = builder.add_body()
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])

        dof = builder.joint_qd_start[joint]

        # Add actuator
        builder.add_actuator(PDActuator, input_indices=[dof], kp=100.0, kd=10.0)

        model = builder.finalize()

        # Verify
        self.assertEqual(len(model.actuators), 1)
        self.assertIsInstance(model.actuators[0], PDActuator)
        self.assertEqual(model.actuators[0].num_actuators, 1)

        kp = model.actuators[0].kp.numpy()
        kd = model.actuators[0].kd.numpy()
        self.assertAlmostEqual(kp[0], 100.0)
        self.assertAlmostEqual(kd[0], 10.0)

    def test_add_multiple_pd_actuators_same_class(self):
        """Test that multiple add_actuator calls with same class accumulate."""
        builder = newton.ModelBuilder()

        # Create 3-link arm
        body0 = builder.add_body()
        body1 = builder.add_body()
        body2 = builder.add_body()

        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = builder.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2, axis=newton.Axis.Y)
        builder.add_articulation([joint0, joint1, joint2])

        dof0 = builder.joint_qd_start[joint0]
        dof1 = builder.joint_qd_start[joint1]
        dof2 = builder.joint_qd_start[joint2]

        # Add actuators with different parameters
        builder.add_actuator(PDActuator, input_indices=[dof0], kp=50.0)
        builder.add_actuator(PDActuator, input_indices=[dof1], kp=100.0, kd=10.0)
        builder.add_actuator(PDActuator, input_indices=[dof2], kp=150.0, kd=15.0, max_force=50.0)

        model = builder.finalize()

        # Should have 1 actuator (all PDActuator calls accumulated)
        self.assertEqual(len(model.actuators), 1)

        actuator = model.actuators[0]
        self.assertIsInstance(actuator, PDActuator)
        self.assertEqual(actuator.num_actuators, 3)

        # Verify indices
        indices = actuator.input_indices.numpy()
        self.assertEqual(indices[0], dof0)
        self.assertEqual(indices[1], dof1)
        self.assertEqual(indices[2], dof2)

        # Verify parameters
        kp = actuator.kp.numpy()
        kd = actuator.kd.numpy()
        max_force = actuator.max_force.numpy()

        np.testing.assert_array_almost_equal(kp, [50.0, 100.0, 150.0])
        np.testing.assert_array_almost_equal(kd, [0.0, 10.0, 15.0])
        self.assertTrue(math.isinf(max_force[0]))
        self.assertTrue(math.isinf(max_force[1]))
        self.assertAlmostEqual(max_force[2], 50.0)

    def test_pd_actuator_default_parameters(self):
        """Test that resolve_arguments fills in defaults correctly."""
        builder = newton.ModelBuilder()

        body = builder.add_body()
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])

        dof = builder.joint_qd_start[joint]

        # Add actuator with minimal parameters
        builder.add_actuator(PDActuator, input_indices=[dof])

        model = builder.finalize()

        actuator = model.actuators[0]

        # Verify defaults: kp=0, kd=0, max_force=inf, gear=1, constant_force=0
        self.assertAlmostEqual(actuator.kp.numpy()[0], 0.0)
        self.assertAlmostEqual(actuator.kd.numpy()[0], 0.0)
        self.assertTrue(math.isinf(actuator.max_force.numpy()[0]))
        self.assertAlmostEqual(actuator.gear.numpy()[0], 1.0)
        self.assertAlmostEqual(actuator.constant_force.numpy()[0], 0.0)

    def test_no_actuators(self):
        """Test model with no actuators has empty list."""
        builder = newton.ModelBuilder()

        body = builder.add_body()
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])

        model = builder.finalize()

        self.assertEqual(len(model.actuators), 0)

    def test_actuator_gear_and_constant_force(self):
        """Test gear and constant_force parameters."""
        builder = newton.ModelBuilder()

        body = builder.add_body()
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])

        dof = builder.joint_qd_start[joint]

        builder.add_actuator(
            PDActuator,
            input_indices=[dof],
            kp=100.0,
            kd=10.0,
            gear=2.5,
            constant_force=5.0,
        )

        model = builder.finalize()

        actuator = model.actuators[0]
        self.assertAlmostEqual(actuator.gear.numpy()[0], 2.5)
        self.assertAlmostEqual(actuator.constant_force.numpy()[0], 5.0)


class TestActuatorMultiWorld(unittest.TestCase):
    """Tests for actuators with multiple worlds (vectorized simulation)."""

    def test_replicate_with_actuators(self):
        """Test that actuators work correctly with replicated worlds."""
        # Create template robot
        template = newton.ModelBuilder()

        body0 = template.add_body()
        body1 = template.add_body()

        joint0 = template.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = template.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        template.add_articulation([joint0, joint1])

        # Get template DOF indices
        template_dof0 = template.joint_qd_start[joint0]
        template_dof1 = template.joint_qd_start[joint1]

        # Add actuators to template
        template.add_actuator(PDActuator, input_indices=[template_dof0], kp=100.0, kd=10.0)
        template.add_actuator(PDActuator, input_indices=[template_dof1], kp=200.0, kd=20.0)

        # Replicate to multiple worlds
        num_worlds = 4
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        # Verify model has correct world count
        self.assertEqual(model.num_worlds, num_worlds)

        # Verify actuators were replicated
        # Each world should have 2 DOFs, so total DOFs = 2 * num_worlds
        # But actuator entries accumulate, so we have 2 actuator entries from template
        # After replicate, the builder should have accumulated all DOFs
        self.assertEqual(len(model.actuators), 1)

        actuator = model.actuators[0]
        # Template has 2 DOFs, replicated 4 times = 8 total actuated DOFs
        self.assertEqual(actuator.num_actuators, 2 * num_worlds)

        # Verify kp/kd values are replicated correctly
        kp = actuator.kp.numpy()
        kd = actuator.kd.numpy()

        # Should be [100, 200, 100, 200, 100, 200, 100, 200]
        expected_kp = np.tile([100.0, 200.0], num_worlds)
        expected_kd = np.tile([10.0, 20.0], num_worlds)

        np.testing.assert_array_almost_equal(kp, expected_kp)
        np.testing.assert_array_almost_equal(kd, expected_kd)

    def test_add_world_with_actuators(self):
        """Test actuators with add_world for explicit world control."""
        # Create template robot
        template = newton.ModelBuilder()

        body = template.add_body()
        joint = template.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        template.add_articulation([joint])

        template_dof = template.joint_qd_start[joint]
        template.add_actuator(PDActuator, input_indices=[template_dof], kp=50.0)

        # Build main model with explicit worlds
        builder = newton.ModelBuilder()
        builder.add_world(template)
        builder.add_world(template)
        builder.add_world(template)

        model = builder.finalize()

        self.assertEqual(model.num_worlds, 3)
        self.assertEqual(len(model.actuators), 1)
        self.assertEqual(model.actuators[0].num_actuators, 3)

        kp = model.actuators[0].kp.numpy()
        np.testing.assert_array_almost_equal(kp, [50.0, 50.0, 50.0])

    def test_multi_world_dof_indices(self):
        """Verify DOF indices are correctly offset per world."""
        template = newton.ModelBuilder()

        body = template.add_body()
        joint = template.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        template.add_articulation([joint])

        template_dof = template.joint_qd_start[joint]
        template.add_actuator(PDActuator, input_indices=[template_dof], kp=100.0)

        # Replicate
        num_worlds = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        actuator = model.actuators[0]
        indices = actuator.input_indices.numpy()

        self.assertEqual(len(indices), num_worlds)

        # Get the DOFs per world from the model
        dofs_per_world = model.joint_dof_count // num_worlds

        # Indices should be offset by dofs_per_world for each world
        for i in range(1, num_worlds):
            expected_offset = dofs_per_world
            actual_offset = indices[i] - indices[i - 1]
            self.assertEqual(actual_offset, expected_offset)


class TestMixedActuatorTypes(unittest.TestCase):
    """Tests for multiple different actuator types."""

    def test_alternating_actuator_types(self):
        """Test PD and PID actuators in alternating order."""
        builder = newton.ModelBuilder()

        # Create 4-link arm
        body0 = builder.add_body()
        body1 = builder.add_body()
        body2 = builder.add_body()
        body3 = builder.add_body()

        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = builder.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        joint2 = builder.add_joint_revolute(parent=body1, child=body2, axis=newton.Axis.Y)
        joint3 = builder.add_joint_revolute(parent=body2, child=body3, axis=newton.Axis.X)
        builder.add_articulation([joint0, joint1, joint2, joint3])

        dof0 = builder.joint_qd_start[joint0]
        dof1 = builder.joint_qd_start[joint1]
        dof2 = builder.joint_qd_start[joint2]
        dof3 = builder.joint_qd_start[joint3]

        # Add actuators in alternating order: PD, PID, PD, PID
        builder.add_actuator(PDActuator, input_indices=[dof0], kp=100.0, kd=10.0)
        builder.add_actuator(PIDActuator, input_indices=[dof1], kp=200.0, ki=5.0, kd=20.0)
        builder.add_actuator(PDActuator, input_indices=[dof2], kp=300.0, kd=30.0)
        builder.add_actuator(PIDActuator, input_indices=[dof3], kp=400.0, ki=10.0, kd=40.0)

        model = builder.finalize()

        # Should have 2 actuators (one PD, one PID)
        self.assertEqual(len(model.actuators), 2)

        # Find PD and PID actuators
        pd_actuator = None
        pid_actuator = None
        for act in model.actuators:
            if isinstance(act, PDActuator) and not isinstance(act, PIDActuator):
                pd_actuator = act
            elif isinstance(act, PIDActuator):
                pid_actuator = act

        self.assertIsNotNone(pd_actuator)
        self.assertIsNotNone(pid_actuator)

        # PD actuator should have DOFs 0 and 2
        self.assertEqual(pd_actuator.num_actuators, 2)
        pd_indices = pd_actuator.input_indices.numpy()
        self.assertEqual(pd_indices[0], dof0)
        self.assertEqual(pd_indices[1], dof2)

        pd_kp = pd_actuator.kp.numpy()
        np.testing.assert_array_almost_equal(pd_kp, [100.0, 300.0])

        # PID actuator should have DOFs 1 and 3
        self.assertEqual(pid_actuator.num_actuators, 2)
        pid_indices = pid_actuator.input_indices.numpy()
        self.assertEqual(pid_indices[0], dof1)
        self.assertEqual(pid_indices[1], dof3)

        pid_kp = pid_actuator.kp.numpy()
        pid_ki = pid_actuator.ki.numpy()
        np.testing.assert_array_almost_equal(pid_kp, [200.0, 400.0])
        np.testing.assert_array_almost_equal(pid_ki, [5.0, 10.0])

    def test_mixed_actuators_with_replication(self):
        """Test mixed actuator types with multi-world replication."""
        template = newton.ModelBuilder()

        # Create 2-link arm
        body0 = template.add_body()
        body1 = template.add_body()

        joint0 = template.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = template.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        template.add_articulation([joint0, joint1])

        template_dof0 = template.joint_qd_start[joint0]
        template_dof1 = template.joint_qd_start[joint1]

        # Joint 0 uses PD, Joint 1 uses PID
        template.add_actuator(PDActuator, input_indices=[template_dof0], kp=100.0, kd=10.0)
        template.add_actuator(PIDActuator, input_indices=[template_dof1], kp=200.0, ki=5.0, kd=20.0)

        # Replicate to 3 worlds
        num_worlds = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        # Should have 2 actuator types
        self.assertEqual(len(model.actuators), 2)

        # Find actuators by type
        pd_actuator = None
        pid_actuator = None
        for act in model.actuators:
            if isinstance(act, PDActuator) and not isinstance(act, PIDActuator):
                pd_actuator = act
            elif isinstance(act, PIDActuator):
                pid_actuator = act

        self.assertIsNotNone(pd_actuator)
        self.assertIsNotNone(pid_actuator)

        # Each actuator type should control num_worlds DOFs
        self.assertEqual(pd_actuator.num_actuators, num_worlds)
        self.assertEqual(pid_actuator.num_actuators, num_worlds)

        # Verify PD parameters are replicated
        pd_kp = pd_actuator.kp.numpy()
        np.testing.assert_array_almost_equal(pd_kp, [100.0] * num_worlds)

        # Verify PID parameters are replicated
        pid_kp = pid_actuator.kp.numpy()
        pid_ki = pid_actuator.ki.numpy()
        np.testing.assert_array_almost_equal(pid_kp, [200.0] * num_worlds)
        np.testing.assert_array_almost_equal(pid_ki, [5.0] * num_worlds)

        # Verify indices are correctly offset per world
        pd_indices = pd_actuator.input_indices.numpy()
        pid_indices = pid_actuator.input_indices.numpy()

        dofs_per_world = model.joint_dof_count // num_worlds

        for i in range(1, num_worlds):
            # PD indices should be offset by dofs_per_world
            self.assertEqual(pd_indices[i] - pd_indices[i - 1], dofs_per_world)
            # PID indices should be offset by dofs_per_world
            self.assertEqual(pid_indices[i] - pid_indices[i - 1], dofs_per_world)

        # PID indices should be 1 more than PD indices within each world
        # (joint1 DOF comes after joint0 DOF)
        for i in range(num_worlds):
            self.assertEqual(pid_indices[i] - pd_indices[i], 1)


class TestActuatorInputOutputIndices(unittest.TestCase):
    """Tests for separate input and output indices."""

    def test_same_input_output_indices(self):
        """Test that omitting output_index uses input_index."""
        builder = newton.ModelBuilder()

        body = builder.add_body()
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=newton.Axis.Z)
        builder.add_articulation([joint])

        dof = builder.joint_qd_start[joint]
        builder.add_actuator(PDActuator, input_indices=[dof], kp=100.0)

        model = builder.finalize()

        actuator = model.actuators[0]
        input_idx = actuator.input_indices.numpy()[0]
        output_idx = actuator.output_indices.numpy()[0]

        self.assertEqual(input_idx, dof)
        self.assertEqual(output_idx, dof)

    def test_different_input_output_indices(self):
        """Test actuator with different input and output indices."""
        builder = newton.ModelBuilder()

        # Create 2 bodies with 2 joints
        body0 = builder.add_body()
        body1 = builder.add_body()

        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = builder.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        builder.add_articulation([joint0, joint1])

        dof0 = builder.joint_qd_start[joint0]
        dof1 = builder.joint_qd_start[joint1]

        # Read from dof0, write to dof1 (e.g., coupled actuator)
        builder.add_actuator(PDActuator, input_indices=[dof0], output_indices=[dof1], kp=100.0)

        model = builder.finalize()

        actuator = model.actuators[0]
        input_idx = actuator.input_indices.numpy()[0]
        output_idx = actuator.output_indices.numpy()[0]

        self.assertEqual(input_idx, dof0)
        self.assertEqual(output_idx, dof1)

    def test_mixed_input_output_with_replication(self):
        """Test different input/output indices replicate correctly."""
        template = newton.ModelBuilder()

        body0 = template.add_body()
        body1 = template.add_body()

        joint0 = template.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = template.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        template.add_articulation([joint0, joint1])

        dof0 = template.joint_qd_start[joint0]
        dof1 = template.joint_qd_start[joint1]

        # Read from dof0, write to dof1
        template.add_actuator(PDActuator, input_indices=[dof0], output_indices=[dof1], kp=100.0)

        # Replicate
        num_worlds = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        actuator = model.actuators[0]
        self.assertEqual(actuator.num_actuators, num_worlds)

        input_indices = actuator.input_indices.numpy()
        output_indices = actuator.output_indices.numpy()

        dofs_per_world = model.joint_dof_count // num_worlds

        # Verify offset is consistent for both input and output
        for i in range(1, num_worlds):
            self.assertEqual(input_indices[i] - input_indices[i - 1], dofs_per_world)
            self.assertEqual(output_indices[i] - output_indices[i - 1], dofs_per_world)

        # Verify output is 1 more than input within each world
        for i in range(num_worlds):
            self.assertEqual(output_indices[i] - input_indices[i], 1)


class TestActuatorUSDParsing(unittest.TestCase):
    """Tests for parsing actuators from USD files."""

    def test_parse_usd_with_actuators(self):
        """Test that actuators are parsed from USD and added to the model."""
        import os

        from control import parse_actuator_prim
        from newton._src.utils.import_usd import parse_usd

        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        builder = newton.ModelBuilder()

        result = parse_usd(builder, usd_path, parse_actuator_fn=parse_actuator_prim)

        self.assertGreaterEqual(result["actuator_count"], 0)

        model = builder.finalize()
        self.assertGreater(len(model.actuators), 0)

    def test_parse_usd_actuator_parameters(self):
        """Test that actuator parameters are correctly parsed from USD."""
        import os

        from control import parse_actuator_prim
        from newton._src.utils.import_usd import parse_usd
        from pxr import Usd

        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        # Parse USD directly to check actuator parameters
        stage = Usd.Stage.Open(usd_path)
        actuator_prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")
        parsed = parse_actuator_prim(actuator_prim)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kwargs.get("kp"), 100.0)
        self.assertEqual(parsed.kwargs.get("kd"), 10.0)

    def test_parse_usd_without_parse_fn(self):
        """Test that USD parsing works normally without parse function."""
        import os

        from newton._src.utils.import_usd import parse_usd

        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        builder = newton.ModelBuilder()

        result = parse_usd(builder, usd_path)

        self.assertEqual(result["actuator_count"], 0)

        model = builder.finalize()
        self.assertEqual(len(model.actuators), 0)


if __name__ == "__main__":
    unittest.main()

