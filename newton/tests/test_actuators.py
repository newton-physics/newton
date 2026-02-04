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

try:
    from newton_actuators import ActuatorDelayedPD, ActuatorPD, ActuatorPID

    HAS_ACTUATORS = True
except ImportError:
    HAS_ACTUATORS = False


@unittest.skipUnless(HAS_ACTUATORS, "newton-actuators not installed")
class TestActuatorBuilder(unittest.TestCase):
    """Tests for ModelBuilder.add_actuator - functionality, multi-world, and scalar params."""

    def test_accumulation_and_parameters(self):
        """Test actuator accumulation, parameters, defaults, and input/output indices."""
        builder = newton.ModelBuilder()

        bodies = [builder.add_body() for _ in range(3)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        # Add actuators - should accumulate into one
        builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=50.0, gear=2.5, constant_force=1.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[1]], kp=100.0, kd=10.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[1]], output_indices=[dofs[2]], kp=150.0, max_force=50.0)

        model = builder.finalize()

        self.assertEqual(len(model.actuators), 1)
        act = model.actuators[0]
        self.assertEqual(act.num_actuators, 3)

        np.testing.assert_array_equal(act.input_indices.numpy(), [dofs[0], dofs[1], dofs[1]])
        np.testing.assert_array_equal(act.output_indices.numpy(), [dofs[0], dofs[1], dofs[2]])
        np.testing.assert_array_almost_equal(act.kp.numpy(), [50.0, 100.0, 150.0])
        np.testing.assert_array_almost_equal(act.kd.numpy(), [0.0, 10.0, 0.0])
        self.assertAlmostEqual(act.gear.numpy()[0], 2.5)
        self.assertAlmostEqual(act.constant_force.numpy()[0], 1.0)
        self.assertAlmostEqual(act.max_force.numpy()[2], 50.0)
        self.assertTrue(math.isinf(act.max_force.numpy()[0]))

    def test_mixed_types_with_replication(self):
        """Test mixed actuator types, replication, DOF offsets, and input/output indices."""
        template = newton.ModelBuilder()

        body0 = template.add_body()
        body1 = template.add_body()
        body2 = template.add_body()

        joint0 = template.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = template.add_joint_revolute(parent=body0, child=body1, axis=newton.Axis.Y)
        joint2 = template.add_joint_revolute(parent=body1, child=body2, axis=newton.Axis.X)
        template.add_articulation([joint0, joint1, joint2])

        dof0 = template.joint_qd_start[joint0]
        dof1 = template.joint_qd_start[joint1]
        dof2 = template.joint_qd_start[joint2]

        template.add_actuator(ActuatorPD, input_indices=[dof0], kp=100.0, kd=10.0)
        template.add_actuator(ActuatorPID, input_indices=[dof1], kp=200.0, ki=5.0, kd=20.0)
        template.add_actuator(ActuatorPD, input_indices=[dof1], output_indices=[dof2], kp=300.0)

        num_worlds = 3
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)

        model = builder.finalize()

        self.assertEqual(model.num_worlds, num_worlds)
        self.assertEqual(len(model.actuators), 2)

        pd_act = next(a for a in model.actuators if type(a) is ActuatorPD)
        pid_act = next(a for a in model.actuators if isinstance(a, ActuatorPID))

        self.assertEqual(pd_act.num_actuators, 2 * num_worlds)
        self.assertEqual(pid_act.num_actuators, num_worlds)

        np.testing.assert_array_almost_equal(pd_act.kp.numpy(), [100.0, 300.0] * num_worlds)
        np.testing.assert_array_almost_equal(pid_act.ki.numpy(), [5.0] * num_worlds)

        pd_in = pd_act.input_indices.numpy()
        pd_out = pd_act.output_indices.numpy()
        dofs_per_world = model.joint_dof_count // num_worlds

        for w in range(1, num_worlds):
            self.assertEqual(pd_in[w * 2] - pd_in[(w - 1) * 2], dofs_per_world)
            self.assertEqual(pd_in[w * 2 + 1] - pd_in[(w - 1) * 2 + 1], dofs_per_world)
            self.assertEqual(pd_out[w * 2 + 1] - pd_out[(w - 1) * 2 + 1], dofs_per_world)

        for w in range(num_worlds):
            self.assertNotEqual(pd_in[w * 2 + 1], pd_out[w * 2 + 1])

    def test_delay_grouping(self):
        """Test: same delay groups, different delays separate, mixed with simple PD."""
        builder = newton.ModelBuilder()

        bodies = [builder.add_body() for _ in range(6)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=100.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[1]], kp=150.0)
        builder.add_actuator(ActuatorDelayedPD, input_indices=[dofs[2]], kp=200.0, delay=3)
        builder.add_actuator(ActuatorDelayedPD, input_indices=[dofs[3]], kp=250.0, delay=3)
        builder.add_actuator(ActuatorDelayedPD, input_indices=[dofs[4]], kp=300.0, delay=7)
        builder.add_actuator(ActuatorDelayedPD, input_indices=[dofs[5]], kp=350.0, delay=7)

        model = builder.finalize()

        self.assertEqual(len(model.actuators), 3)

        pd_act = next(a for a in model.actuators if type(a) is ActuatorPD)
        delay3 = next(a for a in model.actuators if isinstance(a, ActuatorDelayedPD) and a.delay == 3)
        delay7 = next(a for a in model.actuators if isinstance(a, ActuatorDelayedPD) and a.delay == 7)

        self.assertEqual(pd_act.num_actuators, 2)
        self.assertEqual(delay3.num_actuators, 2)
        self.assertEqual(delay7.num_actuators, 2)

        np.testing.assert_array_almost_equal(delay3.kp.numpy(), [200.0, 250.0])


@unittest.skipUnless(HAS_ACTUATORS, "newton-actuators not installed")
class TestActuatorUSDParsing(unittest.TestCase):
    """Tests for parsing actuators from USD files."""

    def test_usd_parsing(self):
        """Test USD parsing with and without actuator parse function, verify parameters."""
        import os

        from newton_actuators import parse_actuator_prim
        from pxr import Usd

        from newton._src.utils.import_usd import parse_usd

        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        # Without parse function - no actuators
        builder1 = newton.ModelBuilder()
        result1 = parse_usd(builder1, usd_path)
        self.assertEqual(result1["actuator_count"], 0)
        model1 = builder1.finalize()
        self.assertEqual(len(model1.actuators), 0)

        # With parse function - actuators parsed
        builder2 = newton.ModelBuilder()
        result2 = parse_usd(builder2, usd_path, parse_actuator_fn=parse_actuator_prim)
        self.assertGreaterEqual(result2["actuator_count"], 0)
        model2 = builder2.finalize()
        self.assertGreater(len(model2.actuators), 0)

        # Verify parsed parameters
        stage = Usd.Stage.Open(usd_path)
        actuator_prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")
        parsed = parse_actuator_prim(actuator_prim)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kwargs.get("kp"), 100.0)
        self.assertEqual(parsed.kwargs.get("kd"), 10.0)


if __name__ == "__main__":
    unittest.main()
