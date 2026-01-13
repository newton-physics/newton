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

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton_actuators import ActuatorPD


class TestSelection(unittest.TestCase):
    def test_no_match(self):
        builder = newton.ModelBuilder()
        builder.add_body()
        model = builder.finalize()
        self.assertRaises(KeyError, ArticulationView, model, pattern="no_match")

    def test_empty_selection(self):
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(child=body)
        builder.add_articulation([joint], key="my_articulation")
        model = builder.finalize()
        control = model.control()
        selection = ArticulationView(model, pattern="my_articulation", exclude_joint_types=[newton.JointType.FREE])
        self.assertEqual(selection.count, 1)
        self.assertEqual(selection.get_root_transforms(model).shape, (1,))
        self.assertEqual(selection.get_dof_positions(model).shape, (1, 0))
        self.assertEqual(selection.get_dof_velocities(model).shape, (1, 0))
        self.assertEqual(selection.get_dof_forces(control).shape, (1, 0))


class TestActuatorSelectionAPI(unittest.TestCase):

    def _build_single_world_model(self):
        builder = newton.ModelBuilder()
        bodies = [builder.add_body() for _ in range(3)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints, key="robot")
        dofs = [builder.joint_qd_start[j] for j in joints]
        builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=100.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[1]], kp=200.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[2]], kp=300.0)
        return builder.finalize(), dofs

    def _build_multi_world_model(self, num_worlds=3):
        template = newton.ModelBuilder()
        bodies = [template.add_body() for _ in range(3)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(template.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        template.add_articulation(joints, key="robot")
        dofs = [template.joint_qd_start[j] for j in joints]
        template.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=100.0)
        template.add_actuator(ActuatorPD, input_indices=[dofs[1]], kp=200.0)
        template.add_actuator(ActuatorPD, input_indices=[dofs[2]], kp=300.0)
        builder = newton.ModelBuilder()
        builder.replicate(template, num_worlds)
        return builder.finalize(), num_worlds

    def test_get_actuator_parameter_single_world(self):
        model, dofs = self._build_single_world_model()
        view = ArticulationView(model, pattern="robot")
        actuator = model.actuators[0]
        kp_values = view.get_actuator_parameter(actuator, "kp")
        self.assertEqual(kp_values.shape, (1, 3))
        np.testing.assert_array_almost_equal(kp_values.numpy().flatten(), [100.0, 200.0, 300.0])

    def test_get_actuator_parameter_multi_world(self):
        num_worlds = 3
        model, _ = self._build_multi_world_model(num_worlds)
        view = ArticulationView(model, pattern="robot*")
        actuator = model.actuators[0]
        kp_values = view.get_actuator_parameter(actuator, "kp")
        self.assertEqual(kp_values.shape, (num_worlds, 3))
        for w in range(num_worlds):
            np.testing.assert_array_almost_equal(kp_values.numpy()[w], [100.0, 200.0, 300.0])

    def test_set_actuator_parameter_single_world(self):
        model, dofs = self._build_single_world_model()
        view = ArticulationView(model, pattern="robot")
        actuator = model.actuators[0]
        new_kp = wp.array([[500.0, 600.0, 700.0]], dtype=float, device=model.device)
        view.set_actuator_parameter(actuator, "kp", new_kp)
        np.testing.assert_array_almost_equal(actuator.kp.numpy(), [500.0, 600.0, 700.0])

    def test_set_actuator_parameter_multi_world(self):
        num_worlds = 3
        model, _ = self._build_multi_world_model(num_worlds)
        view = ArticulationView(model, pattern="robot*")
        actuator = model.actuators[0]
        new_kp = wp.array([
            [500.0, 600.0, 700.0],
            [800.0, 900.0, 1000.0],
            [1100.0, 1200.0, 1300.0],
        ], dtype=float, device=model.device)
        view.set_actuator_parameter(actuator, "kp", new_kp)
        expected = [500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0]
        np.testing.assert_array_almost_equal(actuator.kp.numpy(), expected)

    def test_set_actuator_parameter_with_mask(self):
        num_worlds = 3
        model, _ = self._build_multi_world_model(num_worlds)
        view = ArticulationView(model, pattern="robot*")
        actuator = model.actuators[0]
        mask = wp.array([False, True, False], dtype=bool, device=model.device)
        new_kp = wp.array([
            [999.0, 999.0, 999.0],
            [500.0, 600.0, 700.0],
            [999.0, 999.0, 999.0],
        ], dtype=float, device=model.device)
        view.set_actuator_parameter(actuator, "kp", new_kp, mask=mask)
        kp_np = actuator.kp.numpy()
        np.testing.assert_array_almost_equal(kp_np[0:3], [100.0, 200.0, 300.0])
        np.testing.assert_array_almost_equal(kp_np[3:6], [500.0, 600.0, 700.0])
        np.testing.assert_array_almost_equal(kp_np[6:9], [100.0, 200.0, 300.0])

    def test_get_actuator_parameter_partial_selection(self):
        model, dofs = self._build_single_world_model()
        view = ArticulationView(model, pattern="robot", include_joints=[0, 1])
        actuator = model.actuators[0]
        kp_values = view.get_actuator_parameter(actuator, "kp")
        self.assertEqual(kp_values.shape, (1, 2))
        np.testing.assert_array_almost_equal(kp_values.numpy().flatten(), [100.0, 200.0])

    def test_set_actuator_parameter_partial_selection(self):
        model, dofs = self._build_single_world_model()
        view = ArticulationView(model, pattern="robot", include_joints=[1, 2])
        actuator = model.actuators[0]
        new_kp = wp.array([[555.0, 666.0]], dtype=float, device=model.device)
        view.set_actuator_parameter(actuator, "kp", new_kp)
        np.testing.assert_array_almost_equal(actuator.kp.numpy(), [100.0, 555.0, 666.0])

    def test_get_set_multiple_parameters(self):
        model, dofs = self._build_single_world_model()
        view = ArticulationView(model, pattern="robot")
        actuator = model.actuators[0]
        kd_values = view.get_actuator_parameter(actuator, "kd")
        np.testing.assert_array_almost_equal(kd_values.numpy().flatten(), [0.0, 0.0, 0.0])
        new_kd = wp.array([[10.0, 20.0, 30.0]], dtype=float, device=model.device)
        view.set_actuator_parameter(actuator, "kd", new_kd)
        np.testing.assert_array_almost_equal(actuator.kd.numpy(), [10.0, 20.0, 30.0])

    def test_non_actuated_dof_returns_zero(self):
        builder = newton.ModelBuilder()
        body0 = builder.add_body()
        body1 = builder.add_body()
        joint0 = builder.add_joint_revolute(parent=-1, child=body0, axis=newton.Axis.Z)
        joint1 = builder.add_joint_prismatic(parent=body0, child=body1, axis=newton.Axis.X)
        builder.add_articulation([joint0, joint1], key="robot")
        dof0 = builder.joint_qd_start[joint0]
        builder.add_actuator(ActuatorPD, input_indices=[dof0], kp=100.0)
        model = builder.finalize()
        view = ArticulationView(model, pattern="robot", include_joint_types=[newton.JointType.PRISMATIC])
        actuator = model.actuators[0]
        kp_values = view.get_actuator_parameter(actuator, "kp")
        self.assertEqual(kp_values.shape, (1, 1))
        np.testing.assert_array_almost_equal(kp_values.numpy().flatten(), [0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
