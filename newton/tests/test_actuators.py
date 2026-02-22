# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
import os
import unittest

import numpy as np
import warp as wp
from newton_actuators import ActuatorDelayedPD, ActuatorPD, ActuatorPID, parse_actuator_prim

import newton
from newton._src.utils.import_usd import parse_usd
from newton.selection import ArticulationView

try:
    from pxr import Usd

    HAS_USD = True
except ImportError:
    HAS_USD = False


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

        self.assertEqual(model.world_count, num_worlds)
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

    def test_multi_input_actuator_2d_indices(self):
        """Test actuators with multiple input indices (2D index arrays)."""
        builder = newton.ModelBuilder()

        # Create 6 joints for testing
        bodies = [builder.add_body() for _ in range(6)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        # Add multi-input actuators: each reads from 2 DOFs
        builder.add_actuator(ActuatorPD, input_indices=[dofs[0], dofs[1]], kp=100.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[2], dofs[3]], kp=200.0)
        builder.add_actuator(ActuatorPD, input_indices=[dofs[4], dofs[5]], kp=300.0)

        model = builder.finalize()

        # Should have 1 ActuatorPD instance with 3 actuators, each with 2 inputs
        self.assertEqual(len(model.actuators), 1)
        pd_act = model.actuators[0]

        self.assertEqual(pd_act.num_actuators, 3)

        # Verify input_indices shape is 2D: (3, 2)
        input_arr = pd_act.input_indices.numpy()
        self.assertEqual(input_arr.shape, (3, 2))

        # Verify the indices are correct
        np.testing.assert_array_equal(input_arr[0], [dofs[0], dofs[1]])
        np.testing.assert_array_equal(input_arr[1], [dofs[2], dofs[3]])
        np.testing.assert_array_equal(input_arr[2], [dofs[4], dofs[5]])

        # Verify output_indices also has same shape
        output_arr = pd_act.output_indices.numpy()
        self.assertEqual(output_arr.shape, (3, 2))

        # Verify kp array has correct values (one per actuator)
        np.testing.assert_array_almost_equal(pd_act.kp.numpy(), [100.0, 200.0, 300.0])

    def test_dimension_mismatch_raises_error(self):
        """Test that mixing different input dimensions raises an error."""
        builder = newton.ModelBuilder()

        bodies = [builder.add_body() for _ in range(3)]
        joints = []
        for i, body in enumerate(bodies):
            parent = -1 if i == 0 else bodies[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=body, axis=newton.Axis.Z))
        builder.add_articulation(joints)

        dofs = [builder.joint_qd_start[j] for j in joints]

        # First call: 1 input
        builder.add_actuator(ActuatorPD, input_indices=[dofs[0]], kp=100.0)

        # Second call: 2 inputs - should raise
        with self.assertRaises(ValueError) as ctx:
            builder.add_actuator(ActuatorPD, input_indices=[dofs[1], dofs[2]], kp=200.0)

        self.assertIn("dimension mismatch", str(ctx.exception))


@unittest.skipUnless(HAS_USD, "pxr not installed")
class TestActuatorUSDParsing(unittest.TestCase):
    """Tests for parsing actuators from USD files."""

    def test_usd_parsing(self):
        """Test that USD parsing automatically parses Newton actuators."""
        test_dir = os.path.dirname(__file__)
        usd_path = os.path.join(test_dir, "assets", "actuator_test.usda")

        if not os.path.exists(usd_path):
            self.skipTest(f"Test USD file not found: {usd_path}")

        # Actuators are parsed automatically
        builder = newton.ModelBuilder()
        result = parse_usd(builder, usd_path)
        self.assertGreater(result["actuator_count"], 0)
        model = builder.finalize()
        self.assertGreater(len(model.actuators), 0)

        # Verify parsed parameters
        stage = Usd.Stage.Open(usd_path)
        actuator_prim = stage.GetPrimAtPath("/World/Robot/Joint1Actuator")
        parsed = parse_actuator_prim(actuator_prim)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kwargs.get("kp"), 100.0)
        self.assertEqual(parsed.kwargs.get("kd"), 10.0)


class TestActuatorSelectionAPI(unittest.TestCase):
    """Tests for actuator parameter access via ArticulationView.

    Follows the same parameterised pattern as the joint/link selection tests:
    a single ``run_test_actuator_selection`` helper is driven by four thin
    entry-point tests that cover (use_mask x use_multiple_artics_per_view).
    """

    def run_test_actuator_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        """Test an ArticulationView that includes a subset of joints and that we
        can read/write actuator parameters for the subset with and without a mask.
        Verifies the full flat actuator parameter array."""

        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
      <!-- Third child link with prismatic joint along x -->
      <body name="link3" pos="-0.0 -0.9 0">
        <joint name="joint3" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

        num_joints_per_articulation = 3
        num_articulations_per_world = 2
        num_worlds = 3
        num_actuators = num_joints_per_articulation * num_articulations_per_world * num_worlds

        # Create a single articulation with 3 joints.
        single_articulation_builder = newton.ModelBuilder()
        single_articulation_builder.add_mjcf(mjcf)

        # Add an ActuatorPD for each slide joint: kp = 100, 200, 300
        joint_names = [
            "myart/worldbody/root/link1/joint1",
            "myart/worldbody/root/link2/joint2",
            "myart/worldbody/root/link3/joint3",
        ]
        for i, jname in enumerate(joint_names):
            j_idx = single_articulation_builder.joint_label.index(jname)
            dof = single_articulation_builder.joint_qd_start[j_idx]
            single_articulation_builder.add_actuator(ActuatorPD, input_indices=[dof], kp=100.0 * (i + 1))

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(num_articulations_per_world):
            single_world_builder.add_builder(single_articulation_builder)

        # Customise the articulation labels in single_world_builder
        single_world_builder.articulation_label[1] = "art1"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_label[0] = "art1"
        else:
            single_world_builder.articulation_label[0] = "art0"

        # Create 3 worlds with two articulations per world and 3 actuators per articulation.
        builder = newton.ModelBuilder()
        for _i in range(num_worlds):
            builder.add_world(single_world_builder)

        # Create the model
        model = builder.finalize()

        # Create a view of "art1/joint3"
        joints_to_include = ["joint3"]
        joint_view = ArticulationView(model, "art1", include_joints=joints_to_include)

        actuator = model.actuators[0]

        # Get the kp values for the view's DOFs (only joint3 of selected artics)
        kp_values = joint_view.get_actuator_parameter(actuator, "kp").numpy().copy()

        # Verify shape and initial values
        if use_multiple_artics_per_view:
            self.assertEqual(kp_values.shape, (num_worlds, 2))
            np.testing.assert_array_almost_equal(kp_values, [[300.0, 300.0]] * num_worlds)
        else:
            self.assertEqual(kp_values.shape, (num_worlds, 1))
            np.testing.assert_array_almost_equal(kp_values, [[300.0]] * num_worlds)

        # Modify the kp values with distinguishable per-slot values
        val = 1000.0
        for world_idx in range(kp_values.shape[0]):
            for dof_idx in range(kp_values.shape[1]):
                kp_values[world_idx, dof_idx] = val
                val += 100.0

        mask = None
        if use_mask:
            mask = wp.array([False, True, False], dtype=bool, device=model.device)

        # Set the modified values
        wp_kp = wp.array(kp_values, dtype=float, device=model.device)
        joint_view.set_actuator_parameter(actuator, "kp", wp_kp, mask=mask)

        # Build expected flat kp array and verify against the full actuator.kp array.
        # Initial flat layout: [100, 200, 300] per articulation x 6 articulations.
        expected_kp = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,  # world0/artic0 — not masked
                    100.0,
                    200.0,
                    300.0,  # world0/artic1 — not masked
                    100.0,
                    200.0,
                    1200.0,  # world1/artic0 — masked, joint3 updated
                    100.0,
                    200.0,
                    1300.0,  # world1/artic1 — masked, joint3 updated
                    100.0,
                    200.0,
                    300.0,  # world2/artic0 — not masked
                    100.0,
                    200.0,
                    300.0,  # world2/artic1 — not masked
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,  # world0/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    300.0,  # world0/artic1 (art1) — not masked
                    100.0,
                    200.0,
                    300.0,  # world1/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    1100.0,  # world1/artic1 (art1) — masked, joint3 updated
                    100.0,
                    200.0,
                    300.0,  # world2/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    300.0,  # world2/artic1 (art1) — not masked
                ]
        else:
            if use_multiple_artics_per_view:
                expected_kp = [
                    100.0,
                    200.0,
                    1000.0,  # world0/artic0 — joint3 updated
                    100.0,
                    200.0,
                    1100.0,  # world0/artic1 — joint3 updated
                    100.0,
                    200.0,
                    1200.0,  # world1/artic0 — joint3 updated
                    100.0,
                    200.0,
                    1300.0,  # world1/artic1 — joint3 updated
                    100.0,
                    200.0,
                    1400.0,  # world2/artic0 — joint3 updated
                    100.0,
                    200.0,
                    1500.0,  # world2/artic1 — joint3 updated
                ]
            else:
                expected_kp = [
                    100.0,
                    200.0,
                    300.0,  # world0/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    1000.0,  # world0/artic1 (art1) — joint3 updated
                    100.0,
                    200.0,
                    300.0,  # world1/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    1100.0,  # world1/artic1 (art1) — joint3 updated
                    100.0,
                    200.0,
                    300.0,  # world2/artic0 (art0) — not in view
                    100.0,
                    200.0,
                    1200.0,  # world2/artic1 (art1) — joint3 updated
                ]

        # Verify the full flat actuator kp array
        measured_kp = actuator.kp.numpy()
        for i in range(num_actuators):
            self.assertAlmostEqual(
                expected_kp[i],
                measured_kp[i],
                places=4,
                msg=f"Expected kp value {i}: {expected_kp[i]}, Measured value: {measured_kp[i]}",
            )

    def test_actuator_selection_one_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_no_mask(self):
        self.run_test_actuator_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_actuator_selection_one_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_actuator_selection_two_per_view_with_mask(self):
        self.run_test_actuator_selection(use_mask=True, use_multiple_artics_per_view=True)


class TestActuatorStepIntegration(unittest.TestCase):
    """Tests for actuator.step() with real Model/State/Control objects.

    Verifies the end-to-end flow: set targets on Control -> call actuator.step()
    -> forces written to control.joint_f.

    Uses add_link() (not add_body()) to avoid implicit free joints that would
    cause DOF/coord index mismatch when the actuator indexes into joint_q.
    """

    def _build_chain_model(self, num_joints, actuator_cls, actuator_kwargs):
        """Helper: build a revolute chain with add_link, add one actuator per joint, finalize."""
        builder = newton.ModelBuilder()
        links = [builder.add_link() for _ in range(num_joints)]
        joints = []
        for i, link in enumerate(links):
            parent = -1 if i == 0 else links[i - 1]
            joints.append(builder.add_joint_revolute(parent=parent, child=link, axis=newton.Axis.Z))
        builder.add_articulation(joints)
        dofs = [builder.joint_qd_start[j] for j in joints]
        for dof in dofs:
            builder.add_actuator(actuator_cls, input_indices=[dof], **actuator_kwargs)
        return builder.finalize(), dofs

    def _set_control_array(self, model, control_array, dof_indices, values):
        """Write values into a control array at the given DOF indices."""
        arr_np = control_array.numpy()
        for dof, val in zip(dof_indices, values, strict=True):
            arr_np[dof] = val
        wp.copy(control_array, wp.array(arr_np, dtype=float, device=model.device))

    def test_pd_step_position_error(self):
        """ActuatorPD: force = kp * (target_pos - q) when kd=0, gear=1."""
        model, dofs = self._build_chain_model(3, ActuatorPD, {"kp": 100.0})
        state = model.state()
        control = model.control()
        control.joint_f.zero_()

        self._set_control_array(model, control.joint_target_pos, dofs, [1.0, 2.0, 3.0])

        actuator = model.actuators[0]
        actuator.step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose(
            [forces[d] for d in dofs],
            [100.0, 200.0, 300.0],
            rtol=1e-5,
        )

    def test_pd_step_velocity_error(self):
        """ActuatorPD: force includes kd * (target_vel - qd)."""
        model, dofs = self._build_chain_model(2, ActuatorPD, {"kp": 0.0, "kd": 10.0})
        state = model.state()
        control = model.control()
        control.joint_f.zero_()

        self._set_control_array(model, control.joint_target_vel, dofs, [5.0, -3.0])

        actuator = model.actuators[0]
        actuator.step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose(
            [forces[d] for d in dofs],
            [50.0, -30.0],
            rtol=1e-5,
        )

    def test_pd_step_feedforward_joint_act(self):
        """ActuatorPD: feedforward joint_act term is added to the output force."""
        model, dofs = self._build_chain_model(2, ActuatorPD, {"kp": 0.0})
        state = model.state()
        control = model.control()
        control.joint_f.zero_()

        self._set_control_array(model, control.joint_act, dofs, [7.0, -3.0])

        actuator = model.actuators[0]
        actuator.step(state, control)

        forces = control.joint_f.numpy()
        np.testing.assert_allclose(
            [forces[d] for d in dofs],
            [7.0, -3.0],
            rtol=1e-5,
        )

    def test_pd_step_gear_and_clamp(self):
        """ActuatorPD: gear ratio scales error and force is clamped to max_force."""
        model, dofs = self._build_chain_model(1, ActuatorPD, {"kp": 100.0, "gear": 2.0, "max_force": 50.0})
        state = model.state()
        control = model.control()
        control.joint_f.zero_()

        self._set_control_array(model, control.joint_target_pos, dofs, [1.0])

        actuator = model.actuators[0]
        actuator.step(state, control)

        # gear=2, q=0: force = 2 * (100 * (1 - 2*0)) = 200, clamped to 50
        forces = control.joint_f.numpy()
        self.assertAlmostEqual(forces[dofs[0]], 50.0, places=5)

    def test_pd_step_constant_force(self):
        """ActuatorPD: constant_force offset is included in output."""
        model, dofs = self._build_chain_model(1, ActuatorPD, {"kp": 0.0, "constant_force": 42.0})
        state = model.state()
        control = model.control()
        control.joint_f.zero_()

        actuator = model.actuators[0]
        actuator.step(state, control)

        forces = control.joint_f.numpy()
        self.assertAlmostEqual(forces[dofs[0]], 42.0, places=5)

    def test_pid_step_integral_accumulation(self):
        """ActuatorPID: integral term accumulates over multiple steps."""
        model, dofs = self._build_chain_model(1, ActuatorPID, {"kp": 0.0, "ki": 10.0, "kd": 0.0})
        state = model.state()
        control = model.control()

        self._set_control_array(model, control.joint_target_pos, dofs, [1.0])

        actuator = model.actuators[0]
        state_a = actuator.state()
        state_b = actuator.state()
        dt = 0.01

        forces_over_time = []
        for step_i in range(3):
            control.joint_f.zero_()
            if step_i % 2 == 0:
                current, nxt = state_a, state_b
            else:
                current, nxt = state_b, state_a
            actuator.step(state, control, current, nxt, dt)
            forces_over_time.append(control.joint_f.numpy()[dofs[0]])

        # integral after step 0: 1.0 * 0.01 = 0.01, force = ki * integral = 0.1
        # integral after step 1: 0.01 + 0.01 = 0.02, force = 0.2
        # integral after step 2: 0.02 + 0.01 = 0.03, force = 0.3
        np.testing.assert_allclose(forces_over_time, [0.1, 0.2, 0.3], rtol=1e-4)

    def test_delayed_pd_step_delay_behavior(self):
        """ActuatorDelayedPD: targets are delayed by N steps with real Model objects."""
        delay = 2
        model, dofs = self._build_chain_model(1, ActuatorDelayedPD, {"kp": 1.0, "delay": delay})
        state = model.state()

        actuator = model.actuators[0]
        state_a = actuator.state()
        state_b = actuator.state()
        dt = 0.01

        force_history = []
        for step_i in range(delay + 2):
            control = model.control()
            control.joint_f.zero_()
            target_val = float(step_i + 1) * 10.0
            self._set_control_array(model, control.joint_target_pos, dofs, [target_val])

            if step_i % 2 == 0:
                current, nxt = state_a, state_b
            else:
                current, nxt = state_b, state_a
            actuator.step(state, control, current, nxt, dt)
            force_history.append(control.joint_f.numpy()[dofs[0]])

        # Steps 0..(delay-1) produce zero (buffer filling)
        for i in range(delay):
            self.assertEqual(force_history[i], 0.0, f"Step {i}: expected 0 during fill phase")

        # Step 2 uses target from step 0 (10.0), step 3 uses target from step 1 (20.0)
        self.assertAlmostEqual(force_history[delay], 10.0, places=4)
        self.assertAlmostEqual(force_history[delay + 1], 20.0, places=4)


if __name__ == "__main__":
    unittest.main()
