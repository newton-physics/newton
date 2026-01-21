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
from enum import IntEnum

import newton
from newton.solvers import SolverMuJoCo


class TestMujocoFixedTendon(unittest.TestCase):
    class LimitBreachType(IntEnum):
        UPPER_LIMIT_FROM_ABOVE = 0
        UPPER_LIMIT_FROM_BELOW = 1
        LOWER_LIMIT_FROM_BELOW = 2
        LOWER_LIMIT_FROM_ABOVE = 3

    def test_single_mujoco_fixed_tendon_length_behaviour(self):
        """Test that tendon length works as expected"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon"
		stiffness="2"
		damping="1"
		springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>

</mujoco>

"""

        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf)
        builder = newton.ModelBuilder(gravity=0.0)
        for i in range(0, 2):
            builder.add_world(individual_builder)
        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        dt = 0.02

        coeff0 = 1.0  # from mjcf above
        coeff1 = 1.0  # from mjcf above
        expected_tendon_length = 0.0  # from mjcf above

        # Length of tendon at start is: pos*coef0 + pos1*coef1 = 2*0.5 + 0*0.0 = 1.0
        # Target length is 0.0 (see mjcf above)
        joint_start_positions = [0.5, 0.0, 0.5, 0.0]
        state_in.joint_q.assign(joint_start_positions)

        for _i in range(0, 200):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        # World 0 should have achieved the rest length of the tendon.
        joint_q = state_in.joint_q.numpy()
        q0 = joint_q[0]
        q1 = joint_q[1]
        measured_tendon_length = coeff0 * q0 + coeff1 * q1
        self.assertAlmostEqual(
            expected_tendon_length,
            measured_tendon_length,
            places=3,
            msg=f"Expected tendon length: {expected_tendon_length}, Measured tendon length: {measured_tendon_length}",
        )

        # World 1 and world 0 should have identical state.
        q2 = joint_q[2]
        self.assertAlmostEqual(
            q2,
            q0,
            places=3,
            msg=f"Expected joint_q[2]: {q0}, Measured q2: {q2}",
        )
        q3 = joint_q[3]
        self.assertAlmostEqual(
            q3,
            q1,
            places=3,
            msg=f"Expected joint_q[3]: {q1}, Measured q3: {q3}",
        )

    def run_test_mujoco_fixed_tendon_limit_behavior(self, mode: LimitBreachType):
        """Test that tendons limits are respected"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>

      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom solmix="1.0" type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <!-- Fixed tendon coupling joint1 and joint2 -->
	<fixed
		name="coupling_tendon"
    range = "-1.0 1.0"
		stiffness="0.0"
		damping="0.0"
        solreflimit="0.004 1"
        solimplimit="0.95 0.99 0.001"        
		springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>

</mujoco>

"""
        coeff0 = 1.0  # from mjcf above
        coeff1 = 1.0  # from mjcf above
        lower_limit = -1.0  # from mjcf above
        upper_limit = 1.0  # from mjcf above

        # Configure the start state of world 0
        joint_start_positions = [0.0, 0.0, 0.0, 0.0]
        joint_start_velocities = [0.0, 0.0, 0.0, 0.0]
        if mode is self.LimitBreachType.UPPER_LIMIT_FROM_ABOVE:
            joint_start_positions[0] = upper_limit + 0.1
        elif mode is self.LimitBreachType.UPPER_LIMIT_FROM_BELOW:
            joint_start_positions[0] = upper_limit - 0.1
            joint_start_velocities[0] = 1.0
        elif mode is self.LimitBreachType.LOWER_LIMIT_FROM_BELOW:
            joint_start_positions[0] = lower_limit - 0.1
        elif mode is self.LimitBreachType.LOWER_LIMIT_FROM_ABOVE:
            joint_start_positions[0] = lower_limit + 0.1
            joint_start_velocities[0] = -1.0

        # Configure the start state of world 1 to be identical
        # to that of world 0.
        joint_start_positions[2] = joint_start_positions[0]
        joint_start_velocities[2] = joint_start_velocities[0]

        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(mjcf)
        builder = newton.ModelBuilder(gravity=0.0)
        for i in range(0, 2):
            builder.add_world(individual_builder)
        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(model, iterations=10, ls_iterations=10)

        dt = 0.02

        state_in.joint_q.assign(joint_start_positions)
        state_in.joint_qd.assign(joint_start_velocities)

        for _i in range(0, 20):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        joint_q = state_in.joint_q.numpy()

        # Test that the limits are observed in world 0.
        q0 = joint_q[0]
        q1 = joint_q[1]
        measured_tendon_length = coeff0 * q0 + coeff1 * q1
        has_legal_length = measured_tendon_length > lower_limit and measured_tendon_length < upper_limit
        self.assertTrue(
            has_legal_length,
            f"Allowed range is {lower_limit} to {upper_limit}. measured length is {measured_tendon_length}",
        )

        # World 1 and world 0 should have identical state.
        q2 = joint_q[2]
        self.assertAlmostEqual(
            q2,
            q0,
            places=3,
            msg=f"Expected joint_q[2]: {q0}, Measured q2: {q2}",
        )
        q3 = joint_q[3]
        self.assertAlmostEqual(
            q3,
            q1,
            places=3,
            msg=f"Expected joint_q[3]: {q1}, Measured q3: {q3}",
        )

    def test_upper_tendon_limit_breach_from_above(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.UPPER_LIMIT_FROM_ABOVE)

    def test_upper_tendon_limit_breach_from_below(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.UPPER_LIMIT_FROM_BELOW)

    def test_lower_tendon_limit_breach_from_below(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.LOWER_LIMIT_FROM_BELOW)

    def test_lower_tendon_limit_breach_from_above(self):
        self.run_test_mujoco_fixed_tendon_limit_behavior(self.LimitBreachType.LOWER_LIMIT_FROM_ABOVE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
