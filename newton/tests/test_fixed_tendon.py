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

import newton
from newton.solvers import SolverMuJoCo


class TestMujocoFixedTendon(unittest.TestCase):
    def test_single_mujoco_fixed_tendon_limit_behaviour(self):
        """Test that tendons work"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>

  <option timestep="0.002" gravity="0 0 -9.81"/>

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

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf)
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

        # Length of tendon at start is: pos**coef0 + pos1*coef1 = 2*0.5 + 0*0.0 = 1.0
        # Target length is 0.0 (see mjcf above)
        joint_start_positions = [0.5, 0.0]
        state_in.joint_q.assign(joint_start_positions)

        for _ in range(0, 200):
            solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
            state_in, state_out = state_out, state_in

        joint_q = state_in.joint_q.numpy()
        q0 = joint_q[0]
        q1 = joint_q[1]
        measured_tendon_length = coeff0 * q0 + coeff1 * q1
        self.assertAlmostEqual(
            expected_tendon_length,
            measured_tendon_length,
            places=3,
            msg=f"Expected stiffness value: {expected_tendon_length}, Measured value: {measured_tendon_length}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
