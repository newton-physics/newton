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

import os
import tempfile
import unittest

import warp as wp

import newton
import newton.examples
from newton.utils.selection import ArticulationView


class TestTendonControl(unittest.TestCase):
    def test_tendon_control_integration(self):
        """Test full integration of tendon control from MJCF to simulation"""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test">
    <worldbody>
        <geom type="capsule" pos="-.2 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
        <site name="site0" pos="-.2 .0 .1"/>
        <body>
            <geom type="capsule" pos="0.21 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
            <joint type="hinge" axis="0 1 0" name="hinge"/>
            <site name="site1" pos=".2 .0 .1"/>
        </body>
    </worldbody>

    <tendon>
        <spatial name="spatial0">
            <site site="site0"/>
            <site site="site1"/>
        </spatial>
    </tendon>

    <actuator>
        <position name="spatial0_act" tendon="spatial0" kp="300" />
    </actuator>
</mujoco>
"""

        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "test_tendon.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            # Parse with Newton
            builder = newton.ModelBuilder()
            newton.utils.parse_mjcf(
                mjcf_path,
                builder,
                collapse_fixed_joints=True,
                up_axis="Z",
                enable_self_collisions=False,
            )
            model = builder.finalize()

            # Verify model structure
            self.assertEqual(model.site_count, 2)
            self.assertEqual(model.tendon_count, 1)
            self.assertEqual(model.tendon_actuator_count, 1)
            self.assertEqual(model.joint_count, 1)

            # Create states and control
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()

            # Verify control has tendon arrays
            self.assertIsNotNone(control.tendon_target)
            self.assertEqual(len(control.tendon_target), 1)

            # Set tendon target
            # Need to use numpy to modify warp array
            tendon_targets = control.tendon_target.numpy()
            tendon_targets[0] = -0.05  # Contract tendon by 5cm
            control.tendon_target = wp.array(tendon_targets, dtype=wp.float32, device=model.device)

            # Create solver - let it handle MuJoCo model creation internally
            solver = newton.solvers.MuJoCoSolver(model)

            # Record initial joint position
            initial_joint_pos = state_0.joint_q.numpy()[0]

            # Simulate
            dt = 0.001
            for _ in range(100):
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

            # Verify joint moved due to tendon actuation
            final_joint_pos = state_0.joint_q.numpy()[0]
            self.assertNotAlmostEqual(initial_joint_pos, final_joint_pos, places=3)
            # Joint should have rotated (positive direction due to tendon contraction pulling site1 towards site0)
            self.assertGreater(final_joint_pos, initial_joint_pos)

    def test_control_initialization(self):
        """Test that tendon control arrays are properly initialized"""
        builder = newton.ModelBuilder()

        # Add a simple model with tendons
        builder.add_body(xform=wp.transform())
        site1 = builder.add_site(0, wp.transform(wp.vec3(0, 0, 0)), key="site1")
        site2 = builder.add_site(0, wp.transform(wp.vec3(1, 0, 0)), key="site2")

        tendon_id = builder.add_tendon(
            tendon_type="spatial", site_ids=[site1, site2], stiffness=100.0, key="test_tendon"
        )

        builder.add_tendon_actuator(tendon_id=tendon_id, ke=50.0, key="test_actuator")

        model = builder.finalize()

        # Check control initialization
        control = model.control()
        self.assertIsNotNone(control.tendon_target)
        self.assertEqual(len(control.tendon_target), 1)

        # Check initial values are zero
        self.assertEqual(control.tendon_target.numpy()[0], 0.0)

    def test_tendon_control_selection(self):
        """Test that tendon can be controlled via Selection API"""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <worldbody>
        <geom type="capsule" pos="-.2 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
        <site name="site0" pos="-.2 .0 .1"/>
        <body>
            <geom type="capsule" pos="0.21 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
            <joint type="hinge" axis="0 1 0" name="hinge0"/>
            <site name="site1" pos=".2 .0 .1"/>
            <body>
                <geom type="capsule" pos="0.62 0 0" size="0.1 0.1" axisangle="0 1 0 90"/>
                <joint type="hinge" pos="0.5 0 0" axis="0 1 0" name="hinge1"/>
                <site name="site2" pos=".6 .0 .1"/>
            </body>
        </body>
    </worldbody>

     <tendon>
        <spatial name="spatial0">
            <site site="site0"/>
            <site site="site1"/>
        </spatial>
        <spatial name="spatial1">
            <site site="site1"/>
            <site site="site2"/>
        </spatial>
    </tendon>

    <actuator>
        <position name="spatial0_act" tendon="spatial0" kp="300" />
         <position name="spatial1_act" tendon="spatial1" kp="300" />
    </actuator>
</mujoco>
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mjcf_path = os.path.join(tmpdir, "test_tendon.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)

            # Parse with Newton
            builder = newton.ModelBuilder()
            newton.utils.parse_mjcf(
                mjcf_path,
                builder,
                collapse_fixed_joints=True,
                up_axis="Z",
                enable_self_collisions=False,
            )
            model = builder.finalize()

            # Verify model structure
            self.assertEqual(model.site_count, 3)
            self.assertEqual(model.tendon_count, 2)
            self.assertEqual(model.tendon_actuator_count, 2)
            self.assertEqual(model.joint_count, 2)

            # Create states and control
            state_0 = model.state()
            state_1 = model.state()
            control = model.control()

            # Verify control has tendon arrays
            self.assertIsNotNone(control.tendon_target)
            self.assertEqual(len(control.tendon_target), 2)

            # Set tendon target via selection
            tendons = ArticulationView(model, "articulation_1")
            self.assertEqual(tendons.get_attribute("tendon_target", control).numpy()[0][0], 0)
            self.assertEqual(tendons.get_attribute("tendon_target", control).numpy()[0][1], 0)
            tendons.set_attribute("tendon_target", control, [[-1.0, -1.0]])
            self.assertEqual(tendons.get_attribute("tendon_target", control).numpy()[0][0], -1.0)
            self.assertEqual(tendons.get_attribute("tendon_target", control).numpy()[0][1], -1.0)

            tendons = ArticulationView(model, "articulation_1", exclude_tendons=[0])
            tendons.set_attribute("tendon_target", control, [[-1.0]])

            # Create solver - let it handle MuJoCo model creation internally
            solver = newton.solvers.MuJoCoSolver(model)

            # Record initial joint position
            initial_joint0_pos = state_0.joint_q.numpy()[0]
            initial_joint1_pos = state_0.joint_q.numpy()[1]

            # Simulate
            dt = 0.001
            for _ in range(100):
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

            # Verify joint moved due to tendon actuation
            final_joint0_pos = state_0.joint_q.numpy()[0]
            final_joint1_pos = state_0.joint_q.numpy()[1]
            self.assertNotAlmostEqual(initial_joint0_pos, final_joint0_pos, places=3)
            self.assertNotAlmostEqual(initial_joint1_pos, final_joint1_pos, places=3)
            # Joint should have rotated (positive direction due to gravity)
            self.assertGreater(final_joint0_pos, initial_joint0_pos)
            # Joint should have rotated (negative direction due to tendon contraction pulling site2 towards site1)
            self.assertLess(final_joint1_pos, initial_joint1_pos)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
