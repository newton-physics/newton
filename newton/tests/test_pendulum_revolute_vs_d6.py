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
import unittest

import numpy as np

import newton
from newton.tests.unittest_utils import USD_AVAILABLE


class TestPendulumRevoluteVsD6(unittest.TestCase):
    @unittest.skipUnless(USD_AVAILABLE, "USD not available")
    def test_pendulum_revolute_vs_d6_mujoco(self):
        # Load the USD file
        usd_path = os.path.join(os.path.dirname(__file__), "assets", "pendulum_revolute_vs_d6.usda")

        # Parse and build model
        builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
        builder.add_usd(usd_path, only_load_enabled_rigid_bodies=False)
        model = builder.finalize()

        # Check joint types
        jt = model.joint_type.numpy()

        # Find the revolute and D6 joints by their types
        rev_indices = np.where(jt == newton.JointType.REVOLUTE)[0]
        d6_indices = np.where(jt == newton.JointType.D6)[0]

        if len(rev_indices) == 0 or len(d6_indices) == 0:
            self.fail(f"Expected REVOLUTE and D6 joints not found. types={jt}")

        idx_rev = int(rev_indices[0])
        idx_d6 = int(d6_indices[0])

        # Initial state: give both pendulums the same small initial angle
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        # Set initial joint positions
        q0_model = model.joint_q.numpy()
        qd0_model = model.joint_qd.numpy()

        # Initialize all DOFs to zero first
        q0_model[:] = 0.0
        qd0_model[:] = 0.0

        # Set initial angle - both pendulums start at same angle
        initial_angle = 0.2  # rad

        # Find the q indices for revolute and D6 joints
        q_start_np = model.joint_q_start.numpy()
        rev_qi = q_start_np[idx_rev]
        d6_qi = q_start_np[idx_d6]

        # Set initial positions
        q0_model[rev_qi] = initial_angle
        q0_model[d6_qi] = initial_angle
        model.joint_q.assign(q0_model)

        # Copy the joint positions to state before FK
        state_0.joint_q.assign(model.joint_q)
        state_0.joint_qd.assign(model.joint_qd)

        # Evaluate FK for initial state
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        # Simulate with SolverMuJoCo (Warp backend)
        try:
            solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, disable_contacts=True, iterations=2)
        except Exception as e:
            self.skipTest(f"SolverMuJoCo unavailable: {e}")

        sim_dt = 1.0 / 240.0
        steps = 480
        traj = np.zeros((steps, 2))

        for i in range(steps):
            state_0.clear_forces()
            contacts = None
            solver.step(state_0, state_1, control, contacts, dt=sim_dt)
            state_0, state_1 = state_1, state_0

            # Get joint angles directly
            q_cur = state_0.joint_q.numpy()
            traj[i, 0] = q_cur[rev_qi]
            traj[i, 1] = q_cur[d6_qi]

        # Also print a small preview
        for i in range(min(10, steps)):
            t = i * sim_dt
            qrev = traj[i, 0]
            qd6 = traj[i, 1]
            print(f"t:{t:.6f} qrev:{qrev:.6f} qd6:{qd6:.6f} diff:{abs(qrev - qd6):.6f}")

        # Basic checks: they should have moved and oscillated
        # Check that min and max are different (pendulum is moving)
        rev_min, rev_max = np.min(traj[:, 0]), np.max(traj[:, 0])
        d6_min, d6_max = np.min(traj[:, 1]), np.max(traj[:, 1])

        self.assertNotAlmostEqual(
            rev_min, rev_max, places=3, msg=f"Revolute pendulum did not move: min={rev_min}, max={rev_max}"
        )
        self.assertNotAlmostEqual(d6_min, d6_max, places=3, msg=f"D6 pendulum did not move: min={d6_min}, max={d6_max}")

        # Their trajectories should be close (same physics)
        diff = np.mean(np.abs(traj[:, 0] - traj[:, 1]))
        self.assertLess(diff, 0.1, f"Pendulum behaviors differ too much, mean abs diff = {diff}")

        # Do not clean up to allow re-use across runs
        # os.unlink(usd_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
