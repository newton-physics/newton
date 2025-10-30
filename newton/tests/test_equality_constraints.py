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


class TestEqualityConstraints(unittest.TestCase):
    def test_multiple_constraints(self):
        self.sim_time = 0.0
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / 10

        builder = newton.ModelBuilder()

        builder.add_mjcf(
            os.path.join(os.path.dirname(__file__), "assets", "constraints.xml"),
            ignore_names=["floor", "ground"],
            up_axis="Z",
            skip_equality_constraints=False,
        )

        self.model = builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_cpu=True,
            solver="newton",
            integrator="euler",
            iterations=100,
            ls_iterations=50,
            njmax=100,
            nconmax=50,
        )

        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        for _ in range(1000):
            for _ in range(10):
                self.state_0.clear_forces()
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.frame_dt

        self.assertGreater(
            self.solver.mj_model.eq_type.shape[0], 0
        )  # check if number of equality constraints in mjModel > 0

        # Check constraint violations
        nefc = self.solver.mj_data.nefc  # number of active constraints
        if nefc > 0:
            efc_pos = self.solver.mj_data.efc_pos[:nefc]  # constraint violations
            max_violation = np.max(np.abs(efc_pos))
            self.assertLess(max_violation, 0.01, f"Maximum constraint violation {max_violation} exceeds threshold")

        # Check constraint forces
        if nefc > 0:
            efc_force = self.solver.mj_data.efc_force[:nefc]
            max_force = np.max(np.abs(efc_force))
            self.assertLess(max_force, 1000.0, f"Maximum constraint force {max_force} seems unreasonably large")


if __name__ == "__main__":
    unittest.main(verbosity=2)
