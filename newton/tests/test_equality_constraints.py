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
import warp as wp

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

    def test_equality_constraints_not_duplicated_per_world(self):
        """Test that equality constraints are not duplicated for each world when using separate_worlds=True"""
        # Create a simple robot builder with equality constraints
        robot = newton.ModelBuilder()

        # Add bodies with shapes
        base = robot.add_body(xform=wp.transform((0, 0, 0)), mass=1.0, key="base")
        robot.add_shape_box(base, hx=0.5, hy=0.5, hz=0.5)

        link1 = robot.add_body(xform=wp.transform((1, 0, 0)), mass=1.0, key="link1")
        robot.add_shape_box(link1, hx=0.5, hy=0.5, hz=0.5)

        link2 = robot.add_body(xform=wp.transform((2, 0, 0)), mass=1.0, key="link2")
        robot.add_shape_box(link2, hx=0.5, hy=0.5, hz=0.5)

        # Add joints - connect base to world (-1) first
        robot.add_joint_fixed(
            parent=-1,  # world
            child=base,
            parent_xform=wp.transform((0, 0, 0)),
            child_xform=wp.transform((0, 0, 0)),
            key="joint_fixed",
        )
        robot.add_joint_revolute(
            parent=base,
            child=link1,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="joint1",
        )
        robot.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="joint2",
        )

        # Add 2 equality constraints
        robot.add_equality_constraint_connect(
            body1=base, body2=link2, anchor=wp.vec3(0.5, 0, 0), key="connect_constraint"
        )
        robot.add_equality_constraint_joint(
            joint1=1,  # joint1 (base to link1)
            joint2=2,  # joint2 (link1 to link2)
            polycoef=[1.0, -1.0, 0, 0, 0],
            key="joint_constraint",
        )

        # Build main model with multiple worlds
        main_builder = newton.ModelBuilder()

        # Add ground plane (global, world -1)
        main_builder.current_world = -1
        main_builder.add_ground_plane()

        # Add multiple robot instances
        num_worlds = 3
        for i in range(num_worlds):
            main_builder.add_builder(robot, world=i, xform=wp.transform((i * 5, 0, 0)))

        # Finalize the model
        model = main_builder.finalize()

        # Check that equality constraints count is correct in the Newton model
        # Should be 2 constraints per world * 3 worlds = 6 total
        self.assertEqual(model.equality_constraint_count, 2 * num_worlds)

        # Create MuJoCo solver with separate_worlds=True
        solver = newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=True,
            separate_worlds=True,
            njmax=100,  # Should be enough for 2 constraints, not 6
            nconmax=50,
        )

        # Check that the MuJoCo model has the correct number of equality constraints
        # With separate_worlds=True, it should only have constraints from one world (2)
        self.assertEqual(
            solver.mj_model.neq, 2, f"Expected 2 equality constraints in MuJoCo model, got {solver.mj_model.neq}"
        )

        print(f"Test passed: MuJoCo model has {solver.mj_model.neq} equality constraints (expected 2)")
        print(f"Newton model has {model.equality_constraint_count} total constraints across {num_worlds} worlds")
        
        # Verify that indices are correctly remapped for each world
        # Each world adds 3 bodies, so body indices should be offset by 3 * world_index
        # The first world's base body should be at index 0, second at 3, third at 6
        eq_body1 = model.equality_constraint_body1.numpy()
        eq_body2 = model.equality_constraint_body2.numpy()
        eq_joint1 = model.equality_constraint_joint1.numpy()
        eq_joint2 = model.equality_constraint_joint2.numpy()
        
        for world_idx in range(num_worlds):
            # Each world has 2 constraints
            constraint_idx = world_idx * 2
            
            # For connect constraint: body1 should be base (offset by 3 * world_idx)
            # body2 should be link2 (offset by 3 * world_idx + 2)
            expected_body1 = world_idx * 3 + 0  # base body
            expected_body2 = world_idx * 3 + 2  # link2 body
            self.assertEqual(eq_body1[constraint_idx], expected_body1, 
                           f"World {world_idx} connect constraint body1 index incorrect")
            self.assertEqual(eq_body2[constraint_idx], expected_body2,
                           f"World {world_idx} connect constraint body2 index incorrect")
            
            # For joint constraint: joint1 and joint2 should be offset by 3 * world_idx
            # (each robot has 3 joints: fixed, revolute1, revolute2)
            expected_joint1 = world_idx * 3 + 1  # joint1 (base to link1)
            expected_joint2 = world_idx * 3 + 2  # joint2 (link1 to link2)
            self.assertEqual(eq_joint1[constraint_idx + 1], expected_joint1,
                           f"World {world_idx} joint constraint joint1 index incorrect")
            self.assertEqual(eq_joint2[constraint_idx + 1], expected_joint2,
                           f"World {world_idx} joint constraint joint2 index incorrect")


if __name__ == "__main__":
    unittest.main(verbosity=2)
