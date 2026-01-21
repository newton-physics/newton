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

        eq_keys = self.model.equality_constraint_key
        eq_body1 = self.model.equality_constraint_body1.numpy()
        eq_body2 = self.model.equality_constraint_body2.numpy()
        eq_anchors = self.model.equality_constraint_anchor.numpy()
        eq_torquescale = self.model.equality_constraint_torquescale.numpy()

        c_site_idx = eq_keys.index("c_site")
        self.assertEqual(eq_body1[c_site_idx], -1)
        self.assertEqual(eq_body2[c_site_idx], 0)
        np.testing.assert_allclose(eq_anchors[c_site_idx], [0.0, 0.0, 1.0], rtol=1e-5)

        w_site_idx = eq_keys.index("w_site")
        self.assertEqual(eq_body1[w_site_idx], -1)
        self.assertEqual(eq_body2[w_site_idx], 1)
        np.testing.assert_allclose(eq_anchors[w_site_idx], [0.0, 0.0, 0.0], rtol=1e-5)
        self.assertAlmostEqual(eq_torquescale[w_site_idx], 0.1, places=5)

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

        self.assertEqual(self.solver.mj_model.eq_type.shape[0], 5)

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
        base = robot.add_link(xform=wp.transform((0, 0, 0)), mass=1.0, key="base")
        robot.add_shape_box(base, hx=0.5, hy=0.5, hz=0.5)

        link1 = robot.add_link(xform=wp.transform((1, 0, 0)), mass=1.0, key="link1")
        robot.add_shape_box(link1, hx=0.5, hy=0.5, hz=0.5)

        link2 = robot.add_link(xform=wp.transform((2, 0, 0)), mass=1.0, key="link2")
        robot.add_shape_box(link2, hx=0.5, hy=0.5, hz=0.5)

        # Add joints - connect base to world (-1) first
        joint1 = robot.add_joint_fixed(
            parent=-1,  # world
            child=base,
            parent_xform=wp.transform((0, 0, 0)),
            child_xform=wp.transform((0, 0, 0)),
            key="joint_fixed",
        )
        joint2 = robot.add_joint_revolute(
            parent=base,
            child=link1,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="joint1",
        )
        joint3 = robot.add_joint_revolute(
            parent=link1,
            child=link2,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="joint2",
        )

        # Add articulation
        robot.add_articulation([joint1, joint2, joint3], key="articulation")

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
        main_builder.add_ground_plane()

        # Add multiple robot instances
        num_worlds = 3
        for i in range(num_worlds):
            main_builder.add_world(robot, xform=wp.transform((i * 5, 0, 0)))

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
            self.assertEqual(
                eq_body1[constraint_idx], expected_body1, f"World {world_idx} connect constraint body1 index incorrect"
            )
            self.assertEqual(
                eq_body2[constraint_idx], expected_body2, f"World {world_idx} connect constraint body2 index incorrect"
            )

            # For joint constraint: joint1 and joint2 should be offset by 3 * world_idx
            # (each robot has 3 joints: fixed, revolute1, revolute2)
            expected_joint1 = world_idx * 3 + 1  # joint1 (base to link1)
            expected_joint2 = world_idx * 3 + 2  # joint2 (link1 to link2)
            self.assertEqual(
                eq_joint1[constraint_idx + 1],
                expected_joint1,
                f"World {world_idx} joint constraint joint1 index incorrect",
            )
            self.assertEqual(
                eq_joint2[constraint_idx + 1],
                expected_joint2,
                f"World {world_idx} joint constraint joint2 index incorrect",
            )

    def test_collapse_fixed_joints_with_equality_constraints(self):
        """Test that equality constraints are properly remapped after collapse_fixed_joints,
        including correct transformation of anchor points and relpose."""
        builder = newton.ModelBuilder()

        # Create chain: world -> base (fixed) -> link1 (revolute) -> link2 (fixed) -> link3
        base = builder.add_link(xform=wp.transform((0, 0, 0)), mass=1.0, key="base")
        builder.add_shape_box(base, hx=0.5, hy=0.5, hz=0.5)

        link1 = builder.add_link(xform=wp.transform((1, 0, 0)), mass=1.0, key="link1")
        builder.add_shape_box(link1, hx=0.3, hy=0.3, hz=0.3)

        link2 = builder.add_link(xform=wp.transform((2, 0, 0)), mass=1.0, key="link2")
        builder.add_shape_box(link2, hx=0.3, hy=0.3, hz=0.3)

        link3 = builder.add_link(xform=wp.transform((3, 0, 0)), mass=1.0, key="link3")
        builder.add_shape_box(link3, hx=0.3, hy=0.3, hz=0.3)

        # Fixed joint between link1 and link2 - defines the merge transform
        fixed_parent_xform = wp.transform((0.5, 0.1, 0.0), wp.quat_identity())
        fixed_child_xform = wp.transform((-0.3, 0.0, 0.0), wp.quat_identity())

        joint_fixed_base = builder.add_joint_fixed(
            parent=-1,
            child=base,
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            key="j_base",
        )
        joint1 = builder.add_joint_revolute(
            parent=base,
            child=link1,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="j1",
        )
        joint_fixed_link2 = builder.add_joint_fixed(
            parent=link1,
            child=link2,
            parent_xform=fixed_parent_xform,
            child_xform=fixed_child_xform,
            key="j2_fixed",
        )
        joint3 = builder.add_joint_revolute(
            parent=link2,
            child=link3,
            parent_xform=wp.transform((0.5, 0, 0)),
            child_xform=wp.transform((-0.5, 0, 0)),
            axis=(0, 0, 1),
            key="j3",
        )

        builder.add_articulation([joint_fixed_base, joint1, joint_fixed_link2, joint3], key="articulation")

        original_anchor = wp.vec3(0.1, 0.2, 0.3)
        original_relpose = wp.transform((0.5, 0.1, -0.2), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.3))

        eq_connect = builder.add_equality_constraint_connect(
            body1=base, body2=link3, anchor=wp.vec3(0.5, 0, 0), key="connect_base_link3"
        )
        eq_joint = builder.add_equality_constraint_joint(
            joint1=joint1, joint2=joint3, polycoef=[1.0, -1.0, 0, 0, 0], key="couple_j1_j3"
        )
        eq_weld = builder.add_equality_constraint_weld(
            body1=link2,
            body2=link3,
            anchor=original_anchor,
            relpose=original_relpose,
            key="weld_link2_link3",
        )

        # Compute expected merge transform: parent_xform * inverse(child_xform)
        merge_xform = fixed_parent_xform * wp.transform_inverse(fixed_child_xform)
        expected_anchor = original_anchor
        expected_relpose = merge_xform * original_relpose

        # Verify initial state
        self.assertEqual(builder.body_count, 4)
        self.assertEqual(builder.joint_count, 4)
        self.assertEqual(len(builder.equality_constraint_type), 3)

        # Collapse fixed joints
        result = builder.collapse_fixed_joints(verbose=False)
        body_remap = result["body_remap"]
        joint_remap = result["joint_remap"]

        self.assertEqual(builder.body_count, 3)
        self.assertEqual(builder.joint_count, 3)

        # Verify link2 was merged into link1
        self.assertIn(link2, result["body_merged_parent"])
        self.assertEqual(result["body_merged_parent"][link2], link1)

        # Check index remapping
        new_base = body_remap.get(base, base)
        new_link1 = body_remap.get(link1, link1)
        new_link3 = body_remap.get(link3, link3)
        new_joint1 = joint_remap.get(joint1, -1)
        new_joint3 = joint_remap.get(joint3, -1)

        self.assertNotEqual(new_joint1, -1)
        self.assertNotEqual(new_joint3, -1)
        self.assertEqual(builder.equality_constraint_joint1[eq_joint], new_joint1)
        self.assertEqual(builder.equality_constraint_joint2[eq_joint], new_joint3)
        self.assertEqual(builder.equality_constraint_body1[eq_connect], new_base)
        self.assertEqual(builder.equality_constraint_body2[eq_connect], new_link3)
        self.assertEqual(builder.equality_constraint_body1[eq_weld], new_link1)
        self.assertEqual(builder.equality_constraint_body2[eq_weld], new_link3)

        # Verify anchor was transformed correctly
        actual_anchor = builder.equality_constraint_anchor[eq_weld]
        np.testing.assert_allclose(
            [actual_anchor[0], actual_anchor[1], actual_anchor[2]],
            [expected_anchor[0], expected_anchor[1], expected_anchor[2]],
            rtol=1e-5,
            err_msg="Anchor not correctly transformed after body merge",
        )

        # Verify relpose was transformed correctly
        actual_relpose = builder.equality_constraint_relpose[eq_weld]
        expected_p = wp.transform_get_translation(expected_relpose)
        expected_q = wp.transform_get_rotation(expected_relpose)
        actual_p = wp.transform_get_translation(actual_relpose)
        actual_q = wp.transform_get_rotation(actual_relpose)

        np.testing.assert_allclose(
            [actual_p[0], actual_p[1], actual_p[2]],
            [expected_p[0], expected_p[1], expected_p[2]],
            rtol=1e-5,
            err_msg="Relpose translation not correctly transformed after body merge",
        )
        np.testing.assert_allclose(
            [actual_q[0], actual_q[1], actual_q[2], actual_q[3]],
            [expected_q[0], expected_q[1], expected_q[2], expected_q[3]],
            rtol=1e-5,
            err_msg="Relpose rotation not correctly transformed after body merge",
        )

        # Finalize and verify
        model = builder.finalize()
        self.assertEqual(model.body_count, 3)
        self.assertEqual(model.joint_count, 3)
        self.assertEqual(model.equality_constraint_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
