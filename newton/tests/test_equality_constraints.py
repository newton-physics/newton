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

    def _create_robot_body_based(self):
        """Create a robot with body-based equality constraints using programmatic API."""
        robot = newton.ModelBuilder()

        base = robot.add_link(xform=wp.transform((0, 0, 0)), mass=1.0, key="base")
        robot.add_shape_box(base, hx=0.5, hy=0.5, hz=0.5)

        link1 = robot.add_link(xform=wp.transform((1, 0, 0)), mass=1.0, key="link1")
        robot.add_shape_box(link1, hx=0.5, hy=0.5, hz=0.5)

        link2 = robot.add_link(xform=wp.transform((2, 0, 0)), mass=1.0, key="link2")
        robot.add_shape_box(link2, hx=0.5, hy=0.5, hz=0.5)

        joint1 = robot.add_joint_fixed(
            parent=-1,
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

        robot.add_articulation([joint1, joint2, joint3], key="articulation")

        robot.add_equality_constraint_connect(
            body1=base, body2=link2, anchor=wp.vec3(0.5, 0, 0), key="connect_constraint"
        )
        robot.add_equality_constraint_weld(body1=link1, body2=link2, anchor=wp.vec3(0.5, 0, 0), key="weld_constraint")
        return robot

    def _create_robot_site_based(self):
        """Create a robot with site-based equality constraints from MJCF."""
        mjcf = """<?xml version="1.0"?>
<mujoco>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.5 0.5 0.5"/>
            <site name="base_site" pos="0.5 0 0"/>
            <body name="link1" pos="1 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.5 0.5 0.5"/>
                <site name="link1_site" pos="0.5 0 0"/>
                <body name="link2" pos="1 0 0">
                    <joint name="joint2" type="hinge" axis="0 0 1"/>
                    <geom type="box" size="0.5 0.5 0.5"/>
                    <site name="link2_site" pos="0.5 0 0"/>
                </body>
            </body>
        </body>
    </worldbody>
    <equality>
        <connect name="site_connect" site1="base_site" site2="link2_site"/>
        <weld name="site_weld" site1="link1_site" site2="link2_site"/>
    </equality>
</mujoco>"""
        robot = newton.ModelBuilder()
        robot.add_mjcf(mjcf, parse_sites=True)
        return robot

    def test_equality_constraints_not_duplicated_per_world(self):
        """Test that equality constraints are not duplicated for each world when using separate_worlds=True.

        Tests both body-based (programmatic API) and site-based (MJCF) constraint definitions.
        """
        scenarios = [
            ("body_based", self._create_robot_body_based),
            ("site_based", self._create_robot_site_based),
        ]

        for scenario_name, create_robot_fn in scenarios:
            with self.subTest(constraint_type=scenario_name):
                robot = create_robot_fn()

                main_builder = newton.ModelBuilder()
                main_builder.add_ground_plane()

                num_worlds = 3
                for i in range(num_worlds):
                    main_builder.add_world(robot, xform=wp.transform((i * 5, 0, 0)))

                model = main_builder.finalize()

                # Should be 2 constraints per world * 3 worlds = 6 total
                self.assertEqual(model.equality_constraint_count, 2 * num_worlds)

                solver = newton.solvers.SolverMuJoCo(
                    model,
                    use_mujoco_cpu=True,
                    separate_worlds=True,
                    njmax=100,
                    nconmax=50,
                )

                # With separate_worlds=True, should only have constraints from one world (2)
                self.assertEqual(
                    solver.mj_model.neq,
                    2,
                    f"[{scenario_name}] Expected 2 equality constraints in MuJoCo model, got {solver.mj_model.neq}",
                )

                # Verify body indices are correctly remapped for each world
                eq_body1 = model.equality_constraint_body1.numpy()
                eq_body2 = model.equality_constraint_body2.numpy()

                for world_idx in range(num_worlds):
                    constraint_idx = world_idx * 2
                    expected_base = world_idx * 3 + 0
                    expected_link1 = world_idx * 3 + 1
                    expected_link2 = world_idx * 3 + 2

                    # Connect constraint: base -> link2
                    self.assertEqual(
                        eq_body1[constraint_idx],
                        expected_base,
                        f"[{scenario_name}] World {world_idx} connect body1 incorrect",
                    )
                    self.assertEqual(
                        eq_body2[constraint_idx],
                        expected_link2,
                        f"[{scenario_name}] World {world_idx} connect body2 incorrect",
                    )

                    # Weld constraint: link1 -> link2
                    self.assertEqual(
                        eq_body1[constraint_idx + 1],
                        expected_link1,
                        f"[{scenario_name}] World {world_idx} weld body1 incorrect",
                    )
                    self.assertEqual(
                        eq_body2[constraint_idx + 1],
                        expected_link2,
                        f"[{scenario_name}] World {world_idx} weld body2 incorrect",
                    )

    def _create_chain_body_based(self):
        """Create a chain with fixed joints and body-based constraints using programmatic API."""
        builder = newton.ModelBuilder()

        base = builder.add_link(xform=wp.transform((0, 0, 0)), mass=1.0, key="base")
        builder.add_shape_box(base, hx=0.5, hy=0.5, hz=0.5)

        link1 = builder.add_link(xform=wp.transform((1, 0, 0)), mass=1.0, key="link1")
        builder.add_shape_box(link1, hx=0.3, hy=0.3, hz=0.3)

        link2 = builder.add_link(xform=wp.transform((2, 0, 0)), mass=1.0, key="link2")
        builder.add_shape_box(link2, hx=0.3, hy=0.3, hz=0.3)

        link3 = builder.add_link(xform=wp.transform((3, 0, 0)), mass=1.0, key="link3")
        builder.add_shape_box(link3, hx=0.3, hy=0.3, hz=0.3)

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

        builder.add_equality_constraint_connect(
            body1=base, body2=link3, anchor=wp.vec3(0.5, 0, 0), key="connect_base_link3"
        )
        builder.add_equality_constraint_joint(
            joint1=joint1, joint2=joint3, polycoef=[1.0, -1.0, 0, 0, 0], key="couple_j1_j3"
        )
        builder.add_equality_constraint_weld(
            body1=link2,
            body2=link3,
            anchor=original_anchor,
            relpose=original_relpose,
            key="weld_link2_link3",
        )

        return builder, {
            "base": base,
            "link1": link1,
            "link2": link2,
            "link3": link3,
            "joint1": joint1,
            "joint3": joint3,
            "fixed_parent_xform": fixed_parent_xform,
            "fixed_child_xform": fixed_child_xform,
            "original_anchor": original_anchor,
            "original_relpose": original_relpose,
        }

    def _create_chain_site_based(self):
        """Create a chain with fixed joints and site-based constraints from MJCF."""
        mjcf = """<?xml version="1.0"?>
<mujoco>
    <worldbody>
        <body name="base" pos="0 0 0">
            <geom type="box" size="0.5 0.5 0.5"/>
            <site name="base_site" pos="0.5 0 0"/>
            <body name="link1" pos="1 0 0">
                <joint name="joint1" type="hinge" axis="0 0 1"/>
                <geom type="box" size="0.3 0.3 0.3"/>
                <site name="link1_site" pos="0.3 0.1 0"/>
                <body name="link2" pos="1 0 0">
                    <!-- No joint = fixed joint, will be collapsed -->
                    <geom type="box" size="0.3 0.3 0.3"/>
                    <site name="link2_site" pos="0.2 0 0"/>
                    <body name="link3" pos="1 0 0">
                        <joint name="joint3" type="hinge" axis="0 0 1"/>
                        <geom type="box" size="0.3 0.3 0.3"/>
                        <site name="link3_site" pos="0.3 0 0"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <equality>
        <connect name="site_connect" site1="base_site" site2="link3_site"/>
        <weld name="site_weld" site1="link2_site" site2="link3_site" relpose="0.1 0 0 1 0 0 0"/>
    </equality>
</mujoco>"""
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf, parse_sites=True)
        return builder, None  # No detailed info needed for site-based

    def test_collapse_fixed_joints_with_equality_constraints(self):
        """Test that equality constraints are properly remapped after collapse_fixed_joints.

        Tests both body-based (programmatic API) and site-based (MJCF) constraint definitions.
        """
        scenarios = [
            ("body_based", self._create_chain_body_based, 4, 4, 3),  # initial: 4 bodies, 4 joints, 3 constraints
            ("site_based", self._create_chain_site_based, 4, 3, 2),  # initial: 4 bodies, 3 joints, 2 constraints
        ]

        for scenario_name, create_chain_fn, init_bodies, init_joints, init_constraints in scenarios:
            with self.subTest(constraint_type=scenario_name):
                builder, info = create_chain_fn()

                # Verify initial state
                self.assertEqual(builder.body_count, init_bodies, f"[{scenario_name}] Initial body count wrong")
                self.assertEqual(builder.joint_count, init_joints, f"[{scenario_name}] Initial joint count wrong")
                self.assertEqual(
                    len(builder.equality_constraint_type),
                    init_constraints,
                    f"[{scenario_name}] Initial constraint count wrong",
                )

                # Collapse fixed joints
                result = builder.collapse_fixed_joints(verbose=False)

                # After collapse: one body merged, one joint removed
                self.assertEqual(
                    builder.body_count, init_bodies - 1, f"[{scenario_name}] Body count after collapse wrong"
                )
                self.assertEqual(
                    len(builder.equality_constraint_type),
                    init_constraints,
                    f"[{scenario_name}] Constraint count should remain unchanged",
                )

                # Verify all constraint body indices are valid
                for i, (body1, body2) in enumerate(
                    zip(builder.equality_constraint_body1, builder.equality_constraint_body2, strict=False)
                ):
                    self.assertGreaterEqual(body1, -1, f"[{scenario_name}] Constraint {i} body1 invalid")
                    self.assertLess(body1, builder.body_count, f"[{scenario_name}] Constraint {i} body1 out of range")
                    self.assertGreaterEqual(body2, -1, f"[{scenario_name}] Constraint {i} body2 invalid")
                    self.assertLess(body2, builder.body_count, f"[{scenario_name}] Constraint {i} body2 out of range")

                # Finalize and verify model is valid
                model = builder.finalize()
                self.assertEqual(model.body_count, init_bodies - 1, f"[{scenario_name}] Finalized body count wrong")
                self.assertEqual(
                    model.equality_constraint_count,
                    init_constraints,
                    f"[{scenario_name}] Finalized constraint count wrong",
                )

                # Create solver to verify constraints work
                solver = newton.solvers.SolverMuJoCo(
                    model,
                    use_mujoco_cpu=True,
                    iterations=100,
                    ls_iterations=50,
                    njmax=50,
                    nconmax=50,
                )
                self.assertEqual(
                    solver.mj_model.neq,
                    init_constraints,
                    f"[{scenario_name}] MuJoCo model should have {init_constraints} constraints",
                )

                # Additional detailed checks for body-based scenario
                if info is not None:
                    body_remap = result["body_remap"]
                    joint_remap = result["joint_remap"]

                    # Verify link2 was merged into link1
                    self.assertIn(info["link2"], result["body_merged_parent"])
                    self.assertEqual(result["body_merged_parent"][info["link2"]], info["link1"])

                    # Check index remapping
                    new_joint1 = joint_remap.get(info["joint1"], -1)
                    new_joint3 = joint_remap.get(info["joint3"], -1)
                    self.assertNotEqual(new_joint1, -1)
                    self.assertNotEqual(new_joint3, -1)

                    # Verify relpose was transformed correctly for weld constraint
                    merge_xform = info["fixed_parent_xform"] * wp.transform_inverse(info["fixed_child_xform"])
                    expected_relpose = merge_xform * info["original_relpose"]

                    actual_relpose = builder.equality_constraint_relpose[2]  # weld is constraint index 2
                    expected_p = wp.transform_get_translation(expected_relpose)
                    expected_q = wp.transform_get_rotation(expected_relpose)
                    actual_p = wp.transform_get_translation(actual_relpose)
                    actual_q = wp.transform_get_rotation(actual_relpose)

                    np.testing.assert_allclose(
                        [actual_p[0], actual_p[1], actual_p[2]],
                        [expected_p[0], expected_p[1], expected_p[2]],
                        rtol=1e-5,
                        err_msg=f"[{scenario_name}] Relpose translation not correctly transformed",
                    )
                    np.testing.assert_allclose(
                        [actual_q[0], actual_q[1], actual_q[2], actual_q[3]],
                        [expected_q[0], expected_q[1], expected_q[2], expected_q[3]],
                        rtol=1e-5,
                        err_msg=f"[{scenario_name}] Relpose rotation not correctly transformed",
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
