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
import newton.examples
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import assert_np_equal


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
        self.assertEqual(selection.get_root_transforms(model).shape, (1, 1))
        self.assertEqual(selection.get_dof_positions(model).shape, (1, 1, 0))
        self.assertEqual(selection.get_dof_velocities(model).shape, (1, 1, 0))
        self.assertEqual(selection.get_dof_forces(control).shape, (1, 1, 0))

    def test_selection_shapes(self):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )

        L = 9  # num links
        J = 9  # num joints
        D = 14  # num joint dofs
        C = 15  # num joint coords
        S = 13  # num shapes

        # scene with just one ant
        single_ant_model = ant.finalize()

        single_ant_view = ArticulationView(single_ant_model, "ant")
        self.assertEqual(single_ant_view.count, 1)
        self.assertEqual(single_ant_view.world_count, 1)
        self.assertEqual(single_ant_view.count_per_world, 1)
        self.assertEqual(single_ant_view.get_root_transforms(single_ant_model).shape, (1, 1))
        self.assertEqual(single_ant_view.get_root_velocities(single_ant_model).shape, (1, 1))
        self.assertEqual(single_ant_view.get_link_transforms(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_link_velocities(single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_dof_positions(single_ant_model).shape, (1, 1, C))
        self.assertEqual(single_ant_view.get_dof_velocities(single_ant_model).shape, (1, 1, D))
        self.assertEqual(single_ant_view.get_attribute("body_mass", single_ant_model).shape, (1, 1, L))
        self.assertEqual(single_ant_view.get_attribute("joint_type", single_ant_model).shape, (1, 1, J))
        self.assertEqual(single_ant_view.get_attribute("joint_dof_dim", single_ant_model).shape, (1, 1, J, 2))
        self.assertEqual(single_ant_view.get_attribute("joint_limit_ke", single_ant_model).shape, (1, 1, D))
        self.assertEqual(single_ant_view.get_attribute("shape_thickness", single_ant_model).shape, (1, 1, S))

        W = 10  # num worlds

        # scene with one ant per world
        single_ant_per_world_scene = newton.ModelBuilder()
        single_ant_per_world_scene.replicate(ant, num_worlds=W)
        single_ant_per_world_model = single_ant_per_world_scene.finalize()

        single_ant_per_world_view = ArticulationView(single_ant_per_world_model, "ant")
        self.assertEqual(single_ant_per_world_view.count, W)
        self.assertEqual(single_ant_per_world_view.world_count, W)
        self.assertEqual(single_ant_per_world_view.count_per_world, 1)
        self.assertEqual(single_ant_per_world_view.get_root_transforms(single_ant_per_world_model).shape, (W, 1))
        self.assertEqual(single_ant_per_world_view.get_root_velocities(single_ant_per_world_model).shape, (W, 1))
        self.assertEqual(single_ant_per_world_view.get_link_transforms(single_ant_per_world_model).shape, (W, 1, L))
        self.assertEqual(single_ant_per_world_view.get_link_velocities(single_ant_per_world_model).shape, (W, 1, L))
        self.assertEqual(single_ant_per_world_view.get_dof_positions(single_ant_per_world_model).shape, (W, 1, C))
        self.assertEqual(single_ant_per_world_view.get_dof_velocities(single_ant_per_world_model).shape, (W, 1, D))
        self.assertEqual(
            single_ant_per_world_view.get_attribute("body_mass", single_ant_per_world_model).shape, (W, 1, L)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_type", single_ant_per_world_model).shape, (W, 1, J)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_dof_dim", single_ant_per_world_model).shape, (W, 1, J, 2)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("joint_limit_ke", single_ant_per_world_model).shape, (W, 1, D)
        )
        self.assertEqual(
            single_ant_per_world_view.get_attribute("shape_thickness", single_ant_per_world_model).shape, (W, 1, S)
        )

        A = 3  # num articulations per world

        # scene with multiple ants per world
        multi_ant_world = newton.ModelBuilder()
        for i in range(A):
            multi_ant_world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        multi_ant_per_world_scene = newton.ModelBuilder()
        multi_ant_per_world_scene.replicate(multi_ant_world, num_worlds=W)
        multi_ant_per_world_model = multi_ant_per_world_scene.finalize()

        multi_ant_per_world_view = ArticulationView(multi_ant_per_world_model, "ant")
        self.assertEqual(multi_ant_per_world_view.count, W * A)
        self.assertEqual(multi_ant_per_world_view.world_count, W)
        self.assertEqual(multi_ant_per_world_view.count_per_world, A)
        self.assertEqual(multi_ant_per_world_view.get_root_transforms(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_view.get_root_velocities(multi_ant_per_world_model).shape, (W, A))
        self.assertEqual(multi_ant_per_world_view.get_link_transforms(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(multi_ant_per_world_view.get_link_velocities(multi_ant_per_world_model).shape, (W, A, L))
        self.assertEqual(multi_ant_per_world_view.get_dof_positions(multi_ant_per_world_model).shape, (W, A, C))
        self.assertEqual(multi_ant_per_world_view.get_dof_velocities(multi_ant_per_world_model).shape, (W, A, D))
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("body_mass", multi_ant_per_world_model).shape, (W, A, L)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_type", multi_ant_per_world_model).shape, (W, A, J)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_dof_dim", multi_ant_per_world_model).shape, (W, A, J, 2)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("joint_limit_ke", multi_ant_per_world_model).shape, (W, A, D)
        )
        self.assertEqual(
            multi_ant_per_world_view.get_attribute("shape_thickness", multi_ant_per_world_model).shape, (W, A, S)
        )

    def test_selection_mask(self):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )

        num_worlds = 4
        num_per_world = 3
        num_artis = num_worlds * num_per_world

        # scene with multiple ants per world
        world = newton.ModelBuilder()
        for i in range(num_per_world):
            world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        scene = newton.ModelBuilder()
        scene.replicate(world, num_worlds=num_worlds)
        model = scene.finalize()

        view = ArticulationView(model, "ant")

        # test default mask
        model_mask = view.get_model_articulation_mask()
        expected = np.full(num_artis, 1, dtype=np.bool)
        assert_np_equal(model_mask.numpy(), expected)

        # test per-world mask
        model_mask = view.get_model_articulation_mask(mask=[0, 1, 1, 0])
        expected = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.bool)
        assert_np_equal(model_mask.numpy(), expected)

        # test world-arti mask
        m = [
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ]
        model_mask = view.get_model_articulation_mask(mask=m)
        expected = np.array([0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.bool)
        assert_np_equal(model_mask.numpy(), expected)

    def run_test_joint_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        """Test an ArticulationView that includes a subset of joints and that we
        can write attributes to the subset of joints with and without a mask. Test
        that we can write to model/state/control."""
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
        num_joints = num_joints_per_articulation * num_articulations_per_world * num_worlds

        # Create a single articulation with 3 joints.
        single_articuation_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(single_articuation_builder)
        single_articuation_builder.add_mjcf(mjcf)

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(0, num_articulations_per_world):
            single_world_builder.add_builder(single_articuation_builder)

        # Customise the articulation keys in single_world_builder
        single_world_builder.articulation_key[1] = "art1"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_key[0] = "art1"
        else:
            single_world_builder.articulation_key[0] = "art0"

        # Create 3 worlds with two articulations per world and 3 joints per articulation.
        builder = newton.ModelBuilder()
        for _i in range(0, num_worlds):
            builder.add_world(single_world_builder)

        # Create the model
        model = builder.finalize()
        state_0 = model.state()
        control = model.control()

        # Create a view of "art1/joint3"
        joints_to_include = ["joint3"]
        joint_view = ArticulationView(model, "art1", include_joints=joints_to_include)

        # Get the attributes associated with "joint3"
        joint_dof_positions = joint_view.get_dof_positions(model).numpy()
        joint_limit_lower = joint_view.get_attribute("joint_limit_lower", model).numpy()
        joint_target_pos = joint_view.get_attribute("joint_target_pos", model).numpy()

        # Modify the attributes associated with "joint3"
        val = 1.0
        for world_idx in range(joint_dof_positions.shape[0]):
            for arti_idx in range(joint_dof_positions.shape[1]):
                for joint_idx in range(joint_dof_positions.shape[2]):
                    joint_dof_positions[world_idx, arti_idx, joint_idx] = val
                    joint_limit_lower[world_idx, arti_idx, joint_idx] += val
                    joint_target_pos[world_idx, arti_idx, joint_idx] += 2.0 * val
                    val += 1.0

        mask = None
        if use_mask:
            if use_multiple_artics_per_view:
                mask = [[False, False], [False, True], [False, False]]
            else:
                mask = [[False], [True], [False]]

        expected_dof_positions = []
        expected_joint_limit_lower = []
        expected_joint_target_pos = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -46.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -50.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    8.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
            else:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    2.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -50.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    0.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    0.0,
                ]
        else:
            if use_multiple_artics_per_view:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    1.0,
                    0.0,  # world0/artic1
                    0.0,
                    2.0,
                    0.0,  # world1/artic0
                    0.0,
                    3.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    5.0,
                    0.0,  # world2/artic1
                    0.0,
                    6.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -49.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -47.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -46.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -45.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -44.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    2.0,
                    0.0,  # world0/artic1
                    0.0,
                    4.0,
                    0.0,  # world1/artic0
                    0.0,
                    6.0,
                    0.0,  # world1/artic1
                    0.0,
                    8.0,
                    0.0,  # world2/artic0
                    0.0,
                    10.0,
                    0.0,  # world2/artic1
                    0.0,
                    12.0,
                ]
            else:
                expected_dof_positions = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    1.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    2.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    3.0,
                ]
                expected_joint_limit_lower = [
                    -50.5,  # world0/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world0/artic1
                    -50.5,
                    -49.5,
                    -50.5,  # world1/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world1/artic1
                    -50.5,
                    -48.5,
                    -50.5,  # world2/artic0
                    -50.5,
                    -50.5,
                    -50.5,  # world2/artic1
                    -50.5,
                    -47.5,
                ]
                expected_joint_target_pos = [
                    0.0,  # world0/artic0
                    0.0,
                    0.0,
                    0.0,  # world0/artic1
                    0.0,
                    2.0,
                    0.0,  # world1/artic0
                    0.0,
                    0.0,
                    0.0,  # world1/artic1
                    0.0,
                    4.0,
                    0.0,  # world2/artic0
                    0.0,
                    0.0,
                    0.0,  # world2/artic1
                    0.0,
                    6.0,
                ]

        # Set the values associated with "joint3"
        joint_view.set_dof_positions(state_0, joint_dof_positions, mask)
        joint_view.set_dof_positions(model, joint_dof_positions, mask)
        joint_view.set_attribute("joint_limit_lower", model, joint_limit_lower, mask)
        joint_view.set_attribute("joint_target_pos", control, joint_target_pos, mask)
        joint_view.set_attribute("joint_target_pos", model, joint_target_pos, mask)

        # Get the updated values from model, state, control.
        measured_state_joint_dof_positions = state_0.joint_q.numpy()
        measured_model_joint_dof_positions = model.joint_q.numpy()
        measured_model_joint_limit_lower = model.joint_limit_lower.numpy()
        measured_control_joint_target_pos = control.joint_target_pos.numpy()
        measured_model_joint_target_pos = model.joint_target_pos.numpy()

        # Test that the modified values were correctly set in model, state and control
        for i in range(0, num_joints):
            measured = measured_state_joint_dof_positions[i]
            expected = expected_dof_positions[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected state joint dof position value: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_dof_positions[i]
            expected = expected_dof_positions[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint dof position value: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_limit_lower[i]
            expected = expected_joint_limit_lower[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint limit lower value: {expected}, Measured value: {measured}",
            )

            measured = measured_control_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint target pos value: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint target pos value: {expected}, Measured value: {measured}",
            )

    def run_test_link_selection(self, use_mask: bool, use_multiple_artics_per_view: bool):
        """Test an ArticulationView that excludes a subset of links and that we
        can write attributes to the subset of links with and without a mask"""
        mjcf = """<?xml version="1.0" ?>
<mujoco model="myart">
    <worldbody>
    <!-- Root body (fixed to world) -->
    <body name="root" pos="0 0 0">
      <!-- First child link with prismatic joint along x -->
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Second child link with prismatic joint along x -->
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>

      <!-- Third child link with prismatic joint along x -->
      <body name="link3" pos="-0.0 -0.9 0">
        <joint name="joint3" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
        num_links_per_articulation = 4
        num_articulations_per_world = 2
        num_worlds = 3
        num_links = num_links_per_articulation * num_articulations_per_world * num_worlds

        # Create a single articulation
        single_articuation_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(single_articuation_builder)
        single_articuation_builder.add_mjcf(mjcf)

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(0, num_articulations_per_world):
            single_world_builder.add_builder(single_articuation_builder)

        # Customise the articulation keys in single_world_builder
        single_world_builder.articulation_key[0] = "art0"
        if use_multiple_artics_per_view:
            single_world_builder.articulation_key[1] = "art0"
        else:
            single_world_builder.articulation_key[1] = "art1"

        # Create 3 worlds with 2 articulations per world and 4 links per articulation.
        builder = newton.ModelBuilder()
        for _i in range(0, num_worlds):
            builder.add_world(single_world_builder)

        # Create the model
        model = builder.finalize()
        state_0 = model.state()

        # create a view of art0/"link1" and art0/"link2" by excluding "root" and "link3"
        links_to_exclude = ["root", "link3"]
        link_view = ArticulationView(model, "art0", exclude_links=links_to_exclude)

        # Get the attributes associated with "art0/link1" and "art0/link2"
        link_masses = link_view.get_attribute("body_mass", model).numpy()
        link_vels = link_view.get_attribute("body_qd", model).numpy()

        # Modify the attributes associated with "art0/link1" and "art0/link2"
        val = 1.0
        for world_idx in range(link_masses.shape[0]):
            for arti_idx in range(link_masses.shape[1]):
                for link_idx in range(link_masses.shape[2]):
                    link_masses[world_idx, arti_idx, link_idx] += val
                    link_vels[world_idx, arti_idx, link_idx] = [val, val, val, val, val, val]
                    val += 1.0

        mask = None
        if use_mask:
            if use_multiple_artics_per_view:
                mask = [[False, False], [False, True], [False, False]]
            else:
                mask = [[False], [True], [False]]

        link_view.set_attribute("body_mass", model, link_masses, mask)
        link_view.set_attribute("body_qd", model, link_vels, mask)
        link_view.set_attribute("body_qd", state_0, link_vels, mask)

        expected_body_masses = []
        expected_body_vels = []
        if use_mask:
            if use_multiple_artics_per_view:
                expected_body_masses = [
                    0.01,  # world0/artic0
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world0/artic1
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world1/artic0
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world1/artic1
                    7.01,
                    8.01,
                    0.01,
                    0.01,  # world2/artic0
                    0.01,
                    00.01,
                    0.01,
                    0.01,  # world2/artic1
                    0.01,
                    0.01,
                    0.01,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # world1/artic1/link1
                    [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
            else:
                expected_body_masses = [
                    0.01,  # world0/artic0
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world0/artic1
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world1/artic0
                    3.01,
                    4.01,
                    0.01,
                    0.01,  # world1/artic1
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world2/artic0
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world2/artic1
                    0.01,
                    0.01,
                    0.01,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world1/artic0/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
        else:
            if use_multiple_artics_per_view:
                expected_body_masses = [
                    0.01,  # world0/artic0
                    1.01,
                    2.01,
                    0.01,
                    0.01,  # world0/artic1
                    3.01,
                    4.01,
                    0.01,
                    0.01,  # world1/artic0
                    5.01,
                    6.01,
                    0.01,
                    0.01,  # world1/artic1
                    7.01,
                    8.01,
                    0.01,
                    0.01,  # world2/artic0
                    9.01,
                    10.01,
                    0.01,
                    0.01,  # world2/artic1
                    11.01,
                    12.01,
                    0.01,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # world0/artic0/link1
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world0/artic1/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # world1/artic0/link1
                    [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],  # world1/artic1/link1
                    [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],  # world2/artic0/link1
                    [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [11.0, 11.0, 11.0, 11.0, 11.0, 11.0],  # world2/artic1/link1
                    [12.0, 12.0, 12.0, 12.0, 12.0, 12.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]
            else:
                expected_body_masses = [
                    0.01,  # world0/artic0
                    1.01,
                    2.01,
                    0.01,
                    0.01,  # world0/artic1
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world1/artic0
                    3.01,
                    4.01,
                    0.01,
                    0.01,  # world1/artic1
                    0.01,
                    0.01,
                    0.01,
                    0.01,  # world2/artic0
                    5.01,
                    6.01,
                    0.01,
                    0.01,  # world2/artic1
                    0.01,
                    0.01,
                    0.01,
                ]
                expected_body_vels = [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/root
                    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # world0/artic0/link1
                    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # world0/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world0/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/root
                    [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],  # world1/artic0/link1
                    [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],  # world1/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world1/artic1/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/root
                    [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],  # world2/artic0/link1
                    [6.0, 6.0, 6.0, 6.0, 6.0, 6.0],  # world2/artic0/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic0/link3
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/root
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link2
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # world2/artic1/link3
                ]

        # Get the updated body masses
        measured_body_masses = model.body_mass.numpy()
        measured_model_body_vels = model.body_qd.numpy()
        measured_state_body_vels = state_0.body_qd.numpy()

        # Test that the modified values were correctly set in model
        for i in range(0, num_links):
            measured = measured_body_masses[i]
            expected = expected_body_masses[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected body mass value: {expected}, Measured value: {measured}",
            )

            for j in range(0, 6):
                measured = measured_model_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value: {expected}, Measured value: {measured}",
                )

            for j in range(0, 6):
                measured = measured_state_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value: {expected}, Measured value: {measured}",
                )

    def test_joint_selection_one_per_view_no_mask(self):
        self.run_test_joint_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_joint_selection_two_per_view_no_mask(self):
        self.run_test_joint_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_joint_selection_one_per_view_with_mask(self):
        self.run_test_joint_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_joint_selection_two_per_view_with_mask(self):
        self.run_test_joint_selection(use_mask=True, use_multiple_artics_per_view=True)

    def test_link_selection_one_per_view_no_mask(self):
        self.run_test_link_selection(use_mask=False, use_multiple_artics_per_view=False)

    def test_link_selection_two_per_view_no_mask(self):
        self.run_test_link_selection(use_mask=False, use_multiple_artics_per_view=True)

    def test_link_selection_one_per_view_with_mask(self):
        self.run_test_link_selection(use_mask=True, use_multiple_artics_per_view=False)

    def test_link_selection_two_per_view_with_mask(self):
        self.run_test_link_selection(use_mask=True, use_multiple_artics_per_view=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
