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
        single_ant_per_world_scene.replicate(ant, world_count=W)
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
        multi_ant_per_world_scene.replicate(multi_ant_world, world_count=W)
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

    def test_selection_shape_values_noncontiguous(self):
        """Test that shape attribute values are correct when shape selection is non-contiguous."""
        # Build a 3-link chain: base -> link1 -> link2
        # Each link has one shape with a distinct thickness value
        robot = newton.ModelBuilder()

        thicknesses = [0.001, 0.002, 0.003]

        base = robot.add_link(xform=wp.transform([0, 0, 0], wp.quat_identity()), mass=1.0, key="base")
        robot.add_shape_box(
            base,
            hx=0.1,
            hy=0.1,
            hz=0.1,
            cfg=newton.ModelBuilder.ShapeConfig(thickness=thicknesses[0]),
            key="shape_base",
        )

        link1 = robot.add_link(xform=wp.transform([0, 0, 0.5], wp.quat_identity()), mass=0.5, key="link1")
        robot.add_shape_capsule(
            link1,
            radius=0.05,
            half_height=0.2,
            cfg=newton.ModelBuilder.ShapeConfig(thickness=thicknesses[1]),
            key="shape_link1",
        )

        link2 = robot.add_link(xform=wp.transform([0, 0, 1.0], wp.quat_identity()), mass=0.3, key="link2")
        robot.add_shape_sphere(
            link2,
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(thickness=thicknesses[2]),
            key="shape_link2",
        )

        j0 = robot.add_joint_free(child=base)
        j1 = robot.add_joint_revolute(parent=base, child=link1, axis=[0, 1, 0])
        j2 = robot.add_joint_revolute(parent=link1, child=link2, axis=[0, 1, 0])
        robot.add_articulation([j0, j1, j2], key="robot")

        W = 3
        scene = newton.ModelBuilder()
        # add a ground plane first so shape indices are offset
        scene.add_shape_plane()
        scene.replicate(robot, world_count=W)
        model = scene.finalize()

        # exclude the middle link to make shape indices non-contiguous: [0, 2]
        view = ArticulationView(model, "robot", exclude_links=["link1"])
        self.assertFalse(view.shapes_contiguous, "Expected non-contiguous shape selection")
        self.assertEqual(view.shape_count, 2)

        # read shape_thickness through ArticulationView and check values
        vals = view.get_attribute("shape_thickness", model)
        self.assertEqual(vals.shape, (W, 1, 2))
        vals_np = vals.numpy()

        expected = [thicknesses[0], thicknesses[2]]  # base and link2 (link1 excluded)
        for w in range(W):
            for s, expected_thickness in enumerate(expected):
                self.assertAlmostEqual(
                    float(vals_np[w, 0, s]),
                    expected_thickness,
                    places=6,
                    msg=f"world={w}, shape={s}",
                )

    def test_selection_mask(self):
        # load articulation
        ant = newton.ModelBuilder()
        ant.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )

        world_count = 4
        num_per_world = 3
        num_artis = world_count * num_per_world

        # scene with multiple ants per world
        world = newton.ModelBuilder()
        for i in range(num_per_world):
            world.add_builder(ant, xform=wp.transform((0.0, 0.0, 1.0 + i), wp.quat_identity()))
        scene = newton.ModelBuilder()
        scene.replicate(world, world_count=world_count)
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
        joint_dof_positions = joint_view.get_dof_positions(model).numpy().copy()
        joint_limit_lower = joint_view.get_attribute("joint_limit_lower", model).numpy().copy()
        joint_target_pos = joint_view.get_attribute("joint_target_pos", model).numpy().copy()

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
                mask = wp.array([[False, False], [False, True], [False, False]], dtype=bool, device=model.device)
            else:
                mask = wp.array([[False], [True], [False]], dtype=bool, device=model.device)

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
        wp_joint_dof_positions = wp.array(joint_dof_positions, dtype=float, device=model.device)
        wp_joint_limit_lowers = wp.array(joint_limit_lower, dtype=float, device=model.device)
        wp_joint_target_pos = wp.array(joint_target_pos, dtype=float, device=model.device)
        joint_view.set_dof_positions(state_0, wp_joint_dof_positions, mask)
        joint_view.set_dof_positions(model, wp_joint_dof_positions, mask)
        joint_view.set_attribute("joint_limit_lower", model, wp_joint_limit_lowers, mask)
        joint_view.set_attribute("joint_target_pos", control, wp_joint_target_pos, mask)
        joint_view.set_attribute("joint_target_pos", model, wp_joint_target_pos, mask)

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
                msg=f"Expected state joint dof position value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_dof_positions[i]
            expected = expected_dof_positions[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint dof position value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_limit_lower[i]
            expected = expected_joint_limit_lower[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint limit lower value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_control_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected control joint target pos value {i}: {expected}, Measured value: {measured}",
            )

            measured = measured_model_joint_target_pos[i]
            expected = expected_joint_target_pos[i]
            self.assertAlmostEqual(
                expected,
                measured,
                places=4,
                msg=f"Expected model joint target pos value {i}: {expected}, Measured value: {measured}",
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
        single_articulation_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(single_articulation_builder)
        single_articulation_builder.add_mjcf(mjcf)

        # Create a world with 2 articulations
        single_world_builder = newton.ModelBuilder()
        for _i in range(0, num_articulations_per_world):
            single_world_builder.add_builder(single_articulation_builder)

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
        link_masses = link_view.get_attribute("body_mass", model).numpy().copy()
        link_vels = link_view.get_attribute("body_qd", model).numpy().copy()

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
                mask = wp.array([[False, False], [False, True], [False, False]], dtype=bool, device=model.device)
            else:
                mask = wp.array([[False], [True], [False]], dtype=bool, device=model.device)

        wp_link_masses = wp.array(link_masses, dtype=float, device=model.device)
        wp_link_vels = wp.array(link_vels, dtype=float, device=model.device)
        link_view.set_attribute("body_mass", model, wp_link_masses, mask)
        link_view.set_attribute("body_qd", model, wp_link_vels, mask)
        link_view.set_attribute("body_qd", state_0, wp_link_vels, mask)

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
                msg=f"Expected body mass value {i}: {expected}, Measured value: {measured}",
            )

            for j in range(0, 6):
                measured = measured_model_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value {i}: {expected}, Measured value: {measured}",
                )

            for j in range(0, 6):
                measured = measured_state_body_vels[i][j]
                expected = expected_body_vels[i][j]
                self.assertAlmostEqual(
                    expected,
                    measured,
                    places=4,
                    msg=f"Expected body velocity value {i}: {expected}, Measured value: {measured}",
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


class TestSelectionFixedTendons(unittest.TestCase):
    """Tests for fixed tendon support in ArticulationView."""

    TENDON_MJCF = """<?xml version="1.0" ?>
<mujoco model="two_prismatic_links">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="root" pos="0 0 0">
      <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
      <body name="link1" pos="0.0 -0.5 0">
        <joint name="joint1" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="1 0 0 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
      <body name="link2" pos="-0.0 -0.7 0">
        <joint name="joint2" type="slide" axis="1 0 0" range="-50.5 50.5"/>
        <geom type="cylinder" size="0.05 0.025" rgba="0 0 1 1" euler="0 90 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <fixed name="coupling_tendon" stiffness="2.0" damping="1.0" springlength="0.0">
      <joint joint="joint1" coef="1"/>
      <joint joint="joint2" coef="1"/>
    </fixed>
  </tendon>
</mujoco>
"""

    def test_tendon_count(self):
        """Test that tendon count is correctly detected."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        self.assertEqual(view.tendon_count, 1)

    def test_tendon_selection_shapes(self):
        """Test that tendon selection API returns correct shapes."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1  # num tendons

        # Test generic attribute access
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (1, 1, T))

        damping = view.get_attribute("mujoco.tendon_damping", model)
        self.assertEqual(damping.shape, (1, 1, T))

        tendon_range = view.get_attribute("mujoco.tendon_range", model)
        self.assertEqual(tendon_range.shape, (1, 1, T))  # vec2 trailing dim

    def test_tendon_generic_api(self):
        """Test that tendon attributes are accessible via generic get/set_attribute."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        # Test getters via generic API
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (1, 1, T))
        assert_np_equal(stiffness.numpy(), np.array([[[2.0]]]))

        damping = view.get_attribute("mujoco.tendon_damping", model)
        self.assertEqual(damping.shape, (1, 1, T))
        assert_np_equal(damping.numpy(), np.array([[[1.0]]]))

        springlength = view.get_attribute("mujoco.tendon_springlength", model)
        self.assertEqual(springlength.shape, (1, 1, T))

        tendon_range = view.get_attribute("mujoco.tendon_range", model)
        self.assertEqual(tendon_range.shape, (1, 1, T))

        # Test setters via generic API
        view.set_attribute("mujoco.tendon_damping", model, np.array([[[2.5]]]))
        damping = view.get_attribute("mujoco.tendon_damping", model)
        assert_np_equal(damping.numpy(), np.array([[[2.5]]]))

    def test_tendon_multi_world(self):
        """Test that tendon selection works with multiple worlds."""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(self.TENDON_MJCF)

        W = 4  # num worlds
        scene = newton.ModelBuilder(gravity=0.0)
        scene.replicate(individual_builder, world_count=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        self.assertEqual(view.world_count, W)
        self.assertEqual(view.count_per_world, 1)
        self.assertEqual(view.tendon_count, T)

        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (W, 1, T))

        # Verify values are correct across all worlds
        expected = np.full((W, 1, T), 2.0)
        assert_np_equal(stiffness.numpy(), expected)

    def test_tendon_set_values(self):
        """Test that setting tendon values works correctly."""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(self.TENDON_MJCF)

        W = 2  # num worlds
        scene = newton.ModelBuilder(gravity=0.0)
        scene.replicate(individual_builder, world_count=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")

        # Set new stiffness values via generic API
        new_stiffness = np.array([[[5.0]], [[10.0]]])
        view.set_attribute("mujoco.tendon_stiffness", model, new_stiffness)

        # Verify values were set
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        assert_np_equal(stiffness.numpy(), new_stiffness)

    def test_tendon_names(self):
        """Test that tendon names are correctly populated."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")

        # Check tendon_names is populated
        self.assertEqual(len(view.tendon_names), 1)
        self.assertEqual(view.tendon_names[0], "coupling_tendon")

        # Check that we can look up index from name
        idx = view.tendon_names.index("coupling_tendon")
        self.assertEqual(idx, 0)

    def test_no_tendons_in_articulation(self):
        """Test that articulations without tendons have tendon_count=0."""
        # Use nv_ant.xml which has no tendons
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
        )
        model = builder.finalize()

        view = ArticulationView(model, "ant")
        self.assertEqual(view.tendon_count, 0)
        self.assertEqual(len(view.tendon_names), 0)

    def test_no_tendons_but_model_has_tendons(self):
        """Test accessing tendon attributes on articulation without tendons when model has tendons elsewhere."""
        # Create a model with one articulation that has tendons and one without
        with_tendons_mjcf = self.TENDON_MJCF

        no_tendons_mjcf = """<?xml version="1.0" ?>
<mujoco model="no_tendons_robot">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 0"/>

  <worldbody>
    <body name="simple_robot" pos="0 0 0">
      <joint name="simple_joint" type="slide" axis="1 0 0"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(with_tendons_mjcf)
        builder.add_mjcf(no_tendons_mjcf)
        model = builder.finalize()

        # Select the articulation without tendons
        view = ArticulationView(model, "no_tendons_robot")
        self.assertEqual(view.tendon_count, 0)

        # Attempting to access tendon attributes should raise an error
        # This tests line 969: no tendons found in the selected articulations
        with self.assertRaises(AttributeError) as ctx:
            view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertIn("no tendons were found", str(ctx.exception))

    def test_multiple_articulations_per_world(self):
        """Test tendon selection with multiple articulations in a single world."""
        # Build a single articulation with tendons
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(self.TENDON_MJCF)

        # Create a world with multiple copies of the articulation
        A = 2  # articulations per world
        multi_robot_world = newton.ModelBuilder(gravity=0.0)
        for i in range(A):
            multi_robot_world.add_builder(
                individual_builder, xform=wp.transform((i * 2.0, 0.0, 0.0), wp.quat_identity())
            )

        # Replicate to multiple worlds
        W = 2  # num worlds
        scene = newton.ModelBuilder(gravity=0.0)
        scene.replicate(multi_robot_world, world_count=W)
        model = scene.finalize()

        # Select all articulations
        view = ArticulationView(model, "two_prismatic_links")

        # Should have W worlds, A articulations per world, 1 tendon per articulation
        self.assertEqual(view.world_count, W)
        self.assertEqual(view.count_per_world, A)
        self.assertEqual(view.tendon_count, 1)

        # Test that we can read tendon attributes
        stiffness = view.get_attribute("mujoco.tendon_stiffness", model)
        self.assertEqual(stiffness.shape, (W, A, 1))

        # All stiffness values should be 2.0 (from TENDON_MJCF)
        expected = np.full((W, A, 1), 2.0)
        assert_np_equal(stiffness.numpy(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
