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

    def test_tendon_convenience_methods(self):
        """Test that tendon convenience methods work correctly."""
        builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(self.TENDON_MJCF)
        model = builder.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        # Test getters for implemented properties
        stiffness = view.get_fixed_tendon_stiffness(model)
        self.assertEqual(stiffness.shape, (1, 1, T))
        assert_np_equal(stiffness.numpy(), np.array([[[2.0]]]))

        damping = view.get_fixed_tendon_damping(model)
        self.assertEqual(damping.shape, (1, 1, T))
        assert_np_equal(damping.numpy(), np.array([[[1.0]]]))

        rest_length = view.get_fixed_tendon_rest_length(model)
        self.assertEqual(rest_length.shape, (1, 1, T))

        pos_limits = view.get_fixed_tendon_pos_limits(model)
        self.assertEqual(pos_limits.shape, (1, 1, T))

        # Test setters for implemented properties
        view.set_fixed_tendon_damping(model, np.array([[[2.5]]]))
        damping = view.get_fixed_tendon_damping(model)
        assert_np_equal(damping.numpy(), np.array([[[2.5]]]))

        view.set_fixed_tendon_rest_length(model, np.array([[[[0.1, 0.2]]]]))
        view.set_fixed_tendon_pos_limits(model, np.array([[[[-1.0, 1.0]]]]))

        # Test that unimplemented methods raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            view.get_fixed_tendon_limit_stiffness(model)
        with self.assertRaises(NotImplementedError):
            view.set_fixed_tendon_limit_stiffness(model, np.array([[[1.0]]]))
        with self.assertRaises(NotImplementedError):
            view.get_fixed_tendon_offset(model)
        with self.assertRaises(NotImplementedError):
            view.set_fixed_tendon_offset(model, np.array([[[0.0]]]))

    def test_tendon_multi_world(self):
        """Test that tendon selection works with multiple worlds."""
        individual_builder = newton.ModelBuilder(gravity=0.0)
        SolverMuJoCo.register_custom_attributes(individual_builder)
        individual_builder.add_mjcf(self.TENDON_MJCF)

        W = 4  # num worlds
        scene = newton.ModelBuilder(gravity=0.0)
        scene.replicate(individual_builder, num_worlds=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")
        T = 1

        self.assertEqual(view.world_count, W)
        self.assertEqual(view.count_per_world, 1)
        self.assertEqual(view.tendon_count, T)

        stiffness = view.get_fixed_tendon_stiffness(model)
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
        scene.replicate(individual_builder, num_worlds=W)
        model = scene.finalize()

        view = ArticulationView(model, "two_prismatic_links")

        # Set new stiffness values
        new_stiffness = np.array([[[5.0]], [[10.0]]])
        view.set_fixed_tendon_stiffness(model, new_stiffness)

        # Verify values were set
        stiffness = view.get_fixed_tendon_stiffness(model)
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

        # Attempting to access tendon attributes should raise an error
        with self.assertRaises(AttributeError):
            view.get_fixed_tendon_stiffness(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
