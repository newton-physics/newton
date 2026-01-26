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

"""Tests for MuJoCo general actuator support."""

import unittest

import numpy as np

import newton
from newton import ActuatorMode
from newton._src.solvers.mujoco import CtrlSource
from newton.solvers import SolverMuJoCo

MJCF_ALL_ACTUATOR_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_all_actuators">
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="base" pos="0 0 1">
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            <body name="link_motor" pos="0.2 0 0">
                <joint name="joint_motor" axis="0 0 1" type="hinge"/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="link_position" pos="0.2 0 0">
                    <joint name="joint_position" axis="0 0 1" type="hinge"/>
                    <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                    <body name="link_velocity" pos="0.2 0 0">
                        <joint name="joint_velocity" axis="0 0 1" type="hinge"/>
                        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                        <body name="link_pos_vel" pos="0.2 0 0">
                            <joint name="joint_pos_vel" axis="0 0 1" type="hinge"/>
                            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                            <body name="link_general" pos="0.2 0 0">
                                <joint name="joint_general" axis="0 0 1" type="hinge"/>
                                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                                <body name="link_passive" pos="0.2 0 0">
                                    <joint name="joint_passive" axis="0 0 1" type="hinge"/>
                                    <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint_motor"/>
        <position name="pos1" joint="joint_position" kp="100"/>
        <velocity name="vel1" joint="joint_velocity" kv="10"/>
        <position name="pos2" joint="joint_pos_vel" kp="200"/>
        <velocity name="vel2" joint="joint_pos_vel" kv="20"/>
        <general name="gen1" joint="joint_general" gainprm="50 0 0" biasprm="0 -50 -5" gear="2.5" ctrlrange="-1 1" ctrllimited="true"/>
    </actuator>
</mujoco>
"""


def get_qd_start(builder, joint_name):
    """Get the qd index for a joint by name."""
    joint_idx = builder.joint_key.index(joint_name)
    return sum(builder.joint_dof_dim[i][0] + builder.joint_dof_dim[i][1] for i in range(joint_idx))


class TestMuJoCoActuators(unittest.TestCase):
    """Test MuJoCo actuator parsing, ctrl_direct flag, and multi-world support."""

    def test_actuator_parsing_and_mujoco_model_setup(self):
        """Test actuator parsing into Newton model and subsequent MuJoCo model setup."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(MJCF_ALL_ACTUATOR_TYPES, ctrl_direct=False)
        model = builder.finalize()
        model.ground = False

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 6)

        joint_act_mode = model.joint_act_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.EFFORT))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.POSITION))
        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_position")], 100.0)
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.VELOCITY))
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_velocity")], 10.0)
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.POSITION_VELOCITY))
        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 200.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 20.0)
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_general")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_passive")], int(ActuatorMode.NONE))

        mujoco_attrs = model.mujoco
        ctrl_source = mujoco_attrs.ctrl_source.numpy()

        self.assertEqual(ctrl_source[0], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[1], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[2], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[3], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[4], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[5], CtrlSource.CTRL_DIRECT)

        newton_gainprm = mujoco_attrs.actuator_gainprm.numpy()
        newton_biasprm = mujoco_attrs.actuator_biasprm.numpy()
        newton_gear = mujoco_attrs.actuator_gear.numpy()
        newton_ctrllimited = mujoco_attrs.actuator_ctrllimited.numpy()
        newton_ctrlrange = mujoco_attrs.actuator_ctrlrange.numpy()

        np.testing.assert_allclose(newton_gainprm[5, :3], [50.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(newton_biasprm[5, :3], [0.0, -50.0, -5.0], atol=1e-5)
        self.assertAlmostEqual(newton_gear[5, 0], 2.5, places=5)
        self.assertEqual(newton_ctrllimited[5], True)
        np.testing.assert_allclose(newton_ctrlrange[5], [-1.0, 1.0], atol=1e-5)

        control = model.control()
        self.assertEqual(control.mujoco.ctrl.shape[0], 6)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 5)
        self.assertEqual(mj_model.nq, 6)
        self.assertEqual(mj_model.nv, 6)

        mjc_ctrl_source = solver.mjc_actuator_ctrl_source.numpy()
        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()

        for mj_act_idx in range(mj_model.nu):
            ctrl_src = mjc_ctrl_source[mj_act_idx]
            newton_idx = mjc_to_newton[mj_act_idx]

            if ctrl_src == CtrlSource.CTRL_DIRECT:
                np.testing.assert_allclose(
                    mj_model.actuator_gainprm[mj_act_idx, :3],
                    newton_gainprm[newton_idx, :3],
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    mj_model.actuator_biasprm[mj_act_idx, :3],
                    newton_biasprm[newton_idx, :3],
                    atol=1e-5,
                )
                self.assertAlmostEqual(
                    mj_model.actuator_gear[mj_act_idx, 0],
                    newton_gear[newton_idx, 0],
                    places=5,
                )
                self.assertEqual(
                    mj_model.actuator_ctrllimited[mj_act_idx],
                    newton_ctrllimited[newton_idx],
                )
                np.testing.assert_allclose(
                    mj_model.actuator_ctrlrange[mj_act_idx],
                    newton_ctrlrange[newton_idx],
                    atol=1e-5,
                )

    def test_ctrl_direct_flag(self):
        """Test ctrl_direct=True/False affects Newton model and MuJoCo solver correctly."""
        builder_normal = newton.ModelBuilder()
        builder_normal.add_mjcf(MJCF_ALL_ACTUATOR_TYPES, ctrl_direct=False)
        model_normal = builder_normal.finalize()
        model_normal.ground = False

        joint_act_mode_normal = model_normal.joint_act_mode.numpy()
        self.assertEqual(joint_act_mode_normal[get_qd_start(builder_normal, "joint_motor")], int(ActuatorMode.EFFORT))
        self.assertEqual(
            joint_act_mode_normal[get_qd_start(builder_normal, "joint_position")], int(ActuatorMode.POSITION)
        )
        self.assertEqual(
            joint_act_mode_normal[get_qd_start(builder_normal, "joint_velocity")], int(ActuatorMode.VELOCITY)
        )
        self.assertEqual(
            joint_act_mode_normal[get_qd_start(builder_normal, "joint_pos_vel")], int(ActuatorMode.POSITION_VELOCITY)
        )
        self.assertEqual(joint_act_mode_normal[get_qd_start(builder_normal, "joint_general")], int(ActuatorMode.NONE))

        ctrl_source_normal = model_normal.mujoco.ctrl_source.numpy()
        self.assertEqual(ctrl_source_normal[0], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source_normal[1], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source_normal[2], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source_normal[3], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source_normal[4], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source_normal[5], CtrlSource.CTRL_DIRECT)

        solver_normal = SolverMuJoCo(model_normal, iterations=1, disable_contacts=True)
        mj_normal = solver_normal.mj_model
        self.assertEqual(mj_normal.nu, 5)
        self.assertEqual(mj_normal.nq, 6)
        self.assertEqual(mj_normal.nv, 6)

        builder_direct = newton.ModelBuilder()
        builder_direct.add_mjcf(MJCF_ALL_ACTUATOR_TYPES, ctrl_direct=True)
        model_direct = builder_direct.finalize()
        model_direct.ground = False

        joint_act_mode_direct = model_direct.joint_act_mode.numpy()
        self.assertEqual(joint_act_mode_direct[get_qd_start(builder_direct, "joint_motor")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode_direct[get_qd_start(builder_direct, "joint_position")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode_direct[get_qd_start(builder_direct, "joint_velocity")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode_direct[get_qd_start(builder_direct, "joint_pos_vel")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode_direct[get_qd_start(builder_direct, "joint_general")], int(ActuatorMode.NONE))

        ctrl_source_direct = model_direct.mujoco.ctrl_source.numpy()
        for i in range(6):
            self.assertEqual(ctrl_source_direct[i], CtrlSource.CTRL_DIRECT)

        solver_direct = SolverMuJoCo(model_direct, iterations=1, disable_contacts=True)
        mj_direct = solver_direct.mj_model
        self.assertEqual(mj_direct.nu, 6)
        self.assertEqual(mj_direct.nq, 6)
        self.assertEqual(mj_direct.nv, 6)

        newton_gainprm = model_direct.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model_direct.mujoco.actuator_biasprm.numpy()
        newton_gear = model_direct.mujoco.actuator_gear.numpy()
        newton_ctrllimited = model_direct.mujoco.actuator_ctrllimited.numpy()
        newton_ctrlrange = model_direct.mujoco.actuator_ctrlrange.numpy()

        mjc_to_newton = solver_direct.mjc_actuator_to_newton_idx.numpy()

        for mj_act_idx in range(mj_direct.nu):
            newton_idx = mjc_to_newton[mj_act_idx]
            np.testing.assert_allclose(
                mj_direct.actuator_gainprm[mj_act_idx, :3],
                newton_gainprm[newton_idx, :3],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                mj_direct.actuator_biasprm[mj_act_idx, :3],
                newton_biasprm[newton_idx, :3],
                atol=1e-5,
            )
            self.assertAlmostEqual(
                mj_direct.actuator_gear[mj_act_idx, 0],
                newton_gear[newton_idx, 0],
                places=5,
            )
            self.assertEqual(
                mj_direct.actuator_ctrllimited[mj_act_idx],
                newton_ctrllimited[newton_idx],
            )
            np.testing.assert_allclose(
                mj_direct.actuator_ctrlrange[mj_act_idx],
                newton_ctrlrange[newton_idx],
                atol=1e-5,
            )

    def test_multiworld(self):
        """Test actuator parsing and ctrl_direct in multi-world setup."""
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(MJCF_ALL_ACTUATOR_TYPES, ctrl_direct=False)

        main_builder = newton.ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()
        model.ground = False

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 12)

        actuator_world = model.mujoco.actuator_world.numpy()
        self.assertEqual(len(actuator_world), 12)
        for i in range(6):
            self.assertEqual(actuator_world[i], 0)
        for i in range(6, 12):
            self.assertEqual(actuator_world[i], 1)

        control = model.control()
        self.assertEqual(control.mujoco.ctrl.shape[0], 12)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_model = solver.mj_model
        self.assertEqual(mj_model.nu, 5)
        self.assertEqual(mj_model.nq, 6)
        self.assertEqual(mj_model.nv, 6)

        robot_builder_direct = newton.ModelBuilder()
        robot_builder_direct.add_mjcf(MJCF_ALL_ACTUATOR_TYPES, ctrl_direct=True)

        main_builder_direct = newton.ModelBuilder()
        main_builder_direct.add_world(robot_builder_direct)
        main_builder_direct.add_world(robot_builder_direct)
        model_direct = main_builder_direct.finalize()
        model_direct.ground = False

        ctrl_source_direct = model_direct.mujoco.ctrl_source.numpy()
        for i in range(12):
            self.assertEqual(ctrl_source_direct[i], CtrlSource.CTRL_DIRECT)

        joint_act_mode_direct = model_direct.joint_act_mode.numpy()
        for i in range(model_direct.joint_dof_count):
            self.assertEqual(joint_act_mode_direct[i], int(ActuatorMode.NONE))

        solver_direct = SolverMuJoCo(model_direct, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_direct = solver_direct.mj_model
        self.assertEqual(mj_direct.nu, 6)
        self.assertEqual(mj_direct.nq, 6)
        self.assertEqual(mj_direct.nv, 6)

        newton_gainprm = model_direct.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model_direct.mujoco.actuator_biasprm.numpy()
        newton_gear = model_direct.mujoco.actuator_gear.numpy()

        mjc_to_newton = solver_direct.mjc_actuator_to_newton_idx.numpy()

        for mj_act_idx in range(mj_direct.nu):
            newton_idx = mjc_to_newton[mj_act_idx]
            np.testing.assert_allclose(
                mj_direct.actuator_gainprm[mj_act_idx, :3],
                newton_gainprm[newton_idx, :3],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                mj_direct.actuator_biasprm[mj_act_idx, :3],
                newton_biasprm[newton_idx, :3],
                atol=1e-5,
            )
            self.assertAlmostEqual(
                mj_direct.actuator_gear[mj_act_idx, 0],
                newton_gear[newton_idx, 0],
                places=5,
            )


if __name__ == "__main__":
    unittest.main()
