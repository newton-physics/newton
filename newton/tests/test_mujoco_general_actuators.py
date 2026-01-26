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

"""Tests for MuJoCo actuator parsing and propagation."""

import unittest

import numpy as np

import newton
from newton import ActuatorMode
from newton._src.solvers.mujoco import CtrlSource
from newton.solvers import SolverMuJoCo, SolverNotifyFlags

MJCF_ACTUATORS = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_actuators">
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="floating" pos="0 0 1">
            <freejoint name="free"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            <body name="link_motor" pos="0.2 0 0">
                <joint name="joint_motor" axis="0 0 1" type="hinge"/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="link_pos_vel" pos="0.2 0 0">
                    <joint name="joint_pos_vel" axis="0 0 1" type="hinge"/>
                    <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                    <body name="link_position" pos="0.2 0 0">
                        <joint name="joint_position" axis="0 0 1" type="hinge"/>
                        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                        <body name="link_velocity" pos="0.2 0 0">
                            <joint name="joint_velocity" axis="0 0 1" type="hinge"/>
                            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                            <body name="link_general" pos="0.2 0 0">
                                <joint name="joint_general" axis="0 0 1" type="hinge"/>
                                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    <actuator>
        <motor name="motor1" joint="joint_motor"/>
        <position name="pos1" joint="joint_pos_vel" kp="100"/>
        <velocity name="vel1" joint="joint_pos_vel" kv="10"/>
        <position name="pos2" joint="joint_position" kp="200"/>
        <velocity name="vel2" joint="joint_velocity" kv="20"/>
        <general name="gen1" joint="joint_general" gainprm="50 0 0" biasprm="0 -50 -5" ctrlrange="-1 1" ctrllimited="true"/>
        <general name="body1" body="floating" gainprm="30 0 0" biasprm="0 0 0"/>
    </actuator>
</mujoco>
"""


def get_qd_start(builder, joint_name):
    joint_idx = builder.joint_key.index(joint_name)
    return sum(builder.joint_dof_dim[i][0] + builder.joint_dof_dim[i][1] for i in range(joint_idx))


class TestMuJoCoActuators(unittest.TestCase):
    """Test MuJoCo actuator parsing through builder, Newton model, and MuJoCo model."""

    def test_parsing_ctrl_direct_false(self):
        """Test parsing with ctrl_direct=False."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        self.assertEqual(len(builder.joint_act_mode), 11)
        for i in range(6):
            self.assertEqual(builder.joint_act_mode[i], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.EFFORT))
        self.assertEqual(
            builder.joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.POSITION_VELOCITY)
        )
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.POSITION))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.VELOCITY))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_general")], int(ActuatorMode.NONE))

        self.assertEqual(builder.joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 100.0)
        self.assertEqual(builder.joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 10.0)
        self.assertEqual(builder.joint_target_ke[get_qd_start(builder, "joint_position")], 200.0)
        self.assertEqual(builder.joint_target_kd[get_qd_start(builder, "joint_velocity")], 20.0)

        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 7)

        joint_act_mode = model.joint_act_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        for i in range(6):
            self.assertEqual(joint_act_mode[i], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.EFFORT))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.POSITION_VELOCITY))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.POSITION))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.VELOCITY))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_general")], int(ActuatorMode.NONE))

        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 100.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 10.0)
        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_position")], 200.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_velocity")], 20.0)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for i in range(5):
            self.assertEqual(ctrl_source[i], CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[5], CtrlSource.CTRL_DIRECT)
        self.assertEqual(ctrl_source[6], CtrlSource.CTRL_DIRECT)

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()
        newton_ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        newton_ctrlrange = model.mujoco.actuator_ctrlrange.numpy()
        newton_trntype = model.mujoco.actuator_trntype.numpy()

        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 100.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 10.0)
        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_position")], 200.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_velocity")], 20.0)

        np.testing.assert_allclose(newton_gainprm[5, :3], [50.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(newton_biasprm[5, :3], [0.0, -50.0, -5.0], atol=1e-5)
        self.assertEqual(newton_ctrllimited[5], True)
        np.testing.assert_allclose(newton_ctrlrange[5], [-1.0, 1.0], atol=1e-5)
        self.assertEqual(newton_trntype[5], 0)
        np.testing.assert_allclose(newton_gainprm[6, :3], [30.0, 0.0, 0.0], atol=1e-5)
        self.assertEqual(newton_trntype[6], 4)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 6)
        self.assertEqual(mj_model.nq, 12)
        self.assertEqual(mj_model.nv, 11)

        mjc_ctrl_source = solver.mjc_actuator_ctrl_source.numpy()
        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()

        for mj_idx in range(mj_model.nu):
            if mjc_ctrl_source[mj_idx] == CtrlSource.CTRL_DIRECT:
                newton_idx = mjc_to_newton[mj_idx]
                np.testing.assert_allclose(
                    mj_model.actuator_gainprm[mj_idx, :3],
                    newton_gainprm[newton_idx, :3],
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    mj_model.actuator_biasprm[mj_idx, :3],
                    newton_biasprm[newton_idx, :3],
                    atol=1e-5,
                )
            else:
                idx = mjc_to_newton[mj_idx]
                if idx >= 0:
                    kp = joint_target_ke[idx]
                    kd = joint_target_kd[idx]
                    mode = joint_act_mode[idx]
                    if mode == int(ActuatorMode.POSITION):
                        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 1], -kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 2], -kd, atol=1e-5)
                    elif mode == int(ActuatorMode.POSITION_VELOCITY):
                        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 1], -kp, atol=1e-5)
                else:
                    dof_idx = -(idx + 2)
                    kd = joint_target_kd[dof_idx]
                    np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kd, atol=1e-5)
                    np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 2], -kd, atol=1e-5)

    def test_parsing_ctrl_direct_true(self):
        """Test parsing with ctrl_direct=True."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=True)

        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.NONE))
        self.assertEqual(builder.joint_act_mode[get_qd_start(builder, "joint_general")], int(ActuatorMode.NONE))

        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 7)

        joint_act_mode = model.joint_act_mode.numpy()
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_motor")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_pos_vel")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_position")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_velocity")], int(ActuatorMode.NONE))
        self.assertEqual(joint_act_mode[get_qd_start(builder, "joint_general")], int(ActuatorMode.NONE))

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for i in range(7):
            self.assertEqual(ctrl_source[i], CtrlSource.CTRL_DIRECT)

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 7)
        self.assertEqual(mj_model.nq, 12)
        self.assertEqual(mj_model.nv, 11)

        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()

        for mj_idx in range(mj_model.nu):
            newton_idx = mjc_to_newton[mj_idx]
            np.testing.assert_allclose(
                mj_model.actuator_gainprm[mj_idx, :3],
                newton_gainprm[newton_idx, :3],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                mj_model.actuator_biasprm[mj_idx, :3],
                newton_biasprm[newton_idx, :3],
                atol=1e-5,
            )

    def test_multiworld_ctrl_direct_false(self):
        """Test multiworld with ctrl_direct=False."""
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        main_builder = newton.ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 14)

        actuator_world = model.mujoco.actuator_world.numpy()
        self.assertEqual(len(actuator_world), 14)
        for i in range(7):
            self.assertEqual(actuator_world[i], 0)
        for i in range(7, 14):
            self.assertEqual(actuator_world[i], 1)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for w in range(2):
            offset = w * 7
            for i in range(5):
                self.assertEqual(ctrl_source[offset + i], CtrlSource.JOINT_TARGET)
            self.assertEqual(ctrl_source[offset + 5], CtrlSource.CTRL_DIRECT)
            self.assertEqual(ctrl_source[offset + 6], CtrlSource.CTRL_DIRECT)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 6)
        self.assertEqual(mj_model.nq, 12)
        self.assertEqual(mj_model.nv, 11)

        mjw_gainprm = solver.mjw_model.actuator_gainprm.numpy()
        mjw_biasprm = solver.mjw_model.actuator_biasprm.numpy()

        for world in range(2):
            np.testing.assert_allclose(mjw_gainprm[world, 0, 0], 100.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 0, 1], -100.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 1, 0], 10.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 1, 2], -10.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 2, 0], 200.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 2, 1], -200.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 3, 0], 20.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 3, 2], -20.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 4, 0], 50.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 4, 1], -50.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 5, 0], 30.0, atol=1e-5)

    def test_multiworld_ctrl_direct_true(self):
        """Test multiworld with ctrl_direct=True."""
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=True)

        main_builder = newton.ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 14)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for i in range(14):
            self.assertEqual(ctrl_source[i], CtrlSource.CTRL_DIRECT)

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 7)
        self.assertEqual(mj_model.nq, 12)
        self.assertEqual(mj_model.nv, 11)

        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()

        for mj_idx in range(mj_model.nu):
            newton_idx = mjc_to_newton[mj_idx]
            np.testing.assert_allclose(
                mj_model.actuator_gainprm[mj_idx, :3],
                newton_gainprm[newton_idx, :3],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                mj_model.actuator_biasprm[mj_idx, :3],
                newton_biasprm[newton_idx, :3],
                atol=1e-5,
            )

        mjw_gainprm = solver.mjw_model.actuator_gainprm.numpy()
        mjw_biasprm = solver.mjw_model.actuator_biasprm.numpy()

        for world in range(2):
            for mj_idx in range(mj_model.nu):
                newton_idx = mjc_to_newton[mj_idx]
                world_newton_idx = world * 7 + newton_idx
                np.testing.assert_allclose(
                    mjw_gainprm[world, mj_idx, :3],
                    newton_gainprm[world_newton_idx, :3],
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    mjw_biasprm[world, mj_idx, :3],
                    newton_biasprm[world_newton_idx, :3],
                    atol=1e-5,
                )

    def test_ordering_matches_native_mujoco(self):
        """Test actuator ordering matches native MuJoCo loading."""
        import mujoco  # noqa: PLC0415

        native_model = mujoco.MjModel.from_xml_string(MJCF_ACTUATORS)

        builder = newton.ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        newton_mj = solver.mj_model

        self.assertEqual(native_model.nu, newton_mj.nu)

        for i in range(native_model.nu):
            np.testing.assert_allclose(
                native_model.actuator_gainprm[i, :3],
                newton_mj.actuator_gainprm[i, :3],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                native_model.actuator_biasprm[i, :3],
                newton_mj.actuator_biasprm[i, :3],
                atol=1e-5,
            )
            self.assertEqual(
                native_model.actuator_trnid[i, 0],
                newton_mj.actuator_trnid[i, 0],
            )

    def test_joint_target_gains_update_with_notify_changes(self):
        """Test that JOINT_TARGET actuator gains update when joint_target_ke/kd change."""
        builder = newton.ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        initial_gainprm = solver.mjw_model.actuator_gainprm.numpy().copy()
        initial_biasprm = solver.mjw_model.actuator_biasprm.numpy().copy()

        pos_vel_dof = get_qd_start(builder, "joint_pos_vel")
        position_dof = get_qd_start(builder, "joint_position")
        velocity_dof = get_qd_start(builder, "joint_velocity")

        np.testing.assert_allclose(initial_gainprm[0, 0, 0], 100.0, atol=1e-5)
        np.testing.assert_allclose(initial_biasprm[0, 0, 1], -100.0, atol=1e-5)
        np.testing.assert_allclose(initial_gainprm[0, 1, 0], 10.0, atol=1e-5)
        np.testing.assert_allclose(initial_biasprm[0, 1, 2], -10.0, atol=1e-5)
        np.testing.assert_allclose(initial_gainprm[0, 2, 0], 200.0, atol=1e-5)
        np.testing.assert_allclose(initial_biasprm[0, 2, 1], -200.0, atol=1e-5)
        np.testing.assert_allclose(initial_gainprm[0, 3, 0], 20.0, atol=1e-5)
        np.testing.assert_allclose(initial_biasprm[0, 3, 2], -20.0, atol=1e-5)

        new_ke = model.joint_target_ke.numpy()
        new_kd = model.joint_target_kd.numpy()
        new_ke[pos_vel_dof] = 500.0
        new_kd[pos_vel_dof] = 50.0
        new_ke[position_dof] = 800.0
        new_kd[velocity_dof] = 80.0
        model.joint_target_ke.assign(new_ke)
        model.joint_target_kd.assign(new_kd)

        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        updated_gainprm = solver.mjw_model.actuator_gainprm.numpy()
        updated_biasprm = solver.mjw_model.actuator_biasprm.numpy()

        np.testing.assert_allclose(updated_gainprm[0, 0, 0], 500.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 0, 1], -500.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 1, 0], 50.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 1, 2], -50.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 2, 0], 800.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 2, 1], -800.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 3, 0], 80.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 3, 2], -80.0, atol=1e-5)

        ctrl_direct_gainprm_before = initial_gainprm[0, 4, 0]
        ctrl_direct_gainprm_after = updated_gainprm[0, 4, 0]
        np.testing.assert_allclose(ctrl_direct_gainprm_before, ctrl_direct_gainprm_after, atol=1e-5)

    def test_multiworld_joint_target_gains_update(self):
        """Test that JOINT_TARGET gains update correctly in multiworld setup."""
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        main_builder = newton.ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)

        initial_gainprm = solver.mjw_model.actuator_gainprm.numpy().copy()

        for world in range(2):
            np.testing.assert_allclose(initial_gainprm[world, 0, 0], 100.0, atol=1e-5)
            np.testing.assert_allclose(initial_gainprm[world, 2, 0], 200.0, atol=1e-5)

        new_ke = model.joint_target_ke.numpy()
        new_kd = model.joint_target_kd.numpy()

        dofs_per_world = robot_builder.joint_dof_count
        for world in range(2):
            offset = world * dofs_per_world
            pos_vel_dof = offset + get_qd_start(robot_builder, "joint_pos_vel")
            position_dof = offset + get_qd_start(robot_builder, "joint_position")
            new_ke[pos_vel_dof] = 500.0 + world * 100
            new_kd[pos_vel_dof] = 50.0 + world * 10
            new_ke[position_dof] = 800.0 + world * 100

        model.joint_target_ke.assign(new_ke)
        model.joint_target_kd.assign(new_kd)

        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        updated_gainprm = solver.mjw_model.actuator_gainprm.numpy()
        updated_biasprm = solver.mjw_model.actuator_biasprm.numpy()

        np.testing.assert_allclose(updated_gainprm[0, 0, 0], 500.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 0, 1], -500.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 1, 0], 50.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 1, 2], -50.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 2, 0], 800.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 2, 1], -800.0, atol=1e-5)

        np.testing.assert_allclose(updated_gainprm[1, 0, 0], 600.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 0, 1], -600.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 1, 0], 60.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 1, 2], -60.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 2, 0], 900.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 2, 1], -900.0, atol=1e-5)

        for world in range(2):
            np.testing.assert_allclose(updated_gainprm[world, 4, 0], initial_gainprm[world, 4, 0], atol=1e-5)
            np.testing.assert_allclose(updated_gainprm[world, 5, 0], initial_gainprm[world, 5, 0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
