# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for MuJoCo actuator parsing and propagation."""

import os
import tempfile
import unittest

import numpy as np
import warp as wp

from newton import JointTargetMode, ModelBuilder
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton.tests import get_asset
from newton.tests.unittest_utils import USD_AVAILABLE

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
    <tendon>
        <fixed name="tendon1">
            <joint joint="joint_motor" coef="1.0"/>
            <joint joint="joint_general" coef="-0.5"/>
        </fixed>
    </tendon>
    <actuator>
        <motor name="motor1" joint="joint_motor"/>
        <position name="pos1" joint="joint_pos_vel" kp="100"/>
        <velocity name="vel1" joint="joint_pos_vel" kv="10"/>
        <position name="pos2" joint="joint_position" kp="200"/>
        <velocity name="vel2" joint="joint_velocity" kv="20"/>
        <general name="gen1" joint="joint_general" gainprm="50 0 0" biasprm="0 -50 -5" ctrlrange="-1 1" ctrllimited="true"/>
        <general name="body1" body="floating" gainprm="30 0 0" biasprm="0 0 0"/>
        <motor name="tendon_motor1" tendon="tendon1" gear="2.0"/>
    </actuator>
</mujoco>
"""

USD_MJC_ACTUATOR_TEMPLATE = """#usda 1.0
(
    defaultPrim = "Root"
    kilogramsPerUnit = 1
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Root" (
    apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Cube "Base" (
        apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1
        double size = 0.2
    }

    def Cube "Link" (
        apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1
        double size = 0.2
        double3 xformOp:translate = (0.5, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def PhysicsRevoluteJoint "Hinge" (
        apiSchemas = ["MjcJointAPI"]
    )
    {
        uniform token physics:axis = "Z"
        rel physics:body0 = </Root/Base>
        rel physics:body1 = </Root/Link>
        point3f physics:localPos0 = (0, 0, 0)
        point3f physics:localPos1 = (0, 0, 0)
        quatf physics:localRot0 = (1, 0, 0, 0)
        quatf physics:localRot1 = (1, 0, 0, 0)
    }

    def Scope "Physics"
    {
__ACTUATORS__
    }
}
"""

USD_MJC_POSITION_ACTUATOR = """        def MjcActuator "HingePosition"
        {
            uniform token mjc:biasType = "affine"
            uniform double[] mjc:gainPrm = [12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            uniform double[] mjc:biasPrm = [0, -12, 0, 0, 0, 0, 0, 0, 0, 0]
            uniform double mjc:forceRange:min = -5
            uniform double mjc:forceRange:max = 5
            rel mjc:target = </Root/Hinge>
        }
"""

USD_MJC_DAMPED_POSITION_ACTUATOR = """        def MjcActuator "HingePosition"
        {
            uniform token mjc:biasType = "affine"
            uniform double[] mjc:gainPrm = [12, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            uniform double[] mjc:biasPrm = [0, -12, -3, 0, 0, 0, 0, 0, 0, 0]
            uniform double mjc:forceRange:min = -5
            uniform double mjc:forceRange:max = 5
            rel mjc:target = </Root/Hinge>
        }
"""

USD_MJC_VELOCITY_ACTUATOR = """        def MjcActuator "HingeVelocity"
        {
            uniform token mjc:biasType = "affine"
            uniform double[] mjc:gainPrm = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            uniform double[] mjc:biasPrm = [0, 0, -4, 0, 0, 0, 0, 0, 0, 0]
            rel mjc:target = </Root/Hinge>
        }
"""

USD_MJC_DIRECT_ACTUATOR = """        def MjcActuator "HingeMotor"
        {
            uniform double[] mjc:gainPrm = [7, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            uniform double[] mjc:biasPrm = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            rel mjc:target = </Root/Hinge>
        }
"""


def make_usd_mjc_actuator_stage(*actuator_defs: str) -> str:
    return USD_MJC_ACTUATOR_TEMPLATE.replace("__ACTUATORS__", "\n\n".join(actuator_defs))


def load_usd_mjc_actuator_builder(*actuator_defs: str) -> ModelBuilder:
    with tempfile.NamedTemporaryFile("w", suffix=".usda", delete=False) as f:
        f.write(make_usd_mjc_actuator_stage(*actuator_defs))
        usd_path = f.name

    try:
        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(usd_path)
    finally:
        os.unlink(usd_path)

    return builder


def assert_solver_actuator_mapping(test, model, expected_indices):
    solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
    test.assertEqual(solver.mj_model.nu, len(expected_indices))
    np.testing.assert_array_equal(
        solver.mjc_actuator_ctrl_source.numpy(),
        [SolverMuJoCo.CtrlSource.JOINT_TARGET] * len(expected_indices),
    )
    np.testing.assert_array_equal(solver.mjc_actuator_to_newton_idx.numpy(), expected_indices)


def find_joint_by_name(builder, joint_name):
    """Find a joint index by matching the last segment of hierarchical labels."""
    for i, lbl in enumerate(builder.joint_label):
        if lbl.endswith(f"/{joint_name}") or lbl == joint_name:
            return i
    raise ValueError(f"'{joint_name}' is not in joint labels")


def get_qd_start(builder, joint_name):
    joint_idx = find_joint_by_name(builder, joint_name)
    return sum(builder.joint_dof_dim[i][0] + builder.joint_dof_dim[i][1] for i in range(joint_idx))


class TestMuJoCoActuators(unittest.TestCase):
    """Test MuJoCo actuator parsing through builder, Newton model, and MuJoCo model."""

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_mjc_position_actuator_sets_position_target(self):
        """USD MjcActuator position shortcuts drive Newton position targets."""
        builder = load_usd_mjc_actuator_builder(USD_MJC_POSITION_ACTUATOR)

        dof = get_qd_start(builder, "Hinge")
        self.assertEqual(builder.joint_target_mode[dof], int(JointTargetMode.POSITION))
        self.assertEqual(builder.joint_target_ke[dof], 12.0)
        self.assertEqual(builder.joint_target_kd[dof], 0.0)
        self.assertEqual(builder.joint_effort_limit[dof], 5.0)

        model = builder.finalize()
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 1)
        self.assertEqual(model.joint_target_mode.numpy()[dof], int(JointTargetMode.POSITION))
        np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), [SolverMuJoCo.CtrlSource.JOINT_TARGET])
        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy(), [[dof, 0]])

        assert_solver_actuator_mapping(self, model, [dof])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_mjc_velocity_actuator_sets_velocity_target(self):
        """USD MjcActuator velocity shortcuts drive Newton velocity targets."""
        builder = load_usd_mjc_actuator_builder(USD_MJC_VELOCITY_ACTUATOR)

        dof = get_qd_start(builder, "Hinge")
        self.assertEqual(builder.joint_target_mode[dof], int(JointTargetMode.VELOCITY))
        self.assertEqual(builder.joint_target_ke[dof], 0.0)
        self.assertEqual(builder.joint_target_kd[dof], 4.0)

        model = builder.finalize()
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 1)
        self.assertEqual(model.joint_target_mode.numpy()[dof], int(JointTargetMode.VELOCITY))
        np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), [SolverMuJoCo.CtrlSource.JOINT_TARGET])
        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy(), [[dof, 0]])

        assert_solver_actuator_mapping(self, model, [-(dof + 2)])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_mjc_position_velocity_actuators_set_joint_targets(self):
        """USD MjcActuator position and velocity shortcuts share a target DOF."""
        builder = load_usd_mjc_actuator_builder(USD_MJC_DAMPED_POSITION_ACTUATOR, USD_MJC_VELOCITY_ACTUATOR)

        dof = get_qd_start(builder, "Hinge")
        self.assertEqual(builder.joint_target_mode[dof], int(JointTargetMode.POSITION_VELOCITY))
        self.assertEqual(builder.joint_target_ke[dof], 12.0)
        self.assertEqual(builder.joint_target_kd[dof], 4.0)
        self.assertEqual(builder.joint_effort_limit[dof], 5.0)

        model = builder.finalize()
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 2)
        self.assertEqual(model.joint_target_mode.numpy()[dof], int(JointTargetMode.POSITION_VELOCITY))
        np.testing.assert_array_equal(
            model.mujoco.ctrl_source.numpy(),
            [SolverMuJoCo.CtrlSource.JOINT_TARGET, SolverMuJoCo.CtrlSource.JOINT_TARGET],
        )
        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy(), [[dof, 0], [dof, 0]])

        assert_solver_actuator_mapping(self, model, [dof, -(dof + 2)])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_mjc_direct_actuator_stays_ctrl_direct(self):
        """USD MjcActuator rows that are not position/velocity shortcuts stay direct."""
        builder = load_usd_mjc_actuator_builder(USD_MJC_DIRECT_ACTUATOR)

        dof = get_qd_start(builder, "Hinge")
        self.assertEqual(builder.joint_target_mode[dof], int(JointTargetMode.NONE))

        model = builder.finalize()
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 1)
        self.assertEqual(model.joint_target_mode.numpy()[dof], int(JointTargetMode.NONE))
        np.testing.assert_array_equal(model.mujoco.ctrl_source.numpy(), [SolverMuJoCo.CtrlSource.CTRL_DIRECT])

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        self.assertEqual(solver.mj_model.nu, 1)
        np.testing.assert_array_equal(solver.mjc_actuator_ctrl_source.numpy(), [SolverMuJoCo.CtrlSource.CTRL_DIRECT])
        np.testing.assert_array_equal(solver.mjc_actuator_to_newton_idx.numpy(), [0])

    def test_parsing_ctrl_direct_false(self):
        """Test parsing with ctrl_direct=False."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        self.assertEqual(len(builder.joint_target_mode), 11)
        for i in range(6):
            self.assertEqual(builder.joint_target_mode[i], int(JointTargetMode.NONE))
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_motor")], int(JointTargetMode.NONE))
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "joint_pos_vel")], int(JointTargetMode.POSITION_VELOCITY)
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "joint_position")], int(JointTargetMode.POSITION)
        )
        self.assertEqual(
            builder.joint_target_mode[get_qd_start(builder, "joint_velocity")], int(JointTargetMode.VELOCITY)
        )
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_general")], int(JointTargetMode.NONE))

        self.assertEqual(builder.joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 100.0)
        self.assertEqual(builder.joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 10.0)
        self.assertEqual(builder.joint_target_ke[get_qd_start(builder, "joint_position")], 200.0)
        self.assertEqual(builder.joint_target_kd[get_qd_start(builder, "joint_velocity")], 20.0)

        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 8)

        joint_target_mode = model.joint_target_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        for i in range(6):
            self.assertEqual(joint_target_mode[i], int(JointTargetMode.NONE))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_motor")], int(JointTargetMode.NONE))
        self.assertEqual(
            joint_target_mode[get_qd_start(builder, "joint_pos_vel")], int(JointTargetMode.POSITION_VELOCITY)
        )
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_position")], int(JointTargetMode.POSITION))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_velocity")], int(JointTargetMode.VELOCITY))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_general")], int(JointTargetMode.NONE))

        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_pos_vel")], 100.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_pos_vel")], 10.0)
        self.assertEqual(joint_target_ke[get_qd_start(builder, "joint_position")], 200.0)
        self.assertEqual(joint_target_kd[get_qd_start(builder, "joint_velocity")], 20.0)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
        for i in range(1, 5):
            self.assertEqual(ctrl_source[i], SolverMuJoCo.CtrlSource.JOINT_TARGET)
        self.assertEqual(ctrl_source[5], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
        self.assertEqual(ctrl_source[6], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
        self.assertEqual(ctrl_source[7], SolverMuJoCo.CtrlSource.CTRL_DIRECT)  # tendon actuator

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()
        newton_ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        newton_ctrlrange = model.mujoco.actuator_ctrlrange.numpy()
        newton_trntype = model.mujoco.actuator_trntype.numpy()
        newton_gear = model.mujoco.actuator_gear.numpy()

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
        self.assertEqual(newton_trntype[6], 4)  # body
        # Tendon actuator
        np.testing.assert_allclose(newton_gainprm[7, :3], [1.0, 0.0, 0.0], atol=1e-5)  # motor default
        self.assertEqual(newton_trntype[7], 2)  # tendon
        np.testing.assert_allclose(newton_gear[7], [2.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-5)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 8)
        self.assertEqual(mj_model.nq, 12)
        self.assertEqual(mj_model.nv, 11)

        mjc_ctrl_source = solver.mjc_actuator_ctrl_source.numpy()
        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()

        for mj_idx in range(mj_model.nu):
            if mjc_ctrl_source[mj_idx] == SolverMuJoCo.CtrlSource.CTRL_DIRECT:
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
                np.testing.assert_allclose(
                    mj_model.actuator_gear[mj_idx],
                    newton_gear[newton_idx],
                    atol=1e-5,
                )
            else:
                idx = mjc_to_newton[mj_idx]
                if idx >= 0:
                    kp = joint_target_ke[idx]
                    kd = joint_target_kd[idx]
                    mode = joint_target_mode[idx]
                    if mode == int(JointTargetMode.POSITION):
                        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 1], -kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 2], -kd, atol=1e-5)
                    elif mode == int(JointTargetMode.POSITION_VELOCITY):
                        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kp, atol=1e-5)
                        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 1], -kp, atol=1e-5)
                else:
                    dof_idx = -(idx + 2)
                    kd = joint_target_kd[dof_idx]
                    np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, 0], kd, atol=1e-5)
                    np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, 2], -kd, atol=1e-5)

    def test_parsing_ctrl_direct_true(self):
        """Test parsing with ctrl_direct=True."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=True)

        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_motor")], int(JointTargetMode.NONE))
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_pos_vel")], int(JointTargetMode.NONE))
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_position")], int(JointTargetMode.NONE))
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_velocity")], int(JointTargetMode.NONE))
        self.assertEqual(builder.joint_target_mode[get_qd_start(builder, "joint_general")], int(JointTargetMode.NONE))

        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 8)

        joint_target_mode = model.joint_target_mode.numpy()
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_motor")], int(JointTargetMode.NONE))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_pos_vel")], int(JointTargetMode.NONE))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_position")], int(JointTargetMode.NONE))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_velocity")], int(JointTargetMode.NONE))
        self.assertEqual(joint_target_mode[get_qd_start(builder, "joint_general")], int(JointTargetMode.NONE))

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for i in range(8):
            self.assertEqual(ctrl_source[i], SolverMuJoCo.CtrlSource.CTRL_DIRECT)

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()

        # Verify tendon actuator trntype
        newton_trntype = model.mujoco.actuator_trntype.numpy()
        self.assertEqual(newton_trntype[7], 2)  # tendon

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 8)
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
        robot_builder = ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        main_builder = ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 16)

        actuator_world = model.mujoco.actuator_world.numpy()
        self.assertEqual(len(actuator_world), 16)
        for i in range(8):
            self.assertEqual(actuator_world[i], 0)
        for i in range(8, 16):
            self.assertEqual(actuator_world[i], 1)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for w in range(2):
            offset = w * 8
            self.assertEqual(ctrl_source[offset + 0], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
            for i in range(1, 5):
                self.assertEqual(ctrl_source[offset + i], SolverMuJoCo.CtrlSource.JOINT_TARGET)
            self.assertEqual(ctrl_source[offset + 5], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
            self.assertEqual(ctrl_source[offset + 6], SolverMuJoCo.CtrlSource.CTRL_DIRECT)
            self.assertEqual(ctrl_source[offset + 7], SolverMuJoCo.CtrlSource.CTRL_DIRECT)  # tendon

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 8)
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
            np.testing.assert_allclose(mjw_gainprm[world, 4, 0], 1.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 5, 0], 50.0, atol=1e-5)
            np.testing.assert_allclose(mjw_biasprm[world, 5, 1], -50.0, atol=1e-5)
            np.testing.assert_allclose(mjw_gainprm[world, 6, 0], 30.0, atol=1e-5)

    def test_multiworld_ctrl_direct_true(self):
        """Test multiworld with ctrl_direct=True."""
        robot_builder = ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=True)

        main_builder = ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 16)

        ctrl_source = model.mujoco.ctrl_source.numpy()
        for i in range(16):
            self.assertEqual(ctrl_source[i], SolverMuJoCo.CtrlSource.CTRL_DIRECT)

        newton_gainprm = model.mujoco.actuator_gainprm.numpy()
        newton_biasprm = model.mujoco.actuator_biasprm.numpy()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)
        mj_model = solver.mj_model

        self.assertEqual(mj_model.nu, 8)
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
                world_newton_idx = world * 8 + newton_idx
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
        native_model = SolverMuJoCo.import_mujoco()[0].MjModel.from_xml_string(MJCF_ACTUATORS)

        builder = ModelBuilder()
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

    def test_multiworld_joint_target_gains_update(self):
        """Test that JOINT_TARGET gains update correctly in multiworld setup."""
        robot_builder = ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        main_builder = ModelBuilder()
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
            velocity_dof = offset + get_qd_start(robot_builder, "joint_velocity")
            new_ke[pos_vel_dof] = 500.0 + world * 100
            new_kd[pos_vel_dof] = 50.0 + world * 10
            new_ke[position_dof] = 800.0 + world * 100
            new_kd[velocity_dof] = 80.0 + world * 10

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

        np.testing.assert_allclose(updated_gainprm[1, 0, 0], 600.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 0, 1], -600.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 1, 0], 60.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 1, 2], -60.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 2, 0], 900.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 2, 1], -900.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 3, 0], 90.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 3, 2], -90.0, atol=1e-5)

        for world in range(2):
            np.testing.assert_allclose(updated_gainprm[world, 4, 0], initial_gainprm[world, 4, 0], atol=1e-5)
            np.testing.assert_allclose(updated_gainprm[world, 5, 0], initial_gainprm[world, 5, 0], atol=1e-5)

    def test_multiworld_ctrl_direct_gains_update(self):
        """Test that CTRL_DIRECT actuator gains update correctly in multiworld setup."""
        robot_builder = ModelBuilder()
        robot_builder.add_mjcf(MJCF_ACTUATORS, ctrl_direct=False)

        main_builder = ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)

        initial_gainprm = solver.mjw_model.actuator_gainprm.numpy().copy()
        initial_biasprm = solver.mjw_model.actuator_biasprm.numpy().copy()

        for world in range(2):
            np.testing.assert_allclose(initial_gainprm[world, 4, 0], 1.0, atol=1e-5)
            np.testing.assert_allclose(initial_gainprm[world, 5, 0], 50.0, atol=1e-5)
            np.testing.assert_allclose(initial_biasprm[world, 5, 1], -50.0, atol=1e-5)
            np.testing.assert_allclose(initial_gainprm[world, 6, 0], 30.0, atol=1e-5)

        new_gainprm = model.mujoco.actuator_gainprm.numpy()
        new_biasprm = model.mujoco.actuator_biasprm.numpy()

        actuators_per_world = 8
        for world in range(2):
            offset = world * actuators_per_world
            new_gainprm[offset + 5, 0] = 150.0 + world * 50
            new_biasprm[offset + 5, 1] = -150.0 - world * 50
            new_biasprm[offset + 5, 2] = -15.0 - world * 5
            new_gainprm[offset + 6, 0] = 90.0 + world * 30

        model.mujoco.actuator_gainprm.assign(new_gainprm)
        model.mujoco.actuator_biasprm.assign(new_biasprm)

        solver.notify_model_changed(SolverNotifyFlags.ACTUATOR_PROPERTIES)

        updated_gainprm = solver.mjw_model.actuator_gainprm.numpy()
        updated_biasprm = solver.mjw_model.actuator_biasprm.numpy()

        np.testing.assert_allclose(updated_gainprm[0, 5, 0], 150.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 5, 1], -150.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[0, 5, 2], -15.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[0, 6, 0], 90.0, atol=1e-5)

        np.testing.assert_allclose(updated_gainprm[1, 5, 0], 200.0, atol=1e-5)
        np.testing.assert_allclose(updated_biasprm[1, 5, 1], -200.0, atol=1e-5)
        # biasprm[2] is set per-world from user custom attributes.
        np.testing.assert_allclose(updated_biasprm[1, 5, 2], -20.0, atol=1e-5)
        np.testing.assert_allclose(updated_gainprm[1, 6, 0], 120.0, atol=1e-5)

        for world in range(2):
            np.testing.assert_allclose(updated_gainprm[world, 0, 0], initial_gainprm[world, 0, 0], atol=1e-5)
            np.testing.assert_allclose(updated_gainprm[world, 1, 0], initial_gainprm[world, 1, 0], atol=1e-5)
            np.testing.assert_allclose(updated_gainprm[world, 2, 0], initial_gainprm[world, 2, 0], atol=1e-5)
            np.testing.assert_allclose(updated_gainprm[world, 3, 0], initial_gainprm[world, 3, 0], atol=1e-5)

    def test_combined_joint_per_dof_actuators(self):
        """Test that actuators targeting individual MJCF joints apply gains only to specific DOFs.

        When a body has multiple MJCF joints, Newton combines them into one joint.
        This test verifies that actuators targeting individual MJCF joint names
        correctly apply gains to only the corresponding DOF, not all DOFs.
        """
        # MJCF with multiple joints in one body - will be combined into a single Newton joint
        mjcf_combined_joints = """<?xml version="1.0" encoding="utf-8"?>
        <mujoco model="test_combined_joints">
            <option gravity="0 0 0"/>
            <worldbody>
                <body name="base" pos="0 0 1">
                    <freejoint name="root"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                    <body name="arm" pos="0.2 0 0">
                        <!-- Three joints in one body - combined into one Newton D6 joint -->
                        <joint name="shoulder_x" type="hinge" axis="1 0 0"/>
                        <joint name="shoulder_y" type="hinge" axis="0 1 0"/>
                        <joint name="shoulder_z" type="hinge" axis="0 0 1"/>
                        <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <!-- Target individual MJCF joints with different gains -->
                <position name="pos_x" joint="shoulder_x" kp="100"/>
                <position name="pos_y" joint="shoulder_y" kp="200"/>
                <velocity name="vel_z" joint="shoulder_z" kv="30"/>
            </actuator>
        </mujoco>
        """

        builder = ModelBuilder()
        builder.add_mjcf(mjcf_combined_joints, ctrl_direct=False)

        # Verify the combined joint was created
        combined_name = "test_combined_joints/worldbody/base/arm/shoulder_x_shoulder_y_shoulder_z"
        self.assertIn(combined_name, builder.joint_label)

        # Get the qd_start for the combined joint
        combined_joint_idx = builder.joint_label.index(combined_name)
        qd_start = builder.joint_qd_start[combined_joint_idx]

        # The free joint has 6 DOFs (0-5), so the combined joint DOFs start at 6
        # shoulder_x -> DOF 6, shoulder_y -> DOF 7, shoulder_z -> DOF 8
        self.assertEqual(qd_start, 6)

        # Verify gains are applied to specific DOFs, not all DOFs
        # DOF 6 (shoulder_x): kp=100, kv=0 -> POSITION mode
        self.assertEqual(builder.joint_target_ke[6], 100.0)
        self.assertEqual(builder.joint_target_kd[6], 0.0)
        self.assertEqual(builder.joint_target_mode[6], int(JointTargetMode.POSITION))

        # DOF 7 (shoulder_y): kp=200, kv=0 -> POSITION mode
        self.assertEqual(builder.joint_target_ke[7], 200.0)
        self.assertEqual(builder.joint_target_kd[7], 0.0)
        self.assertEqual(builder.joint_target_mode[7], int(JointTargetMode.POSITION))

        # DOF 8 (shoulder_z): kp=0, kv=30 -> VELOCITY mode
        self.assertEqual(builder.joint_target_ke[8], 0.0)
        self.assertEqual(builder.joint_target_kd[8], 30.0)
        self.assertEqual(builder.joint_target_mode[8], int(JointTargetMode.VELOCITY))

        # Verify freejoint DOFs (0-5) are not affected
        for i in range(6):
            self.assertEqual(builder.joint_target_mode[i], int(JointTargetMode.NONE))

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_actuator_cartpole(self):
        """Test basic actuator parsing from the MjcActuator schema"""
        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)

        builder.add_usd(get_asset("cartpole_mjc.usda"))

        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        self.assertTrue(hasattr(model, "mujoco"))
        self.assertTrue(hasattr(model.mujoco, "actuator_gear"))
        np.testing.assert_array_equal(model.mujoco.actuator_ctrllimited.numpy(), [True])
        np.testing.assert_allclose(model.mujoco.actuator_ctrlrange.numpy(), [[-3.0, 3.0]])
        np.testing.assert_allclose(model.mujoco.actuator_gear.numpy(), [[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_array_equal(solver.mjw_model.actuator_ctrllimited.numpy(), [True])
        np.testing.assert_allclose(solver.mjw_model.actuator_ctrlrange.numpy(), [[[-3.0, 3.0]]])
        np.testing.assert_allclose(solver.mjw_model.actuator_gear.numpy(), [[[50.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        np.testing.assert_array_equal(solver.mjw_model.actuator_trnid.numpy(), [[0, -1]])
        np.testing.assert_array_equal(solver.mjw_model.actuator_trntype.numpy(), [0])


MJCF_SITE_ACTUATOR = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_site_actuator">
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <body name="base" pos="0 0 1">
            <freejoint name="root"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            <site name="sensor_site" pos="0.1 0 0"/>
            <body name="arm" pos="0.2 0 0">
                <joint name="elbow" axis="0 1 0" type="hinge"/>
                <geom type="box" size="0.05 0.05 0.2" mass="0.5"/>
                <site name="ee_site" pos="0 0 0.2"/>
            </body>
        </body>
    </worldbody>
    <actuator>
        <general name="site_motor" site="ee_site" gainprm="25 0 0" biasprm="0 -25 -2"/>
        <motor name="elbow_motor" joint="elbow"/>
    </actuator>
</mujoco>
"""


class TestMuJoCoSiteActuators(unittest.TestCase):
    """Tests for site-targeted actuator support in SolverMuJoCo."""

    def test_site_actuator_parsed_from_mjcf(self):
        """Site actuator is correctly parsed with trntype=SITE from MJCF."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        model = builder.finalize()

        trntype = model.mujoco.actuator_trntype.numpy()

        # Find the site actuator among all parsed actuators
        site_trntype = int(SolverMuJoCo.TrnType.SITE)
        site_indices = [i for i in range(len(trntype)) if trntype[i] == site_trntype]
        self.assertEqual(len(site_indices), 1, "Expected exactly one SITE actuator")

        gainprm = model.mujoco.actuator_gainprm.numpy()
        np.testing.assert_allclose(gainprm[site_indices[0], :3], [25.0, 0.0, 0.0], atol=1e-5)

    def test_site_actuator_exported_to_mujoco(self):
        """Site actuator is exported to MuJoCo model with correct trntype and target."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        mj_model = solver.mj_model

        # At least 2 actuators: site_motor and elbow_motor
        self.assertGreaterEqual(mj_model.nu, 2)

        # Find the actuator with site transmission type (mjTRN_SITE = 4 in native MuJoCo)
        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts = [i for i in range(mj_model.nu) if mj_model.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(site_acts), 1, "Expected exactly one site actuator in MuJoCo model")

        mj_idx = site_acts[0]
        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, :3], [25.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(mj_model.actuator_biasprm[mj_idx, :3], [0.0, -25.0, -2.0], atol=1e-5)

        # The trnid should point to a valid site index
        site_id = mj_model.actuator_trnid[mj_idx, 0]
        self.assertGreaterEqual(site_id, 0)
        self.assertLess(site_id, mj_model.nsite)

    def test_site_actuator_matches_native_mujoco(self):
        """Site actuator properties match native MuJoCo loading."""
        native_model = SolverMuJoCo.import_mujoco()[0].MjModel.from_xml_string(MJCF_SITE_ACTUATOR)

        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        newton_mj = solver.mj_model

        self.assertEqual(native_model.nu, newton_mj.nu)

        for i in range(native_model.nu):
            self.assertEqual(
                native_model.actuator_trntype[i],
                newton_mj.actuator_trntype[i],
                f"Actuator {i} trntype mismatch",
            )
            np.testing.assert_array_equal(
                native_model.actuator_trnid[i],
                newton_mj.actuator_trnid[i],
                err_msg=f"Actuator {i} trnid mismatch",
            )
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

    def test_site_actuator_with_include_sites_false(self):
        """Site actuator is resolved even when include_sites=False."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, include_sites=False)
        mj_model = solver.mj_model

        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts = [i for i in range(mj_model.nu) if mj_model.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(site_acts), 1, "Site actuator should be exported even with include_sites=False")

        mj_idx = site_acts[0]
        np.testing.assert_allclose(mj_model.actuator_gainprm[mj_idx, :3], [25.0, 0.0, 0.0], atol=1e-5)

        site_id = mj_model.actuator_trnid[mj_idx, 0]
        self.assertGreaterEqual(site_id, 0)
        self.assertLess(site_id, mj_model.nsite)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_site_actuator_from_usd(self):
        """Site actuator is parsed from a USD MjcActuator prim targeting a MjcSiteAPI prim."""
        builder = ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(get_asset("site_actuator_mjc.usda"))
        model = builder.finalize()

        # USD parse should yield exactly one actuator with trntype=SITE
        site_trntype = int(SolverMuJoCo.TrnType.SITE)
        trntype = model.mujoco.actuator_trntype.numpy()
        site_indices = [i for i in range(len(trntype)) if trntype[i] == site_trntype]
        self.assertEqual(len(site_indices), 1, "Expected exactly one SITE actuator parsed from USD")

        solver = SolverMuJoCo(model, separate_worlds=False)
        mj_model = solver.mj_model

        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts = [i for i in range(mj_model.nu) if mj_model.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(site_acts), 1, "Expected exactly one site actuator in MuJoCo model")

        mj_idx = site_acts[0]
        site_id = mj_model.actuator_trnid[mj_idx, 0]
        self.assertGreaterEqual(site_id, 0)
        self.assertLess(site_id, mj_model.nsite)

        # 6-DoF gear round-trips from USD `uniform double[] mjc:gear = [0, 0, 10, 0, 0, 2]`
        np.testing.assert_allclose(mj_model.actuator_gear[mj_idx, :6], [0.0, 0.0, 10.0, 0.0, 0.0, 2.0], atol=1e-5)

    def test_site_actuator_applies_force_at_site(self):
        """Stepping with a site actuator produces qfrc_actuator matching native MuJoCo."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        builder.request_state_attributes("mujoco:qfrc_actuator")
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        import mujoco

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Identify the Newton-side actuator ordering so we can set matching ctrl
        # values on Newton and native MuJoCo.
        newton_trntype = model.mujoco.actuator_trntype.numpy()
        site_trntype = int(SolverMuJoCo.TrnType.SITE)
        newton_site_idx = int(np.where(newton_trntype == site_trntype)[0][0])
        newton_joint_idx = int(np.where(newton_trntype != site_trntype)[0][0])

        ctrl = np.zeros(model.mujoco.actuator_trntype.shape[0], dtype=np.float32)
        ctrl[newton_site_idx] = 0.5  # site-actuator input
        ctrl[newton_joint_idx] = 0.7  # joint-motor input
        control.mujoco.ctrl = wp.array(ctrl, dtype=wp.float32, device=model.device)

        solver.step(state_0, state_1, control, None, dt=0.001)
        qfrc_newton = state_1.mujoco.qfrc_actuator.numpy()

        # Native MuJoCo reference: mj_forward with the same ctrl mapped through
        # the MuJoCo-side actuator ordering (which may permute Newton's).
        native_model = SolverMuJoCo.import_mujoco()[0].MjModel.from_xml_string(MJCF_SITE_ACTUATOR)
        data = mujoco.MjData(native_model)
        mjc_to_newton = solver.mjc_actuator_to_newton_idx.numpy()
        for mj_idx in range(native_model.nu):
            data.ctrl[mj_idx] = ctrl[mjc_to_newton[mj_idx]]
        mujoco.mj_forward(native_model, data)
        qfrc_native = np.array(data.qfrc_actuator)

        self.assertGreater(
            float(np.max(np.abs(qfrc_native))),
            0.0,
            "Test fixture must produce a nonzero reference qfrc for the assertion to be meaningful",
        )
        np.testing.assert_allclose(qfrc_newton, qfrc_native, atol=1e-4)

    def test_site_actuator_multiworld_separate(self):
        """With separate_worlds=True, site actuators in each world are dispatched correctly."""
        robot_builder = ModelBuilder()
        robot_builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)

        main_builder = ModelBuilder()
        main_builder.add_world(robot_builder)
        main_builder.add_world(robot_builder)
        model = main_builder.finalize()

        # actuator_world should partition actuators between the two worlds.
        actuator_world = model.mujoco.actuator_world.numpy()
        per_world = len(actuator_world) // 2
        self.assertGreater(per_world, 0)
        for i in range(per_world):
            self.assertEqual(actuator_world[i], 0, f"Actuator {i} should belong to world 0")
        for i in range(per_world, 2 * per_world):
            self.assertEqual(actuator_world[i], 1, f"Actuator {i} should belong to world 1")

        # Construction must not raise -- exercises the per-world shape filter
        # for SITE trntype in _init_mjc_model_for_world.
        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, separate_worlds=True)

        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts_per_world = [
            i for i in range(solver.mj_model.nu) if solver.mj_model.actuator_trntype[i] == mj_site_trntype
        ]
        self.assertEqual(
            len(site_acts_per_world),
            1,
            "Each separate_worlds=True MuJoCo model should export exactly one site actuator",
        )

    def test_site_actuator_required_shape_preserved_with_include_sites_false(self):
        """With include_sites=False, the site shape referenced by an actuator is still exported."""
        builder = ModelBuilder()
        builder.add_mjcf(MJCF_SITE_ACTUATOR, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True, include_sites=False)
        mj_model = solver.mj_model

        # The referenced site must have survived the `include_sites=False` filter
        # because the actuator declared it as required.
        self.assertGreaterEqual(mj_model.nsite, 1, "Actuator-referenced site must be preserved")

        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts = [i for i in range(mj_model.nu) if mj_model.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(site_acts), 1)
        mj_idx = site_acts[0]

        # The actuator's trnid must point at a site that actually exists in the
        # exported MuJoCo model (verifies the shape was not silently dropped).
        site_id = int(mj_model.actuator_trnid[mj_idx, 0])
        self.assertGreaterEqual(site_id, 0)
        self.assertLess(site_id, mj_model.nsite)
        self.assertGreaterEqual(int(mj_model.site_bodyid[site_id]), 0)

    def test_site_actuator_6dof_gear_mjcf(self):
        """Non-trivial 6-DoF gear on a site actuator round-trips through MJCF."""
        mjcf_6dof_gear = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_site_gear">
    <option gravity="0 0 -9.81"/>
    <worldbody>
        <body name="base" pos="0 0 1">
            <freejoint name="root"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            <site name="wrench_site" pos="0 0 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="wrench" site="wrench_site" gear="1 2 3 4 5 6"/>
    </actuator>
</mujoco>
"""

        builder = ModelBuilder()
        builder.add_mjcf(mjcf_6dof_gear, ctrl_direct=True)
        model = builder.finalize()

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)
        newton_mj = solver.mj_model

        import mujoco

        mj_site_trntype = mujoco.mjtTrn.mjTRN_SITE
        site_acts = [i for i in range(newton_mj.nu) if newton_mj.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(site_acts), 1)
        mj_idx = site_acts[0]
        np.testing.assert_allclose(newton_mj.actuator_gear[mj_idx, :6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-5)

        # Parity check: native MuJoCo sees the same gear vector on its site actuator.
        native_model = SolverMuJoCo.import_mujoco()[0].MjModel.from_xml_string(mjcf_6dof_gear)
        native_site_acts = [i for i in range(native_model.nu) if native_model.actuator_trntype[i] == mj_site_trntype]
        self.assertEqual(len(native_site_acts), 1)
        np.testing.assert_allclose(
            newton_mj.actuator_gear[mj_idx, :6],
            native_model.actuator_gear[native_site_acts[0], :6],
            atol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
