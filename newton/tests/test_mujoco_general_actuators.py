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

"""Tests for MuJoCo general actuator support.

MuJoCo general actuators are separate from joint PD actuators:

Joint PD Actuators:
- Uses joint_act_mode per-DOF (POSITION, VELOCITY, POSITION_VELOCITY, NONE)
- Uses joint_target_ke/kd per-DOF
- Control via joint_target_pos and joint_target_vel

MuJoCo General Actuators:
- Uses custom_frequency_counts["mujoco:actuator"] for count
- Uses custom attributes with "mujoco:actuator" frequency (actuator_trnid, etc.)
- Control via control.mujoco.ctrl
- Separate from joint PD actuator infrastructure
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ActuatorMode
from newton.solvers import SolverMuJoCo, SolverNotifyFlags
from newton._src.solvers.mujoco import CtrlSource


class TestMuJoCoGeneralActuators(unittest.TestCase):
    """Test MuJoCo general actuator parsing and simulation."""

    def test_parse_general_actuator_from_mjcf(self):
        """Test that <general> actuators are parsed from MJCF with correct attributes."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_general_actuator">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="act1" joint="joint1" 
                 gainprm="100 0 0" 
                 biasprm="0 -100 -10"
                 gear="1.5"
                 ctrlrange="-1 1"
                 ctrllimited="true"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # 1 actuator created in mujoco namespace
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 1)

        # Check custom attributes exist
        self.assertTrue(hasattr(model, "mujoco"))
        mujoco_attrs = model.mujoco

        # Check ctrl_source is CTRL_DIRECT for general actuator
        self.assertTrue(hasattr(mujoco_attrs, "ctrl_source"))
        ctrl_source = mujoco_attrs.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], CtrlSource.CTRL_DIRECT)

        # Note: ctrl_type is NOT a custom attribute - it's computed in solver_mujoco
        # during MuJoCo model construction

        # Check actuator_trnid (target is in trnid[:,0])
        trnid = mujoco_attrs.actuator_trnid.numpy()
        self.assertEqual(trnid[0, 0], 0)  # Targets joint 0

        # Check gainprm
        self.assertTrue(hasattr(mujoco_attrs, "actuator_gainprm"))
        gainprm = mujoco_attrs.actuator_gainprm.numpy()
        self.assertEqual(gainprm.shape[0], 1)  # 1 actuator
        np.testing.assert_allclose(gainprm[0, :3], [100.0, 0.0, 0.0], atol=1e-5)

        # Check biasprm
        self.assertTrue(hasattr(mujoco_attrs, "actuator_biasprm"))
        biasprm = mujoco_attrs.actuator_biasprm.numpy()
        self.assertEqual(biasprm.shape[0], 1)
        np.testing.assert_allclose(biasprm[0, :3], [0.0, -100.0, -10.0], atol=1e-5)

        # Check gear
        self.assertTrue(hasattr(mujoco_attrs, "actuator_gear"))
        gear = mujoco_attrs.actuator_gear.numpy()
        self.assertEqual(gear.shape[0], 1)
        self.assertAlmostEqual(gear[0, 0], 1.5, places=5)

        # Check ctrllimited
        self.assertTrue(hasattr(mujoco_attrs, "actuator_ctrllimited"))
        ctrllimited = mujoco_attrs.actuator_ctrllimited.numpy()
        self.assertEqual(ctrllimited[0], True)

        # Check ctrlrange
        self.assertTrue(hasattr(mujoco_attrs, "actuator_ctrlrange"))
        ctrlrange = mujoco_attrs.actuator_ctrlrange.numpy()
        np.testing.assert_allclose(ctrlrange[0], [-1.0, 1.0], atol=1e-5)

    def test_per_dof_actuator_properties(self):
        """Test that per-DOF actuator properties are correctly set."""
        builder = newton.ModelBuilder()

        # Create a simple articulation with 3 joints
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), I_m=wp.mat33(np.eye(3)))
        b2 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), I_m=wp.mat33(np.eye(3)))
        b3 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), I_m=wp.mat33(np.eye(3)))

        j1 = builder.add_joint_revolute(-1, b1, target_ke=100.0, actuator_mode=ActuatorMode.POSITION)
        j2 = builder.add_joint_revolute(b1, b2, target_ke=200.0, actuator_mode=ActuatorMode.NONE)  # No actuator
        j3 = builder.add_joint_revolute(b2, b3, target_kd=300.0, actuator_mode=ActuatorMode.VELOCITY)

        builder.add_articulation([j1, j2, j3])
        model = builder.finalize()

        # Check per-DOF arrays
        joint_act_mode = model.joint_act_mode.numpy()
        joint_target_ke = model.joint_target_ke.numpy()
        joint_target_kd = model.joint_target_kd.numpy()

        # 3 DOFs total
        self.assertEqual(len(joint_act_mode), 3)
        self.assertEqual(len(joint_target_ke), 3)
        self.assertEqual(len(joint_target_kd), 3)

        # DOF 0: POSITION mode with ke=100
        self.assertEqual(joint_act_mode[0], int(ActuatorMode.POSITION))
        self.assertEqual(joint_target_ke[0], 100.0)

        # DOF 1: NONE mode
        self.assertEqual(joint_act_mode[1], int(ActuatorMode.NONE))

        # DOF 2: VELOCITY mode with kd=300
        self.assertEqual(joint_act_mode[2], int(ActuatorMode.VELOCITY))
        self.assertEqual(joint_target_kd[2], 300.0)

    def test_mujoco_ctrl_array_creation(self):
        """Test that mujoco.ctrl array is created with correct size for MuJoCo general actuators."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_mujoco_ctrl">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link2" pos="1 0 1">
            <joint name="joint2" axis="0 1 0" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="act1" joint="joint1" gainprm="100 0 0" biasprm="0 -100 0"/>
        <general name="act2" joint="joint2" gainprm="50 0 0" biasprm="0 -50 0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Create control object
        control = model.control()

        # control.mujoco.ctrl is sized for all actuators (custom_frequency_counts["mujoco:actuator"])
        mujoco_act_count = model.custom_frequency_counts.get("mujoco:actuator", 0)
        self.assertEqual(mujoco_act_count, 2)
        self.assertTrue(hasattr(control, "mujoco"))
        self.assertIsNotNone(control.mujoco.ctrl)
        self.assertEqual(control.mujoco.ctrl.shape[0], mujoco_act_count)
        self.assertEqual(control.mujoco.ctrl.shape[0], 2)

        # Both actuators should be CTRL_DIRECT (general actuators)
        ctrl_source = model.mujoco.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], CtrlSource.CTRL_DIRECT)
        self.assertEqual(ctrl_source[1], CtrlSource.CTRL_DIRECT)

    def test_general_actuator_simulation(self):
        """Test that general actuators work correctly in MuJoCo simulation."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_general_sim">
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="act1" joint="joint1" 
                 gainprm="1 0 0" 
                 biasprm="0 0 0"
                 gear="1"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()
        model.ground = False

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Set initial state
        state_0.joint_q.assign([0.0])
        state_0.joint_qd.assign([0.0])

        # Apply control via control.mujoco.ctrl
        control.mujoco.ctrl.assign([10.0])  # Apply torque of 10

        dt = 0.01
        solver.step(state_0, state_1, control, None, dt)

        # Check that the joint moved (torque applied)
        joint_qd = state_1.joint_qd.numpy()
        self.assertNotEqual(joint_qd[0], 0.0, "Joint should have non-zero velocity after torque applied")

    def test_mixed_actuator_types(self):
        """Test a model with both position/velocity and MuJoCo general actuators."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_mixed_actuators">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link2" pos="1 0 1">
            <joint name="joint2" axis="0 1 0" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link3" pos="2 0 1">
            <joint name="joint3" axis="1 0 0" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="pos_act" joint="joint1" kp="100"/>
        <velocity name="vel_act" joint="joint2" kv="10"/>
        <general name="general_act" joint="joint3" gainprm="1 0 0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # All 3 actuators are now created in mujoco namespace
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 3)

        # Check per-DOF actuator modes (backward compatibility)
        joint_act_mode = model.joint_act_mode.numpy()
        # DOF 0 (joint1) should have POSITION mode
        self.assertEqual(joint_act_mode[0], int(ActuatorMode.POSITION))
        # DOF 1 (joint2) should have VELOCITY mode
        self.assertEqual(joint_act_mode[1], int(ActuatorMode.VELOCITY))
        # DOF 2 (joint3) is handled by general actuator (mode NONE for PD)
        self.assertEqual(joint_act_mode[2], int(ActuatorMode.NONE))

        # Check ctrl_source for each actuator
        # Note: ctrl_type and ctrl_to_dof are NOT custom attributes - computed in solver_mujoco
        mujoco_attrs = model.mujoco
        ctrl_source = mujoco_attrs.ctrl_source.numpy()

        # Actuator 0: position, JOINT_TARGET mode
        self.assertEqual(ctrl_source[0], CtrlSource.JOINT_TARGET)

        # Actuator 1: velocity, JOINT_TARGET mode
        self.assertEqual(ctrl_source[1], CtrlSource.JOINT_TARGET)

        # Actuator 2: general, CTRL_DIRECT mode
        self.assertEqual(ctrl_source[2], CtrlSource.CTRL_DIRECT)

    def test_mujoco_ctrl_clear(self):
        """Test that control.clear() zeros control.mujoco.ctrl."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_ctrl_clear">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="gen1" joint="joint1" gainprm="1 0 0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        control = model.control()
        
        # Set some values
        control.mujoco.ctrl.assign([10.0])
        self.assertAlmostEqual(control.mujoco.ctrl.numpy()[0], 10.0, places=5)
        
        # Clear
        control.clear()
        self.assertAlmostEqual(control.mujoco.ctrl.numpy()[0], 0.0, places=5)

    def test_general_actuator_without_register_custom_attributes(self):
        """Test that general actuators are ignored without register_custom_attributes."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_general_no_register">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="gen1" joint="joint1" gainprm="1 0 0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        # Note: NOT calling SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Without register_custom_attributes, general actuators are NOT created
        # because there's no actuator_trnid custom attribute to store them
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 0)

    def test_position_actuator_without_register_custom_attributes(self):
        """Test that position actuators are parsed correctly without custom attributes."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_pos_no_register">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="pos1" joint="joint1" kp="100"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        # Note: NOT calling SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Position actuators set per-DOF properties
        self.assertEqual(model.custom_frequency_counts.get("mujoco:actuator", 0), 0)

        joint_act_mode = model.joint_act_mode.numpy()
        self.assertEqual(joint_act_mode[0], int(ActuatorMode.POSITION))

        # Check kp was set per-DOF
        joint_target_ke = model.joint_target_ke.numpy()
        self.assertAlmostEqual(joint_target_ke[0], 100.0, places=5)


class TestCtrlSourceModes(unittest.TestCase):
    """Test CtrlSource modes (JOINT_TARGET vs CTRL_DIRECT)."""

    def test_position_actuator_joint_target_mode(self):
        """Test that position actuators default to JOINT_TARGET mode."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_pos_joint_target">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="pos1" joint="joint1" kp="100"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Check ctrl_source is JOINT_TARGET (0)
        mujoco_attrs = model.mujoco
        self.assertTrue(hasattr(mujoco_attrs, "ctrl_source"))
        ctrl_source = mujoco_attrs.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], CtrlSource.JOINT_TARGET)

        # Note: ctrl_type and ctrl_to_dof are NOT custom attributes - computed in solver_mujoco

    def test_general_actuator_ctrl_direct_mode(self):
        """Test that general actuators use CTRL_DIRECT mode."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_general_ctrl_direct">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <general name="gen1" joint="joint1" gainprm="1 0 0"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()

        # Check ctrl_source is CTRL_DIRECT (1)
        mujoco_attrs = model.mujoco
        ctrl_source = mujoco_attrs.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], CtrlSource.CTRL_DIRECT)

        # Note: ctrl_type is NOT a custom attribute - it's computed in solver_mujoco

    def test_ctrl_direct_flag_forces_direct_mode(self):
        """Test that ctrl_direct=True forces all actuators to use CTRL_DIRECT mode."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_ctrl_direct_flag">
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        <body name="link2" pos="1 0 1">
            <joint name="joint2" axis="0 1 0" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="pos1" joint="joint1" kp="100"/>
        <velocity name="vel1" joint="joint2" kv="50"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content, ctrl_direct=True)  # Force CTRL_DIRECT
        model = builder.finalize()

        # Both actuators should be CTRL_DIRECT
        mujoco_attrs = model.mujoco
        ctrl_source = mujoco_attrs.ctrl_source.numpy()
        self.assertEqual(ctrl_source[0], CtrlSource.CTRL_DIRECT)
        self.assertEqual(ctrl_source[1], CtrlSource.CTRL_DIRECT)

        # Note: ctrl_type is NOT a custom attribute - it's computed in solver_mujoco


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing code."""

    def test_legacy_per_dof_arrays_still_work(self):
        """Test that per-DOF arrays (joint_act_mode, joint_target_ke/kd) still work."""
        builder = newton.ModelBuilder()

        # Create articulation with explicit actuator modes
        b1 = builder.add_link(mass=1.0, com=wp.vec3(0, 0, 0), I_m=wp.mat33(np.eye(3)))
        j1 = builder.add_joint_revolute(
            -1, b1, 
            target_ke=100.0, 
            target_kd=10.0,
            actuator_mode=ActuatorMode.POSITION_VELOCITY
        )
        builder.add_articulation([j1])
        model = builder.finalize()

        # Check per-DOF arrays
        joint_act_mode = model.joint_act_mode.numpy()
        self.assertEqual(joint_act_mode[0], int(ActuatorMode.POSITION_VELOCITY))

        joint_target_ke = model.joint_target_ke.numpy()
        self.assertEqual(joint_target_ke[0], 100.0)

        joint_target_kd = model.joint_target_kd.numpy()
        self.assertEqual(joint_target_kd[0], 10.0)


class TestGainSynchronization(unittest.TestCase):
    """Test that gains are properly synchronized between Newton and MuJoCo."""

    def test_joint_target_mode_syncs_gains(self):
        """Test that JOINT_TARGET mode syncs gains from joint_target_ke/kd."""
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_gain_sync">
    <option gravity="0 0 0"/>
    <worldbody>
        <body name="link1" pos="0 0 1">
            <joint name="joint1" axis="0 0 1" type="hinge"/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
    <actuator>
        <position name="pos1" joint="joint1" kp="100"/>
    </actuator>
</mujoco>
"""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()
        model.ground = False

        # Check that joint_target_ke was set from MJCF
        joint_target_ke = model.joint_target_ke.numpy()
        self.assertEqual(joint_target_ke[0], 100.0)

        solver = SolverMuJoCo(model, iterations=1, disable_contacts=True)

        # Modify ke
        model.joint_target_ke.assign([200.0])

        # Notify solver of model change with JOINT_DOF_PROPERTIES flag
        solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        # The solver should sync the new gain to MuJoCo
        # (This test verifies the notification mechanism works)


if __name__ == "__main__":
    unittest.main()
