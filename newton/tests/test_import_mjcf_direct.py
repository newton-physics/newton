# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the direct native import experiment with independently compiled MJCF."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco.import_mjcf_native import parse_mjcf as import_mjcf_direct
from newton.solvers import SolverMuJoCo


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is required")
class TestImportMjcfDirect(unittest.TestCase):
    def test_native_defaults_without_xml_bridge(self):
        """Resolve inherited actuator/material defaults without the legacy parser."""
        xml = """<mujoco><default><general gear="7"/><material rgba="1 0 0 1"/></default>
            <asset><material name="red"/></asset><worldbody><body name="b"><joint name="j"/>
            <geom size=".1" material="red"/></body></worldbody>
            <actuator><motor joint="j"/></actuator></mujoco>"""
        builder = newton.ModelBuilder()
        with patch("newton._src.utils.import_mjcf.parse_mjcf", side_effect=AssertionError("legacy parser called")):
            import_mjcf_direct(builder, xml)
        np.testing.assert_allclose(builder.shape_color[0], [1, 0, 0])
        np.testing.assert_allclose(builder.custom_attributes["mujoco:actuator_gear"].values[0], [7, 0, 0, 0, 0, 0])

    def test_native_inertia_and_primitives(self):
        """Preserve compiled inertia tensors and primitive dimensions and poses."""
        import mujoco

        xml = """<mujoco><compiler settotalmass="10"/><worldbody><body name="b" euler="20 30 40">
            <joint/><geom type="box" size=".1 .2 .3" pos=".1 .2 .3" euler="10 20 30"
                friction=".4 .03 .002" margin=".01" gap=".003" priority="3" condim="4"
                solref=".01 1" solimp=".8 .9 .002 .4 3" solmix=".3"/>
            <geom type="sphere" size=".1" pos=".4 0 0"/>
            <geom type="capsule" size=".02" fromto="0 0 0 .4 .2 .3"/>
            <geom type="cylinder" size=".1 .2"/><geom type="ellipsoid" size=".1 .2 .3"/>
            </body></worldbody></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.0
        import_mjcf_direct(builder, xml)
        np.testing.assert_allclose(builder.body_mass, reference.body_mass[1:], rtol=1e-6)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, reference.body_iquat[1])
        rotation = rotation.reshape(3, 3)
        expected = rotation @ np.diag(reference.body_inertia[1]) @ rotation.T
        np.testing.assert_allclose(np.asarray(builder.body_inertia[0]).reshape(3, 3), expected, atol=1e-6)
        # SolverMuJoCo zeroes native box margins for CCD compatibility.
        np.testing.assert_allclose(builder.shape_margin, reference.geom_margin, atol=1e-6)
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True, use_mujoco_contacts=False)
        for i, dimensions in enumerate((3, 1, 2, 2, 3)):
            np.testing.assert_allclose(
                solver.mj_model.geom_size[i, :dimensions], reference.geom_size[i, :dimensions], atol=1e-6
            )
        np.testing.assert_allclose(solver.mj_model.geom_pos, reference.geom_pos, atol=1e-6)
        for field in (
            "geom_friction",
            "geom_gap",
            "geom_priority",
            "geom_condim",
            "geom_solref",
            "geom_solimp",
            "geom_solmix",
        ):
            with self.subTest(field=field):
                np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field), atol=1e-6)

    def test_include_replicate_frames(self):
        """Let native parsing expand includes and replicated frames."""
        import mujoco

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "part.xml").write_text(
                '<mujoco><replicate count="3" offset="1 0 0"><frame pos="0 0 2">'
                '<body name="b"><geom size=".1"/></body></frame></replicate></mujoco>'
            )
            source = path / "scene.xml"
            source.write_text('<mujoco><worldbody><include file="part.xml"/></worldbody></mujoco>')
            reference = mujoco.MjModel.from_xml_path(str(source))
            builder = newton.ModelBuilder()
            import_mjcf_direct(builder, source)
            self.assertEqual(builder.body_count, 3)
            self.assertEqual(len(builder.articulation_start), 3)
            np.testing.assert_allclose([list(q.p) for q in builder.body_q], reference.body_pos[1:])

    def test_scalar_joint_parameters_and_kinematics(self):
        """Preserve joint references, passive parameters, anchors, and moving poses."""
        import mujoco

        xml = """<mujoco><worldbody><body name="base" pos=".2 .3 .4" euler="20 30 40">
            <joint name="hinge" pos=".1 .2 .3" axis="0 1 0" ref="15" springref="10"
                range="-30 60" stiffness="2" damping="3" armature=".2" frictionloss=".1" margin=".01"/>
            <geom size=".1" mass="1"/><body name="slider" pos=".2 .1 .4" euler="30 10 20">
            <joint name="slide" type="slide" axis="1 0 0" ref=".2" springref=".1" range="-.3 .5"/>
            <geom type="box" size=".1 .2 .3" mass="2"/></body></body></worldbody></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        import_mjcf_direct(builder, xml)
        model = builder.finalize()
        solver = SolverMuJoCo(model, disable_contacts=True)
        for field in (
            "qpos0",
            "qpos_spring",
            "jnt_range",
            "jnt_stiffness",
            "jnt_margin",
            "dof_damping",
            "dof_armature",
            "dof_frictionloss",
        ):
            with self.subTest(field=field):
                np.testing.assert_allclose(getattr(solver.mj_model, field), getattr(reference, field), atol=2e-6)
        state = model.state()
        model.joint_q.assign(np.array([0.23, -0.12], dtype=np.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        data = mujoco.MjData(reference)
        data.qpos[:] = reference.qpos0 + np.array([0.23, -0.12])
        mujoco.mj_forward(reference, data)
        poses = state.body_q.numpy()
        np.testing.assert_allclose(poses[:, :3], data.xpos[1:], atol=2e-6)
        for i in range(2):
            q = poses[i, 3:][[3, 0, 1, 2]]
            self.assertAlmostEqual(abs(float(q @ data.xquat[i + 1])), 1.0, places=5)

    def test_composed_actuator_damping_and_forces(self):
        """Resolve arm and finger damping using the complete assembled inertia."""
        import mujoco

        arm_geom = (
            '<joint name="shoulder" axis="0 1 0"/><geom type="capsule" size=".05" fromto="0 0 0 0 0 1" mass="1"/>'
        )
        hand_body = '<geom size=".1" mass="2"/><body name="finger" pos=".1 0 0"><joint name="finger_joint" axis="0 1 0"/><geom type="capsule" size=".02" fromto="0 0 0 0 0 .2" mass=".1"/></body>'
        for damping in ('dampratio="1"', 'kv="3"'):
            with self.subTest(damping=damping):
                arm_act = f'<position name="arm_servo" joint="shoulder" kp="100" {damping}/>'
                hand_act = '<position name="finger_servo" joint="finger_joint" kp="10" dampratio=".7"/>'
                arm = f'<mujoco><worldbody><body name="arm">{arm_geom}</body></worldbody><actuator>{arm_act}</actuator></mujoco>'
                hand = f'<mujoco><worldbody><body name="hand">{hand_body}</body></worldbody><actuator>{hand_act}</actuator></mujoco>'
                assembled = f'<mujoco><worldbody><body name="arm">{arm_geom}<body name="hand" pos="0 0 1">{hand_body}</body></body></worldbody><actuator>{arm_act}{hand_act}</actuator></mujoco>'
                reference = mujoco.MjModel.from_xml_string(assembled)
                standalone = mujoco.MjModel.from_xml_string(arm)
                builder = newton.ModelBuilder()
                import_mjcf_direct(builder, arm)
                import_mjcf_direct(builder, hand, parent_body=0, xform=wp.transform((0, 0, 1), wp.quat_identity()))
                self.assertEqual(len(builder.articulation_start), 1)
                self.assertAlmostEqual(builder.custom_attributes["mujoco:actuator_biasprm"].values[1][2], 0.7, places=6)
                solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                np.testing.assert_allclose(
                    solver.mj_model.actuator_biasprm, reference.actuator_biasprm, rtol=2e-5, atol=1e-6
                )
                np.testing.assert_allclose(
                    solver.mjw_model.actuator_biasprm.numpy().reshape(-1, 10),
                    reference.actuator_biasprm,
                    rtol=2e-5,
                    atol=1e-6,
                )
                if "dampratio" in damping:
                    self.assertGreater(
                        abs(reference.actuator_biasprm[0, 2]), 2 * abs(standalone.actuator_biasprm[0, 2])
                    )
                for qpos, qvel, ctrl in (([0, 0], [0, 0], [0.2, 0.1]), ([0.3, -0.2], [0.4, -0.5], [-0.1, 0.25])):
                    actual = mujoco.MjData(solver.mj_model)
                    expected = mujoco.MjData(reference)
                    for native, data in ((solver.mj_model, actual), (reference, expected)):
                        data.qpos[:], data.qvel[:], data.ctrl[:] = qpos, qvel, ctrl
                        mujoco.mj_forward(native, data)
                    for field in ("actuator_force", "qfrc_actuator", "actuator_length", "actuator_velocity"):
                        np.testing.assert_allclose(
                            getattr(actual, field), getattr(expected, field), rtol=3e-5, atol=2e-6
                        )

    def test_composition_collapse_and_identical_replication(self):
        """Match assembled native physics after attachment, collapsing, and identical replication."""
        import mujoco
        import mujoco_warp

        compiler = '<compiler angle="radian"><lengthrange uselimit="true"/></compiler>'
        arm_body = """<joint name="shoulder" axis="0 1 0" springdamper=".2 .7"/>
            <geom type="capsule" size=".05" fromto="0 0 0 0 0 1" mass="1"/>"""
        hand_body = """<geom size=".1" mass="2"/><site name="palm" pos="0 0 .1"/>
            <body name="mount" pos=".1 0 .2"><geom size=".02" mass=".2"/>
            <body name="finger"><joint name="finger_joint" type="slide" axis="0 0 1" ref=".1"
            range=".1 .4" springdamper=".3 .8"/><geom size=".03" mass=".1"/>
            <site name="tip" pos="0 0 .2"/></body></body>"""
        tendon = """<tendon><spatial stiffness="5" damping=".2" springlength="-1">
            <site site="palm"/><site site="tip"/></spatial></tendon>"""
        arm_actuator = '<position name="arm_servo" joint="shoulder" kp="50" dampratio=".7"/>'
        hand_actuators = """<position name="finger_servo" joint="finger_joint" kp="20" dampratio=".5"/>
            <muscle name="muscle" joint="finger_joint"/>"""
        arm = f'<mujoco>{compiler}<worldbody><body name="arm">{arm_body}</body></worldbody><actuator>{arm_actuator}</actuator></mujoco>'
        hand = f'<mujoco>{compiler}<worldbody><body name="hand">{hand_body}</body></worldbody>{tendon}<actuator>{hand_actuators}</actuator></mujoco>'
        assembled = f"""<mujoco>{compiler}<worldbody><body name="arm">{arm_body}
            <body name="hand" pos="0 0 1">{hand_body}</body></body></worldbody>
            {tendon}<actuator>{arm_actuator}{hand_actuators}</actuator></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(assembled)
        fields = (
            "jnt_type",
            "jnt_limited",
            "jnt_stiffness",
            "dof_damping",
            "dof_armature",
            "qpos0",
            "qpos_spring",
            "actuator_trnid",
            "actuator_biasprm",
            "actuator_gainprm",
            "actuator_lengthrange",
            "tendon_lengthspring",
        )
        for collapse in (False, True):
            for copies in (1, 2):
                with self.subTest(collapse=collapse, copies=copies):
                    asset = newton.ModelBuilder()
                    asset.add_mjcf_native(arm, ctrl_direct=True)
                    asset.add_mjcf_native(
                        hand,
                        parent_body=0,
                        xform=wp.transform((0, 0, 1), wp.quat_identity()),
                        ctrl_direct=True,
                        parse_mujoco_options=False,
                        collapse_fixed_joints=collapse,
                    )
                    self.assertEqual(asset.body_count, 2 if collapse else 4)
                    self.assertEqual(asset.joint_dof_count, reference.nv)
                    builder = newton.ModelBuilder()
                    builder.replicate(asset, copies)
                    solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
                    self.assertEqual((solver.mj_model.nv, solver.mj_model.nu), (reference.nv, reference.nu))
                    limited = reference.jnt_limited.astype(bool)
                    np.testing.assert_allclose(
                        solver.mj_model.jnt_range[limited], reference.jnt_range[limited], rtol=5e-5, atol=3e-6
                    )
                    for field in fields:
                        np.testing.assert_allclose(
                            getattr(solver.mj_model, field),
                            getattr(reference, field),
                            rtol=5e-5,
                            atol=3e-6,
                            err_msg=field,
                        )
                    for q, v, ctrl, act in (
                        ([0, 0.1], [0, 0], [0.1, 0.2, 0.3], [0.2]),
                        ([0.2, 0.25], [0.3, -0.1], [-0.2, 0.3, 0.5], [0.4]),
                    ):
                        expected = mujoco.MjData(reference)
                        expected.qpos[:], expected.qvel[:], expected.ctrl[:], expected.act[:] = q, v, ctrl, act
                        mujoco.mj_forward(reference, expected)
                        for field, values in (("qpos", q), ("qvel", v), ("ctrl", ctrl), ("act", act)):
                            getattr(solver.mjw_data, field).assign(np.tile(values, (copies, 1)).astype(np.float32))
                        mujoco_warp.forward(solver.mjw_model, solver.mjw_data)
                        for field in ("qfrc_passive", "qfrc_actuator", "qfrc_bias", "actuator_force", "act_dot"):
                            np.testing.assert_allclose(
                                getattr(solver.mjw_data, field).numpy(),
                                np.tile(getattr(expected, field), (copies, 1)),
                                rtol=5e-5,
                                atol=3e-5,
                                err_msg=field,
                            )

    def test_reject_unsupported_before_mutation(self):
        """Reject unsupported semantics and native XML errors before adding bodies."""
        for content, error, message in (
            ('<geom size=".1" densitty="2"/>', ValueError, "densitty"),
            ('<joint type="slide"/><joint type="ball"/><geom size=".1"/>', NotImplementedError, "ball"),
        ):
            with self.subTest(content=content):
                builder = newton.ModelBuilder()
                with self.assertRaisesRegex(error, message):
                    import_mjcf_direct(builder, f"<mujoco><worldbody><body>{content}</body></worldbody></mujoco>")
                self.assertEqual((builder.body_count, builder.joint_count, builder.shape_count), (0, 0, 0))

    def test_actuator_parameters_and_forces(self):
        """Preserve gear, inherited ranges, activation dynamics and force limits."""
        import mujoco

        xml = """<mujoco><default><general gear="2" forcelimited="true" forcerange="-2 3"/></default>
            <worldbody><body><joint name="j" range="-30 60" ref="15"/><geom size=".1" mass="1"/></body></worldbody>
            <actuator><motor joint="j" ctrlrange="-.2 .3"/>
            <position joint="j" kp="12" kv=".3" timeconst=".02" inheritrange=".8"/>
            <velocity joint="j" kv="4"/>
            <intvelocity joint="j" kp="3" actrange="-.2 .4"/>
            <general joint="j" dyntype="filterexact" dynprm=".1" gainprm="2"/>
            </actuator></mujoco>"""
        reference = mujoco.MjModel.from_xml_string(xml)
        builder = newton.ModelBuilder()
        import_mjcf_direct(builder, xml)
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        for field in (
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_dynprm",
            "actuator_gear",
            "actuator_ctrlrange",
            "actuator_forcerange",
            "actuator_actrange",
            "actuator_ctrllimited",
            "actuator_forcelimited",
            "actuator_actlimited",
            "actuator_dyntype",
            "actuator_gaintype",
            "actuator_biastype",
            "actuator_actnum",
        ):
            with self.subTest(field=field):
                np.testing.assert_allclose(
                    getattr(solver.mj_model, field), getattr(reference, field), rtol=2e-5, atol=1e-6
                )
        actual, expected = mujoco.MjData(solver.mj_model), mujoco.MjData(reference)
        for native, data in ((solver.mj_model, actual), (reference, expected)):
            data.qpos[:], data.qvel[:], data.ctrl[:], data.act[:] = 0.4, -0.3, 0.7, 0.15
            mujoco.mj_forward(native, data)
        for field in ("actuator_force", "qfrc_actuator", "act_dot"):
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), rtol=2e-5, atol=1e-6)

    def test_attach_to_newton_authored_parent(self):
        """Resolve native hand damping after attaching to a builder-authored arm."""
        import mujoco

        finger = '<body name="finger"><joint name="finger_joint" axis="0 1 0"/><geom type="capsule" size=".02" fromto="0 0 0 0 0 .2" mass=".1"/></body>'
        actuator = '<actuator><position joint="finger_joint" kp="10" dampratio=".7"/></actuator>'
        hand = f'<mujoco><worldbody><body name="hand">{finger}</body></worldbody>{actuator}</mujoco>'
        assembled = f'<mujoco><worldbody><body name="arm"><joint axis="0 1 0"/><inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/><body name="hand" pos="0 0 1">{finger}</body></body></worldbody>{actuator}</mujoco>'
        reference = mujoco.MjModel.from_xml_string(assembled)
        builder = newton.ModelBuilder()
        arm = builder.add_link(mass=1, inertia=wp.mat33(0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1))
        joint = builder.add_joint_revolute(-1, arm, axis=(0, 1, 0))
        builder.add_articulation([joint])
        import_mjcf_direct(builder, hand, parent_body=arm, xform=wp.transform((0, 0, 1), wp.quat_identity()))
        solver = SolverMuJoCo(builder.finalize(), disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.actuator_biasprm, reference.actuator_biasprm, rtol=2e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
