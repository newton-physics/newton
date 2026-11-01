# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import tempfile
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestMuJoCoJointLimits(unittest.TestCase):
    def _make_model(self, joint_type, *, lower=None, upper=None, effort=None, device="cpu"):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/Scene")
        scene.CreateGravityMagnitudeAttr(0.0)
        root = UsdGeom.Xform.Define(stage, "/Robot")
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        base = UsdGeom.Xform.Define(stage, "/Robot/Base")
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/Root")
        fixed.CreateBody1Rel().SetTargets([base.GetPath()])
        link = UsdGeom.Xform.Define(stage, "/Robot/Link")
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(1.0)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
        schema = UsdPhysics.RevoluteJoint if joint_type == "revolute" else UsdPhysics.PrismaticJoint
        joint = schema.Define(stage, "/Robot/Joint")
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([link.GetPath()])
        if lower is not None:
            joint.CreateLowerLimitAttr(lower)
        if upper is not None:
            joint.CreateUpperLimitAttr(upper)
        if effort is not None:
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular" if joint_type == "revolute" else "linear")
            drive.CreateStiffnessAttr(100.0)
            drive.CreateDampingAttr(10.0)
            drive.CreateTargetPositionAttr(1.0)
            drive.CreateTargetVelocityAttr(1.0)
            drive.CreateMaxForceAttr(effort)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        return builder.finalize(device=device)

    def _backends(self):
        yield "cpu", True
        if wp.is_cuda_available():
            yield "cuda:0", False
        else:
            yield "cpu", False

    def test_usd_equal_joint_limits(self):
        """Build and constrain USD hinge and slide joints with equal limits."""
        for (device, use_cpu), joint_type, value in itertools.product(
            self._backends(), ("revolute", "prismatic"), (0.0, 0.25)
        ):
            with self.subTest(device=device, use_cpu=use_cpu, joint_type=joint_type, value=value):
                model = self._make_model(joint_type, lower=value, upper=value, device=device)
                model.joint_limit_ke.fill_(1.0e4)
                model.joint_limit_kd.fill_(200.0)
                lower = model.joint_limit_lower.numpy().copy()
                with self.assertWarnsRegex(UserWarning, r"/Robot/Joint.*equal.*limits"):
                    solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu)
                np.testing.assert_array_equal(model.joint_limit_lower.numpy(), lower)
                np.testing.assert_array_equal(model.joint_limit_upper.numpy(), lower)
                self.assertTrue(solver.mj_model.jnt_limited[0])
                np.testing.assert_array_equal(solver.mjw_model.jnt_range.numpy()[0], np.column_stack((lower, lower)))
                model.joint_q.assign(lower)
                state, next_state = model.state(), model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                control = model.control()
                for force in (1.0, -1.0):
                    control.joint_f.fill_(force)
                    for _ in range(100):
                        solver.step(state, next_state, control, None, 0.002)
                        state, next_state = next_state, state
                    np.testing.assert_allclose(state.joint_q.numpy(), lower, atol=1e-3)

    def test_usd_equal_joint_limits_mjcf_export(self):
        """Export a compilable narrow range around equal limits with a reference offset."""
        import mujoco

        for joint_type in ("revolute", "prismatic"):
            with self.subTest(joint_type=joint_type), tempfile.TemporaryDirectory() as tmp:
                model = self._make_model(joint_type, lower=0.25, upper=0.25)
                model.mujoco.dof_ref.fill_(0.5)
                expected = float(model.joint_limit_lower.numpy()[0]) + 0.5
                filename = os.path.join(tmp, "model.xml")
                with self.assertWarnsRegex(UserWarning, r"/Robot/Joint.*equal.*limits"):
                    solver = SolverMuJoCo(model, use_mujoco_cpu=True, save_to_mjcf=filename)
                np.testing.assert_allclose(solver.mj_model.jnt_range[0], [expected, expected], atol=1e-7)
                saved = mujoco.MjModel.from_xml_path(filename)
                self.assertLess(float(saved.jnt_range[0, 0]), expected)
                self.assertGreater(float(saved.jnt_range[0, 1]), expected)
                np.testing.assert_allclose(saved.jnt_range[0], [expected, expected], atol=2e-6)

    def test_usd_zero_drive_force(self):
        """Disable zero-force USD drives and allow runtime effort-limit updates."""
        modes = (
            newton.JointTargetMode.POSITION,
            newton.JointTargetMode.VELOCITY,
            newton.JointTargetMode.POSITION_VELOCITY,
        )
        for (device, use_cpu), joint_type, mode in itertools.product(
            self._backends(), ("revolute", "prismatic"), modes
        ):
            with self.subTest(device=device, use_cpu=use_cpu, joint_type=joint_type, mode=mode):
                model = self._make_model(joint_type, effort=0.0, device=device)
                model.joint_target_mode.fill_(int(mode))
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu)
                state, next_state = model.state(), model.state()
                newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                control = model.control()
                for _ in range(10):
                    solver.step(state, next_state, control, None, 0.002)
                    state, next_state = next_state, state
                np.testing.assert_array_equal(state.joint_q.numpy(), [0.0])
                np.testing.assert_array_equal(state.joint_qd.numpy(), [0.0])

                model.joint_effort_limit.fill_(2.0)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                solver.step(state, next_state, control, None, 0.002)
                self.assertGreater(float(next_state.joint_qd.numpy()[0]), 0.0)

                model.joint_effort_limit.zero_()
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                solver.step(next_state, state, control, None, 0.002)
                np.testing.assert_allclose(state.joint_qd.numpy(), next_state.joint_qd.numpy(), atol=1e-7)

    def test_d6_limits_per_world(self):
        """Restore per-world D6 limits and effort clamps after compiling the template."""
        template = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(template)
        body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        linear = newton.ModelBuilder.JointDofConfig(limit_lower=0.0, limit_upper=0.0, effort_limit=0.0)
        angular = newton.ModelBuilder.JointDofConfig(limit_lower=-1.0, limit_upper=1.0, effort_limit=0.0)
        joint = template.add_joint_d6(-1, body, linear_axes=[linear], angular_axes=[angular], label="d6")
        template.add_articulation([joint])
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        model = builder.finalize(device=device)
        lower = np.array([0.0, -1.0, 0.25, -1.0], dtype=np.float32)
        upper = np.array([0.0, 1.0, 0.25, 1.0], dtype=np.float32)
        ref = np.array([0.5, -0.25, 0.7, -0.5], dtype=np.float32)
        effort = np.array([0.0, 0.0, 3.0, 5.0], dtype=np.float32)
        model.joint_limit_lower.assign(lower)
        model.joint_limit_upper.assign(upper)
        model.joint_effort_limit.assign(effort)
        model.mujoco.dof_ref.assign(ref)
        with self.assertWarnsRegex(UserWarning, r"d6.*equal.*limits"):
            solver = SolverMuJoCo(model)
        np.testing.assert_allclose(
            solver.mjw_model.jnt_range.numpy(), np.column_stack((lower + ref, upper + ref)).reshape(2, 2, 2)
        )
        np.testing.assert_array_equal(
            solver.mjw_model.jnt_actfrcrange.numpy(), np.column_stack((-effort, effort)).reshape(2, 2, 2)
        )

    def test_usd_zero_drive_force_mjcf_export(self):
        """Reject MJCF export rather than saving the temporary nonzero effort limit."""
        model = self._make_model("revolute", effort=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            filename = os.path.join(tmp, "model.xml")
            with self.assertRaisesRegex(ValueError, r"/Robot/Joint.*joint_effort_limit=0.*save_to_mjcf"):
                SolverMuJoCo(model, use_mujoco_cpu=True, save_to_mjcf=filename)
            self.assertFalse(os.path.exists(filename))

    def test_usd_reversed_joint_limits(self):
        """Identify the Newton joint and reversed limits before MuJoCo compilation."""
        for joint_type in ("revolute", "prismatic"):
            with self.subTest(joint_type=joint_type):
                model = self._make_model(joint_type, lower=2.0, upper=1.0)
                with self.assertRaisesRegex(ValueError, r"/Robot/Joint.*limit_lower.*limit_upper"):
                    SolverMuJoCo(model, use_mujoco_cpu=True)

    def test_usd_negative_drive_force(self):
        """Identify the Newton joint and negative effort limit before compilation."""
        for joint_type in ("revolute", "prismatic"):
            with self.subTest(joint_type=joint_type):
                model = self._make_model(joint_type, effort=-1.0)
                with self.assertRaisesRegex(ValueError, r"/Robot/Joint.*effort_limit.*-1"):
                    SolverMuJoCo(model, use_mujoco_cpu=True)


if __name__ == "__main__":
    unittest.main()
