# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from contextlib import nullcontext
from itertools import product

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverSemiImplicit
from newton.solvers.experimental.coupled import SolverCoupled
from newton.tests.unittest_utils import USD_AVAILABLE
from newton.usd import SchemaResolverMjc, SchemaResolverNewton


def _slider_builder(*, register=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if register:
        SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf("""<mujoco>
      <compiler angle="radian"/>
      <worldbody><body name="slider">
        <joint type="slide" axis="1 0 0" ref="0.1" springref="0.35" stiffness="2"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body></worldbody>
    </mujoco>""")
    return builder


class TestJointSprings(unittest.TestCase):
    def test_import_without_solver_registration(self):
        """Import passive springs into core fields without explicit solver registration."""
        model = _slider_builder().finalize(device="cpu")
        np.testing.assert_allclose(model.joint_stiffness.numpy(), [2.0])
        np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.25])

    def test_imported_spring_force(self):
        """Accelerate an undriven unit-mass slider using its imported spring."""
        for solver_cls in (SolverFeatherstone, SolverSemiImplicit, SolverMuJoCo):
            with self.subTest(solver=solver_cls.__name__):
                model = _slider_builder(register=True).finalize(device="cpu")
                kwargs = {"use_mujoco_cpu": True} if solver_cls == SolverMuJoCo else {}
                solver = solver_cls(model, **kwargs)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.0005], rtol=1e-4, atol=1e-7)

    def test_config_names_and_preload(self):
        """Keep spring preload outside limits while exposing the canonical target names."""
        cfg = newton.ModelBuilder.JointDofConfig(
            stiffness=3.0, rest_q=2.0, target_q=0.2, target_qd=0.4, limit_lower=-1.0, limit_upper=1.0
        )
        self.assertEqual(cfg.rest_q, 2.0)
        self.assertEqual(cfg.target_q, 0.2)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(cfg.target_pos, cfg.target_q)
        with self.assertWarns(DeprecationWarning):
            cfg.target_vel = 0.8
        self.assertEqual(cfg.target_qd, 0.8)

    def test_mjcf_reference_units_and_defaults(self):
        """Convert scalar references once, including an omitted spring reference."""
        for angle, ref, springref, expected in (
            ("degree", 30.0, 45.0, np.pi / 12),
            ("radian", 0.4, 0.7, 0.3),
            ("degree", 30.0, None, -np.pi / 6),
        ):
            with self.subTest(angle=angle, springref=springref):
                spring = f'springref="{springref}"' if springref is not None else ""
                builder = newton.ModelBuilder()
                builder.add_mjcf(f'''<mujoco><compiler angle="{angle}"/><worldbody><body>
                    <joint type="hinge" ref="{ref}" {spring} stiffness="3"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                </body></worldbody></mujoco>''')
                model = builder.finalize(device="cpu")
                np.testing.assert_allclose(model.joint_rest_q.numpy(), [expected], atol=1e-7)
                np.testing.assert_allclose(model.joint_stiffness.numpy(), [3])

    def test_mjcf_hinge_to_ball_spring_conversion(self):
        """Report lost scalar spring references during the optional hinge-to-ball approximation."""
        builder = newton.ModelBuilder()
        with self.assertWarnsRegex(UserWarning, "identity spring rest orientation") as warning:
            builder.add_mjcf(
                """<mujoco><worldbody><body>
                    <joint axis="1 0 0" ref="10" stiffness="2"/>
                    <joint axis="0 1 0" ref="20" stiffness="2"/>
                    <joint axis="0 0 1" ref="30" stiffness="2"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                </body></worldbody></mujoco>""",
                convert_3d_hinge_to_ball_joints=True,
            )
        self.assertEqual(warning.filename, __file__)
        model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.joint_rest_q.numpy(), [0, 0, 0, 1])
        np.testing.assert_allclose(model.joint_stiffness.numpy(), [2, 2, 2])

    def test_mjcf_scaled_spring_rest_coordinates(self):
        """Scale imported linear spring rest coordinates while preserving angular coordinates."""
        for scale in (0.25, 2.0):
            with self.subTest(scale=scale):
                builder = newton.ModelBuilder()
                builder.add_mjcf(
                    """<mujoco><compiler angle="degree"/><worldbody><body>
                        <joint type="slide" axis="1 0 0" ref="0.1" springref="0.35" stiffness="2"/>
                        <joint type="hinge" axis="0 0 1" ref="30" springref="45" stiffness="3"/>
                        <geom type="sphere" size="0.1" mass="1"/>
                    </body></worldbody></mujoco>""",
                    scale=scale,
                )
                model = builder.finalize(device="cpu")
                np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.25 * scale, np.pi / 12], atol=1e-7)

    def test_mjcf_hinge_to_ball_without_springs(self):
        """Avoid spring approximation warnings for hinges without passive stiffness."""
        builder = newton.ModelBuilder()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder.add_mjcf(
                """<mujoco><worldbody><body>
                    <joint axis="1 0 0" ref="10"/>
                    <joint axis="0 1 0" ref="20"/>
                    <joint axis="0 0 1" ref="30"/>
                    <geom type="sphere" size="0.1" mass="1"/>
                </body></worldbody></mujoco>""",
                convert_3d_hinge_to_ball_joints=True,
            )
        np.testing.assert_allclose(builder.joint_rest_q, [0, 0, 0, 1])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_without_solver_registration(self):
        """Read MuJoCo USD spring properties into core fields without solver registration."""
        from pxr import Sdf, Usd, UsdPhysics

        for angle, ref, springref, rest in (("degree", 30, 45, np.pi / 12), ("radian", 0.4, 0.7, 0.3)):
            for register, merged in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(angle=angle, register=register, merged=merged):
                    stage = Usd.Stage.CreateInMemory()
                    stage.GetRootLayer().ImportFromString(f'''#usda 1.0
                    def PhysicsScene "scene" (prepend apiSchemas = ["MjcSceneAPI"]) {{
                        uniform token mjc:compiler:angle = "{angle}"
                    }}
                    def Xform "root" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {{
                        def Cube "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{
                            float physics:mass = 1
                            float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
                            double size = 0.1
                        }}
                        def PhysicsRevoluteJoint "hinge" (prepend apiSchemas = ["MjcJointAPI"]) {{
                            rel physics:body1 = </root/body>
                            token physics:axis = "Z"
                            float physics:lowerLimit = -30
                            float physics:upperLimit = 30
                            float mjc:ref = {ref}
                            float mjc:springref = {springref}
                            double mjc:stiffness = 3
                        }}
                    }}''')
                    if merged:
                        slide = UsdPhysics.PrismaticJoint.Define(stage, "/root/slide")
                        slide.CreateBody1Rel().SetTargets(["/root/body"])
                        slide.CreateAxisAttr().Set("X")
                        prim = slide.GetPrim()
                        prim.AddAppliedSchema("MjcJointAPI")
                        prim.CreateAttribute("mjc:stiffness", Sdf.ValueTypeNames.Double).Set(5)
                        prim.CreateAttribute("mjc:ref", Sdf.ValueTypeNames.Float).Set(0.1)
                        prim.CreateAttribute("mjc:springref", Sdf.ValueTypeNames.Float).Set(0.3)
                    builder = newton.ModelBuilder()
                    if register:
                        SolverMuJoCo.register_custom_attributes(builder)
                    builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc(), SchemaResolverNewton()])
                    model = builder.finalize(device="cpu")
                    np.testing.assert_allclose(model.joint_stiffness.numpy(), [5, 3] if merged else [3])
                    np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.2, rest] if merged else [rest], atol=1e-6)
                    offset = np.deg2rad(ref) if angle == "degree" else ref
                    np.testing.assert_allclose(model.joint_limit_lower.numpy()[-1], -np.pi / 6 - offset, atol=1e-6)
                    np.testing.assert_allclose(model.joint_limit_upper.numpy()[-1], np.pi / 6 - offset, atol=1e-6)

    def test_coupled_spring_updates(self):
        """Apply core rest edits to compacted solvers with core or legacy builder inputs."""
        for solver_cls, legacy in product((SolverFeatherstone, SolverSemiImplicit, SolverMuJoCo), (False, True)):
            with self.subTest(solver=solver_cls.__name__, legacy=legacy):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                SolverMuJoCo.register_custom_attributes(builder)
                builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                attributes = {"mujoco:dof_ref": 0.1}
                if legacy:
                    attributes.update({"mujoco:dof_passive_stiffness": 2, "mujoco:dof_springref": 0.35})
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    stiffness=0 if legacy else 2,
                    rest_q=None if legacy else 0.25,
                    custom_attributes=attributes,
                )
                builder.add_articulation([joint])
                with self.assertWarns(DeprecationWarning) if legacy else nullcontext():
                    model = builder.finalize(device="cpu")
                factory = (
                    (lambda view: SolverMuJoCo(view, use_mujoco_cpu=True)) if solver_cls == SolverMuJoCo else solver_cls
                )
                solver = SolverCoupled(
                    model, entries=[SolverCoupled.Entry(name="slider", solver=factory, bodies=[body], joints=[joint])]
                )
                stiffness = model.joint_stiffness.numpy()
                stiffness[-1] = 4.0
                model.joint_stiffness.assign(stiffness)
                rest = model.joint_rest_q.numpy()
                rest[-1] = 0.5
                model.joint_rest_q.assign(rest)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy()[-1], 0.002, atol=1e-7)

    def test_legacy_spring_inputs_are_builder_only(self):
        """Expose only core spring arrays after converting legacy builder inputs."""
        model = _slider_builder(register=True).finalize(device="cpu")
        self.assertIs(type(model.mujoco), newton.Model.AttributeNamespace)
        self.assertFalse(hasattr(model.mujoco, "dof_passive_stiffness"))
        self.assertFalse(hasattr(model.mujoco, "dof_springref"))
        self.assertIsNone(model._attribute_spec("mujoco:dof_passive_stiffness"))
        self.assertIsNone(model._attribute_spec("mujoco:dof_springref"))

    def test_builder_composition_and_coordinate_layout(self):
        """Preserve spring data through fixed-joint collapse and world replication with quaternion joints."""
        builder = newton.ModelBuilder()
        root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        free = builder.add_joint_free(root)
        ball_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        ball = builder.add_joint_ball(root, ball_body)
        fixed_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        fixed = builder.add_joint_fixed(ball_body, fixed_body)
        slider_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        slider = builder.add_joint_prismatic(fixed_body, slider_body, stiffness=4.0, rest_q=0.6)
        builder.add_articulation([free, ball, fixed, slider])
        builder.collapse_fixed_joints()
        expected_rest = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0.6])
        combined = newton.ModelBuilder()
        combined.replicate(builder, 2)
        model = combined.finalize(device="cpu")
        self.assertEqual(model.joint_rest_q.shape, model.joint_q.shape)
        np.testing.assert_allclose(model.joint_rest_q.numpy(), np.tile(expected_rest, 2))
        np.testing.assert_allclose(model.joint_stiffness.numpy(), np.tile([0] * 9 + [4], 2))

    def test_target_keyword_alias_conflicts(self):
        """Accept deprecated targets and reject contradictory old and new keyword values."""
        with self.assertWarns(DeprecationWarning):
            cfg = newton.ModelBuilder.JointDofConfig(target_pos=0.4, target_vel=0.8)
        self.assertEqual(cfg.target_q, 0.4)
        self.assertEqual(cfg.target_qd, 0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaisesRegex(ValueError, "Conflicting target_q"):
                newton.ModelBuilder.JointDofConfig(target_q=0.0, target_pos=0.4)
        for method in ("add_joint_revolute", "add_joint_prismatic"):
            builder = newton.ModelBuilder()
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            with self.assertWarns(DeprecationWarning):
                joint = getattr(builder, method)(-1, body, target_pos=0.4, target_vel=0.8)
            builder.add_articulation([joint])
            model = builder.finalize(device="cpu")
            np.testing.assert_allclose(model.joint_target_q.numpy(), [0.4])
            np.testing.assert_allclose(model.joint_target_qd.numpy(), [0.8])

    def test_passive_spring_preload_and_drive(self):
        """Apply passive spring and damping forces independently of drive suppression at a joint limit."""
        for solver_cls, joint_type, outside in product(
            (SolverFeatherstone, SolverSemiImplicit), ("revolute", "prismatic"), (False, True)
        ):
            with self.subTest(solver=solver_cls.__name__, joint_type=joint_type, outside=outside):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                joint = getattr(builder, "add_joint_" + joint_type)(
                    -1,
                    body,
                    stiffness=2,
                    rest_q=2,
                    damping=0.5,
                    target_q=0.5,
                    target_qd=0.0,
                    target_ke=3,
                    target_kd=0,
                    limit_lower=-1,
                    limit_upper=1,
                    limit_ke=0,
                    limit_kd=0,
                )
                builder.add_articulation([joint])
                q = 1.2 if outside else 0.2
                builder.joint_q[0] = q
                builder.joint_qd[0] = 0.1
                model = builder.finalize(device="cpu")
                solver = solver_cls(model, angular_damping=0.0)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                force = 2 * (2 - q) - 0.5 * 0.1 + (0 if outside else 3 * (0.5 - q))
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.1 + 0.001 * force], atol=1e-6)

    def test_ball_spring_compatibility(self):
        """Preserve isotropic MuJoCo ball springs across rest-orientation updates."""
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
        joint = builder.add_joint_ball(-1, body, stiffness=2)
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        rest = np.array([0, 0, np.sin(0.2), np.cos(0.2)], dtype=np.float32)
        model.joint_rest_q.assign(rest)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        np.testing.assert_allclose(solver.mj_model.qpos_spring, rest[[3, 0, 1, 2]], atol=1e-6)
        rest = np.array([np.sin(0.3), 0, 0, np.cos(0.3)], dtype=np.float32)
        model.joint_rest_q.assign(rest)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(solver.mj_model.qpos_spring, rest[[3, 0, 1, 2]], atol=1e-6)

    def test_ball_spring_rest_with_rotated_child_frame(self):
        """Hold a MuJoCo ball joint at its rest orientation with an authored or runtime-edited child frame."""
        rest = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)
        child_xform = wp.transform((0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5))
        for device, runtime_edit in product(wp.get_devices(), (False, True)):
            with self.subTest(device=device, runtime_edit=runtime_edit):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                authored = wp.transform_identity() if runtime_edit else child_xform
                builder.add_articulation([builder.add_joint_ball(-1, body, child_xform=authored, stiffness=25.0)])
                builder.joint_q[0:4] = list(rest)
                builder.joint_rest_q[0:4] = list(rest)
                model = builder.finalize(device=device)
                solver = SolverMuJoCo(model, use_mujoco_cpu=device.is_cpu)
                if runtime_edit:
                    model.joint_X_c.assign([child_xform])
                    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.0, 0.0, 0.0], atol=1e-6)

    def test_ball_frame_update_preserves_pending_rest(self):
        """Publish child-frame edits separately from reference-pose edits on both backends."""
        rest = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4)
        pending_rest = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.6)
        child_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5)
        for device in wp.get_devices():
            with self.subTest(device=device):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                builder.add_articulation([builder.add_joint_ball(-1, body, stiffness=25.0)])
                builder.joint_rest_q[:] = list(rest)
                model = builder.finalize(device=device)
                solver = SolverMuJoCo(model, use_mujoco_cpu=device.is_cpu)
                model.joint_rest_q.assign(list(pending_rest))
                model.joint_X_c.assign([wp.transform((0.0, 0.0, 0.0), child_rotation)])
                for flags, published_rest in (
                    (newton.ModelFlags.JOINT_PROPERTIES, rest),
                    (newton.ModelFlags.JOINT_REFERENCE_POSE_PROPERTIES, pending_rest),
                ):
                    solver.notify_model_changed(flags)
                    expected = np.array(child_rotation * published_rest * wp.quat_inverse(child_rotation))[[3, 0, 1, 2]]
                    np.testing.assert_allclose(solver.mjw_model.qpos_spring.numpy()[0], expected, atol=1e-6)
                    if device.is_cpu:
                        np.testing.assert_allclose(solver.mj_model.qpos_spring, expected, atol=1e-6)

    def test_angular_spring_quaternion_sign_and_preload(self):
        """Equivalent body quaternions produce the same scalar spring force without wrapping preload."""
        for device, joint_type, negative, (angle, rest) in product(
            wp.get_devices(), ("revolute", "d6"), (False, True), ((0.4, 0.4), (-2.8, 3.0), (2.8, -3.0), (0.4, 7.0))
        ):
            with self.subTest(device=device, joint_type=joint_type, negative=negative, angle=angle, rest=rest):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                if joint_type == "revolute":
                    joint = builder.add_joint_revolute(-1, body, axis=newton.Axis.Z, stiffness=10.0, rest_q=rest)
                else:
                    joint = builder.add_joint_d6(
                        -1,
                        body,
                        angular_axes=[
                            newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, stiffness=10.0, rest_q=rest)
                        ],
                    )
                builder.add_articulation([joint])
                builder.joint_q[:] = [angle]
                model = builder.finalize(device=device)
                solver = SolverSemiImplicit(model, angular_damping=0.0)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                if negative:
                    poses = state.body_q.numpy()
                    poses[:, 3:] *= -1.0
                    state.body_q.assign(poses)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.01 * (rest - angle)], atol=1e-6)

    def test_explicit_zero_rest_conflicts(self):
        """Distinguish an authored zero from an omitted rest value during legacy conversion."""
        for use_default, springref in product((False, True), (0.2, 0.5)):
            with self.subTest(use_default=use_default, springref=springref):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                if use_default:
                    builder.default_joint_cfg.rest_q = 0.0
                parent = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                fixed = builder.add_joint_fixed(-1, parent)
                joint = builder.add_joint_prismatic(
                    parent,
                    body,
                    rest_q=None if use_default else 0.0,
                    custom_attributes={"mujoco:dof_ref": 0.2, "mujoco:dof_springref": springref},
                )
                builder.add_articulation([fixed, joint])
                builder.collapse_fixed_joints()
                combined = newton.ModelBuilder()
                combined.replicate(builder, 2)
                with self.assertWarns(DeprecationWarning):
                    if springref == 0.5:
                        with self.assertRaisesRegex(ValueError, "Conflicting core joint spring value"):
                            combined.finalize(device="cpu")
                    else:
                        model = combined.finalize(device="cpu")
                        np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.0, 0.0], atol=1e-7)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required for graph capture")
    def test_cuda_graph_runtime_updates(self):
        """Apply core spring updates to already captured solver steps."""
        for solver_cls in (SolverFeatherstone, SolverSemiImplicit, SolverMuJoCo):
            with self.subTest(solver=solver_cls.__name__):
                model = _slider_builder(register=True).finalize(device="cuda:0")
                kwargs = {"use_mujoco_cpu": False} if solver_cls == SolverMuJoCo else {}
                solver = solver_cls(model, **kwargs)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                control = model.control()
                solver.step(state, out, control, None, 0.001)
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(state, out, control, None, 0.001)
                model.joint_stiffness.assign([4.0])
                model.joint_rest_q.assign([0.5])
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                wp.capture_launch(capture.graph)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.002], atol=2e-6)
                model.joint_rest_q.assign([0.75])
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                wp.capture_launch(capture.graph)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.003], atol=2e-6)

    def test_imported_core_edits_are_authoritative(self):
        """Allow core edits after registered import and preserve the rest pose when only ref changes."""
        builder = _slider_builder(register=True)
        builder.joint_stiffness[0] = 5.0
        builder.joint_rest_q[0] = 0.6
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        model.mujoco.dof_ref.assign([0.4])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.6])
        np.testing.assert_allclose(solver.mj_model.qpos_spring, [1.0], atol=1e-6)
        np.testing.assert_allclose(solver.mj_model.jnt_stiffness, [5.0])

    def test_legacy_authored_values_and_conflicts(self):
        """Convert legacy inputs once, preserve them through replication, and reject core conflicts."""
        for conflict in (None, "stiffness", "rest_q"):
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_revolute(
                -1,
                body,
                stiffness=5 if conflict == "stiffness" else 0,
                rest_q=0.25 if conflict == "rest_q" else None,
                custom_attributes={
                    "mujoco:dof_passive_stiffness": 3.0,
                    "mujoco:dof_springref": 0.7,
                    "mujoco:dof_ref": 0.2,
                },
            )
            builder.add_articulation([joint])
            if conflict:
                with (
                    self.assertWarns(DeprecationWarning),
                    self.assertRaisesRegex(ValueError, "Conflicting core joint spring"),
                ):
                    builder.finalize(device="cpu")
            else:
                with self.assertWarnsRegex(DeprecationWarning, "joint_stiffness") as warning:
                    model = builder.finalize(device="cpu")
                self.assertEqual(warning.filename, __file__)
                np.testing.assert_allclose(model.joint_stiffness.numpy(), [3])
                np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.5])
                solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                model.mujoco.dof_ref.assign([0.3])
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                np.testing.assert_allclose(model.joint_rest_q.numpy(), [0.5], atol=1e-7)
                np.testing.assert_allclose(solver.mj_model.qpos_spring, [0.8], atol=1e-7)
                combined = newton.ModelBuilder()
                combined.replicate(builder, 2)
                with self.assertWarns(DeprecationWarning):
                    replicated = combined.finalize(device="cpu")
                np.testing.assert_allclose(replicated.joint_stiffness.numpy(), [3, 3])
                np.testing.assert_allclose(replicated.joint_rest_q.numpy(), [0.5, 0.5], atol=1e-7)

    def test_legacy_springref_default(self):
        """Default legacy springref to zero without changing core-authored rest coordinates."""
        for joint_type, legacy, rest_q in product(("revolute", "prismatic"), (False, True), (None, 0.0, 0.4)):
            with self.subTest(joint_type=joint_type, legacy=legacy, rest_q=rest_q):
                builder = newton.ModelBuilder()
                SolverMuJoCo.register_custom_attributes(builder)
                ball_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_articulation([builder.add_joint_ball(-1, ball_body)])
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                attributes = {"mujoco:dof_ref": 0.2}
                if legacy:
                    attributes["mujoco:dof_passive_stiffness"] = 3.0
                joint = getattr(builder, "add_joint_" + joint_type)(
                    -1, body, stiffness=0 if legacy else 3, rest_q=rest_q, custom_attributes=attributes
                )
                builder.add_articulation([joint])
                expected_rest = (-0.2 if legacy else 0.0) if rest_q is None else rest_q
                with self.assertWarns(DeprecationWarning) if legacy else nullcontext():
                    model = builder.finalize(device="cpu")
                np.testing.assert_allclose(model.joint_rest_q.numpy(), [0, 0, 0, 1, expected_rest], atol=1e-7)
                solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                np.testing.assert_allclose(solver.mj_model.qpos_spring[-1], expected_rest + 0.2, atol=1e-7)
                combined = newton.ModelBuilder()
                combined.replicate(builder, 2)
                with self.assertWarns(DeprecationWarning) if legacy else nullcontext():
                    replicated = combined.finalize(device="cpu")
                np.testing.assert_allclose(
                    replicated.joint_rest_q.numpy(), np.tile([0, 0, 0, 1, expected_rest], 2), atol=1e-7
                )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_legacy_springref_default(self):
        """Preserve MuJoCo's omitted springref with default USD schema resolvers."""
        from pxr import Usd

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().ImportFromString("""#usda 1.0
        def Xform "root" {
            def Cube "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
                float physics:mass = 1
                float3 physics:diagonalInertia = (0.01, 0.01, 0.01)
                double size = 0.1
            }
            def PhysicsPrismaticJoint "slider" (prepend apiSchemas = ["MjcJointAPI"]) {
                rel physics:body1 = </root/body>
                token physics:axis = "X"
                float mjc:ref = 0.2
                double mjc:stiffness = 3
            }
        }""")
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        builder.add_articulation([0])
        with self.assertWarns(DeprecationWarning):
            model = builder.finalize(device="cpu")
        np.testing.assert_allclose(model.joint_stiffness.numpy(), [3])
        np.testing.assert_allclose(model.joint_rest_q.numpy(), [-0.2], atol=1e-7)
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        np.testing.assert_allclose(solver.mj_model.qpos_spring, [0], atol=1e-7)

    def test_d6_springs_after_quaternion_joint(self):
        """Index D6 rest coordinates correctly after a ball joint changes the coordinate offset."""
        for solver_cls in (SolverFeatherstone, SolverSemiImplicit, SolverMuJoCo):
            with self.subTest(solver=solver_cls.__name__):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                ball = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                builder.add_articulation([builder.add_joint_ball(-1, ball)])
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X, stiffness=2, rest_q=0.3),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Y, stiffness=4, rest_q=0.2),
                    ],
                    angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z, stiffness=5, rest_q=0.1)],
                )
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                kwargs = {"use_mujoco_cpu": True} if solver_cls == SolverMuJoCo else {}
                solver = solver_cls(model, **kwargs)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_qd.numpy()[3:], [0.0006, 0.0008, 0.0005], atol=1e-7)

    def test_spring_parameter_gradients(self):
        """Differentiate a slider's acceleration with respect to its stiffness and rest coordinate."""
        model = _slider_builder().finalize(device="cpu", requires_grad=True)
        solver = SolverFeatherstone(model)
        state, out = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        with wp.Tape() as tape:
            solver.step(state, out, model.control(), None, 0.001)
        tape.backward(grads={out.joint_qd: wp.ones(1, dtype=float, device="cpu")})
        np.testing.assert_allclose(model.joint_stiffness.grad.numpy(), [0.00025], atol=1e-7)
        np.testing.assert_allclose(model.joint_rest_q.grad.numpy(), [0.002], atol=1e-7)

    def test_spring_equilibrium(self):
        """Keep scalar joints at their spring rest coordinates without installing drives."""
        for solver_cls, joint_type in product(
            (SolverFeatherstone, SolverSemiImplicit, SolverMuJoCo), ("revolute", "prismatic")
        ):
            with self.subTest(solver=solver_cls.__name__, joint_type=joint_type):
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), lock_inertia=True)
                joint = getattr(builder, "add_joint_" + joint_type)(-1, body, stiffness=3, rest_q=0.4)
                builder.add_articulation([joint])
                builder.joint_q[0] = 0.4
                model = builder.finalize(device="cpu")
                kwargs = {"use_mujoco_cpu": True} if solver_cls == SolverMuJoCo else {}
                solver = solver_cls(model, **kwargs)
                state, out = model.state(), model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                solver.step(state, out, model.control(), None, 0.001)
                newton.eval_ik(model, out, out.joint_q, out.joint_qd)
                np.testing.assert_allclose(out.joint_q.numpy(), [0.4], atol=1e-7)
                np.testing.assert_allclose(out.joint_qd.numpy(), [0.0], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
