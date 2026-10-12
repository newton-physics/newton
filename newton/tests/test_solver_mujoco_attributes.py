# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.tests.utils.mujoco import _create_actuator_test_stage


class TestMuJoCoAttributes(unittest.TestCase):
    def test_unresolved_usd_site_actuator_has_no_owner(self):
        """Leave an unresolved USD site actuator unowned."""
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_revolute(parent=-1, child=body)
        builder.add_shape_box(body=body, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([joint])
        builder.add_custom_values(
            **{
                "mujoco:actuator_trnid": wp.vec2i(0, -1),
                "mujoco:actuator_trntype": int(SolverMuJoCo.TrnType.SITE),
                "mujoco:actuator_target_label": "/World/missing_site",
            }
        )

        self.assertEqual(SolverMuJoCo._shape_owners(builder), [0])
        self.assertEqual(SolverMuJoCo._resolve_mujoco_actuator_owners(builder), [-1])

    def test_custom_attributes_from_code(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        b0 = builder.add_link()
        j0 = builder.add_joint_revolute(-1, b0, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b0, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:condim": 6})
        b1 = builder.add_link()
        j1 = builder.add_joint_revolute(b0, b1, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b1, hx=0.1, hy=0.1, hz=0.1, custom_attributes={"mujoco:condim": 4})
        b2 = builder.add_link()
        j2 = builder.add_joint_revolute(b1, b2, axis=(0.0, 0.0, 1.0))
        builder.add_shape_box(body=b2, hx=0.1, hy=0.1, hz=0.1)
        builder.add_articulation([j0, j1, j2])
        model = builder.finalize()

        # Should work fine with single world
        solver = SolverMuJoCo(model, separate_worlds=False)

        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    def test_custom_attributes_from_mjcf(self):
        mjcf = """
        <mujoco>
            <worldbody>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" condim="6" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" condim="4" />
                </body>
                <body>
                    <joint type="hinge" axis="0 0 1" />
                    <geom type="box" size="0.1 0.1 0.1" />
                </body>
            </worldbody>
        </mujoco>
        """
        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    def test_custom_attributes_from_urdf(self):
        urdf = """
        <robot name="test_robot">
            <link name="body1">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry condim="6">
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <link name="body2">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry condim="4">
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <link name="body3">
                <joint type="revolute" axis="0 0 1" />
                <collision>
                    <geometry>
                        <box size="0.1 0.1 0.1" />
                    </geometry>
                </collision>
            </link>
            <joint name="joint1" type="revolute">
                <parent link="body1" />
                <child link="body2" />
            </joint>
            <joint name="joint2" type="revolute">
                <parent link="body2" />
                <child link="body3" />
            </joint>
        </robot>
        """
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_urdf(urdf)
        model = builder.finalize()
        solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6, 4, 3])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6, 4, 3])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_custom_attributes_from_usd(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        self.assertTrue(stage)

        body_path = "/body"
        shape = UsdGeom.Cube.Define(stage, body_path)
        prim = shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.ArticulationRootAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int, True).Set(6)

        joint_path = "/joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([body_path])

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "standalone world roots"):
            solver = SolverMuJoCo(model, separate_worlds=False)
        assert hasattr(model, "mujoco")
        assert hasattr(model.mujoco, "condim")
        assert np.allclose(model.mujoco.condim.numpy(), [6])
        assert np.allclose(solver.mjw_model.geom_condim.numpy(), [6])

    def test_invalid_tendon_joint_range_remains_unowned(self):
        """Keep a tendon unowned when its joint range exceeds the available rows."""
        mjcf = """
            <mujoco model="robot">
              <worldbody>
                <body name="root">
                  <body name="link">
                    <joint name="hinge" type="hinge"/>
                    <geom type="sphere" size="0.1"/>
                    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
                  </body>
                </body>
              </worldbody>
              <tendon>
                <fixed name="tendon">
                  <joint joint="hinge" coef="1"/>
                </fixed>
              </tendon>
            </mujoco>
            """

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf)
        builder.add_mjcf(mjcf)
        builder.custom_attributes["mujoco:tendon_joint_adr"].values = [1, 1]
        builder.custom_attributes["mujoco:tendon_joint_num"].values = [2, 1]

        model = builder.finalize()

        np.testing.assert_array_equal(model.custom_frequency_articulation["mujoco:tendon"].numpy(), [-1, 1])

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_actuator_auto_limits_and_partial_ranges(self):
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene_prim = scene.GetPrim()
        scene_prim.CreateAttribute("mjc:compiler:autoLimits", Sdf.ValueTypeNames.Bool, True).Set(True)

        base = UsdGeom.Xform.Define(stage, "/World/base").GetPrim()
        link = UsdGeom.Xform.Define(stage, "/World/link").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.ArticulationRootAPI.Apply(base)

        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/joint")
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/link")])

        tendon = stage.DefinePrim("/World/fixed_tendon", "MjcTendon")
        tendon.CreateAttribute("mjc:type", Sdf.ValueTypeNames.Token, True).Set("fixed")
        tendon.CreateRelationship("mjc:path", True).SetTargets([Sdf.Path("/World/joint")])
        tendon.CreateAttribute("mjc:path:indices", Sdf.ValueTypeNames.IntArray, True).Set(Vt.IntArray([0]))
        tendon.CreateAttribute("mjc:path:coef", Sdf.ValueTypeNames.DoubleArray, True).Set(Vt.DoubleArray([1.0]))

        # Joint actuator with ctrlRange:max only (min omitted on purpose).
        act_joint = stage.DefinePrim("/World/act_joint", "MjcActuator")
        act_joint.CreateRelationship("mjc:target", True).SetTargets([Sdf.Path("/World/joint")])
        act_joint.CreateAttribute("mjc:ctrlRange:max", Sdf.ValueTypeNames.Double, True).Set(1.22173)
        act_joint.CreateAttribute("mjc:forceRange:min", Sdf.ValueTypeNames.Double, True).Set(-2.0)
        act_joint.CreateAttribute("mjc:forceRange:max", Sdf.ValueTypeNames.Double, True).Set(2.0)

        # Tendon actuator with ctrlRange:max only (min omitted on purpose).
        act_tendon = stage.DefinePrim("/World/act_tendon", "MjcActuator")
        act_tendon.CreateRelationship("mjc:target", True).SetTargets([Sdf.Path("/World/fixed_tendon")])
        act_tendon.CreateAttribute("mjc:ctrlRange:max", Sdf.ValueTypeNames.Double, True).Set(3.1415)
        act_tendon.CreateAttribute("mjc:forceRange:min", Sdf.ValueTypeNames.Double, True).Set(-1.0)
        act_tendon.CreateAttribute("mjc:forceRange:max", Sdf.ValueTypeNames.Double, True).Set(1.0)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts["mujoco:actuator"], 2)

        target_labels = list(model.mujoco.actuator_target_label)
        joint_act_idx = target_labels.index("/World/joint")
        tendon_act_idx = target_labels.index("/World/fixed_tendon")

        ctrlrange = model.mujoco.actuator_ctrlrange.numpy()
        forcerange = model.mujoco.actuator_forcerange.numpy()
        ctrllimited = model.mujoco.actuator_ctrllimited.numpy()
        forcelimited = model.mujoco.actuator_forcelimited.numpy()

        assert_np_equal(ctrlrange[joint_act_idx], np.array([0.0, 1.22173], dtype=np.float32), tol=1e-6)
        assert_np_equal(ctrlrange[tendon_act_idx], np.array([0.0, 3.1415], dtype=np.float32), tol=1e-6)
        assert_np_equal(forcerange[joint_act_idx], np.array([-2.0, 2.0], dtype=np.float32), tol=1e-6)
        assert_np_equal(forcerange[tendon_act_idx], np.array([-1.0, 1.0], dtype=np.float32), tol=1e-6)

        self.assertTrue(bool(ctrllimited[joint_act_idx]))
        self.assertTrue(bool(ctrllimited[tendon_act_idx]))
        self.assertTrue(bool(forcelimited[joint_act_idx]))
        self.assertTrue(bool(forcelimited[tendon_act_idx]))

    def test_mjc_damping_from_usd_via_schema_resolver(self):
        """Test mjc:damping attributes are parsed via SchemaResolverMjc."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        from newton.usd import SchemaResolverMjc  # noqa: PLC0415

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        # Create root body
        root_path = "/robot"
        root_shape = UsdGeom.Cube.Define(stage, root_path)
        root_prim = root_shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        UsdPhysics.CollisionAPI.Apply(root_prim)

        # Create child body
        child_path = "/robot/child"
        child_shape = UsdGeom.Cube.Define(stage, child_path)
        child_prim = child_shape.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(child_prim)
        UsdPhysics.CollisionAPI.Apply(child_prim)

        # Create joint with mjc:damping
        joint_path = "/robot/child/joint"
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        joint.CreateAxisAttr().Set("Z")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint_prim = joint.GetPrim()
        joint_prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double, True).Set(5.0)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverMjc()])
        model = builder.finalize()

        assert hasattr(model, "mujoco")
        damping_values = model.joint_damping.numpy()
        # 6 DOFs from floating base (all 0.0) + 1 DOF from revolute joint (5.0)
        assert damping_values[-1] == 5.0, f"Expected last DOF damping to be 5.0, got {damping_values}"

    def test_ref_coordinate_conversion(self):
        """Verify ref offset in coordinate conversion.

        With a hinge joint at ref=90 degrees, setting joint_q=0 in Newton
        should produce qpos=pi/2 in MuJoCo after _update_mjc_data.
        """
        mjcf_content = """<?xml version="1.0" encoding="utf-8"?>
<mujoco model="test_ref">
    <worldbody>
        <body name="base">
            <geom type="box" size="0.1 0.1 0.1"/>
            <body name="child" pos="0 0 1">
                <joint name="hinge" type="hinge" axis="0 1 0" ref="90"/>
                <geom type="box" size="0.1 0.1 0.1"/>
            </body>
        </body>
    </worldbody>
</mujoco>"""

        builder = newton.ModelBuilder()
        builder.add_mjcf(mjcf_content)
        model = builder.finalize()
        solver = SolverMuJoCo(model)

        # joint_q=0 should map to qpos=ref (pi/2)
        state = model.state()
        solver._update_mjc_data(solver.mjw_data, model, state)
        qpos = solver.mjw_data.qpos.numpy()
        np.testing.assert_allclose(qpos[0, 0], np.pi / 2, atol=1e-5, err_msg="joint_q=0 should map to qpos=ref")

        solver._mujoco_warp.kinematics(solver.mjw_model, solver.mjw_data)

        # Use _update_newton_state to get body transforms from MuJoCo
        solver._update_newton_state(model, state, solver.mjw_data, state_prev=state)

        # Compare Newton's body_q (now from MuJoCo) with MuJoCo's xpos/xquat
        newton_body_q = state.body_q.numpy()
        mjc_body_to_newton = solver.mjc_body_to_newton.numpy()

        for body_name in ["child"]:
            newton_body_idx = next(
                (i for i, lbl in enumerate(model.body_label) if lbl.endswith(f"/{body_name}")),
                None,
            )
            self.assertIsNotNone(newton_body_idx, f"Expected a body with '{body_name}' in its label")
            mjc_body_idx = np.where(mjc_body_to_newton[0] == newton_body_idx)[0][0]

            # Get Newton body position and quaternion (populated from MuJoCo via update_newton_state)
            newton_pos = newton_body_q[newton_body_idx, 0:3]
            newton_quat = newton_body_q[newton_body_idx, 3:7]  # [x, y, z, w]

            # Get MuJoCo Warp body position and quaternion
            mj_pos = solver.mjw_data.xpos.numpy()[0, mjc_body_idx]
            mj_quat_wxyz = solver.mjw_data.xquat.numpy()[0, mjc_body_idx]  # MuJoCo uses [w, x, y, z]
            mj_quat = np.array([mj_quat_wxyz[1], mj_quat_wxyz[2], mj_quat_wxyz[3], mj_quat_wxyz[0]])

            # Compare positions
            assert np.allclose(newton_pos, mj_pos, atol=0.01), (
                f"Position mismatch for {body_name}: Newton={newton_pos}, MuJoCo={mj_pos}"
            )

            # Compare quaternions (sign-invariant since q and -q represent the same rotation)
            quat_dist = min(np.linalg.norm(newton_quat - mj_quat), np.linalg.norm(newton_quat + mj_quat))
            assert quat_dist < 0.01, f"Quaternion mismatch for {body_name}: Newton={newton_quat}, MuJoCo={mj_quat}"


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdActuatorTypeAttributes(unittest.TestCase):
    """Verify dynType, gainType, biasType are parsed from MjcActuator USD prims."""

    @classmethod
    def setUpClass(cls):
        from pxr import Sdf, Vt

        def set_actuator_attrs(act):
            act.CreateAttribute("mjc:dynType", Sdf.ValueTypeNames.Token, True).Set("filter")
            act.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token, True).Set("fixed")
            act.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token, True).Set("affine")
            act.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            act.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([0.0, -100.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )

        stage = _create_actuator_test_stage(extra_actuator_attrs=set_actuator_attrs)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_type_attributes_parsed_and_compiled(self):
        """dynType, gainType, biasType should be parsed from USD and reflected in the compiled model."""
        # Newton model stores the parsed enum integers
        self.assertEqual(int(self.model.mujoco.actuator_dyntype.numpy()[0]), 2)  # filter
        self.assertEqual(int(self.model.mujoco.actuator_gaintype.numpy()[0]), 0)  # fixed
        self.assertEqual(int(self.model.mujoco.actuator_biastype.numpy()[0]), 1)  # affine

        # Compiled MuJoCo model must match
        self.assertEqual(self.solver.mj_model.actuator_dyntype[0], 2)
        self.assertEqual(self.solver.mj_model.actuator_gaintype[0], 0)
        self.assertEqual(self.solver.mj_model.actuator_biastype[0], 1)

    def test_unsupported_type_warns_naming_the_prim(self):
        """Warn on an unsupported USD gainType, naming the offending actuator prim."""
        from pxr import Sdf

        def set_actuator_attrs(act):
            act.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token, True).Set("pid")

        stage = _create_actuator_test_stage(extra_actuator_attrs=set_actuator_attrs)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        with self.assertWarnsRegex(RuntimeWarning, r"gaintype 'pid' on prim '/World/actuator'"):
            builder.add_usd(stage)
