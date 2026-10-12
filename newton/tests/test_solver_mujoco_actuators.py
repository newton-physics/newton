# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton._src.solvers.mujoco.enums import _ActuatorBiasType, _ActuatorDynamicsType, _ActuatorGainType
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE, assert_np_equal
from newton.tests.utils.mujoco import _create_actuator_test_stage


class TestActuatorDampratio(unittest.TestCase):
    """Verify dampratio on position actuator shortcuts produces correct biasprm[2].

    MuJoCo's <position kp="K" dampratio="D"/> computes kd = D * 2 * sqrt(K * acc0)
    during mj_setConst. Newton must use set_to_position on the MjSpec actuator so
    the compiler resolves dampratio correctly.
    """

    MJCF = """<?xml version="1.0" ?>
    <mujoco>
        <worldbody>
            <body name="base" pos="0 0 1">
                <freejoint/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="child" pos="0 0 0.5">
                    <joint name="j1" type="hinge" axis="0 1 0"/>
                    <geom type="box" size="0.05 0.05 0.05" mass="0.5"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <position name="pos_dampratio" joint="j1" kp="100" dampratio="1"/>
            <position name="pos_kv" joint="j1" kp="100" kv="5"/>
            <motor name="motor_plain" joint="j1"/>
        </actuator>
    </mujoco>"""

    @classmethod
    def setUpClass(cls):
        builder = newton.ModelBuilder()
        builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)
        cls.native_model = cls.solver._mujoco.MjModel.from_xml_string(cls.MJCF)

    def test_dampratio_biasprm2_nonzero(self):
        """Position actuator with dampratio should have nonzero biasprm[2]."""
        bp = self.solver.mj_model.actuator_biasprm
        # Actuator 0: kp=100, dampratio=1 -> biasprm[2] computed by mj_setConst
        self.assertAlmostEqual(bp[0, 0], 0.0, places=5)
        self.assertAlmostEqual(bp[0, 1], -100.0, places=3)
        self.assertNotAlmostEqual(bp[0, 2], 0.0, places=3, msg="dampratio should produce nonzero biasprm[2]")
        self.assertLess(bp[0, 2], 0.0, "biasprm[2] should be negative (damping)")

    def test_explicit_kv_biasprm2(self):
        """Position actuator with explicit kv should have biasprm[2] = -kv."""
        bp = self.solver.mj_model.actuator_biasprm
        # Actuator 1: kp=100, kv=5 -> biasprm[2] = -5
        self.assertAlmostEqual(bp[1, 2], -5.0, places=3)

    def test_motor_biasprm_zero(self):
        """Motor actuator should have all-zero biasprm."""
        bp = self.solver.mj_model.actuator_biasprm
        np.testing.assert_allclose(bp[2], 0.0, atol=1e-6)

    def test_dampratio_matches_native_compile(self):
        expected = self.native_model.actuator_biasprm[0, :3]
        np.testing.assert_allclose(self.solver.mj_model.actuator_biasprm[0, :3], expected, atol=1e-6)
        np.testing.assert_allclose(self.solver.mjw_model.actuator_biasprm.numpy()[0, 0, :3], expected, atol=1e-6)

    def test_dampratio_custom_attribute_parsed(self):
        """dampratio should be encoded as unresolved biasprm[2] > 0."""
        biasprm = self.model.mujoco.actuator_biasprm.numpy()
        self.assertAlmostEqual(float(biasprm[0, 2]), 1.0, places=5, msg="dampratio=1 should be stored in biasprm[2]")
        self.assertAlmostEqual(float(biasprm[1, 2]), -5.0, places=5, msg="kv should store negative damping")
        self.assertAlmostEqual(float(biasprm[2, 2]), 0.0, places=5, msg="motor has zero biasprm[2]")

    def test_runtime_dampratio_update(self):
        """Updating biasprm[2] with unresolved dampratio should re-resolve via set_const_0."""
        biasprm = self.model.mujoco.actuator_biasprm.numpy()
        biasprm[0, 2] = 0.5
        self.model.mujoco.actuator_biasprm.assign(biasprm)
        self.solver.notify_model_changed(ModelFlags.ACTUATOR_PROPERTIES)

        resolved = self.solver.mjw_model.actuator_biasprm.numpy()[0, 0, 2]
        self.assertLess(resolved, 0.0, "resolved biasprm[2] should be negative damping")


class TestActuatorDampratioMultiWorld(unittest.TestCase):
    """Verify dampratio biasprm[2] is propagated to all worlds, not just world 0."""

    MJCF = TestActuatorDampratio.MJCF
    NUM_WORLDS = 4

    @classmethod
    def setUpClass(cls):
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()
        builder.replicate(robot_builder, cls.NUM_WORLDS)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_biasprm2_consistent_across_worlds(self):
        """biasprm[2] should be identical across all worlds for dampratio actuators."""
        bp = self.solver.mjw_model.actuator_biasprm.numpy()
        self.assertGreater(bp.shape[0], 1, "mjw_model should have multiple worlds")
        # Actuator 0 has dampratio=1, so biasprm[2] should be nonzero and identical
        world0_val = bp[0, 0, 2]
        self.assertNotAlmostEqual(float(world0_val), 0.0, places=3)
        for w in range(1, self.NUM_WORLDS):
            np.testing.assert_allclose(
                bp[w, 0, 2],
                world0_val,
                atol=1e-6,
                err_msg=f"World {w} biasprm[2] should match world 0",
            )


class TestActuatorDampratioMultiWorldRuntime(unittest.TestCase):
    """Verify per-world derived quantities after mass randomization."""

    MJCF = """<?xml version="1.0" ?>
    <mujoco>
        <worldbody>
            <body name="base">
                <joint name="j1" type="hinge" axis="0 0 1"/>
                <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            </body>
        </worldbody>
        <actuator>
            <position name="pos_dampratio" joint="j1" kp="100" dampratio="1.0"/>
        </actuator>
    </mujoco>
    """

    @classmethod
    def setUpClass(cls):
        robot_builder = newton.ModelBuilder()
        robot_builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot_builder, 2)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_per_world_acc0_and_dampratio(self):
        masses = self.model.body_mass.numpy()
        inertias = self.model.body_inertia.numpy()
        bodies_per_world = self.model.body_count // self.model.world_count

        for world in range(self.model.world_count):
            scale = 1.0 + world
            start = world * bodies_per_world
            end = start + bodies_per_world
            masses[start:end] *= scale
            inertias[start:end] *= scale

        self.model.body_mass.assign(masses)
        self.model.body_inertia.assign(inertias)

        self.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES | ModelFlags.ACTUATOR_PROPERTIES)

        acc0 = self.solver.mjw_model.actuator_acc0.numpy()
        biasprm = self.solver.mjw_model.actuator_biasprm.numpy()
        meaninertia = self.solver.mjw_model.stat.meaninertia.numpy()

        self.assertNotAlmostEqual(float(acc0[0, 0]), float(acc0[1, 0]), places=6)
        self.assertLess(float(biasprm[0, 0, 2]), 0.0)
        self.assertLess(float(biasprm[1, 0, 2]), 0.0)
        self.assertNotAlmostEqual(float(biasprm[0, 0, 2]), float(biasprm[1, 0, 2]), places=6)
        self.assertEqual(meaninertia.shape, (self.model.world_count,))
        np.testing.assert_allclose(meaninertia[1], 2.0 * meaninertia[0], rtol=1e-5)


class TestActuatorLengthRange(unittest.TestCase):
    """Verify that actuator length ranges keep MuJoCo's compiled values."""

    MUSCLE_PRM = "0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2 0"
    MUSCLE_MJCF = f"""<?xml version="1.0" ?>
    <mujoco>
        <worldbody>
            <body pos="0 0 1">
                <joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-1 1"/>
                <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1"/>
                <site name="s1" pos="0.3 0 0.05"/>
            </body>
            <site name="s0" pos="0 0 1.2"/>
        </worldbody>
        <tendon>
            <spatial name="t1"><site site="s0"/><site site="s1"/></spatial>
        </tendon>
        <actuator>
            <general name="joint_muscle" joint="j1" gaintype="muscle" biastype="muscle" dyntype="muscle"
                     gainprm="{MUSCLE_PRM}" biasprm="{MUSCLE_PRM}"/>
            <general name="tendon_muscle" tendon="t1" gaintype="muscle" biastype="muscle" dyntype="muscle"
                     gainprm="{MUSCLE_PRM}" biasprm="{MUSCLE_PRM}"/>
        </actuator>
    </mujoco>
    """

    MOTOR_MJCF = """<?xml version="1.0" ?>
    <mujoco>
        <worldbody>
            <body>
                <joint name="j1" type="hinge" axis="0 0 1" limited="true" range="-90 90"/>
                <geom type="capsule" size="0.05" fromto="0 0 0 0.5 0 0" mass="1.0"/>
            </body>
        </worldbody>
        <actuator>
            <motor name="motor1" joint="j1" gear="2"/>
            <general name="authored" joint="j1" lengthrange="-0.3 0.4"/>
        </actuator>
    </mujoco>
    """

    @staticmethod
    def _build(mjcf: str, world_count: int = 2) -> newton.Model:
        robot_builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(robot_builder)
        robot_builder.add_mjcf(mjcf, ctrl_direct=True)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.replicate(robot_builder, world_count)
        return builder.finalize()

    def test_muscle_lengthrange_matches_compiler(self):
        """Keep compiled joint and tendon muscle length ranges through model notifications."""
        import mujoco

        expected = mujoco.MjModel.from_xml_string(self.MUSCLE_MJCF).actuator_lengthrange
        self.assertTrue(np.all(expected[:, 0] < expected[:, 1]))
        for use_mujoco_cpu in (True, False):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                # The MuJoCo CPU backend supports a single world.
                model = self._build(self.MUSCLE_MJCF, world_count=1 if use_mujoco_cpu else 2)
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                for flags in (None, ModelFlags.ALL):
                    if flags is not None:
                        solver.notify_model_changed(flags)
                    np.testing.assert_allclose(solver.mj_model.actuator_lengthrange, expected, rtol=1e-6)
                    for world_lengthrange in solver.mjw_model.actuator_lengthrange.numpy():
                        np.testing.assert_allclose(world_lengthrange, expected, rtol=1e-5)

    def test_muscle_forces_match_cpu_backend(self):
        """Produce the same joint and tendon muscle forces on MuJoCo Warp as on MuJoCo CPU."""
        model = self._build(self.MUSCLE_MJCF, world_count=1)
        forces = {}
        for use_mujoco_cpu in (True, False):
            solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            control.mujoco.ctrl.fill_(1.0)
            for _ in range(5):
                solver.step(state_0, state_1, control, None, 0.002)
                state_0, state_1 = state_1, state_0
            forces[use_mujoco_cpu] = (
                solver.mj_data.actuator_force.copy() if use_mujoco_cpu else solver.mjw_data.actuator_force.numpy()[0]
            )
        self.assertTrue(np.all(np.abs(forces[True]) > 0.1))
        np.testing.assert_allclose(forces[False], forces[True], rtol=1e-4)

    def test_runtime_updates_preserve_lengthrange(self):
        """Forward authored ranges and keep compiled ranges after runtime gear changes."""
        expected = [[0.0, 0.0], [-0.3, 0.4]]
        for use_mujoco_cpu in (True, False):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                model = self._build(self.MOTOR_MJCF, world_count=1 if use_mujoco_cpu else 2)
                np.testing.assert_allclose(model.mujoco.actuator_lengthrange.numpy()[:2], expected)
                solver = SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                np.testing.assert_allclose(solver.mj_model.actuator_lengthrange, expected, rtol=1e-6)

                gear = model.mujoco.actuator_gear.numpy()
                gear[:, 0] = 3.0
                model.mujoco.actuator_gear.assign(gear)
                solver.notify_model_changed(
                    ModelFlags.ACTUATOR_PROPERTIES | ModelFlags.JOINT_DOF_PROPERTIES | ModelFlags.TENDON_PROPERTIES
                )

                np.testing.assert_allclose(solver.mjw_model.actuator_gear.numpy()[..., 0], 3.0)
                for world_lengthrange in solver.mjw_model.actuator_lengthrange.numpy():
                    np.testing.assert_allclose(world_lengthrange, expected, rtol=1e-6)

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_authored_lengthrange(self):
        """Forward an authored USD mjc:lengthRange to MuJoCo."""
        from pxr import Sdf, Vt

        def set_actuator_attrs(act):
            # A non-default gear keeps the actuator CTRL_DIRECT.
            act.CreateAttribute("mjc:gear", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            act.CreateAttribute("mjc:lengthRange:min", Sdf.ValueTypeNames.Double, True).Set(-0.3)
            act.CreateAttribute("mjc:lengthRange:max", Sdf.ValueTypeNames.Double, True).Set(0.4)

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(_create_actuator_test_stage(extra_actuator_attrs=set_actuator_attrs))
        model = builder.finalize()
        np.testing.assert_allclose(model.mujoco.actuator_lengthrange.numpy(), [[-0.3, 0.4]])
        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.actuator_lengthrange, [[-0.3, 0.4]], rtol=1e-6)


class TestActuatorInheritrange(unittest.TestCase):
    """Verify inheritrange on position actuators copies joint range to ctrlrange."""

    MJCF = """<?xml version="1.0" ?>
    <mujoco>
        <compiler angle="radian"/>
        <worldbody>
            <body name="base" pos="0 0 1">
                <freejoint/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="child" pos="0 0 0.5">
                    <joint name="j1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
                    <geom type="box" size="0.05 0.05 0.05" mass="0.5"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <position name="pos_inherit" joint="j1" kp="100" inheritrange="1"/>
        </actuator>
    </mujoco>"""

    @classmethod
    def setUpClass(cls):
        builder = newton.ModelBuilder()
        builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_ctrlrange_matches_joint_range(self):
        """ctrlrange should match joint range when inheritrange=1."""
        cr = self.solver.mj_model.actuator_ctrlrange[0]
        np.testing.assert_allclose(cr, [-1.57, 1.57], atol=1e-6)

    def test_ctrllimited_set(self):
        """ctrllimited should be 1 when inheritrange resolves a range."""
        self.assertEqual(self.solver.mj_model.actuator_ctrllimited[0], 1)


class TestActuatorInheritrangeFractional(unittest.TestCase):
    """Verify fractional inheritrange scales ctrlrange around the midpoint."""

    MJCF = """<?xml version="1.0" ?>
    <mujoco>
        <compiler angle="radian"/>
        <worldbody>
            <body name="base" pos="0 0 1">
                <freejoint/>
                <geom type="box" size="0.1 0.1 0.1" mass="1"/>
                <body name="child" pos="0 0 0.5">
                    <joint name="j1" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                    <geom type="box" size="0.05 0.05 0.05" mass="0.5"/>
                </body>
            </body>
        </worldbody>
        <actuator>
            <position name="pos_half" joint="j1" kp="100" inheritrange="0.5"/>
        </actuator>
    </mujoco>"""

    @classmethod
    def setUpClass(cls):
        builder = newton.ModelBuilder()
        builder.add_mjcf(cls.MJCF, ctrl_direct=True)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_ctrlrange_is_half(self):
        """ctrlrange should be half the joint range centered on the midpoint."""
        # Joint range: [-2.0, 2.0], mean=0.0, half-width=2.0
        # inheritrange=0.5 → radius = 2.0 * 0.5 = 1.0 → ctrlrange = [-1.0, 1.0]
        cr = self.solver.mj_model.actuator_ctrlrange[0]
        np.testing.assert_allclose(cr, [-1.0, 1.0], atol=1e-6)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdActuatorInheritrange(unittest.TestCase):
    """Verify inheritRange from USD scales ctrlrange around the joint-range midpoint.

    Per the MjcActuator schema, a positive inheritRange X resolves the actuator's
    ctrlrange to [midpoint - half_width*X, midpoint + half_width*X] where midpoint
    and half_width come from the transmission target's joint range.

    """

    JOINT_LO_DEG = -90.0
    JOINT_HI_DEG = 90.0

    CASES = (0.8, 1.0, 1.2)

    def _build_model(self, inherit_range_value, *, dof_ref=0.0, compiler_angle=None):
        from pxr import Sdf, Vt

        lo, hi = self.JOINT_LO_DEG, self.JOINT_HI_DEG

        def set_joint_limits(joint):
            joint.CreateLowerLimitAttr().Set(lo)
            joint.CreateUpperLimitAttr().Set(hi)
            joint.GetPrim().CreateAttribute("mjc:ref", Sdf.ValueTypeNames.Float, True).Set(dof_ref)

        def set_actuator_attrs(act):
            act.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token, True).Set("fixed")
            act.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token, True).Set("affine")
            act.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            act.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([0.0, -100.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            act.CreateAttribute("mjc:inheritRange", Sdf.ValueTypeNames.Double, True).Set(inherit_range_value)

        stage = _create_actuator_test_stage(
            extra_joint_attrs=set_joint_limits,
            extra_actuator_attrs=set_actuator_attrs,
        )
        if compiler_angle is not None:
            stage.GetPrimAtPath("/physicsScene").CreateAttribute(
                "mjc:compiler:angle", Sdf.ValueTypeNames.Token, True
            ).Set(compiler_angle)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        return builder.finalize()

    def test_inheritrange_ctrlrange(self):
        """Parsed ctrlrange should scale around midpoint for each inheritRange value."""
        lo_rad = self.JOINT_LO_DEG * np.pi / 180.0
        hi_rad = self.JOINT_HI_DEG * np.pi / 180.0
        mean = (lo_rad + hi_rad) / 2.0
        half_width = (hi_rad - lo_rad) / 2.0

        for ir in self.CASES:
            with self.subTest(inheritRange=ir):
                model = self._build_model(ir)

                radius = half_width * ir
                cr = model.mujoco.actuator_ctrlrange.numpy()[0]
                np.testing.assert_allclose(cr, [mean - radius, mean + radius], atol=1e-4)

                self.assertEqual(int(model.mujoco.actuator_ctrllimited.numpy()[0]), 1)

    def test_inheritrange_ctrlrange_with_joint_ref(self):
        """Normalize a default-degree ref and retain absolute control ranges."""
        dof_ref_degrees = 30.0
        dof_ref = np.deg2rad(dof_ref_degrees)
        model = self._build_model(1.0, dof_ref=dof_ref_degrees)

        lo_rad = self.JOINT_LO_DEG * np.pi / 180.0
        hi_rad = self.JOINT_HI_DEG * np.pi / 180.0
        expected = [lo_rad, hi_rad]
        np.testing.assert_allclose(model.mujoco.dof_ref.numpy()[-1], dof_ref, atol=1e-5)
        np.testing.assert_allclose(model.joint_limit_lower.numpy()[-1], lo_rad - dof_ref, atol=1e-5)
        np.testing.assert_allclose(model.joint_limit_upper.numpy()[-1], hi_rad - dof_ref, atol=1e-5)
        cr = model.mujoco.actuator_ctrlrange.numpy()[0]
        np.testing.assert_allclose(cr, expected, atol=1e-4)

        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.jnt_range[-1], expected, atol=1e-4)
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], expected, atol=1e-4)

    def test_converter_degree_ref_uses_absolute_limits(self):
        """Normalize converter-authored degree refs and absolute USD limits."""
        dof_ref_degrees = 30.0
        model = self._build_model(1.0, dof_ref=dof_ref_degrees, compiler_angle="degree")

        lo_rad = self.JOINT_LO_DEG * np.pi / 180.0
        hi_rad = self.JOINT_HI_DEG * np.pi / 180.0
        ref_rad = dof_ref_degrees * np.pi / 180.0
        np.testing.assert_allclose(model.mujoco.dof_ref.numpy()[-1], ref_rad, atol=1e-5)
        np.testing.assert_allclose(model.joint_limit_lower.numpy()[-1], lo_rad - ref_rad, atol=1e-5)
        np.testing.assert_allclose(model.joint_limit_upper.numpy()[-1], hi_rad - ref_rad, atol=1e-5)
        np.testing.assert_allclose(model.mujoco.actuator_ctrlrange.numpy()[0], [lo_rad, hi_rad], atol=1e-5)

        solver = SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_allclose(solver.mj_model.qpos0[-1], ref_rad, atol=1e-5)
        np.testing.assert_allclose(solver.mj_model.jnt_range[-1], [lo_rad, hi_rad], atol=1e-5)
        np.testing.assert_allclose(solver.mj_model.actuator_ctrlrange[0], [lo_rad, hi_rad], atol=1e-5)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdPositionShortcutBiasprmDampratio(unittest.TestCase):
    """Verify that positive biasprm[2] in a position-shortcut pattern is treated as dampratio."""

    KP = 200.0
    DAMPRATIO_PLACEHOLDER = 1.5

    @classmethod
    def setUpClass(cls):
        from pxr import Sdf, Vt

        kp = cls.KP
        dampratio_placeholder = cls.DAMPRATIO_PLACEHOLDER

        def set_actuator_attrs(act):
            act.CreateAttribute("mjc:gainType", Sdf.ValueTypeNames.Token, True).Set("fixed")
            act.CreateAttribute("mjc:biasType", Sdf.ValueTypeNames.Token, True).Set("affine")
            act.CreateAttribute("mjc:gainPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([kp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
            act.CreateAttribute("mjc:biasPrm", Sdf.ValueTypeNames.DoubleArray, True).Set(
                Vt.DoubleArray([0.0, -kp, dampratio_placeholder, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )

        stage = _create_actuator_test_stage(extra_actuator_attrs=set_actuator_attrs)
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_usd(stage)
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_position_shortcut_with_dampratio_from_biasprm(self):
        """Position shortcut should detect positive biasprm[2] as dampratio and resolve kd."""
        mj = self.solver.mj_model
        self.assertEqual(mj.actuator_biastype[0], 1)
        np.testing.assert_allclose(mj.actuator_gainprm[0, 0], self.KP, atol=1e-5)
        np.testing.assert_allclose(mj.actuator_biasprm[0, 1], -self.KP, atol=1e-5)

        bp2 = mj.actuator_biasprm[0, 2]
        self.assertLess(bp2, 0.0, "biasprm[2] should be negative (resolved kd)")


class TestMultiWorldQfrcActuatorCom(unittest.TestCase):
    """Multi-world qfrc_actuator must use each world's own body CoM.

    Regression test: convert_qfrc_actuator_from_mj_kernel indexed
    joint_type, joint_child, and joint_dof_dim with per-world jntid
    instead of the global index, so world 1 read world 0's CoM for
    the free-joint force transform.
    """

    MJCF = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body pos="0 0 1" quat="0.7071 0.7071 0 0">
          <freejoint name="root"/>
          <geom type="box" size="0.1 0.1 0.05" mass="1"/>
        </body>
      </worldbody>
      <actuator>
        <motor joint="root" gear="0 0 1 0 0 0"/>
      </actuator>
    </mujoco>
    """

    @classmethod
    def setUpClass(cls):
        def make_template(com):
            t = newton.ModelBuilder()
            t.add_mjcf(cls.MJCF, ctrl_direct=True)
            t.body_com[0] = com
            return t

        builder = newton.ModelBuilder()
        builder.add_world(make_template(wp.vec3(0.0, 0.0, 0.0)))
        builder.add_world(make_template(wp.vec3(0.1, -0.05, 0.03)))
        cls.model = builder.finalize()
        cls.solver = SolverMuJoCo(cls.model)

    def test_world1_uses_own_com(self):
        """World 1 angular qfrc must reflect its non-zero CoM, not world 0's."""
        state = self.model.state()
        observables = self.solver.observables({self.solver.ObservableFlags.QFRC_ACTUATOR})
        ctrl = self.model.control()
        ctrl.mujoco.ctrl = wp.array([10.0, 10.0], dtype=wp.float32)
        self.solver.step(state, state, ctrl, None, dt=0.01, observables=observables)

        qfrc = observables.qfrc_actuator.numpy()
        dofs_per_world = self.model.joint_dof_count // self.model.world_count

        # World 0 has CoM at origin — cross product is zero
        np.testing.assert_allclose(qfrc[3:6], 0.0, atol=0.01)
        # World 1: tau = -cross(R * com, f_lin) where R is 90deg around X,
        # com = (0.1, -0.05, 0.03), f_lin = (0, 0, 10)
        # => R*com = (0.1, -0.03, -0.05), tau = (0.3, 1.0, 0.0)
        # The bug would produce zeros here (reading world 0's zero CoM).
        angular_1 = qfrc[dofs_per_world + 3 : dofs_per_world + 6]
        np.testing.assert_allclose(angular_1, [0.3, 1.0, 0.0], atol=0.05)


class TestActuatorTypes(unittest.TestCase):
    """Actuator type enums and their MJCF round trip."""

    MJCF = """<mujoco>
        <worldbody>
            <body>
                <joint name="j" type="hinge" axis="0 0 1"/>
                <geom type="box" size=".1 .1 .1"/>
            </body>
        </worldbody>
        <actuator>
            <general name="a" joint="j" dyntype="{dyntype}" gaintype="{gaintype}" biastype="{biastype}" {extra}/>
        </actuator>
    </mujoco>"""

    def _mjcf(self, dyntype="none", gaintype="fixed", biastype="none", extra=""):
        return self.MJCF.format(dyntype=dyntype, gaintype=gaintype, biastype=biastype, extra=extra)

    def test_enums_match_mujoco(self):
        """Pin every actuator enum ordinal against the matching mujoco.mjt* member."""
        mujoco, _ = SolverMuJoCo.import_mujoco()
        for enum, mj_enum, prefix in (
            (_ActuatorBiasType, mujoco.mjtBias, "mjBIAS_"),
            (_ActuatorDynamicsType, mujoco.mjtDyn, "mjDYN_"),
            (_ActuatorGainType, mujoco.mjtGain, "mjGAIN_"),
        ):
            for member in enum:
                with self.subTest(enum=enum.__name__, member=member.name):
                    self.assertEqual(int(member), int(mj_enum.__members__[prefix + member.name.replace("_", "")]))

    def test_types_match_native(self):
        """Compile actuator types through Newton and check they match native MuJoCo on the same MJCF."""
        mujoco, _ = SolverMuJoCo.import_mujoco()
        for dyntype, gaintype, biastype, extra in (
            ("none", "fixed", "none", ""),
            ("none", "fixed", "affine", ""),
            ("integrator", "fixed", "none", ""),
            ("filter", "fixed", "none", ""),
            ("filterexact", "fixed", "none", ""),
            ("none", "affine", "affine", ""),
            ("user", "user", "user", 'actearly="true"'),
        ):
            with self.subTest(dyntype=dyntype, gaintype=gaintype, biastype=biastype):
                mjcf = self._mjcf(dyntype=dyntype, gaintype=gaintype, biastype=biastype, extra=extra)
                builder = newton.ModelBuilder()
                builder.add_mjcf(mjcf)
                mj_model = SolverMuJoCo(builder.finalize()).mj_model
                native = mujoco.MjModel.from_xml_string(mjcf)
                assert_np_equal(mj_model.actuator_dyntype, native.actuator_dyntype)
                assert_np_equal(mj_model.actuator_gaintype, native.actuator_gaintype)
                assert_np_equal(mj_model.actuator_biastype, native.actuator_biastype)

    def test_unsupported_type_warns(self):
        """Warn on an unsupported gaintype, naming the attribute, the value and the MJCF actuator."""
        for gaintype in ("dcmotor", "so3", "pid", "bogus"):
            with self.subTest(gaintype=gaintype):
                builder = newton.ModelBuilder()
                with self.assertWarnsRegex(RuntimeWarning, rf"gaintype '{gaintype}' on actuator 'a'"):
                    builder.add_mjcf(self._mjcf(gaintype=gaintype))
