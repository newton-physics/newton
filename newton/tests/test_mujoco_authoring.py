# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for programmatic MuJoCo-specific model authoring helpers."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import mujoco


def _add_revolute(builder: newton.ModelBuilder, label: str) -> tuple[int, int]:
    """Add a single-body revolute articulation and return its body and joint."""
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        label=f"{label}_body",
    )
    joint = builder.add_joint_revolute(parent=-1, child=body, axis=wp.vec3(0.0, 0.0, 1.0), label=label)
    builder.add_articulation([joint], label=f"{label}_articulation")
    return body, joint


class TestMuJoCoActuatorAuthoring(unittest.TestCase):
    """Tests for public MuJoCo actuator authoring helpers."""

    def test_add_actuator_dcmotor(self):
        """Create a DC-motor row from high-level parameters."""
        builder = newton.ModelBuilder()
        _, joint = _add_revolute(builder, "hinge")

        actuator = mujoco.add_actuator_dcmotor(
            builder,
            target=mujoco.ActuatorTarget.joint(joint),
            motorconst=(0.05, 0.06),
            resistance=2.0,
            nominal=(24.0, 0.2, 100.0),
            saturation=(2.0, 4.0, 7.0),
            inductance=(0.01, 20.0),
            cogging=(0.1, 6.0, 0.2),
            controller=(5.0, 1.0, 0.2, 10.0, 2.0, 3.0),
            thermal=(0.004, 10.0, 30.0, 0.001, 0.4, 90.0),
            lugre=(0.3, 0.4, 0.5, 12.0, 0.02),
            input_mode="position",
            gear=(3.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            damping=0.7,
            armature=0.02,
        )
        model = builder.finalize()

        self.assertEqual(actuator, 0)
        self.assertEqual(model.custom_frequency_counts["mujoco:actuator"], 1)
        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy(), [[0, -1]])
        np.testing.assert_array_equal(
            model.mujoco.actuator_trntype.numpy(),
            [int(newton.solvers.SolverMuJoCo.TrnType.JOINT)],
        )
        np.testing.assert_array_equal(
            model.mujoco.ctrl_type.numpy(),
            [int(newton.solvers.SolverMuJoCo.CtrlType.DCMOTOR)],
        )
        np.testing.assert_allclose(model.mujoco.actuator_dcmotor_motorconst.numpy(), [[0.05, 0.06]])
        np.testing.assert_allclose(model.mujoco.actuator_dcmotor_lugre.numpy(), [[0.3, 0.4, 0.5, 12.0, 0.02]])
        np.testing.assert_array_equal(model.mujoco.actuator_dcmotor_input.numpy(), [1])
        np.testing.assert_array_equal(model.mujoco.actuator_ctrlspec.numpy(), [1])

        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        self.assertEqual(solver.mj_model.nu, 1)
        native_mujoco = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        np.testing.assert_array_equal(
            solver.mj_model.actuator_dyntype,
            [native_mujoco.mjtDyn.mjDYN_DCMOTOR],
        )

    def test_add_actuator_shortcuts_and_ranges(self):
        """Normalize shortcut parameters and authored range metadata."""
        builder = newton.ModelBuilder()
        _, joint = _add_revolute(builder, "hinge")
        target = mujoco.ActuatorTarget.joint(joint)

        mujoco.add_actuator_motor(builder, target=target, ctrlrange=(-2.0, 2.0), ctrllimited=True)
        mujoco.add_actuator_position(builder, target=target, kp=10.0, kv=2.0)
        mujoco.add_actuator_velocity(builder, target=target, kv=3.0)
        mujoco.add_actuator_general(
            builder,
            target=target,
            dyntype="filterexact",
            dynprm=(0.05,),
            gainprm=(4.0,),
            biasprm=(0.0, -4.0, -0.5),
        )
        model = builder.finalize()

        self.assertEqual(model.custom_frequency_counts["mujoco:actuator"], 4)
        np.testing.assert_array_equal(model.mujoco.actuator_has_ctrlrange.numpy(), [1, 0, 0, 0])
        np.testing.assert_array_equal(model.mujoco.actuator_ctrllimited.numpy(), [1, 2, 2, 2])
        np.testing.assert_allclose(model.mujoco.actuator_gainprm.numpy()[:, 0], [1.0, 10.0, 3.0, 4.0])
        np.testing.assert_allclose(model.mujoco.actuator_biasprm.numpy()[1, :3], [0.0, -10.0, -2.0])
        np.testing.assert_allclose(model.mujoco.actuator_biasprm.numpy()[2, :3], [0.0, 0.0, -3.0])

    def test_dcmotor_input_modes_match_mujoco(self):
        """Preserve named and numeric input modes with MuJoCo's input bitmask."""
        native_mujoco = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        for name, ctrlspec, controller in (
            ("voltage", native_mujoco.mjtCtrlInput.mjINPUT_VOLTAGE, (0.0,) * 6),
            ("position", native_mujoco.mjtCtrlInput.mjINPUT_POS, (5.0, 1.0, 0.2, 10.0, 2.0, 3.0)),
            ("velocity", native_mujoco.mjtCtrlInput.mjINPUT_VEL, (0.0, 0.0, 0.2, 10.0, 2.0, 3.0)),
        ):
            for input_mode in (name, int(ctrlspec)):
                with self.subTest(input_mode=input_mode):
                    builder = newton.ModelBuilder()
                    _, joint = _add_revolute(builder, "hinge")
                    mujoco.add_actuator_dcmotor(
                        builder,
                        target=mujoco.ActuatorTarget.joint(joint),
                        motorconst=(0.05, 0.06),
                        resistance=2.0,
                        controller=controller,
                        input_mode=input_mode,
                    )
                    solver = newton.solvers.SolverMuJoCo(builder.finalize(), use_mujoco_cpu=True, disable_contacts=True)
                    np.testing.assert_array_equal(solver.mj_model.actuator_ctrlspec, [ctrlspec])

    def test_add_actuator_dcmotor_matches_native_mujoco(self):
        """Compile helper-authored DC motors like native MJCF and keep runtime updates."""
        native_mujoco = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        native = native_mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody>
                <body>
                  <joint name="hinge" type="hinge" axis="0 0 1"/>
                  <geom type="sphere" size="0.1"/>
                </body>
              </worldbody>
              <actuator>
                <dcmotor joint="hinge" motorconst="0.05 0.06" resistance="2" input="pos"
                         controller="5 1 0.2 10 2 3" ctrlrange="-1 1" saturation="2 4 7"/>
              </actuator>
            </mujoco>
            """
        )

        for use_mujoco_cpu in (True, False):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                builder = newton.ModelBuilder()
                _, joint = _add_revolute(builder, "hinge")
                mujoco.add_actuator_dcmotor(
                    builder,
                    target=mujoco.ActuatorTarget.joint(joint),
                    motorconst=(0.05, 0.06),
                    resistance=2.0,
                    controller=(5.0, 1.0, 0.2, 10.0, 2.0, 3.0),
                    input_mode="position",
                    ctrlrange=(-1.0, 1.0),
                    saturation=(2.0, 4.0, 7.0),
                )
                model = builder.finalize()
                solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                for name in (
                    "actuator_gainprm",
                    "actuator_biasprm",
                    "actuator_dynprm",
                    "actuator_ctrlspec",
                    "actuator_forcerange",
                    "actuator_forcelimited",
                    "actuator_actlimited",
                ):
                    np.testing.assert_allclose(getattr(solver.mj_model, name), getattr(native, name), err_msg=name)

                model.mujoco.actuator_ctrlrange.assign([[-0.5, 0.5]])
                solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
                if use_mujoco_cpu:
                    ctrlrange = solver.mj_model.actuator_ctrlrange[0]
                    gainprm = solver.mj_model.actuator_gainprm[0]
                else:
                    ctrlrange = solver.mjw_model.actuator_ctrlrange.numpy()[0, 0]
                    gainprm = solver.mjw_model.actuator_gainprm.numpy()[0, 0]
                    # MuJoCo-Warp reads the legacy DC-motor input slot (1 = position).
                    self.assertEqual(gainprm[8], 1.0)
                np.testing.assert_allclose(ctrlrange, [-0.5, 0.5])
                np.testing.assert_allclose(gainprm[:8], native.actuator_gainprm[0, :8], rtol=1e-6)

    def test_reject_unsupported_dcmotor_input_mode(self):
        """Reject DC-motor input signatures that SolverMuJoCo cannot drive."""
        builder = newton.ModelBuilder()
        _, joint = _add_revolute(builder, "hinge")

        for input_mode in ("feedforward", 0, 3):
            with self.subTest(input_mode=input_mode), self.assertRaises(ValueError):
                mujoco.add_actuator_dcmotor(
                    builder,
                    target=mujoco.ActuatorTarget.joint(joint),
                    motorconst=(0.05, 0.06),
                    resistance=2.0,
                    input_mode=input_mode,
                )

    def test_reject_multidof_joint_without_dof(self):
        """Require an explicit local DOF for multi-DOF joint targets."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_ball(parent=-1, child=body)
        builder.add_articulation([joint])

        with self.assertRaisesRegex(ValueError, "dof"):
            mujoco.add_actuator_motor(builder, target=mujoco.ActuatorTarget.joint(joint))

    def test_multidof_actuator_drives_selected_axis(self):
        """Local and absolute DOF targets select the corresponding ball/free/D6 axis."""
        native_mujoco = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        for joint_type, dof_count in (("ball", 3), ("free", 6), ("d6", 2)):
            for absolute in (False, True):
                with self.subTest(joint_type=joint_type, absolute=absolute):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    _add_revolute(builder, "prefix")
                    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                    kwargs = (
                        {
                            "angular_axes": [
                                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y)
                            ]
                        }
                        if joint_type == "d6"
                        else {}
                    )
                    joint = getattr(builder, f"add_joint_{joint_type}")(parent=-1, child=body, **kwargs)
                    builder.add_articulation([joint])
                    for dof in range(dof_count):
                        target = (
                            mujoco.ActuatorTarget.joint_dof(builder.joint_qd_start[joint] + dof)
                            if absolute
                            else mujoco.ActuatorTarget.joint(joint, dof=dof)
                        )
                        mujoco.add_actuator_motor(builder, target, gear=(2.0,))
                    # An explicit vector describes a native MuJoCo transmission direction.
                    mujoco.add_actuator_motor(builder, target, gear=(0.5, 0.75, 1.0))
                    solver = newton.solvers.SolverMuJoCo(builder.finalize(), use_mujoco_cpu=True, disable_contacts=True)
                    data = native_mujoco.MjData(solver.mj_model)
                    for dof in range(dof_count):
                        data.ctrl[:] = 0.0
                        data.ctrl[dof] = 1.0
                        native_mujoco.mj_forward(solver.mj_model, data)
                        expected = np.zeros(1 + dof_count)
                        expected[1 + dof] = 2.0
                        np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-7)
                    data.ctrl[:] = 0.0
                    data.ctrl[-1] = 1.0
                    native_mujoco.mj_forward(solver.mj_model, data)
                    expected = np.zeros(1 + dof_count)
                    if joint_type == "d6":
                        expected[-1] = 0.5
                    else:
                        expected[1:4] = [0.5, 0.75, 1.0]
                    np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-7)

    def test_reject_actuator_targets_across_worlds(self):
        """Reject primary and secondary transmission targets from another world."""
        builder = newton.ModelBuilder()
        global_site = builder.add_site(-1)
        builder.begin_world()
        body0, joint0 = _add_revolute(builder, "world0")
        site0 = builder.add_site(body0)
        tendon0 = mujoco.add_tendon_fixed(builder, [(joint0, 1.0)])
        builder.end_world()
        builder.begin_world()
        body1, joint1 = _add_revolute(builder, "world1")
        site1 = builder.add_site(body1)
        targets = (
            mujoco.ActuatorTarget.joint(joint0),
            mujoco.ActuatorTarget.joint_dof(builder.joint_qd_start[joint0]),
            mujoco.ActuatorTarget.body(body0),
            mujoco.ActuatorTarget.tendon(tendon0),
            mujoco.ActuatorTarget.site(site0),
            mujoco.ActuatorTarget.site(site1, refsite=site0),
            mujoco.ActuatorTarget.slider_crank(site0, site1),
            mujoco.ActuatorTarget.slider_crank(site1, site0),
        )
        for target in targets:
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "belongs to world 0"):
                mujoco.add_actuator_motor(builder, target, cranklength=0.1)
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:actuator", 0), 0)
        mujoco.add_actuator_motor(builder, mujoco.ActuatorTarget.joint(joint1))
        mujoco.add_actuator_motor(builder, mujoco.ActuatorTarget.site(site1, refsite=global_site))
        builder.end_world()

    def test_reject_dcmotor_compiler_managed_limits(self):
        """DC-motor shortcuts must not accept limits that compilation discards."""
        builder = newton.ModelBuilder()
        _, joint = _add_revolute(builder, "hinge")
        for keyword, value in (
            ("forcerange", (-1.0, 1.0)),
            ("forcelimited", True),
            ("actrange", (-0.5, 0.5)),
            ("actlimited", True),
        ):
            with self.subTest(keyword=keyword), self.assertRaisesRegex(TypeError, keyword):
                mujoco.add_actuator_dcmotor(
                    builder,
                    mujoco.ActuatorTarget.joint(joint),
                    motorconst=(0.05, 0.06),
                    resistance=2.0,
                    **{keyword: value},
                )

    def test_actuator_target_kinds(self):
        """Tag each actuator target factory with its enum kind."""
        kind = mujoco.ActuatorTarget.Kind
        self.assertEqual(mujoco.ActuatorTarget.joint(2, dof=1), mujoco.ActuatorTarget(kind.JOINT, 2, dof=1))
        self.assertEqual(mujoco.ActuatorTarget.joint_dof(3).kind, kind.JOINT_DOF)
        self.assertEqual(mujoco.ActuatorTarget.tendon(0).kind, kind.TENDON)
        self.assertEqual(mujoco.ActuatorTarget.site(4, refsite=5), mujoco.ActuatorTarget(kind.SITE, 4, 5))
        self.assertEqual(mujoco.ActuatorTarget.body(1).kind, kind.BODY)
        self.assertEqual(mujoco.ActuatorTarget.slider_crank(4, 5), mujoco.ActuatorTarget(kind.SLIDER_CRANK, 4, 5))

    def test_reject_malformed_actuator_ranges(self):
        """Require two-value ranges instead of silently padding them."""
        builder = newton.ModelBuilder()
        _, joint = _add_revolute(builder, "hinge")

        with self.assertRaisesRegex(ValueError, "ctrlrange requires exactly 2 values"):
            mujoco.add_actuator_motor(builder, target=mujoco.ActuatorTarget.joint(joint), ctrlrange=(1.0,))

    def test_remap_actuator_target_when_merging_builders(self):
        """Offset heterogeneous actuator targets during builder composition."""
        blueprint = newton.ModelBuilder()
        blueprint_body, blueprint_joint = _add_revolute(blueprint, "blueprint_hinge")
        blueprint_site0 = blueprint.add_site(blueprint_body, label="blueprint_site0")
        blueprint_site1 = blueprint.add_site(blueprint_body, label="blueprint_site1")
        blueprint_tendon = mujoco.add_tendon_fixed(blueprint, joints=[(blueprint_joint, 1.0)])
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.joint(blueprint_joint))
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.tendon(blueprint_tendon))
        mujoco.add_actuator_motor(
            blueprint, target=mujoco.ActuatorTarget.site(blueprint_site0, refsite=blueprint_site1)
        )
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.body(blueprint_body))
        mujoco.add_actuator_motor(
            blueprint,
            target=mujoco.ActuatorTarget.slider_crank(blueprint_site0, blueprint_site1),
            cranklength=0.1,
        )

        scene = newton.ModelBuilder()
        scene_body, scene_joint = _add_revolute(scene, "scene_hinge")
        scene.add_site(scene_body, label="scene_site")
        mujoco.add_tendon_fixed(scene, joints=[(scene_joint, 1.0)])
        scene.add_builder(blueprint)
        model = scene.finalize()

        np.testing.assert_array_equal(
            model.mujoco.actuator_trnid.numpy(),
            [[1, -1], [1, -1], [1, 2], [1, -1], [1, 2]],
        )

    def test_remap_mjcf_actuator_targets_when_merging_builders(self):
        """Offset imported ctrl-direct actuator targets for each merged builder copy."""
        mjcf = """
        <mujoco>
          <worldbody>
            <body name="link">
              <joint name="hinge" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.1"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="hinge"/>
          </actuator>
        </mujoco>
        """
        robot = newton.ModelBuilder()
        robot.add_mjcf(mjcf, ctrl_direct=True)

        scene = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
        scene.add_builder(robot)
        scene.add_builder(robot)
        model = scene.finalize()

        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy()[:, 0], [0, 1])
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, disable_contacts=True)
        np.testing.assert_array_equal(solver.mj_model.actuator_trnid[:, 0], [0, 1])


class TestMuJoCoEntityAuthoring(unittest.TestCase):
    """Tests for non-actuator MuJoCo authoring helpers."""

    def test_add_contact_pair(self):
        """Create an explicit MuJoCo contact-pair row."""
        builder = newton.ModelBuilder()
        shape0 = builder.add_shape_box(body=-1, hx=0.1, hy=0.1, hz=0.1)
        shape1 = builder.add_shape_sphere(body=-1, radius=0.1)

        pair = mujoco.add_contact_pair(
            builder,
            shape0,
            shape1,
            condim=4,
            friction=(0.8, 0.7, 0.01, 0.02, 0.03),
            margin=0.01,
        )
        model = builder.finalize()

        self.assertEqual(pair, 0)
        np.testing.assert_array_equal(model.mujoco.pair_geom1.numpy(), [shape0])
        np.testing.assert_array_equal(model.mujoco.pair_geom2.numpy(), [shape1])
        np.testing.assert_array_equal(model.mujoco.pair_condim.numpy(), [4])
        np.testing.assert_allclose(model.mujoco.pair_friction.numpy(), [[0.8, 0.7, 0.01, 0.02, 0.03]])

    def test_reject_contact_pair_across_worlds(self):
        """Reject contact pairs whose shapes belong to another world."""
        builder = newton.ModelBuilder()
        builder.begin_world()
        shape0 = builder.add_shape_sphere(body=-1, radius=0.1)
        builder.end_world()
        builder.begin_world()
        shape1 = builder.add_shape_sphere(body=-1, radius=0.1)

        with self.assertRaisesRegex(ValueError, "belongs to world 0"):
            mujoco.add_contact_pair(builder, shape0, shape1)
        builder.end_world()

    def test_add_fixed_tendon_and_tendon_actuator(self):
        """Create a fixed tendon and target it with an actuator."""
        builder = newton.ModelBuilder()
        _, joint0 = _add_revolute(builder, "hinge0")
        _, joint1 = _add_revolute(builder, "hinge1")

        tendon = mujoco.add_tendon_fixed(
            builder,
            joints=[(joint0, 1.0), (joint1, -0.5)],
            label="coupling",
            stiffness=20.0,
        )
        mujoco.add_actuator_motor(builder, target=mujoco.ActuatorTarget.tendon(tendon), gear=(2.0,))
        model = builder.finalize()

        self.assertEqual(tendon, 0)
        np.testing.assert_array_equal(model.mujoco.tendon_joint_adr.numpy(), [0])
        np.testing.assert_array_equal(model.mujoco.tendon_joint_num.numpy(), [2])
        np.testing.assert_array_equal(model.mujoco.tendon_joint.numpy(), [joint0, joint1])
        np.testing.assert_allclose(model.mujoco.tendon_coef.numpy(), [1.0, -0.5])
        np.testing.assert_array_equal(
            model.mujoco.actuator_trntype.numpy(),
            [int(newton.solvers.SolverMuJoCo.TrnType.TENDON)],
        )
        np.testing.assert_array_equal(model.mujoco.actuator_trnid.numpy(), [[tendon, -1]])

    def test_add_spatial_tendon(self):
        """Create a spatial tendon while hiding its child-row addresses."""
        builder = newton.ModelBuilder()
        body, _ = _add_revolute(builder, "hinge")
        site0 = builder.add_site(body, label="site0")
        geom = builder.add_shape_sphere(body, radius=0.1, label="geom")
        site1 = builder.add_site(body, label="site1")

        tendon = mujoco.add_tendon_spatial(
            builder,
            path=[
                mujoco.TendonWrapSite(site0),
                mujoco.TendonWrapGeom(geom, sidesite=site1),
                mujoco.TendonWrapPulley(2.0),
            ],
            label="spatial",
        )
        model = builder.finalize()

        self.assertEqual(tendon, 0)
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_adr.numpy(), [0])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_num.numpy(), [3])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_type.numpy(), [0, 1, 2])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_shape.numpy(), [site0, geom, -1])
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_sidesite.numpy(), [-1, site1, -1])
        np.testing.assert_allclose(model.mujoco.tendon_wrap_prm.numpy(), [0.0, 0.0, 2.0])

    def test_reject_fixed_tendon_joint_without_mujoco_mapping(self):
        """Reject tendon joints that the solver would silently drop on export."""
        for joint_type in ("d6", "fixed"):
            with self.subTest(joint_type=joint_type):
                builder = newton.ModelBuilder()
                body = builder.add_link(mass=1.0)
                kwargs = (
                    {"linear_axes": [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0))]}
                    if joint_type == "d6"
                    else {}
                )
                joint = getattr(builder, f"add_joint_{joint_type}")(parent=-1, child=body, **kwargs)
                builder.add_articulation([joint])
                with self.assertRaisesRegex(ValueError, "fixed tendon"):
                    mujoco.add_tendon_fixed(builder, [(joint, 1.0)])
                self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon", 0), 0)
                self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon_joint", 0), 0)

    def test_weld_default_preserves_initial_relative_pose(self):
        """An omitted weld pose preserves the offset; explicit identity removes it."""
        for use_mujoco_cpu in (True, False):
            for relpose, expected_offset in ((None, 0.5), (wp.transform_identity(), 0.0)):
                with self.subTest(use_mujoco_cpu=use_mujoco_cpu, relpose=relpose):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    bodies = [
                        builder.add_body(
                            xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                            mass=1.0,
                            inertia=wp.mat33(np.eye(3)),
                        )
                        for x in (0.0, 0.5)
                    ]
                    mujoco.add_equality_weld(builder, *bodies, relpose=relpose)
                    model = builder.finalize()
                    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                    solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
                    state0, state1 = model.state(), model.state()
                    control = model.control()
                    for _ in range(200):
                        solver.step(state0, state1, control, None, 0.002)
                        state0, state1 = state1, state0
                    positions = state0.body_q.numpy()[:, :3]
                    np.testing.assert_allclose(positions[1] - positions[0], [expected_offset, 0.0, 0.0], atol=1e-5)

    def test_add_equality_helpers(self):
        """Create typed MuJoCo equality-constraint rows."""
        builder = newton.ModelBuilder()
        body0, joint0 = _add_revolute(builder, "hinge0")
        body1, joint1 = _add_revolute(builder, "hinge1")

        connect = mujoco.add_equality_connect(builder, body0, body1, anchor=(0.1, 0.2, 0.3))
        weld = mujoco.add_equality_weld(builder, body0, body1, torquescale=2.0)
        joint = mujoco.add_equality_joint(builder, joint0, joint1, polycoef=(1.0, 2.0))
        model = builder.finalize()

        self.assertEqual((connect, weld, joint), (0, 1, 2))
        np.testing.assert_array_equal(
            model.mujoco.equality_constraint_type.numpy(),
            [
                int(newton.solvers.SolverMuJoCo.EqType.CONNECT),
                int(newton.solvers.SolverMuJoCo.EqType.WELD),
                int(newton.solvers.SolverMuJoCo.EqType.JOINT),
            ],
        )
        np.testing.assert_allclose(model.mujoco.equality_constraint_anchor.numpy()[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(model.mujoco.equality_constraint_polycoef.numpy()[2], [1.0, 2.0, 0.0, 0.0, 0.0])

    def test_reject_self_referencing_equalities(self):
        """Reject equality constraints whose two operands are identical."""
        builder = newton.ModelBuilder()
        body, joint = _add_revolute(builder, "hinge")

        with self.assertRaisesRegex(ValueError, "two distinct bodies"):
            mujoco.add_equality_connect(builder, body, body)
        with self.assertRaisesRegex(ValueError, "two distinct bodies"):
            mujoco.add_equality_weld(builder, body, body)
        with self.assertRaisesRegex(ValueError, "two distinct joints"):
            mujoco.add_equality_joint(builder, joint, joint)


if __name__ == "__main__":
    unittest.main()
