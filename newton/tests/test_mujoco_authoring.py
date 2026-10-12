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

    def test_reject_multidof_joint_without_dof(self):
        """Require an explicit local DOF for multi-DOF joint targets."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_d6(
            parent=-1,
            child=body,
            angular_axes=[newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y)],
        )
        builder.add_articulation([joint])

        with self.assertRaisesRegex(ValueError, "dof"):
            mujoco.add_actuator_motor(builder, target=mujoco.ActuatorTarget.joint(joint))

    def test_d6_actuator_drives_selected_axis(self):
        """Drive each D6 axis through both local and absolute DOF targets."""
        native = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        for absolute in (False, True):
            with self.subTest(absolute=absolute):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                _add_revolute(builder, "prefix")
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    angular_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y)
                    ],
                )
                builder.add_articulation([joint])
                for dof in range(2):
                    target = (
                        mujoco.ActuatorTarget.joint_dof(builder.joint_qd_start[joint] + dof)
                        if absolute
                        else mujoco.ActuatorTarget.joint(joint, dof=dof)
                    )
                    mujoco.add_actuator_motor(builder, target, gear=(2.0,))
                solver = newton.solvers.SolverMuJoCo(builder.finalize(), use_mujoco_cpu=True, disable_contacts=True)
                data = native.MjData(solver.mj_model)
                for dof in range(2):
                    data.ctrl[:] = 0.0
                    data.ctrl[dof] = 1.0
                    native.mj_forward(solver.mj_model, data)
                    expected = np.zeros(3)
                    expected[1 + dof] = 2.0
                    np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-7)

    def test_ball_free_targets_require_native_gear(self):
        """Reject ambiguous Newton DOF selection on native ball/free transmissions."""
        rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 2)
        for joint_type, gear in (
            ("ball", (1.0, 0.0, 0.0)),
            ("free", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
            ("free", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ):
            with self.subTest(joint_type=joint_type, gear=gear):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_link(
                    mass=1.0,
                    inertia=wp.mat33(np.eye(3)),
                    com=wp.vec3(0.1, 0.2, 0.0) if joint_type == "free" else wp.vec3(),
                )
                joint = getattr(builder, f"add_joint_{joint_type}")(parent=-1, child=body)
                builder.add_articulation([joint])
                q_start = builder.joint_q_start[joint] + (3 if joint_type == "free" else 0)
                builder.joint_q[q_start : q_start + 4] = list(rotation)
                for target in (mujoco.ActuatorTarget.joint(joint, dof=0), mujoco.ActuatorTarget.joint_dof(0)):
                    with self.subTest(target=target), self.assertRaisesRegex(ValueError, "native.*gear"):
                        mujoco.add_actuator_motor(builder, target, gear=gear)
                with self.assertRaisesRegex(ValueError, "gear"):
                    mujoco.add_actuator_motor(builder, mujoco.ActuatorTarget.joint(joint))
                self.assertEqual(builder._custom_frequency_counts.get("mujoco:actuator", 0), 0)

                mujoco.add_actuator_motor(builder, mujoco.ActuatorTarget.joint(joint), gear=gear, ctrl=1.0)
                model = builder.finalize()
                velocities = []
                for use_mujoco_cpu in (True, False):
                    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                    np.testing.assert_allclose(solver.mj_model.actuator_gear[0], list(gear) + [0.0] * (6 - len(gear)))
                    state0, state1 = model.state(), model.state()
                    newton.eval_fk(model, model.joint_q, model.joint_qd, state0)
                    solver.step(state0, state1, model.control(), None, 0.001)
                    velocities.append(state1.joint_qd.numpy())
                np.testing.assert_allclose(velocities[0], velocities[1], atol=1e-6)
                if joint_type == "free" and gear[0] != 0.0:
                    # World-frame X force at the body origin also produces COM torque.
                    self.assertGreater(velocities[0][0], 0.0)
                    self.assertGreater(velocities[0][5], 0.0)
                else:
                    # Native child-frame X torque acts about parent-frame Y after rotation.
                    angular_start = 3 if joint_type == "free" else 0
                    self.assertGreater(velocities[0][angular_start + 1], 0.0)
                    self.assertAlmostEqual(velocities[0][angular_start], 0.0, places=6)

    def test_zero_feedback_gains(self):
        """Allow a damping-only position actuator and a disabled velocity actuator."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        _, joint = _add_revolute(builder, "hinge")
        target = mujoco.ActuatorTarget.joint(joint)
        mujoco.add_actuator_position(builder, target, kp=0.0, kv=2.0)
        mujoco.add_actuator_velocity(builder, target, kv=0.0)
        mujoco.add_actuator_position(builder, target, kp=0.0, dampratio=1.0)
        solver = newton.solvers.SolverMuJoCo(builder.finalize(), use_mujoco_cpu=True, disable_contacts=True)
        native = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        data = native.MjData(solver.mj_model)
        data.qvel[:] = 0.5
        data.ctrl[:] = 1.0
        native.mj_forward(solver.mj_model, data)
        np.testing.assert_allclose(data.actuator_force, [-1.0, 0.0, 0.0])
        for gains in ({"kp": -1.0}, {"kv": -1.0}, {"dampratio": -1.0}):
            with self.subTest(gains=gains), self.assertRaisesRegex(ValueError, "nonnegative"):
                mujoco.add_actuator_position(builder, target, **gains)

    def test_import_actuator_damping_remains_out_of_scope(self):
        """Preserve upstream import behavior for scalar and vector actuator damping."""
        for damping in ("3", "1 0 0"):
            with self.subTest(damping=damping):
                builder = newton.ModelBuilder()
                builder.add_mjcf(
                    f"""<mujoco><worldbody><body><joint name="hinge"/>
                  <geom type="sphere" size="0.1"/></body></worldbody>
                  <actuator><motor joint="hinge" damping="{damping}" armature="0.5"/></actuator></mujoco>""",
                    ctrl_direct=True,
                )
                solver = newton.solvers.SolverMuJoCo(builder.finalize(), use_mujoco_cpu=True, disable_contacts=True)
                np.testing.assert_array_equal(solver.mj_model.actuator_damping, 0.0)
                np.testing.assert_array_equal(solver.mj_model.actuator_armature, 0.0)

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
        for actuator in (
            '<motor joint="hinge"/>',
            '<muscle joint="hinge" force="1" lengthrange="0.5 1.5"/>',
        ):
            with self.subTest(actuator=actuator):
                mjcf = f"""
                <mujoco>
                  <worldbody>
                    <body name="link">
                      <joint name="hinge" type="hinge" axis="0 0 1"/>
                      <geom type="sphere" size="0.1"/>
                    </body>
                  </worldbody>
                  <actuator>
                    {actuator}
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
        """Compile complete spatial tendon branches with a wrapping geom and pulley."""
        builder = newton.ModelBuilder()
        body, _ = _add_revolute(builder, "hinge")
        site0 = builder.add_site(body, xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()))
        site1 = builder.add_site(body, xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()))
        sidesite = builder.add_site(body, xform=wp.transform(wp.vec3(0.0, 0.2, 0.0), wp.quat_identity()))
        geom = builder.add_shape_sphere(body, radius=0.1, label="wrap_geom")
        tendon = mujoco.add_tendon_spatial(
            builder,
            path=[
                mujoco.TendonWrapSite(site0),
                mujoco.TendonWrapGeom(geom, sidesite=sidesite),
                mujoco.TendonWrapSite(site1),
                mujoco.TendonWrapPulley(2.0),
                mujoco.TendonWrapSite(site0),
                mujoco.TendonWrapSite(site1),
            ],
            label="spatial",
        )
        model = builder.finalize()
        self.assertEqual(tendon, 0)
        np.testing.assert_array_equal(model.mujoco.tendon_wrap_shape.numpy(), [site0, geom, site1, -1, site0, site1])
        native = newton.solvers.SolverMuJoCo.import_mujoco()[0]
        for use_mujoco_cpu in (True, False):
            with self.subTest(use_mujoco_cpu=use_mujoco_cpu):
                solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=use_mujoco_cpu, disable_contacts=True)
                self.assertEqual(solver.mj_model.ntendon, 1)
                np.testing.assert_array_equal(solver.mj_model.tendon_num, [6])
                np.testing.assert_array_equal(
                    solver.mj_model.wrap_type,
                    [
                        native.mjtWrap.mjWRAP_SITE,
                        native.mjtWrap.mjWRAP_SPHERE,
                        native.mjtWrap.mjWRAP_SITE,
                        native.mjtWrap.mjWRAP_PULLEY,
                        native.mjtWrap.mjWRAP_SITE,
                        native.mjtWrap.mjWRAP_SITE,
                    ],
                )
                self.assertAlmostEqual(solver.mj_model.wrap_prm[3], 2.0)
                self.assertGreater(solver.mj_data.ten_length[0], 0.0)

    def test_reject_fixed_tendon_across_worlds(self):
        """Reject foreign joints before appending any tendon or child rows."""
        builder = newton.ModelBuilder()
        _, global_joint = _add_revolute(builder, "global")
        builder.begin_world()
        _, joint0 = _add_revolute(builder, "world0")
        builder.end_world()
        builder.begin_world()
        _, joint1 = _add_revolute(builder, "world1")

        with self.assertRaisesRegex(ValueError, "belongs to world 0"):
            mujoco.add_tendon_fixed(builder, [(joint1, 1.0), (joint0, -1.0)])
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon", 0), 0)
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon_joint", 0), 0)

        tendon = mujoco.add_tendon_fixed(builder, [(joint1, 1.0), (global_joint, -1.0)])
        self.assertEqual(tendon, 0)
        self.assertEqual(builder.custom_attributes["mujoco:tendon_joint"].values, [joint1, global_joint])
        builder.end_world()
        with self.assertRaisesRegex(ValueError, "belongs to world 1"):
            mujoco.add_tendon_fixed(builder, [(joint1, 1.0)])

    def test_reject_spatial_tendon_across_worlds(self):
        """Check every site, geom, and sidesite while allowing global shapes."""
        builder = newton.ModelBuilder()
        global_site = builder.add_site(-1)
        global_geom = builder.add_shape_sphere(-1, radius=0.1)
        builder.begin_world()
        site0 = builder.add_site(-1)
        geom0 = builder.add_shape_sphere(-1, radius=0.1)
        builder.end_world()
        builder.begin_world()
        site1 = builder.add_site(-1)
        geom1 = builder.add_shape_sphere(-1, radius=0.1)

        entries = (
            mujoco.TendonWrapSite(site0),
            mujoco.TendonWrapGeom(geom0),
            mujoco.TendonWrapGeom(geom1, sidesite=site0),
        )
        for entry in entries:
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "belongs to world 0"):
                mujoco.add_tendon_spatial(builder, [mujoco.TendonWrapSite(site1), entry])
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon", 0), 0)
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:tendon_wrap", 0), 0)

        tendon = mujoco.add_tendon_spatial(
            builder,
            [
                mujoco.TendonWrapSite(site1),
                mujoco.TendonWrapGeom(geom1, sidesite=global_site),
                mujoco.TendonWrapSite(global_site),
                mujoco.TendonWrapGeom(global_geom, sidesite=site1),
                mujoco.TendonWrapSite(site1),
            ],
        )
        self.assertEqual(tendon, 0)
        self.assertEqual(builder._custom_frequency_counts["mujoco:tendon_wrap"], 5)
        builder.end_world()
        with self.assertRaisesRegex(ValueError, "belongs to world 1"):
            mujoco.add_tendon_spatial(builder, [mujoco.TendonWrapSite(site1)])

    def test_reject_unsupported_fixed_tendon_joints(self):
        """Require a scalar joint with a single MuJoCo joint mapping."""
        for joint_type in ("d6", "fixed", "ball", "free"):
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

    def test_reject_equalities_across_worlds(self):
        """Validate both equality operands before appending rows, preserving global references."""
        builder = newton.ModelBuilder()
        global_body, global_joint = _add_revolute(builder, "global")
        builder.begin_world()
        body0, joint0 = _add_revolute(builder, "world0")
        builder.end_world()
        builder.begin_world()
        body1, joint1 = _add_revolute(builder, "world1")
        for helper, foreign, local, global_index in (
            (mujoco.add_equality_connect, body0, body1, global_body),
            (mujoco.add_equality_weld, body0, body1, global_body),
            (mujoco.add_equality_joint, joint0, joint1, global_joint),
        ):
            before = builder._custom_frequency_counts.get("mujoco:equality_constraint", 0)
            for operands in ((foreign, local), (local, foreign)):
                with self.subTest(helper=helper.__name__, operands=operands):
                    with self.assertRaisesRegex(ValueError, "belongs to world 0"):
                        helper(builder, *operands)
                    self.assertEqual(builder._custom_frequency_counts.get("mujoco:equality_constraint", 0), before)
            helper(builder, local, global_index)
        mujoco.add_equality_connect(builder, body1)
        mujoco.add_equality_weld(builder, body1)
        builder.end_world()

    def test_reject_invalid_equality_custom_attributes(self):
        """Reject managed overrides and invalid extras without leaving partial rows."""
        for helper in (mujoco.add_equality_connect, mujoco.add_equality_weld, mujoco.add_equality_joint):
            builder = newton.ModelBuilder()
            body0, joint0 = _add_revolute(builder, "first")
            body1, joint1 = _add_revolute(builder, "second")
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name="body_value",
                    namespace="test",
                    frequency=newton.Model.AttributeFrequency.BODY,
                    dtype=float,
                )
            )
            operands = (joint0, joint1) if helper is mujoco.add_equality_joint else (body0, body1)
            for extras, error in (
                ({"mujoco:equality_constraint_world": 1}, ValueError),
                ({"test:missing": 1}, AttributeError),
                ({"test:body_value": 1}, ValueError),
            ):
                with self.subTest(helper=helper.__name__, extras=extras):
                    with self.assertRaises(error):
                        helper(builder, *operands, custom_attributes=extras)
                    self.assertEqual(builder._custom_frequency_counts.get("mujoco:equality_constraint", 0), 0)

    def test_weld_preserves_each_world_initial_pose(self):
        """Infer weld-to-world poses per copy and preserve them across notifications."""
        blueprint = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = blueprint.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        mujoco.add_equality_weld(blueprint, body)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(blueprint)
        builder.add_world(blueprint, xform=wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity()))
        model = builder.finalize()
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=False, separate_worlds=True, disable_contacts=True)
        initial = solver.mjw_model.eq_data.numpy().copy()
        solver.notify_model_changed(newton.ModelFlags.CONSTRAINT_PROPERTIES)
        solver.notify_model_changed(newton.ModelFlags.ACTUATOR_PROPERTIES)
        np.testing.assert_allclose(solver.mjw_model.eq_data.numpy(), initial)
        state0, state1 = model.state(), model.state()
        for _ in range(100):
            solver.step(state0, state1, model.control(), None, 0.005)
            state0, state1 = state1, state0
        np.testing.assert_allclose(state0.body_q.numpy(), model.body_q.numpy(), atol=1e-5)

    def test_reject_contact_pair_sites(self):
        """Reject sites instead of silently dropping the pair during solver construction."""
        builder = newton.ModelBuilder()
        site = builder.add_site(-1)
        shape = builder.add_shape_sphere(-1, radius=0.1)
        for shapes in ((site, shape), (shape, site)):
            with self.subTest(shapes=shapes), self.assertRaisesRegex(ValueError, "site"):
                mujoco.add_contact_pair(builder, *shapes)
        self.assertEqual(builder._custom_frequency_counts.get("mujoco:pair", 0), 0)

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
