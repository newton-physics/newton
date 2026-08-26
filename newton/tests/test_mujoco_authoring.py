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

    def test_reject_multidof_joint_without_dof(self):
        """Require an explicit local DOF for multi-DOF joint targets."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_ball(parent=-1, child=body)
        builder.add_articulation([joint])

        with self.assertRaisesRegex(ValueError, "dof"):
            mujoco.add_actuator_motor(builder, target=mujoco.ActuatorTarget.joint(joint))

    def test_remap_actuator_target_when_merging_builders(self):
        """Offset heterogeneous actuator targets during builder composition."""
        blueprint = newton.ModelBuilder()
        blueprint_body, blueprint_joint = _add_revolute(blueprint, "blueprint_hinge")
        blueprint_site0 = blueprint.add_site(blueprint_body, label="blueprint_site0")
        blueprint_site1 = blueprint.add_site(blueprint_body, label="blueprint_site1")
        blueprint_tendon = mujoco.add_tendon_fixed(blueprint, joints=[(blueprint_joint, 1.0)])
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.joint(blueprint_joint))
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.tendon(blueprint_tendon))
        mujoco.add_actuator_motor(blueprint, target=mujoco.ActuatorTarget.site(blueprint_site0, blueprint_site1))
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


if __name__ == "__main__":
    unittest.main()
