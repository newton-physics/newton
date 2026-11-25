# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton import JointDriveType, JointTargetMode, ModelFlags, NewtonWarning
from newton.solvers import (
    SolverFeatherstone,
    SolverKamino,
    SolverMuJoCo,
    SolverSemiImplicit,
    SolverSteel,
    SolverVBD,
    SolverXPBD,
)
from newton.tests.unittest_utils import MUJOCO_MODULES, get_test_devices, requires


@requires(*MUJOCO_MODULES)
class TestJointDrive(unittest.TestCase):
    def compute_expected_velocity_outcome(
        self,
        world_id,
        g,
        dt,
        joint_type,
        free_axis,
        pos_targets,
        vel_targets,
        target_kes,
        target_kds,
        joint_qs,
        joint_qds,
        masses,
        inertias,
    ) -> float:
        pos_target = pos_targets[world_id]
        vel_target = vel_targets[world_id]
        ke = target_kes[world_id]
        kd = target_kds[world_id]
        q = joint_qs[world_id]
        qd = joint_qds[world_id]
        mass = masses[world_id]
        inertia = inertias[world_id]

        M = 0.0
        if joint_type == newton.JointType.PRISMATIC:
            M = mass
        elif joint_type == newton.JointType.REVOLUTE:
            M = inertia[free_axis][free_axis]
        else:
            print("unsupported joint type")

        pos_err = pos_target - q
        vel_err = vel_target - qd
        F = ke * pos_err + kd * vel_err

        F += M * g

        qdNew = qd + F * dt / M
        return qdNew

    def run_test_joint_drive_no_limits(self, is_prismatic: bool, world_up_axis: int, joint_motion_axis: int):
        g = 0.0
        if is_prismatic and world_up_axis == joint_motion_axis:
            g = 5.0

        dt = 0.01

        joint_type = newton.JointType.PRISMATIC
        if is_prismatic:
            joint_type = newton.JointType.PRISMATIC
        else:
            joint_type = newton.JointType.REVOLUTE

        nb_worlds = 2
        body_masses = [10.0, 20.0]
        body_coms = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)]
        body_inertias = [
            wp.mat33(4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0),
            wp.mat33(8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0),
        ]
        joint_start_positions = [100.0, 205.0]
        joint_start_velocities = [10.0, 25.0]
        joint_pos_targets = [200.0, 300.0]
        joint_vel_targets = [0.0, 0.0]
        joint_drive_stiffnesses = [100.0, 200.0]
        joint_drive_dampings = [10.0, 20.0]

        gravity_vector = tuple(component * g for component in newton.Axis.from_any(world_up_axis).to_vector())
        main_builder = newton.ModelBuilder(gravity=gravity_vector, up_axis=world_up_axis)
        for i in range(0, nb_worlds):
            body_mass = body_masses[i]
            body_com = body_coms[i]
            body_inertia = body_inertias[i]
            drive_stiffness = joint_drive_stiffnesses[i]
            joint_pos_target = joint_pos_targets[i]
            joint_vel_target = joint_vel_targets[i]
            joint_drive_damping = joint_drive_dampings[i]
            joint_start_position = joint_start_positions[i]
            joint_start_velocity = joint_start_velocities[i]

            # Create a single body jointed to the world with a prismatic joint
            # Make sure that we use the mass properties specified here by setting shape density to 0.0
            world_builder = newton.ModelBuilder(gravity=gravity_vector, up_axis=world_up_axis)
            bodyIndex = world_builder.add_link(mass=body_mass, inertia=body_inertia, com=body_com)
            world_builder.add_shape_sphere(
                radius=1.0, body=bodyIndex, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
            )
            if is_prismatic:
                world_builder.add_joint_prismatic(
                    axis=joint_motion_axis,
                    parent=-1,
                    child=bodyIndex,
                    target_q=joint_pos_target,
                    target_qd=joint_vel_target,
                    target_ke=drive_stiffness,
                    target_kd=joint_drive_damping,
                    armature=0.0,
                    effort_limit=1000000000000.0,
                    velocity_limit=100000000000000000.0,
                    friction=0.0,
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                )
            else:
                world_builder.add_joint_revolute(
                    axis=joint_motion_axis,
                    parent=-1,
                    child=bodyIndex,
                    target_q=joint_pos_target,
                    target_qd=joint_vel_target,
                    target_ke=drive_stiffness,
                    target_kd=joint_drive_damping,
                    armature=0.0,
                    effort_limit=1000000000000.0,
                    velocity_limit=100000000000000000.0,
                    friction=0.0,
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                )

            # Add the joint to an articulation
            world_builder.add_articulation([0])

            main_builder.add_world(world_builder)

            # Set the start pos and vel of the dof.
            main_builder.joint_q[i] = joint_start_position
            main_builder.joint_qd[i] = joint_start_velocity

        # Create the MujocoSolver instance
        model = main_builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        collision_pipeline.collide(state_in, contacts)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = SolverMuJoCo(
            model, iterations=1, ls_iterations=1, disable_contacts=True, use_mujoco_cpu=False, integrator="euler"
        )

        # Compute the expected velocity outcome after a single sim step.
        vNew = [0.0] * nb_worlds
        for i in range(0, nb_worlds):
            vNew[i] = self.compute_expected_velocity_outcome(
                world_id=i,
                g=g,
                dt=dt,
                joint_type=joint_type,
                free_axis=joint_motion_axis,
                pos_targets=joint_pos_targets,
                vel_targets=[0.0, 0.0],
                target_kes=joint_drive_stiffnesses,
                target_kds=joint_drive_dampings,
                joint_qs=joint_start_positions,
                joint_qds=joint_start_velocities,
                masses=body_masses,
                inertias=body_inertias,
            )

        # Perform 1 sim step.
        solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nb_worlds):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)
        state_in, state_out = state_out, state_in

        #########################

        # Update the stiffness and damping values and reset to start state stored in model.joint_q and model.joint_qd
        joint_drive_stiffnesses[0] *= 2.0
        joint_drive_stiffnesses[1] *= 2.5
        joint_drive_dampings[0] *= 2.75
        joint_drive_dampings[1] *= 3.5
        model.joint_target_ke.assign(joint_drive_stiffnesses)
        model.joint_target_kd.assign(joint_drive_dampings)
        state_in.joint_q.assign(joint_start_positions)
        state_in.joint_qd.assign(joint_start_velocities)
        control.joint_target_q.assign(joint_pos_targets)
        control.joint_target_qd.assign(joint_vel_targets)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        # Recompute the expected velocity outcomes
        for i in range(0, nb_worlds):
            vNew[i] = self.compute_expected_velocity_outcome(
                world_id=i,
                g=g,
                dt=dt,
                joint_type=joint_type,
                free_axis=joint_motion_axis,
                pos_targets=joint_pos_targets,
                vel_targets=[0.0, 0.0],
                target_kes=joint_drive_stiffnesses,
                target_kds=joint_drive_dampings,
                joint_qs=joint_start_positions,
                joint_qds=joint_start_velocities,
                masses=body_masses,
                inertias=body_inertias,
            )

        # Run a sim step with the new values of ke and kd
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nb_worlds):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)
        state_in, state_out = state_out, state_in

        ################################
        # Change to velocity control and reset to start state
        joint_vel_targets = [20.0, 300.0]
        joint_drive_stiffnesses = [0.0, 0.0]
        joint_drive_dampings = [10.0, 20.0]
        joint_start_positions = [0.0, 0.0]
        joint_start_velocities = [0.0, 0.0]

        model.joint_target_ke.assign(joint_drive_stiffnesses)
        model.joint_target_kd.assign(joint_drive_dampings)
        control.joint_target_qd.assign(joint_vel_targets)
        state_in.joint_q.assign(joint_start_positions)
        state_in.joint_qd.assign(joint_start_velocities)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        # Recompute the expected velocity outcomes
        for i in range(0, nb_worlds):
            vNew[i] = self.compute_expected_velocity_outcome(
                world_id=i,
                g=g,
                dt=dt,
                joint_type=joint_type,
                free_axis=joint_motion_axis,
                pos_targets=[0.0, 0.0],
                vel_targets=joint_vel_targets,
                target_kes=joint_drive_stiffnesses,
                target_kds=joint_drive_dampings,
                joint_qs=joint_start_positions,
                joint_qds=joint_start_velocities,
                masses=body_masses,
                inertias=body_inertias,
            )

        # Run a sim step with the new drive type
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nb_worlds):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)
        state_in, state_out = state_out, state_in

        ################################

        # Now run again with no control (zero stiffness/damping) and reset back to the start state.

        joint_drive_stiffnesses = [0.0, 0.0]
        joint_drive_dampings = [0.0, 0.0]
        model.joint_target_ke.assign(joint_drive_stiffnesses)
        model.joint_target_kd.assign(joint_drive_dampings)

        state_in.joint_q.assign(joint_start_positions)
        state_in.joint_qd.assign(joint_start_velocities)
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        # Recompute the expected velocity outcomes
        for i in range(0, nb_worlds):
            vNew[i] = self.compute_expected_velocity_outcome(
                world_id=i,
                g=g,
                dt=dt,
                joint_type=joint_type,
                free_axis=joint_motion_axis,
                pos_targets=[0.0, 0.0],
                vel_targets=[0.0, 0.0],
                target_kes=joint_drive_stiffnesses,
                target_kds=joint_drive_dampings,
                joint_qs=joint_start_positions,
                joint_qds=joint_start_velocities,
                masses=body_masses,
                inertias=body_inertias,
            )

        # Run a sim step with the new drive type
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        solver.step(state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nb_worlds):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)

    def test_joint_drive_prismatic_upX_motionX(self):
        self.run_test_joint_drive_no_limits(True, 0, 0)

    def test_joint_drive_prismatic_upX_motionY(self):
        self.run_test_joint_drive_no_limits(True, 0, 1)

    def test_joint_drive_prismatic_upX_motionZ(self):
        self.run_test_joint_drive_no_limits(True, 0, 2)

    def test_joint_drive_prismatic_upY_motionX(self):
        self.run_test_joint_drive_no_limits(True, 1, 0)

    def test_joint_drive_prismatic_upY_motionY(self):
        self.run_test_joint_drive_no_limits(True, 1, 1)

    def test_joint_drive_prismatic_upY_motionZ(self):
        self.run_test_joint_drive_no_limits(True, 1, 2)

    def test_joint_drive_prismatic_upZ_motionX(self):
        self.run_test_joint_drive_no_limits(True, 2, 0)

    def test_joint_drive_prismatic_upZ_motionY(self):
        self.run_test_joint_drive_no_limits(True, 2, 1)

    def test_joint_drive_prismatic_upZ_motionZ(self):
        self.run_test_joint_drive_no_limits(True, 2, 2)

    def test_joint_drive_revolute_upX_motionX(self):
        self.run_test_joint_drive_no_limits(False, 0, 0)

    def test_joint_drive_revolute_upX_motionY(self):
        self.run_test_joint_drive_no_limits(False, 0, 1)

    def test_joint_drive_revolute_upX_motionZ(self):
        self.run_test_joint_drive_no_limits(False, 0, 2)

    def test_joint_drive_revolute_upY_motionX(self):
        self.run_test_joint_drive_no_limits(False, 1, 0)

    def test_joint_drive_revolute_upY_motionY(self):
        self.run_test_joint_drive_no_limits(False, 1, 1)

    def test_joint_drive_revolute_upY_motionZ(self):
        self.run_test_joint_drive_no_limits(False, 1, 2)

    def test_joint_drive_revolute_upZ_motionX(self):
        self.run_test_joint_drive_no_limits(False, 2, 0)

    def test_joint_drive_revolute_upZ_motionY(self):
        self.run_test_joint_drive_no_limits(False, 2, 1)

    def test_joint_drive_revolute_upZ_motionZ(self):
        self.run_test_joint_drive_no_limits(False, 2, 2)


class TestJointAccelerationDrive(unittest.TestCase):
    """Tests for JointDriveType.ACCELERATION across USD import, solvers, and diagnostics."""

    @staticmethod
    def _build_usd_prismatic_stage(
        mode: str,
        mass: float,
        *,
        stiffness: float = 10.0,
        damping: float = 0.0,
        target_position: float = 1.0,
        target_velocity: float = 0.0,
        max_force: float | None = None,
    ):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, "/Rig").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name, link_mass in [("Base", 1.0), ("Slide", mass)]:
            link = UsdGeom.Xform.Define(stage, "/Rig/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link)
            props = UsdPhysics.MassAPI.Apply(link)
            props.CreateMassAttr(link_mass)
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(link_mass / 6.0))
            props.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/Rig/Joint")
        joint.CreateBody0Rel().SetTargets(["/Rig/Base"])
        joint.CreateBody1Rel().SetTargets(["/Rig/Slide"])
        joint.CreateAxisAttr("X")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateTypeAttr(mode)
        drive.CreateTargetPositionAttr(target_position)
        drive.CreateTargetVelocityAttr(target_velocity)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        if max_force is not None:
            drive.CreateMaxForceAttr(max_force)
        return stage

    @staticmethod
    def _build_usd_revolute_stage(
        mode: str,
        mass: float,
        inertia_diag: float,
        *,
        stiffness_deg: float = 10.0 * math.pi / 180.0,
        target_position_deg: float = math.degrees(1.0),
    ):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, "/Rig").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name, link_mass, link_i in [("Base", 1.0, 1.0), ("Link", mass, inertia_diag)]:
            link = UsdGeom.Xform.Define(stage, "/Rig/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link)
            props = UsdPhysics.MassAPI.Apply(link)
            props.CreateMassAttr(link_mass)
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(link_i))
            props.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/Rig/Joint")
        joint.CreateBody0Rel().SetTargets(["/Rig/Base"])
        joint.CreateBody1Rel().SetTargets(["/Rig/Link"])
        joint.CreateAxisAttr("Z")
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr(mode)
        drive.CreateTargetPositionAttr(target_position_deg)
        drive.CreateStiffnessAttr(stiffness_deg)
        return stage

    @requires("pxr.Usd", *MUJOCO_MODULES)
    def test_usd_prismatic_acceleration_vs_force_mujoco(self):
        """Verify USD prismatic acceleration drives give mass-independent acceleration in SolverMuJoCo."""
        dt = 1e-4
        backends = [("cpu", True)]
        if wp.is_cuda_available():
            backends.append(("cuda:0", False))

        for device, use_cpu in backends:
            for mode, expected in (("force", (10.0, 1.0)), ("acceleration", (10.0, 10.0))):
                measured = []
                for mass in (1.0, 10.0):
                    stage = self._build_usd_prismatic_stage(mode, mass)
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    builder.add_usd(stage, floating=False)
                    model = builder.finalize(device=device)
                    state_in, state_out = model.state(), model.state()
                    solver = SolverMuJoCo(model, use_mujoco_cpu=use_cpu)
                    solver.step(state_in, state_out, model.control(), None, dt)
                    measured.append(float(state_out.joint_qd.numpy()[0]) / dt)
                with self.subTest(device=device, use_cpu=use_cpu, mode=mode):
                    for got, exp in zip(measured, expected, strict=True):
                        self.assertAlmostEqual(got, exp, places=2)

    @requires("pxr.Usd")
    def test_usd_prismatic_acceleration_vs_force_featherstone(self):
        """Verify USD prismatic acceleration drives give mass-independent acceleration in SolverFeatherstone."""
        dt = 1e-4
        for device in get_test_devices():
            for mode, expected in (("force", (10.0, 1.0)), ("acceleration", (10.0, 10.0))):
                measured = []
                for mass in (1.0, 10.0):
                    stage = self._build_usd_prismatic_stage(mode, mass)
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    builder.add_usd(stage, floating=False)
                    model = builder.finalize(device=device)
                    state_in, state_out = model.state(), model.state()
                    solver = SolverFeatherstone(model)
                    solver.step(state_in, state_out, model.control(), None, dt)
                    measured.append(float(state_out.joint_qd.numpy()[0]) / dt)
                with self.subTest(device=device, mode=mode):
                    for got, exp in zip(measured, expected, strict=True):
                        self.assertAlmostEqual(got, exp, places=2)

    @requires("pxr.Usd", *MUJOCO_MODULES)
    def test_usd_revolute_acceleration_drive(self):
        """Verify USD revolute acceleration drives are inertia-independent on MuJoCo and Featherstone."""
        dt = 1e-4
        for solver_kind in ("mujoco_cpu", "featherstone"):
            for mode, expected in (("force", (10.0, 2.0)), ("acceleration", (10.0, 10.0))):
                measured = []
                for inertia_diag in (1.0, 5.0):
                    stage = self._build_usd_revolute_stage(mode, mass=1.0, inertia_diag=inertia_diag)
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    builder.add_usd(stage, floating=False)
                    model = builder.finalize(device="cpu")
                    state_in, state_out = model.state(), model.state()
                    if solver_kind == "mujoco_cpu":
                        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                    else:
                        solver = SolverFeatherstone(model)
                    solver.step(state_in, state_out, model.control(), None, dt)
                    measured.append(float(state_out.joint_qd.numpy()[0]) / dt)
                with self.subTest(solver=solver_kind, mode=mode):
                    for got, exp in zip(measured, expected, strict=True):
                        self.assertAlmostEqual(got, exp, places=2)

    @requires("pxr.Usd", *MUJOCO_MODULES)
    def test_usd_d6_mixed_drive_types(self):
        """Verify USD D6 joints support mixed acceleration and force drives on independent axes."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        dt = 1e-4
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        root = UsdGeom.Xform.Define(stage, "/Rig").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name, mass in [("Base", 1.0), ("Body", 10.0)]:
            link = UsdGeom.Xform.Define(stage, "/Rig/" + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(link)
            props = UsdPhysics.MassAPI.Apply(link)
            props.CreateMassAttr(mass)
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(5.0))
            props.CreateCenterOfMassAttr(Gf.Vec3f(0.0))

        joint = UsdPhysics.Joint.Define(stage, "/Rig/D6")
        joint.CreateBody0Rel().SetTargets(["/Rig/Base"])
        joint.CreateBody1Rel().SetTargets(["/Rig/Body"])
        for locked in ("transZ", "rotX", "rotY"):
            lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), locked)
            lim.CreateLowAttr(1.0)
            lim.CreateHighAttr(-1.0)
        for free_axis, lo, hi in (("transX", -100.0, 100.0), ("transY", -100.0, 100.0), ("rotZ", -180.0, 180.0)):
            lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), free_axis)
            lim.CreateLowAttr(lo)
            lim.CreateHighAttr(hi)

        d_tx = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "transX")
        d_tx.CreateTypeAttr("acceleration")
        d_tx.CreateTargetPositionAttr(1.0)
        d_tx.CreateStiffnessAttr(10.0)

        d_ty = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "transY")
        d_ty.CreateTypeAttr("force")
        d_ty.CreateTargetPositionAttr(1.0)
        d_ty.CreateStiffnessAttr(10.0)

        d_rz = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "rotZ")
        d_rz.CreateTypeAttr("acceleration")
        d_rz.CreateTargetPositionAttr(math.degrees(1.0))
        d_rz.CreateStiffnessAttr(10.0 * math.pi / 180.0)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_usd(stage, floating=False)
        model = builder.finalize(device="cpu")

        for solver_cls, kwargs in ((SolverMuJoCo, {"use_mujoco_cpu": True}), (SolverFeatherstone, {})):
            with self.subTest(solver=solver_cls.__name__):
                state_in, state_out = model.state(), model.state()
                solver = solver_cls(model, **kwargs)
                solver.step(state_in, state_out, model.control(), None, dt)
                acc = state_out.joint_qd.numpy() / dt
                # transX: accel drive -> 10 m/s^2; transY: force drive -> 10 / 10 = 1 m/s^2; rotZ: accel -> 10 rad/s^2
                self.assertAlmostEqual(float(acc[0]), 10.0, places=2)
                self.assertAlmostEqual(float(acc[1]), 1.0, places=2)
                self.assertAlmostEqual(float(acc[2]), 10.0, places=2)

    @requires(*MUJOCO_MODULES)
    def test_acceleration_drive_with_armature_and_floating_base(self):
        """Account for joint armature and floating-base reduced mass in acceleration drives."""
        dt = 1e-4

        # Case 1: Single root prismatic joint with nonzero armature (tests exact CPU mj_solveM path).
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=10.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        j = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            axis=newton.Axis.X,
            target_q=1.0,
            target_ke=10.0,
            armature=5.0,
            drive_type=JointDriveType.ACCELERATION,
        )
        builder.add_articulation([j])

        backends = [("cpu", True, SolverMuJoCo), ("cpu", False, SolverFeatherstone)]
        if wp.is_cuda_available():
            backends.append(("cuda:0", False, SolverMuJoCo))
            backends.append(("cuda:0", False, SolverFeatherstone))

        for device, use_cpu, solver_cls in backends:
            with self.subTest(case="armature", device=device, solver=solver_cls.__name__):
                model = builder.finalize(device=device)
                state_in, state_out = model.state(), model.state()
                solver = solver_cls(model, use_mujoco_cpu=use_cpu) if solver_cls is SolverMuJoCo else solver_cls(model)
                solver.step(state_in, state_out, model.control(), None, dt)
                self.assertAlmostEqual(float(state_out.joint_qd.numpy()[0]) / dt, 10.0, places=2)

        # Case 2: Floating base (mass=2) + prismatic child (mass=10) -> reduced mass = 20/12.
        fb = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = fb.add_link(mass=2.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        slide = fb.add_link(mass=10.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        j_free = fb.add_joint_free(base)
        j_slide = fb.add_joint_prismatic(
            parent=base,
            child=slide,
            axis=newton.Axis.X,
            target_q=1.0,
            target_ke=10.0,
            drive_type=JointDriveType.ACCELERATION,
        )
        fb.add_articulation([j_free, j_slide])
        for solver_cls, kwargs in ((SolverMuJoCo, {"use_mujoco_cpu": True}), (SolverFeatherstone, {})):
            with self.subTest(case="floating_base", solver=solver_cls.__name__):
                model = fb.finalize(device="cpu")
                state_in, state_out = model.state(), model.state()
                solver = solver_cls(model, **kwargs)
                solver.step(state_in, state_out, model.control(), None, dt)
                # DOF 6 is the prismatic slide joint relative acceleration.
                self.assertAlmostEqual(float(state_out.joint_qd.numpy()[6]) / dt, 10.0, places=2)

    @requires(*MUJOCO_MODULES)
    def test_acceleration_drive_velocity_mode_and_effort_limit(self):
        """Verify velocity/PD acceleration drives and force-space effort limit clamping."""
        dt = 1e-4

        # Velocity-only acceleration drive: target_qd=2.0, target_kd=5.0 -> accel = 10 m/s^2 for any mass.
        for mass in (1.0, 10.0):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=mass, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            j = builder.add_joint_prismatic(
                parent=-1,
                child=link,
                axis=newton.Axis.X,
                target_qd=2.0,
                target_kd=5.0,
                actuator_mode=JointTargetMode.VELOCITY,
                drive_type=JointDriveType.ACCELERATION,
            )
            builder.add_articulation([j])
            model = builder.finalize(device="cpu")
            state_in, state_out = model.state(), model.state()
            solver = SolverMuJoCo(model, use_mujoco_cpu=True)
            solver.step(state_in, state_out, model.control(), None, dt)
            self.assertAlmostEqual(float(state_out.joint_qd.numpy()[0]) / dt, 10.0, places=2)

        # Effort limit (force ceiling): 10 kg body commanded at 10 m/s^2 needs 100 N; effort_limit=25 N clamps to 2.5 m/s^2.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=10.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        j = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            axis=newton.Axis.X,
            target_q=1.0,
            target_ke=10.0,
            effort_limit=25.0,
            drive_type=JointDriveType.ACCELERATION,
        )
        builder.add_articulation([j])
        model = builder.finalize(device="cpu")
        state_in, state_out = model.state(), model.state()
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        solver.step(state_in, state_out, model.control(), None, dt)
        self.assertAlmostEqual(float(state_out.joint_qd.numpy()[0]) / dt, 2.5, places=2)

    @requires(*MUJOCO_MODULES)
    def test_acceleration_drive_notify_model_changed_and_multiworld(self):
        """Verify acceleration drives stay mass-independent across notify_model_changed and multi-world GPU."""
        dt = 1e-4

        # Single-world CPU: test mass update, armature update, and drive-type toggling.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        link = builder.add_link(mass=2.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        j = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            axis=newton.Axis.X,
            target_q=1.0,
            target_ke=10.0,
            drive_type=JointDriveType.ACCELERATION,
        )
        builder.add_articulation([j])
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCo(model, use_mujoco_cpu=True)

        def measure_accel() -> float:
            s_in, s_out = model.state(), model.state()
            solver.step(s_in, s_out, model.control(), None, dt)
            return float(s_out.joint_qd.numpy()[0]) / dt

        self.assertAlmostEqual(measure_accel(), 10.0, places=2)

        # Change body mass from 2 kg to 10 kg: acceleration stays 10 m/s^2.
        model.body_mass.assign(np.array([10.0], dtype=np.float32))
        model.body_inv_mass.assign(np.array([0.1], dtype=np.float32))
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
        self.assertAlmostEqual(measure_accel(), 10.0, places=2)

        # Add joint armature = 5 kg: acceleration stays 10 m/s^2.
        model.joint_armature.assign(np.array([5.0], dtype=np.float32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES)
        self.assertAlmostEqual(measure_accel(), 10.0, places=2)

        # Switch drive type to FORCE: acceleration drops to 10 / (10 + 5) = 2/3 m/s^2.
        model.joint_drive_type.assign(np.array([int(JointDriveType.FORCE)], dtype=np.int32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        self.assertAlmostEqual(measure_accel(), 10.0 / 15.0, places=2)

        # Switch back to ACCELERATION with new gain ke=20: acceleration becomes 20 m/s^2.
        model.joint_drive_type.assign(np.array([int(JointDriveType.ACCELERATION)], dtype=np.int32))
        model.joint_target_ke.assign(np.array([20.0], dtype=np.float32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
        self.assertAlmostEqual(measure_accel(), 20.0, places=2)

        # Multi-world GPU: world 0 has mass 1 kg, world 1 has mass 10 kg; both accelerate at 10 m/s^2.
        if wp.is_cuda_available():
            main_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            for mass in (1.0, 10.0):
                wb = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                b_idx = wb.add_link(mass=mass, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
                j_idx = wb.add_joint_prismatic(
                    parent=-1,
                    child=b_idx,
                    axis=newton.Axis.X,
                    target_q=1.0,
                    target_ke=10.0,
                    drive_type=JointDriveType.ACCELERATION,
                )
                wb.add_articulation([j_idx])
                main_builder.add_world(wb)
            gpu_model = main_builder.finalize(device="cuda:0")
            gpu_solver = SolverMuJoCo(gpu_model, use_mujoco_cpu=False)
            s_in, s_out = gpu_model.state(), gpu_model.state()
            gpu_solver.step(s_in, s_out, gpu_model.control(), None, dt)
            accs = s_out.joint_qd.numpy() / dt
            self.assertAlmostEqual(float(accs[0]), 10.0, places=2)
            self.assertAlmostEqual(float(accs[1]), 10.0, places=2)

    def test_builder_collapse_fixed_joints_preserves_drive_type(self):
        """Preserve joint_drive_type when collapsing fixed joints in ModelBuilder."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        base = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        mid = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        tip = builder.add_link(mass=5.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        j0 = builder.add_joint_fixed(parent=-1, child=base)
        j1 = builder.add_joint_fixed(parent=base, child=mid)
        j2 = builder.add_joint_prismatic(
            parent=mid,
            child=tip,
            axis=newton.Axis.X,
            target_q=1.0,
            target_ke=10.0,
            drive_type=JointDriveType.ACCELERATION,
        )
        builder.add_articulation([j0, j1, j2])
        builder.collapse_fixed_joints()
        model = builder.finalize(device="cpu")
        self.assertEqual(model.joint_drive_type.numpy().tolist(), [int(JointDriveType.ACCELERATION)])

    def test_unsupported_solvers_warn_on_acceleration_drives(self):
        """Emit a NewtonWarning when constructing or updating unsupported solvers with acceleration drives."""

        def make_model(drive_type: JointDriveType):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            base = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            slide = builder.add_link(mass=10.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            j_fix = builder.add_joint_fixed(parent=-1, child=base)
            j_prismatic = builder.add_joint_prismatic(
                parent=base,
                child=slide,
                axis=newton.Axis.X,
                target_q=1.0,
                target_ke=10.0,
                drive_type=drive_type,
            )
            builder.add_articulation([j_fix, j_prismatic])
            builder.color()
            return builder.finalize(device="cpu")

        accel_model = make_model(JointDriveType.ACCELERATION)
        force_model = make_model(JointDriveType.FORCE)

        for solver_cls in (SolverSemiImplicit, SolverXPBD, SolverVBD, SolverSteel, SolverKamino):
            with self.subTest(solver=solver_cls.__name__, mode="acceleration_init"):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver_cls(accel_model)
                matching = [
                    w
                    for w in caught
                    if issubclass(w.category, NewtonWarning) and "JointDriveType.ACCELERATION" in str(w.message)
                ]
                self.assertEqual(len(matching), 1)

            with self.subTest(solver=solver_cls.__name__, mode="force_no_warning"):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    solver = solver_cls(force_model)
                matching = [
                    w
                    for w in caught
                    if issubclass(w.category, NewtonWarning) and "JointDriveType.ACCELERATION" in str(w.message)
                ]
                self.assertEqual(len(matching), 0)

                # Switching a force drive to acceleration and notifying should also emit the diagnostic.
                force_model.joint_drive_type.assign(np.array([int(JointDriveType.ACCELERATION)], dtype=np.int32))
                try:
                    with warnings.catch_warnings(record=True) as caught_notify:
                        warnings.simplefilter("always")
                        solver.notify_model_changed(ModelFlags.JOINT_DOF_FORCE_PROPERTIES)
                    matching_notify = [
                        w
                        for w in caught_notify
                        if issubclass(w.category, NewtonWarning) and "JointDriveType.ACCELERATION" in str(w.message)
                    ]
                    self.assertEqual(len(matching_notify), 1)
                finally:
                    force_model.joint_drive_type.assign(np.array([int(JointDriveType.FORCE)], dtype=np.int32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
