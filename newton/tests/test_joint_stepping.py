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

"""Tests for joint stepping: armature, limits, friction, and drive.

Armature adds to the diagonal of the joint-space inertia matrix, so for a 1-DOF
revolute joint with applied torque tau, the acceleration should be approximately
tau / (I + armature), where I is the natural rotational inertia about the joint axis.
For a prismatic joint, the effective mass becomes (m + armature).

Joint limits are enforced via spring-damper penalty forces.  When a joint position
exceeds its limit, the solver applies a restoring force proportional to the
penetration depth (limit_ke) and a damping force (limit_kd).
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone, SolverMuJoCo, SolverNotifyFlags


class Sim:
    """Holds the simulation objects for a single joint test."""

    def __init__(self, model, solver, state_in, state_out, control):
        self.model = model
        self.solver = solver
        self.state_in = state_in
        self.state_out = state_out
        self.control = control


class TestJointSteppingBase:
    def _create_solver(self, model):
        raise NotImplementedError

    def _build_model(
        self,
        armature: float,
        body_inertia: float,
        joint_type: str,
        body_mass: float = 1.0,
        motion_axis: int = 2,
        limit_lower: float | None = None,
        limit_upper: float | None = None,
        friction: float = 0.0,
        target_ke: float | None = None,
        target_kd: float | None = None,
        target_pos: float | None = None,
        effort_limit: float = 1e12,
    ):
        """Build a single joint attached to the world with specified armature.

        Args:
            armature: Armature value for the joint DOF [kg*m^2 or kg].
            body_inertia: Diagonal rotational inertia of the body [kg*m^2].
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                or ``"d6_prismatic"``.
            body_mass: Mass of the body [kg].
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
            limit_lower: Lower joint limit [m or rad]. None for unlimited.
            limit_upper: Upper joint limit [m or rad]. None for unlimited.
            friction: Joint friction loss [N*m or N].
            target_ke: Position drive stiffness [N*m/rad or N/m]. None for default.
            target_kd: Position drive damping [N*m*s/rad or N*s/m]. None for default.
            target_pos: Target position for the drive [rad or m]. None for default.
            effort_limit: Maximum actuator force [N*m or N].

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        builder = newton.ModelBuilder(gravity=0.0, up_axis=1)

        inertia_mat = wp.mat33(
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
            0.0,
            0.0,
            0.0,
            body_inertia,
        )
        body = builder.add_link(
            mass=body_mass,
            inertia=inertia_mat,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        if joint_type == "prismatic":
            joint = builder.add_joint_prismatic(
                axis=motion_axis,
                parent=-1,
                child=body,
                armature=armature,
                effort_limit=effort_limit,
                velocity_limit=1e12,
                friction=friction,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                target_ke=target_ke,
                target_kd=target_kd,
                target_pos=target_pos,
            )
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                axis=motion_axis,
                parent=-1,
                child=body,
                armature=armature,
                effort_limit=effort_limit,
                velocity_limit=1e12,
                friction=friction,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
                target_ke=target_ke,
                target_kd=target_kd,
                target_pos=target_pos,
            )
        elif joint_type in ("d6_prismatic", "d6_revolute", "ball"):
            dof_kwargs = {}
            if limit_lower is not None:
                dof_kwargs["limit_lower"] = limit_lower
            if limit_upper is not None:
                dof_kwargs["limit_upper"] = limit_upper
            if target_ke is not None:
                dof_kwargs["target_ke"] = target_ke
            if target_kd is not None:
                dof_kwargs["target_kd"] = target_kd
            if target_pos is not None:
                dof_kwargs["target_pos"] = target_pos
            dof_cfg = newton.ModelBuilder.JointDofConfig(
                axis=motion_axis,
                armature=armature,
                friction=friction,
                effort_limit=effort_limit,
                **dof_kwargs,
            )
            if joint_type == "d6_prismatic":
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    linear_axes=[dof_cfg],
                )
            elif joint_type == "d6_revolute":
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    angular_axes=[dof_cfg],
                )
            elif joint_type == "ball":
                dof_x = newton.ModelBuilder.JointDofConfig(
                    axis=0, armature=armature, friction=friction, effort_limit=effort_limit, **dof_kwargs
                )
                dof_y = newton.ModelBuilder.JointDofConfig(
                    axis=1, armature=armature, friction=friction, effort_limit=effort_limit, **dof_kwargs
                )
                dof_z = newton.ModelBuilder.JointDofConfig(
                    axis=2, armature=armature, friction=friction, effort_limit=effort_limit, **dof_kwargs
                )
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    angular_axes=[dof_x, dof_y, dof_z],
                )
        else:
            raise ValueError(joint_type)
        builder.add_articulation(joints=[joint])

        model = builder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()

        solver = self._create_solver(model)

        return Sim(model, solver, state_in, state_out, control)

    def _qd_index(self, joint_type, motion_axis):
        """Return the index into ``joint_qd`` for the tested DOF."""
        return motion_axis if joint_type == "ball" else 0

    def _make_dof_array(self, value, joint_type, motion_axis):
        """Create a per-DOF array with ``value`` on the tested axis.

        For ball joints (3 DOFs), returns a 3-element array with ``value`` at
        ``motion_axis``.  For 1-DOF joints, returns a 1-element array.

        Args:
            value: Scalar value to place on the tested axis.
            joint_type: Joint type string.
            motion_axis: Axis index (0=X, 1=Y, 2=Z).
        """
        if joint_type == "ball":
            arr = np.zeros(3, dtype=np.float32)
            arr[motion_axis] = value
            return arr
        return np.array([value], dtype=np.float32)

    def _make_force_array(self, value, joint_type, motion_axis, num_dof):
        """Create a joint force array with ``value`` on the tested axis.

        Args:
            value: Force or torque magnitude [N or N*m].
            joint_type: Joint type string.
            motion_axis: Axis index (0=X, 1=Y, 2=Z).
            num_dof: Total number of DOFs for the joint.
        """
        force_arr = np.zeros(num_dof, dtype=np.float32)
        force_arr[self._qd_index(joint_type, motion_axis)] = value
        return force_arr


class TestJointArmatureBase(TestJointSteppingBase):
    def _test_armature_reduces_joint_speed(self, joint_type: str, motion_axis: int):
        """Applying the same force with higher armature yields lower joint speed.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                ``"d6_prismatic"``, or ``"ball"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """
        dt = 0.01
        force = 100.0
        effective_inertia = 4.0
        armatures = [0.0, 1.0, 10.0, 10000000000.0]
        num_armatures = len(armatures)

        # Build and step a sim for each armature value
        sims = [None] * num_armatures
        measured_qds = [0.0] * num_armatures
        for i in range(0, num_armatures):
            sim = self._build_model(
                armature=armatures[i],
                body_inertia=effective_inertia,
                joint_type=joint_type,
                body_mass=effective_inertia,
                motion_axis=motion_axis,
            )

            # Apply a force to the joint for 1 sim step and measure the joint speed.
            # For multi-DOF joints (ball), apply force only on the motion_axis DOF.
            num_dof = sim.state_in.joint_qd.numpy().shape[0]
            force_arr = self._make_force_array(force, joint_type, motion_axis, num_dof)
            sim.control.joint_f.assign(force_arr)
            sim.solver.step(state_in=sim.state_in, state_out=sim.state_out, control=sim.control, dt=dt, contacts=None)
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            qd_index = self._qd_index(joint_type, motion_axis)
            measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
            self.assertGreater(measured_qd, 0.0)

            sims[i] = sim
            measured_qds[i] = measured_qd

        # Higher armature should produce lower speed
        for i in range(0, num_armatures - 1):
            self.assertGreater(measured_qds[i], measured_qds[i + 1])
        self.assertAlmostEqual(measured_qds[num_armatures - 1], 0.0, places=5)

        # Check expected values: qd = force * dt / (effective_inertia + armature)
        for i in range(0, num_armatures):
            expected_qd = force * dt / (effective_inertia + armatures[i])
            self.assertAlmostEqual(measured_qds[i], expected_qd, delta=1e-4)

        # Change armature at runtime on the first model and verify
        armature_changed = 20.0
        armature_arr = np.full(num_dof, armature_changed, dtype=np.float32)
        sims[0].model.joint_armature.assign(armature_arr)
        sims[0].state_in.joint_q.assign(self._make_dof_array(0.0, joint_type, motion_axis))
        sims[0].state_in.joint_qd.assign(self._make_dof_array(0.0, joint_type, motion_axis))

        sims[0].solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        sims[0].control.joint_f.assign(force_arr)
        sims[0].solver.step(
            state_in=sims[0].state_in, state_out=sims[0].state_out, control=sims[0].control, dt=dt, contacts=None
        )
        sims[0].state_in, sims[0].state_out = sims[0].state_out, sims[0].state_in

        measured_qd_changed = sims[0].state_in.joint_qd.numpy()[qd_index]
        expected_qd_changed = force * dt / (effective_inertia + armature_changed)
        self.assertAlmostEqual(measured_qd_changed, expected_qd_changed, delta=1e-4)
        self.assertGreater(measured_qds[0], measured_qd_changed)

    def test_armature_reduces_joint_speed_revolute_x(self):
        """Higher armature yields lower joint speed for a revolute joint about X."""
        self._test_armature_reduces_joint_speed(joint_type="revolute", motion_axis=0)

    def test_armature_reduces_joint_speed_revolute_y(self):
        """Higher armature yields lower joint speed for a revolute joint about Y."""
        self._test_armature_reduces_joint_speed(joint_type="revolute", motion_axis=1)

    def test_armature_reduces_joint_speed_revolute_z(self):
        """Higher armature yields lower joint speed for a revolute joint about Z."""
        self._test_armature_reduces_joint_speed(joint_type="revolute", motion_axis=2)

    def test_armature_reduces_joint_speed_prismatic_x(self):
        """Higher armature yields lower joint speed for a prismatic joint along X."""
        self._test_armature_reduces_joint_speed(joint_type="prismatic", motion_axis=0)

    def test_armature_reduces_joint_speed_prismatic_y(self):
        """Higher armature yields lower joint speed for a prismatic joint along Y."""
        self._test_armature_reduces_joint_speed(joint_type="prismatic", motion_axis=1)

    def test_armature_reduces_joint_speed_prismatic_z(self):
        """Higher armature yields lower joint speed for a prismatic joint along Z."""
        self._test_armature_reduces_joint_speed(joint_type="prismatic", motion_axis=2)

    def test_armature_reduces_joint_speed_d6_revolute_x(self):
        """Higher armature yields lower joint speed for a D6 revolute joint about X."""
        self._test_armature_reduces_joint_speed(joint_type="d6_revolute", motion_axis=0)

    def test_armature_reduces_joint_speed_d6_revolute_y(self):
        """Higher armature yields lower joint speed for a D6 revolute joint about Y."""
        self._test_armature_reduces_joint_speed(joint_type="d6_revolute", motion_axis=1)

    def test_armature_reduces_joint_speed_d6_revolute_z(self):
        """Higher armature yields lower joint speed for a D6 revolute joint about Z."""
        self._test_armature_reduces_joint_speed(joint_type="d6_revolute", motion_axis=2)

    def test_armature_reduces_joint_speed_d6_prismatic_x(self):
        """Higher armature yields lower joint speed for a D6 prismatic joint along X."""
        self._test_armature_reduces_joint_speed(joint_type="d6_prismatic", motion_axis=0)

    def test_armature_reduces_joint_speed_d6_prismatic_y(self):
        """Higher armature yields lower joint speed for a D6 prismatic joint along Y."""
        self._test_armature_reduces_joint_speed(joint_type="d6_prismatic", motion_axis=1)

    def test_armature_reduces_joint_speed_d6_prismatic_z(self):
        """Higher armature yields lower joint speed for a D6 prismatic joint along Z."""
        self._test_armature_reduces_joint_speed(joint_type="d6_prismatic", motion_axis=2)

    def test_armature_reduces_joint_speed_ball_x(self):
        """Higher armature yields lower joint speed for a ball joint about X."""
        self._test_armature_reduces_joint_speed(joint_type="ball", motion_axis=0)

    def test_armature_reduces_joint_speed_ball_y(self):
        """Higher armature yields lower joint speed for a ball joint about Y."""
        self._test_armature_reduces_joint_speed(joint_type="ball", motion_axis=1)

    def test_armature_reduces_joint_speed_ball_z(self):
        """Higher armature yields lower joint speed for a ball joint about Z."""
        self._test_armature_reduces_joint_speed(joint_type="ball", motion_axis=2)


class TestJointLimitBase(TestJointSteppingBase):
    def _expected_limit_qd(
        self, ke: float, kd: float, q: float, qd: float, limit_q: float, dt: float, inertia: float
    ) -> float:
        """Compute the expected velocity after one step with a limit penalty force.

        Args:
            ke: Limit stiffness [N/m or N*m/rad].
            kd: Limit damping [N*s/m or N*m*s/rad].
            q: Current joint position [m or rad].
            qd: Current joint velocity [m/s or rad/s].
            limit_q: The limit value being violated [m or rad].
            dt: Time step [s].
            inertia: Effective inertia of the body [kg*m^2 or kg].
        """
        force = -ke * (q - limit_q) - kd * qd
        return qd + dt * force / inertia

    def _test_joint_limits(self, joint_type: str, motion_axis: int):
        """Verify that the penalty force from a joint limit breach produces the expected velocity.

        Starts the joint past its limit with zero velocity and steps once.
        The solver applies a spring-damper penalty force proportional to the
        penetration depth (``limit_ke * penetration``).  The test checks that
        the resulting ``joint_qd`` matches the analytically expected value.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                ``"d6_prismatic"``, or ``"ball"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """
        dt = 0.01
        body_mass = 1.0
        qd_index = self._qd_index(joint_type, motion_axis)

        # Configure the limits for the initial limit test.
        initial_upper_limit = 0.5
        initial_lower_limit = -0.3
        initial_limits_for_newton = [[None, initial_upper_limit], [initial_lower_limit, None]]
        initial_limit_qs = [initial_upper_limit, initial_lower_limit]
        initial_start_qs = [1.1 * initial_upper_limit, 1.1 * initial_lower_limit]
        initial_start_qds = [0.0, 0.0]

        # We will change the limits and test again.
        changed_upper_limit = 0.25
        changed_lower_limit = -0.15
        changed_limits_for_newton = [[None, changed_upper_limit], [changed_lower_limit, None]]
        changed_limit_qs = [changed_upper_limit, changed_lower_limit]
        changed_start_qs = [1.1 * changed_upper_limit, 1.1 * changed_lower_limit]
        changed_start_qds = [0.0, 0.0]

        # Configure the start state.
        num_limits = len(initial_limit_qs)
        for i in range(0, num_limits):
            sim = self._build_model(
                armature=0.0,
                body_inertia=body_mass,
                joint_type=joint_type,
                body_mass=body_mass,
                motion_axis=motion_axis,
                limit_lower=initial_limits_for_newton[i][0],
                limit_upper=initial_limits_for_newton[i][1],
            )
            num_dof = sim.state_in.joint_qd.numpy().shape[0]
            sim.state_in.joint_q.assign(self._make_dof_array(initial_start_qs[i], joint_type, motion_axis))
            sim.state_in.joint_qd.assign(self._make_dof_array(initial_start_qds[i], joint_type, motion_axis))

            # Compute the expected velocity after applying the penalty force to repair the limit breach.
            ke = sim.model.joint_limit_ke.numpy()[qd_index]
            kd = sim.model.joint_limit_kd.numpy()[qd_index]
            expected_qd = self._expected_limit_qd(
                ke, kd, initial_start_qs[i], initial_start_qds[i], initial_limit_qs[i], dt, body_mass
            )

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
            self.assertAlmostEqual(measured_qd, expected_qd, places=4)

            # Change the lower and upper limit at runtime, reset state, and verify the new limit is enforced.
            # Need to be careful here because setting model.joint_limit_upper to None does not work with SolverMujoco.
            # https://github.com/newton-physics/newton/issues/2072
            if i == 0:
                sim.model.joint_limit_upper.assign(np.full(num_dof, changed_limits_for_newton[i][1], dtype=np.float32))
            elif i == 1:
                sim.model.joint_limit_lower.assign(np.full(num_dof, changed_limits_for_newton[i][0], dtype=np.float32))
            else:
                raise ValueError(f"Unexpected index {i}")

            sim.state_in.joint_q.assign(self._make_dof_array(changed_start_qs[i], joint_type, motion_axis))
            sim.state_in.joint_qd.assign(self._make_dof_array(changed_start_qds[i], joint_type, motion_axis))
            sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            expected_qd = self._expected_limit_qd(
                ke, kd, changed_start_qs[i], changed_start_qds[i], changed_limit_qs[i], dt, body_mass
            )

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
            self.assertAlmostEqual(measured_qd, expected_qd, places=4)

            # Change ke at runtime, reset state, and verify the new stiffness is reflected.
            # Remember that the limits are currently set to changed_limits_for_newton.
            changed_ke = ke * 2.0
            sim.model.joint_limit_ke.assign(np.full(num_dof, changed_ke, dtype=np.float32))
            sim.state_in.joint_q.assign(self._make_dof_array(changed_start_qs[i], joint_type, motion_axis))
            sim.state_in.joint_qd.assign(self._make_dof_array(changed_start_qds[i], joint_type, motion_axis))
            sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            expected_qd = self._expected_limit_qd(
                changed_ke, kd, changed_start_qs[i], changed_start_qds[i], changed_limit_qs[i], dt, body_mass
            )

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
            self.assertAlmostEqual(measured_qd, expected_qd, places=4)

            # Change kd at runtime with a non-zero initial velocity to test damping.
            # Set ke=0 to isolate the damping force.
            # Only Featherstone matches the analytical formula; MuJoCo's constraint solver differs.
            if isinstance(sim.solver, SolverFeatherstone):
                changed_ke = 0
                changed_kd = kd * 3.0
                start_qd_for_kd_test = -5.0 if changed_start_qs[i] > 0 else 5.0
                sim.model.joint_limit_ke.assign(np.full(num_dof, changed_ke, dtype=np.float32))
                sim.model.joint_limit_kd.assign(np.full(num_dof, changed_kd, dtype=np.float32))
                sim.state_in.joint_q.assign(self._make_dof_array(changed_start_qs[i], joint_type, motion_axis))
                sim.state_in.joint_qd.assign(self._make_dof_array(start_qd_for_kd_test, joint_type, motion_axis))
                sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
                expected_qd = self._expected_limit_qd(
                    changed_ke,
                    changed_kd,
                    changed_start_qs[i],
                    start_qd_for_kd_test,
                    changed_limit_qs[i],
                    dt,
                    body_mass,
                )

                sim.solver.step(
                    state_in=sim.state_in,
                    state_out=sim.state_out,
                    control=sim.control,
                    dt=dt,
                    contacts=None,
                )
                sim.state_in, sim.state_out = sim.state_out, sim.state_in
                measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
                self.assertAlmostEqual(measured_qd, expected_qd, places=4)

    def test_joint_limits_revolute_x(self):
        """Joint limit is enforced for a revolute joint about X."""
        self._test_joint_limits(joint_type="revolute", motion_axis=0)

    def test_joint_limits_revolute_y(self):
        """Joint limit is enforced for a revolute joint about Y."""
        self._test_joint_limits(joint_type="revolute", motion_axis=1)

    def test_joint_limits_revolute_z(self):
        """Joint limit is enforced for a revolute joint about Z."""
        self._test_joint_limits(joint_type="revolute", motion_axis=2)

    def test_joint_limits_prismatic_x(self):
        """Joint limit is enforced for a prismatic joint along X."""
        self._test_joint_limits(joint_type="prismatic", motion_axis=0)

    def test_joint_limits_prismatic_y(self):
        """Joint limit is enforced for a prismatic joint along Y."""
        self._test_joint_limits(joint_type="prismatic", motion_axis=1)

    def test_joint_limits_prismatic_z(self):
        """Joint limit is enforced for a prismatic joint along Z."""
        self._test_joint_limits(joint_type="prismatic", motion_axis=2)

    def test_joint_limits_d6_revolute_x(self):
        """Joint limit is enforced for a D6 revolute joint about X."""
        self._test_joint_limits(joint_type="d6_revolute", motion_axis=0)

    def test_joint_limits_d6_revolute_y(self):
        """Joint limit is enforced for a D6 revolute joint about Y."""
        self._test_joint_limits(joint_type="d6_revolute", motion_axis=1)

    def test_joint_limits_d6_revolute_z(self):
        """Joint limit is enforced for a D6 revolute joint about Z."""
        self._test_joint_limits(joint_type="d6_revolute", motion_axis=2)

    def test_joint_limits_d6_prismatic_x(self):
        """Joint limit is enforced for a D6 prismatic joint along X."""
        self._test_joint_limits(joint_type="d6_prismatic", motion_axis=0)

    def test_joint_limits_d6_prismatic_y(self):
        """Joint limit is enforced for a D6 prismatic joint along Y."""
        self._test_joint_limits(joint_type="d6_prismatic", motion_axis=1)

    def test_joint_limits_d6_prismatic_z(self):
        """Joint limit is enforced for a D6 prismatic joint along Z."""
        self._test_joint_limits(joint_type="d6_prismatic", motion_axis=2)

    def test_joint_limits_ball_x(self):
        """Joint limit is enforced for a ball joint about X."""
        self._test_joint_limits(joint_type="ball", motion_axis=0)

    def test_joint_limits_ball_y(self):
        """Joint limit is enforced for a ball joint about Y."""
        self._test_joint_limits(joint_type="ball", motion_axis=1)

    def test_joint_limits_ball_z(self):
        """Joint limit is enforced for a ball joint about Z."""
        self._test_joint_limits(joint_type="ball", motion_axis=2)


class TestJointFrictionBase(TestJointSteppingBase):
    def _test_joint_friction(self, joint_type: str, motion_axis: int):
        """Verify that joint friction reduces velocity compared to the frictionless case.

        Applies a force to a joint with and without friction, steps once, and
        checks that the friction case produces a lower velocity.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                ``"d6_prismatic"``, or ``"ball"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """
        dt = 0.01
        force = 100.0
        body_inertia = 4.0
        friction_value = 50.0

        sim_no_friction = self._build_model(
            armature=0.0,
            body_inertia=body_inertia,
            joint_type=joint_type,
            body_mass=body_inertia,
            motion_axis=motion_axis,
            friction=0.0,
        )
        sim_with_friction = self._build_model(
            armature=0.0,
            body_inertia=body_inertia,
            joint_type=joint_type,
            body_mass=body_inertia,
            motion_axis=motion_axis,
            friction=friction_value,
        )

        # For multi-DOF joints (ball), apply force only on the motion_axis DOF.
        num_dof = sim_no_friction.state_in.joint_qd.numpy().shape[0]
        force_arr = self._make_force_array(force, joint_type, motion_axis, num_dof)
        qd_index = self._qd_index(joint_type, motion_axis)

        # Apply the same force to both and step.
        sim_no_friction.control.joint_f.assign(force_arr)
        sim_no_friction.solver.step(
            state_in=sim_no_friction.state_in,
            state_out=sim_no_friction.state_out,
            control=sim_no_friction.control,
            dt=dt,
            contacts=None,
        )
        sim_no_friction.state_in, sim_no_friction.state_out = sim_no_friction.state_out, sim_no_friction.state_in
        qd_no_friction = sim_no_friction.state_in.joint_qd.numpy()[qd_index]

        sim_with_friction.control.joint_f.assign(force_arr)
        sim_with_friction.solver.step(
            state_in=sim_with_friction.state_in,
            state_out=sim_with_friction.state_out,
            control=sim_with_friction.control,
            dt=dt,
            contacts=None,
        )
        sim_with_friction.state_in, sim_with_friction.state_out = (
            sim_with_friction.state_out,
            sim_with_friction.state_in,
        )
        qd_with_friction = sim_with_friction.state_in.joint_qd.numpy()[qd_index]

        # Both should be positive (force is positive).
        self.assertGreater(qd_no_friction, 0.0)
        self.assertGreater(qd_with_friction, 0.0)

        # Friction should reduce the velocity.
        self.assertGreater(qd_no_friction, qd_with_friction)

    def test_joint_friction_revolute_x(self):
        """Joint friction reduces velocity for a revolute joint about X."""
        self._test_joint_friction(joint_type="revolute", motion_axis=0)

    def test_joint_friction_revolute_y(self):
        """Joint friction reduces velocity for a revolute joint about Y."""
        self._test_joint_friction(joint_type="revolute", motion_axis=1)

    def test_joint_friction_revolute_z(self):
        """Joint friction reduces velocity for a revolute joint about Z."""
        self._test_joint_friction(joint_type="revolute", motion_axis=2)

    def test_joint_friction_prismatic_x(self):
        """Joint friction reduces velocity for a prismatic joint along X."""
        self._test_joint_friction(joint_type="prismatic", motion_axis=0)

    def test_joint_friction_prismatic_y(self):
        """Joint friction reduces velocity for a prismatic joint along Y."""
        self._test_joint_friction(joint_type="prismatic", motion_axis=1)

    def test_joint_friction_prismatic_z(self):
        """Joint friction reduces velocity for a prismatic joint along Z."""
        self._test_joint_friction(joint_type="prismatic", motion_axis=2)

    def test_joint_friction_d6_revolute_x(self):
        """Joint friction reduces velocity for a D6 revolute joint about X."""
        self._test_joint_friction(joint_type="d6_revolute", motion_axis=0)

    def test_joint_friction_d6_revolute_y(self):
        """Joint friction reduces velocity for a D6 revolute joint about Y."""
        self._test_joint_friction(joint_type="d6_revolute", motion_axis=1)

    def test_joint_friction_d6_revolute_z(self):
        """Joint friction reduces velocity for a D6 revolute joint about Z."""
        self._test_joint_friction(joint_type="d6_revolute", motion_axis=2)

    def test_joint_friction_d6_prismatic_x(self):
        """Joint friction reduces velocity for a D6 prismatic joint along X."""
        self._test_joint_friction(joint_type="d6_prismatic", motion_axis=0)

    def test_joint_friction_d6_prismatic_y(self):
        """Joint friction reduces velocity for a D6 prismatic joint along Y."""
        self._test_joint_friction(joint_type="d6_prismatic", motion_axis=1)

    def test_joint_friction_d6_prismatic_z(self):
        """Joint friction reduces velocity for a D6 prismatic joint along Z."""
        self._test_joint_friction(joint_type="d6_prismatic", motion_axis=2)

    def test_joint_friction_ball_x(self):
        """Joint friction reduces velocity for a ball joint about X."""
        self._test_joint_friction(joint_type="ball", motion_axis=0)

    def test_joint_friction_ball_y(self):
        """Joint friction reduces velocity for a ball joint about Y."""
        self._test_joint_friction(joint_type="ball", motion_axis=1)

    def test_joint_friction_ball_z(self):
        """Joint friction reduces velocity for a ball joint about Z."""
        self._test_joint_friction(joint_type="ball", motion_axis=2)


class TestJointLimitMuJoCo(TestJointLimitBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            iterations=1,
            ls_iterations=1,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


class TestJointLimitFeatherstone(TestJointLimitBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverFeatherstone(
            model,
            angular_damping=0.0,
        )


class TestJointArmatureMuJoCo(TestJointArmatureBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            iterations=1,
            ls_iterations=1,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


class TestJointArmatureFeatherstone(TestJointArmatureBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverFeatherstone(
            model,
            angular_damping=0.0,
        )


class TestJointFrictionMuJoCo(TestJointFrictionBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            iterations=10,
            ls_iterations=10,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


class TestJointDriveBase(TestJointSteppingBase):
    def _drive_force(self, target_ke: float, target_kd: float, q: float, qd: float, target_pos: float) -> float:
        """Compute the drive force for a position-controlled joint.

        Args:
            target_ke: Position stiffness [N*m/rad or N/m].
            target_kd: Velocity damping [N*m*s/rad or N*s/m].
            q: Current joint position [m or rad].
            qd: Current joint velocity [m/s or rad/s].
            target_pos: Target position [m or rad].
        """
        return target_ke * (target_pos - q) + target_kd * (0.0 - qd)

    def _expected_drive_qd(
        self, target_ke: float, target_kd: float, q: float, qd: float, target_pos: float, dt: float, inertia: float
    ) -> float:
        """Compute the expected velocity after one step with a position drive.

        Args:
            target_ke: Position stiffness [N*m/rad or N/m].
            target_kd: Velocity damping [N*m*s/rad or N*s/m].
            q: Current joint position [m or rad].
            qd: Current joint velocity [m/s or rad/s].
            target_pos: Target position [m or rad].
            dt: Time step [s].
            inertia: Effective inertia of the body [kg*m^2 or kg].
        """
        force = self._drive_force(target_ke, target_kd, q, qd, target_pos)
        return qd + dt * force / inertia

    def _test_joint_drive(self, joint_type: str, motion_axis: int):
        """Verify that the joint drive produces the expected velocity.

        Builds a joint with a position drive (target_ke, target_kd, target_pos),
        starts from a displaced position, and checks that the resulting velocity
        matches the analytical expectation.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                ``"d6_prismatic"``, or ``"ball"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """
        dt = 0.01
        body_inertia = 4.0
        drive_ke = 100.0
        drive_kd = 10.0
        target_pos = 1.0
        start_q = 0.0
        start_qd = 0.0

        sim = self._build_model(
            armature=0.0,
            body_inertia=body_inertia,
            joint_type=joint_type,
            body_mass=body_inertia,
            motion_axis=motion_axis,
            target_ke=drive_ke,
            target_kd=drive_kd,
            target_pos=target_pos,
        )
        num_dof = sim.state_in.joint_qd.numpy().shape[0]
        qd_index = self._qd_index(joint_type, motion_axis)
        sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
        sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))

        expected_qd = self._expected_drive_qd(drive_ke, drive_kd, start_q, start_qd, target_pos, dt, body_inertia)

        sim.solver.step(
            state_in=sim.state_in,
            state_out=sim.state_out,
            control=sim.control,
            dt=dt,
            contacts=None,
        )
        sim.state_in, sim.state_out = sim.state_out, sim.state_in
        measured_qd = sim.state_in.joint_qd.numpy()[qd_index]
        self.assertAlmostEqual(measured_qd, expected_qd, places=4)

        # Higher drive stiffness should produce higher velocity toward target.
        higher_ke = drive_ke * 3.0
        sim.model.joint_target_ke.assign(np.full(num_dof, higher_ke, dtype=np.float32))
        sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
        sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        expected_qd2 = self._expected_drive_qd(higher_ke, drive_kd, start_q, start_qd, target_pos, dt, body_inertia)

        sim.solver.step(
            state_in=sim.state_in,
            state_out=sim.state_out,
            control=sim.control,
            dt=dt,
            contacts=None,
        )
        sim.state_in, sim.state_out = sim.state_out, sim.state_in
        measured_qd2 = sim.state_in.joint_qd.numpy()[qd_index]
        self.assertAlmostEqual(measured_qd2, expected_qd2, places=4)
        self.assertGreater(measured_qd2, measured_qd)

        # Change target_pos at runtime and verify the drive responds.
        # Re-derive control from model so that control.joint_target_pos picks up the new value.
        changed_target_pos = -1.0
        sim.model.joint_target_ke.assign(np.full(num_dof, drive_ke, dtype=np.float32))
        sim.model.joint_target_pos.assign(np.full(num_dof, changed_target_pos, dtype=np.float32))
        sim.control = sim.model.control()
        sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
        sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        expected_qd_tp = self._expected_drive_qd(
            drive_ke, drive_kd, start_q, start_qd, changed_target_pos, dt, body_inertia
        )

        sim.solver.step(
            state_in=sim.state_in,
            state_out=sim.state_out,
            control=sim.control,
            dt=dt,
            contacts=None,
        )
        sim.state_in, sim.state_out = sim.state_out, sim.state_in
        measured_qd_tp = sim.state_in.joint_qd.numpy()[qd_index]
        self.assertAlmostEqual(measured_qd_tp, expected_qd_tp, places=4)
        # Original target_pos was positive, changed is negative, so velocity should flip sign.
        self.assertLess(measured_qd_tp, 0.0)
        self.assertGreater(measured_qd, 0.0)

        # Set target_pos via Control instead of Model and verify the drive responds.
        ctrl_target_pos = 2.0
        sim.control.joint_target_pos.assign(np.full(num_dof, ctrl_target_pos, dtype=np.float32))
        sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
        sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        expected_qd_ctrl = self._expected_drive_qd(
            drive_ke, drive_kd, start_q, start_qd, ctrl_target_pos, dt, body_inertia
        )

        sim.solver.step(
            state_in=sim.state_in,
            state_out=sim.state_out,
            control=sim.control,
            dt=dt,
            contacts=None,
        )
        sim.state_in, sim.state_out = sim.state_out, sim.state_in
        measured_qd_ctrl = sim.state_in.joint_qd.numpy()[qd_index]
        self.assertAlmostEqual(measured_qd_ctrl, expected_qd_ctrl, places=4)
        # ctrl_target_pos (2.0) is further from start_q than original target_pos (1.0),
        # so the drive force and resulting velocity should be larger.
        self.assertGreater(measured_qd_ctrl, measured_qd)

        # Test effort limit: a small effort limit should cap the drive force.
        # Featherstone does not support effort_limit; only MuJoCo does.
        if isinstance(sim.solver, SolverMuJoCo):
            small_effort = 10.0
            sim.model.joint_target_ke.assign(np.full(num_dof, drive_ke, dtype=np.float32))
            sim.model.joint_target_pos.assign(np.full(num_dof, target_pos, dtype=np.float32))
            sim.model.joint_effort_limit.assign(np.full(num_dof, small_effort, dtype=np.float32))
            sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
            sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))
            sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

            unclamped_force = self._drive_force(drive_ke, drive_kd, start_q, start_qd, target_pos)
            clamped_force = unclamped_force
            if unclamped_force > small_effort:
                clamped_force = small_effort
            elif unclamped_force < -small_effort:
                clamped_force = -small_effort
            self.assertGreater(abs(unclamped_force), abs(clamped_force))
            expected_qd3 = start_qd + dt * clamped_force / body_inertia

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd3 = sim.state_in.joint_qd.numpy()[qd_index]

            # The effort-limited velocity should be less than the unlimited one.
            self.assertGreater(measured_qd3, 0.0)
            self.assertGreater(measured_qd, measured_qd3)
            self.assertAlmostEqual(measured_qd3, expected_qd3, places=4)

        # Apply a joint force with ke=kd=0 to verify control.joint_f works without drive interference.
        joint_force = 50.0
        sim.model.joint_target_ke.assign(np.full(num_dof, 0.0, dtype=np.float32))
        sim.model.joint_target_kd.assign(np.full(num_dof, 0.0, dtype=np.float32))
        sim.control = sim.model.control()
        sim.control.joint_f.assign(self._make_force_array(joint_force, joint_type, motion_axis, num_dof))
        sim.state_in.joint_q.assign(self._make_dof_array(start_q, joint_type, motion_axis))
        sim.state_in.joint_qd.assign(self._make_dof_array(start_qd, joint_type, motion_axis))
        sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)

        expected_qd_f = start_qd + dt * joint_force / body_inertia

        sim.solver.step(
            state_in=sim.state_in,
            state_out=sim.state_out,
            control=sim.control,
            dt=dt,
            contacts=None,
        )
        sim.state_in, sim.state_out = sim.state_out, sim.state_in
        measured_qd_f = sim.state_in.joint_qd.numpy()[qd_index]
        self.assertAlmostEqual(measured_qd_f, expected_qd_f, places=4)

    def test_joint_drive_revolute_x(self):
        """Joint drive produces expected velocity for a revolute joint about X."""
        self._test_joint_drive(joint_type="revolute", motion_axis=0)

    def test_joint_drive_revolute_y(self):
        """Joint drive produces expected velocity for a revolute joint about Y."""
        self._test_joint_drive(joint_type="revolute", motion_axis=1)

    def test_joint_drive_revolute_z(self):
        """Joint drive produces expected velocity for a revolute joint about Z."""
        self._test_joint_drive(joint_type="revolute", motion_axis=2)

    def test_joint_drive_prismatic_x(self):
        """Joint drive produces expected velocity for a prismatic joint along X."""
        self._test_joint_drive(joint_type="prismatic", motion_axis=0)

    def test_joint_drive_prismatic_y(self):
        """Joint drive produces expected velocity for a prismatic joint along Y."""
        self._test_joint_drive(joint_type="prismatic", motion_axis=1)

    def test_joint_drive_prismatic_z(self):
        """Joint drive produces expected velocity for a prismatic joint along Z."""
        self._test_joint_drive(joint_type="prismatic", motion_axis=2)

    def test_joint_drive_d6_revolute_x(self):
        """Joint drive produces expected velocity for a D6 revolute joint about X."""
        self._test_joint_drive(joint_type="d6_revolute", motion_axis=0)

    def test_joint_drive_d6_revolute_y(self):
        """Joint drive produces expected velocity for a D6 revolute joint about Y."""
        self._test_joint_drive(joint_type="d6_revolute", motion_axis=1)

    def test_joint_drive_d6_revolute_z(self):
        """Joint drive produces expected velocity for a D6 revolute joint about Z."""
        self._test_joint_drive(joint_type="d6_revolute", motion_axis=2)

    def test_joint_drive_d6_prismatic_x(self):
        """Joint drive produces expected velocity for a D6 prismatic joint along X."""
        self._test_joint_drive(joint_type="d6_prismatic", motion_axis=0)

    def test_joint_drive_d6_prismatic_y(self):
        """Joint drive produces expected velocity for a D6 prismatic joint along Y."""
        self._test_joint_drive(joint_type="d6_prismatic", motion_axis=1)

    def test_joint_drive_d6_prismatic_z(self):
        """Joint drive produces expected velocity for a D6 prismatic joint along Z."""
        self._test_joint_drive(joint_type="d6_prismatic", motion_axis=2)

    def test_joint_drive_ball_x(self):
        """Joint drive produces expected velocity for a ball joint about X."""
        self._test_joint_drive(joint_type="ball", motion_axis=0)

    def test_joint_drive_ball_y(self):
        """Joint drive produces expected velocity for a ball joint about Y."""
        self._test_joint_drive(joint_type="ball", motion_axis=1)

    def test_joint_drive_ball_z(self):
        """Joint drive produces expected velocity for a ball joint about Z."""
        self._test_joint_drive(joint_type="ball", motion_axis=2)


class TestJointDriveFeatherstone(TestJointDriveBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverFeatherstone(
            model,
            angular_damping=0.0,
        )


class TestJointDriveMuJoCo(TestJointDriveBase, unittest.TestCase):
    def _create_solver(self, model):
        return SolverMuJoCo(
            model,
            iterations=1,
            ls_iterations=1,
            disable_contacts=True,
            use_mujoco_cpu=False,
            integrator="euler",
        )


# Note: XPBD, SemiImplicit, and Featherstone do not support joint friction or effort_limit.
# Note: XPBD and SemiImplicit do not support armature or joint limits.


if __name__ == "__main__":
    unittest.main(verbosity=2)
