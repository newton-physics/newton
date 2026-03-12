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

"""Tests for joint stepping: armature and limits.

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
                effort_limit=1e12,
                velocity_limit=1e12,
                friction=0.0,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                axis=motion_axis,
                parent=-1,
                child=body,
                armature=armature,
                effort_limit=1e12,
                velocity_limit=1e12,
                friction=0.0,
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )
        elif joint_type == "d6_prismatic" or joint_type == "d6_revolute":
            dof_kwargs = {}
            if limit_lower is not None:
                dof_kwargs["limit_lower"] = limit_lower
            if limit_upper is not None:
                dof_kwargs["limit_upper"] = limit_upper
            dof_cfg = newton.ModelBuilder.JointDofConfig(
                axis=motion_axis,
                armature=armature,
                **dof_kwargs,
            )
            if joint_type == "d6_prismatic":
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    linear_axes=[dof_cfg],
                )
            else:
                joint = builder.add_joint_d6(
                    -1,
                    body,
                    angular_axes=[dof_cfg],
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


class TestJointArmatureBase(TestJointSteppingBase):
    def _test_armature_reduces_joint_speed(self, joint_type: str, motion_axis: int):
        """Applying the same force with higher armature yields lower joint speed.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                or ``"d6_prismatic"``.
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
            sim.control.joint_f.assign(np.array([force], dtype=np.float32))
            sim.solver.step(state_in=sim.state_in, state_out=sim.state_out, control=sim.control, dt=dt, contacts=None)
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[0]
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
        sims[0].model.joint_armature.assign(np.array([armature_changed], dtype=np.float32))
        sims[0].state_in.joint_q.assign(np.array([0.0], dtype=np.float32))
        sims[0].state_in.joint_qd.assign(np.array([0.0], dtype=np.float32))

        sims[0].solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
        sims[0].control.joint_f.assign(np.array([force], dtype=np.float32))
        sims[0].solver.step(
            state_in=sims[0].state_in, state_out=sims[0].state_out, control=sims[0].control, dt=dt, contacts=None
        )
        sims[0].state_in, sims[0].state_out = sims[0].state_out, sims[0].state_in

        measured_qd_changed = sims[0].state_in.joint_qd.numpy()[0]
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


class TestJointLimitBase(TestJointSteppingBase):
    def _expected_limit_qd(self, ke: float, kd: float, q: float, qd: float, limit_q: float, dt: float) -> float:
        """Compute the expected velocity after one step with a limit penalty force.

        Args:
            ke: Limit stiffness [N/m or N*m/rad].
            kd: Limit damping [N*s/m or N*m*s/rad].
            q: Current joint position [m or rad].
            qd: Current joint velocity [m/s or rad/s].
            limit_q: The limit value being violated [m or rad].
            dt: Time step [s].
        """
        return dt * (-ke * (q - limit_q) - kd * qd)

    def _test_joint_limits(self, joint_type: str, motion_axis: int):
        """Verify that the penalty force from a joint limit breach produces the expected velocity.

        Starts the joint past its limit with zero velocity and steps once.
        The solver applies a spring-damper penalty force proportional to the
        penetration depth (``limit_ke * penetration``).  The test checks that
        the resulting ``joint_qd`` matches the analytically expected value.

        Args:
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                or ``"d6_prismatic"``.
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).
        """
        dt = 0.01

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
                body_inertia=1.0,
                joint_type=joint_type,
                body_mass=1.0,
                motion_axis=motion_axis,
                limit_lower=initial_limits_for_newton[i][0],
                limit_upper=initial_limits_for_newton[i][1],
            )
            sim.state_in.joint_q.assign(np.array([initial_start_qs[i]], dtype=np.float32))
            sim.state_in.joint_qd.assign(np.array([initial_start_qds[i]], dtype=np.float32))

            # Compute the expected velocity after applying the penalty force to repair the limit breach.
            ke = sim.model.joint_limit_ke.numpy()[0]
            kd = sim.model.joint_limit_kd.numpy()[0]
            expected_qd = self._expected_limit_qd(
                ke, kd, initial_start_qs[i], initial_start_qds[i], initial_limit_qs[i], dt
            )

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[0]
            self.assertAlmostEqual(measured_qd, expected_qd, places=4)

            # Change the lower and upper limit at runtime, reset state, and verify the new limit is enforced.
            # Need to be careful here because setting model.joint_limit_upper to None does not work with SolverMujoco.
            # https://github.com/newton-physics/newton/issues/2072
            if i == 0:
                sim.model.joint_limit_upper.assign(np.array([changed_limits_for_newton[i][1]], dtype=np.float32))
            elif i == 1:
                sim.model.joint_limit_lower.assign(np.array([changed_limits_for_newton[i][0]], dtype=np.float32))
            else:
                raise ValueError(f"Unexpected index {i}")

            sim.state_in.joint_q.assign(np.array([changed_start_qs[i]], dtype=np.float32))
            sim.state_in.joint_qd.assign(np.array([changed_start_qds[i]], dtype=np.float32))
            sim.solver.notify_model_changed(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
            expected_qd = self._expected_limit_qd(
                ke, kd, changed_start_qs[i], changed_start_qds[i], changed_limit_qs[i], dt
            )

            sim.solver.step(
                state_in=sim.state_in,
                state_out=sim.state_out,
                control=sim.control,
                dt=dt,
                contacts=None,
            )
            sim.state_in, sim.state_out = sim.state_out, sim.state_in
            measured_qd = sim.state_in.joint_qd.numpy()[0]
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


# MuJoCo limits are soft constraints and cannot be tested with exact clamping assertions.


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


# Note: XPBD and SemiImplicit both document that they do not support
# armature or joint limits.


if __name__ == "__main__":
    unittest.main(verbosity=2)
