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

"""Tests for joint armature.

Armature adds to the diagonal of the joint-space inertia matrix, so for a 1-DOF
revolute joint with applied torque tau, the acceleration should be approximately
tau / (I + armature), where I is the natural rotational inertia about the joint axis.

For a prismatic joint, the effective mass becomes (m + armature).
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


class TestJointArmatureBase:
    def _create_solver(self, model):
        raise NotImplementedError

    def _build_model(
        self,
        armature: float,
        body_inertia: float,
        joint_type: str = "revolute",
        body_mass: float = 1.0,
        motion_axis: int = 2,
    ):
        """Build a single joint attached to the world with specified armature.

        Args:
            armature: Armature value for the joint DOF [kg*m^2 or kg].
            body_inertia: Diagonal rotational inertia of the body [kg*m^2].
            joint_type: One of ``"revolute"``, ``"prismatic"``, ``"d6_revolute"``,
                or ``"d6_prismatic"``.
            body_mass: Mass of the body [kg].
            motion_axis: Joint motion axis (0=X, 1=Y, 2=Z).

        Returns:
            A :class:`Sim` containing the model, solver, states, and control.
        """
        builder = newton.ModelBuilder(gravity=0.0, up_axis=1)

        inertia_mat = wp.mat33(
            body_inertia, 0.0, 0.0,
            0.0, body_inertia, 0.0,
            0.0, 0.0, body_inertia,
        )
        body = builder.add_link(
            mass=body_mass,
            inertia=inertia_mat,
            com=wp.vec3(0.0, 0.0, 0.0),
        )
        if joint_type == "prismatic":
            joint = builder.add_joint_prismatic(
                axis=motion_axis, parent=-1, child=body,
                armature=armature, effort_limit=1e12, velocity_limit=1e12, friction=0.0,
            )
        elif joint_type == "revolute":
            joint = builder.add_joint_revolute(
                axis=motion_axis, parent=-1, child=body,
                armature=armature, effort_limit=1e12, velocity_limit=1e12, friction=0.0,
            )
        elif joint_type == "d6_prismatic" or joint_type == "d6_revolute":
            dof_cfg = newton.ModelBuilder.JointDofConfig(
                axis=motion_axis,
                armature=armature,
            )
            if joint_type == "d6_prismatic":
                joint = builder.add_joint_d6(
                    -1, body, linear_axes=[dof_cfg],
                )
            else:
                joint = builder.add_joint_d6(
                    -1, body, angular_axes=[dof_cfg],
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
                armature=armatures[i], body_inertia=effective_inertia, joint_type=joint_type,
                body_mass=effective_inertia, motion_axis=motion_axis,
            )

            # Apply a force to the joint for 1 sim step and measure the joint speed.
            sim.control.joint_f.assign(np.array([force], dtype=np.float32))
            sim.solver.step(state_in=sim.state_in, state_out=sim.state_out, control=sim.control, dt=dt, contacts=None)
            measured_qd = sim.state_out.joint_qd.numpy()[0]
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
        sims[0].solver.step(state_in=sims[0].state_in, state_out=sims[0].state_out, control=sims[0].control, dt=dt, contacts=None)

        measured_qd_changed = sims[0].state_out.joint_qd.numpy()[0]
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

# Note XPBD and SemiImplicit both document that they do not support armature.


if __name__ == "__main__":
    unittest.main(verbosity=2)
