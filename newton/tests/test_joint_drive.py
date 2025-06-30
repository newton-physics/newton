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

import unittest

import warp as wp

import newton
from newton.solvers import MuJoCoSolver


class TestJointDrive(unittest.TestCase):
    def run_test_spring_linear(self, worldUpAxis: int, jointMotionAxis: int):
        nbenvs = 1
        body_masses = [10.0, 15.0]
        body_coms = [wp.vec3(-1.0, 2.0, 3.0), wp.vec3(2.0, -0.5, 6.0)]
        body_inertias = [
            wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
            wp.mat33(6.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 10.0),
        ]
        joint_start_positions = [100.0, 67.0]
        joint_start_velocities = [6.0, 14.0]
        joint_drive_target_positions = [200.0, 300.0]
        joint_drive_stiffnesses = [100.0, 220.0]
        joint_drive_dampings = [10.0, 18.0]

        gravity = 8.0
        dt = 0.01

        worldBuilder = newton.ModelBuilder(gravity=gravity, up_axis=worldUpAxis)
        for i in range(0, nbenvs):
            body_mass = body_masses[i]
            body_com = body_coms[i]
            body_inertia = body_inertias[i]
            drive_stiffness = joint_drive_stiffnesses[i]
            joint_drive_target_position = joint_drive_target_positions[i]
            joint_drive_damping = joint_drive_dampings[i]
            joint_start_position = joint_start_positions[i]
            joint_start_velocity = joint_start_velocities[i]

            # Create a single body jointed to the world with a prismatic joint
            # Make sure that we use the mass properties specified here by setting shape density to 0.0
            environmentBuilder = newton.ModelBuilder(gravity=gravity, up_axis=worldUpAxis)
            bodyIndex = environmentBuilder.add_body(mass=body_mass, I_m=body_inertia, armature=0.0, com=body_com)
            environmentBuilder.add_shape_sphere(
                radius=1.0, body=bodyIndex, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
            )
            environmentBuilder.add_joint_prismatic(
                axis=jointMotionAxis,
                parent=-1,
                child=bodyIndex,
                target=joint_drive_target_position,
                target_ke=drive_stiffness,
                target_kd=joint_drive_damping,
                armature=0.0,
                effort_limit=1000000000000.0,
                velocity_limit=100000000000000000.0,
                friction=0.0,
            )
            worldBuilder.add_builder(environmentBuilder)

            # Set the start pos and vel of the dof.
            worldBuilder.joint_q[i] = joint_start_position
            worldBuilder.joint_qd[i] = joint_start_velocity

        # Create the MujocoSolver instance
        model = worldBuilder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)
        newton.sim.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = MuJoCoSolver(model, iterations=1, ls_iterations=1, disable_contacts=True, use_mujoco=False)

        # Compute the expected velocity outcome after a single sim step.
        vNew = [0.0] * nbenvs
        g = 0.0
        if worldUpAxis == jointMotionAxis:
            g = gravity
        for i in range(0, nbenvs):
            ke = joint_drive_stiffnesses[i]
            kd = joint_drive_dampings[i]
            xT = joint_drive_target_positions[i]
            M = body_masses[i]
            x = joint_start_positions[i]
            v = joint_start_velocities[i]
            F = ke * (xT - x) - kd * v + M * g
            vNew[i] = v + F * dt / M

        # Perform 1 sim step.
        solver.step(model=model, state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nbenvs):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)

    def run_test_spring_angular(self, worldUpAxis: int, jointMotionAxis: int):
        nbenvs = 1
        body_masses = [10.0, 15.0]
        body_coms = [wp.vec3(-1.0, 2.0, 3.0), wp.vec3(2.0, -0.5, 6.0)]
        body_inertias = [
            wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
            wp.mat33(6.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 10.0),
        ]
        joint_start_positions = [wp.pi * 0.15, wp.pi * 0.5]
        joint_start_velocities = [wp.pi * 0.12, wp.pi * 0.18]
        joint_drive_target_positions = [wp.pi * 0.5, -wp.pi * 0.25]
        joint_drive_stiffnesses = [100.0, 220.0]
        joint_drive_dampings = [10.0, 18.0]

        gravity = 0.0
        dt = 0.01

        worldBuilder = newton.ModelBuilder(gravity=gravity, up_axis=worldUpAxis)
        for i in range(0, nbenvs):
            body_mass = body_masses[i]
            body_com = body_coms[i]
            body_inertia = body_inertias[i]
            drive_stiffness = joint_drive_stiffnesses[i]
            joint_drive_target_position = joint_drive_target_positions[i]
            joint_drive_damping = joint_drive_dampings[i]
            joint_start_position = joint_start_positions[i]
            joint_start_velocity = joint_start_velocities[i]

            # Create a single body jointed to the world with a prismatic joint
            # Make sure that we use the mass properties specified here by setting shape density to 0.0
            environmentBuilder = newton.ModelBuilder(gravity=gravity, up_axis=worldUpAxis)
            bodyIndex = environmentBuilder.add_body(mass=body_mass, I_m=body_inertia, armature=0.0, com=body_com)
            environmentBuilder.add_shape_sphere(
                radius=1.0, body=bodyIndex, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
            )
            environmentBuilder.add_joint_prismatic(
                axis=jointMotionAxis,
                parent=-1,
                child=bodyIndex,
                target=joint_drive_target_position,
                target_ke=drive_stiffness,
                target_kd=joint_drive_damping,
                armature=0.0,
                effort_limit=1000000000000.0,
                velocity_limit=100000000000000000.0,
                friction=0.0,
            )
            worldBuilder.add_builder(environmentBuilder)

            # Set the start pos and vel of the dof.
            worldBuilder.joint_q[i] = joint_start_position
            worldBuilder.joint_qd[i] = joint_start_velocity

        # Create the MujocoSolver instance
        model = worldBuilder.finalize()
        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.collide(state_in)
        newton.sim.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        solver = MuJoCoSolver(model, iterations=1, ls_iterations=1, disable_contacts=True, use_mujoco=False)

        # Compute the expected velocity outcome after a single sim step.
        vNew = [0.0] * nbenvs
        g = 0.0
        if worldUpAxis == jointMotionAxis:
            g = gravity
        for i in range(0, nbenvs):
            ke = joint_drive_stiffnesses[i]
            kd = joint_drive_dampings[i]
            xT = joint_drive_target_positions[i]
            M = body_masses[i]
            x = joint_start_positions[i]
            v = joint_start_velocities[i]
            F = ke * (xT - x) - kd * v + M * g
            vNew[i] = v + F * dt / M

        # Perform 1 sim step.
        solver.step(model=model, state_in=state_in, state_out=state_out, contacts=contacts, control=control, dt=dt)
        for i in range(0, nbenvs):
            self.assertAlmostEqual(vNew[i], state_out.joint_qd.numpy()[i], delta=0.0001)

    def test_spring_linear_upX_motionX(self):
        self.run_test_spring_linear(0, 0)

    def test_spring_linear_upX_motionY(self):
        self.run_test_spring_linear(0, 1)

    def test_spring_linear_upX_motionZ(self):
        self.run_test_spring_linear(0, 2)

    def test_spring_linear_upY_motionX(self):
        self.run_test_spring_linear(1, 0)

    def test_spring_linear__upY_motionY(self):
        self.run_test_spring_linear(1, 1)

    def test_spring_linear_upY_motionZ(self):
        self.run_test_spring_linear(1, 2)

    def test_spring_linear_upZ_motionX(self):
        self.run_test_spring_linear(1, 0)

    def test_spring_linear_upZ_motionY(self):
        self.run_test_spring_linear(1, 1)

    def test_spring_linear_upZ_motionZ(self):
        self.run_test_spring_linear(1, 2)

    def test_spring_angular_upX_motionX(self):
        self.run_test_spring_angular(0, 0)

    def test_spring_angular_upX_motionY(self):
        self.run_test_spring_angular(0, 1)

    def test_spring_angular_upX_motionZ(self):
        self.run_test_spring_angular(0, 2)

    def test_spring_angular_upY_motionX(self):
        self.run_test_spring_angular(1, 0)

    def test_spring_angular__upY_motionY(self):
        self.run_test_spring_angular(1, 1)

    def test_spring_angular_upY_motionZ(self):
        self.run_test_spring_angular(1, 2)

    def test_spring_angular_upZ_motionX(self):
        self.run_test_spring_angular(1, 0)

    def test_spring_angular_upZ_motionY(self):
        self.run_test_spring_angular(1, 1)

    def test_spring_angular_upZ_motionZ(self):
        self.run_test_spring_linear(1, 2)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
