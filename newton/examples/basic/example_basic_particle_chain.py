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

###########################################################################
# Example Basic Particle Chain
#
# Shows how to use the ModelBuilder API to create a chain of 100 particles
# connected by springs. The first particle is anchored (kinematic) and the
# chain hangs down under gravity like a rope or string.
#
# Command: python -m newton.examples basic_particle_chain
#
###########################################################################

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        # particle chain parameters
        self.num_particles = 100
        particle_spacing = 0.05  # distance between particles
        particle_mass = 0.1
        particle_radius = 0.02
        spring_stiffness = 1.0e4
        spring_damping = 1.0e1
        start_height = 3.0

        builder = newton.ModelBuilder()

        # add ground plane
        builder.add_ground_plane()

        # Create the first particle (anchored/kinematic with mass=0)
        builder.add_particle(
            pos=(0.0, 0.0, start_height),
            vel=(0.0, 0.0, 0.0),
            mass=0.0,  # kinematic particle (fixed)
            radius=particle_radius,
        )

        # Create chain of dynamic particles and connect them with springs
        for i in range(1, self.num_particles):
            # particles extend horizontally along the x-axis
            builder.add_particle(
                pos=(i * particle_spacing, 0.0, start_height),
                vel=(0.0, 0.0, 0.0),
                mass=particle_mass,
                radius=particle_radius,
            )
            # connect to previous particle with a spring
            builder.add_spring(
                i=i - 1,
                j=i,
                ke=spring_stiffness,
                kd=spring_damping,
                control=0.0,
            )

        # finalize model
        self.model = builder.finalize()

        # soft contact parameters for particle-ground collision
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        self.solver = newton.solvers.SolverXPBD(self.model, iterations=10)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Create collision pipeline from command-line args
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, self.args)
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test_final(self):
        # Verify that the anchor particle (first particle) is still at rest
        newton.examples.test_particle_state(
            self.state_0,
            "anchor particle is stationary",
            lambda q, qd: wp.length(qd) < 1e-6,
            indices=[0],
        )

        # Verify all particles are above the ground
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] >= 0.0,
        )

        # Verify particles are within reasonable bounds
        p_lower = wp.vec3(-1.0, -2.0, -0.1)
        p_upper = wp.vec3(6.0, 2.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within reasonable bounds",
            lambda q, qd: newton.utils.vec_inside_limits(q, p_lower, p_upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Enable particle visualization if using GL viewer
    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    # Create viewer and run
    example = Example(viewer, args)

    newton.examples.run(example, args)
