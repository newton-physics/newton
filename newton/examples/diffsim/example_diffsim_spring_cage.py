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
# Example Diffsim Spring Cage
#
# A single particle is attached with springs to each point of a cage.
# The objective is to optimize the rest length of the springs in order
# for the particle to be pulled towards a target position.
#
# Command: python -m newton.examples diffsim_spring_cage
#
###########################################################################
import numpy as np
import warp as wp
import warp.render

import newton
import newton.examples
from newton.tests.unittest_utils import most


@wp.kernel
def compute_loss_kernel(
    pos: wp.array(dtype=wp.vec3),
    target_pos: wp.vec3,
    loss: wp.array(dtype=float),
):
    loss[0] = wp.length_sq(pos[0] - target_pos)


@wp.kernel()
def apply_gradient_kernel(
    spring_rest_lengths_grad: wp.array(dtype=float),
    train_rate: float,
    spring_rest_lengths: wp.array(dtype=float),
):
    tid = wp.tid()

    spring_rest_lengths[tid] -= spring_rest_lengths_grad[tid] * train_rate


class Example:
    def __init__(self, viewer, verbose=False):
        # setup simulation parameters first
        self.fps = 30
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 30  # 1.0 seconds
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        # setup training parameters
        self.train_iter = 0

        # Factor by which the rest lengths of the springs are adjusted after each
        # iteration, relatively to the corresponding gradients. Lower values
        # converge more slowly but have less chances to miss the local minimum.
        self.train_rate = 0.5

        # Target position that we want the main particle to reach by optimising
        # the rest lengths of the springs.
        self.target_pos = (0.375, 0.125, 0.25)

        # Initialize a loss value that will represent the distance of the main
        # particle to the target position. It needs to be defined as an array
        # so that it can be written out by a kernel.
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []

        # setup rendering
        self.viewer = viewer

        # setup simulation scene (spring cage)
        scene = newton.ModelBuilder()

        # define main particle at the origin
        particle_mass = 1.0
        scene.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), particle_mass)

        # Define the cage made of points that will be pulling our main particle
        # using springs.
        # fmt: off
        scene.add_particle(( 0.2, -0.7,  0.8), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle(( 1.1,  0.0,  0.2), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle((-1.2,  0.1,  0.1), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle(( 0.4,  0.6,  0.4), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle((-0.2,  0.7, -0.9), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle(( 0.1, -0.8, -0.8), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle((-0.8, -0.9,  0.2), (0.0, 0.0, 0.0), 0.0)
        scene.add_particle((-0.1,  1.0,  0.4), (0.0, 0.0, 0.0), 0.0)
        # fmt: on

        # Define the spring constraints between the main particle and the cage points.
        spring_elastic_stiffness = 100.0
        spring_elastic_damping = 10.0
        for i in range(1, scene.particle_count):
            scene.add_spring(0, i, spring_elastic_stiffness, spring_elastic_damping, 0)

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = scene.finalize(requires_grad=True)
        # ! manually disable particle-particle contact handling because the gradient computation
        # ! involving the eval_particle_contact kernel is not correct
        self.model.particle_grid = None

        # Use the SemiImplicit integrator for stepping through the simulation.
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # allocate sim states for trajectory (control and contacts are not used in this example)
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.contacts = None

        # rendering
        self.viewer.set_model(self.model)

        # capture forward/backward passes
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            # Capture all the kernel launches into a CUDA graph so that they can
            # all be run in a single graph launch, which helps with performance.
            with wp.ScopedCapture() as capture:
                self.forward_backward()
            self.graph = capture.graph
        else:
            self.graph = None

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.forward()
        self.tape.backward(self.loss)

    def forward(self):
        # run simulation loop
        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        wp.launch(
            compute_loss_kernel,
            dim=1,
            inputs=(
                self.states[-1].particle_q,
                self.target_pos,
            ),
            outputs=(self.loss,),
        )

        return self.loss

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def check_grad(self):
        param = self.model.spring_rest_length
        x_c = param.numpy().flatten()
        x_grad_numeric = np.zeros_like(x_c)
        for i in range(len(x_c)):
            eps = 1.0e-3
            step = np.zeros_like(x_c)
            step[i] = eps
            x_1 = x_c + step
            x_0 = x_c - step
            param.assign(x_1)
            l_1 = self.forward().numpy()[0]
            param.assign(x_0)
            l_0 = self.forward().numpy()[0]
            dldx = (l_1 - l_0) / (eps * 2.0)
            x_grad_numeric[i] = dldx
        param.assign(x_c)
        tape = wp.Tape()
        with tape:
            l = self.forward()
        tape.backward(l)
        x_grad_analytic = param.grad.numpy()[0].copy()
        return x_grad_numeric, x_grad_analytic

    def test_final(self):
        x_grad_numeric, x_grad_analytic = self.check_grad()
        assert np.allclose(x_grad_numeric, x_grad_analytic, atol=0.2)
        assert all(np.array(self.loss_history) < 0.3)
        assert most(np.diff(self.loss_history) < -0.0, min_ratio=0.5)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()
        self.loss_history.append(self.loss.numpy()[0])

        x = self.model.spring_rest_length

        if self.verbose:
            print(f"Train iter: {self.train_iter} Loss: {self.loss}")
            x_np = x.flatten().numpy()
            x_grad_np = x.grad.flatten().numpy()
            print(f"    x_min: {x_np.min()} x_max: {x_np.max()} g_min: {x_grad_np.min()} g_max: {x_grad_np.max()}")

        # gradient descent step
        wp.launch(
            apply_gradient_kernel,
            dim=self.model.spring_count,
            inputs=[
                x.grad,
                self.train_rate,
            ],
            outputs=[x],
        )

        # clear grads for next iteration
        self.tape.zero()

        self.train_iter += 1

    def render(self):
        # for interactive viewing, we just render the final state at every frame
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            start_frame = self.sim_steps
        else:
            start_frame = 0

        for i in range(start_frame, self.sim_steps + 1):
            state = self.states[i]
            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target_pos, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )

            # TODO: Draw springs inside log_state()
            q = state.particle_q.numpy()

            lines_starts = []
            lines_ends = []
            half_lengths = []
            colors = []

            for j in range(1, len(q)):
                lines_starts.append(q[0])
                lines_ends.append(q[j])
                half_lengths.append(0.5 * np.linalg.norm(q[0] - q[j]))

            min_length = min(half_lengths)
            max_length = max(half_lengths)
            for l in range(len(half_lengths)):
                color = wp.render.bourke_color_map(min_length, max_length, half_lengths[l])
                colors.append(color)

            # Draw line as sanity check
            self.viewer.log_lines(
                "/springs_lines",
                wp.array(lines_starts, dtype=wp.vec3),
                wp.array(lines_ends, dtype=wp.vec3),
                wp.array(colors, dtype=wp.vec3),
                0.02,
            )
            self.viewer.end_frame()

            self.frame += 1


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)
    if isinstance(viewer, newton.viewer.ViewerGL):
        viewer.show_particles = True

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
