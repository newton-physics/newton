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
# Example Diffsim Cloth
#
# Shows how to use Newton to optimize the initial velocities of a piece of
# cloth such that its center of mass hits a target after a specified time.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
# Command: python -m newton.examples diffsim_cloth
#
###########################################################################
import numpy as np
import warp as wp
import warp.render

import newton
import newton.examples
from newton.tests.unittest_utils import most


@wp.kernel
def com_kernel(positions: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    # compute center of mass
    wp.atomic_add(com, 0, positions[tid] / float(n))


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # sq. distance to target
    delta = com[0] - target

    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer, verbose=False):
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 120  # 2.0 seconds
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        # setup training parameters
        self.train_iter = 0
        self.train_rate = 5.0
        self.target = (0.0, 8.0, 0.0)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_history = []

        # setup rendering
        self.viewer = viewer

        # setup simulation scene (cloth grid)
        scene = newton.ModelBuilder()
        scene.default_particle_radius = 0.01

        dim_x = 16
        dim_y = 16
        scene.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            vel=wp.vec3(0.0, 0.1, 0.1),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -wp.pi * 0.75),
            dim_x=dim_x,
            dim_y=dim_y,
            cell_x=1.0 / dim_x,
            cell_y=1.0 / dim_y,
            mass=1.0,
            tri_ke=10000.0,
            tri_ka=10000.0,
            tri_kd=100.0,
            tri_lift=10.0,
            tri_drag=5.0,
        )

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = scene.finalize(requires_grad=True)

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.solver.enable_tri_contact = False

        # allocate sim states for trajectory (control and contacts are not used in this example)
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.contacts = None

        # rendering
        self.viewer.set_model(self.model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            pos = type(self.viewer.camera.pos)(12.5, 0.0, 2.0)
            self.viewer.camera.pos = pos

        # capture forward/backward passes
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
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

        # compute loss on final state
        self.com.zero_()
        wp.launch(
            com_kernel,
            dim=self.model.particle_count,
            inputs=[self.states[-1].particle_q, self.model.particle_count, self.com],
        )
        wp.launch(loss_kernel, dim=1, inputs=[self.com, self.target, self.loss])

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        x = self.states[0].particle_qd

        if self.verbose:
            print(f"Train iter: {self.train_iter} Loss: {self.loss}")
            x_np = x.flatten().numpy()
            x_grad_np = x.grad.flatten().numpy()
            print(f"    x_min: {x_np.min()} x_max: {x_np.max()} g_min: {x_grad_np.min()} g_max: {x_grad_np.max()}")

        # gradient descent step
        wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

        # clear grads for next iteration
        self.tape.zero()

        self.train_iter += 1
        self.loss_history.append(self.loss.numpy()[0])

    def test_final(self):
        assert all(np.array(self.loss_history) < 300.0)
        assert most(np.diff(self.loss_history[:-1]) < -1.0)

    def render(self):
        if self.frame > 0 and self.train_iter % 4 != 0:
            return

        # draw trajectory
        traj_verts = [self.states[0].particle_q.numpy().mean(axis=0)]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(state.particle_q.numpy().mean(axis=0))

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(1.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(traj_verts[0:-1], dtype=wp.vec3),
                wp.array(traj_verts[1:], dtype=wp.vec3),
                wp.render.bourke_color_map(0.0, 269.0, self.loss.numpy()[0]),
            )
            self.viewer.end_frame()

            self.frame += 1


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
