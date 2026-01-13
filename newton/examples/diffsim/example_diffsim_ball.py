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
# Example Diffsim Ball
#
# Shows how to use Newton to optimize the initial velocity of a particle
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the initial velocity, followed by
# a simple gradient-descent optimization step.
#
# Command: python -m newton.examples diffsim_ball
#
###########################################################################
import numpy as np
import warp as wp
from warp.render import bourke_color_map

import newton
import newton.examples
from newton.tests.unittest_utils import assert_np_equal


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    # distance to target
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, viewer, args=None, verbose=False):
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 36
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        self.train_iter = 0
        self.train_rate = 0.02
        self.target = (0.0, -2.0, 1.5)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_history = []

        self.viewer = viewer
        self.viewer.show_particles = True

        # setup simulation scene
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)

        scene.add_particle(pos=wp.vec3(0.0, -0.5, 1.0), vel=wp.vec3(0.0, 5.0, -5.0), mass=1.0)

        # add wall and ground plane
        ke = 1.0e4
        kf = 0.0
        kd = 1.0e1
        mu = 0.2

        scene.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
            hx=1.0,
            hy=0.25,
            hz=1.0,
            cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu),
        )

        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = scene.finalize(requires_grad=True)

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu
        self.model.soft_contact_restitution = 1.0

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # allocate sim states, initialize control and one-shot contacts (valid for simple collisions against constant plane)
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()

        # Create collision pipeline (default: unified)
        self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
        if self.collision_pipeline is not None:
            self.collision_pipeline.soft_contact_margin = 10.0
        self.contacts = self.model.collide(
            self.states[0], collision_pipeline=self.collision_pipeline, soft_contact_margin=10.0
        )

        self.viewer.set_model(self.model)

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
        wp.launch(loss_kernel, dim=1, inputs=[self.states[-1].particle_q, self.target, self.loss])

        return self.loss

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
            print(f"    x: {x} g: {x.grad}")

        # gradient descent step
        wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, self.train_rate])

        # clear grads for next iteration
        self.tape.zero()

        self.train_iter += 1
        self.loss_history.append(self.loss.numpy()[0])

    def test_final(self):
        x_grad_numeric, x_grad_analytic = self.check_grad()
        assert_np_equal(x_grad_numeric, x_grad_analytic, tol=5e-2)
        assert all(np.array(self.loss_history) < 10.0)
        # skip the last loss because there could be some bouncing around the optimum
        assert all(np.diff(self.loss_history[:-1]) < -1e-3)

    def render(self):
        if self.frame > 0 and self.train_iter % 16 != 0:
            return

        # draw trajectory
        traj_verts = [self.states[0].particle_q.numpy()[0].tolist()]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(state.particle_q.numpy()[0].tolist())

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_scalar("/loss", self.loss.numpy()[0])
            self.viewer.log_state(state)
            self.viewer.log_contacts(self.contacts, state)
            self.viewer.log_shapes(
                "/target",
                newton.GeoType.BOX,
                (0.1, 0.1, 0.1),
                wp.array([wp.transform(self.target, wp.quat_identity())], dtype=wp.transform),
                wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3),
            )
            self.viewer.log_lines(
                f"/traj_{self.train_iter - 1}",
                wp.array(traj_verts[0:-1], dtype=wp.vec3),
                wp.array(traj_verts[1:], dtype=wp.vec3),
                bourke_color_map(0.0, 7.0, self.loss.numpy()[0]),
            )
            self.viewer.end_frame()

            self.frame += 1

    def check_grad(self):
        param = self.states[0].particle_qd

        # initial value
        x_c = param.numpy().flatten()

        # compute numeric gradient
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

        # reset initial state
        param.assign(x_c)

        # compute analytic gradient
        tape = wp.Tape()
        with tape:
            l = self.forward()

        tape.backward(l)

        x_grad_analytic = param.grad.numpy()[0].copy()

        print(f"numeric grad: {x_grad_numeric}")
        print(f"analytic grad: {x_grad_analytic}")

        tape.zero()

        return x_grad_numeric, x_grad_analytic


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, args=args, verbose=args.verbose)

    # Check gradients
    example.check_grad()

    # Run example
    newton.examples.run(example, args)
