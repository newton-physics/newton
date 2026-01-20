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
# Example Diffsim Soft Body
#
# Shows how to use Newton to optimize for the material parameters of a soft body,
# such that it bounces off the wall and floor in order to hit a target.
#
# This example uses the built-in wp.Tape() object to compute gradients of
# the distance to target (loss) w.r.t the material parameters, followed by
# a simple gradient-descent optimization step.
#
# Command: python -m newton.examples diffsim_soft_body
#
###########################################################################

import numpy as np
import warp as wp
import warp.optim

import newton
import newton.examples
from newton.tests.unittest_utils import most
from newton.utils import bourke_color_map


@wp.kernel
def assign_param(params: wp.array(dtype=wp.float32), tet_materials: wp.array2d(dtype=wp.float32)):
    tid = wp.tid()
    params_idx = 2 * wp.tid() % params.shape[0]
    tet_materials[tid, 0] = params[params_idx]
    tet_materials[tid, 1] = params[params_idx + 1]


@wp.kernel
def com_kernel(particle_q: wp.array(dtype=wp.vec3), com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    point = particle_q[tid]
    a = point / wp.float32(particle_q.shape[0])

    # Atomically add the point coordinates to the accumulator
    wp.atomic_add(com, 0, a)


@wp.kernel
def loss_kernel(
    target: wp.vec3,
    com: wp.array(dtype=wp.vec3),
    pos_error: wp.array(dtype=float),
    loss: wp.array(dtype=float),
):
    diff = com[0] - target
    pos_error[0] = wp.length(diff)
    loss[0] = wp.dot(diff, diff)


@wp.kernel
def enforce_constraint_kernel(lower_bound: wp.float32, upper_bound: wp.float32, x: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if x[tid] < lower_bound:
        x[tid] = lower_bound
    elif x[tid] > upper_bound:
        x[tid] = upper_bound


class Example:
    def __init__(self, viewer, material_behavior="anisotropic", verbose=False):
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = 60  # 1.0 seconds
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose
        self.material_behavior = material_behavior

        # setup training parameters
        self.train_iter = 0
        self.train_rate = 1e7
        self.target = wp.vec3(0.0, -1.0, 1.5)
        self.com = wp.array([wp.vec3(0.0, 0.0, 0.0)], dtype=wp.vec3, requires_grad=True)
        self.pos_error = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.loss_history = []

        # setup rendering
        self.viewer = viewer

        # Create FEM model.
        self.model = self.create_model()

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # allocate sim states for trajectory, control and contacts
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.contacts = self.model.collide(self.states[0], soft_contact_margin=0.001)

        # Initialize material parameters to be optimized from model
        if self.material_behavior == "anisotropic":
            # Different Lame parameters for each tet
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[:, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )
        else:
            # Same Lame parameters for all tets
            self.material_params = wp.array(
                self.model.tet_materials.numpy()[0, :2].flatten(),
                dtype=wp.float32,
                requires_grad=True,
            )

            # Scale learning rate for isotropic material
            scale = self.material_params.size / float(self.model.tet_count)
            self.train_rate = self.train_rate * scale

        # setup hard bounds for material parameters
        self.hard_lower_bound = wp.float32(500.0)
        self.hard_upper_bound = wp.float32(4e6)

        # Create optimizer
        self.optimizer = warp.optim.SGD(
            [self.material_params],
            lr=self.train_rate,
            nesterov=False,
        )

        # rendering
        self.viewer.set_model(self.model)

        # capture forward/backward passes
        self.capture()

    def create_model(self):
        # setup simulation scene
        scene = newton.ModelBuilder()
        scene.default_particle_radius = 0.0005

        # setup grid parameters
        cell_dim = 2
        cell_size = 0.1

        # compute particle density
        total_mass = 0.2
        num_particles = (cell_dim + 1) ** 3
        particle_mass = total_mass / num_particles
        particle_density = particle_mass / (cell_size**3)
        if self.verbose:
            print(f"Particle density: {particle_density}")

        # compute Lame parameters
        young_mod = 1.5 * 1e4
        poisson_ratio = 0.3
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

        # add soft grid to scene
        scene.add_soft_grid(
            pos=wp.vec3(-0.5 * cell_size * cell_dim, -0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 6.0, -6.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=particle_density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=0.0,
            tri_ke=1e-4,
            tri_ka=1e-4,
            tri_kd=1e-4,
            tri_drag=0.0,
            tri_lift=0.0,
            fix_bottom=False,
        )

        # add wall and ground plane to scene
        ke = 1.0e3
        kf = 0.0
        kd = 1.0e0
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

        # use `requires_grad=True` to create a model for differentiable simulation
        model = scene.finalize(requires_grad=True)

        model.soft_contact_ke = ke
        model.soft_contact_kf = kf
        model.soft_contact_kd = kd
        model.soft_contact_mu = mu
        model.soft_contact_restitution = 1.0

        return model

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
        wp.launch(
            kernel=assign_param,
            dim=self.model.tet_count,
            inputs=(self.material_params,),
            outputs=(self.model.tet_materials,),
        )

        # run simulation loop
        for sim_step in range(self.sim_steps):
            self.simulate(sim_step)

        # Update loss
        # Compute the center of mass for the last time step.
        wp.launch(
            kernel=com_kernel,
            dim=self.model.particle_count,
            inputs=(self.states[-1].particle_q,),
            outputs=(self.com,),
        )

        # calculate loss
        wp.launch(
            kernel=loss_kernel,
            dim=1,
            inputs=(
                self.target,
                self.com,
            ),
            outputs=(self.pos_error, self.loss),
        )

        return self.loss

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.contacts = self.model.collide(self.states[t], soft_contact_margin=0.001)
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        if self.verbose:
            self.log_step()

        self.optimizer.step([self.material_params.grad])

        wp.launch(
            kernel=enforce_constraint_kernel,
            dim=self.material_params.shape[0],
            inputs=(
                self.hard_lower_bound,
                self.hard_upper_bound,
            ),
            outputs=(self.material_params,),
        )

        self.loss_history.append(self.loss.numpy()[0])

        # clear grads for next iteration
        self.tape.zero()
        self.loss.zero_()
        self.com.zero_()
        self.pos_error.zero_()

        self.train_iter = self.train_iter + 1

    def log_step(self):
        x = self.material_params.numpy().reshape(-1, 2)
        x_grad = self.material_params.grad.numpy().reshape(-1, 2)

        print(f"Train iter: {self.train_iter} Loss: {self.loss.numpy()[0]}")

        print(f"Pos error: {self.pos_error.numpy()[0]}")

        print(
            f"Max Mu: {np.max(x[:, 0])}, Min Mu: {np.min(x[:, 0])}, "
            f"Max Lambda: {np.max(x[:, 1])}, Min Lambda: {np.min(x[:, 1])}"
        )

        print(
            f"Max Mu Grad: {np.max(x_grad[:, 0])}, Min Mu Grad: {np.min(x_grad[:, 0])}, "
            f"Max Lambda Grad: {np.max(x_grad[:, 1])}, Min Lambda Grad: {np.min(x_grad[:, 1])}"
        )

    def test_final(self):
        assert all(np.array(self.loss_history) < 0.8)
        assert most(np.diff(self.loss_history) < -0.0, min_ratio=0.8)

    def render(self):
        if self.frame > 0 and self.train_iter % 10 != 0:
            return

        # draw trajectory
        traj_verts = [np.mean(self.states[0].particle_q.numpy(), axis=0).tolist()]

        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]
            traj_verts.append(np.mean(state.particle_q.numpy(), axis=0).tolist())

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
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
                bourke_color_map(0.0, self.loss_history[0], self.loss_history[-1]),
            )
            self.viewer.end_frame()

            self.frame += 1


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--material_behavior",
        default="anisotropic",
        choices=["anisotropic", "isotropic"],
        help="Set material behavior to be Anisotropic or Isotropic.",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, material_behavior=args.material_behavior, verbose=args.verbose)

    # Run example
    newton.examples.run(example, args)
