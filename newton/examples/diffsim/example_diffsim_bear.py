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
# Example Walker
#
# Trains a tetrahedral mesh quadruped to run. Feeds 8 time-varying input
# phases as inputs into a single layer fully connected network with a tanh
# activation function. Interprets the output of the network as tet
# activations, which are fed into the soft mesh model. This is simulated
# forward in time and then evaluated based on the center of mass momentum
# of the mesh.
#
# Command: python -m newton.examples diffsim_bear
#
###########################################################################

import numpy as np
import warp as wp
import warp.optim
from pxr import Usd, UsdGeom

import newton
import newton.examples

PHASE_COUNT = 8
PHASE_STEP = wp.constant((2.0 * wp.pi) / PHASE_COUNT)
PHASE_FREQ = wp.constant(5.0)
ACTIVATION_STRENGTH = wp.constant(0.3)

TILE_TETS = wp.constant(8)
TILE_THREADS = 64


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    vx = com[tid][0]
    vy = com[tid][1]
    vz = com[tid][2]
    delta = wp.sqrt(vy * vy) + wp.sqrt(vz * vz) - vx

    wp.atomic_add(loss, 0, delta)


@wp.kernel
def com_kernel(velocities: wp.array(dtype=wp.vec3), n: int, com: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    v = velocities[tid]
    a = v / wp.float32(n)
    wp.atomic_add(com, 0, a)


@wp.kernel
def compute_phases(phases: wp.array(dtype=float), sim_time: float):
    tid = wp.tid()
    phases[tid] = wp.sin(PHASE_FREQ * sim_time + wp.float32(tid) * PHASE_STEP)


@wp.func
def tanh(x: float):
    return wp.tanh(x) * ACTIVATION_STRENGTH


@wp.kernel
def network(
    phases: wp.array2d(dtype=float), weights: wp.array2d(dtype=float), tet_activations: wp.array2d(dtype=float)
):
    # output tile index
    i = wp.tid()

    # GEMM
    p = wp.tile_load(phases, shape=(PHASE_COUNT, 1))
    w = wp.tile_load(weights, shape=(TILE_TETS, PHASE_COUNT), offset=(i * TILE_TETS, 0))
    out = wp.tile_matmul(w, p)

    # activation
    activations = wp.tile_map(tanh, out)
    wp.tile_store(tet_activations, activations, offset=(i * TILE_TETS, 0))


class Example:
    def __init__(self, viewer, verbose=False):
        # setup simulation parameters first
        self.fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_steps = 300
        self.sim_substeps = 80
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.verbose = verbose

        # load bear mesh
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bear.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bear"))

        tet_vertices = np.array(usd_geom.GetPointsAttr().Get())
        tet_indices = np.array(usd_geom.GetPrim().GetAttribute("tetraIndices").Get())

        quat_0 = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.HALF_PI)
        quat_1 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.HALF_PI)

        num_tets = len(tet_indices) // 4

        # setup training parameters
        self.train_iter = 0
        self.train_rate = 0.025
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.coms = []
        for _i in range(self.sim_steps):
            self.coms.append(wp.zeros(1, dtype=wp.vec3, requires_grad=True))

        # model input (Neural network)
        self.network_tiles = int(np.ceil(num_tets / TILE_TETS))
        self.phase_count = PHASE_COUNT
        self.phases = []
        for _i in range(self.sim_steps):
            self.phases.append(wp.zeros(self.phase_count, dtype=float, requires_grad=True))

        # weights matrix for linear network
        rng = np.random.default_rng(42)
        k = 1.0 / self.phase_count
        weights = rng.uniform(-np.sqrt(k), np.sqrt(k), (num_tets, self.phase_count))
        self.weights = wp.array(weights, dtype=float, requires_grad=True)

        # tanh activation layer array
        self.tet_activations = []
        for _i in range(self.sim_steps):
            self.tet_activations.append(wp.zeros(num_tets, dtype=float, requires_grad=True))

        # setup rendering
        self.viewer = viewer

        # setup simulation scene
        scene = newton.ModelBuilder()

        scene.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=quat_0 * quat_1,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=tet_vertices,
            indices=tet_indices,
            density=1.0,
            k_mu=2000.0,
            k_lambda=2000.0,
            k_damp=2.0,
            tri_ke=0.0,
            tri_ka=1e-8,
            tri_kd=0.0,
            tri_drag=0.0,
            tri_lift=0.0,
        )

        # Add ground
        ke = 2.0e3
        kd = 0.1
        kf = 10.0
        mu = 0.7
        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = scene.finalize(requires_grad=True)

        # Set soft contact parameters
        self.model.soft_contact_ke = ke
        self.model.soft_contact_kd = kd
        self.model.soft_contact_kf = kf
        self.model.soft_contact_mu = mu

        self.model.particle_radius = wp.full(self.model.particle_count, 0.05, dtype=float)

        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.solver.enable_tri_contact = False

        # allocate sim states, initialize control and contacts
        self.states = [self.model.state() for _ in range(self.sim_steps * self.sim_substeps + 1)]
        self.control = self.model.control()
        self.contacts = self.model.collide(self.states[0], soft_contact_margin=40.0)

        # optimization
        self.optimizer = wp.optim.Adam([self.weights.flatten()], lr=self.train_rate)

        # rendering
        self.viewer.set_model(self.model)

        if isinstance(self.viewer, newton.viewer.ViewerGL):
            pos = type(self.viewer.camera.pos)(0.0, -25.0, 2.0)
            self.viewer.camera.pos = pos
            self.viewer.camera.yaw = 90.0

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
        for sim_step in range(self.sim_steps):
            with wp.ScopedTimer("network", active=self.verbose):
                # build sinusoidal input phases
                wp.launch(
                    kernel=compute_phases,
                    dim=self.phase_count,
                    inputs=[self.phases[sim_step], self.sim_time],
                )

                # apply linear network with tanh activation
                wp.launch_tiled(
                    kernel=network,
                    dim=self.network_tiles,
                    inputs=[self.phases[sim_step].reshape((self.phase_count, 1)), self.weights],
                    outputs=[self.tet_activations[sim_step].reshape((self.model.tet_count, 1))],
                    block_dim=TILE_THREADS,
                )
                self.control.tet_activations = self.tet_activations[sim_step]

            with wp.ScopedTimer("simulate", active=self.verbose):
                self.simulate(sim_step)

            with wp.ScopedTimer("loss", active=self.verbose):
                # compute center of mass velocity
                wp.launch(
                    com_kernel,
                    dim=self.model.particle_count,
                    inputs=[
                        self.states[(sim_step + 1) * self.sim_substeps].particle_qd,
                        self.model.particle_count,
                        self.coms[sim_step],
                    ],
                    outputs=[],
                )
                # compute loss
                wp.launch(loss_kernel, dim=1, inputs=[self.coms[sim_step], self.loss], outputs=[])

    def simulate(self, sim_step):
        for i in range(self.sim_substeps):
            t = sim_step * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(self.states[t], self.states[t + 1], self.control, self.contacts, self.sim_dt)
            self.sim_time += self.sim_dt

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        x = self.weights

        if self.verbose:
            print(f"Train iter {self.train_iter}: {self.loss}")
            x_np = x.flatten().numpy()
            x_grad_np = x.grad.flatten().numpy()
            print(f"    x_min: {x_np.min()} x_max: {x_np.max()} g_min: {x_grad_np.min()} g_max: {x_grad_np.max()}")

        # optimization
        self.optimizer.step([x.grad.flatten()])

        # reset sim
        self.sim_time = 0.0
        self.states[0] = self.model.state(requires_grad=True)

        # clear grads and zero arrays for next iteration
        self.tape.zero()
        self.loss.zero_()
        for i in range(self.sim_steps):
            self.coms[i].zero_()

        self.train_iter += 1

    def test(self):
        pass

    def render(self):
        for i in range(self.sim_steps + 1):
            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(self.states[i * self.sim_substeps])
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
    newton.examples.run(example)
