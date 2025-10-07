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
# Example Diffsim Bear
#
# Trains a tetrahedral mesh bear to run. Feeds 8 time-varying input
# phases as inputs into a single-layer fully connected network with a tanh
# activation function. Interprets the output of the network as tet
# activations, which are fed into Newton's soft mesh model. This is
# simulated forward in time and then evaluated based on the center of mass
# momentum of the mesh.
#
###########################################################################

import math
import os

import numpy as np
import warp as wp
import warp.optim
from pxr import Usd, UsdGeom

import newton
import newton.examples
from newton.tests.unittest_utils import most

PHASE_COUNT = 8
PHASE_STEP = (2.0 * math.pi) / PHASE_COUNT
PHASE_FREQ = 5.0
ACTIVATION_STRENGTH = 0.3

TILE_TETS = 8
TILE_THREADS = 64

DEFAULT_BEAR_PATH = os.path.join(newton.examples.get_asset_directory(), "bear.usd")  # Path to input bear asset


@wp.kernel
def loss_kernel(com: wp.array(dtype=wp.vec3), loss: wp.array(dtype=float)):
    tid = wp.tid()
    vx = com[tid][0]
    vy = com[tid][1]
    vz = com[tid][2]
    delta = wp.abs(vy) + wp.abs(vz) - vx

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
    def __init__(self, viewer, verbose=False, sim_steps=300):
        # setup simulation parameters first
        fps = 60
        self.frame = 0
        self.frame_dt = 1.0 / fps

        self.sim_steps = sim_steps
        self.sim_substeps = 80
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.phase_count = PHASE_COUNT
        self.verbose = verbose

        # setup training parameters
        self.train_iter = 0
        self.train_rate = 0.025

        # setup rendering
        self.viewer = viewer

        # load bear asset
        asset_stage = Usd.Stage.Open(DEFAULT_BEAR_PATH)

        geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bear/bear"))
        points = geom.GetPointsAttr().Get()

        self.points = [wp.vec3(point) for point in points]
        self.tet_indices = geom.GetPrim().GetAttribute("tetraIndices").Get()

        # create sim model
        scene = newton.ModelBuilder()

        scene.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 0.5),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=self.points,
            indices=self.tet_indices,
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

        # add ground plane to scene
        ke = 2.0e3
        kf = 10.0
        kd = 0.1
        mu = 0.7

        scene.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=kf, kd=kd, mu=mu))

        # finalize model
        # use `requires_grad=True` to create a model for differentiable simulation
        self.model = scene.finalize(requires_grad=True)

        self.model.soft_contact_ke = ke
        self.model.soft_contact_kf = kf
        self.model.soft_contact_kd = kd
        self.model.soft_contact_mu = mu

        # allocate sim states
        self.states = []
        for _i in range(self.sim_steps * self.sim_substeps + 1):
            self.states.append(self.model.state(requires_grad=True))

        # initialize control and one-shot contacts (valid for simple collisions against constant plane)
        self.control = self.model.control()
        self.contacts = self.model.collide(self.states[0], soft_contact_margin=10.0)

        # initialize the solver.
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)

        # model input
        self.phases = []
        for _i in range(self.sim_steps):
            self.phases.append(wp.zeros(self.phase_count, dtype=float, requires_grad=True))

        # weights matrix for linear network
        rng = np.random.default_rng(42)
        k = 1.0 / self.phase_count
        weights = rng.uniform(-np.sqrt(k), np.sqrt(k), (self.model.tet_count, self.phase_count))
        self.weights = wp.array(weights, dtype=float, requires_grad=True)

        # tanh activation layer array
        self.tet_activations = []
        for _i in range(self.sim_steps):
            self.tet_activations.append(wp.zeros(self.model.tet_count, dtype=float, requires_grad=True))

        # optimization
        self.loss = wp.zeros(1, dtype=float, requires_grad=True)
        self.loss_history = []
        self.coms = []
        for _i in range(self.sim_steps):
            self.coms.append(wp.zeros(1, dtype=wp.vec3, requires_grad=True))
        self.optimizer = warp.optim.Adam([self.weights.flatten()], lr=self.train_rate)

        # rendering
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(25.0, -20.0, 10.0), pitch=-20.0, yaw=130.0)

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
            for i in range(self.sim_steps):
                self.forward(i)
        self.tape.backward(self.loss)

    def forward(self, frame):
        # build sinusoidal input phases
        wp.launch(kernel=compute_phases, dim=self.phase_count, inputs=[self.phases[frame], self.sim_time])

        # apply linear network with tanh activation
        wp.launch_tiled(
            kernel=network,
            dim=math.ceil(self.model.tet_count / TILE_TETS),
            inputs=[self.phases[frame].reshape((self.phase_count, 1)), self.weights],
            outputs=[self.tet_activations[frame].reshape((self.model.tet_count, 1))],
            block_dim=TILE_THREADS,
        )
        self.control.tet_activations = self.tet_activations[frame]

        # run simulation loop
        for i in range(self.sim_substeps):
            t = frame * self.sim_substeps + i
            self.states[t].clear_forces()
            self.solver.step(
                self.states[t],
                self.states[t + 1],
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.sim_time += self.sim_dt

        # compute center of mass velocity
        wp.launch(
            com_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.states[(frame + 1) * self.sim_substeps].particle_qd,
                self.model.particle_count,
                self.coms[frame],
            ],
            outputs=[],
        )

        # compute loss
        wp.launch(loss_kernel, dim=1, inputs=[self.coms[frame], self.loss], outputs=[])

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.forward_backward()

        # optimization
        self.optimizer.step([self.weights.grad.flatten()])

        loss = self.loss.numpy()
        if self.verbose:
            print(f"Iteration {self.train_iter}: {loss}")
        self.loss_history.append(loss[0])

        # reset sim
        self.sim_time = 0.0
        self.states[0] = self.model.state(requires_grad=True)

        # clear grads and zero arrays for next iteration
        self.tape.zero()
        self.loss.zero_()
        for i in range(self.sim_steps):
            self.coms[i].zero_()

        self.train_iter += 1

    def render(self):
        # draw training run
        for i in range(self.sim_steps + 1):
            state = self.states[i * self.sim_substeps]

            self.viewer.begin_frame(self.frame * self.frame_dt)
            self.viewer.log_state(state)
            self.viewer.end_frame()

            self.frame += 1

    def test(self):
        assert most(np.diff(self.loss_history) < -0.0, min_ratio=0.8)


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--sim-steps", type=int, default=300, help="Number of simulation steps to execute in a training run."
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, verbose=args.verbose, sim_steps=args.sim_steps)

    # Run example
    newton.examples.run(example, args)
