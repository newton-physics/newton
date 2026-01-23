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
# Example Selection Cartpole
#
# Demonstrates batch control of multiple cartpole worlds using
# ArticulationView. This example spawns multiple cartpole robots and applies
# simple random control policy.
#
# To limit the number of worlds to render use the max-worlds argument.
# Command: python -m newton.examples selection_cartpole --num-worlds 16 --max-worlds 8
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton.selection import ArticulationView

USE_TORCH = False
COLLAPSE_FIXED_JOINTS = False


@wp.kernel
def randomize_states_kernel(joint_q: wp.array3d(dtype=float), seed: int):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    joint_q[tid, 0, 0] = 2.0 - 4.0 * wp.randf(rng)
    joint_q[tid, 0, 1] = wp.pi / 8.0 - wp.pi / 4.0 * wp.randf(rng)
    joint_q[tid, 0, 2] = wp.pi / 8.0 - wp.pi / 4.0 * wp.randf(rng)


@wp.kernel
def apply_forces_kernel(joint_q: wp.array3d(dtype=float), joint_f: wp.array3d(dtype=float)):
    tid = wp.tid()
    if joint_q[tid, 0, 0] > 0.0:
        joint_f[tid, 0, 0] = -20.0
    else:
        joint_f[tid, 0, 0] = 20.0


class Example:
    def __init__(self, viewer, num_worlds=16, max_worlds=None, verbose=True):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        world = newton.ModelBuilder()
        world.add_usd(
            newton.examples.get_asset("cartpole.usda"),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
            enable_self_collisions=False,
        )

        scene = newton.ModelBuilder()
        scene.replicate(world, num_worlds=self.num_worlds)

        # finalize model
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, disable_contacts=True)

        self.viewer = viewer

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # =======================
        # get cartpole view
        # =======================
        self.cartpoles = ArticulationView(self.model, "/cartPole", verbose=verbose)

        # =========================
        # randomize initial state
        # =========================
        if USE_TORCH:
            import torch  # noqa: PLC0415

            cart_positions = 2.0 - 4.0 * torch.rand(num_worlds)
            pole1_angles = torch.pi / 8.0 - torch.pi / 4.0 * torch.rand(num_worlds)
            pole2_angles = torch.pi / 8.0 - torch.pi / 4.0 * torch.rand(num_worlds)
            joint_q = torch.stack([cart_positions, pole1_angles, pole2_angles], dim=1)
        else:
            joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
            wp.launch(randomize_states_kernel, dim=num_worlds, inputs=[joint_q, 42])

        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q)

        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.cartpoles.eval_fk(self.state_0)

        self.viewer.set_model(self.model, max_worlds=max_worlds)
        self.viewer.set_world_offsets((2.0, 0.0, 0.0))

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # ====================================
        # get observations and apply controls
        # ====================================
        if USE_TORCH:
            import torch  # noqa: PLC0415

            joint_q = wp.to_torch(self.cartpoles.get_attribute("joint_q", self.state_0))
            joint_f = wp.to_torch(self.cartpoles.get_attribute("joint_f", self.control))
            joint_f[..., 0] = torch.where(joint_q[..., 0] > 0, -20, 20)
        else:
            joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
            joint_f = self.cartpoles.get_attribute("joint_f", self.control)
            wp.launch(
                apply_forces_kernel,
                dim=joint_f.shape,
                inputs=[joint_q, joint_f],
            )

        self.cartpoles.set_attribute("joint_f", self.control, joint_f)

        # simulate
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        num_bodies_per_world = self.model.body_count // self.num_worlds
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart is at ground level and has correct orientation",
            lambda q, qd: q[2] == 0.0 and newton.utils.vec_allclose(q.q, wp.quat_identity()),
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "cart only moves along y direction",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and wp.length_sq(wp.spatial_bottom(qd)) == 0.0,
            indices=[i * num_bodies_per_world for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole1 only has y-axis linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and qd[2] == 0.0
            and abs(qd[3]) > 0.3
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 1 for i in range(self.num_worlds)],
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "pole2 only has yz-plane linear velocity and x-axis angular velocity",
            lambda q, qd: qd[0] == 0.0
            and abs(qd[1]) > 0.05
            and abs(qd[2]) > 0.05
            and abs(qd[3]) > 0.2
            and qd[4] == 0.0
            and qd[5] == 0.0,
            indices=[i * num_bodies_per_world + 2 for i in range(self.num_worlds)],
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--num-worlds",
        type=int,
        default=16,
        help="Total number of simulated worlds.",
    )

    viewer, args = newton.examples.init(parser)

    if USE_TORCH:
        import torch

        torch.set_default_device(args.device)

    example = Example(viewer, num_worlds=args.num_worlds, max_worlds=args.max_worlds)

    newton.examples.run(example, args)
