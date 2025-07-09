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

from __future__ import annotations

import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples import compute_env_offsets
from newton.utils.selection import ArticulationView

USE_TORCH = False
COLLAPSE_FIXED_JOINTS = False
VERBOSE = True


@wp.kernel
def randomize_states_kernel(joint_q: wp.array2d(dtype=float), seed: int):
    tid = wp.tid()
    rng = wp.rand_init(seed, tid)
    joint_q[tid, 0] = 2.0 - 4.0 * wp.randf(rng)
    joint_q[tid, 1] = wp.pi / 8.0 - wp.pi / 4.0 * wp.randf(rng)
    joint_q[tid, 2] = wp.pi / 8.0 - wp.pi / 4.0 * wp.randf(rng)


@wp.kernel
def apply_forces_kernel(joint_q: wp.array2d(dtype=float), joint_f: wp.array2d(dtype=float)):
    tid = wp.tid()
    if joint_q[tid, 0] > 0.0:
        joint_f[tid, 0] = -20.0
    else:
        joint_f[tid, 0] = 20.0


class Example:
    def __init__(self, stage_path: str | None = "example_selection_cartpole.usd", num_envs=16, use_cuda_graph=True):
        self.num_envs = num_envs

        up_axis = newton.Axis.Z

        articulation_builder = newton.ModelBuilder(up_axis=up_axis)
        newton.utils.parse_urdf(
            newton.examples.get_asset("cartpole.urdf"),
            articulation_builder,
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
            enable_self_collisions=False,
            floating=False,
        )

        env_offsets = compute_env_offsets(num_envs, env_offset=(4.0, 4.0, 0.0), up_axis=up_axis)

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(env_offsets[i], wp.quat_identity()))

        # finalize model
        self.model = builder.finalize()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.solver = newton.solvers.MuJoCoSolver(self.model, disable_contacts=True)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # =======================
        # get cartpole view
        # =======================
        self.cartpoles = ArticulationView(self.model, "cartpole", verbose=VERBOSE)

        # =========================
        # randomize initial state
        # =========================
        if USE_TORCH:
            import torch  # noqa: PLC0415

            cart_positions = 2.0 - 4.0 * torch.rand(num_envs)
            pole1_angles = torch.pi / 8.0 - torch.pi / 4.0 * torch.rand(num_envs)
            pole2_angles = torch.pi / 8.0 - torch.pi / 4.0 * torch.rand(num_envs)
            joint_q = torch.stack([cart_positions, pole1_angles, pole2_angles], dim=1)
        else:
            joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
            wp.launch(randomize_states_kernel, dim=num_envs, inputs=[joint_q, 42])

        self.cartpoles.set_attribute("joint_q", self.state_0, joint_q)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.cartpoles.eval_fk(self.state_0)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=1.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 3, 10),
            )

        self.use_cuda_graph = wp.get_device().is_cuda and use_cuda_graph
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
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
            joint_f[:, 0] = torch.where(joint_q[:, 0] > 0, -20, 20)
        else:
            joint_q = self.cartpoles.get_attribute("joint_q", self.state_0)
            joint_f = self.cartpoles.get_attribute("joint_f", self.control)
            wp.launch(apply_forces_kernel, dim=joint_f.shape, inputs=[joint_q, joint_f])

        self.cartpoles.set_attribute("joint_f", self.control, joint_f)

        # simulate
        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


# scoped device manager for both Warp and Torch
class ScopedDevice:
    def __init__(self, device):
        self.warp_scoped_device = wp.ScopedDevice(device)
        if USE_TORCH:
            import torch  # noqa: PLC0415

            self.torch_scoped_device = torch.device(wp.device_to_torch(device))

    def __enter__(self):
        self.warp_scoped_device.__enter__()
        if USE_TORCH:
            self.torch_scoped_device.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warp_scoped_device.__exit__(exc_type, exc_val, exc_tb)
        if USE_TORCH:
            self.torch_scoped_device.__exit__(exc_type, exc_val, exc_tb)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_selection_cartpole.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=12000, help="Total number of frames.")
    parser.add_argument("--num-envs", type=int, default=16, help="Total number of simulated environments.")
    parser.add_argument("--use-cuda-graph", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs, use_cuda_graph=args.use_cuda_graph)

        for frame_idx in range(args.num_frames):
            example.step()
            example.render()

            if not example.renderer:
                print(f"[{frame_idx:4d}/{args.num_frames}]")

        if example.renderer:
            example.renderer.save()
