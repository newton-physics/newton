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

# RANDOMIZE_PER_ENV determines how shape material values are randomized.
# - If True, all shapes in the same environment get the same random value.
# - If False, each shape in each environment gets its own random value.
RANDOMIZE_PER_ENV = True


@wp.kernel
def compute_middle_kernel(
    lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float), middle: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    middle[i, j] = 0.5 * (lower[i, j] + upper[i, j])


@wp.kernel
def reset_materials_kernel(mu: wp.array2d(dtype=float), seed: int, num_envs: int):
    i, j = wp.tid()

    if RANDOMIZE_PER_ENV:
        rng = wp.rand_init(seed, i)
    else:
        rng = wp.rand_init(seed, i * num_envs + j)

    mu[i, j] = wp.randf(rng)  # random coefficient of friction


class Example:
    def __init__(self, stage_path: str | None = "example_selection_materials.usd", num_envs=16, use_cuda_graph=True):
        self.num_envs = num_envs

        up_axis = newton.Axis.Z

        env_builder = newton.ModelBuilder(up_axis=up_axis)
        newton.utils.parse_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            env_builder,
            ignore_names=["floor", "ground"],
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )

        env_offsets = compute_env_offsets(num_envs, env_offset=(4.0, 4.0, 0.0), up_axis=up_axis)

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_builder(env_builder, xform=wp.transform(env_offsets[i], wp.quat_identity()))

        builder.add_ground_plane()

        # finalize model
        self.model = builder.finalize()

        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None
        if stage_path:
            self.renderer = newton.utils.SimRendererOpenGL(
                path=stage_path,
                model=self.model,
                scaling=2.0,
                up_axis=str(up_axis),
                screen_width=1280,
                screen_height=720,
                camera_pos=(0, 4, 30),
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.next_reset = 0.0
        self.reset_count = 0

        # ===========================================================
        # create articulation view
        # ===========================================================
        self.ants = ArticulationView(self.model, "ant", verbose=VERBOSE, exclude_joint_types=[newton.JOINT_FREE])

        if USE_TORCH:
            # default ant root states
            self.default_ant_root_transforms = wp.to_torch(self.ants.get_root_transforms(self.model)).clone()
            self.default_ant_root_velocities = wp.to_torch(self.ants.get_root_velocities(self.model)).clone()

            # set ant DOFs to the middle of their range by default
            dof_limit_lower = wp.to_torch(self.ants.get_attribute("joint_limit_lower", self.model))
            dof_limit_upper = wp.to_torch(self.ants.get_attribute("joint_limit_upper", self.model))
            self.default_ant_dof_positions = 0.5 * (dof_limit_lower + dof_limit_upper)
            self.default_ant_dof_velocities = wp.to_torch(self.ants.get_dof_velocities(self.model)).clone()
        else:
            # default ant root states
            self.default_ant_root_transforms = wp.clone(self.ants.get_root_transforms(self.model))
            self.default_ant_root_velocities = wp.clone(self.ants.get_root_velocities(self.model))

            # set ant DOFs to the middle of their range by default
            dof_limit_lower = self.ants.get_attribute("joint_limit_lower", self.model)
            dof_limit_upper = self.ants.get_attribute("joint_limit_upper", self.model)
            self.default_ant_dof_positions = wp.empty_like(dof_limit_lower)
            wp.launch(
                compute_middle_kernel,
                dim=self.default_ant_dof_positions.shape,
                inputs=[dof_limit_lower, dof_limit_upper, self.default_ant_dof_positions],
            )
            self.default_ant_dof_velocities = wp.clone(self.ants.get_dof_velocities(self.model))

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 2.0

        self.use_cuda_graph = wp.get_device().is_cuda and use_cuda_graph
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                contacts = self.model.collide(self.state_0)
            else:
                contacts = None

            self.solver.step(self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()
            self.next_reset = self.sim_time + 2.0

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def reset(self, mask=None):
        # ========================================
        # update velocities and materials
        # ========================================
        if USE_TORCH:
            import torch  # noqa: PLC0415

            # flip velocities
            if self.reset_count % 2 == 0:
                self.default_ant_root_velocities[:, 4] = 5.0
            else:
                self.default_ant_root_velocities[:, 4] = -5.0

            # randomize materials
            if RANDOMIZE_PER_ENV:
                material_mu = torch.rand(self.ants.count).unsqueeze(1).repeat(1, self.ants.shape_count)
            else:
                material_mu = torch.rand((self.ants.count, self.ants.shape_count))
        else:
            # flip velocities
            if self.reset_count % 2 == 0:
                self.default_ant_root_velocities.fill_(wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 5.0, 0.0))
            else:
                self.default_ant_root_velocities.fill_(wp.spatial_vector(0.0, 0.0, 0.0, 0.0, -5.0, 0.0))

            # randomize materials
            material_mu = self.ants.get_attribute("shape_materials.mu", self.model)
            wp.launch(
                reset_materials_kernel, dim=material_mu.shape, inputs=[material_mu, self.reset_count, self.num_envs]
            )

        self.ants.set_attribute("shape_materials.mu", self.model, material_mu)

        # check values in model
        # print(self.ants.get_attribute("shape_materials.mu", self.model))
        # print(self.model.shape_materials.mu)

        # !!! Notify solver of material changes !!!
        self.solver.notify_model_changed(newton.sim.NOTIFY_FLAG_SHAPE_PROPERTIES)

        # ================================
        # reset transforms and velocities
        # ================================
        self.ants.set_root_transforms(self.state_0, self.default_ant_root_transforms, mask=mask)
        self.ants.set_root_velocities(self.state_0, self.default_ant_root_velocities, mask=mask)
        self.ants.set_dof_positions(self.state_0, self.default_ant_dof_positions, mask=mask)
        self.ants.set_dof_velocities(self.state_0, self.default_ant_dof_velocities, mask=mask)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.ants.eval_fk(self.state_0, mask=mask)

        self.reset_count += 1

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
        default="example_selection_materials.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=1200, help="Total number of frames.")
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
