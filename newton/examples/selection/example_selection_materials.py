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
# Example Selection Materials
#
# Demonstrates runtime material property modification using ArticulationView.
# This example spawns multiple ant robots across worlds and dynamically
# changes their friction coefficients during simulation. The ants alternate
# between forward and backward movement with randomized material properties,
# showcasing how to efficiently modify physics parameters for batches of
# objects using the selection API.
#
# Command: python -m newton.examples selection_materials
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton.selection import ArticulationView
from newton.solvers import SolverNotifyFlags

USE_TORCH = False
COLLAPSE_FIXED_JOINTS = False
VERBOSE = True

# RANDOMIZE_PER_WORLD determines how shape material values are randomized.
# - If True, all shapes in the same world get the same random value.
# - If False, each shape in each world gets its own random value.
RANDOMIZE_PER_WORLD = True


@wp.kernel
def compute_middle_kernel(
    lower: wp.array2d(dtype=float),
    upper: wp.array2d(dtype=float),
    middle: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    middle[i, j] = 0.5 * (lower[i, j] + upper[i, j])


@wp.kernel
def reset_materials_kernel(mu: wp.array2d(dtype=float), seed: int, shape_count: int):
    i, j = wp.tid()

    if RANDOMIZE_PER_WORLD:
        rng = wp.rand_init(seed, i)
    else:
        rng = wp.rand_init(seed, i * shape_count + j)

    mu[i, j] = wp.randf(rng)  # random coefficient of friction


class Example:
    def __init__(self, viewer, num_worlds=16):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        world_template = newton.ModelBuilder()
        world_template.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )

        scene = newton.ModelBuilder()

        scene.add_ground_plane()
        scene.replicate(world_template, num_worlds=self.num_worlds)

        # finalize model
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=50)

        self.viewer = viewer

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        self.next_reset = 0.0
        self.reset_count = 0

        # ===========================================================
        # create articulation view
        # ===========================================================
        self.ants = ArticulationView(
            self.model,
            "ant",
            verbose=VERBOSE,
            exclude_joint_types=[newton.JointType.FREE],
        )

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
                inputs=[
                    dof_limit_lower,
                    dof_limit_upper,
                    self.default_ant_dof_positions,
                ],
            )
            self.default_ant_dof_velocities = wp.clone(self.ants.get_dof_velocities(self.model))

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets((4.0, 4.0, 0.0))

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # reset all
        self.reset()
        self.capture()

        self.next_reset = self.sim_time + 2.0

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

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
                self.contacts = self.model.collide(self.state_0)
            else:
                self.contacts = None

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset()
            self.next_reset = self.sim_time + 2.0

        if self.graph:
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
            if RANDOMIZE_PER_WORLD:
                material_mu = torch.rand(self.ants.count).unsqueeze(1).repeat(1, self.ants.shape_count)
            else:
                material_mu = torch.rand((self.ants.count, self.ants.shape_count))
        else:
            # flip velocities
            if self.reset_count % 2 == 0:
                self.default_ant_root_velocities.fill_(wp.spatial_vector(0.0, 5.0, 0.0, 0.0, 0.0, 0.0))
            else:
                self.default_ant_root_velocities.fill_(wp.spatial_vector(0.0, -5.0, 0.0, 0.0, 0.0, 0.0))

            # randomize materials
            material_mu = self.ants.get_attribute("shape_material_mu", self.model)
            wp.launch(
                reset_materials_kernel,
                dim=material_mu.shape,
                inputs=[material_mu, self.reset_count, self.ants.shape_count],
            )

        self.ants.set_attribute("shape_material_mu", self.model, material_mu)

        # check values in model
        # print(self.ants.get_attribute("shape_material_mu", self.model))
        # print(self.model.shape_material_mu)

        # !!! Notify solver of material changes !!!
        self.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

        # ================================
        # reset transforms and velocities
        # ================================
        self.ants.set_root_transforms(self.state_0, self.default_ant_root_transforms, mask=mask)
        self.ants.set_root_velocities(self.state_0, self.default_ant_root_velocities, mask=mask)
        self.ants.set_dof_positions(self.state_0, self.default_ant_dof_positions, mask=mask)
        self.ants.set_dof_velocities(self.state_0, self.default_ant_dof_velocities, mask=mask)

        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.ants.eval_fk(self.state_0, mask=mask)

        self.reset_count += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.01,
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

        torch.set_device(args.device)

    example = Example(viewer, num_worlds=args.num_worlds)

    newton.examples.run(example, args)
