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
# Example Selection Multiple Articulations
#
# Shows how to control multiple articulation per world in a single ArticulationView.
#
# Command: python -m newton.examples selection_multiple
#
###########################################################################

from __future__ import annotations

import warp as wp

import newton
import newton.examples
from newton.selection import ArticulationView

USE_TORCH = False
COLLAPSE_FIXED_JOINTS = False
VERBOSE = True


@wp.kernel
def compute_middle_kernel(
    lower: wp.array3d(dtype=float), upper: wp.array3d(dtype=float), middle: wp.array3d(dtype=float)
):
    world, arti, dof = wp.tid()
    middle[world, arti, dof] = 0.5 * (lower[world, arti, dof] + upper[world, arti, dof])


@wp.kernel
def init_masks(mask_0: wp.array(dtype=bool), mask_1: wp.array(dtype=bool)):
    tid = wp.tid()
    yes = tid % 2 == 0
    mask_0[tid] = yes
    mask_1[tid] = not yes


@wp.func
def check_mask(mask: wp.array(dtype=bool), idx: int) -> bool:
    # mask could be None, in which case return True
    if mask:
        return mask[idx]
    else:
        return True


@wp.kernel
def reset_kernel(
    root_velocities: wp.array2d(dtype=wp.spatial_vector),  # root velocities
    mask: wp.array(dtype=bool),  # world mask (optional, can be None)
    max_jump: float,  # maximum jump speed
    max_spin: float,  # maximum spin speed
    seed: int,  # random seed
):
    world, arti = wp.tid()

    if check_mask(mask, world):
        # random jump velocity per world
        jump_rng = wp.rand_init(seed, world)
        jump_vel = max_jump * wp.randf(jump_rng)
        # random spin velocity per articulation
        num_artis = root_velocities.shape[1]
        spin_rng = wp.rand_init(seed, world * num_artis + arti)
        spin_vel = max_spin * (1.0 - 2.0 * wp.randf(spin_rng))
        root_velocities[world, arti] = wp.spatial_vector(0.0, 0.0, jump_vel, 0.0, 0.0, spin_vel)


@wp.kernel
def random_forces_kernel(
    dof_forces: wp.array3d(dtype=float),  # dof forces (output)
    max_magnitude: float,  # maximum force magnitude
    seed: int,  # random seed
):
    world, arti, dof = wp.tid()
    num_artis, num_dofs = dof_forces.shape[1], dof_forces.shape[2]
    rng = wp.rand_init(seed, num_dofs * (world * num_artis + arti) + dof)
    dof_forces[world, arti, dof] = max_magnitude * (1.0 - 2.0 * wp.randf(rng))


class Example:
    def __init__(self, viewer, num_worlds=16):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        # load articulation
        arti = newton.ModelBuilder()
        arti.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )

        # create world with multiple articulations
        world = newton.ModelBuilder()
        world.add_builder(arti, xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
        world.add_builder(arti, xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()))
        world.add_builder(arti, xform=wp.transform((0.0, 0.0, 3.0), wp.quat_identity()))

        # create scene
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        scene.replicate(world, num_worlds=self.num_worlds)

        # finalize model
        self.model = scene.finalize()

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=100, nconmax=70)

        self.viewer = viewer

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Contacts only needed for non-MuJoCo solvers
        self.contacts = self.model.contacts() if not isinstance(self.solver, newton.solvers.SolverMuJoCo) else None

        self.next_reset = 0.0
        self.step_count = 0

        # ===========================================================
        # create multi-articulation view
        # ===========================================================
        self.ants = ArticulationView(self.model, "ant", verbose=VERBOSE, exclude_joint_types=[newton.JointType.FREE])

        self.num_per_world = self.ants.count_per_world

        if USE_TORCH:
            import torch  # noqa: PLC0415

            # default ant root states
            self.default_root_transforms = wp.to_torch(self.ants.get_root_transforms(self.model)).clone()
            self.default_root_velocities = wp.to_torch(self.ants.get_root_velocities(self.model)).clone()

            # set ant DOFs to the middle of their range by default
            dof_limit_lower = wp.to_torch(self.ants.get_attribute("joint_limit_lower", self.model))
            dof_limit_upper = wp.to_torch(self.ants.get_attribute("joint_limit_upper", self.model))
            self.default_dof_positions = 0.5 * (dof_limit_lower + dof_limit_upper)
            self.default_dof_velocities = wp.to_torch(self.ants.get_dof_velocities(self.model)).clone()

            # create disjoint subsets to alternate resets
            all_indices = torch.arange(num_worlds, dtype=torch.int32)
            self.mask_0 = torch.zeros(num_worlds, dtype=bool)
            self.mask_0[all_indices[::2]] = True
            self.mask_1 = torch.zeros(num_worlds, dtype=bool)
            self.mask_1[all_indices[1::2]] = True
        else:
            # default ant root states
            self.default_root_transforms = wp.clone(self.ants.get_root_transforms(self.model))
            self.default_root_velocities = wp.clone(self.ants.get_root_velocities(self.model))

            # set ant DOFs to the middle of their range by default
            dof_limit_lower = self.ants.get_attribute("joint_limit_lower", self.model)
            dof_limit_upper = self.ants.get_attribute("joint_limit_upper", self.model)
            self.default_dof_positions = wp.empty_like(dof_limit_lower)
            wp.launch(
                compute_middle_kernel,
                dim=self.default_dof_positions.shape,
                inputs=[dof_limit_lower, dof_limit_upper, self.default_dof_positions],
            )
            self.default_dof_velocities = wp.clone(self.ants.get_dof_velocities(self.model))

            # create disjoint subsets to alternate resets
            self.mask_0 = wp.empty(num_worlds, dtype=bool)
            self.mask_1 = wp.empty(num_worlds, dtype=bool)
            wp.launch(init_masks, dim=num_worlds, inputs=[self.mask_0, self.mask_1])

        self.viewer.set_model(self.model)

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

            # explicit collisions needed without MuJoCo solver
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset(mask=self.mask_0)
            self.mask_0, self.mask_1 = self.mask_1, self.mask_0
            self.next_reset = self.sim_time + 2.0

        # ================================
        # apply random controls
        # ================================
        if USE_TORCH:
            import torch  # noqa: PLC0415

            dof_forces = 2.0 - 4.0 * torch.rand((self.num_worlds, self.num_per_world, self.ants.joint_dof_count))
        else:
            dof_forces = self.ants.get_dof_forces(self.control)
            wp.launch(
                random_forces_kernel,
                dim=dof_forces.shape,
                inputs=[dof_forces, 2.0, self.step_count],
            )

        self.ants.set_dof_forces(self.control, dof_forces)

        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt
        self.step_count += 1

    def reset(self, mask=None):
        # ================================
        # reset transforms and velocities
        # ================================

        if USE_TORCH:
            import torch  # noqa: PLC0415

            # random jump speed per world
            self.default_root_velocities[..., 2] = 3.0 * torch.rand((self.num_worlds, 1))
            # random spin speed per articulation
            self.default_root_velocities[..., 5] = (
                4.0 * torch.pi * (0.5 - torch.rand((self.num_worlds, self.num_per_world)))
            )
        else:
            wp.launch(
                reset_kernel,
                dim=(self.num_worlds, self.num_per_world),
                inputs=[self.default_root_velocities, mask, 3.0, 2.0 * wp.pi, self.step_count],
            )

        self.ants.set_root_transforms(self.state_0, self.default_root_transforms, mask=mask)
        self.ants.set_root_velocities(self.state_0, self.default_root_velocities, mask=mask)
        self.ants.set_dof_positions(self.state_0, self.default_dof_positions, mask=mask)
        self.ants.set_dof_velocities(self.state_0, self.default_dof_velocities, mask=mask)

        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.ants.eval_fk(self.state_0, mask=mask)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.01,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=16, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    if USE_TORCH:
        import torch

        torch.set_default_device(args.device)

    example = Example(viewer, num_worlds=args.num_worlds)

    newton.examples.run(example, args)
