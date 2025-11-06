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
# Example Contact Sensor
#
# Shows how to use the ContactSensor class to evaluate contact forces and torques.
# The example spawns ant and humanoid robots across multiple worlds,
# applies random forces to their joints, and performs selective resets on subsets of worlds.
#
# Command: python -m newton.examples sensor_contact
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton import Contacts
from newton.selection import ArticulationView
from newton.sensors import ContactSensor, populate_contacts
from newton.tests.unittest_utils import find_nonfinite_members

USE_TORCH = False
COLLAPSE_FIXED_JOINTS = False
VERBOSE = True


@wp.kernel
def compute_middle_kernel(
    lower: wp.array2d(dtype=float), upper: wp.array2d(dtype=float), middle: wp.array2d(dtype=float)
):
    i, j = wp.tid()
    middle[i, j] = 0.5 * (lower[i, j] + upper[i, j])


@wp.kernel
def init_masks(mask_0: wp.array(dtype=bool), mask_1: wp.array(dtype=bool)):
    tid = wp.tid()
    yes = tid % 2 == 0
    mask_0[tid] = yes
    mask_1[tid] = not yes


@wp.kernel
def reset_kernel(
    ant_root_velocities: wp.array(dtype=wp.spatial_vector),
    hum_root_velocities: wp.array(dtype=wp.spatial_vector),
    mask: wp.array(dtype=bool),  # optional, can be None
    seed: int,
):
    tid = wp.tid()

    if mask:
        do_it = mask[tid]
    else:
        do_it = True

    if do_it:
        rng = wp.rand_init(seed, tid)
        spin_vel = 4.0 * wp.pi * (0.5 - wp.randf(rng))
        jump_vel = 3.0 * wp.randf(rng)
        ant_root_velocities[tid] = wp.spatial_vector(0.0, 0.0, jump_vel, 0.0, 0.0, spin_vel)
        hum_root_velocities[tid] = wp.spatial_vector(0.0, 0.0, jump_vel, 0.0, 0.0, -spin_vel)


@wp.kernel
def random_forces_kernel(dof_forces: wp.array2d(dtype=float), seed: int, num_worlds: int):
    i, j = wp.tid()
    rng = wp.rand_init(seed, i * num_worlds + j)
    dof_forces[i, j] = 5.0 - 10.0 * wp.randf(rng)


class Example:
    def __init__(self, viewer, num_worlds=16, use_cuda_graph=True):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds
        self.viewer = viewer

        up_axis = newton.Axis.Z

        world_builder = newton.ModelBuilder(up_axis=up_axis)
        world_builder.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            ignore_names=["floor", "ground"],
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )
        world_builder.add_mjcf(
            newton.examples.get_asset("nv_humanoid.xml"),
            ignore_names=["floor", "ground"],
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 3.5), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
            parse_sites=False,  # AD: remove once asset is fixed
        )

        builder = newton.ModelBuilder()
        builder.replicate(world_builder, self.num_worlds)

        builder.add_ground_plane()
        # stores contact info required by contact sensors
        self.contacts = Contacts(0, 0)

        # finalize model
        self.model = builder.finalize()

        self.torso_all_contact_sensor = ContactSensor(self.model, sensing_obj_shapes="torso_geom", verbose=True)
        self.arm_ground_contact_sensor = ContactSensor(
            self.model, sensing_obj_bodies="*arm", counterpart_shapes="ground_plane", verbose=True
        )
        self.foot_arm_contact_sensor = ContactSensor(
            self.model,
            sensing_obj_bodies="*foot",
            counterpart_shapes="*arm",
            include_total=False,
            verbose=True,
            prune_noncolliding=True,
        )

        self.solver = newton.solvers.SolverMuJoCo(self.model, njmax=100, nconmax=100)

        self.viewer.set_model(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.next_reset = 0.0
        self.seed = 0

        # ===========================================================
        # create articulation views
        # ===========================================================
        self.ants = ArticulationView(self.model, "ant", verbose=VERBOSE, exclude_joint_types=[newton.JointType.FREE])
        self.humanoids = ArticulationView(
            self.model, "humanoid", verbose=VERBOSE, exclude_joint_types=[newton.JointType.FREE]
        )

        if USE_TORCH:
            import torch  # noqa: PLC0415

            # default ant root states
            self.default_ant_root_transforms = wp.to_torch(self.ants.get_root_transforms(self.model)).clone()
            self.default_ant_root_velocities = wp.to_torch(self.ants.get_root_velocities(self.model)).clone()

            # set ant DOFs to the middle of their range by default
            dof_limit_lower = wp.to_torch(self.ants.get_attribute("joint_limit_lower", self.model))
            dof_limit_upper = wp.to_torch(self.ants.get_attribute("joint_limit_upper", self.model))
            self.default_ant_dof_positions = 0.5 * (dof_limit_lower + dof_limit_upper)
            self.default_ant_dof_velocities = wp.to_torch(self.ants.get_dof_velocities(self.model)).clone()

            # default humanoid states
            self.default_hum_root_transforms = wp.to_torch(self.humanoids.get_root_transforms(self.model)).clone()
            self.default_hum_root_velocities = wp.to_torch(self.humanoids.get_root_velocities(self.model)).clone()
            self.default_hum_dof_positions = wp.to_torch(self.humanoids.get_dof_positions(self.model)).clone()
            self.default_hum_dof_velocities = wp.to_torch(self.humanoids.get_dof_velocities(self.model)).clone()

            # create disjoint subsets to alternate resets
            all_indices = torch.arange(num_worlds, dtype=torch.int32)
            self.mask_0 = torch.zeros(num_worlds, dtype=bool)
            self.mask_0[all_indices[::2]] = True
            self.mask_1 = torch.zeros(num_worlds, dtype=bool)
            self.mask_1[all_indices[1::2]] = True
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

            # default humanoid states
            self.default_hum_root_transforms = wp.clone(self.humanoids.get_root_transforms(self.model))
            self.default_hum_root_velocities = wp.clone(self.humanoids.get_root_velocities(self.model))
            self.default_hum_dof_positions = wp.clone(self.humanoids.get_dof_positions(self.model))
            self.default_hum_dof_velocities = wp.clone(self.humanoids.get_dof_velocities(self.model))

            # create disjoint subsets to alternate resets
            self.mask_0 = wp.empty(num_worlds, dtype=bool)
            self.mask_1 = wp.empty(num_worlds, dtype=bool)
            wp.launch(init_masks, dim=num_worlds, inputs=[self.mask_0, self.mask_1])

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 2.0

        self.use_cuda_graph = wp.get_device().is_cuda and use_cuda_graph
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
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

            dof_forces = 5.0 - 10.0 * torch.rand((self.num_worlds, self.ants.joint_dof_count))
        else:
            dof_forces = self.ants.get_dof_forces(self.control)
            wp.launch(random_forces_kernel, dim=dof_forces.shape, inputs=[dof_forces, self.seed, self.num_worlds])

        self.ants.set_dof_forces(self.control, dof_forces)

        with wp.ScopedTimer("step", active=False):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        populate_contacts(self.contacts, self.solver)
        self.torso_all_contact_sensor.eval(self.contacts)
        print(f"Torso net forces: {self.torso_all_contact_sensor.net_force}")
        self.arm_ground_contact_sensor.eval(self.contacts)
        self.foot_arm_contact_sensor.eval(self.contacts)

        self.sim_time += self.frame_dt

    def reset(self, mask=None):
        # ================================
        # reset transforms and velocities
        # ================================

        if USE_TORCH:
            import torch  # noqa: PLC0415

            # randomize ant velocities
            self.default_ant_root_velocities[:, 2] = 4.0 * torch.pi * (0.5 - torch.rand(self.num_worlds))
            self.default_ant_root_velocities[:, 5] = 3.0 * torch.rand(self.num_worlds)

            # humanoids spin in the opposite direction
            self.default_hum_root_velocities[:, 2] = -self.default_ant_root_velocities[:, 2]
            # humanoids move up at the same speed
            self.default_hum_root_velocities[:, 5] = self.default_ant_root_velocities[:, 5]
        else:
            wp.launch(
                reset_kernel,
                dim=self.num_worlds,
                inputs=[self.default_ant_root_velocities, self.default_hum_root_velocities, mask, self.seed],
            )

        self.ants.set_root_transforms(self.state_0, self.default_ant_root_transforms, mask=mask)
        self.ants.set_root_velocities(self.state_0, self.default_ant_root_velocities, mask=mask)
        self.ants.set_dof_positions(self.state_0, self.default_ant_dof_positions, mask=mask)
        self.ants.set_dof_velocities(self.state_0, self.default_ant_dof_velocities, mask=mask)

        self.humanoids.set_root_transforms(self.state_0, self.default_hum_root_transforms, mask=mask)
        self.humanoids.set_root_velocities(self.state_0, self.default_hum_root_velocities, mask=mask)
        self.humanoids.set_dof_positions(self.state_0, self.default_hum_dof_positions, mask=mask)
        self.humanoids.set_dof_velocities(self.state_0, self.default_hum_dof_velocities, mask=mask)

        if not isinstance(self.solver, newton.solvers.SolverMuJoCo):
            self.ants.eval_fk(self.state_0, mask=mask)

    def test(self):
        assert len(find_nonfinite_members(self.torso_all_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.arm_ground_contact_sensor)) == 0
        assert len(find_nonfinite_members(self.foot_arm_contact_sensor)) == 0

    def render(self):
        with wp.ScopedTimer("render", active=False):
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.state_0)
            self.viewer.end_frame()


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
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=16, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    with ScopedDevice(args.device):
        # Create viewer and run
        example = Example(viewer, num_worlds=args.num_worlds)

        newton.examples.run(example, args)
