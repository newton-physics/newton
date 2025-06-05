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

import math

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.examples import compute_env_offsets
from newton.utils.selection import ArticulationView, ContactView, ContactViewManager

COLLAPSE_FIXED_JOINTS = True
VERBOSE = True


class Example:
    def __init__(self, stage_path=None, num_envs=8):
        self.num_envs = num_envs

        up_axis = newton.Axis.Z

        articulation_builder = newton.ModelBuilder(up_axis=up_axis)
        newton.utils.parse_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            articulation_builder,
            ignore_names=["floor", "ground"],
            up_axis=up_axis,
            xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            collapse_fixed_joints=COLLAPSE_FIXED_JOINTS,
        )

        env_offsets = compute_env_offsets(num_envs, env_offset=(4.0, 4.0, 0.0), up_axis=up_axis)

        builder = newton.ModelBuilder()
        for i in range(self.num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(env_offsets[i], wp.quat_identity()))

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

        # ===========================================================
        # create articulation view
        # ===========================================================
        self.ants = ArticulationView(self.model, "ant", verbose=VERBOSE, exclude_joint_types=[newton.JOINT_FREE])

        print(f"joint_q shape:      {self.ants.get_attribute('joint_q', self.model).shape}")
        print(f"joint_qd shape:     {self.ants.get_attribute('joint_qd', self.model).shape}")
        print(f"joint_f shape:      {self.ants.get_attribute('joint_f', self.model).shape}")
        print(f"joint_target shape: {self.ants.get_attribute('joint_target', self.model).shape}")
        print(f"body_q shape:       {self.ants.get_attribute('body_q', self.model).shape}")
        print(f"body_qd shape:      {self.ants.get_attribute('body_qd', self.model).shape}")

        # default root states
        self.default_root_transforms = wp.to_torch(self.ants.get_root_transforms(self.model)).clone()
        self.default_root_velocities = wp.to_torch(self.ants.get_root_velocities(self.model)).clone()
        self.default_root_velocities[:, 2] = 0.5 * math.pi  # rotate about z-axis
        self.default_root_velocities[:, 5] = 5.0  # move up z-axis

        # set all DOFs to the middle of their range by default
        self.contact_mgr = ContactViewManager(self.model)

        self.contacts_feet_ground = ContactView(
            self.contact_mgr,
            "/World/envs/*/Robot/*_foot",  # all robots' feet
            "/World/ground/GroundPlane/CollisionPlane",
        )

        self.contacts_torso_ground = ContactView(
            self.contact_mgr,
            "/World/envs/*/Robot/torso",  # all robots' torsos
            "/World/ground/GroundPlane/CollisionPlane",
        )

        self.contact_mgr.finalize(self.solver)

        # Precompute foot body indices for efficient color updates
        self.foot_body_indices = []
        for body_key in self.contacts_feet_ground.query_keys[0]:
            body_idx = self.model.body_key.index(body_key)
            self.foot_body_indices.append(body_idx)

        # Precompute torso body indices for efficient color updates
        self.torso_body_indices = []
        for body_key in self.contacts_torso_ground.query_keys[0]:
            body_idx = self.model.body_key.index(body_key)
            self.torso_body_indices.append(body_idx)

        # set all dofs to the middle of their range by default
        dof_limit_lower = wp.to_torch(self.ants.get_attribute("joint_limit_lower", self.model))
        dof_limit_upper = wp.to_torch(self.ants.get_attribute("joint_limit_upper", self.model))
        self.default_dof_positions = 0.5 * (dof_limit_lower + dof_limit_upper)
        self.default_dof_velocities = wp.to_torch(self.ants.get_dof_velocities(self.model)).clone()

        # create disjoint subsets to alternate resets
        all_indices = torch.arange(num_envs, dtype=torch.int32)
        self.mask_0 = torch.zeros(num_envs, dtype=bool)
        self.mask_0[all_indices[::2]] = True
        self.mask_1 = torch.zeros(num_envs, dtype=bool)
        self.mask_1[all_indices[1::2]] = True

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 2.0

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

        self.t = 0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                contacts = self.model.collide(self.state_0)
            else:
                contacts = None

            self.solver.step(self.model, self.state_0, self.state_1, self.control, contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset(mask=self.mask_0)
            self.mask_0, self.mask_1 = self.mask_1, self.mask_0
            self.next_reset = self.sim_time + 2.0

        # =========================
        # apply random controls
        # =========================
        dof_forces = 5.0 - 10.0 * torch.rand((self.num_envs, self.ants.joint_dof_count))
        self.ants.set_dof_forces(self.control, dof_forces)

        contact = self.model.contact()
        contact.dist = self.solver.mjw_data.contact.dist
        contact.geom = self.solver.mjw_data.contact.geom
        contact.frame = self.solver.mjw_data.contact.frame
        contact.worldid = self.solver.mjw_data.contact.worldid

        n_contacts = self.solver.mjw_data.ncon
        self.contact_mgr.contact_reporter.select_aggregate(contact, n_contacts)

        feet_entities, feet_matrix = self.contacts_feet_ground.get_contact_dist()

        def colormap(dist):
            if dist > 1:
                return (0.5, 0.5, 0.5)
            d = 0.2 + min(-float(dist[0]) * 15, 1) * 0.8
            v = 0.5 + d / 2
            return (v, v * (1 - d), v * (1 - d))

        # Update shape colors based on contact distances
        body_colors = {}
        for foot_nr, dist in enumerate(feet_matrix.numpy()):
            body_idx = self.foot_body_indices[foot_nr]
            body_colors[body_idx] = colormap(dist)

        # Handle torso contacts
        torso_entities, torso_matrix = self.contacts_torso_ground.get_contact_dist()
        for torso_nr, dist in enumerate(torso_matrix.numpy()):
            body_idx = self.torso_body_indices[torso_nr]
            if dist < 1:  # in contact
                body_colors[body_idx] = (1.0, 0.0, 1.0)  # magenta
            else:  # not in contact
                body_colors[body_idx] = (1.0, 0.5, 0.0)  # orange

        self._set_shape_colors(body_colors)

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def reset(self, mask=None):
        # ==============================
        # set transforms and velocities
        # ==============================
        self.ants.set_root_transforms(self.state_0, self.default_root_transforms, mask=mask)
        self.ants.set_root_velocities(self.state_0, self.default_root_velocities, mask=mask)
        self.ants.set_dof_positions(self.state_0, self.default_dof_positions, mask=mask)
        self.ants.set_dof_velocities(self.state_0, self.default_dof_velocities, mask=mask)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.ants.eval_fk(self.state_0, mask=mask)

    def _set_shape_colors(self, body_colors):
        """Set colors for all shapes of specified bodies in the renderer."""
        if self.renderer is None:
            return

        inst = list(self.renderer._instances.values())
        inst_keys = list(self.renderer._instances.keys())

        for body_idx, color in body_colors.items():
            # Get all shapes for this body
            for shape_idx in self.model.body_shapes[body_idx]:
                inst_copy = list(inst[shape_idx])
                inst_copy[5] = color
                inst_copy[6] = np.array(color)
                self.renderer._instances[inst_keys[shape_idx]] = tuple(inst_copy)

        self.renderer.update_instance_colors()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_selection_ant.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=16, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device), torch.device(wp.device_to_torch(args.device)):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

            # import time
            # time.sleep(0.2)

        if example.renderer:
            example.renderer.save()
