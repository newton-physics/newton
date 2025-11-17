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
# Example Robot UR10
#
# Shows how to set up a simulation of a UR10 robot arm
# from a USD file using newton.ModelBuilder.add_usd() and
# applies a sinusoidal trajectory to the joint targets.
#
# Command: python -m newton.examples robot_ur10 --num-worlds 16
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.selection import ArticulationView


@wp.kernel
def update_joint_target_trajectory_kernel(
    joint_target_trajectory: wp.array3d(dtype=wp.float32),
    time: wp.array(dtype=wp.float32),
    dt: wp.float32,
    # output
    joint_target: wp.array2d(dtype=wp.float32),
):
    world_idx = wp.tid()
    t = time[world_idx]
    t = wp.mod(t + dt, float(joint_target_trajectory.shape[0] - 1))
    step = int(t)
    time[world_idx] = t

    num_dofs = joint_target.shape[1]
    for dof in range(num_dofs):
        # add world_idx here to make the sequence of dofs different for each world
        di = (dof + world_idx) % num_dofs
        joint_target[world_idx, dof] = wp.lerp(
            joint_target_trajectory[step, world_idx, di],
            joint_target_trajectory[step + 1, world_idx, di],
            wp.frac(t),
        )


class Example:
    def __init__(self, viewer, num_worlds=4):
        self.fps = 50
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_worlds = num_worlds

        self.viewer = viewer

        self.device = wp.get_device()

        ur10 = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(ur10)

        asset_path = newton.utils.download_asset("universal_robots_ur10")
        asset_file = str(asset_path / "usd" / "ur10_instanceable.usda")
        height = 1.2
        ur10.add_usd(
            asset_file,
            xform=wp.transform(wp.vec3(0.0, 0.0, height)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        # create a pedestal
        ur10.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(0, 0, height / 2)), half_height=height / 2, radius=0.08)

        for i in range(len(ur10.joint_target_ke)):
            ur10.joint_target_ke[i] = 500
            ur10.joint_target_kd[i] = 50

        builder = newton.ModelBuilder()
        builder.replicate(ur10, self.num_worlds, spacing=(2, 2, 0))

        # set random joint configurations
        rng = np.random.default_rng(42)
        joint_q = rng.uniform(-wp.pi, wp.pi, builder.joint_dof_count)
        builder.joint_q = joint_q.tolist()

        builder.add_ground_plane()

        self.model = builder.finalize()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = None

        self.articulation_view = ArticulationView(
            self.model, "*ur10*", exclude_joint_types=[newton.JointType.FREE, newton.JointType.DISTANCE]
        )
        assert self.articulation_view.count == self.num_worlds, (
            "Number of worlds must match the number of articulations"
        )
        dof_count = self.articulation_view.joint_dof_count
        joint_target_trajectory = np.zeros((0, self.num_worlds, dof_count), dtype=np.float32)

        self.control_speed = 50.0

        dof_lower = self.articulation_view.get_attribute("joint_limit_lower", self.model)[0].numpy()
        dof_upper = self.articulation_view.get_attribute("joint_limit_upper", self.model)[0].numpy()
        joint_q = self.articulation_view.get_attribute("joint_q", self.state_0).numpy()
        for i in range(dof_count):
            # generate sinusoidal control signal for this dof
            lower = dof_lower[i]
            upper = dof_upper[i]
            if not np.isfinite(lower) or abs(lower) > 6.0:
                # no limits; assume the joint dof is angular
                lower = -wp.pi
                upper = wp.pi
            # first determine the phase shift such that the signal starts at the dof's initial joint_q
            limit_range = upper - lower
            normalized = (joint_q[:, i] - lower) / limit_range * 2.0 - 1.0
            phase_shift = np.zeros(self.articulation_view.count)
            mask = abs(normalized) < 1.0
            phase_shift[mask] = np.arcsin(normalized[mask])

            traj = np.sin(np.linspace(phase_shift, 2 * np.pi + phase_shift, int(limit_range * 50)))
            traj = traj * (upper - lower) * 0.5 + 0.5 * (upper + lower)

            target_trajectory = np.tile(joint_q, (len(traj), 1, 1))
            target_trajectory[:, :, i] = traj

            joint_target_trajectory = np.concatenate((joint_target_trajectory, target_trajectory), axis=0)

        self.joint_target_trajectory = wp.array(joint_target_trajectory, dtype=wp.float32, device=self.device)
        self.time_step = wp.zeros(self.num_worlds, dtype=wp.float32, device=self.device)

        self.ctrl = self.articulation_view.get_attribute("joint_target_pos", self.control)

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            disable_contacts=True,
        )

        self.viewer.set_model(self.model)

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

            wp.launch(
                update_joint_target_trajectory_kernel,
                dim=self.num_worlds,
                inputs=[self.joint_target_trajectory, self.time_step, self.sim_dt * self.control_speed],
                outputs=[self.ctrl],
                device=self.device,
            )
            self.articulation_view.set_attribute("joint_target_pos", self.control, self.ctrl)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--num-worlds", type=int, default=100, help="Total number of simulated worlds.")

    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args.num_worlds)

    newton.examples.run(example, args)
