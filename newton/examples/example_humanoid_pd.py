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

"""
Example demonstrating PD control with position targets in MuJoCo.
This modified version of the humanoid example shows how to:
1. Set up PD control gains (kp, kd)
2. Set position targets that update each iteration
3. Verify that the control values reach MuJoCo correctly
4. Simulate simple motion patterns to verify control effectiveness
"""

import math

import torch
import warp as wp
import newton
import newton.examples
import newton.utils
from newton.utils.isaaclab import replicate_environment
from newton.utils.selection import ArticulationView


class Example:
    def __init__(self, stage_path=None, num_envs=8):
        self.num_envs = num_envs

        builder, stage_info = replicate_environment(
            newton.examples.get_asset("envs/humanoid_env.usd"),
            "/World/envs/env_0",
            "/World/envs/env_{}",
            num_envs,
            (5.0, 5.0, 0.0),
            # USD importer args
            collapse_fixed_joints=True,
            joint_ordering="dfs",
        )

        up_axis = stage_info.get("up_axis") or newton.Axis.Z

        # ===========================================================
        # Configure PD control for all joints
        # ===========================================================
        
        # Set joint modes to position control AND disable limits
        for i in range(len(builder.joint_axis_mode)):
            if builder.joint_axis_mode[i] != newton.JOINT_MODE_FORCE: 
                builder.joint_axis_mode[i] = newton.JOINT_MODE_TARGET_VELOCITY # if tracking only position use newton.JOINT_MODE_TARGET_POSITION, kd_gain will be used as  kd * q_dot
        
        for i in range(len(builder.joint_limit_lower)):
            builder.joint_limit_lower[i] = -100.0
            builder.joint_limit_upper[i] = 100.0

        kp_gain = 3.0
        kd_gain = 2.0
        
        for i in range(len(builder.joint_target_ke)):
            if builder.joint_axis_mode[i] == newton.JOINT_MODE_TARGET_VELOCITY: # also change here for position
                builder.joint_target_ke[i] = kp_gain
                builder.joint_target_kd[i] = kd_gain

        self.model = builder.finalize()
        self.model.ground = False
        
        # DISABLE GRAVITY 
        self.model.gravity = wp.vec3(0.0, 0.0, 0.0)

        self.solver = newton.solvers.MuJoCoSolver(self.model, integrator="implicit") # set implicit integrator to ensure that D term is interfated implicitly

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
        fps = 60.0
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.next_reset = 0.0
        self.humanoids = ArticulationView(self.model, "/World/envs/*/Robot/torso", include_free_joint=False)

        if self.humanoids.include_free_joint:
            # combined root and dof transforms
            self.default_transforms = wp.to_torch(self.humanoids.get_attribute("joint_q", self.model)).clone()
            self.default_transforms[:, 2] = 2.0  # Float in mid-air
            # Keep upright orientation (no rotation)
            self.default_transforms[:, 3] = 0.0  # qx
            self.default_transforms[:, 4] = 0.0  # qy 
            self.default_transforms[:, 5] = 0.0  # qz
            self.default_transforms[:, 6] = 1.0  # qw (identity quaternion)
            # combined root and dof velocities
            self.default_velocities = wp.to_torch(self.humanoids.get_attribute("joint_qd", self.model)).clone()
        else:
            self.default_root_transforms = wp.to_torch(self.humanoids.get_root_transforms(self.model)).clone()
            self.default_root_transforms[:, 2] = 2.0
            self.default_root_transforms[:, 3] = 0.0
            self.default_root_transforms[:, 4] = 0.0 
            self.default_root_transforms[:, 5] = 0.0 
            self.default_dof_transforms = wp.to_torch(self.humanoids.get_attribute("joint_q", self.model)).clone()
            self.default_root_velocities = wp.to_torch(self.humanoids.get_root_velocities(self.model)).clone()
            self.default_dof_velocities = wp.to_torch(self.humanoids.get_attribute("joint_qd", self.model)).clone()


        num_joints = self.humanoids.joint_dof_count
        if num_joints >= 12: 
            self.leg_joint_indices = list(range(min(12, num_joints)))
        else:
            self.leg_joint_indices = list(range(num_joints))

        # create disjoint index groups to alternate between
        all_indices = torch.arange(num_envs, dtype=torch.int32)
        self.indices_0 = all_indices[::2]
        self.indices_1 = all_indices[1::2]

        # reset all
        self.reset()
        self.next_reset = self.sim_time + 16.0  # Longer reset time to see full movement sequence

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # explicit collisions needed without MuJoCo solver
            if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
                newton.collision.collide(self.model, self.state_0)

            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time >= self.next_reset:
            self.reset(self.indices_0)
            self.next_reset = self.sim_time + 16.0
            self.indices_0, self.indices_1 = self.indices_1, self.indices_0


        # =========================
        # position controls
        # =========================
        target_positions = torch.zeros((self.num_envs, self.humanoids.joint_dof_count))
        frequency = 0.2  
        amplitude = 1.047 
        angle = amplitude * math.sin(2 * math.pi * frequency * self.sim_time)   
        target_positions[:, 10] = angle  # ONLY joint 10 (foot)
        
        for i in range(target_positions.shape[1]):
            if i != 10: 
                target_positions[:, i] = 0.0
   
        self.humanoids.set_attribute("joint_target", self.control, target_positions)

        # =========================
        # velocity controls
        # =========================
        target_velocities = torch.zeros((self.num_envs, self.humanoids.joint_dof_count))
        
        vel_frequency = 0.3
        vel_amplitude = 2.0  
        velocity_command = vel_amplitude * math.sin(2 * math.pi * vel_frequency * self.sim_time)
        target_velocities[:, 10] = velocity_command

        self.humanoids.set_attribute("joint_target_velocity", self.control, target_velocities)

        # =========================
        # apply random controls
        # =========================
        joint_forces = 20.0 - 40.0 * torch.rand((self.num_envs, self.humanoids.joint_dof_count))
        if self.humanoids.include_free_joint:
            joint_forces = torch.cat([torch.zeros((self.num_envs, 6)), joint_forces], axis=1)
            
        #self.humanoids.set_attribute("joint_f", self.control, joint_forces)   # uncomment to apply force as well

        with wp.ScopedTimer("step", active=False):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def reset(self, indices=None):
        print(f"Resetting environments at t={self.sim_time:.2f}s")
        
        # ==============================
        # set transforms and velocities
        # ==============================
        if self.humanoids.include_free_joint:
            # set root and dof transforms together
            self.humanoids.set_attribute("joint_q", self.state_0, self.default_transforms, indices=indices)
            # set root and dof velocities together
            self.humanoids.set_attribute("joint_qd", self.state_0, self.default_velocities, indices=indices)
        else:
            # set root and dof transforms separately
            self.humanoids.set_root_transforms(self.state_0, self.default_root_transforms, indices=indices)
            self.humanoids.set_attribute("joint_q", self.state_0, self.default_dof_transforms, indices=indices)
            # set root and dof velocities separately
            self.humanoids.set_root_velocities(self.state_0, self.default_root_velocities, indices=indices)
            self.humanoids.set_attribute("joint_qd", self.state_0, self.default_dof_velocities, indices=indices)

        if not isinstance(self.solver, newton.solvers.MuJoCoSolver):
            self.humanoids.eval_fk(self.state_0, indices=indices)

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
        default="example_selection_humanoid_pd.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=1, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for frame in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()