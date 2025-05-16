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
# Anymal C demo
#
###########################################################################

import math

import numpy as np
import torch
import warp as wp

import newton
import newton.collision
import newton.core.articulation
import newton.examples
import newton.utils
from newton.core import Control, State


@wp.kernel
def compute_observations_anymal_dflex(
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    basis_vec0: wp.vec3,
    basis_vec1: wp.vec3,
    dof_q: int,
    dof_qd: int,
    # outputs
    obs: wp.array(dtype=float, ndim=2),
):
    env_id = wp.tid()

    torso_pos = wp.vec3(
        joint_q[dof_q * env_id + 0],
        joint_q[dof_q * env_id + 1],
        joint_q[dof_q * env_id + 2],
    )
    torso_quat = wp.quat(
        joint_q[dof_q * env_id + 3],
        joint_q[dof_q * env_id + 4],
        joint_q[dof_q * env_id + 5],
        joint_q[dof_q * env_id + 6],
    )
    lin_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 3],
        joint_qd[dof_qd * env_id + 4],
        joint_qd[dof_qd * env_id + 5],
    )
    ang_vel = wp.vec3(
        joint_qd[dof_qd * env_id + 0],
        joint_qd[dof_qd * env_id + 1],
        joint_qd[dof_qd * env_id + 2],
    )

    # convert the linear velocity of the torso from twist representation to the velocity of the center of mass in world frame
    lin_vel = lin_vel - wp.cross(torso_pos, ang_vel)

    up_vec = wp.quat_rotate(torso_quat, basis_vec1)
    heading_vec = wp.quat_rotate(torso_quat, basis_vec0)

    obs[env_id, 0] = torso_pos[1]  # 0
    for i in range(4):  # 1:5
        obs[env_id, 1 + i] = torso_quat[i]
    for i in range(3):  # 5:8
        obs[env_id, 5 + i] = lin_vel[i]
    for i in range(3):  # 8:11
        obs[env_id, 8 + i] = ang_vel[i]
    for i in range(12):  # 11:23
        obs[env_id, 11 + i] = joint_q[dof_q * env_id + 7 + i]
    for i in range(12):  # 23:35
        obs[env_id, 23 + i] = joint_qd[dof_qd * env_id + 6 + i]
    obs[env_id, 35] = up_vec[1]  # 35
    obs[env_id, 36] = heading_vec[0]  # 36


class T(torch.nn.Module):
    def __init__(self, mean, var):
        super().__init__()
        self.mean = mean
        self.var = var

    def forward(self, obs):
        return torch.clamp((obs - self.mean) / torch.sqrt(1e-5 + self.var), min=-5, max=5).float()


@wp.kernel
def apply_joint_position_pd_control(
    actions: wp.array(dtype=wp.float32, ndim=1),
    action_scale: wp.float32,
    default_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    joint_torque_limit: wp.array(dtype=wp.float32),
    Kp: wp.float32,
    Kd: wp.float32,
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis_start: wp.array(dtype=wp.int32),
    # outputs
    target_joint_q: wp.array(dtype=wp.float32),
    joint_act: wp.array(dtype=wp.float32),
):
    joint_id = wp.tid()
    ai = joint_axis_start[joint_id]
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    dim = joint_axis_dim[joint_id, 0] + joint_axis_dim[joint_id, 1]
    for j in range(dim):
        qj = qi + j
        qdj = qdi + j
        aj = ai + j
        q = joint_q[qj]
        qd = joint_qd[qdj]

        tq = wp.clamp(actions[aj], -1.0, 1.0) * action_scale + default_joint_q[qj]

        target_joint_q[aj] = tq

        tq = Kp * (tq - q) - Kd * qd
        # tq = wp.clamp(tq, -joint_torque_limit[aj], joint_torque_limit[aj])

        joint_act[aj] = tq


def load_sequential_policy(obs_dim, hidden_dims, action_dim, state_dict, prefix=""):
    """Create a sequential model and load the policy"""
    layers = []
    cur_dim = obs_dim

    for hidden_dim in hidden_dims:
        layers.append(torch.nn.Linear(cur_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
        cur_dim = hidden_dim

    layers.append(torch.nn.Linear(cur_dim, action_dim))
    net = torch.nn.Sequential(*layers)
    p = len(prefix)
    layers = {name[p:]: data for name, data in state_dict.items() if name.startswith(prefix)}
    net.load_state_dict(layers)
    return net


class Example:
    def __init__(self, stage_path="example_quadruped.usd", num_envs=8):
        self.device = wp.get_device()
        builder = newton.ModelBuilder()

        newton.utils.parse_urdf(
            newton.examples.get_asset("../../assets/anymal_c_simple_description/urdf/anymal.urdf"),
            builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 60
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.control_dim = 12
        action_strength = 150.0
        self.control_gains_wp = wp.array(
            np.array(
                [
                    50.0,  # LF_HAA
                    40.0,  # LF_HFE
                    8.0,  # LF_KFE
                    50.0,  # RF_HAA
                    40.0,  # RF_HFE
                    8.0,  # RF_KFE
                    50.0,  # LH_HAA
                    40.0,  # LH_HFE
                    8.0,  # LH_KFE
                    50.0,  # RH_HAA
                    40.0,  # RH_HFE
                    8.0,  # RH_KFE
                ]
            )
            * action_strength
            / 100.0,
            dtype=float,
        )
        self.num_envs = num_envs

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)

        self.basis_vec0 = wp.vec3(1.0, 0.0, 0.0)
        self.basis_vec1 = wp.vec3(0.0, 0.0, 1.0)

        self.bodies_per_env = 1
        self.dof_q_per_env = builder.joint_coord_count
        self.dof_qd_per_env = builder.joint_dof_count

        builder.joint_q[:7] = [
            0.0,
            0.7,
            0.0,
            *self.start_rot,
        ]

        builder.joint_q[7:] = [
            0.03,  # LF_HAA
            0.4,  # LF_HFE
            -0.8,  # LF_KFE
            -0.03,  # RF_HAA
            0.4,  # RF_HFE
            -0.8,  # RF_KFE
            0.03,  # LH_HAA
            -0.4,  # LH_HFE
            0.8,  # LH_KFE
            -0.03,  # RH_HAA
            -0.4,  # RH_HFE
            0.8,  # RH_KFE
        ]

        for i in range(builder.joint_axis_count):
            builder.joint_axis_mode[i] = newton.core.JOINT_MODE_FORCE

        self.torques = wp.zeros(
            self.num_envs * builder.joint_axis_count,
            dtype=wp.float32,
            device=self.device,
        )

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.default_joint_q = self.model.joint_q
        self.joint_torque_limit = self.control_gains_wp
        self.action_scale = 0.5
        self.Kp = 85.0
        self.Kd = 2.0

        self.target_joint_q = wp.empty((self.num_envs * self.control_dim), dtype=wp.float32, device=self.device)

        self.policy_model = torch.load(newton.examples.get_asset("anymal_walking_policy.pt"), weights_only=False).cuda()

        self.solver = newton.solvers.FeatherstoneSolver(self.model)

        self.renderer = newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        obs_dim = 37
        self.obs_buf = wp.empty(
            (1, obs_dim),
            dtype=wp.float32,
            device=self.device,
        )

        newton.core.articulation.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None
        self.compute_observations(self.state_0, self.control, self.obs_buf)

    def compute_observations(
        self,
        state: State,
        control: Control,
        observations: wp.array,
    ):
        # dflex observations
        wp.launch(
            compute_observations_anymal_dflex,
            dim=self.num_envs,
            inputs=[
                state.joint_q,
                state.joint_qd,
                self.basis_vec0,
                self.basis_vec1,
                self.dof_q_per_env,
                self.dof_qd_per_env,
            ],
            outputs=[observations],
            device=self.device,
        )

    def assign_control(self, actions: wp.array, control: Control, state: State):
        wp.launch(
            kernel=apply_joint_position_pd_control,
            dim=self.model.joint_count,
            inputs=[
                wp.from_torch(wp.to_torch(actions).reshape(-1)),
                self.action_scale,
                self.default_joint_q,
                state.joint_q,
                state.joint_qd,
                self.joint_torque_limit,
                self.Kp,
                self.Kd,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_axis_dim,
                self.model.joint_axis_start,
            ],
            outputs=[self.target_joint_q, control.joint_act],
            device=self.model.device,
        )

    def simulate(self):
        self.compute_observations(self.state_0, self.control, self.obs_buf)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.collision.collide(self.model, self.state_0)
            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        obs_torch = wp.to_torch(self.obs_buf).detach()
        ctrl = wp.array(self.policy_model(obs_torch).detach(), dtype=float)
        print(ctrl.numpy())
        self.assign_control(ctrl, self.control, self.state_0)
        self.sim_time += self.frame_dt

        print("Joint_act:", self.control.joint_act.numpy())
        print("Observations:", self.obs_buf.numpy())
        # wp.launch(fill_array, 12, inputs=[self.control.joint_act, wp.array(np.random.randn(12), dtype=float)])

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
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
        default="example_quadruped.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
