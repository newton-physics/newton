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
# Example Anymal C walk
#
# Shows how to control Anymal C with a pretrained policy.
#
# Example usage:
# uv run --extra cu12 newton/examples/example_anymal_c_walk.py
#
###########################################################################

import sys

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.solvers.euler.kernels  # For graph capture on CUDA <12.3
import newton.utils
from newton.sim import Control, State


@wp.kernel
def compute_observations_anymal(
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


@wp.kernel
def apply_joint_position_pd_control(
    actions: wp.array(dtype=wp.float32, ndim=1),
    action_scale: wp.float32,
    default_joint_q: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    Kp: wp.float32,
    Kd: wp.float32,
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    joint_f: wp.array(dtype=wp.float32),
):
    joint_id = wp.tid()
    if joint_id == 0:
        return  # skip the free joint
    qi = joint_q_start[joint_id]
    qdi = joint_qd_start[joint_id]
    dim = joint_dof_dim[joint_id, 0] + joint_dof_dim[joint_id, 1]
    for j in range(dim):
        qj = qi + j
        qdj = qdi + j
        q = joint_q[qj]
        qd = joint_qd[qdj]

        tq = wp.clamp(actions[qdj - 6], -1.0, 1.0) * action_scale + default_joint_q[qj]
        tq = Kp * (tq - q) - Kd * qd

        joint_f[qdj] = tq


class AnymalController:
    """Controller for Anymal with pretrained policy."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
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
        self.action_scale = 0.5
        self.Kp = 140.0
        self.Kd = 2.0
        self.joint_torque_limit = self.control_gains_wp
        self.default_joint_q = self.model.joint_q

        self.basis_vec0 = wp.vec3(1.0, 0.0, 0.0)
        self.basis_vec1 = wp.vec3(0.0, 0.0, 1.0)

        self.policy_model = torch.jit.load(newton.examples.get_asset("anymal_walking_policy.pt")).cuda()

        self.dof_q_per_env = model.joint_coord_count
        self.dof_qd_per_env = model.joint_dof_count
        self.num_envs = 1
        self.ctrl = None
        obs_dim = 37
        self.obs_buf = wp.empty(
            (self.num_envs, obs_dim),
            dtype=wp.float32,
            device=self.device,
        )

    def compute_observations(self, state: State, observations: wp.array):
        wp.launch(
            compute_observations_anymal,
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

    def assign_control(self, control: Control, state: State):
        wp.launch(
            kernel=apply_joint_position_pd_control,
            dim=self.model.joint_count,
            inputs=[
                wp.from_torch(wp.to_torch(self.ctrl).reshape(-1)),
                self.action_scale,
                self.default_joint_q,
                state.joint_q,
                state.joint_qd,
                self.Kp,
                self.Kd,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_dof_dim,
            ],
            outputs=[
                control.joint_f,
            ],
            device=self.model.device,
        )

    def get_control(self, state: State):
        self.compute_observations(state, self.obs_buf)
        obs_torch = wp.to_torch(self.obs_buf).detach()
        self.ctrl = wp.array(torch.clamp(self.policy_model(obs_torch).detach(), -1, 1), dtype=float)


class Example:
    def __init__(self, stage_path="example_anymal_c_walk.usd", headless=False):
        self.device = wp.get_device()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anymal_c_simple_description")

        newton.utils.parse_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            builder,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 60
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [
            0.0,
            0.7,
            0.0,
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

        # finalize model
        self.model = builder.finalize()

        # the policy was trained with the following inertia tensors
        # fmt: off
        self.model.body_inertia = wp.array(
            [
                [[1.30548920,  0.00067627, 0.05068519], [ 0.000676270, 2.74363500,  0.00123380], [0.05068519,  0.00123380, 2.82926230]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109795, 0.05645556], [-0.001097950, 0.20255709, -0.00183889], [0.05645556, -0.00183889, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474725], [-0.008252360, 0.01629626,  0.00638789], [0.00474725,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111040, 0.05645556], [ 0.001110400, 0.20255709,  0.00183910], [0.05645556,  0.00183910, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368, -0.00825236, 0.00474726], [-0.008252360, 0.01629626,  0.00638789], [0.00474726,  0.00638789, 0.02370901]],
                [[0.18137439,  0.00111041, 0.05645556], [ 0.001110410, 0.20255709,  0.00183909], [0.05645556,  0.00183909, 0.02763401]],
                [[0.03070243, -0.00022458, 0.00102368], [-0.000224580, 0.02828139,  0.00652076], [0.00102368,  0.00652076, 0.00269065]],
                [[0.01809368,  0.00826303, 0.00475366], [ 0.008263030, 0.01629626, -0.00638789], [0.00475366, -0.00638789, 0.02370901]],
                [[0.18137439, -0.00109796, 0.05645556], [-0.001097960, 0.20255709, -0.00183888], [0.05645556, -0.00183888, 0.02763401]],
                [[0.03070243,  0.00022458, 0.00102368], [ 0.000224580, 0.02828139, -0.00652076], [0.00102368, -0.00652076, 0.00269065]]
            ],
            dtype=wp.mat33f,
        )
        self.model.body_mass = wp.array([27.99286, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505, 2.51203, 3.27327, 0.55505], dtype=wp.float32,)
        # fmt: on

        self.solver = newton.solvers.FeatherstoneSolver(self.model)
        self.renderer = None if headless else newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self.controller = AnymalController(self.model, self.device)
        self.controller.get_control(self.state_0)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            # Initial graph launch, load modules (necessary for drivers prior to CUDA 12.3)
            wp.load_module(newton.solvers.euler.kernels, device=wp.get_device())

            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.controller.assign_control(self.control, self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.controller.get_control(self.state_0)
        self.sim_time += self.frame_dt

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
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default="example_anymal_c_walk.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    if wp.get_device(args.device).is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
