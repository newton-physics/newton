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
# Shows how to control Anymal C with a policy pretrained in physx.
#
# Example usage:
# uv run --extra cu12 newton/examples/example_anymal_c_walk_physx_policy.py
#
###########################################################################


import torch
import warp as wp

import newton
import newton.utils
from newton.sim import State

lab_to_mujoco = [9, 3, 6, 0, 10, 4, 7, 1, 11, 5, 8, 2]
mujoco_to_lab = [3, 7, 11, 1, 5, 9, 2, 6, 10, 0, 4, 8]


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.    Args:
    q: The quaternion in (x, y, z, w). Shape is (..., 4).
    v: The vector in (x, y, z). Shape is (..., 3).    Returns:
    The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(actions, state: State, joint_pos_initial, device, indices, gravity_vec, command):
    root_quat_w = torch.tensor(state.joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_w = torch.tensor(state.joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(state.joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    joint_pos_current = torch.tensor(state.joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(state.joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)
    return obs


class Example:
    def __init__(self, stage_path=None, headless=False):
        self.device = wp.get_device()
        # Convert Warp device to PyTorch device string
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        if stage_path is None:
            asset_path = newton.utils.download_asset("anymal_c_simple_description")
            stage_path = str(asset_path / "urdf" / "anymal.urdf")
        newton.utils.parse_urdf(
            stage_path,
            builder,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0e0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder.joint_q[:3] = [0.0, 0.0, 0.62]

        builder.joint_q[3:7] = [
            0.0,
            0.0,
            0.7071,
            0.7071,
        ]

        builder.joint_q[7:] = [
            0.0,
            -0.4,
            0.8,
            0.0,
            -0.4,
            0.8,
            0.0,
            0.4,
            -0.8,
            0.0,
            0.4,
            -0.8,
        ]
        for i in range(len(builder.joint_dof_mode)):
            builder.joint_dof_mode[i] = newton.JOINT_MODE_TARGET_POSITION

        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        self.model = builder.finalize()
        self.solver = newton.solvers.MuJoCoSolver(self.model)

        self.renderer = None if headless else newton.utils.SimRendererOpenGL(self.model, stage_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        newton.sim.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Pre-compute tensors that don't change during simulation
        self.lab_to_mujoco_indices = torch.tensor(
            [lab_to_mujoco[i] for i in range(len(lab_to_mujoco))], device=self.torch_device
        )
        self.mujoco_to_lab_indices = torch.tensor(
            [mujoco_to_lab[i] for i in range(len(mujoco_to_lab))], device=self.torch_device
        )
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.command[0, 0] = 1

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            torch_tensor = torch.zeros(18, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step"):
            obs = compute_obs(
                self.act,
                self.state_0,
                self.joint_pos_initial,
                self.torch_device,
                self.lab_to_mujoco_indices,
                self.gravity_vec,
                self.command,
            )
            with torch.no_grad():
                self.act = self.policy(obs)
                self.rearranged_act = torch.gather(self.act, 1, self.mujoco_to_lab_indices.unsqueeze(0))
                a = self.joint_pos_initial + 0.5 * self.rearranged_act
                a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
                a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
                wp.copy(
                    self.control.joint_target, a_wp
                )  # this can actually be optimized by doing  wp.copy(self.solver.mjw_data.ctrl[0], a_wp) and not launching  apply_mjc_control_kernel each step. Typically we update position and velocity targets at the rate of the outer control loop.
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
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
        help="Path to the output URDF file.",
    )
    parser.add_argument("--num-frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, headless=args.headless)

        # Download the policy from the newton-assets repository
        policy_asset_path = newton.utils.download_asset("anymal_c_policies")
        policy_path = str(policy_asset_path / "anymal_walking_policy_physx.pt")

        example.policy = torch.jit.load(policy_path, map_location=example.torch_device)
        example.joint_pos_initial = torch.tensor(
            example.state_0.joint_q[7:], device=example.torch_device, dtype=torch.float32
        ).unsqueeze(0)
        example.joint_vel_initial = torch.tensor(
            example.state_0.joint_qd[6:], device=example.torch_device, dtype=torch.float32
        )
        example.act = torch.zeros(1, 12, device=example.torch_device, dtype=torch.float32)
        example.rearranged_act = torch.zeros(1, 12, device=example.torch_device, dtype=torch.float32)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
