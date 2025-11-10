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
# Example Robot ANYmal C Walk
#
# Shows how to simulate ANYmal C using SolverMuJoCo and control it with a
# policy trained in PhysX.
#
# Command: python -m newton.examples robot_anymal_c_walk
#
###########################################################################

import torch
import warp as wp
from warp.torch import device_to_torch

wp.config.enable_backward = False

import newton
import newton.examples
import newton.utils
from newton import State
from newton.geometry import generate_terrain_grid

lab_to_mujoco = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
mujoco_to_lab = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


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
    root_lin_vel_w = torch.tensor(state.joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(state.joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
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
    def __init__(self, viewer, args=None):
        self.viewer = viewer
        self.device = wp.get_device()
        self.torch_device = device_to_torch(self.device)
        self.is_test = args is not None and args.test

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        stage_path = str(asset_path / "urdf" / "anymal.urdf")
        builder.add_urdf(
            stage_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.62), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5)),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Generate procedural terrain for visual demonstration (but not during unit tests)
        if not self.is_test:
            vertices, indices = generate_terrain_grid(
                grid_size=(8, 3),  # 3x8 grid for forward walking
                block_size=(3.0, 3.0),
                terrain_types=["random_grid", "flat", "wave", "gap", "pyramid_stairs"],
                terrain_params={
                    "pyramid_stairs": {"step_width": 0.3, "step_height": 0.02, "platform_width": 0.6},
                    "random_grid": {"grid_width": 0.3, "grid_height_range": (0, 0.02)},
                    "wave": {"wave_amplitude": 0.15, "wave_frequency": 2.0},
                },
                seed=42,
            )
            terrain_mesh = newton.Mesh(vertices, indices)
            terrain_offset = wp.transform(p=wp.vec3(-5, -2.0, 0.01), q=wp.quat_identity())
            builder.add_shape_mesh(body=-1, mesh=terrain_mesh, xform=terrain_offset)
        builder.add_ground_plane()

        self.sim_time = 0.0
        self.sim_step = 0
        fps = 50
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        # set initial joint positions
        initial_q = {
            "RH_HAA": 0.0,
            "RH_HFE": -0.4,
            "RH_KFE": 0.8,
            "LH_HAA": 0.0,
            "LH_HFE": -0.4,
            "LH_KFE": 0.8,
            "RF_HAA": 0.0,
            "RF_HFE": 0.4,
            "RF_KFE": -0.8,
            "LF_HAA": 0.0,
            "LF_HFE": 0.4,
            "LF_KFE": -0.8,
        }
        # Set initial joint positions (skip first 7 position coordinates which are the free joint), e.g. for "LF_HAA" value will be written at index 1+6 = 7.
        for key, value in initial_q.items():
            builder.joint_q[builder.joint_key.index(key) + 6] = value

        for i in range(len(builder.joint_target_ke)):
                builder.joint_target_ke[i] = 150
                builder.joint_target_kd[i] = 5

        self.model = builder.finalize()

        # Create collision pipeline for terrain mesh collisions
        self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
            self.model,
            rigid_contact_max_per_pair=10,
            rigid_contact_margin=0.01,
            broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
        )

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            use_mujoco_contacts=False,  # Use Newton contacts from collision pipeline
            ls_parallel=True,
            njmax=50,
        )

        self.viewer.set_model(self.model)

        self.follow_cam = True

        if isinstance(self.viewer, newton.viewer.ViewerGL):

            def toggle_follow_cam(imgui):
                changed, follow_cam = imgui.checkbox("Follow Camera", self.follow_cam)
                if changed:
                    self.follow_cam = follow_cam

            self.viewer.register_ui_callback(toggle_follow_cam, position="side")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Evaluate forward kinematics to update body poses based on initial joint configuration
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Initialize contacts using collision pipeline
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        # Download the policy from the newton-assets repository
        policy_asset_path = newton.utils.download_asset("anybotics_anymal_c")
        policy_path = str(policy_asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")

        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)
        self.joint_pos_initial = torch.tensor(
            self.state_0.joint_q[7:], device=self.torch_device, dtype=torch.float32
        ).unsqueeze(0)
        self.joint_vel_initial = torch.tensor(self.state_0.joint_qd[6:], device=self.torch_device, dtype=torch.float32)
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        # Pre-compute tensors that don't change during simulation
        self.lab_to_mujoco_indices = torch.tensor(lab_to_mujoco, device=self.torch_device)
        self.mujoco_to_lab_indices = torch.tensor(mujoco_to_lab, device=self.torch_device)
        self.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=self.torch_device, dtype=torch.float32)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.command[0, 0] = 1

        self.capture()

    def capture(self):
        if self.device.is_cuda:
            torch_tensor = torch.zeros(18, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
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

            # Compute contacts using collision pipeline for terrain mesh
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
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
                self.control.joint_target_pos, a_wp
            )  # this can actually be optimized by doing  wp.copy(self.solver.mjw_data.ctrl[0], a_wp) and not launching  apply_mjc_control_kernel each step. Typically we update position and velocity targets at the rate of the outer control loop.
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.follow_cam:
            self.viewer.set_camera(
                pos=wp.vec3(*self.state_0.joint_q.numpy()[:3]) + wp.vec3(10.0, 0.0, 2.0), pitch=0.0, yaw=-180.0
            )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test(self):
        assert self.model.body_key == [
            "base",
            "LF_HIP",
            "LF_THIGH",
            "LF_SHANK",
            "RF_HIP",
            "RF_THIGH",
            "RF_SHANK",
            "LH_HIP",
            "LH_THIGH",
            "LH_SHANK",
            "RH_HIP",
            "RH_THIGH",
            "RH_SHANK",
        ]
        assert self.model.joint_key == [
            "floating_base",
            "LF_HAA",
            "LF_HFE",
            "LF_KFE",
            "RF_HAA",
            "RF_HFE",
            "RF_KFE",
            "LH_HAA",
            "LH_HFE",
            "LH_KFE",
            "RH_HAA",
            "RH_HFE",
            "RH_KFE",
        ]

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.1,
        )

        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot went in the right direction",
            lambda q, qd: q[1] > 9.0,  # This threshold assumes 500 frames
        )

        forward_vel_min = wp.spatial_vector(-0.5, 0.9, -0.2, -0.8, -0.5, -0.5)
        forward_vel_max = wp.spatial_vector(0.5, 1.1, 0.2, 0.8, 0.5, 0.5)
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "the robot is moving forward and not falling",
            lambda q, qd: newton.utils.vec_inside_limits(qd, forward_vel_min, forward_vel_max),
            indices=[0],
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    example = Example(viewer, args)

    newton.examples.run(example, args)
