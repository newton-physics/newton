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
# Example Anymal C walk Coupled with Sand
#
# Shows Anymal C with a pretrained policy coupled with implicit mpm sand.
#
###########################################################################

import math
from typing import List

import numpy as np
import torch
import warp as wp

import newton
import newton.collision
import newton.core.articulation
import newton.examples
import newton.utils
from newton.core import Control, State
from newton.solvers.solver_implicit_mpm import ImplicitMPMSolver


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
    joint_axis_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_axis_start: wp.array(dtype=wp.int32),
    # outputs
    joint_f: wp.array(dtype=wp.float32),
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
        tq = Kp * (tq - q) - Kd * qd

        # skip the 6 dofs of the free joint
        joint_f[6 + aj] = tq


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
        obs_dim = 37
        self.obs_buf = wp.empty(
            (self.num_envs, obs_dim),
            dtype=wp.float32,
            device=self.device,
        )

    def compute_observations(
        self,
        state: State,
        observations: wp.array,
    ):
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
                self.Kp,
                self.Kd,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.model.joint_axis_dim,
                self.model.joint_axis_start,
            ],
            outputs=[
                control.joint_f,
            ],
            device=self.model.device,
        )

    def get_control(self, state: State, control: Control):
        self.compute_observations(state, self.obs_buf)
        obs_torch = wp.to_torch(self.obs_buf).detach()
        ctrl = wp.array(torch.clamp(self.policy_model(obs_torch).detach(), -1, 1), dtype=float)
        self.assign_control(ctrl, control, state)


@wp.kernel
def update_collider_mesh(
    src_points: wp.array(dtype=wp.vec3),
    src_shape: wp.array(dtype=int),
    res_mesh: wp.uint64,
    shape_transforms: wp.array(dtype=wp.transform),
    shape_body_id: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    dt: float,
):
    v = wp.tid()
    res = wp.mesh_get(res_mesh)

    shape_id = src_shape[v]
    p = wp.transform_point(shape_transforms[shape_id], src_points[v])

    X_wb = body_q[shape_body_id[shape_id]]

    cur_p = res.points[v] + dt * res.velocities[v]
    next_p = wp.transform_point(X_wb, p)
    res.velocities[v] = (next_p - cur_p) / dt
    res.points[v] = cur_p


class Example:
    def __init__(self, urdf_path: str, voxel_size=0.05, particles_per_cell=3, tolerance=1.0e-5, headless=False):
        self.device = wp.get_device()
        builder = newton.ModelBuilder(up_axis=newton.Axis.Y)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=2.0e3,
            kd=5.0e2,
            kf=1.0e2,
            mu=0.75,
        )

        newton.utils.parse_urdf(
            # newton.examples.get_asset("../../assets/anymal_c_simple_description/urdf/anymal.urdf"),
            urdf_path,
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

        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.start_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)

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

        # add sand particles

        max_fraction = 1.0
        particle_lo = np.array([-0.5, 0.0, -0.5])
        particle_hi = np.array([2.5, 0.15, 0.5])
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        _spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

        builder.set_ground_plane(offset=np.min(builder.particle_q[:, 1]))

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

        self.model.particle_mu = 0.48

        # lower the ground slightly for the sand and the renderer.
        # this makes the robot "float" a little bit, preventing impossible kinematic boundary conditions when the feet intersect the ground
        # proper solution will be to have full two-way coupling between the sand and the robot
        self.model.ground_plane_params = (
            *self.model.ground_plane_params[:-1],
            self.model.ground_plane_params[-1] - 0.025,
        )

        ## Grab meshes for collisions
        collider_body_idx = [idx for idx, key in enumerate(builder.body_key) if "SHANK" in key]
        collider_shape_ids = np.concatenate(
            [[m for m in self.model.body_shapes[b] if self.model.shape_geo_src[m]] for b in collider_body_idx]
        )

        collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
            [self.model.shape_geo_src[m].vertices for m in collider_shape_ids],
            [self.model.shape_geo_src[m].indices for m in collider_shape_ids],
            [self.model.shape_geo.scale.numpy()[m] for m in collider_shape_ids],
            collider_shape_ids,
        )

        self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        self.collider_rest_points = collider_points
        self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        self.solver = newton.solvers.FeatherstoneSolver(self.model)

        options = ImplicitMPMSolver.Options()
        options.voxel_size = voxel_size
        options.max_fraction = max_fraction
        options.tolerance = tolerance
        options.unilateral = False
        options.max_iterations = 50
        # options.gauss_seidel = False
        # options.dynamic_grid = False
        # options.grid_padding = 3

        self.mpm_solver = ImplicitMPMSolver(self.model, options)
        self.mpm_solver.setup_collider(self.model, [self.collider_mesh])

        self.renderer = None if headless else newton.utils.SimRendererOpenGL(self.model, urdf_path)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        newton.core.articulation.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._update_collider_mesh(self.state_0)

        self.control = self.model.control()
        self.controller = AnymalController(self.model, self.device)

        self.use_cuda_graph = self.device.is_cuda and wp.is_mempool_enabled(wp.get_device())
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.robot_graph = capture.graph
        else:
            self.robot_graph = None

    def simulate_robot(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.collision.collide(self.model, self.state_0)
            self.solver.step(self.model, self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        self._update_collider_mesh(self.state_0)
        # solve in-place, avoids having to resync robot sim state
        self.mpm_solver.step(self.model, self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        with wp.ScopedTimer("step", synchronize=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.robot_graph)
            else:
                self.simulate_robot()

            self.simulate_sand()
            self.controller.get_control(self.state_0, self.control)

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", synchronize=True):
            self.renderer.begin_frame(self.sim_time)

            self.renderer.render(self.state_0)

            self.renderer.end_frame()

    def _update_collider_mesh(self, state):
        wp.launch(
            update_collider_mesh,
            dim=self.collider_rest_points.shape[0],
            inputs=[
                self.collider_rest_points,
                self.collider_shape_ids,
                self.collider_mesh.id,
                self.model.shape_transform,
                self.model.shape_body,
                state.body_q,
                self.frame_dt,
            ],
        )
        self.collider_mesh.refit()


def _spawn_particles(builder: newton.ModelBuilder, res, bounds_lo, bounds_hi, packing_fraction):
    Nx = res[0]
    Ny = res[1]
    Nz = res[2]

    px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
    py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
    pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

    points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

    cell_size = (bounds_hi - bounds_lo) / res
    cell_volume = np.prod(cell_size)

    radius = np.max(cell_size) * 0.5
    volume = np.prod(cell_volume) * packing_fraction

    points += 2.0 * radius * (np.random.rand(*points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel
    builder.particle_mass = np.full(points.shape[0], volume)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.zeros(points.shape[0], dtype=int)

    print("Particle count: ", points.shape[0])


def _merge_meshes(
    points: List[np.array],
    indices: List[np.array],
    scales: List[np.array],
    shape_ids: List[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "urdf_path",
        type=lambda x: None if x == "None" else str(x),
        help="Path to the Anymal C URDF file from newton-assets.",
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=10000, help="Total number of frames.")
    parser.add_argument("--voxel_size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles_per_cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction)

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            urdf_path=args.urdf_path,
            voxel_size=args.voxel_size,
            particles_per_cell=args.particles_per_cell,
            tolerance=args.tolerance,
            headless=args.headless,
        )

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
