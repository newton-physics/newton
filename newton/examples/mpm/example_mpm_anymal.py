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
# Example MPM Anymal
#
# Shows Anymal C with a pretrained policy coupled with implicit MPM sand.
#
# Example usage (via unified runner):
#   python -m newton.examples mpm_anymal --viewer gl
###########################################################################

import sys

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM

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


def compute_obs(actions, state: newton.State, joint_pos_initial, indices, gravity_vec, command):
    q = wp.to_torch(state.joint_q)
    qd = wp.to_torch(state.joint_qd)
    root_quat_w = q[3:7].unsqueeze(0)
    root_lin_vel_w = qd[3:6].unsqueeze(0)
    root_ang_vel_w = qd[:3].unsqueeze(0)
    joint_pos_current = q[7:].unsqueeze(0)
    joint_vel_current = qd[6:].unsqueeze(0)
    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)
    return obs


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
    def __init__(
        self,
        viewer,
        voxel_size=0.05,
        particles_per_cell=3,
        tolerance=1.0e-5,
        sand_friction=0.48,
        dynamic_grid=True,
    ):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.device = wp.get_device()

        # import the robot model
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

        asset_path = newton.utils.download_asset("anybotics_anymal_c")
        builder.add_urdf(
            str(asset_path / "urdf" / "anymal.urdf"),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        builder.add_ground_plane()

        # setup robot joint properties
        builder.joint_q[:3] = [0.0, 0.0, 0.62]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
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
            builder.joint_dof_mode[i] = newton.JointMode.TARGET_POSITION
        for i in range(len(builder.joint_target_ke)):
            builder.joint_target_ke[i] = 150
            builder.joint_target_kd[i] = 5

        # add sand particles
        max_fraction = 1.0
        particle_lo = np.array([-0.5, -0.5, 0.0])  # emission lower bound
        particle_hi = np.array([0.5, 2.5, 0.15])  # emission upper bound
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )
        _spawn_particles(builder, particle_res, particle_lo, particle_hi, max_fraction)

        # finalize model
        self.model = builder.finalize()
        self.model.particle_mu = sand_friction

        # Select and merge meshes for robot/sand collisions
        collider_body_idx = [idx for idx, key in enumerate(builder.body_key) if "SHANK" in key]
        collider_shape_ids = np.concatenate(
            [[m for m in self.model.body_shapes[b] if self.model.shape_source[m]] for b in collider_body_idx]
        )

        collider_points, collider_indices, collider_v_shape_ids = _merge_meshes(
            [self.model.shape_source[m].vertices for m in collider_shape_ids],
            [self.model.shape_source[m].indices for m in collider_shape_ids],
            [self.model.shape_scale.numpy()[m] for m in collider_shape_ids],
            collider_shape_ids,
        )

        self.collider_mesh = wp.Mesh(wp.clone(collider_points), collider_indices, wp.zeros_like(collider_points))
        self.collider_rest_points = collider_points
        self.collider_shape_ids = wp.array(collider_v_shape_ids, dtype=int)

        # setup solvers
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # setup mpm solver
        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.max_fraction = max_fraction
        mpm_options.tolerance = tolerance
        mpm_options.unilateral = False
        mpm_options.max_iterations = 50
        mpm_options.dynamic_grid = dynamic_grid
        if not dynamic_grid:
            mpm_options.grid_padding = 5

        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)
        self.mpm_solver.setup_collider(self.model, [self.collider_mesh])

        # simulation state
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.mpm_solver.enrich_state(self.state_0)
        self.mpm_solver.enrich_state(self.state_1)

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._update_collider_mesh(self.state_0)

        # Setup control policy
        self.control = self.model.control()

        q0 = wp.to_torch(self.state_0.joint_q)
        self.torch_device = q0.device
        self.joint_pos_initial = q0[7:].unsqueeze(0).detach().clone()
        self.act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, 12, device=self.torch_device, dtype=torch.float32)

        # Download the policy from the newton-assets repository
        policy_path = str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt")
        self.policy = torch.jit.load(policy_path, map_location=self.torch_device)

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

        # set model on viewer and setup capture
        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate_robot()
            self.graph = capture.graph

    def apply_control(self):
        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial,
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
            # copy action targets to control buffer
            wp.copy(self.control.joint_target, a_wp)

    def simulate_robot(self):
        # robot substeps
        self.contacts = self.model.collide(self.state_0, rigid_contact_margin=0.1)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate_sand(self):
        # sand step (in-place on frame dt)
        self._update_collider_mesh(self.state_0)
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def step(self):
        # compute control before graph/step
        self.apply_control()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate_robot()

        # MPM solver step is not graph-capturable yet
        self.simulate_sand()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

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

    rng = np.random.default_rng()
    points += 2.0 * radius * (rng.random(points.shape) - 0.5)
    vel = np.zeros_like(points)

    builder.particle_q = points
    builder.particle_qd = vel
    builder.particle_mass = np.full(points.shape[0], volume)
    builder.particle_radius = np.full(points.shape[0], radius)
    builder.particle_flags = np.zeros(points.shape[0], dtype=int)

    print("Particle count: ", points.shape[0])


def _merge_meshes(
    points: list[np.array],
    indices: list[np.array],
    scales: list[np.array],
    shape_ids: list[int],
):
    pt_count = np.array([len(pts) for pts in points])
    offsets = np.cumsum(pt_count) - pt_count

    mesh_id = np.repeat(np.arange(len(points), dtype=int), repeats=pt_count)

    merged_points = np.vstack([pts * scale for pts, scale in zip(points, scales, strict=False)])

    merged_indices = np.concatenate([idx + offsets[k] for k, idx in enumerate(indices)])

    return (
        wp.array(merged_points, dtype=wp.vec3),
        wp.array(merged_indices, dtype=int),
        wp.array(np.array(shape_ids)[mesh_id], dtype=int),
    )


if __name__ == "__main__":
    import argparse

    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.03)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=3.0)
    parser.add_argument("--sand-friction", "-mu", type=float, default=0.48)
    parser.add_argument("--tolerance", "-tol", type=float, default=1.0e-5)
    parser.add_argument("--dynamic-grid", action=argparse.BooleanOptionalAction, default=True)

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # This example requires a GPU device
    if wp.get_device().is_cpu:
        print("Error: This example requires a GPU device.")
        sys.exit(1)

    # Create example and load policy
    example = Example(
        viewer,
        voxel_size=args.voxel_size,
        particles_per_cell=args.particles_per_cell,
        tolerance=args.tolerance,
        sand_friction=args.sand_friction,
        dynamic_grid=args.dynamic_grid,
    )

    # Run via unified example runner
    newton.examples.run(example)
