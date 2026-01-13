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
# Example MPM Humanoid 2-Way Coupling
#
# Shows Unitree G1 locomotion with a pretrained policy coupled with implicit MPM sand.
#
# Example usage (via unified runner):
#   uv run newton/examples/mpm/example_mpm_g1_twoway_coupling.py
###########################################################################

import sys
import os
from collections.abc import Sequence
import yaml

import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverImplicitMPM

from newton.examples.utils import CircularBuffer, find_physx_mjwarp_mapping


"""
math utils
"""

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
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


"""
kernels
"""

@wp.kernel
def compute_body_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Compute forces applied by sand to rigid bodies.

    Sum the impulses applied on each mpm grid node and convert to
    forces and torques at the body's center of mass.
    """

    i = wp.tid()

    cid = collider_ids[i]
    if cid >= 0 and cid < body_ids.shape[0]:
        body_index = body_ids[cid]
        if body_index == -1:
            return

        f_world = collider_impulses[i] / dt

        X_wb = body_q[body_index]
        X_com = body_com[body_index]
        r = collider_impulse_pos[i] - wp.transform_point(X_wb, X_com)
        body_wrench = wp.spatial_vector(f_world, wp.cross(r, f_world))
        wp.atomic_add(body_f, body_index, body_wrench)


@wp.kernel
def subtract_body_force(
    dt: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_f: wp.array(dtype=wp.spatial_vector),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    body_q_res: wp.array(dtype=wp.transform),
    body_qd_res: wp.array(dtype=wp.spatial_vector),
):
    """Update the rigid bodies velocity to remove the forces applied by sand at the last step.

    This is necessary to compute the total impulses that are required to enforce the complementarity-based
    frictional contact boundary conditions.
    """

    body_id = wp.tid()

    # Remove previously applied force
    f = body_f[body_id]
    inv_mass = wp.where(body_mass[body_id] > 0.0, 1.0 / body_mass[body_id], 0.0)
    delta_v = dt * inv_mass * wp.spatial_top(f)
    r = wp.transform_get_rotation(body_q[body_id])

    delta_w = dt * wp.quat_rotate(r, body_inv_inertia[body_id] * wp.quat_rotate_inv(r, wp.spatial_bottom(f)))

    body_q_res[body_id] = body_q[body_id]
    body_qd_res[body_id] = body_qd[body_id] - wp.spatial_vector(delta_v, delta_w)



"""
Env class
"""

class NewtonEnv:
    def __init__(
        self,
        viewer,
        config,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
    ):
        self.config = config

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        # sim time step track
        self.sim_time = 0.0
        self.sim_step = 0

        self.viewer = viewer
        self.history_length = self.config.get("history_length", 10)

        """
        setup rigid body builder
        """
        builder = newton.ModelBuilder()
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06,
            limit_ke=1.0e3,
            limit_kd=1.0e1,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        # add robot
        self.add_robot(builder)
        # add ground plane
        builder.add_ground_plane()

        """
        setup sand builder
        """
        sand_builder = newton.ModelBuilder()
        voxel_size = 0.03
        voxel_size_vis = 0.04  # for rendering
        self.add_sand(sand_builder, voxel_size=voxel_size_vis)

        """
        finalize models
        """
        self.model = builder.finalize()
        self.sand_model = sand_builder.finalize()

        # device setup
        self.device = self.model.device
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        """
        setup MPM solver
        """
        tolerance=1.0e-6
        grid_type = 'fixed'  # 'fixed' or 'sparse'

        mpm_options = SolverImplicitMPM.Options()
        mpm_options.voxel_size = voxel_size
        mpm_options.tolerance = tolerance
        mpm_options.transfer_scheme = "pic"
        mpm_options.collider_basis = "pic27"
        mpm_options.collider_velocity_mode = "finite_difference" 
        mpm_options.grid_type = grid_type
        mpm_options.grid_padding = 50
        mpm_options.max_active_cell_count = 2**16

        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.young_modulus = 15.0e6 # 15-20 MPa is reasonable range

        # particle internal friction
        self.sand_model.particle_mu = 0.48

        mpm_model = SolverImplicitMPM.Model(self.sand_model, mpm_options)
        # read colliders from the RB model rather than the sand model

        # NOTE: 
        # G1's foot mass is 0.608kg. and total mass is 32.34 kg.
        # assume 2 foot support total weight.
        coupling_relaxation = 26.6 # (32.34/2)/0.608 = 26.6
        mpm_model.setup_collider(
            model=self.model, 
            # body_mass=wp.zeros_like(self.model.body_mass) # kinematic collider
            body_mass=self.model.body_mass * coupling_relaxation,
            body_inv_inertia=self.model.body_inv_inertia / coupling_relaxation,
        )
        self.mpm_solver = SolverImplicitMPM(mpm_model, mpm_options)

        """
        setup rigid body solver
        """
        self.rb_solver = newton.solvers.SolverMuJoCo(self.model, ls_parallel=True, njmax=500)

        """
        prepare simulation states
        """

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.sand_state_0 = self.sand_model.state()
        self.mpm_solver.enrich_state(self.sand_state_0)

        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        """
        setup viewer
        """
        self.viewer.set_model(self.model)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.register_ui_callback(self.render_ui, position="side")
        self.viewer.show_particles = True
        self.show_impulses = True
        self.follow_cam = True

        # not required for MuJoCo, but required for other solvers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)
        self._initial_sand_q = wp.clone(self.sand_state_0.particle_q)
        self._initial_sand_qd = wp.clone(self.sand_state_0.particle_qd)

        # Additional buffers for tracking two-way coupling forces
        max_nodes = 2**16 # max number of collider nodes to track
        self.collider_impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=self.model.device)
        self.collider_impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=self.model.device)
        self.collect_collider_impulses()

        # map from collider index to body index
        self.collider_body_id = mpm_model.collider.collider_body_index

        # per-body forces and torques applied by sand to rigid bodies
        self.body_sand_forces = wp.zeros_like(self.state_0.body_f)

        self.particle_render_colors = wp.full(
            self.sand_model.particle_count, 
            # value=wp.vec3(0.27, 0.24, 0.21), 
            value=wp.vec3(0.7, 0.6, 0.4),
            dtype=wp.vec3, 
            device=self.sand_model.device
        )
        self.capture()
        
        """
        policy setups
        """
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False
        # create observation buffer
        self.create_obs_buffer()
        # Load policy and setup tensors
        self.load_policy_and_setup_tensors()


    """
    initializations
    """

    def add_robot(self, builder: newton.ModelBuilder):
        asset_path = self.config["asset_path"]
        if asset_path.endswith(".usd"):
            builder.add_usd(
                source=self.config["asset_path"], 
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                collapse_fixed_joints=False,
                enable_self_collisions=False,
                joint_ordering="dfs",
                hide_collision_shapes=True,
            )
        elif asset_path.endswith(".urdf"):
            builder.add_urdf(
                source=self.config["asset_path"],
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                ignore_inertial_definitions=False,
            )
        elif asset_path.endswith(".xml"):
            builder.add_mjcf(
                source=self.config["asset_path"],
                xform=wp.transform(wp.vec3(*self.config["mjw_init_pos"])),
                floating=True,
                enable_self_collisions=False,
                # parse_visuals_as_colliders=True,
                ignore_inertial_definitions=False,
            )
        # builder.approximate_meshes("bounding_box")
        
        # -- set initial pose
        builder.joint_q[:3] = self.config.get("mjw_init_pos", [0.0, -0.5, 0.8])
        builder.joint_q[3:7] = self.config.get("mjw_init_quat", [0.0, 0.0, 0.0, 1.0])
        builder.joint_q[7:] = self.config["mjw_joint_pos"]
        # -- set joint gains
        for i in range(len(self.config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = self.config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = self.config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = self.config["mjw_joint_armature"][i]

        # Disable collisions with bodies other than shanks
        for body in range(builder.body_count):
            if "ankle" not in builder.body_key[body] and "knee" not in builder.body_key[body]:
                for shape in builder.body_shapes[body]:
                    builder.shape_flags[shape] = builder.shape_flags[shape] & ~newton.ShapeFlags.COLLIDE_PARTICLES

    
    def add_sand(self, sand_builder: newton.ModelBuilder, voxel_size=0.05):
        particles_per_cell = 3.0
        density = 2500.0 # bulk density kg/m3
        particle_lo = np.array([-1.0, 0.0, 0.0])  # emission lower bound
        particle_hi = np.array([1.0, 3.5, 0.11])  # emission upper bound
        particle_res = np.array(
            np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size),
            dtype=int,
        )

        cell_size = (particle_hi - particle_lo) / particle_res
        cell_volume = np.prod(cell_size)
        radius = float(np.max(cell_size) * 0.5)
        mass = float(np.prod(cell_volume) * density)

        sand_builder.add_particle_grid(
            pos=wp.vec3(particle_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
        )

    def create_obs_buffer(self):
        self.base_ang_vel = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.velocity_command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self.joint_pos = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.joint_vel = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.last_actions = torch.zeros((1, self.config["num_dofs"]), device=self.torch_device, dtype=torch.float32)
        self.group_obs_term_hisotry_buffer = {
            "base_ang_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "projected_gravity": CircularBuffer(self.history_length, 1, self.torch_device),
            "velocity_command": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_pos": CircularBuffer(self.history_length, 1, self.torch_device),
            "joint_vel": CircularBuffer(self.history_length, 1, self.torch_device),
            "last_actions": CircularBuffer(self.history_length, 1, self.torch_device),
        }
        obs_dim = self.history_length * (
            self.base_ang_vel.shape[1] + self.projected_gravity.shape[1] + self.velocity_command.shape[1] + \
                self.joint_pos.shape[1] + self.joint_vel.shape[1] + self.last_actions.shape[1])
        self.obs = torch.zeros(1, obs_dim, device=self.torch_device, dtype=torch.float32)

    """
    graph
    """
    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None


    """
    physics step
    """

    def step(self):
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            # option1: control max speed with keyboard
            fwd = 2.5 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)

            # option2: incremental control with keyboard
            # fwd = 0.1 if self.viewer.is_key_down("i") else (-0.1 if self.viewer.is_key_down("k") else 0.0)
            # lat = 0.05 if self.viewer.is_key_down("j") else (-0.05 if self.viewer.is_key_down("l") else 0.0)
            # rot = 0.1 if self.viewer.is_key_down("u") else (-0.1 if self.viewer.is_key_down("o") else 0.0)
            # self.command[0, 0] += float(fwd)
            # self.command[0, 1] += float(lat)
            # self.command[0, 2] += float(rot)
            # # clip 
            # self.command[:, 0].clamp_(min=-2.0, max=3.0)
            # self.command[:, 1].clamp_(min=-0.5, max=0.5)
            # self.command[:, 2].clamp_(min=-1.0, max=1.0)
            # if self.viewer.is_key_down("z"):
            #     self.command *= 0.0
            # print(f"[INFO] Command: fwd={self.command[0,0]:.2f}, lat={self.command[0,1]:.2f}, rot={self.command[0,2]:.2f}")

            # Reset when 'P' is pressed (edge-triggered)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        """
        apply control from RL policy
        """
        self.apply_control()

        """
        step physics
        """
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)

        if self.follow_cam:
            self.viewer.set_camera(
                pos=wp.vec3(*self.state_0.joint_q.numpy()[:3]) + wp.vec3(3.0, 0.0, 0.1), pitch=-5.0, yaw=-180.0
            )

        self.viewer.log_points(
            "/sand",
            points=self.sand_state_0.particle_q,
            radii=self.sand_model.particle_radius,
            colors=self.particle_render_colors,
            hidden=not self.viewer.show_particles,
        )

        if self.show_impulses:
            impulses, pos, _cid = self.mpm_solver.collect_collider_impulses(self.sand_state_0)
            self.viewer.log_lines(
                "/impulses",
                starts=pos,
                ends=pos + impulses,
                colors=wp.full(pos.shape[0], value=wp.vec3(1.0, 0.0, 0.0), dtype=wp.vec3),
            )
        else:
            self.viewer.log_lines("/impulses", None, None, None)

        self.viewer.end_frame()

    """
    physics
    """

    def simulate(self):
        # robot substeps
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            wp.launch(
                compute_body_forces,
                dim=self.collider_impulse_ids.shape[0],
                inputs=[
                    self.sim_dt,
                    self.collider_impulse_ids,
                    self.collider_impulses,
                    self.collider_impulse_pos,
                    self.collider_body_id,
                    self.state_0.body_q,
                    self.model.body_com,
                    self.state_0.body_f,
                ],
            )
            # saved applied force to subtract later on
            self.body_sand_forces.assign(self.state_0.body_f)

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.contacts = self.model.collide(self.state_0)
            self.rb_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

            self.simulate_sand()

    def simulate_sand(self):
        # Subtract previously applied impulses from body velocities

        if self.sand_state_0.body_q is not None:
            wp.launch(
                subtract_body_force,
                dim=self.sand_state_0.body_q.shape,
                inputs=[
                    self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.body_sand_forces,
                    self.mpm_solver.mpm_model.collider_body_inv_inertia,
                    self.mpm_solver.mpm_model.collider_body_mass,
                    self.sand_state_0.body_q,
                    self.sand_state_0.body_qd,
                ],
            )

        self.mpm_solver.step(self.sand_state_0, self.sand_state_0, contacts=None, control=None, dt=self.sim_dt)

        # Save impulses to apply back to rigid bodies
        self.collect_collider_impulses()

    def collect_collider_impulses(self):
        collider_impulses, collider_impulse_pos, collider_impulse_ids = self.mpm_solver.collect_collider_impulses(
            self.sand_state_0
        )
        self.collider_impulse_ids.fill_(-1)
        n_colliders = min(collider_impulses.shape[0], self.collider_impulses.shape[0])
        self.collider_impulses[:n_colliders].assign(collider_impulses[:n_colliders])
        self.collider_impulse_pos[:n_colliders].assign(collider_impulse_pos[:n_colliders])
        self.collider_impulse_ids[:n_colliders].assign(collider_impulse_ids[:n_colliders])

    """
    mdp
    """
    
    def compute_obs(self):
        # Extract state information with proper handling
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        joint_qd = self.state_0.joint_qd if self.state_0.joint_qd is not None else []

        root_quat_w = torch.tensor(joint_q[3:7], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        # root_lin_vel_w = torch.tensor(joint_qd[:3], device=self.torch_device, dtype=torch.float32).unsqueeze(0) # maybe used 
        root_ang_vel_w = torch.tensor(joint_qd[3:6], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        joint_pos_current = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        joint_vel_current = torch.tensor(joint_qd[6:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)

        self.base_ang_vel = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
        self.projected_gravity = quat_rotate_inverse(root_quat_w, self.gravity_vec)
        self.joint_pos = torch.index_select(joint_pos_current - self.joint_pos_initial, 1, self.physx_to_mjc_indices)
        self.joint_vel = torch.index_select(joint_vel_current, 1, self.physx_to_mjc_indices)
        self.velocity_command = self.command
        self.last_actions = self.act
        
        # add to history buffer
        self.group_obs_term_hisotry_buffer["base_ang_vel"].append(self.base_ang_vel * self.config.get("ang_vel_scale", 1.0))
        self.group_obs_term_hisotry_buffer["projected_gravity"].append(self.projected_gravity * self.config.get("projected_gravity_scale", 1.0))
        self.group_obs_term_hisotry_buffer["velocity_command"].append(self.velocity_command * self.config.get("command_scale", 1.0))
        self.group_obs_term_hisotry_buffer["joint_pos"].append(self.joint_pos * self.config.get("joint_pos_scale", 1.0))
        self.group_obs_term_hisotry_buffer["joint_vel"].append(self.joint_vel * self.config.get("joint_vel_scale", 1.0))
        self.group_obs_term_hisotry_buffer["last_actions"].append(self.last_actions * self.config.get("last_action_scale", 1.0))
        
        self.obs = torch.cat([
            self.group_obs_term_hisotry_buffer["base_ang_vel"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["projected_gravity"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["velocity_command"].buffer.reshape(1, -1), # (1, history_length * 3)
            self.group_obs_term_hisotry_buffer["joint_pos"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["joint_vel"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
            self.group_obs_term_hisotry_buffer["last_actions"].buffer.reshape(1, -1), # (1, history_length * num_dofs)
        ], dim=1)

    def apply_control(self):
        self.compute_obs()
        with torch.no_grad():
            self.act = self.policy(self.obs)
            self.rearranged_act = torch.index_select(self.act, 1, self.mjc_to_physx_indices)
            a = self.joint_pos_initial + self.config["action_scale"] * self.rearranged_act
            # add floating-base dof zeros
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

    """
    reset
    """

    def reset(self):
        print("[INFO] Resetting example")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

        # sand reset
        wp.copy(self.sand_state_0.particle_q, self._initial_sand_q)
        wp.copy(self.sand_state_0.particle_qd, self._initial_sand_qd)
        self.mpm_solver.enrich_state(self.sand_state_0)
        self.collect_collider_impulses()

        # reset control 
        a = self.joint_pos_initial
        a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
        a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
        wp.copy(self.control.joint_target_pos, a_wp)


        
    """
    utilities
    """
    def render_ui(self, imgui):
        _changed, self.show_impulses = imgui.checkbox("Show Impulses", self.show_impulses)
        
    def load_policy_and_setup_tensors(self):
        """Load policy and setup initial tensors for robot control.
        """
        print("[INFO] Loading policy from:", self.config["policy_path"])
        self.policy = torch.jit.load(self.config["policy_path"], map_location=self.torch_device)

        # Handle potential None state
        joint_q = self.state_0.joint_q if self.state_0.joint_q is not None else []
        self.joint_pos_initial = torch.tensor(joint_q[7:], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.act = torch.zeros(1, self.config["num_dofs"], device=self.torch_device, dtype=torch.float32)
        self.rearranged_act = torch.zeros(1, self.config["num_dofs"], device=self.torch_device, dtype=torch.float32)


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds
    # example-specific ones
    parser = newton.examples.create_parser()

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    asset_dir = str(newton.utils.download_asset("unitree_g1"))

    # Load robot configuration from YAML file in the downloaded assets
    yaml_file_path = os.path.join(asset_dir, "rl_policies/g1_29dof_rev_1_0.yaml")
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    # load asset from newton-assets
    # config["policy_path"] = "rl_policies/g1_29DOF_rigid.pt"
    config["policy_path"] = os.path.join(asset_dir, config["policy_path"])
    config["asset_path"] = os.path.join(asset_dir, config["asset_path"])

    # if args.physx:
    if "physx_joint_names" in config.keys():
        # when importing policy trained in IsaacLab
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])

    env = NewtonEnv(viewer, config, mjc_to_physx, physx_to_mjc)

    total_sim_time = 100.0  # seconds
    while env.sim_time < total_sim_time:
        if not env.viewer.is_paused():
            with wp.ScopedTimer("step", active=False):
                env.step()

        with wp.ScopedTimer("render", active=False):
            env.render()
    print("[INFO] Simulation completed")
    env.viewer.close()