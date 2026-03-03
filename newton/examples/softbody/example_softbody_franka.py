# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Soft Body Franka
#
# Demonstrates a Franka Panda robot grasping a deformable rubber duck
# on a table. The robot uses Jacobian-based IK to follow keyframed
# end-effector poses. The duck is a tetrahedral mesh simulated with VBD.
#
# The simulation runs in meter scale.
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.utils
from newton import Model, ModelBuilder, State, eval_fk
from newton.math import transform_twist
from newton.solvers import SolverFeatherstone, SolverVBD

# Hardcoded local path for now (asset not yet published in newton-assets repo)
DUCK_ASSET = "D:/Code/Graphics/newton-assets/manipulation_objects/rubber_duck/mesh.usd"


@wp.kernel
def compute_ee_delta(
    body_q: wp.array(dtype=wp.transform),
    offset: wp.transform,
    body_id: int,
    bodies_per_world: int,
    target: wp.transform,
    # outputs
    ee_delta: wp.array(dtype=wp.spatial_vector),
):
    world_id = wp.tid()
    tf = body_q[bodies_per_world * world_id + body_id] * offset
    pos = wp.transform_get_translation(tf)
    pos_des = wp.transform_get_translation(target)
    pos_diff = pos_des - pos
    rot = wp.transform_get_rotation(tf)
    rot_des = wp.transform_get_rotation(target)
    ang_diff = rot_des * wp.quat_inverse(rot)
    ee_delta[world_id] = wp.spatial_vector(pos_diff[0], pos_diff[1], pos_diff[2], ang_diff[0], ang_diff[1], ang_diff[2])


class Example:
    def __init__(self, viewer, args=None):
        # simulation parameters (meter scale)
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # contact (meter scale)
        self.particle_radius = 0.005
        self.soft_body_contact_margin = 0.01
        self.particle_self_contact_radius = 0.003
        self.particle_self_contact_margin = 0.005

        self.soft_contact_ke = 2e6
        self.soft_contact_kd = 1e-7
        self.self_contact_friction = 0.5

        self.scene = ModelBuilder(gravity=-9.81)

        self.viewer = viewer

        # create robot
        franka = ModelBuilder()
        self.create_articulation(franka)
        self.scene.add_world(franka)
        self.bodies_per_world = franka.body_count
        self.dof_q_per_world = franka.joint_coord_count
        self.dof_qd_per_world = franka.joint_dof_count

        # add a table (meter scale)
        table_hx = 0.4
        table_hy = 0.4
        table_hz = 0.1
        table_pos = wp.vec3(0.0, -0.5, 0.1)
        self.scene.add_shape_box(
            -1,
            wp.transform(table_pos, wp.quat_identity()),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
        )

        # load pre-computed tetrahedral mesh from USD
        usd_stage = Usd.Stage.Open(DUCK_ASSET)
        prim = usd_stage.GetPrimAtPath("/TetModel")
        tetmesh = newton.TetMesh.create_from_usd(prim)

        # Duck USDA is in meters (metersPerUnit=1.0), small-head variant.
        # Table top is at z=0.2m. Duck center offset ~0.05m above table.
        self.scene.add_soft_mesh(
            pos=wp.vec3(0.0, -0.5, 0.23),
            rot=wp.quat_identity(),
            scale=1.0,  # already in meters
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=tetmesh,
            density=100.0,
            k_mu=1.0e6,
            k_lambda=1.0e6,
            k_damp=1e-6,
            particle_radius=self.particle_radius,
        )

        self.scene.color()
        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # contact material properties
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke[...] = self.soft_contact_ke
        shape_kd[...] = self.soft_contact_kd
        shape_mu[...] = 1.5
        self.model.shape_material_ke = wp.array(
            shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.model.shape_material_ke.device
        )
        self.model.shape_material_kd = wp.array(
            shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.model.shape_material_kd.device
        )
        self.model.shape_material_mu = wp.array(
            shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.model.shape_material_mu.device
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.target_joint_qd = wp.empty_like(self.state_0.joint_qd)

        self.control = self.model.control()

        # collision pipeline for soft body - robot contacts
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=self.soft_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_control()

        # soft body solver
        self.soft_solver = SolverVBD(
            self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=False,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=64,
            particle_collision_detection_interval=-1,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(1, dtype=wp.vec3)
        self.gravity_earth = wp.array(wp.vec3(0.0, 0.0, -9.81), dtype=wp.vec3)

        # evaluate FK for initial state
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture for performance
        self.capture()

    def set_up_control(self):
        self.control = self.model.control()

        out_dim = 6
        in_dim = self.model.joint_dof_count

        def onehot(i, out_dim):
            x = wp.array([1.0 if j == i else 0.0 for j in range(out_dim)], dtype=float)
            return x

        self.Jacobian_one_hots = [onehot(i, out_dim) for i in range(out_dim)]

        @wp.kernel
        def compute_body_out(body_qd: wp.array(dtype=wp.spatial_vector), body_out: wp.array(dtype=float)):
            mv = transform_twist(wp.static(self.endeffector_offset), body_qd[wp.static(self.endeffector_id)])
            for i in range(6):
                body_out[i] = mv[i]

        self.compute_body_out_kernel = compute_body_out
        self.temp_state_for_jacobian = self.model.state(requires_grad=True)

        self.body_out = wp.empty(out_dim, dtype=float, requires_grad=True)
        self.J_flat = wp.empty(out_dim * in_dim, dtype=float)
        self.J_shape = wp.array((out_dim, in_dim), dtype=int)
        self.ee_delta = wp.empty(1, dtype=wp.spatial_vector)
        self.initial_pose = self.model.joint_q.numpy()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform((-0.5, -0.5, -0.1), wp.quat_identity()),
            floating=False,
            scale=1.0,  # URDF is in meters
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        gripper_open = 1.0
        gripper_close = 0.65

        # Keyframe sequence: approach, descend, pinch, lift, hold, place, release, retract
        # [duration, px, py, pz, qx, qy, qz, qw, gripper_activation] (positions in meters)
        self.robot_key_poses = np.array(
            [
                # approach: move above the duck
                [2.5, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_open],
                # descend: lower to duck body
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_open],
                # pinch: close gripper on duck
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_close],
                # lift: raise duck off table
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_close],
                # hold: pause in air
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_close],
                # place: lower back to table
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_close],
                # release: open gripper
                [1.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_open],
                # retract: move away
                [2.0, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_open],
            ],
            dtype=np.float32,
        )

        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]
        self.target = self.targets[0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_count - 3
        self.endeffector_offset = wp.transform([0.0, 0.0, 0.22], wp.quat_identity())

    def compute_body_jacobian(
        self,
        model: Model,
        joint_q: wp.array,
        joint_qd: wp.array,
        include_rotation: bool = False,
    ):
        joint_q.requires_grad = True
        joint_qd.requires_grad = True

        in_dim = model.joint_dof_count
        out_dim = 6 if include_rotation else 3

        tape = wp.Tape()
        with tape:
            eval_fk(model, joint_q, joint_qd, self.temp_state_for_jacobian)
            wp.launch(
                self.compute_body_out_kernel, 1, inputs=[self.temp_state_for_jacobian.body_qd], outputs=[self.body_out]
            )

        for i in range(out_dim):
            tape.backward(grads={self.body_out: self.Jacobian_one_hots[i]})
            wp.copy(self.J_flat[i * in_dim : (i + 1) * in_dim], joint_qd.grad)
            tape.zero()

    def generate_control_joint_qd(self, state_in: State):
        if self.sim_time >= self.robot_key_poses_time[-1]:
            self.target_joint_qd.zero_()
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        self.target = self.targets[current_interval]

        include_rotation = True

        wp.launch(
            compute_ee_delta,
            dim=1,
            inputs=[
                state_in.body_q,
                self.endeffector_offset,
                self.endeffector_id,
                self.bodies_per_world,
                wp.transform(*self.target[:7]),
            ],
            outputs=[self.ee_delta],
        )

        self.compute_body_jacobian(
            self.model,
            state_in.joint_q,
            state_in.joint_qd,
            include_rotation=include_rotation,
        )
        J = self.J_flat.numpy().reshape(-1, self.model.joint_dof_count)
        delta_target = self.ee_delta.numpy()[0]
        J_inv = np.linalg.pinv(J)

        I = np.eye(J.shape[1], dtype=np.float32)
        N = I - J_inv @ J

        q = state_in.joint_q.numpy()
        q_des = q.copy()
        q_des[1:] = self.initial_pose[1:]

        K_null = 1.0
        delta_q_null = K_null * (q_des - q)
        delta_q = J_inv @ delta_target + N @ delta_q_null

        # gripper finger control
        delta_q[-2] = self.target[-1] * 0.04 - q[-2]
        delta_q[-1] = self.target[-1] * 0.04 - q[-1]

        self.target_joint_qd.assign(delta_q)

    def step(self):
        self.generate_control_joint_qd(self.state_0)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def simulate(self):
        self.soft_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # robot sim (disable particles temporarily)
            particle_count = self.model.particle_count
            self.model.particle_count = 0
            self.model.gravity.assign(self.gravity_zero)
            self.model.shape_contact_pair_count = 0

            self.state_0.joint_qd.assign(self.target_joint_qd)
            self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0.particle_f.zero_()
            self.model.particle_count = particle_count
            self.model.gravity.assign(self.gravity_earth)

            # collision detection
            self.collision_pipeline.collide(self.state_0, self.contacts)

            # soft body sim
            self.soft_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.5, -1.0, -0.05)
        p_upper = wp.vec3(0.5, 0.0, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 2.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 0.7,
        )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=1000)
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
