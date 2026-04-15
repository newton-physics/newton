# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Soft Body Franka
#
# Demonstrates a Franka Panda robot grasping a deformable rubber duck
# on a table. The robot is positioned via Newton's GPU IK solver;
# Featherstone is used as a kinematic integrator to produce proper
# body velocities for VBD friction. The duck is a tetrahedral mesh
# simulated with VBD.
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
import newton.ik as ik
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def compute_joint_qd(
    target_q: wp.array[float],
    current_q: wp.array[float],
    out_qd: wp.array[float],
    inv_frame_dt: float,
):
    i = wp.tid()
    out_qd[i] = (target_q[i] - current_q[i]) * inv_frame_dt


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
        duck_path = newton.utils.download_asset("manipulation_objects/rubber_duck")
        usd_stage = Usd.Stage.Open(str(duck_path / "model.usda"))
        prim = usd_stage.GetPrimAtPath("/root/Model/TetMesh")
        tetmesh = newton.TetMesh.create_from_usd(prim)

        # Duck USDA is in meters (metersPerUnit=1.0).
        # Table top is at z=0.2m. Duck center offset ~0.03m above table.
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

        self.model.shape_material_ke.fill_(self.soft_contact_ke)
        self.model.shape_material_kd.fill_(self.soft_contact_kd)
        self.model.shape_material_mu.fill_(1.5)

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

        # robot solver (Featherstone as kinematic integrator for body velocities)
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)

        # IK solver setup
        self.set_up_ik()

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
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture for performance
        self.capture()

    def set_up_ik(self):
        """Set up GPU IK solver for end-effector pose tracking."""
        # Evaluate FK to get initial EE transform
        state = self.model.state()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)

        # IK joint coordinates (1 problem, all DOFs)
        self.n_coords = self.model.joint_coord_count
        self.n_dofs = self.model.joint_dof_count
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.n_coords))

        # Finger DOF indices (last two)
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1

        # Finger position buffer (wp.array so it works with graph capture)
        self.finger_pos_buf = wp.zeros(1, dtype=float)

        # 1D buffer for IK result (target joint config)
        self.target_joint_q = wp.zeros(self.n_coords, dtype=float)

        # Initial target from keyframe
        target_pos = wp.vec3(*self.targets[0][:3].tolist())
        target_rot = wp.vec4(*self.targets[0][3:7].tolist())

        # Position objective (with EE offset along tool axis)
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=wp.vec3(0.0, 0.0, 0.22),
            target_positions=wp.array([target_pos], dtype=wp.vec3),
        )

        # Rotation objective
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([target_rot], dtype=wp.vec4),
        )

        # Joint limit objective
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )

        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        self.ik_iters = 24

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
        gripper_close = 0.5

        # Keyframe sequence: approach, descend, pinch, lift, hold, place, release, retract
        # [duration, px, py, pz, qx, qy, qz, qw, gripper_activation] (positions in meters)
        self.robot_key_poses = np.array(
            [
                # approach: move above the duck
                [2.5, -0.005, -0.5, 0.35, 1, 0.0, 0.0, 0.0, gripper_open],
                # descend: lower to duck body
                [2.0, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_open],
                # pinch: close gripper on duck
                [2.5, -0.005, -0.5, 0.21, 1, 0.0, 0.0, 0.0, gripper_close],
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

    def update_ik_targets(self):
        """Interpolate keyframes and update IK target arrays (CPU, called before graph launch)."""
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return

        current_interval = np.searchsorted(self.robot_key_poses_time, self.sim_time)

        # Interpolate between previous and current keyframe target
        t_start = self.robot_key_poses_time[current_interval - 1] if current_interval > 0 else 0.0
        t_end = self.robot_key_poses_time[current_interval]
        alpha = float(np.clip((self.sim_time - t_start) / (t_end - t_start), 0.0, 1.0))

        target_cur = self.targets[current_interval]
        target_prev = self.targets[current_interval - 1] if current_interval > 0 else target_cur
        target_interp = (1.0 - alpha) * target_prev + alpha * target_cur

        # Update IK target arrays on GPU (read by IK solver inside captured graph)
        self.pos_obj.set_target_position(0, wp.vec3(*target_interp[:3].tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*target_interp[3:7].tolist()))

        # Update gripper finger position buffer
        finger_pos = float(target_interp[-1]) * 0.04
        self.finger_pos_buf.fill_(finger_pos)

    def step(self):
        self.update_ik_targets()
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.simulate()

    def simulate(self):
        # IK solve once per frame (GPU, captured in graph)
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)

        # Set gripper finger positions from buffer
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
        )

        # Copy IK result to target buffer (2D -> 1D, contiguous memory)
        wp.copy(self.target_joint_q, self.ik_joint_q, dest_offset=0, src_offset=0, count=self.n_coords)

        # Compute joint velocity: qd = (target - current) / frame_dt
        wp.launch(
            compute_joint_qd,
            dim=self.n_dofs,
            inputs=[self.target_joint_q, self.state_0.joint_q, self.target_joint_qd, 1.0 / self.frame_dt],
        )

        self.soft_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # Featherstone as kinematic integrator (disable particles + gravity)
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
