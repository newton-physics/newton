# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Franka
#
# This simulation demonstrates a coupled robot-cloth simulation
# using the VBD solver for the cloth and Featherstone for the robot,
# showcasing its ability to handle complex contacts while ensuring it
# remains intersection-free. The robot tracks end-effector keyframes
# via Newton's IK solver.
#
# The simulation runs in centimeter scale for better numerical behavior
# of the VBD solver. A vis_state is used to convert back to meter scale
# for visualization.
#
# Command: python -m newton.examples cloth_franka
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton import ModelBuilder, eval_fk
from newton.solvers import SolverFeatherstone, SolverVBD


@wp.kernel
def scale_positions(src: wp.array[wp.vec3], scale: float, dst: wp.array[wp.vec3]):
    i = wp.tid()
    dst[i] = src[i] * scale


@wp.kernel
def scale_body_transforms(src: wp.array[wp.transform], scale: float, dst: wp.array[wp.transform]):
    i = wp.tid()
    p = wp.transform_get_translation(src[i])
    q = wp.transform_get_rotation(src[i])
    dst[i] = wp.transform(p * scale, q)


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    joint_q[0, idx0] = finger_pos[0]
    joint_q[0, idx1] = finger_pos[0]


@wp.kernel
def compute_joint_qd(
    target_q: wp.array2d[float],
    current_q: wp.array[float],
    inv_frame_dt: float,
    # outputs
    out_qd: wp.array[float],
):
    # Indexing q and qd alike only holds for a fixed base; a floating base would
    # need joint_q_start/joint_qd_start and a quaternion difference.
    i = wp.tid()
    out_qd[i] = (target_q[0, i] - current_q[i]) * inv_frame_dt


class Example:
    def __init__(self, viewer, args):
        # parameters
        #   simulation (centimeter scale)
        self.add_cloth = True
        self.add_robot = True
        self.sim_substeps = 10
        self.iterations = 5
        self.fps = 60
        self.frame_dt = 1 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        # visualization: simulation in cm, viewer in meters
        self.viz_scale = 0.01

        #   contact (cm scale)
        #       body-cloth contact
        self.cloth_particle_radius = 0.8
        self.cloth_body_contact_margin = 0.8
        #       self-contact
        self.particle_self_contact_margin = 0.2
        self.particle_self_contact_gap = 0.0

        self.soft_contact_ke = 1e4
        self.soft_contact_kd = 1e1

        self.robot_contact_ke = 5e4
        self.robot_contact_kd = 5e1
        self.robot_contact_mu = 1.5

        self.self_contact_friction = 0.25

        #   elasticity
        self.tri_ke = 1e4
        self.tri_ka = 1e4
        self.tri_kd = 1.5e-2

        self.bending_ke = 5
        self.bending_kd = 5e-1

        self.scene = ModelBuilder(gravity=(0.0, 0.0, -981.0))

        self.viewer = viewer

        if self.add_robot:
            franka = ModelBuilder()
            self.create_articulation(franka)

            self.scene.add_world(franka)
            self.bodies_per_world = franka.body_count
            self.dof_q_per_world = franka.joint_coord_count
            self.dof_qd_per_world = franka.joint_dof_count

        # add a table (cm scale)
        self.table_hx_cm = 40.0
        self.table_hy_cm = 40.0
        self.table_hz_cm = 10.0
        self.table_pos_cm = wp.vec3(0.0, -50.0, 10.0)
        self.table_shape_idx = self.scene.shape_count
        self.scene.add_shape_box(
            -1,
            xform=wp.transform(
                self.table_pos_cm,
                wp.quat_identity(),
            ),
            hx=self.table_hx_cm,
            hy=self.table_hy_cm,
            hz=self.table_hz_cm,
        )

        # add the T-shirt
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = shirt_mesh.vertices
        mesh_indices = shirt_mesh.indices
        vertices = [wp.vec3(v) for v in mesh_points]

        if self.add_cloth:
            self.scene.add_cloth_mesh(
                vertices=vertices,
                indices=mesh_indices,
                rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
                pos=wp.vec3(0.0, 70.0, 30.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                density=0.02,
                scale=1.0,
                tri_ke=self.tri_ke,
                tri_ka=self.tri_ka,
                tri_kd=self.tri_kd,
                edge_ke=self.bending_ke,
                edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius,
            )

            self.scene.color()

        self.scene.add_ground_plane()

        self.model = self.scene.finalize(requires_grad=False)

        # Hide the table box from automatic shape rendering -- the GL viewer
        # bakes primitive dimensions into the mesh and ignores shape_scale,
        # so we render it manually at meter scale in render() instead.
        flags = self.model.shape_flags.numpy()
        flags[self.table_shape_idx] &= ~int(newton.ShapeFlags.VISIBLE)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

        # Pre-compute meter-scale table viz data
        self.table_viz_xform = wp.array(
            [
                wp.transform(
                    (
                        float(self.table_pos_cm[0]) * self.viz_scale,
                        float(self.table_pos_cm[1]) * self.viz_scale,
                        float(self.table_pos_cm[2]) * self.viz_scale,
                    ),
                    wp.quat_identity(),
                )
            ],
            dtype=wp.transform,
        )
        self.table_viz_scale = (
            self.table_hx_cm * self.viz_scale,
            self.table_hy_cm * self.viz_scale,
            self.table_hz_cm * self.viz_scale,
        )
        self.table_viz_color = wp.array([wp.vec3(0.5, 0.5, 0.5)], dtype=wp.vec3)

        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.self_contact_friction

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()

        shape_ke[...] = self.robot_contact_ke
        shape_kd[...] = self.robot_contact_kd
        shape_mu[...] = self.robot_contact_mu

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

        # Explicit collision pipeline for cloth-body contacts with custom margin
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=self.cloth_body_contact_margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        self.sim_time = 0.0

        # initialize robot solver
        self.robot_solver = SolverFeatherstone(self.model, update_mass_matrix_interval=self.sim_substeps)
        self.set_up_ik()

        self.cloth_solver: SolverVBD | None = None
        if self.add_cloth:
            self.model.edge_rest_angle.zero_()
            self.cloth_solver = SolverVBD(
                self.model,
                iterations=self.iterations,
                integrate_with_external_rigid_solver=True,
                particle_self_contact_margin=self.particle_self_contact_margin,
                particle_self_contact_gap=self.particle_self_contact_gap,
                particle_topological_contact_filter_threshold=1,
                particle_rest_shape_contact_exclusion_radius=0.5,
                particle_enable_self_contact=True,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                collision_frequency_type={
                    newton.solvers.SolverBase.CollisionSlot.SOFT_SELF_CONTACT: newton.solvers.SolverBase.CollisionFrequencyType.PRE_INIT,
                },
            )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.6, 0.6, 1.24), -42.0, -58.0)

        # Visualization state for meter-scale rendering
        self.viz_state = self.model.state()

        # Pre-compute scaled shape data for meter-scale visualization.
        # Two paths need updating:
        #   1) The GL viewer's CUDA path reads model.shape_transform / model.shape_scale
        #      directly, so we swap them temporarily in render().
        #   2) The base viewer path caches shapes.xforms / shapes.scales during
        #      set_model(), so we permanently scale those cached copies here.
        self.sim_shape_transform = self.model.shape_transform
        self.sim_shape_scale = self.model.shape_scale

        xform_np = self.model.shape_transform.numpy().copy()
        xform_np[:, :3] *= self.viz_scale
        self.viz_shape_transform = wp.array(xform_np, dtype=wp.transform, device=self.model.device)

        scale_np = self.model.shape_scale.numpy().copy()
        scale_np *= self.viz_scale
        self.viz_shape_scale = wp.array(scale_np, dtype=wp.vec3, device=self.model.device)

        # Scale particle radii from cm to meters for visualization
        self.sim_particle_radius = self.model.particle_radius
        if self.model.particle_radius is not None:
            radius_np = self.model.particle_radius.numpy().copy()
            radius_np *= self.viz_scale
            self.viz_particle_radius = wp.array(radius_np, dtype=wp.float32, device=self.model.device)
        else:
            self.viz_particle_radius = None

        # Scale the viewer's cached shape instance data (base viewer / GL fallback path)
        if hasattr(self.viewer, "_shape_instances"):
            for shapes in self.viewer._shape_instances.values():
                xi = shapes.xforms.numpy()
                xi[:, :3] *= self.viz_scale
                shapes.xforms = wp.array(xi, dtype=wp.transform, device=shapes.device)

                sc = shapes.scales.numpy()
                sc *= self.viz_scale
                shapes.scales = wp.array(sc, dtype=wp.vec3, device=shapes.device)

        # gravity arrays for swapping during simulation
        self.gravity_zero = wp.zeros(self.model.gravity.shape[0], dtype=wp.vec3, device=self.model.device)
        # gravity in cm/s²
        self.gravity_earth = wp.full(
            self.model.gravity.shape[0], wp.vec3(0.0, 0.0, -981.0), dtype=wp.vec3, device=self.model.device
        )

        # Ensure FK evaluation (for non-MuJoCo solvers):
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # graph capture
        self.graph = None
        if self.add_cloth:
            self.capture()

    def set_up_ik(self):
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.finger_pos = wp.zeros(1, dtype=float)

        # Interpolate the first keyframe from the initial end-effector pose
        state = self.model.state()
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        ee_tf = wp.transform(*state.body_q.numpy()[self.endeffector_id])
        ee_pos = wp.transform_point(ee_tf, self.endeffector_offset)
        ee_rot = wp.transform_get_rotation(ee_tf)
        self.initial_target = np.array([*ee_pos, *ee_rot, self.targets[0][-1]], dtype=np.float32)

        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.endeffector_id,
            link_offset=self.endeffector_offset,
            target_positions=wp.array([ee_pos], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.endeffector_id,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*ee_rot)], dtype=wp.vec4),
            # keeps orientation tracking tight against the posture term below
            weight=10.0,
        )
        joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        # Zero-width limits act as a quadratic pull toward the initial posture, which
        # keeps the redundant DOF from drifting; base yaw and fingers are left free.
        rest_q = self.model.joint_q.numpy()
        rest_lower = rest_q.copy()
        rest_upper = rest_q.copy()
        free = [self.base_joint_idx, self.finger_idx0, self.finger_idx1]
        rest_lower[free] = -1.0e6
        rest_upper[free] = 1.0e6
        posture_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(rest_lower, dtype=float),
            joint_limit_upper=wp.array(rest_upper, dtype=float),
            weight=0.3,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, joint_limit_obj, posture_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def create_articulation(self, builder):
        asset_path = newton.utils.download_asset("franka_emika_panda")

        builder.add_urdf(
            str(asset_path / "urdf" / "fr3_franka_hand.urdf"),
            xform=wp.transform(
                (-50.0, -50.0, 0.0),
                wp.quat_identity(),
            ),
            floating=False,
            scale=100,  # URDF is in meters, scale to cm
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
            joint_damping_mode="passive",
        )
        builder.joint_q[:6] = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307]

        clamp_close_activation_val = 0.1
        clamp_open_activation_val = 0.8

        self.robot_key_poses = np.array(
            [
                # translation_duration, gripper transform (3D position [cm], 4D quaternion), gripper activation
                # descend to working height before approaching the cloth
                [4, 31.0, -60.0, 40.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top left
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [2, 31.0, -60.0, 20.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 26.0, -60.0, 26.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 12.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -6.0, -60.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # bottom left
                [2, 15.0, -33.0, 31.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                [3, 15.0, -33.0, 21.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [2, 15.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [3, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_close_activation_val],
                [1, -2.0, -33.0, 28.0, 0.8536, -0.3536, 0.3536, -0.1464, clamp_open_activation_val],
                # top right
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 20.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -18.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 5.0, -60.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom right
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [3, -18.0, -30.0, 20.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [3, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, -3.0, -30.0, 31.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                # bottom
                [2, 0.0, -20.0, 30.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, 0.0, -20.0, 19.5, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [2, 0.0, -20.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -30.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_close_activation_val],
                [1.5, 0.0, -40.0, 35.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
                [2, -28.0, -60.0, 28.0, 0.9239, -0.3827, 0.0, 0.0, clamp_open_activation_val],
            ],
            dtype=np.float32,
        )
        self.targets = self.robot_key_poses[:, 1:]
        self.transition_duration = self.robot_key_poses[:, 0]

        self.robot_key_poses_time = np.cumsum(self.robot_key_poses[:, 0])
        self.endeffector_id = builder.body_label.index("fr3/fr3_link7")
        joint_q_idx = {label: builder.joint_q_start[j] for j, label in enumerate(builder.joint_label)}
        self.base_joint_idx = joint_q_idx["fr3/fr3_joint1"]
        self.finger_idx0 = joint_q_idx["fr3/fr3_finger_joint1"]
        self.finger_idx1 = joint_q_idx["fr3/fr3_finger_joint2"]
        self.endeffector_offset = wp.vec3(0.0, 0.0, 22.0)

    def update_ik_targets(self):
        # After the key poses sequence ends, hold the last target
        if self.sim_time >= self.robot_key_poses_time[-1]:
            return

        i = np.searchsorted(self.robot_key_poses_time, self.sim_time)
        t_start = self.robot_key_poses_time[i - 1] if i > 0 else 0.0
        alpha = np.clip((self.sim_time - t_start) / self.transition_duration[i], 0.0, 1.0)
        prev = self.targets[i - 1] if i > 0 else self.initial_target
        target = (1.0 - alpha) * prev + alpha * self.targets[i]
        rot = target[3:7] / np.linalg.norm(target[3:7])

        self.pos_obj.set_target_position(0, wp.vec3(*target[:3]))
        self.rot_obj.set_target_rotation(0, wp.vec4(*rot))
        # finger positions in cm
        self.finger_pos.fill_(float(target[-1]) * 4.0)

    def step(self):
        self.update_ik_targets()
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
        else:
            self.simulate()

    def simulate(self):
        self.ik_solver.step(self.state_0.joint_q.reshape((1, -1)), self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=1,
            inputs=[self.ik_joint_q, self.finger_pos, self.finger_idx0, self.finger_idx1],
        )
        # Joint velocity that reaches the IK solution within one frame
        wp.launch(
            compute_joint_qd,
            dim=self.model.joint_dof_count,
            inputs=[self.ik_joint_q, self.state_0.joint_q, 1.0 / self.frame_dt],
            outputs=[self.target_joint_qd],
        )

        self.cloth_solver.rebuild_bvh(self.state_0)
        for _step in range(self.sim_substeps):
            # robot sim
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            # apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            if self.add_robot:
                particle_count = self.model.particle_count
                # set particle_count = 0 to disable particle simulation in robot solver
                self.model.particle_count = 0
                self.model.gravity.assign(self.gravity_zero)

                # Update the robot pose - this will modify state_0 and copy to state_1
                self.state_0.joint_qd.assign(self.target_joint_qd)
                # Just update the forward kinematics to get body positions from joint coordinates
                self.robot_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

                self.state_0.particle_f.zero_()

                # restore original settings
                self.model.particle_count = particle_count
                self.model.gravity.assign(self.gravity_earth)

            # cloth sim
            self.collision_pipeline.collide(self.state_0, self.contacts)

            if self.add_cloth:
                self.cloth_solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

            self.sim_time += self.sim_dt

    def render(self):
        if self.viewer is None:
            return

        # Scale particle and body positions from cm to meters for visualization
        wp.launch(
            scale_positions,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.viz_scale],
            outputs=[self.viz_state.particle_q],
        )
        if self.model.body_count > 0:
            wp.launch(
                scale_body_transforms,
                dim=self.model.body_count,
                inputs=[self.state_0.body_q, self.viz_scale],
                outputs=[self.viz_state.body_q],
            )

        # Swap model shape and particle data to meter-scale for rendering
        self.model.shape_transform = self.viz_shape_transform
        self.model.shape_scale = self.viz_shape_scale
        self.model.particle_radius = self.viz_particle_radius

        try:
            self.viewer.begin_frame(self.sim_time)
            self.viewer.log_state(self.viz_state)
            # Render the table box manually at meter scale
            self.viewer.log_shapes(
                "/table",
                newton.GeoType.BOX,
                self.table_viz_scale,
                self.table_viz_xform,
                self.table_viz_color,
            )
            self.viewer.end_frame()
        finally:
            # Restore simulation shape and particle data
            self.model.shape_transform = self.sim_shape_transform
            self.model.shape_scale = self.sim_shape_scale
            self.model.particle_radius = self.sim_particle_radius

    def test_final(self):
        p_lower = wp.vec3(-36.0, -95.0, -5.0)
        p_upper = wp.vec3(36.0, 5.0, 56.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 200.0,
        )
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "body velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 70.0,
        )


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=3850)
    viewer, args = newton.examples.init(parser)

    # Create example and run
    newton.examples.run(Example(viewer, args), args)
