# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Reset DR Legs
#
# Demonstrates the reset configurations supported by SolverKamino.reset():
# default state, base pose, base pose + twist, joint-space FK, and
# actuator-space FK. Cycles through one actuated joint at a time, applying
# a brief feed-forward torque to it right after each reset, and advances to
# the next reset mode once every actuated joint has been cycled through.
#
###########################################################################

import argparse

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation  # noqa: TID253

import newton
import newton.examples
from newton._src.solvers.kamino._src.utils.sim.viewer_recording import enable_recording

###
# Kernels
###


@wp.kernel
def _actuate_selected_joint(
    dt: wp.float32,
    dofs_per_world: wp.int32,
    has_started_resets: wp.array[wp.bool],
    reset_dof_index: wp.array[wp.int32],
    actuated_dof_idx: wp.array[wp.int32],
    reset_time: wp.array[wp.float32],
    joint_f: wp.array[wp.float32],
):
    """Apply a brief feed-forward torque to the actuated joint currently selected by the reset-cycling demo."""
    # Skip if no joint is selected for actuation yet (i.e. before the first reset)
    if not has_started_resets[0]:
        return

    # Hack to handle negative reset index
    dof_index = reset_dof_index[0]
    if dof_index < 0:
        dof_index = actuated_dof_idx.shape[0] - 1

    # Track how long it has been since the last reset, since public models
    # don't expose the solver's internal simulation-time array
    t = reset_time[0]
    reset_time[0] = t + dt

    # Define the time window for the active feed-forward torque
    t_end = wp.float32(0.5)

    # Ad-hoc torque magnitude based on the selected joint
    # because we want higher actuation for the two hip joints
    local_index = dof_index % dofs_per_world
    if local_index == 0 or local_index == 6:
        torque = wp.float32(0.01)
    else:
        torque = wp.float32(0.001)

    # Reverse torque direction for the first leg
    if local_index < 6:
        torque = -torque

    dof = actuated_dof_idx[dof_index]
    if t < t_end:
        joint_f[dof] = torque
    else:
        joint_f[dof] = 0.0


###
# Example class
###


class Example:
    def __init__(self, viewer: newton.viewer.ViewerBase, args=None):
        # Set simulation run-time configurations
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.001
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.world_count = args.world_count if args else 1
        self.max_steps = args.max_reset_steps if args else 400
        self.viewer = viewer
        self.device = wp.get_device()

        # Internal counters driving the reset-mode / joint cycling demo
        self.sim_steps = 0
        self.sim_reset_mode = 0

        video_output_filename = getattr(args, "video_path", None)
        self.record_video = enable_recording(
            viewer=self.viewer,
            record_video=args.record_video if args else False,
            start_clip=True,
            output_path=video_output_filename if video_output_filename is not None else "recording.mp4",
            max_frames=getattr(args, "max_video_frames", 1000),
            fps=self.fps,
        )

        # Create a single-robot model builder and register the Kamino-specific custom attributes
        robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(robot_builder)

        # Load the DR Legs USD and add it to the builder
        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs" / "usd" / "dr_legs_with_meshes_and_boxes.usda")
        robot_builder.add_usd(
            asset_file,
            xform=wp.transformf((0.0, 0.0, 0.265), wp.quat_identity(dtype=wp.float32)),
            force_show_colliders=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        # Update joint target mode to enable force actuation
        robot_builder.joint_target_mode = [
            mode if mode == newton.JointTargetMode.NONE else newton.JointTargetMode.EFFORT
            for mode in robot_builder.joint_target_mode
        ]

        # Create the multi-world model by duplicating the single-robot builder.
        # Gravity is disabled: this example demonstrates reset behavior, not free-fall/locomotion.
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
        for _ in range(self.world_count):
            builder.add_world(robot_builder)

        # Create the model from the builder
        self.model = builder.finalize(skip_validation_joints=True)
        self.model.gravity.zero_()

        # Create and configure the Kamino solver for the given model
        self.config = newton.solvers.SolverKamino.Config.from_model(self.model)
        self.config.use_fk_solver = True
        self.config.constraints.alpha = 0.1
        self.config.padmm.primal_tolerance = 1e-4
        self.config.padmm.dual_tolerance = 1e-4
        self.config.padmm.compl_tolerance = 1e-4
        self.config.padmm.max_iterations = 100
        self.config.padmm.eta = 1e-5
        self.config.padmm.rho_0 = 0.02
        self.config.padmm.rho_min = 0.01
        self.config.padmm.use_acceleration = True
        self.config.padmm.warmstart_mode = "containers"
        self.config.padmm.contact_warmstart_method = "geom_pair_net_force"
        self.config.collect_solver_info = False
        self.config.compute_solution_metrics = False
        self.solver = newton.solvers.SolverKamino(self.model, config=self.config)

        # Create state and control data containers
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Determine which joints are actuated (driven by FK on reset), in joint order.
        joint_q_start_np = self.model.joint_q_start.numpy()
        joint_qd_start_np = self.model.joint_qd_start.numpy()
        joint_target_mode_np = self.model.joint_target_mode.numpy()
        actuated_joints = [
            j
            for j in range(self.model.joint_count)
            if np.any(
                joint_target_mode_np[joint_qd_start_np[j] : joint_qd_start_np[j + 1]]
                != int(newton.JointTargetMode.NONE)
            )
        ]
        self.actuated_joint_q_idx_np = joint_q_start_np[actuated_joints].astype(np.int32)
        self.actuated_joint_dof_idx_np = joint_qd_start_np[actuated_joints].astype(np.int32)
        self.num_actuated_dofs = len(actuated_joints)

        # Allocate utility arrays for resetting and for the joint-cycling actuation demo
        with wp.ScopedDevice(self.device):
            self.base_q = wp.zeros(shape=(self.world_count,), dtype=wp.transformf)
            self.base_u = wp.zeros(shape=(self.world_count,), dtype=wp.spatial_vectorf)
            self.joint_q = wp.zeros(shape=(self.model.joint_coord_count,), dtype=wp.float32)
            self.joint_u = wp.zeros(shape=(self.model.joint_dof_count,), dtype=wp.float32)
            self.actuator_q = wp.zeros(shape=(self.num_actuated_dofs,), dtype=wp.float32)
            self.actuator_u = wp.zeros(shape=(self.num_actuated_dofs,), dtype=wp.float32)
            self.actuated_joint_dof_idx = wp.array(self.actuated_joint_dof_idx_np, dtype=wp.int32)
            self.reset_time = wp.zeros(shape=(1,), dtype=wp.float32)
            self.has_started_resets = wp.full(shape=(1,), dtype=wp.bool, value=False)
            self.reset_dof_index = wp.full(shape=(1,), dtype=wp.int32, value=-1)

        # Attach the model to the viewer for visualization
        self.viewer.set_model(self.model)

        # Warm-start the simulation
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        self.solver.reset(self.state_0)
        self.solver.reset(self.state_1)

        # Capture the simulation graph if running on CUDA
        # NOTE: This only has an effect on GPU devices
        self.graph = None
        self.capture()

        # If only a single world is created, set initial camera position for better view of the system
        if self.world_count == 1 and hasattr(self.viewer, "set_camera"):
            camera_pos = wp.vec3(0.6, 0.6, 0.3)
            pitch = -10.0
            yaw = 225.0
            self.viewer.set_camera(camera_pos, pitch, yaw)

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.launch(
                _actuate_selected_joint,
                dim=1,
                inputs=[
                    self.sim_dt,
                    self.num_actuated_dofs // self.model.world_count,
                    self.has_started_resets,
                    self.reset_dof_index,
                    self.actuated_joint_dof_idx,
                    self.reset_time,
                    self.control.joint_f,
                ],
                device=self.device,
            )
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_steps += 1

    def update_reset_selection(self):
        """Advance to the next actuated joint, moving to the next reset mode once all have been cycled through."""
        self.sim_steps = 0
        self.control.joint_f.zero_()
        self.reset_time.zero_()
        self.has_started_resets.fill_(True)
        dof_index = self.reset_dof_index.numpy()[0]
        dof_index = (dof_index + 1) % self.num_actuated_dofs
        # If all actuated joints have been cycled through, proceed to the next reset mode
        if dof_index == self.num_actuated_dofs - 1:
            self.sim_reset_mode = (self.sim_reset_mode + 1) % 5
            dof_index = -1
        self.reset_dof_index.fill_(dof_index)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_steps += self.sim_substeps
        else:
            self.simulate()
        self.sim_time += self.frame_dt

        # Demo of resetting to the default state defined in the model
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 0:
            self.update_reset_selection()
            self.solver.reset(self.state_0)
            self.solver.reset(self.state_1)

        # Demo of resetting only the base pose
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 1:
            self.update_reset_selection()
            reset_config = self._reset_config_base_pose()
            self.solver.reset(state=self.state_0, config=reset_config)
            self.solver.reset(state=self.state_1, config=reset_config)

        # Demo of resetting the base pose and twist
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 2:
            self.update_reset_selection()
            reset_config = self._reset_config_base_pose_and_twist(
                u_base=wp.spatial_vectorf(0.0, 0.0, 0.05, 0.0, 0.0, 0.3)
            )
            self.solver.reset(state=self.state_0, config=reset_config)
            self.solver.reset(state=self.state_1, config=reset_config)

        # Demo of resetting the base state and joint configurations to specific poses
        # NOTE: This will invoke the FK solver to update body poses
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 3:
            self.update_reset_selection()
            dof_index = self.reset_dof_index.numpy()[0]
            joint_q_np = np.zeros(self.model.joint_coord_count, dtype=np.float32)
            joint_q_np[self.actuated_joint_q_idx_np[dof_index]] = self._selected_joint_angle(dof_index)
            self.joint_q.assign(joint_q_np)
            reset_config = self._reset_config_base_pose_and_twist(
                u_base=wp.spatial_vectorf(0.0, 0.0, -0.05, 0.0, 0.0, 0.3),
                body_poses=newton.solvers.SolverKamino.ResetConfig.FromJointQ(self.joint_q),
                body_velocities=newton.solvers.SolverKamino.ResetConfig.FromJointU(self.joint_u),
            )
            self.solver.reset(state=self.state_0, config=reset_config)
            self.solver.reset(state=self.state_1, config=reset_config)

        # Demo of resetting the base state and actuator configurations to specific poses
        # NOTE: This will invoke the FK solver to update body poses
        if self.sim_steps >= self.max_steps and self.sim_reset_mode == 4:
            self.update_reset_selection()
            dof_index = self.reset_dof_index.numpy()[0]
            actuator_q_np = np.zeros(self.num_actuated_dofs, dtype=np.float32)
            actuator_q_np[dof_index] = self._selected_joint_angle(dof_index)
            self.actuator_q.assign(actuator_q_np)
            reset_config = self._reset_config_base_pose_and_twist(
                u_base=wp.spatial_vectorf(0.0, 0.0, -0.05, 0.0, 0.0, -0.3),
                body_poses=newton.solvers.SolverKamino.ResetConfig.FromActuatorQ(self.actuator_q),
                body_velocities=newton.solvers.SolverKamino.ResetConfig.FromActuatorU(self.actuator_u),
            )
            self.solver.reset(state=self.state_0, config=reset_config)
            self.solver.reset(state=self.state_1, config=reset_config)

    def _selected_joint_angle(self, dof_index: int) -> float:
        """Target angle for the actuated joint currently selected by the reset-cycling demo."""
        local_index = dof_index % (self.num_actuated_dofs // self.world_count)
        return np.pi / 12 if local_index < 6 else -np.pi / 12

    def _reset_config_base_pose(self):
        R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
        q_b = R_b.as_quat()  # x, y, z, w
        q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
        self.base_q.assign([q_base] * self.world_count)
        return newton.solvers.SolverKamino.ResetConfig(
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q)
        )

    def _reset_config_base_pose_and_twist(self, u_base: wp.spatial_vectorf, **kwargs):
        R_b = Rotation.from_rotvec(np.pi / 4 * np.array([0, 0, 1]))
        q_b = R_b.as_quat()  # x, y, z, w
        q_base = wp.transformf((0.1, 0.1, 0.3), q_b)
        self.base_q.assign([q_base] * self.world_count)
        self.base_u.assign([u_base] * self.world_count)
        return newton.solvers.SolverKamino.ResetConfig(
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self.base_q),
            base_velocity=newton.solvers.SolverKamino.ResetConfig.FromBaseU(self.base_u),
            **kwargs,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        pass  # TODO: Add some assertions here once we have a more meaningful test scenario

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--max-reset-steps",
            type=int,
            default=400,
            help="Number of simulation steps between each reset-mode demo.",
        )
        parser.add_argument(
            "--record-video",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Record a video of the viewer, up to 1000 frames.",
        )
        parser.add_argument(
            "--video-path",
            type=str,
            default=None,
            help="Output video path (defaults to 'recording.mp4').",
        )
        parser.add_argument(
            "--max-video-frames",
            type=int,
            default=1000,
            help="Maximum number of frames recorded for the video (defaults to 1000).",
        )
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
    if hasattr(viewer, "finish_clip"):
        viewer.finish_clip()
