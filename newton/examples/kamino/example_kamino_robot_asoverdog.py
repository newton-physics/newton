# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Kamino Robot asOverDog
#
# Simulates the Bennett, Planar, and Spherical asOverDog variants with
# SolverKamino and robot-specific locomotion policies.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
#
# Command: python -m newton.examples kamino_robot_asoverdog
###########################################################################

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from warp_nn import nn
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes


@dataclass(frozen=True)
class RobotConfig:
    home_position: tuple[float, ...]
    enable_self_collisions: bool


ROBOT_CONFIGS = {
    "bennett": RobotConfig(
        home_position=(-0.67, 1.91, -0.74, 0.67, -1.91, 0.74, 0.67, -1.91, 0.74, -0.67, 1.91, -0.74),
        enable_self_collisions=False,
    ),
    "planar": RobotConfig(
        home_position=(
            -1.3788101091,
            2.3911010752,
            -1.5184364492,
            1.3788101091,
            -2.3911010752,
            1.5184364492,
            1.3788101091,
            -2.3911010752,
            1.5184364492,
            -1.3788101091,
            2.3911010752,
            -1.5184364492,
        ),
        enable_self_collisions=True,
    ),
    "spherical": RobotConfig(
        home_position=(
            -0.9424777961,
            2.4260076603,
            -1.5707963268,
            0.9424777961,
            -2.4260076603,
            1.5707963268,
            0.9424777961,
            -2.4260076603,
            1.5707963268,
            -0.9424777961,
            2.4260076603,
            -1.5707963268,
        ),
        enable_self_collisions=False,
    ),
}

MOTOR_JOINTS = (
    "motor_hip_FL",
    "motor_thigh_FL",
    "motor_calf_FL",
    "motor_hip_FR",
    "motor_thigh_FR",
    "motor_calf_FR",
    "motor_hip_RL",
    "motor_thigh_RL",
    "motor_calf_RL",
    "motor_hip_RR",
    "motor_thigh_RR",
    "motor_calf_RR",
)
POLICY_MOTOR_JOINTS = (
    "motor_hip_FL",
    "motor_hip_FR",
    "motor_hip_RL",
    "motor_hip_RR",
    "motor_thigh_FL",
    "motor_thigh_FR",
    "motor_thigh_RL",
    "motor_thigh_RR",
    "motor_calf_FL",
    "motor_calf_FR",
    "motor_calf_RL",
    "motor_calf_RR",
)
LOOP_JOINTS = ("end_joint_FL", "end_joint_FR", "end_joint_RL", "end_joint_RR")

ACTION_COUNT = 12
OBSERVATION_FRAME_SIZE = 61
OBSERVATION_HISTORY = 25
OBSERVATION_SIZE = OBSERVATION_FRAME_SIZE * OBSERVATION_HISTORY
ACTUATOR_HISTORY = 6


@wp.kernel
def _build_observation_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    base_body: int,
    policy_coord_indices: wp.array[int],
    policy_dof_indices: wp.array[int],
    policy_home: wp.array[float],
    command: wp.array[wp.vec3],
    phase: wp.array[wp.vec4],
    action_history: wp.array2d[float],
    observation: wp.array2d[float],
):
    for i in range((OBSERVATION_HISTORY - 1) * OBSERVATION_FRAME_SIZE):
        observation[0, i] = observation[0, i + OBSERVATION_FRAME_SIZE]

    cmd = command[0]
    standing = wp.abs(cmd[0]) < 0.1 and wp.abs(cmd[1]) < 0.1 and wp.abs(cmd[2]) < 0.05
    effective_cmd = cmd
    if standing:
        effective_cmd = wp.vec3(0.0, 0.0, 0.0)

    gait_phase = phase[0] + wp.vec4(0.1, 0.1, 0.1, 0.1)
    for i in range(4):
        if gait_phase[i] >= 2.0:
            gait_phase[i] = gait_phase[i] - 2.0
    phase[0] = gait_phase

    base = body_q[base_body]
    rotation = wp.transform_get_rotation(base)
    angular_velocity = wp.quat_rotate_inv(rotation, wp.spatial_bottom(body_qd[base_body]))
    gravity = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, -1.0))
    offset = (OBSERVATION_HISTORY - 1) * OBSERVATION_FRAME_SIZE

    for i in range(4):
        clock = wp.sin(wp.pi * gait_phase[i])
        if standing:
            clock = 0.5
        observation[0, offset + i] = clock

    observation[0, offset + 4] = gravity[0]
    observation[0, offset + 5] = gravity[1]
    observation[0, offset + 6] = gravity[2]
    observation[0, offset + 7] = 2.0 * angular_velocity[0]
    observation[0, offset + 8] = 2.0 * angular_velocity[1]
    observation[0, offset + 9] = 2.0 * angular_velocity[2]
    observation[0, offset + 10] = 2.0 * effective_cmd[0]
    observation[0, offset + 11] = 2.0 * effective_cmd[1]
    observation[0, offset + 12] = 2.0 * effective_cmd[2]

    for i in range(ACTION_COUNT):
        observation[0, offset + 13 + i] = joint_q[policy_coord_indices[i]] - policy_home[i]
        observation[0, offset + 25 + i] = 0.05 * joint_qd[policy_dof_indices[i]]
        observation[0, offset + 37 + i] = action_history[0, i]
        observation[0, offset + 49 + i] = action_history[1, i]


@wp.kernel
def _update_action_kernel(
    policy_action: wp.array2d[float],
    action_history: wp.array2d[float],
):
    i = wp.tid()
    action_history[1, i] = action_history[0, i]
    action_history[0, i] = wp.clamp(policy_action[0, i], -10.0, 10.0)


@wp.kernel
def _build_actuator_input_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    motor_coord_indices: wp.array[int],
    motor_dof_indices: wp.array[int],
    sim_from_policy: wp.array[int],
    home_position: wp.array[float],
    action_history: wp.array2d[float],
    position_error_history: wp.array2d[float],
    velocity_history: wp.array2d[float],
    actuator_input: wp.array2d[float],
):
    motor = wp.tid()
    for history in range(ACTUATOR_HISTORY - 1, 0, -1):
        position_error_history[history, motor] = position_error_history[history - 1, motor]
        velocity_history[history, motor] = velocity_history[history - 1, motor]

    target = home_position[motor] + 0.25 * action_history[0, sim_from_policy[motor]]
    position_error_history[0, motor] = target - joint_q[motor_coord_indices[motor]]
    velocity_history[0, motor] = joint_qd[motor_dof_indices[motor]]
    for history in range(ACTUATOR_HISTORY):
        actuator_input[motor, history] = -position_error_history[history, motor]
        actuator_input[motor, ACTUATOR_HISTORY + history] = velocity_history[history, motor]


@wp.kernel
def _apply_motor_torque_kernel(
    raw_torque: wp.array2d[float],
    joint_qd: wp.array[float],
    motor_dof_indices: wp.array[int],
    joint_force: wp.array[float],
):
    motor = wp.tid()
    dof = motor_dof_indices[motor]
    velocity = wp.clamp(joint_qd[dof], -44.0, 44.0)
    upper = wp.min(17.0 * (1.0 - velocity / 22.0), 17.0)
    lower = wp.max(17.0 * (-1.0 - velocity / 22.0), -17.0)
    joint_force[dof] = wp.clamp(raw_torque[motor, 0], lower, upper)


@wp.kernel
def _compute_loop_error_kernel(
    body_q: wp.array[wp.transform],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    loop_joints: wp.array[int],
    error: wp.array[float],
):
    i = wp.tid()
    joint = loop_joints[i]
    parent_anchor = wp.transform_multiply(body_q[joint_parent[joint]], joint_X_p[joint])
    child_anchor = wp.transform_multiply(body_q[joint_child[joint]], joint_X_c[joint])
    error[i] = wp.length(wp.transform_get_translation(parent_anchor) - wp.transform_get_translation(child_anchor))


def _short_label(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _find_unique(labels: list[str], name: str) -> int:
    matches = [index for index, label in enumerate(labels) if _short_label(label) == name]
    if len(matches) != 1:
        raise ValueError(f"Label {name!r} matched {len(matches)} entries; expected exactly one")
    return matches[0]


def _load_actuator(path: Path, device: wp.Device) -> nn.Sequential:
    actuator = nn.Sequential(
        nn.Linear(12, 64),
        nn.SoftSign(),
        nn.Linear(64, 64),
        nn.SoftSign(),
        nn.Linear(64, 1),
    ).to(device)
    with np.load(path) as weights:
        state = {
            name: np.asarray(value).reshape((-1, 1)) if name.endswith(".bias") else np.asarray(value)
            for name, value in weights.items()
        }
    actuator.load_state_dict(state)
    return actuator


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.device = wp.get_device()
        self.robot_name = args.robot
        self.robot_config = ROBOT_CONFIGS[self.robot_name]
        self.frame_dt = 1.0 / 50.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self._reset_key_previous = False
        self._fallback_command = wp.vec3(args.vx, args.vy, args.yaw_rate)

        asset_dir = newton.utils.download_asset("asoverdog")
        usd_path = asset_dir / "usd" / f"{self.robot_name}.usdc"
        policy_path = asset_dir / "policies" / f"{self.robot_name}.onnx"
        actuator_path = asset_dir / "actuators" / "robstride_250hz_history_len_6.npz"

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverKamino.register_custom_attributes(builder)
        builder.request_contact_attributes("force")
        builder.add_usd(
            str(usd_path),
            collapse_fixed_joints=False,
            enable_self_collisions=self.robot_config.enable_self_collisions,
            hide_collision_shapes=True,
        )

        self.base_body = _find_unique(builder.body_label, "MainBody")
        motor_joints = [_find_unique(builder.joint_label, name) for name in MOTOR_JOINTS]
        loop_joints = [_find_unique(builder.joint_label, name) for name in LOOP_JOINTS]
        motor_coords = [builder.joint_q_start[index] for index in motor_joints]
        motor_dofs = [builder.joint_qd_start[index] for index in motor_joints]
        if any(builder.joint_articulation[index] != -1 for index in loop_joints):
            raise ValueError("asOverDog loop-closing joints must be outside the articulation")

        builder.joint_target_mode[:] = [int(newton.JointTargetMode.NONE)] * builder.joint_dof_count
        for dof in motor_dofs:
            builder.joint_target_mode[dof] = int(newton.JointTargetMode.EFFORT)
            builder.joint_effort_limit[dof] = 17.0
            builder.joint_velocity_limit[dof] = 22.0
            builder.joint_armature[dof] = 0.0042
            builder.joint_damping[dof] = 0.008
        builder.shape_material_mu[:] = [0.6] * builder.shape_count
        builder.shape_material_restitution[:] = [0.0] * builder.shape_count
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.0, restitution=0.0),
        )

        self.model = builder.finalize(device=self.device, skip_validation_joints=True)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.rigid_contact_max = 256
        solver_config = newton.solvers.SolverKamino.Config.from_model(
            self.model,
            dynamics_solver="dvi",
            sparse_jacobian=True,
            sparse_dynamics=True,
        )
        solver_config.use_fk_solver = True
        solver_config.use_collision_detector = False
        solver_config.integrator = "euler"
        solver_config.constraints.alpha = 0.1
        solver_config.constraints.beta = 0.011
        solver_config.constraints.gamma = 0.015
        solver_config.dynamics.preconditioning = False
        solver_config.dynamics.linear_solver_type = "CR"
        solver_config.dynamics.linear_solver_kwargs = {"maxiter": 9}
        solver_config.dvi.bilateral_solver_type = "LLTBRCM"
        solver_config.dvi.bilateral_solver_kwargs = {"parallel_factorization": True}
        solver_config.dvi.tolerance = 1.0e-4
        solver_config.dvi.regularization = 1.0e-5
        solver_config.dvi.max_alternating_iterations = 4
        solver_config.dvi.inequality_sweeps_per_iteration = 3
        solver_config.dvi.bilateral_solve_interval = 1
        solver_config.dvi.warmstart_mode = "containers"
        solver_config.dvi.contact_warmstart_method = "key_and_position_with_net_force_backup"
        solver_config.materials.friction_mix_mode = "multiply"
        solver_config.materials.restitution_mix_mode = "multiply"
        self.solver = newton.solvers.SolverKamino(self.model, config=solver_config)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)

        self._motor_coord_indices = wp.array(motor_coords, dtype=wp.int32, device=self.device)
        self._motor_dof_indices = wp.array(motor_dofs, dtype=wp.int32, device=self.device)
        self._loop_joint_indices = wp.array(loop_joints, dtype=wp.int32, device=self.device)
        policy_coords = [motor_coords[MOTOR_JOINTS.index(name)] for name in POLICY_MOTOR_JOINTS]
        policy_dofs = [motor_dofs[MOTOR_JOINTS.index(name)] for name in POLICY_MOTOR_JOINTS]
        policy_home = [self.robot_config.home_position[MOTOR_JOINTS.index(name)] for name in POLICY_MOTOR_JOINTS]
        sim_from_policy = [POLICY_MOTOR_JOINTS.index(name) for name in MOTOR_JOINTS]
        self._policy_coord_indices = wp.array(policy_coords, dtype=wp.int32, device=self.device)
        self._policy_dof_indices = wp.array(policy_dofs, dtype=wp.int32, device=self.device)
        self._policy_home = wp.array(policy_home, dtype=wp.float32, device=self.device)
        self._home_position = wp.array(self.robot_config.home_position, dtype=wp.float32, device=self.device)
        self._sim_from_policy = wp.array(sim_from_policy, dtype=wp.int32, device=self.device)

        self.policy = OnnxRuntime(str(policy_path), device=self.device)
        self.policy_input_name = self.policy.input_names[0]
        self.policy_output_name = self.policy.output_names[0]
        validate_policy_io_shapes(
            str(policy_path),
            self.policy_input_name,
            self.policy_output_name,
            obs_width=OBSERVATION_SIZE,
            action_width=ACTION_COUNT,
            context="example_kamino_robot_asoverdog",
        )
        self.actuator = _load_actuator(actuator_path, self.device)
        self._command = wp.array([self._fallback_command], dtype=wp.vec3, device=self.device)
        self._phase = wp.array([wp.vec4(1.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.device)
        self._action_history = wp.zeros((2, ACTION_COUNT), dtype=wp.float32, device=self.device)
        self._observation = wp.zeros((1, OBSERVATION_SIZE), dtype=wp.float32, device=self.device)
        self._position_error_history = wp.zeros((ACTUATOR_HISTORY, ACTION_COUNT), dtype=wp.float32, device=self.device)
        self._velocity_history = wp.zeros_like(self._position_error_history)
        self._actuator_input = wp.zeros((ACTION_COUNT, 2 * ACTUATOR_HISTORY), dtype=wp.float32, device=self.device)
        self._loop_error = wp.zeros(len(LOOP_JOINTS), dtype=wp.float32, device=self.device)

        self._base_q = wp.array(
            [wp.transform((0.0, 0.0, 0.34), wp.quat_identity())],
            dtype=wp.transform,
            device=self.device,
        )
        self._base_u = wp.zeros(1, dtype=wp.spatial_vector, device=self.device)
        self._actuator_u = wp.zeros(ACTION_COUNT, dtype=wp.float32, device=self.device)
        self._reset_config = newton.solvers.SolverKamino.ResetConfig(
            body_poses=newton.solvers.SolverKamino.ResetConfig.FromActuatorQ(self._home_position),
            body_velocities=newton.solvers.SolverKamino.ResetConfig.FromActuatorU(self._actuator_u),
            base_pose=newton.solvers.SolverKamino.ResetConfig.FromBaseQ(self._base_q),
            base_velocity=newton.solvers.SolverKamino.ResetConfig.FromBaseU(self._base_u),
        )
        self.reset()
        self._warmup_networks()
        self.capture()

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.6, 1.2, 0.65), -12.0, -140.0)

    def _warmup_networks(self):
        output = self.policy({self.policy_input_name: self._observation})
        self.actuator(self._actuator_input)
        if output[self.policy_output_name].shape != (1, ACTION_COUNT):
            raise ValueError("asOverDog policy output must have shape (1, 12)")

    def _simulate_frame(self):
        wp.launch(
            _build_observation_kernel,
            dim=1,
            inputs=[
                self.state_0.body_q,
                self.state_0.body_qd,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.base_body,
                self._policy_coord_indices,
                self._policy_dof_indices,
                self._policy_home,
                self._command,
                self._phase,
                self._action_history,
                self._observation,
            ],
            device=self.device,
        )
        policy_output = self.policy({self.policy_input_name: self._observation})[self.policy_output_name]
        wp.launch(
            _update_action_kernel,
            dim=ACTION_COUNT,
            inputs=[policy_output, self._action_history],
            device=self.device,
        )

        for _ in range(self.sim_substeps):
            wp.launch(
                _build_actuator_input_kernel,
                dim=ACTION_COUNT,
                inputs=[
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    self._motor_coord_indices,
                    self._motor_dof_indices,
                    self._sim_from_policy,
                    self._home_position,
                    self._action_history,
                    self._position_error_history,
                    self._velocity_history,
                    self._actuator_input,
                ],
                device=self.device,
            )
            raw_torque = self.actuator(self._actuator_input)
            self.control.joint_f.zero_()
            wp.launch(
                _apply_motor_torque_kernel,
                dim=ACTION_COUNT,
                inputs=[raw_torque, self.state_0.joint_qd, self._motor_dof_indices, self.control.joint_f],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.solver.update_contacts(self.contacts, self.state_1)
            self.state_0, self.state_1 = self.state_1, self.state_0

        wp.launch(
            _compute_loop_error_kernel,
            dim=len(LOOP_JOINTS),
            inputs=[
                self.state_0.body_q,
                self.model.joint_parent,
                self.model.joint_child,
                self.model.joint_X_p,
                self.model.joint_X_c,
                self._loop_joint_indices,
                self._loop_error,
            ],
            device=self.device,
        )

    def capture(self):
        self.graph = None
        if self.device.is_cuda and self.device.is_mempool_enabled:
            with wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph
            self.reset()

    def reset(self):
        self.solver.reset(self.state_0, config=self._reset_config)
        self.solver.reset(self.state_1, config=self._reset_config)
        self.control.joint_f.zero_()
        self._action_history.zero_()
        self._observation.zero_()
        self._position_error_history.zero_()
        self._velocity_history.zero_()
        self._phase.assign([wp.vec4(1.0, 0.0, 0.0, 1.0)])

    def step(self):
        command = self._fallback_command
        if hasattr(self.viewer, "is_key_down"):
            forward = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lateral = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            yaw_rate = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            if forward != 0.0 or lateral != 0.0 or yaw_rate != 0.0:
                command = wp.vec3(forward, lateral, yaw_rate)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_previous:
                self.reset()
            self._reset_key_previous = reset_down
        self._command.assign([command])

        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        loop_error = self._loop_error.numpy()
        if not np.isfinite(body_q).all():
            raise AssertionError("asOverDog body state contains non-finite values")
        if body_q[self.base_body, 2] <= 0.15:
            raise AssertionError("asOverDog base fell below the expected operating height")
        if not np.isfinite(loop_error).all() or loop_error.max() >= 1.0e-3:
            raise AssertionError(f"asOverDog loop-closure error is too large: {loop_error.max():.6g} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(render_fps=50.0)
        parser.add_argument("--robot", choices=tuple(ROBOT_CONFIGS), default="bennett", help="Leg mechanism to load.")
        parser.add_argument("--vx", type=float, default=0.0, help="Fallback forward velocity command in m/s.")
        parser.add_argument("--vy", type=float, default=0.0, help="Fallback lateral velocity command in m/s.")
        parser.add_argument("--yaw-rate", type=float, default=0.0, help="Fallback yaw-rate command in rad/s.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
