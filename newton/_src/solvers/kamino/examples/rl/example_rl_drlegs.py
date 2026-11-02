# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: DR Legs walk policy play-back
#
# Runs a trained ONNX walk policy on the DR Legs robot using the Kamino
# solver with implicit PD joint control. Velocity commands come from an
# Xbox gamepad or keyboard via the 3-D viewer.
#
# Usage:
#   python example_rl_drlegs.py --policy path/to/model.onnx
#   python example_rl_drlegs.py --policy path/to/model.onnx --mode async
#   python example_rl_drlegs.py --headless --num-steps 200
###########################################################################

import argparse
from pathlib import Path
from typing import ClassVar

import warp as wp
import yaml

import newton
from newton._src.solvers.kamino._src.utils import logger as msg
from newton._src.solvers.kamino._src.utils.viewer import MeshColors, ViewerConfig
from newton._src.solvers.kamino.examples import run_headless
from newton._src.solvers.kamino.examples.rl.joystick import JoystickConfig, JoystickController
from newton._src.solvers.kamino.examples.rl.onnx_policy import WarpOnnxPolicy
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.simulation_runner import SimulationRunner

wp.set_module_options({"enable_backward": False})

_DEFAULTS = {
    "action_scale": 0.4,
    "contact_duration": 0.3,
    "phase_embedding_k": 2,
    "vel_cmd_max": 0.3,
    "yaw_cmd_max": 0.8,
    "pd_kp": 15.0,
    "pd_kd": 0.6,
    "pd_armature": 0.01,
    "path_deviation_scale": 0.1,
    "linear_path_error_limit": 0.1,
    "standing_height": 0.265,
    "height_cmd_min": 0.16,
    "height_cmd_max": 0.27,
    "height_error_scale": 0.05,
    "sim_dt": 0.004,
    "control_decimation": 5,
    "body_pose_offset_z": 0.265,
    "usd_model": "dr_legs/usd/dr_legs_with_meshes_and_boxes.usda",
    "policy_file": "drlegs_walk.onnx",
}

_DRLEGS_ASSET_REF = "a0547548eaa966c2f5478bee496c3cfba1fa98fc"
_DRLEGS_ACTION_WIDTH = 12
_DRLEGS_OBSERVATION_WIDTH = 94


@wp.kernel
def _set_pd_gains_kernel(
    actuated_dof_indices: wp.array[wp.int32],
    kp: wp.float32,
    kd: wp.float32,
    armature: wp.float32,
    joint_kp: wp.array[wp.float32],
    joint_kd: wp.array[wp.float32],
    joint_armature: wp.array[wp.float32],
):
    action_index = wp.tid()
    dof_index = actuated_dof_indices[action_index]
    joint_kp[dof_index] = kp
    joint_kd[dof_index] = kd
    joint_armature[dof_index] = armature


@wp.kernel
def _build_observation_kernel(
    body_q: wp.array[wp.transformf],
    body_u: wp.array[wp.spatial_vectorf],
    joint_q: wp.array[wp.float32],
    actions: wp.array2d[wp.float32],
    command: wp.array[wp.vec4f],
    phase: wp.array[wp.float32],
    path_heading: wp.array[wp.float32],
    path_position: wp.array[wp.vec2f],
    action_history: wp.array2d[wp.float32],
    action_history_prev: wp.array2d[wp.float32],
    root_body_index: wp.int32,
    root_coords_offset: wp.int32,
    root_coords_count: wp.int32,
    joint_coord_count: wp.int32,
    env_dt: wp.float32,
    phase_rate: wp.float32,
    action_scale: wp.float32,
    path_deviation_scale: wp.float32,
    path_error_limit: wp.float32,
    height_error_scale: wp.float32,
    observation: wp.array2d[wp.float32],
):
    cmd = command[0]
    current_phase = wp.mod(phase[0] + env_dt * phase_rate, 1.0)
    phase[0] = current_phase

    heading = path_heading[0]
    mid_heading = heading + 0.5 * env_dt * cmd[2]
    path_delta = wp.quat_rotate(
        wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), mid_heading),
        wp.vec3f(cmd[0], cmd[1], 0.0),
    )
    path = path_position[0] + wp.vec2f(path_delta[0], path_delta[1]) * env_dt
    heading += env_dt * cmd[2]
    path_heading[0] = heading

    root_transform = body_q[root_body_index]
    root_position = wp.transform_get_translation(root_transform)
    root_rotation = wp.transform_get_rotation(root_transform)
    path_error = path - wp.vec2f(root_position[0], root_position[1])
    path_error_length = wp.length(path_error)
    if path_error_length > path_error_limit:
        path = wp.vec2f(root_position[0], root_position[1]) + path_error * path_error_limit / path_error_length
    path_position[0] = path

    path_rotation = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), heading)
    root_in_path = wp.quat_inverse(path_rotation) * root_rotation
    root_rotation_matrix = wp.quat_to_matrix(root_in_path)
    for row in range(3):
        for column in range(3):
            observation[0, row * 3 + column] = root_rotation_matrix[row, column]

    deviation_world = wp.vec3f(root_position[0] - path[0], root_position[1] - path[1], 0.0)
    deviation_path = wp.quat_rotate_inv(path_rotation, deviation_world)
    inverse_deviation_scale = 1.0 / path_deviation_scale
    observation[0, 9] = deviation_path[0] * inverse_deviation_scale
    observation[0, 10] = deviation_path[1] * inverse_deviation_scale

    root_heading = wp.atan2(
        2.0 * (root_in_path[2] * root_in_path[3] + root_in_path[0] * root_in_path[1]),
        root_in_path[3] * root_in_path[3]
        + root_in_path[0] * root_in_path[0]
        - root_in_path[1] * root_in_path[1]
        - root_in_path[2] * root_in_path[2],
    )
    heading_rotation = wp.quat_from_axis_angle(wp.vec3f(0.0, 0.0, 1.0), root_heading)
    deviation_heading = wp.quat_rotate_inv(heading_rotation, -deviation_path)
    observation[0, 11] = deviation_heading[0] * inverse_deviation_scale
    observation[0, 12] = deviation_heading[1] * inverse_deviation_scale

    observation[0, 13] = cmd[0]
    observation[0, 14] = cmd[1]
    observation[0, 15] = cmd[2]

    command_linear_root = wp.quat_rotate_inv(root_in_path, wp.vec3f(cmd[0], cmd[1], 0.0))
    command_angular_root = wp.quat_rotate_inv(root_in_path, wp.vec3f(0.0, 0.0, cmd[2]))
    for axis in range(3):
        observation[0, 16 + axis] = command_linear_root[axis]
        observation[0, 19 + axis] = command_angular_root[axis]

    observation[0, 22] = wp.cos(2.0 * wp.pi * current_phase)
    observation[0, 23] = wp.sin(2.0 * wp.pi * current_phase)
    observation[0, 24] = wp.cos(4.0 * wp.pi * current_phase)
    observation[0, 25] = wp.sin(4.0 * wp.pi * current_phase)

    root_velocity = body_u[root_body_index]
    root_linear_velocity = wp.quat_rotate_inv(root_rotation, wp.spatial_top(root_velocity))
    root_angular_velocity = wp.quat_rotate_inv(root_rotation, wp.spatial_bottom(root_velocity))
    for axis in range(3):
        observation[0, 26 + axis] = root_linear_velocity[axis]
        observation[0, 29 + axis] = root_angular_velocity[axis]

    observation[0, 32] = cmd[3]
    observation[0, 33] = (root_position[2] - cmd[3]) / height_error_scale

    output_index = wp.int32(34)
    root_coords_end = root_coords_offset + root_coords_count
    for coord_index in range(joint_coord_count):
        if coord_index < root_coords_offset or coord_index >= root_coords_end:
            observation[0, output_index] = joint_q[coord_index]
            output_index += 1

    for action_index in range(_DRLEGS_ACTION_WIDTH):
        previous_action = action_history[0, action_index]
        scaled_action = action_scale * actions[0, action_index]
        action_history_prev[0, action_index] = previous_action
        action_history[0, action_index] = scaled_action
        observation[0, output_index + action_index] = scaled_action
        observation[0, output_index + _DRLEGS_ACTION_WIDTH + action_index] = previous_action


@wp.kernel
def _apply_actions_kernel(
    actions: wp.array2d[wp.float32],
    actuated_coord_indices: wp.array[wp.int32],
    actuated_dof_indices: wp.array[wp.int32],
    action_scale: wp.float32,
    joint_position_target: wp.array[wp.float32],
    joint_velocity_target: wp.array[wp.float32],
):
    action_index = wp.tid()
    joint_position_target[actuated_coord_indices[action_index]] = action_scale * actions[0, action_index]
    joint_velocity_target[actuated_dof_indices[action_index]] = 0.0


@wp.kernel
def _random_actions_kernel(step: wp.int32, actions: wp.array2d[wp.float32]):
    action_index = wp.tid()
    random_state = wp.rand_init(42, step * _DRLEGS_ACTION_WIDTH + action_index)
    actions[0, action_index] = wp.randf(random_state, -1.0, 1.0)


@wp.kernel
def _reset_path_kernel(
    body_q: wp.array[wp.transformf],
    root_body_index: wp.int32,
    path_position: wp.array[wp.vec2f],
):
    root_position = wp.transform_get_translation(body_q[root_body_index])
    path_position[0] = wp.vec2f(root_position[0], root_position[1])


def _load_drlegs_config(asset_path: Path) -> dict:
    """Load walk configuration from the asset, falling back to built-in defaults."""
    config = dict(_DEFAULTS)
    yaml_path = asset_path / "dr_legs" / "rl_policies" / "drlegs_walk.yaml"
    if yaml_path.exists():
        with open(yaml_path, encoding="utf-8") as file:
            config.update(yaml.safe_load(file) or {})
        msg.info(f"Loaded config from {yaml_path}")
    else:
        msg.info("No YAML config found, using built-in defaults")
    config["phase_rate"] = 1.0 / (2.0 * config["contact_duration"])
    if config["phase_embedding_k"] != 2:
        raise ValueError("The DR Legs ONNX policy requires phase_embedding_k=2")
    return config


class Example:
    """Run the DR Legs ONNX walk policy without a PyTorch dependency."""

    BODY_GROUP_COLORS: ClassVar[dict] = {
        "pelvis": MeshColors.BONE,
        "hip_servos": MeshColors.DARK,
        "upperleg_link": MeshColors.SAGEGREY,
        "lowerleg_link": MeshColors.BONE,
        "ankle_bracket": MeshColors.SAGEGREY,
        "foot": MeshColors.DARK,
        "servohorn": MeshColors.DARK,
        "upperleg_rod": MeshColors.DARK,
    }

    def __init__(
        self,
        config: dict,
        device: wp.DeviceLike = None,
        policy=None,
        headless: bool = False,
        max_steps: int = 10000,
    ):
        self.cfg = config
        self.sim_dt = config["sim_dt"]
        self.control_decimation = config["control_decimation"]
        self.env_dt = self.sim_dt * self.control_decimation
        self.max_steps = max_steps
        self.policy = policy
        self._step_count = 0
        self.device = wp.get_device(device)

        asset_path = newton.utils.download_asset("disneyresearch", ref=_DRLEGS_ASSET_REF)
        usd_model_path = str(asset_path / config["usd_model"])
        self.sim_wrapper = RigidBodySim(
            usd_model_path=usd_model_path,
            num_worlds=1,
            sim_dt=self.sim_dt,
            device=self.device,
            headless=headless,
            body_pose_offset=(0.0, 0.0, config["body_pose_offset_z"], 0.0, 0.0, 0.0, 1.0),
            use_cuda_graph=True,
            render_config=ViewerConfig(diffuse_scale=1.0, specular_scale=0.3, shadow_radius=10.0),
            use_torch=False,
        )
        self._apply_body_group_colors()

        model = self.sim_wrapper.sim.model
        self._root_body_index = int(model.info.base_body_index.numpy()[0])
        root_joint_index = int(model.info.base_joint_index.numpy()[0])
        if self._root_body_index < 0 or root_joint_index < 0:
            raise ValueError("The DR Legs policy requires a floating-base articulation")
        self._root_coords_offset = int(model.joints.coords_offset.numpy()[root_joint_index])
        self._root_coords_count = int(model.joints.num_coords.numpy()[root_joint_index])
        self._joint_coord_count = model.size.max_of_num_joint_coords
        policy_joint_coord_count = self._joint_coord_count - self._root_coords_count
        if policy_joint_coord_count != 36:
            raise ValueError(
                f"The DR Legs policy requires 36 non-root joint coordinates, got {policy_joint_coord_count}"
            )
        if self.sim_wrapper.num_actuated != _DRLEGS_ACTION_WIDTH:
            raise ValueError(
                f"The DR Legs policy requires {_DRLEGS_ACTION_WIDTH} actuated joints, "
                f"got {self.sim_wrapper.num_actuated}"
            )

        self._configure_pd_gains()
        self._observation = wp.zeros((1, _DRLEGS_OBSERVATION_WIDTH), dtype=wp.float32, device=self.device)
        self.actions = wp.zeros((1, _DRLEGS_ACTION_WIDTH), dtype=wp.float32, device=self.device)
        self._action_history = wp.zeros_like(self.actions)
        self._action_history_prev = wp.zeros_like(self.actions)
        self._phase = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._path_heading = wp.zeros(1, dtype=wp.float32, device=self.device)
        self._path_position = wp.zeros(1, dtype=wp.vec2f, device=self.device)
        self._command_height = config["standing_height"]
        self._command = wp.array([wp.vec4f(0.0, 0.0, 0.0, self._command_height)], dtype=wp.vec4f, device=self.device)

        self.joystick = JoystickController(
            dt=self.env_dt,
            viewer=self.sim_wrapper.viewer,
            num_worlds=1,
            config=JoystickConfig(
                forward_velocity_base=config["vel_cmd_max"],
                forward_velocity_turbo=0.0,
                lateral_velocity_base=config["vel_cmd_max"],
                lateral_velocity_turbo=0.0,
                angular_velocity_base=config["yaw_cmd_max"],
                angular_velocity_turbo=0.0,
            ),
            track_path=False,
        )
        self.reset()

    @property
    def viewer(self):
        return self.sim_wrapper.viewer

    def _configure_pd_gains(self) -> None:
        joints = self.sim_wrapper.sim.model.joints
        joints.k_p_j.zero_()
        joints.k_d_j.zero_()
        joints.b_j.zero_()
        wp.launch(
            _set_pd_gains_kernel,
            dim=_DRLEGS_ACTION_WIDTH,
            inputs=[
                self.sim_wrapper.actuated_dof_indices_tensor,
                self.cfg["pd_kp"],
                self.cfg["pd_kd"],
                self.cfg["pd_armature"],
                joints.k_p_j,
                joints.k_d_j,
                joints.a_j,
            ],
            device=self.device,
        )

    def _apply_body_group_colors(self) -> None:
        if self.viewer is None:
            return
        model = self.sim_wrapper._newton_model
        shape_body = model.shape_body.numpy()
        for shape_index in range(model.shape_count):
            body_index = int(shape_body[shape_index])
            if body_index < 0:
                continue
            body_name = model.body_label[body_index].rsplit("/", 1)[-1]
            for prefix, color in self.BODY_GROUP_COLORS.items():
                if body_name.startswith(prefix):
                    model.shape_color[shape_index : shape_index + 1].fill_(wp.vec3(color))
                    break

    def poll_input(self) -> None:
        """Update scalar input commands without constructing tensor objects."""
        self.joystick.update()
        if self.joystick.input_mode == "joystick":
            pitch = self.joystick.head_pitch
            if pitch >= 0.0:
                ratio = min(1.0, pitch / self.joystick.head_pitch_up_limit)
                self._command_height = self.cfg["standing_height"] + ratio * (
                    self.cfg["height_cmd_max"] - self.cfg["standing_height"]
                )
            else:
                ratio = min(1.0, -pitch / self.joystick.head_pitch_down_limit)
                self._command_height = self.cfg["standing_height"] - ratio * (
                    self.cfg["standing_height"] - self.cfg["height_cmd_min"]
                )
        elif self.viewer is not None and hasattr(self.viewer, "is_key_down"):
            if self.viewer.is_key_down("y"):
                self._command_height = min(self._command_height + 0.001, self.cfg["height_cmd_max"])
            if self.viewer.is_key_down("n"):
                self._command_height = max(self._command_height - 0.001, self.cfg["height_cmd_min"])

        self._command.assign(
            [
                wp.vec4f(
                    self.joystick.forward_velocity,
                    self.joystick.lateral_velocity,
                    self.joystick.angular_velocity,
                    self._command_height,
                )
            ]
        )

    def _build_observation(self) -> None:
        wp.launch(
            _build_observation_kernel,
            dim=1,
            inputs=[
                self.sim_wrapper.sim.state.q_i,
                self.sim_wrapper.sim.state.u_i,
                self.sim_wrapper.sim.state.q_j,
                self.actions,
                self._command,
                self._phase,
                self._path_heading,
                self._path_position,
                self._action_history,
                self._action_history_prev,
                self._root_body_index,
                self._root_coords_offset,
                self._root_coords_count,
                self._joint_coord_count,
                self.env_dt,
                self.cfg["phase_rate"],
                self.cfg["action_scale"],
                self.cfg["path_deviation_scale"],
                self.cfg["linear_path_error_limit"],
                self.cfg["height_error_scale"],
                self._observation,
            ],
            device=self.device,
        )

    def _apply_actions(self) -> None:
        control = self.sim_wrapper.sim.control
        control.q_j_ref.zero_()
        control.dq_j_ref.zero_()
        wp.launch(
            _apply_actions_kernel,
            dim=_DRLEGS_ACTION_WIDTH,
            inputs=[
                self.actions,
                self.sim_wrapper.actuated_coord_indices_tensor,
                self.sim_wrapper.actuated_dof_indices_tensor,
                self.cfg["action_scale"],
                control.q_j_ref,
                control.dq_j_ref,
            ],
            device=self.device,
        )

    def reset(self) -> None:
        """Reset the simulation and Warp-native policy state."""
        self.sim_wrapper.reset()
        self.actions.zero_()
        self._action_history.zero_()
        self._action_history_prev.zero_()
        self._phase.zero_()
        self._path_heading.zero_()
        self._step_count = 0
        self._command_height = self.cfg["standing_height"]
        self._command.fill_(wp.vec4f(0.0, 0.0, 0.0, self._command_height))
        self.sim_wrapper.sim.control.q_j_ref.zero_()
        self.sim_wrapper.sim.control.dq_j_ref.zero_()
        wp.launch(
            _reset_path_kernel,
            dim=1,
            inputs=[self.sim_wrapper.sim.state.q_i, self._root_body_index, self._path_position],
            device=self.device,
        )
        self.joystick.reset()

    def sim_step(self) -> None:
        """Build observations, evaluate the policy, apply actions, and step physics."""
        self._build_observation()
        if self.policy is not None:
            wp.copy(self.actions, self.policy(self._observation))
        else:
            wp.launch(
                _random_actions_kernel,
                dim=_DRLEGS_ACTION_WIDTH,
                inputs=[self._step_count, self.actions],
                device=self.device,
            )
        self._apply_actions()
        for _ in range(self.control_decimation):
            self.sim_wrapper.step()
        self._step_count += 1

    def step_once(self) -> None:
        """Advance one policy step for headless execution."""
        self.sim_step()

    def step(self) -> None:
        """Process input and advance one policy step."""
        if self.joystick.check_reset():
            self.reset()
        self.poll_input()
        self.sim_step()

    def render(self) -> None:
        """Render the current simulation state."""
        self.sim_wrapper.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DR Legs walk policy play example")
    parser.add_argument("--device", type=str, help="The compute device to use")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False, help="Run headlessly")
    parser.add_argument("--num-steps", type=int, default=10000, help="Policy steps for headless mode")
    parser.add_argument(
        "--control-decimation", type=int, default=None, help="Physics substeps per policy step (overrides YAML)"
    )
    parser.add_argument("--sim-dt", type=float, default=None, help="Physics timestep in seconds (overrides YAML)")
    parser.add_argument("--policy", type=str, default=None, help="ONNX policy path (overrides the asset default)")
    parser.add_argument("--mode", choices=["sync", "async"], default="sync", help="Simulation loop mode")
    parser.add_argument("--render-fps", type=float, default=30.0, help="Target render rate in async mode")
    args = parser.parse_args()

    msg.set_log_level(msg.LogLevel.INFO)
    device = wp.get_device(args.device) if args.device else wp.get_preferred_device()
    wp.set_device(device)
    msg.info(f"device: {device}")

    asset_path = newton.utils.download_asset("disneyresearch", ref=_DRLEGS_ASSET_REF)
    config = _load_drlegs_config(asset_path)
    if args.sim_dt is not None:
        config["sim_dt"] = args.sim_dt
    if args.control_decimation is not None:
        config["control_decimation"] = args.control_decimation

    policy = None
    if args.policy:
        policy_path = Path(args.policy)
        if policy_path.suffix.lower() != ".onnx" or not policy_path.is_file():
            raise FileNotFoundError(f"Expected an existing ONNX policy, got '{policy_path}'")
        policy = WarpOnnxPolicy(policy_path, device=device, batch_size=1, action_width=_DRLEGS_ACTION_WIDTH)
        msg.info(f"Loaded policy from: {policy_path}")
    else:
        policy_path = asset_path / "dr_legs" / "rl_policies" / config["policy_file"]
        if policy_path.exists():
            policy = WarpOnnxPolicy(policy_path, device=device, batch_size=1, action_width=_DRLEGS_ACTION_WIDTH)
            msg.info(f"Loaded default policy from: {policy_path}")
        else:
            msg.info(f"No policy at {policy_path} -- using random actions")

    example = Example(config=config, device=device, policy=policy, headless=args.headless, max_steps=args.num_steps)
    try:
        if args.headless:
            msg.notif("Running in headless mode...")
            run_headless(example, progress=True)
        else:
            msg.notif(f"Running in Viewer mode ({args.mode})...")
            if hasattr(example.viewer, "set_camera"):
                example.viewer.set_camera(wp.vec3(0.6, 0.6, 0.3), -10.0, 225.0)
            SimulationRunner(example, mode=args.mode, render_fps=args.render_fps).run()
    except KeyboardInterrupt:
        pass
    finally:
        example.joystick.close()
