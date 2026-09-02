# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot asRoBallet
#
# Demonstrates LQR and learned-policy control for the asRoBallet robot
# (RSS 2026: https://arxiv.org/abs/2604.24916).
#
# Hold "i"/"k" to move forward/backward and "j"/"l" to move left/right.
# Hold "u"/"o" to turn left/right. Nonzero velocity arguments select fixed
# command mode instead of keyboard control. The policy controller is used by
# default; pass "--controller lqr" to use LQR.
# Pass "--virtual-ball-joint" to constrain the ball center to the base using
# the original asRoBallet spherical joint.
#
# Command: python -m newton.examples robot_asroballet
#
###########################################################################

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from scipy import linalg, signal

import newton
import newton.examples
import newton.utils
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes

ASROBALLET_SCENE_PATH = Path("mjcf") / "scene_floating_ball.xml"
ASROBALLET_VIRTUAL_BALL_JOINT_SCENE_PATH = Path("mjcf") / "scene.xml"
ASROBALLET_VELOCITY_POLICY_PATH = Path("rl_policies") / "velocity_tracking.onnx"
ASROBALLET_STATION_POLICY_PATH = Path("rl_policies") / "station_keeping.onnx"
BODY_COLLISION_GROUP = 1
BALL_COLLISION_GROUP = 2
ROLLER_COLLISION_GROUP = -1
GROUND_COLLISION_GROUP = -1
# Compensate for the effective rolling ratio of the STL ball-and-roller assembly.
VELOCITY_COMMAND_SCALE = 0.9
# Compensate for torsional friction in steady-state yaw tracking.
YAW_COMMAND_SCALE = 1.04
KEYBOARD_LINEAR_SPEED = 0.5
KEYBOARD_YAW_RATE = 1.0
KEYBOARD_LINEAR_ACCELERATION = 0.5
KEYBOARD_YAW_ACCELERATION = 2.0
MESH_MAX_HULL_VERTICES = 1024


def download_asroballet_assets() -> Path:
    """Download the asRoBallet model and policies at their pinned revision."""
    return newton.utils.download_asset("asroballet")


def resolve_asroballet_model_path(asset_root: Path, virtual_ball_joint: bool) -> str:
    """Resolve the shared MJCF scene used by every controller."""
    scene_path = ASROBALLET_VIRTUAL_BALL_JOINT_SCENE_PATH if virtual_ball_joint else ASROBALLET_SCENE_PATH
    return str(asset_root / scene_path)


def keyboard_velocity_command(viewer) -> np.ndarray:
    """Return the body-relative velocity commanded by held keyboard keys."""
    forward = float(viewer.is_key_down("i")) - float(viewer.is_key_down("k"))
    lateral = float(viewer.is_key_down("j")) - float(viewer.is_key_down("l"))
    yaw = float(viewer.is_key_down("u")) - float(viewer.is_key_down("o"))
    return np.array(
        [
            KEYBOARD_LINEAR_SPEED * forward,
            KEYBOARD_LINEAR_SPEED * lateral,
            KEYBOARD_YAW_RATE * yaw,
        ],
        dtype=np.float64,
    )


def interactive_velocity_command(viewer, gui_command: np.ndarray | None = None) -> np.ndarray:
    """Combine held keyboard and GUI controls into one body-relative command."""
    if gui_command is None:
        gui_command = np.zeros(3, dtype=np.float64)
    command = keyboard_velocity_command(viewer) + gui_command
    limits = np.array([KEYBOARD_LINEAR_SPEED, KEYBOARD_LINEAR_SPEED, KEYBOARD_YAW_RATE])
    return np.clip(command, -limits, limits)


def should_use_keyboard_control(viewer_name: str, headless: bool, fixed_command: np.ndarray) -> bool:
    """Return whether a run should accept interactive keyboard commands."""
    return viewer_name in {"gl", "rtx"} and not headless and not np.any(fixed_command)


def move_toward(current: np.ndarray, target: np.ndarray, max_delta: np.ndarray) -> np.ndarray:
    """Move each current value toward its target without overshooting."""
    delta = np.clip(target - current, -max_delta, max_delta)
    return current + delta


@wp.kernel
def _lqr_control_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    base_body: int,
    wheel_q_indices: wp.array[int],
    wheel_ctrl_indices: wp.array[int],
    command: wp.array[wp.vec3],
    position_reference: wp.array[wp.vec3],
    track_planar_velocity: bool,
    roll_gain: wp.array[float],
    pitch_gain: wp.array[float],
    yaw_gain: wp.array[float],
    ball_to_control: wp.array2d[float],
    wheel_state_to_ball: wp.array2d[float],
    sphere_radius: float,
    control_limit: float,
    control: wp.array[float],
):
    pose = body_q[base_body]
    position = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    velocity = wp.quat_rotate_inv(rotation, wp.spatial_top(body_qd[base_body]))
    angular_velocity = wp.quat_rotate_inv(rotation, wp.spatial_bottom(body_qd[base_body]))
    gravity_direction = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, -1.0))

    roll = wp.atan2(-gravity_direction[1], -gravity_direction[2])
    pitch = wp.atan2(gravity_direction[0], -gravity_direction[2])
    desired_velocity = command[0]

    wheel_position = wp.vec3(
        joint_q[wheel_q_indices[0]],
        joint_q[wheel_q_indices[1]],
        joint_q[wheel_q_indices[2]],
    )
    ball_angle = wp.vec3()
    for axis in range(3):
        ball_angle[axis] = (
            wheel_state_to_ball[axis, 0] * wheel_position[0]
            + wheel_state_to_ball[axis, 1] * wheel_position[1]
            + wheel_state_to_ball[axis, 2] * wheel_position[2]
        )

    if not track_planar_velocity:
        reference = position_reference[0]
        position_error = wp.quat_rotate_inv(rotation, position - reference)
        ball_angle[0] = -position_error[1] / sphere_radius
        ball_angle[1] = position_error[0] / sphere_radius

    roll_rate_error = -(velocity[1] - VELOCITY_COMMAND_SCALE * desired_velocity[1]) / sphere_radius
    pitch_rate_error = (velocity[0] - VELOCITY_COMMAND_SCALE * desired_velocity[0]) / sphere_radius

    roll_torque = -(
        roll_gain[0] * roll
        + roll_gain[1] * ball_angle[0]
        + roll_gain[2] * angular_velocity[0]
        + roll_gain[3] * roll_rate_error
    )
    pitch_torque = -(
        pitch_gain[0] * pitch
        + pitch_gain[1] * ball_angle[1]
        + pitch_gain[2] * angular_velocity[1]
        + pitch_gain[3] * pitch_rate_error
    )
    yaw_torque = -yaw_gain[0] * (angular_velocity[2] - YAW_COMMAND_SCALE * desired_velocity[2])

    virtual_torque = wp.vec3(roll_torque, pitch_torque, yaw_torque)
    for wheel in range(3):
        wheel_control = (
            ball_to_control[wheel, 0] * virtual_torque[0]
            + ball_to_control[wheel, 1] * virtual_torque[1]
            + ball_to_control[wheel, 2] * virtual_torque[2]
        )
        control[wheel_ctrl_indices[wheel]] = wp.clamp(wheel_control, -control_limit, control_limit)


@wp.kernel
def _latch_position_reference_kernel(
    body_q: wp.array[wp.transform],
    base_body: int,
    position_reference: wp.array[wp.vec3],
):
    position_reference[0] = wp.transform_get_translation(body_q[base_body])


def launch_lqr_control(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    control: wp.array[float],
    base_body: int,
    wheel_q_indices: wp.array[int],
    wheel_ctrl_indices: wp.array[int],
    command: wp.array[wp.vec3],
    position_reference: wp.array[wp.vec3],
    track_planar_velocity: bool,
    roll_gain: wp.array[float],
    pitch_gain: wp.array[float],
    yaw_gain: wp.array[float],
    ball_to_control: wp.array2d[float],
    wheel_state_to_ball: wp.array2d[float],
    sphere_radius: float,
    control_limit: float,
    device,
) -> None:
    """Launch one asRoBallet LQR control update."""
    wp.launch(
        _lqr_control_kernel,
        dim=1,
        inputs=[
            body_q,
            body_qd,
            joint_q,
            base_body,
            wheel_q_indices,
            wheel_ctrl_indices,
            command,
            position_reference,
            track_planar_velocity,
            roll_gain,
            pitch_gain,
            yaw_gain,
            ball_to_control,
            wheel_state_to_ball,
            sphere_radius,
            control_limit,
            control,
        ],
        device=device,
    )


@dataclass(frozen=True)
class LqrModelParameters:
    """Rigid-body parameters of the asRoBallet balancing model.

    Args:
        upper_mass: Upper-body mass [kg].
        sphere_mass: Ball mass [kg].
        sphere_radius: Ball radius [m].
        com_height: Upper-body center-of-mass height above the ball center [m].
        upper_inertia_roll: Upper-body roll inertia [kg m^2].
        upper_inertia_pitch: Upper-body pitch inertia [kg m^2].
        upper_inertia_yaw: Upper-body yaw inertia [kg m^2].
        sphere_inertia: Ball inertia about a diameter [kg m^2].
        gravity: Gravitational acceleration magnitude [m/s^2].
        rolling_damping: Viscous rolling damping coefficient [N m s/rad].
    """

    upper_mass: float = 13.37187446932556
    sphere_mass: float = 3.021143163744471
    sphere_radius: float = 0.115
    com_height: float = 0.5290457799688545
    upper_inertia_roll: float = 1.3594327
    upper_inertia_pitch: float = 1.18643043
    upper_inertia_yaw: float = 0.21719836
    sphere_inertia: float = 0.02453707
    gravity: float = 9.81
    rolling_damping: float = 0.1


def build_wheel_control_matrix(
    sphere_radius: float = 0.115,
    wheel_radius: float = 0.05,
    wheel_gear: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mappings between wheel torque, virtual ball torque, and actuator control."""
    radial_directions = np.array(
        [
            [0.62120968, 0.0, 0.78364439],
            [-0.30924857, 0.53757604, 0.78445990],
            [-0.30947051, -0.53796185, 0.78410781],
        ],
        dtype=np.float64,
    )
    axle_axes = np.array(
        [
            [0.70710678, 0.0, -0.70710678],
            [-0.35355309, 0.61237215, -0.70710718],
            [-0.35355309, -0.61237215, -0.70710718],
        ],
        dtype=np.float64,
    )
    drive_directions = np.cross(radial_directions, axle_axes)
    drive_directions /= np.linalg.norm(drive_directions, axis=1, keepdims=True)

    wheel_to_ball = np.column_stack(
        [
            np.cross(sphere_radius * radial, drive / wheel_radius)
            for radial, drive in zip(radial_directions, drive_directions, strict=True)
        ]
    )
    ball_to_control = np.linalg.inv(wheel_to_ball) / wheel_gear
    return wheel_to_ball, ball_to_control


def build_wheel_state_matrix(wheel_to_ball: np.ndarray) -> np.ndarray:
    """Build the mapping from wheel joint state to virtual ball state."""
    return np.linalg.inv(wheel_to_ball.T)


def build_planar_state_space(
    parameters: LqrModelParameters,
    axis: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the full-state continuous-time model for one balancing plane."""
    if axis == "roll":
        upper_inertia = parameters.upper_inertia_roll
    elif axis == "pitch":
        upper_inertia = parameters.upper_inertia_pitch
    else:
        raise ValueError(f"Unsupported balancing axis '{axis}'.")

    m_u = parameters.upper_mass
    m_s = parameters.sphere_mass
    r_s = parameters.sphere_radius
    l_u = parameters.com_height
    i_s = parameters.sphere_inertia
    f_r = parameters.rolling_damping

    upper_pivot_inertia = m_u * l_u**2 + upper_inertia
    rolling_inertia = (m_s + m_u) * r_s**2 + i_s
    determinant = upper_pivot_inertia * rolling_inertia - (m_u * r_s * l_u) ** 2

    a = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [
                m_u * parameters.gravity * l_u * rolling_inertia / determinant,
                0.0,
                0.0,
                m_u * r_s * l_u * f_r / determinant,
            ],
            [
                -(m_u**2) * r_s * l_u**2 * parameters.gravity / determinant,
                0.0,
                0.0,
                -upper_pivot_inertia * f_r / determinant,
            ],
        ],
        dtype=np.float64,
    )
    b = np.array(
        [
            [0.0],
            [0.0],
            [-(rolling_inertia + m_u * r_s * l_u) / determinant],
            [(upper_pivot_inertia + m_u * r_s * l_u) / determinant],
        ],
        dtype=np.float64,
    )
    return a, b


def build_yaw_state_space(parameters: LqrModelParameters) -> tuple[np.ndarray, np.ndarray]:
    """Build the continuous-time yaw-rate model."""
    a = np.array([[-parameters.rolling_damping / parameters.upper_inertia_yaw]], dtype=np.float64)
    b = np.array([[-1.0 / parameters.upper_inertia_yaw]], dtype=np.float64)
    return a, b


def solve_discrete_lqr(
    continuous_a: np.ndarray,
    continuous_b: np.ndarray,
    timestep: float,
    state_cost: np.ndarray,
    control_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discretize a continuous model and solve its infinite-horizon LQR."""
    state_count = continuous_a.shape[0]
    discrete_a, discrete_b, _, _, _ = signal.cont2discrete(
        (
            continuous_a,
            continuous_b,
            np.eye(state_count, dtype=np.float64),
            np.zeros((state_count, continuous_b.shape[1]), dtype=np.float64),
        ),
        timestep,
    )
    riccati = linalg.solve_discrete_are(discrete_a, discrete_b, state_cost, control_cost)
    gain = np.linalg.solve(
        discrete_b.T @ riccati @ discrete_b + control_cost,
        discrete_b.T @ riccati @ discrete_a,
    )
    return discrete_a, discrete_b, gain


def find_asroballet_joint_indices(builder: newton.ModelBuilder, joint_name: str) -> tuple[int, int]:
    """Locate one joint coordinate and its MuJoCo control index."""
    matching_joints = [joint for joint, label in enumerate(builder.joint_label) if label.endswith(f"/{joint_name}")]
    if len(matching_joints) != 1:
        raise ValueError(f"Expected one joint named '{joint_name}', found {len(matching_joints)}.")

    joint = matching_joints[0]
    q_index = builder.joint_q_start[joint]
    qd_index = builder.joint_qd_start[joint]
    actuator_targets = builder.custom_attributes["mujoco:actuator_trnid"].values
    matching_actuators = [actuator for actuator, target in enumerate(actuator_targets) if int(target[0]) == qd_index]
    if len(matching_actuators) != 1:
        raise ValueError(f"Expected one actuator for '{joint_name}', found {len(matching_actuators)}.")

    return q_index, matching_actuators[0]


def configure_asroballet_downward_arm_pose(builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
    """Set both elbows to the vertically downward pose and locate their actuators."""
    elbow_q_indices = []
    elbow_ctrl_indices = []

    for elbow_name in ("right_elbow_joint", "left_elbow_joint"):
        q_index, ctrl_index = find_asroballet_joint_indices(builder, elbow_name)
        builder.joint_q[q_index] = math.pi / 2.0
        elbow_q_indices.append(q_index)
        elbow_ctrl_indices.append(ctrl_index)

    return elbow_q_indices, elbow_ctrl_indices


def find_asroballet_wheel_indices(builder: newton.ModelBuilder) -> tuple[list[int], list[int]]:
    """Locate the three driven wheel coordinates and control indices."""
    wheel_q_indices = []
    wheel_ctrl_indices = []
    for wheel in range(1, 4):
        q_index, ctrl_index = find_asroballet_joint_indices(builder, f"wheel_{wheel}_axle_joint")
        wheel_q_indices.append(q_index)
        wheel_ctrl_indices.append(ctrl_index)

    return wheel_q_indices, wheel_ctrl_indices


def initialize_asroballet_control(control: newton.Control, elbow_ctrl_indices: list[int]) -> None:
    """Initialize wheel controls to zero and hold both elbows downward."""
    ctrl = [0.0] * control.mujoco.ctrl.shape[0]
    for actuator in elbow_ctrl_indices:
        ctrl[actuator] = math.pi / 2.0
    control.mujoco.ctrl.assign(ctrl)


def validate_asroballet_ball_topology(
    builder: newton.ModelBuilder,
    virtual_ball_joint: bool,
) -> None:
    """Validate whether the imported ball uses the selected topology."""
    base_bodies = [body for body, label in enumerate(builder.body_label) if label.endswith("/base_link")]
    ball_bodies = [body for body, label in enumerate(builder.body_label) if label.endswith("/ball_link")]
    if len(base_bodies) != 1 or len(ball_bodies) != 1:
        raise ValueError(
            "Expected one asRoBallet base and ball body, "
            f"found {len(base_bodies)} base and {len(ball_bodies)} ball bodies."
        )

    base_body = base_bodies[0]
    ball_body = ball_bodies[0]
    ball_joints = [joint for joint, child in enumerate(builder.joint_child) if child == ball_body]
    if len(ball_joints) != 1:
        raise ValueError(f"Expected one asRoBallet ball root joint, found {len(ball_joints)}.")

    ball_joint = ball_joints[0]
    expected_type = newton.JointType.BALL if virtual_ball_joint else newton.JointType.FREE
    expected_parent = base_body if virtual_ball_joint else -1
    if builder.joint_type[ball_joint] != expected_type or builder.joint_parent[ball_joint] != expected_parent:
        mode = "virtual ball-joint" if virtual_ball_joint else "independent floating-ball"
        raise ValueError(f"The selected MJCF does not provide the expected {mode} topology.")


def build_asroballet_builder(model_path: str, virtual_ball_joint: bool) -> newton.ModelBuilder:
    """Build the shared asRoBallet simulation model."""
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.rigid_gap = 0.0

    builder.add_mjcf(
        model_path,
        ctrl_direct=True,
        enable_self_collisions=True,
        ignore_inertial_definitions=False,
        # Preserve the contact geometry used to train the policies and used by LQR.
        mesh_maxhullvert=MESH_MAX_HULL_VERTICES,
    )
    validate_asroballet_ball_topology(builder, virtual_ball_joint)

    body_shapes, roller_shapes, ball_shape = classify_asroballet_collision_shapes(builder, shape_start=0)
    # The MJCF "rubber" class is imported as a collider, but the ball is also its visual geometry.
    builder.shape_flags[ball_shape] |= int(newton.ShapeFlags.VISIBLE)
    builder.shape_color[ball_shape] = wp.vec3(0.2, 0.2, 0.2)
    ground_shapes = [
        shape
        for shape in range(builder.shape_count)
        if builder.shape_label[shape].endswith("/floor") and builder.shape_body[shape] == -1
    ]
    if len(ground_shapes) != 1:
        raise ValueError(f"Expected one asRoBallet scene ground shape, found {len(ground_shapes)}.")

    ground_shape = ground_shapes[0]
    builder.shape_flags[ground_shape] |= int(newton.ShapeFlags.VISIBLE)
    builder.shape_color[ground_shape] = wp.vec3(0.125, 0.125, 0.15)
    configure_asroballet_collision_groups(
        builder,
        body_shapes=body_shapes,
        roller_shapes=roller_shapes,
        ball_shape=ball_shape,
        ground_shape=ground_shape,
    )

    return builder


def classify_asroballet_collision_shapes(
    builder: newton.ModelBuilder,
    shape_start: int,
) -> tuple[list[int], list[int], int]:
    """Classify colliding shapes imported from the asRoBallet MJCF."""
    body_shapes = []
    roller_shapes = []
    ball_shapes = []

    for shape in range(shape_start, builder.shape_count):
        if not builder.shape_flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue

        label = builder.shape_label[shape]
        if builder.shape_body[shape] == -1:
            continue
        if "_roller_" in label:
            roller_shapes.append(shape)
        elif label.endswith("/ball_link/ball_link_geom"):
            ball_shapes.append(shape)
        else:
            body_shapes.append(shape)

    actual_counts = (len(body_shapes), len(roller_shapes), len(ball_shapes))
    expected_counts = (10, 36, 1)
    if actual_counts != expected_counts:
        raise ValueError(
            "Unexpected asRoBallet collision topology: "
            f"expected body/roller/ball counts {expected_counts}, got {actual_counts}."
        )

    return body_shapes, roller_shapes, ball_shapes[0]


def configure_asroballet_collision_groups(
    builder: newton.ModelBuilder,
    body_shapes: list[int],
    roller_shapes: list[int],
    ball_shape: int,
    ground_shape: int,
) -> None:
    """Configure Newton collision groups equivalent to the asRoBallet MJCF masks."""
    builder.shape_collision_group[ground_shape] = GROUND_COLLISION_GROUP
    builder.shape_collision_group[ball_shape] = BALL_COLLISION_GROUP

    for shape in body_shapes:
        builder.shape_collision_group[shape] = BODY_COLLISION_GROUP
    for shape in roller_shapes:
        builder.shape_collision_group[shape] = ROLLER_COLLISION_GROUP

    for body_shape in body_shapes:
        for roller_shape in roller_shapes:
            builder.add_shape_collision_filter_pair(body_shape, roller_shape)

    for roller_index, roller_shape in enumerate(roller_shapes):
        for other_roller_shape in roller_shapes[roller_index + 1 :]:
            builder.add_shape_collision_filter_pair(roller_shape, other_roller_shape)


class _AsRoBalletController:
    """Own the simulation state shared by all asRoBallet controllers."""

    def __init__(self, viewer, model_path: str, virtual_ball_joint: bool):
        newton.use_coord_layout_targets = True
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.control_decimation = 5
        self.sim_time = 0.0
        self.viewer = viewer
        self.gui_velocity_command = np.zeros(3, dtype=np.float64)

        builder = build_asroballet_builder(model_path, virtual_ball_joint)
        _, self.elbow_ctrl_indices = configure_asroballet_downward_arm_pose(builder)
        wheel_q_indices, wheel_ctrl_indices = find_asroballet_wheel_indices(builder)
        base_bodies = [body for body, label in enumerate(builder.body_label) if label.endswith("/base_link")]
        if len(base_bodies) != 1:
            raise ValueError(f"Expected one asRoBallet base body, found {len(base_bodies)}.")
        self.base_body = base_bodies[0]

        self.model = builder.finalize()
        self.device = self.model.device
        self.wheel_q_indices = wp.array(wheel_q_indices, dtype=wp.int32, device=self.device)
        self.wheel_ctrl_indices = wp.array(wheel_ctrl_indices, dtype=wp.int32, device=self.device)
        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            iterations=100,
            ls_iterations=50,
            njmax=512,
            nconmax=1024,
            use_mujoco_cpu=True,
            use_mujoco_contacts=True,
            enable_multiccd=True,
        )
        self.solver.mj_model.opt.noslip_iterations = 1
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        initialize_asroballet_control(self.control, self.elbow_ctrl_indices)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.5, -2.5, 1.5), pitch=-12.0, yaw=135.0)

    def simulate(self):
        for _ in range(self.sim_substeps // self.control_decimation):
            self.update_control()
            for _ in range(self.control_decimation):
                self.state_0.clear_forces()
                self.viewer.apply_forces(self.state_0)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

    def update_control(self):
        raise NotImplementedError

    def after_simulate(self):
        pass

    def gui(self, ui):
        """Draw press-and-hold motion controls and keyboard help."""
        ui.text("Motion Controls")
        ui.text("Hold a button or its keyboard shortcut.")
        ui.text("Forward/back: I / K")
        ui.text("Left/right: J / L")
        ui.text("Turn left/right: U / O")
        ui.separator()

        interactive = getattr(self, "keyboard_control", False)
        ui.begin_disabled(not interactive)

        ui.button("Forward [I]##asroballet_forward")
        forward = ui.is_item_active()
        ui.same_line()
        ui.button("Backward [K]##asroballet_backward")
        backward = ui.is_item_active()

        ui.button("Left [J]##asroballet_left")
        left = ui.is_item_active()
        ui.same_line()
        ui.button("Right [L]##asroballet_right")
        right = ui.is_item_active()

        ui.button("Turn Left [U]##asroballet_turn_left")
        turn_left = ui.is_item_active()
        ui.same_line()
        ui.button("Turn Right [O]##asroballet_turn_right")
        turn_right = ui.is_item_active()

        ui.end_disabled()

        self.gui_velocity_command[:] = [
            KEYBOARD_LINEAR_SPEED * (forward - backward),
            KEYBOARD_LINEAR_SPEED * (left - right),
            KEYBOARD_YAW_RATE * (turn_left - turn_right),
        ]
        if not interactive:
            self.gui_velocity_command.fill(0.0)
            ui.text("Fixed CLI command active; interactive controls are disabled.")

        command = getattr(self, "velocity_command", np.zeros(3))
        ui.separator()
        ui.text(f"Command: vx={command[0]:+.2f}, vy={command[1]:+.2f} m/s")
        ui.text(f"Yaw rate: {command[2]:+.2f} rad/s")

    def step(self):
        self.update_command()
        self.simulate()
        self.after_simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


class _LqrController(_AsRoBalletController):
    def __init__(self, viewer, args):
        asset_root = download_asroballet_assets()
        super().__init__(
            viewer,
            resolve_asroballet_model_path(asset_root, args.virtual_ball_joint),
            args.virtual_ball_joint,
        )

        parameters = LqrModelParameters()
        control_cost = np.array([[20.0]])
        static_state_cost = np.diag([100.0, 2000.0, 10.0, 1000.0])
        tracking_state_cost = np.diag([100.0, 1.0e-6, 10.0, 1000.0])
        _, _, static_roll_gain = solve_discrete_lqr(
            *build_planar_state_space(parameters, "roll"),
            timestep=self.sim_dt * self.control_decimation,
            state_cost=static_state_cost,
            control_cost=control_cost,
        )
        _, _, static_pitch_gain = solve_discrete_lqr(
            *build_planar_state_space(parameters, "pitch"),
            timestep=self.sim_dt * self.control_decimation,
            state_cost=static_state_cost,
            control_cost=control_cost,
        )
        _, _, tracking_roll_gain = solve_discrete_lqr(
            *build_planar_state_space(parameters, "roll"),
            timestep=self.sim_dt * self.control_decimation,
            state_cost=tracking_state_cost,
            control_cost=control_cost,
        )
        _, _, tracking_pitch_gain = solve_discrete_lqr(
            *build_planar_state_space(parameters, "pitch"),
            timestep=self.sim_dt * self.control_decimation,
            state_cost=tracking_state_cost,
            control_cost=control_cost,
        )
        _, _, yaw_gain = solve_discrete_lqr(
            *build_yaw_state_space(parameters),
            timestep=self.sim_dt * self.control_decimation,
            state_cost=np.array([[8000.0]]),
            control_cost=control_cost,
        )
        wheel_to_ball, ball_to_control = build_wheel_control_matrix(sphere_radius=parameters.sphere_radius)
        self.static_roll_gain = wp.array(static_roll_gain.ravel(), dtype=wp.float32, device=self.device)
        self.static_pitch_gain = wp.array(static_pitch_gain.ravel(), dtype=wp.float32, device=self.device)
        self.tracking_roll_gain = wp.array(tracking_roll_gain.ravel(), dtype=wp.float32, device=self.device)
        self.tracking_pitch_gain = wp.array(tracking_pitch_gain.ravel(), dtype=wp.float32, device=self.device)
        self.yaw_gain = wp.array(yaw_gain.ravel(), dtype=wp.float32, device=self.device)
        self.ball_to_control = wp.array2d(ball_to_control, dtype=wp.float32, device=self.device)
        self.wheel_state_to_ball = wp.array2d(
            build_wheel_state_matrix(wheel_to_ball),
            dtype=wp.float32,
            device=self.device,
        )
        self.velocity_command = np.array([args.velocity_x, args.velocity_y, args.yaw_rate], dtype=np.float64)
        self.keyboard_control = should_use_keyboard_control(
            args.viewer,
            headless=args.headless,
            fixed_command=self.velocity_command,
        )
        self.track_planar_velocity = bool(np.any(self.velocity_command[:2]))
        self.roll_gain = self.tracking_roll_gain if self.track_planar_velocity else self.static_roll_gain
        self.pitch_gain = self.tracking_pitch_gain if self.track_planar_velocity else self.static_pitch_gain
        self.command = wp.array(
            [wp.vec3(*self.velocity_command)],
            dtype=wp.vec3,
            device=self.device,
        )
        self.sphere_radius = parameters.sphere_radius
        self.control_limit = 2.0

        self.initial_base_position = self.state_0.body_q.numpy()[self.base_body, :3].copy()
        self.position_reference = wp.array(
            [wp.vec3(*self.initial_base_position)],
            dtype=wp.vec3,
            device=self.device,
        )

    def update_control(self):
        launch_lqr_control(
            body_q=self.state_0.body_q,
            body_qd=self.state_0.body_qd,
            joint_q=self.state_0.joint_q,
            control=self.control.mujoco.ctrl,
            base_body=self.base_body,
            wheel_q_indices=self.wheel_q_indices,
            wheel_ctrl_indices=self.wheel_ctrl_indices,
            command=self.command,
            position_reference=self.position_reference,
            track_planar_velocity=self.track_planar_velocity,
            roll_gain=self.roll_gain,
            pitch_gain=self.pitch_gain,
            yaw_gain=self.yaw_gain,
            ball_to_control=self.ball_to_control,
            wheel_state_to_ball=self.wheel_state_to_ball,
            sphere_radius=self.sphere_radius,
            control_limit=self.control_limit,
            device=self.device,
        )

    def update_command(self):
        if not self.keyboard_control:
            return

        target_command = interactive_velocity_command(self.viewer, self.gui_velocity_command)
        max_delta = self.frame_dt * np.array(
            [KEYBOARD_LINEAR_ACCELERATION, KEYBOARD_LINEAR_ACCELERATION, KEYBOARD_YAW_ACCELERATION],
            dtype=np.float64,
        )
        was_tracking = self.track_planar_velocity
        self.velocity_command = move_toward(self.velocity_command, target_command, max_delta)
        self.track_planar_velocity = bool(np.any(target_command[:2]) or np.any(self.velocity_command[:2]))

        if was_tracking and not self.track_planar_velocity:
            wp.launch(
                _latch_position_reference_kernel,
                dim=1,
                inputs=[self.state_0.body_q, self.base_body, self.position_reference],
                device=self.device,
            )

        self.roll_gain = self.tracking_roll_gain if self.track_planar_velocity else self.static_roll_gain
        self.pitch_gain = self.tracking_pitch_gain if self.track_planar_velocity else self.static_pitch_gain
        self.command.assign([wp.vec3(*self.velocity_command)])

    def test_final(self):
        pose = self.state_0.body_q.numpy()[self.base_body]
        x, y, z, w = pose[3:7]
        rotation_20 = 2.0 * (x * z - w * y)
        rotation_21 = 2.0 * (y * z + w * x)
        rotation_22 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(rotation_21, rotation_22)
        pitch = math.atan2(-rotation_20, rotation_22)

        assert math.isfinite(roll) and math.isfinite(pitch), "Base orientation is non-finite."
        assert abs(roll) < 0.2 and abs(pitch) < 0.2, f"Base lost balance: roll={roll:.3f} rad, pitch={pitch:.3f} rad."
        if self.sim_time >= 3.0:
            rotation = np.array(
                [
                    [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                    [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
                    [rotation_20, rotation_21, rotation_22],
                ]
            )
            body_velocity = self.state_0.body_qd.numpy()[self.base_body]
            local_velocity = rotation.T @ body_velocity[:3]
            local_angular_velocity = rotation.T @ body_velocity[3:]

            if not self.track_planar_velocity:
                position_reference = self.position_reference.numpy()[0]
                planar_displacement = np.linalg.norm(pose[:2] - position_reference[:2])
                planar_speed = np.linalg.norm(local_velocity[:2])
                assert planar_displacement < 0.05, f"Static position error is too large: {planar_displacement:.3f} m."
                assert planar_speed < 0.05, f"Static planar speed is too large: {planar_speed:.3f} m/s."
            else:
                velocity_error = np.linalg.norm(local_velocity[:2] - self.velocity_command[:2])
                assert velocity_error < 0.08, f"Planar velocity tracking error is too large: {velocity_error:.3f} m/s."
            if self.velocity_command[2] != 0.0:
                yaw_rate_error = abs(local_angular_velocity[2] - self.velocity_command[2])
                assert yaw_rate_error < 0.15, f"Yaw-rate tracking error is too large: {yaw_rate_error:.3f} rad/s."


OBSERVATION_DIM = 16
STATION_OBSERVATION_DIM = 17
ACTION_DIM = 3
STATION_SWITCH_LINEAR_SPEED = 0.15
STATION_SWITCH_YAW_RATE = 0.1
STATION_SWITCH_SETTLE_TIME = 0.2


def load_policy_runtime(policy_path: str, device):
    """Load an exported policy through Warp-NN."""
    from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415

    return OnnxRuntime(policy_path, device=device)


@wp.kernel
def _build_policy_observation_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    base_body: int,
    ball_body: int,
    command: wp.array[wp.vec3],
    last_action: wp.array2d[float],
    body_count: int,
    observation: wp.array2d[float],
):
    base_pose = body_q[base_body]
    base_rotation = wp.transform_get_rotation(base_pose)
    base_twist = body_qd[base_body]
    angular_velocity_world = wp.spatial_bottom(base_twist)

    # Newton stores body_qd at the COM, while the MuJoCo velocimeter used
    # during training is attached to the base body origin.
    base_com_offset_world = wp.transform_vector(base_pose, body_com[base_body])
    origin_velocity_world = wp.spatial_top(base_twist) - wp.cross(
        angular_velocity_world,
        base_com_offset_world,
    )
    local_velocity = wp.quat_rotate_inv(base_rotation, origin_velocity_world)
    local_angular_velocity = wp.quat_rotate_inv(base_rotation, angular_velocity_world)

    x = base_rotation[0]
    y = base_rotation[1]
    z = base_rotation[2]
    w = base_rotation[3]
    roll = wp.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = wp.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = wp.asin(sin_pitch)
    total_mass = float(0.0)
    subtree_com_world = wp.vec3()
    for body in range(body_count):
        mass = body_mass[body]
        subtree_com_world += mass * wp.transform_point(body_q[body], body_com[body])
        total_mass += mass
    subtree_com_world /= total_mass

    ball_com_world = wp.transform_point(body_q[ball_body], body_com[ball_body])
    local_com_proxy = 10.0 * wp.quat_rotate_inv(
        base_rotation,
        subtree_com_world - ball_com_world,
    )
    desired_velocity = command[0]

    observation[0, 0] = local_velocity[0]
    observation[0, 1] = local_velocity[1]
    observation[0, 2] = desired_velocity[0]
    observation[0, 3] = desired_velocity[1]
    observation[0, 4] = desired_velocity[2]
    observation[0, 5] = roll
    observation[0, 6] = pitch
    observation[0, 7] = local_com_proxy[0]
    observation[0, 8] = local_com_proxy[1]
    observation[0, 9] = local_com_proxy[2]
    observation[0, 10] = local_angular_velocity[0]
    observation[0, 11] = local_angular_velocity[1]
    observation[0, 12] = local_angular_velocity[2]
    observation[0, 13] = last_action[0, 0]
    observation[0, 14] = last_action[0, 1]
    observation[0, 15] = last_action[0, 2]


@wp.kernel
def _build_station_policy_observation_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    base_body: int,
    ball_body: int,
    station_site_transform: wp.transform,
    reference_position_xy: wp.array[wp.vec2],
    reference_yaw: wp.array[float],
    last_action: wp.array2d[float],
    body_count: int,
    observation: wp.array2d[float],
):
    base_pose = body_q[base_body]
    base_rotation = wp.transform_get_rotation(base_pose)
    base_twist = body_qd[base_body]
    angular_velocity_world = wp.spatial_bottom(base_twist)

    base_com_offset_world = wp.transform_vector(base_pose, body_com[base_body])
    origin_velocity_world = wp.spatial_top(base_twist) - wp.cross(
        angular_velocity_world,
        base_com_offset_world,
    )
    local_velocity = wp.quat_rotate_inv(base_rotation, origin_velocity_world)
    local_angular_velocity = wp.quat_rotate_inv(base_rotation, angular_velocity_world)

    x = base_rotation[0]
    y = base_rotation[1]
    z = base_rotation[2]
    w = base_rotation[3]
    roll = wp.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = wp.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = wp.asin(sin_pitch)
    yaw = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    total_mass = float(0.0)
    subtree_com_world = wp.vec3()
    for body in range(body_count):
        mass = body_mass[body]
        subtree_com_world += mass * wp.transform_point(body_q[body], body_com[body])
        total_mass += mass
    subtree_com_world /= total_mass

    ball_com_world = wp.transform_point(body_q[ball_body], body_com[ball_body])
    local_com_proxy = 10.0 * wp.quat_rotate_inv(
        base_rotation,
        subtree_com_world - ball_com_world,
    )

    station_site_pose = wp.transform_multiply(base_pose, station_site_transform)
    station_position = wp.transform_get_translation(station_site_pose)
    position_error_world = wp.vec2(
        station_position[0] - reference_position_xy[0][0],
        station_position[1] - reference_position_xy[0][1],
    )
    cos_reference = wp.cos(reference_yaw[0])
    sin_reference = wp.sin(reference_yaw[0])
    relative_position = wp.vec2(
        cos_reference * position_error_world[0] + sin_reference * position_error_world[1],
        -sin_reference * position_error_world[0] + cos_reference * position_error_world[1],
    )
    relative_yaw = yaw - reference_yaw[0]

    observation[0, 0] = local_velocity[0]
    observation[0, 1] = local_velocity[1]
    observation[0, 2] = relative_position[0]
    observation[0, 3] = relative_position[1]
    observation[0, 4] = roll
    observation[0, 5] = pitch
    observation[0, 6] = wp.sin(relative_yaw)
    observation[0, 7] = wp.cos(relative_yaw)
    observation[0, 8] = local_com_proxy[0]
    observation[0, 9] = local_com_proxy[1]
    observation[0, 10] = local_com_proxy[2]
    observation[0, 11] = local_angular_velocity[0]
    observation[0, 12] = local_angular_velocity[1]
    observation[0, 13] = local_angular_velocity[2]
    observation[0, 14] = last_action[0, 0]
    observation[0, 15] = last_action[0, 1]
    observation[0, 16] = last_action[0, 2]


@wp.kernel
def _latch_station_reference_kernel(
    body_q: wp.array[wp.transform],
    base_body: int,
    station_site_transform: wp.transform,
    reference_position_xy: wp.array[wp.vec2],
    reference_yaw: wp.array[float],
):
    base_pose = body_q[base_body]
    station_site_pose = wp.transform_multiply(base_pose, station_site_transform)
    station_position = wp.transform_get_translation(station_site_pose)
    base_rotation = wp.transform_get_rotation(base_pose)
    x = base_rotation[0]
    y = base_rotation[1]
    z = base_rotation[2]
    w = base_rotation[3]

    reference_position_xy[0] = wp.vec2(station_position[0], station_position[1])
    reference_yaw[0] = wp.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@wp.kernel
def _measure_station_switch_motion_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base_body: int,
    motion: wp.array[wp.vec2],
):
    base_pose = body_q[base_body]
    base_rotation = wp.transform_get_rotation(base_pose)
    base_twist = body_qd[base_body]
    angular_velocity_world = wp.spatial_bottom(base_twist)
    base_com_offset_world = wp.transform_vector(base_pose, body_com[base_body])
    origin_velocity_world = wp.spatial_top(base_twist) - wp.cross(
        angular_velocity_world,
        base_com_offset_world,
    )
    local_velocity = wp.quat_rotate_inv(base_rotation, origin_velocity_world)
    local_angular_velocity = wp.quat_rotate_inv(base_rotation, angular_velocity_world)
    motion[0] = wp.vec2(
        wp.length(wp.vec2(local_velocity[0], local_velocity[1])),
        wp.abs(local_angular_velocity[2]),
    )


@wp.kernel
def _apply_policy_action_kernel(
    action: wp.array2d[float],
    wheel_ctrl_indices: wp.array[int],
    control: wp.array[float],
    last_action: wp.array2d[float],
):
    wheel = wp.tid()
    value = wp.clamp(action[0, wheel], -1.0, 1.0)
    last_action[0, wheel] = value
    control[wheel_ctrl_indices[wheel]] = value


def launch_policy_observation(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    base_body: int,
    ball_body: int,
    command: wp.array[wp.vec3],
    last_action: wp.array2d[float],
    observation: wp.array2d[float],
    device,
) -> None:
    """Build one observation with the layout used during policy training."""
    wp.launch(
        _build_policy_observation_kernel,
        dim=1,
        inputs=[
            body_q,
            body_qd,
            body_com,
            body_mass,
            base_body,
            ball_body,
            command,
            last_action,
            body_mass.shape[0],
            observation,
        ],
        device=device,
    )


def launch_station_policy_observation(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    base_body: int,
    ball_body: int,
    station_site_transform: wp.transform,
    reference_position_xy: wp.array[wp.vec2],
    reference_yaw: wp.array[float],
    last_action: wp.array2d[float],
    observation: wp.array2d[float],
    device,
) -> None:
    """Build one station observation relative to the latched release pose."""
    wp.launch(
        _build_station_policy_observation_kernel,
        dim=1,
        inputs=[
            body_q,
            body_qd,
            body_com,
            body_mass,
            base_body,
            ball_body,
            station_site_transform,
            reference_position_xy,
            reference_yaw,
            last_action,
            body_mass.shape[0],
            observation,
        ],
        device=device,
    )


def latch_station_reference(
    body_q: wp.array[wp.transform],
    base_body: int,
    station_site_transform: wp.transform,
    reference_position_xy: wp.array[wp.vec2],
    reference_yaw: wp.array[float],
    device,
) -> None:
    """Latch the current station sensor position and base yaw."""
    wp.launch(
        _latch_station_reference_kernel,
        dim=1,
        inputs=[
            body_q,
            base_body,
            station_site_transform,
            reference_position_xy,
            reference_yaw,
        ],
        device=device,
    )


def measure_station_switch_motion(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    base_body: int,
    motion: wp.array[wp.vec2],
    device,
) -> np.ndarray:
    """Measure planar and yaw speed for the station transition."""
    wp.launch(
        _measure_station_switch_motion_kernel,
        dim=1,
        inputs=[body_q, body_qd, body_com, base_body, motion],
        device=device,
    )
    return motion.numpy()[0]


def launch_policy_action(
    action: wp.array2d[float],
    wheel_ctrl_indices: wp.array[int],
    control: wp.array[float],
    last_action: wp.array2d[float],
    device,
) -> None:
    """Apply clipped wheel actions and retain them for the next observation."""
    wp.launch(
        _apply_policy_action_kernel,
        dim=ACTION_DIM,
        inputs=[action, wheel_ctrl_indices, control, last_action],
        device=device,
    )


class _PolicyController(_AsRoBalletController):
    def __init__(self, viewer, args):
        asset_root = download_asroballet_assets()
        model_path = resolve_asroballet_model_path(asset_root, args.virtual_ball_joint)
        policy_path = str(asset_root / ASROBALLET_VELOCITY_POLICY_PATH)
        station_policy_path = str(asset_root / ASROBALLET_STATION_POLICY_PATH)
        super().__init__(viewer, model_path, args.virtual_ball_joint)

        ball_bodies = [body for body, label in enumerate(self.model.body_label) if label.endswith("/ball_link")]
        if len(ball_bodies) != 1:
            raise ValueError(f"Expected one asRoBallet ball body, found {len(ball_bodies)}.")
        self.ball_body = ball_bodies[0]
        shape_body = self.model.shape_body.numpy()
        shape_flags = self.model.shape_flags.numpy()
        station_sites = [
            shape
            for shape, label in enumerate(self.model.shape_label)
            if label.endswith("/slam_site")
            and shape_body[shape] == self.base_body
            and shape_flags[shape] & int(newton.ShapeFlags.SITE)
        ]
        if len(station_sites) != 1:
            raise ValueError(f"Expected one asRoBallet slam_site, found {len(station_sites)}.")
        station_site_transform = self.model.shape_transform.numpy()[station_sites[0]]
        self.station_site_transform = wp.transform(
            wp.vec3(*station_site_transform[:3]),
            wp.quat(*station_site_transform[3:]),
        )

        self.policy = load_policy_runtime(policy_path, device=self.device)
        self.policy_input_name = self.policy.input_names[0]
        self.policy_output_name = self.policy.output_names[0]
        validate_policy_io_shapes(
            policy_path,
            self.policy_input_name,
            self.policy_output_name,
            obs_width=OBSERVATION_DIM,
            action_width=ACTION_DIM,
            context="asRoBallet policy",
        )
        self.station_policy = load_policy_runtime(station_policy_path, device=self.device)
        self.station_policy_input_name = self.station_policy.input_names[0]
        self.station_policy_output_name = self.station_policy.output_names[0]
        validate_policy_io_shapes(
            station_policy_path,
            self.station_policy_input_name,
            self.station_policy_output_name,
            obs_width=STATION_OBSERVATION_DIM,
            action_width=ACTION_DIM,
            context="asRoBallet station policy",
        )
        self.observation = wp.zeros((1, OBSERVATION_DIM), dtype=wp.float32, device=self.device)
        self.station_observation = wp.zeros(
            (1, STATION_OBSERVATION_DIM),
            dtype=wp.float32,
            device=self.device,
        )
        self.last_action = wp.zeros((1, ACTION_DIM), dtype=wp.float32, device=self.device)
        self.station_reference_position_xy = wp.zeros(1, dtype=wp.vec2, device=self.device)
        self.station_reference_yaw = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.station_switch_motion = wp.zeros(1, dtype=wp.vec2, device=self.device)
        self.station_switch_settle_frames = max(1, math.ceil(STATION_SWITCH_SETTLE_TIME / self.frame_dt))
        self.station_switch_settle_count = 0
        latch_station_reference(
            body_q=self.state_0.body_q,
            base_body=self.base_body,
            station_site_transform=self.station_site_transform,
            reference_position_xy=self.station_reference_position_xy,
            reference_yaw=self.station_reference_yaw,
            device=self.device,
        )

        self.velocity_command = np.array([args.velocity_x, args.velocity_y, args.yaw_rate], dtype=np.float64)
        self.station_mode = not np.any(self.velocity_command)
        self.braking_mode = False
        self.keyboard_control = should_use_keyboard_control(
            args.viewer,
            headless=args.headless,
            fixed_command=self.velocity_command,
        )
        self.command = wp.array(
            [wp.vec3(*self.velocity_command)],
            dtype=wp.vec3,
            device=self.device,
        )

    def update_command(self):
        if not self.keyboard_control:
            return

        self.velocity_command = interactive_velocity_command(self.viewer, self.gui_velocity_command)
        self.command.assign([wp.vec3(*self.velocity_command)])
        if np.any(self.velocity_command):
            self.station_mode = False
            self.braking_mode = False
            self.station_switch_settle_count = 0
        elif not self.station_mode:
            self.braking_mode = True

    def update_station_mode(self):
        if not self.braking_mode:
            return

        planar_speed, yaw_rate = measure_station_switch_motion(
            body_q=self.state_0.body_q,
            body_qd=self.state_0.body_qd,
            body_com=self.model.body_com,
            base_body=self.base_body,
            motion=self.station_switch_motion,
            device=self.device,
        )
        if planar_speed < STATION_SWITCH_LINEAR_SPEED and yaw_rate < STATION_SWITCH_YAW_RATE:
            self.station_switch_settle_count += 1
        else:
            self.station_switch_settle_count = 0

        if self.station_switch_settle_count < self.station_switch_settle_frames:
            return

        latch_station_reference(
            body_q=self.state_0.body_q,
            base_body=self.base_body,
            station_site_transform=self.station_site_transform,
            reference_position_xy=self.station_reference_position_xy,
            reference_yaw=self.station_reference_yaw,
            device=self.device,
        )
        self.station_mode = True
        self.braking_mode = False
        self.station_switch_settle_count = 0

    def update_control(self):
        if self.station_mode:
            launch_station_policy_observation(
                body_q=self.state_0.body_q,
                body_qd=self.state_0.body_qd,
                body_com=self.model.body_com,
                body_mass=self.model.body_mass,
                base_body=self.base_body,
                ball_body=self.ball_body,
                station_site_transform=self.station_site_transform,
                reference_position_xy=self.station_reference_position_xy,
                reference_yaw=self.station_reference_yaw,
                last_action=self.last_action,
                observation=self.station_observation,
                device=self.device,
            )
            output = self.station_policy({self.station_policy_input_name: self.station_observation})
            action = output[self.station_policy_output_name]
        else:
            launch_policy_observation(
                body_q=self.state_0.body_q,
                body_qd=self.state_0.body_qd,
                body_com=self.model.body_com,
                body_mass=self.model.body_mass,
                base_body=self.base_body,
                ball_body=self.ball_body,
                command=self.command,
                last_action=self.last_action,
                observation=self.observation,
                device=self.device,
            )
            output = self.policy({self.policy_input_name: self.observation})
            action = output[self.policy_output_name]
        launch_policy_action(
            action=action,
            wheel_ctrl_indices=self.wheel_ctrl_indices,
            control=self.control.mujoco.ctrl,
            last_action=self.last_action,
            device=self.device,
        )

    def after_simulate(self):
        self.update_station_mode()

    def test_final(self):
        """Verify policy balance and applied-action feedback."""
        pose = self.state_0.body_q.numpy()[self.base_body]
        x, y, z, w = pose[3:7]
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = math.asin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))

        assert math.isfinite(roll) and math.isfinite(pitch), "Base orientation is non-finite."
        assert abs(roll) < 0.35 and abs(pitch) < 0.35, f"Base lost balance: roll={roll:.3f} rad, pitch={pitch:.3f} rad."
        wheel_ctrl_indices = self.wheel_ctrl_indices.numpy()
        applied_action = self.control.mujoco.ctrl.numpy()[wheel_ctrl_indices]
        np.testing.assert_allclose(self.last_action.numpy()[0], applied_action)


class Example:
    def __init__(self, viewer, args):
        controller_type = _PolicyController if args.controller == "policy" else _LqrController
        self._controller = controller_type(viewer, args)

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def step(self):
        self._controller.step()

    def render(self):
        self._controller.render()

    def gui(self, ui):
        self._controller.gui(ui)

    def test_final(self):
        self._controller.test_final()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--controller",
            choices=("policy", "lqr"),
            default="policy",
            help="Controller to run (default: policy).",
        )
        parser.add_argument(
            "--virtual-ball-joint",
            action="store_true",
            help="Constrain the ball center to the base with the original spherical joint.",
        )
        parser.add_argument("--velocity-x", type=float, default=0.0, help="Commanded forward velocity [m/s].")
        parser.add_argument("--velocity-y", type=float, default=0.0, help="Commanded lateral velocity [m/s].")
        parser.add_argument("--yaw-rate", type=float, default=0.0, help="Commanded yaw rate [rad/s].")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
