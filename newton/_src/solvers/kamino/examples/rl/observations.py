# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Python
from abc import ABC, abstractmethod

# Thirdparty
import torch  # noqa: TID253
import warp as wp

from newton._src.solvers.kamino._src.utils.sim import Simulator
from newton._src.solvers.kamino.examples.rl.simulation import RigidBodySim
from newton._src.solvers.kamino.examples.rl.utils import StackedIndices, periodic_encoding

# ---------------------------------------------------------------------------
# Warp helpers for BipedalObservation
# ---------------------------------------------------------------------------

_Z_AXIS = wp.constant(wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _projected_yaw(q: wp.quat) -> float:
    """Extract the yaw angle from a quaternion (no warp builtin equivalent)."""
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    return wp.atan2(2.0 * (qz * qw + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)


@wp.func
def _write_vec3(obs: wp.array[wp.float32], idx: int, v: wp.vec3):
    obs[idx + 0] = v[0]
    obs[idx + 1] = v[1]
    obs[idx + 2] = v[2]


# Observation group indices into the offsets array (must match _OBS_NAMES order).
_OBS_ORI = wp.constant(0)
_OBS_PATH_DEV = wp.constant(1)
_OBS_PATH_DEV_H = wp.constant(2)
_OBS_PATH_CMD = wp.constant(3)
_OBS_PATH_LIN_VEL = wp.constant(4)
_OBS_PATH_ANG_VEL = wp.constant(5)
_OBS_PHASE_ENC = wp.constant(6)
_OBS_NECK = wp.constant(7)
_OBS_ROOT_LIN_VEL = wp.constant(8)
_OBS_ROOT_ANG_VEL = wp.constant(9)
_OBS_JOINT_POS = wp.constant(10)
_OBS_JOINT_VEL = wp.constant(11)

# Python-side list matching the constant order above.
_OBS_NAMES = [
    "orientation_root_to_path",
    "path_deviation",
    "path_deviation_in_heading",
    "path_cmd",
    "path_lin_vel_in_root",
    "path_ang_vel_in_root",
    "phase_encoding",
    "neck_cmd",
    "root_lin_vel_in_root",
    "root_ang_vel_in_root",
    "normalized_joint_positions",
    "normalized_joint_velocities",
]


@wp.kernel
def _compute_bipedal_obs_core(
    obs: wp.array[wp.float32],
    q_i: wp.array[wp.float32],
    u_i: wp.array[wp.float32],
    q_j: wp.array[wp.float32],
    dq_j: wp.array[wp.float32],
    command: wp.array[wp.float32],
    phase: wp.array[wp.float32],
    freq_2pi: wp.array[wp.float32],
    offset_enc: wp.array[wp.float32],
    joint_default: wp.array[wp.float32],
    joint_range: wp.array[wp.float32],
    obs_offsets: wp.array[wp.int32],
    num_bodies: int,
    num_joint_coords: int,
    num_joint_dofs: int,
    num_obs: int,
    cmd_dim: int,
    inv_path_dev_scale: float,
    inv_joint_vel_scale: float,
    phase_enc_dim: int,
    num_joints: int,
):
    w = wp.tid()

    # Flat array strides
    qi_base = w * num_bodies * 7
    ui_base = w * num_bodies * 6
    qj_base = w * num_joint_coords
    dqj_base = w * num_joint_dofs
    cmd_base = w * cmd_dim
    o = w * num_obs

    # Root pose & velocities (body 0)
    root_quat = wp.quat(q_i[qi_base + 3], q_i[qi_base + 4], q_i[qi_base + 5], q_i[qi_base + 6])
    root_lin_vel = wp.vec3(u_i[ui_base + 0], u_i[ui_base + 1], u_i[ui_base + 2])
    root_ang_vel = wp.vec3(u_i[ui_base + 3], u_i[ui_base + 4], u_i[ui_base + 5])

    # Commands
    path_heading = command[cmd_base + 0]
    cmd_vel = wp.vec3(command[cmd_base + 3], command[cmd_base + 4], 0.0)
    cmd_yaw_rate = command[cmd_base + 5]

    # root_orientation_in_path = inv(path_quat) * root_quat
    path_quat = wp.quat_from_axis_angle(_Z_AXIS, path_heading)
    root_in_path = wp.mul(wp.quat_inverse(path_quat), root_quat)

    # Orientation as flattened 3x3 rotation matrix
    rot = wp.quat_to_matrix(root_in_path)
    oi = o + obs_offsets[_OBS_ORI]
    for i in range(3):
        for j in range(3):
            obs[oi + i * 3 + j] = rot[i, j]

    # Path deviation in path frame (scaled)
    diff = wp.vec3(q_i[qi_base + 0] - command[cmd_base + 1], q_i[qi_base + 1] - command[cmd_base + 2], 0.0)
    rtp = wp.quat_rotate_inv(path_quat, diff)
    obs[o + obs_offsets[_OBS_PATH_DEV] + 0] = rtp[0] * inv_path_dev_scale
    obs[o + obs_offsets[_OBS_PATH_DEV] + 1] = rtp[1] * inv_path_dev_scale

    # Path deviation in heading frame (scaled)
    root_heading = _projected_yaw(root_in_path)
    heading_quat = wp.quat_from_axis_angle(_Z_AXIS, root_heading)
    pth = wp.quat_rotate_inv(heading_quat, wp.vec3(-rtp[0], -rtp[1], 0.0))
    obs[o + obs_offsets[_OBS_PATH_DEV_H] + 0] = pth[0] * inv_path_dev_scale
    obs[o + obs_offsets[_OBS_PATH_DEV_H] + 1] = pth[1] * inv_path_dev_scale

    # Path command
    _write_vec3(obs, o + obs_offsets[_OBS_PATH_CMD], wp.vec3(cmd_vel[0], cmd_vel[1], cmd_yaw_rate))

    # Path velocities in root frame
    _write_vec3(obs, o + obs_offsets[_OBS_PATH_LIN_VEL], wp.quat_rotate_inv(root_in_path, cmd_vel))
    _write_vec3(
        obs, o + obs_offsets[_OBS_PATH_ANG_VEL], wp.quat_rotate_inv(root_in_path, wp.vec3(0.0, 0.0, cmd_yaw_rate))
    )

    # Phase encoding
    p = phase[w]
    op = o + obs_offsets[_OBS_PHASE_ENC]
    for i in range(phase_enc_dim):
        obs[op + i] = wp.sin(p * freq_2pi[i] + offset_enc[i])

    # Neck command
    on = o + obs_offsets[_OBS_NECK]
    for i in range(4):
        obs[on + i] = command[cmd_base + 6 + i]

    # Root velocities in root frame
    _write_vec3(obs, o + obs_offsets[_OBS_ROOT_LIN_VEL], wp.quat_rotate_inv(root_quat, root_lin_vel))
    _write_vec3(obs, o + obs_offsets[_OBS_ROOT_ANG_VEL], wp.quat_rotate_inv(root_quat, root_ang_vel))

    # Normalized joint positions & velocities
    ojp = o + obs_offsets[_OBS_JOINT_POS]
    ojv = o + obs_offsets[_OBS_JOINT_VEL]
    for j in range(num_joints):
        obs[ojp + j] = (q_j[qj_base + j] - joint_default[j]) / joint_range[j]
        obs[ojv + j] = dq_j[dqj_base + j] * inv_joint_vel_scale


class PhaseRate(torch.nn.Module):
    """
    Defines the mapping between robot measurements and a pretrained phase rate.
    """

    def __init__(self, path, obs_cmd_range) -> None:
        super().__init__()
        self.obs_cmd_idx = list(obs_cmd_range)

        # Load pre-trained model
        model = torch.load(path, weights_only=False)
        model.eval()

        # Turn off gradients of the pretrained parameters
        for param in model.parameters():
            param.requires_grad_(False)

        super().add_module("_phase_rate", model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        path_cmd = input[:, self.obs_cmd_idx]
        return self._phase_rate(path_cmd)


# ---------------------------------------------------------------------------
# Warp helpers for BipedalObservation
# ---------------------------------------------------------------------------

_Z_AXIS = wp.constant(wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _projected_yaw(q: wp.quat) -> float:
    """Extract the yaw angle from a quaternion (no warp builtin equivalent)."""
    qx = q[0]
    qy = q[1]
    qz = q[2]
    qw = q[3]
    return wp.atan2(2.0 * (qz * qw + qx * qy), qw * qw + qx * qx - qy * qy - qz * qz)


@wp.func
def _write_vec3(obs: wp.array[wp.float32], idx: int, v: wp.vec3):
    obs[idx + 0] = v[0]
    obs[idx + 1] = v[1]
    obs[idx + 2] = v[2]


# Observation group indices into the offsets array (must match _OBS_NAMES order).
_OBS_ORI = wp.constant(0)
_OBS_PATH_DEV = wp.constant(1)
_OBS_PATH_DEV_H = wp.constant(2)
_OBS_PATH_CMD = wp.constant(3)
_OBS_PATH_LIN_VEL = wp.constant(4)
_OBS_PATH_ANG_VEL = wp.constant(5)
_OBS_PHASE_ENC = wp.constant(6)
_OBS_NECK = wp.constant(7)
_OBS_ROOT_LIN_VEL = wp.constant(8)
_OBS_ROOT_ANG_VEL = wp.constant(9)
_OBS_JOINT_POS = wp.constant(10)
_OBS_JOINT_VEL = wp.constant(11)

# Python-side list matching the constant order above.
_OBS_NAMES = [
    "orientation_root_to_path",
    "path_deviation",
    "path_deviation_in_heading",
    "path_cmd",
    "path_lin_vel_in_root",
    "path_ang_vel_in_root",
    "phase_encoding",
    "neck_cmd",
    "root_lin_vel_in_root",
    "root_ang_vel_in_root",
    "normalized_joint_positions",
    "normalized_joint_velocities",
]


@wp.kernel
def _compute_bipedal_obs_core(
    obs: wp.array[wp.float32],
    q_i: wp.array[wp.float32],
    u_i: wp.array[wp.float32],
    q_j: wp.array[wp.float32],
    dq_j: wp.array[wp.float32],
    command: wp.array[wp.float32],
    phase: wp.array[wp.float32],
    freq_2pi: wp.array[wp.float32],
    offset_enc: wp.array[wp.float32],
    joint_default: wp.array[wp.float32],
    joint_range: wp.array[wp.float32],
    obs_offsets: wp.array[wp.int32],
    num_bodies: int,
    num_joint_coords: int,
    num_joint_dofs: int,
    num_obs: int,
    cmd_dim: int,
    inv_path_dev_scale: float,
    inv_joint_vel_scale: float,
    phase_enc_dim: int,
    num_joints: int,
):
    w = wp.tid()

    # Flat array strides
    qi_base = w * num_bodies * 7
    ui_base = w * num_bodies * 6
    qj_base = w * num_joint_coords
    dqj_base = w * num_joint_dofs
    cmd_base = w * cmd_dim
    o = w * num_obs

    # Root pose & velocities (body 0)
    root_quat = wp.quat(q_i[qi_base + 3], q_i[qi_base + 4], q_i[qi_base + 5], q_i[qi_base + 6])
    root_lin_vel = wp.vec3(u_i[ui_base + 0], u_i[ui_base + 1], u_i[ui_base + 2])
    root_ang_vel = wp.vec3(u_i[ui_base + 3], u_i[ui_base + 4], u_i[ui_base + 5])

    # Commands
    path_heading = command[cmd_base + 0]
    cmd_vel = wp.vec3(command[cmd_base + 3], command[cmd_base + 4], 0.0)
    cmd_yaw_rate = command[cmd_base + 5]

    # root_orientation_in_path = inv(path_quat) * root_quat
    path_quat = wp.quat_from_axis_angle(_Z_AXIS, path_heading)
    root_in_path = wp.mul(wp.quat_inverse(path_quat), root_quat)

    # Orientation as flattened 3x3 rotation matrix
    rot = wp.quat_to_matrix(root_in_path)
    oi = o + obs_offsets[_OBS_ORI]
    for i in range(3):
        for j in range(3):
            obs[oi + i * 3 + j] = rot[i, j]

    # Path deviation in path frame (scaled)
    diff = wp.vec3(q_i[qi_base + 0] - command[cmd_base + 1], q_i[qi_base + 1] - command[cmd_base + 2], 0.0)
    rtp = wp.quat_rotate_inv(path_quat, diff)
    obs[o + obs_offsets[_OBS_PATH_DEV] + 0] = rtp[0] * inv_path_dev_scale
    obs[o + obs_offsets[_OBS_PATH_DEV] + 1] = rtp[1] * inv_path_dev_scale

    # Path deviation in heading frame (scaled)
    root_heading = _projected_yaw(root_in_path)
    heading_quat = wp.quat_from_axis_angle(_Z_AXIS, root_heading)
    pth = wp.quat_rotate_inv(heading_quat, wp.vec3(-rtp[0], -rtp[1], 0.0))
    obs[o + obs_offsets[_OBS_PATH_DEV_H] + 0] = pth[0] * inv_path_dev_scale
    obs[o + obs_offsets[_OBS_PATH_DEV_H] + 1] = pth[1] * inv_path_dev_scale

    # Path command
    _write_vec3(obs, o + obs_offsets[_OBS_PATH_CMD], wp.vec3(cmd_vel[0], cmd_vel[1], cmd_yaw_rate))

    # Path velocities in root frame
    _write_vec3(obs, o + obs_offsets[_OBS_PATH_LIN_VEL], wp.quat_rotate_inv(root_in_path, cmd_vel))
    _write_vec3(
        obs, o + obs_offsets[_OBS_PATH_ANG_VEL], wp.quat_rotate_inv(root_in_path, wp.vec3(0.0, 0.0, cmd_yaw_rate))
    )

    # Phase encoding
    p = phase[w]
    op = o + obs_offsets[_OBS_PHASE_ENC]
    for i in range(phase_enc_dim):
        obs[op + i] = wp.sin(p * freq_2pi[i] + offset_enc[i])

    # Neck command
    on = o + obs_offsets[_OBS_NECK]
    for i in range(4):
        obs[on + i] = command[cmd_base + 6 + i]

    # Root velocities in root frame
    _write_vec3(obs, o + obs_offsets[_OBS_ROOT_LIN_VEL], wp.quat_rotate_inv(root_quat, root_lin_vel))
    _write_vec3(obs, o + obs_offsets[_OBS_ROOT_ANG_VEL], wp.quat_rotate_inv(root_quat, root_ang_vel))

    # Normalized joint positions & velocities
    ojp = o + obs_offsets[_OBS_JOINT_POS]
    ojv = o + obs_offsets[_OBS_JOINT_VEL]
    for j in range(num_joints):
        obs[ojp + j] = (q_j[qj_base + j] - joint_default[j]) / joint_range[j]
        obs[ojv + j] = dq_j[dqj_base + j] * inv_joint_vel_scale


class PhaseRate(torch.nn.Module):
    """
    Defines the mapping between robot measurements and a pretrained phase rate.
    """

    def __init__(self, path, obs_cmd_range) -> None:
        super().__init__()
        self.obs_cmd_idx = list(obs_cmd_range)

        # Load pre-trained model
        model = torch.load(path, weights_only=False)
        model.eval()

        # Turn off gradients of the pretrained parameters
        for param in model.parameters():
            param.requires_grad_(False)

        super().add_module("_phase_rate", model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        path_cmd = input[:, self.obs_cmd_idx]
        return self._phase_rate(path_cmd)


class ObservationBuilder(ABC):
    """Base class for building observation tensors from a Kamino Simulator.

    Subclasses define which signals to extract and concatenate.  The builder
    maintains internal buffers (e.g. action history) and provides a uniform
    ``compute()`` interface suitable for inference loops.

    Args:
        sim: A Kamino ``Simulator`` instance.
        num_worlds: Number of parallel simulation worlds.
        device: Torch device string (e.g. ``"cuda:0"``).
        command_dim: Dimensionality of the external command vector
            (0 means no command input; todo reserved for future joystick / keyboard velocity commands).
    """

    def __init__(
        self,
        sim: Simulator,
        num_worlds: int,
        device: str,
        command_dim: int = 0,
    ) -> None:
        self._sim = sim
        self._num_worlds = num_worlds
        self._device = device
        self._command_dim = command_dim

        # Pre-allocated command tensor (empty if command_dim == 0).
        self._command: torch.Tensor = torch.zeros(
            (num_worlds, max(command_dim, 0)),
            device=device,
            dtype=torch.float32,
        )

    @property
    @abstractmethod
    def num_observations(self) -> int:
        """Total observation dimensionality (per environment)."""
        ...

    @property
    def command_dim(self) -> int:
        """Dimensionality of the external command vector."""
        return self._command_dim

    @property
    def command(self) -> torch.Tensor:
        """Current command tensor, shape ``(num_worlds, command_dim)``.

        External code (keyboard handler, joystick) writes into this tensor
        so that ``compute()`` can include it in the observation.
        """
        return self._command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self._command = value

    @abstractmethod
    def compute(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        """Build the observation tensor from the current simulator state.

        Args:
            actions: The most recent actions applied, shape
                ``(num_worlds, num_actions)``.  Pass ``None`` on the very
                first step (before any action has been applied).

        Returns:
            Observation tensor of shape ``(num_worlds, num_observations)``.
        """
        ...

    @abstractmethod
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset internal buffers (e.g. action history) for given envs.

        Args:
            env_ids: Which environments to reset.  ``None`` resets all.
        """
        ...

    def _get_joint_positions(self) -> torch.Tensor:
        """Joint positions as a PyTorch tensor ``(num_worlds, num_joint_coords)``.

        Zero-copy view of the underlying Warp array.
        """
        num_joint_coords = self._sim.model.size.max_of_num_joint_coords
        return wp.to_torch(self._sim.state.q_j).reshape(self._num_worlds, num_joint_coords)

    def _get_joint_velocities(self) -> torch.Tensor:
        """Joint velocities as a PyTorch tensor ``(num_worlds, num_joint_dofs)``.

        Zero-copy view of the underlying Warp array.
        """
        num_joint_dofs = self._sim.model.size.max_of_num_joint_dofs
        return wp.to_torch(self._sim.state.dq_j).reshape(self._num_worlds, num_joint_dofs)

    def _get_root_positions(self) -> torch.Tensor:
        """Root body positions as a PyTorch tensor ``(num_worlds, 3)``.

        Zero-copy view of the underlying Warp array (body 0, xyz).
        """
        num_bodies = self._sim.model.size.max_of_num_bodies
        q_i = wp.to_torch(self._sim.state.q_i).reshape(self._num_worlds, num_bodies, 7)
        return q_i[:, 0, :3]


class DrlegsBaseObservation(ObservationBuilder):
    """Base observation builder for DR Legs.

    Observation vector (63D):
        * root position        (3D  — pelvis xyz)
        * DOF positions        (36D — all joints, including passive linkages)
        * action history t-0   (12D — actuated joints, current step)
        * action history t-1   (12D — actuated joints, previous step)

    Args:
        body_sim: A :class:`RigidBodySim` instance.
        action_scale: Scale applied to raw actions before storing in history.
    """

    def __init__(
        self,
        body_sim: RigidBodySim,
        action_scale: float = 0.25,
    ) -> None:
        super().__init__(
            sim=body_sim.sim,
            num_worlds=body_sim.num_worlds,
            device=body_sim.torch_device,
            command_dim=0,
        )
        self._body_sim = body_sim
        self._num_actions = body_sim.num_actuated
        self._num_dofs = body_sim.num_joint_coords
        self._action_scale = action_scale

        # Action history buffers (actuated joints only).
        self._action_history: torch.Tensor = torch.zeros(
            (body_sim.num_worlds, self._num_actions),
            device=body_sim.torch_device,
            dtype=torch.float32,
        )
        self._action_history_prev: torch.Tensor = torch.zeros(
            (body_sim.num_worlds, self._num_actions),
            device=body_sim.torch_device,
            dtype=torch.float32,
        )

        # Pre-allocated observation buffer (eliminates torch.cat)
        self._obs_buffer = torch.zeros(
            (body_sim.num_worlds, 3 + self._num_dofs + 2 * self._num_actions),
            device=body_sim.torch_device,
            dtype=torch.float32,
        )

    @property
    def num_observations(self) -> int:
        return 3 + self._num_dofs + self._num_actions + self._num_actions  # 63

    def compute(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        if actions is not None:
            self._action_history_prev[:] = self._action_history
            self._action_history[:] = self._action_scale * actions

        root_pos = self._get_root_positions()
        q_j = self._get_joint_positions()

        d = self._num_dofs
        a = self._num_actions
        self._obs_buffer[:, :3] = root_pos
        self._obs_buffer[:, 3 : 3 + d] = q_j
        self._obs_buffer[:, 3 + d : 3 + d + a] = self._action_history
        self._obs_buffer[:, 3 + d + a :] = self._action_history_prev
        return self._obs_buffer

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._action_history.zero_()
            self._action_history_prev.zero_()
        else:
            self._action_history[env_ids] = 0.0
            self._action_history_prev[env_ids] = 0.0


# ---------------------------------------------------------------------------
# BipedalObservation — standalone Bipedal observation builder
# ---------------------------------------------------------------------------


class BipedalObservation(ObservationBuilder, torch.nn.Module):
    """Bipedal observation builder for inference.

    Reads commands from :pyattr:`command` (shape ``(num_worlds, 10)``),
    simulator state from a :class:`RigidBodySim`, and maintains action
    history and gait phase internally.

    Command tensor layout (10 dims)::

         [0]      path_heading         (1)
         [1:3]    path_position_2d     (2)
         [3:5]    cmd_vel_xy           (2)
         [5]      cmd_yaw_rate         (1)
         [6:10]   neck_cmd             (4)

    Phase is managed internally via a pretrained :class:`PhaseRate` model
    that predicts the gait frequency from the path command.
    """

    # -- Command tensor indices --
    CMD_DIM = 10
    CMD_PATH_HEADING = 0
    CMD_PATH_POSITION = slice(1, 3)
    CMD_VEL = slice(3, 5)
    CMD_YAW_RATE = 5
    CMD_HEAD = slice(6, 10)

    def __init__(
        self,
        body_sim: RigidBodySim,
        joint_position_default: list[float],
        joint_position_range: list[float],
        joint_velocity_scale: float,
        path_deviation_scale: float,
        phase_embedding_dim: int,
        phase_rate_policy_path: str,
        dt: float = 0.02,
        num_joints: int = 14,
    ) -> None:
        torch.nn.Module.__init__(self)
        ObservationBuilder.__init__(
            self,
            sim=body_sim.sim,
            num_worlds=body_sim.num_worlds,
            device=body_sim.torch_device,
            command_dim=self.CMD_DIM,
        )
        self._body_sim = body_sim
        self._dt = dt

        self.joint_velocity_scale = joint_velocity_scale
        self.path_deviation_scale = path_deviation_scale

        # Joint normalization
        self.register_buffer(
            "_joint_position_default",
            torch.tensor(joint_position_default, dtype=torch.float),
        )
        self.register_buffer(
            "_joint_position_range",
            torch.tensor(joint_position_range, dtype=torch.float),
        )

        # Phase encoding
        freq_2pi, offset = periodic_encoding(k=phase_embedding_dim // 2)
        self.register_buffer("_freq_2pi", torch.from_numpy(freq_2pi).to(dtype=torch.float))
        self.register_buffer("_offset", torch.from_numpy(offset).to(dtype=torch.float))

        phase_encoding_size = len(freq_2pi)

        # History size
        self.history_size = num_joints * 2

        # Observation structure (matches the GTC training layout)
        self.obs_idx = StackedIndices(
            [
                ("orientation_root_to_path", 9),
                ("path_deviation", 2),
                ("path_deviation_in_heading", 2),
                ("path_cmd", 3),
                ("path_lin_vel_in_root", 3),
                ("path_ang_vel_in_root", 3),
                ("phase_encoding", phase_encoding_size),
                ("neck_cmd", 4),
                ("root_lin_vel_in_root", 3),
                ("root_ang_vel_in_root", 3),
                ("normalized_joint_positions", num_joints),
                ("normalized_joint_velocities", num_joints),
                ("history", self.history_size),
            ]
        )
        self.num_obs = len(self.obs_idx)

        # Action history (normalised joint-position setpoints)
        self._action_hist_0 = torch.zeros(self._num_worlds, num_joints, device=self._device, dtype=torch.float)
        self._action_hist_1 = torch.zeros(self._num_worlds, num_joints, device=self._device, dtype=torch.float)

        # Internal gait phase state
        self._phase = torch.zeros(self._num_worlds, device=self._device, dtype=torch.float)
        self._phase_rate = PhaseRate(phase_rate_policy_path, self.obs_idx.path_cmd)

        # Cached intermediates (populated by compute, used by subclasses
        # for privileged observations)
        self._root_orientation_in_path: torch.Tensor | None = None
        self._root_lin_vel_in_root: torch.Tensor | None = None
        self._root_ang_vel_in_root: torch.Tensor | None = None
        self._skip_obs = torch.empty((self._num_worlds, 0), dtype=torch.float, device=self._device)

        # Move registered buffers to device
        self.to(self._device)

        # -- Pre-allocated observation buffer --
        self._obs_buffer = torch.zeros(self._num_worlds, self.num_obs, device=self._device, dtype=torch.float)
        self._wp_obs = wp.from_torch(self._obs_buffer.reshape(-1))

        # Phase rate
        self._phase_rate_input = torch.zeros(self._num_worlds, 3, device=self._device, dtype=torch.float)

        # Warp views of simulator state
        self._wp_q_i = wp.from_torch(body_sim.q_i.reshape(-1))
        self._wp_u_i = wp.from_torch(body_sim.u_i.reshape(-1))
        self._wp_q_j = wp.from_torch(body_sim.q_j.reshape(-1))
        self._wp_dq_j = wp.from_torch(body_sim.dq_j.reshape(-1))
        self._wp_command = wp.from_torch(self._command.reshape(-1))
        self._wp_phase = wp.from_torch(self._phase)
        self._wp_freq_2pi = wp.from_torch(self._freq_2pi)
        self._wp_offset = wp.from_torch(self._offset)
        self._wp_joint_default = wp.from_torch(self._joint_position_default)
        self._wp_joint_range = wp.from_torch(self._joint_position_range)

        # Observation group offsets from StackedIndices (matches _OBS_NAMES order)
        obs_offsets = torch.tensor(
            [self.obs_idx[name].start for name in _OBS_NAMES],
            dtype=torch.int32,
            device=self._device,
        )
        self._wp_obs_offsets = wp.from_torch(obs_offsets)

        # Stride constants for kernel
        self._num_bodies = body_sim.num_bodies
        self._num_joint_coords = body_sim.num_joint_coords
        self._num_joint_dofs = body_sim.num_joint_dofs
        self._num_joints = num_joints
        self._phase_enc_dim = phase_encoding_size
        self._wp_device = body_sim.device

        # Pre-computed inverse scales
        self._inv_path_dev_scale = 1.0 / path_deviation_scale
        self._inv_joint_vel_scale = 1.0 / joint_velocity_scale

        # Action history slice indices
        self._hist_start = self.obs_idx.history.start
        self._hist_mid = self._hist_start + num_joints

        # Indices for cached velocity views
        self._root_lin_vel_start = self.obs_idx.root_lin_vel_in_root.start
        self._root_ang_vel_start = self.obs_idx.root_ang_vel_in_root.start

    def get_feature_module(self) -> BipedalObservation:
        return self

    @property
    def num_observations(self) -> int:
        return self.num_obs

    def compute(self, setpoints: torch.Tensor | None = None) -> torch.Tensor:
        """Build the observation tensor from current simulator state.

        Args:
            setpoints: Latest joint-position setpoints (raw, un-normalised),
                shape ``(num_worlds, num_joints)``.  ``None`` on the very
                first step before any action has been applied.
        """
        nw = self._num_worlds

        # -- Phase advance --
        self._phase_rate_input[:, :2] = self._command[:, self.CMD_VEL]
        self._phase_rate_input[:, 2] = self._command[:, self.CMD_YAW_RATE]
        with torch.inference_mode():
            rate = self._phase_rate._phase_rate(self._phase_rate_input).squeeze(-1)
        self._phase.add_(rate * self._dt).remainder_(1.0)

        # -- Warp kernel: obs[0:hist_start] --
        wp.launch(
            _compute_bipedal_obs_core,
            dim=nw,
            inputs=[
                self._wp_obs,
                self._wp_q_i,
                self._wp_u_i,
                self._wp_q_j,
                self._wp_dq_j,
                self._wp_command,
                self._wp_phase,
                self._wp_freq_2pi,
                self._wp_offset,
                self._wp_joint_default,
                self._wp_joint_range,
                self._wp_obs_offsets,
                self._num_bodies,
                self._num_joint_coords,
                self._num_joint_dofs,
                self.num_obs,
                self.CMD_DIM,
                self._inv_path_dev_scale,
                self._inv_joint_vel_scale,
                self._phase_enc_dim,
                self._num_joints,
            ],
            device=self._wp_device,
        )

        # -- Action history: pointer swap (no copy) then overwrite --
        self._action_hist_0, self._action_hist_1 = self._action_hist_1, self._action_hist_0
        if setpoints is not None:
            torch.sub(setpoints, self._joint_position_default, out=self._action_hist_0)
            self._action_hist_0.div_(self._joint_position_range)

        # -- Write action history into pre-allocated buffer --
        self._obs_buffer[:, self._hist_start : self._hist_mid] = self._action_hist_0
        self._obs_buffer[:, self._hist_mid : self._hist_start + self.history_size] = self._action_hist_1

        # -- Cache velocity views for subclasses --
        s = self._root_lin_vel_start
        self._root_lin_vel_in_root = self._obs_buffer[:, s : s + 3]
        s = self._root_ang_vel_start
        self._root_ang_vel_in_root = self._obs_buffer[:, s : s + 3]

        return self._obs_buffer

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset action history and phase for the given environments."""
        if env_ids is None:
            normalized = (self._body_sim.q_j - self._joint_position_default) / self._joint_position_range
            self._action_hist_0[:] = normalized
            self._action_hist_1[:] = normalized
            self._phase.zero_()
        else:
            normalized = (self._body_sim.q_j[env_ids] - self._joint_position_default) / self._joint_position_range
            self._action_hist_0[env_ids] = normalized
            self._action_hist_1[env_ids] = normalized
            self._phase[env_ids] = 0.0
