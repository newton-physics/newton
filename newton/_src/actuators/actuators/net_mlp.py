# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import warp as wp

from ..kernels import nn_output_kernel
from .base import Actuator

if TYPE_CHECKING:
    from ...sim.control import Control
    from ...sim.state import State


class ActuatorNetMLP(Actuator):
    """MLP-based neural network actuator.

    Uses a pre-trained MLP to compute joint torques from position error
    and velocity history. The network takes scaled, concatenated position
    errors and velocities from selected history timesteps.

    Input preparation (per actuator)::

        pos_input = [pos_error[t] for t in input_idx] * pos_scale
        vel_input = [velocity[t] for t in input_idx] * vel_scale
        network_input = cat(pos_input, vel_input)  # or vel, pos if input_order="vel_pos"

    Output: ``torque = network(input) * torque_scale``, clamped to ``±max_force``.

    Stateful: maintains position error and velocity history buffers.

    .. note::
        Requires PyTorch. Not CUDA-graph compatible
        (:meth:`is_graphable` returns ``False``).
    """

    SCALAR_PARAMS: ClassVar[set[str]] = {"network_path"}

    @dataclass
    class State:
        """History buffers for MLP actuator."""

        pos_error_history: Any = None
        vel_history: Any = None

        def reset(self) -> None:
            """Zero history buffers in-place."""
            self.pos_error_history.zero_()
            self.vel_history.zero_()

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve arguments with defaults. Requires ``network_path``.

        Args:
            args: User-provided arguments.

        Returns:
            Arguments with defaults.

        Raises:
            ValueError: If ``network_path`` not provided.
        """
        if "network_path" not in args:
            raise ValueError("ActuatorNetMLP requires 'network_path' argument")
        return {
            "network_path": args["network_path"],
            "max_force": args.get("max_force", math.inf),
            "pos_scale": args.get("pos_scale", 1.0),
            "vel_scale": args.get("vel_scale", 1.0),
            "torque_scale": args.get("torque_scale", 1.0),
            "input_order": args.get("input_order", "pos_vel"),
            "input_idx": args.get("input_idx", None),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        max_force: wp.array[float],
        pos_scale: float = 1.0,
        vel_scale: float = 1.0,
        torque_scale: float = 1.0,
        input_order: str = "pos_vel",
        input_idx: list[int] | None = None,
        network: Any = None,
        network_path: str | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_output_attr: str = "joint_f",
    ):
        """Initialize MLP actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            max_force: Per-actuator force limits [N or N·m]. Shape ``(N,)``.
            pos_scale: Scaling factor for position error inputs.
            vel_scale: Scaling factor for velocity inputs.
            torque_scale: Scaling factor for network output torques.
            input_order: Concatenation order, ``"pos_vel"`` or ``"vel_pos"``.
            input_idx: History timestep indices to feed the network. 0 = current step,
                1 = one step ago, etc. Default ``[0]`` (current only).
            network: Pre-trained network (``torch.nn.Module``). If ``None``, loaded from
                *network_path*.
            network_path: Path to a TorchScript model file. Used when *network* is ``None``.
            state_pos_attr: Attribute on :class:`~newton.State` for joint positions.
            state_vel_attr: Attribute on :class:`~newton.State` for joint velocities.
            control_target_pos_attr: Attribute on :class:`~newton.Control` for target positions.
            control_output_attr: Attribute on :class:`~newton.Control` for output forces.
        """
        import torch

        super().__init__(input_indices, output_indices, control_output_attr)

        if len(max_force) != self.num_actuators:
            raise ValueError(f"max_force length ({len(max_force)}) must match num_actuators ({self.num_actuators})")

        self.max_force = max_force
        self.pos_scale = pos_scale
        self.vel_scale = vel_scale
        self.torque_scale = torque_scale
        if input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{input_order}'")
        self.input_order = input_order
        self.input_idx = input_idx
        if not self.input_idx:
            self.input_idx = [0]
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        device = input_indices.device
        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")

        self.network_path = network_path
        if network is not None:
            self.network = network.to(self._torch_device).eval()
        elif network_path is not None:
            self.network = torch.jit.load(network_path, map_location=self._torch_device).eval()
        else:
            raise ValueError("Either 'network' or 'network_path' must be provided")

        self._torch_indices = torch.tensor(input_indices.numpy(), dtype=torch.long, device=self._torch_device)

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr

    def _run_controller(
        self,
        sim_state: State,
        sim_control: Control,
        controller_output: wp.array[float],
        output_indices: wp.array[wp.uint32],
        current_state: ActuatorNetMLP.State,
        dt: float,
    ) -> None:
        """Compute MLP network torques."""
        import torch

        current_pos = wp.to_torch(getattr(sim_state, self.state_pos_attr))
        current_vel = wp.to_torch(getattr(sim_state, self.state_vel_attr))
        target_pos = wp.to_torch(getattr(sim_control, self.control_target_pos_attr))

        pos_error = target_pos[self._torch_indices] - current_pos[self._torch_indices]
        vel = current_vel[self._torch_indices]

        # Write current values at index 0; state manager will roll this into history.
        current_state.pos_error_history[0] = pos_error
        current_state.vel_history[0] = vel

        pos_input = torch.stack([current_state.pos_error_history[i] for i in self.input_idx], dim=1)
        vel_input = torch.stack([current_state.vel_history[i] for i in self.input_idx], dim=1)

        if self.input_order == "pos_vel":
            net_input = torch.cat([pos_input * self.pos_scale, vel_input * self.vel_scale], dim=1)
        else:
            net_input = torch.cat([vel_input * self.vel_scale, pos_input * self.pos_scale], dim=1)

        with torch.inference_mode():
            torques = self.network(net_input)

        torques = torques.reshape(self.num_actuators) * self.torque_scale
        torques_wp = wp.from_torch(torques.contiguous(), dtype=wp.float32)

        wp.launch(
            kernel=nn_output_kernel,
            dim=self.num_actuators,
            inputs=[torques_wp, self.max_force, output_indices],
            outputs=[controller_output],
        )

    def _run_state_manager(
        self,
        sim_state: State,
        sim_control: Control,
        current_state: ActuatorNetMLP.State,
        next_state: ActuatorNetMLP.State,
        dt: float,
    ) -> None:
        """Persist updated history buffers to next state."""
        if next_state is None:
            return
        next_state.pos_error_history = current_state.pos_error_history.roll(1, 0)
        next_state.vel_history = current_state.vel_history.roll(1, 0)

    def state(self) -> ActuatorNetMLP.State:
        """Return a new state with zeroed history buffers."""
        import torch

        return ActuatorNetMLP.State(
            pos_error_history=torch.zeros(self.history_length, self.num_actuators, device=self._torch_device),
            vel_history=torch.zeros(self.history_length, self.num_actuators, device=self._torch_device),
        )
