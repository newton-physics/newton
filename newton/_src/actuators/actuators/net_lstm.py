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


class ActuatorNetLSTM(Actuator):
    """LSTM-based neural network actuator.

    Uses a pre-trained LSTM network to compute joint torques from position
    error and velocity. The network maintains hidden and cell state across
    timesteps to capture temporal dynamics of the actuator.

    The network must be callable as::

        torques, (h_new, c_new) = network(input, (h, c))

    where *input* has shape ``(batch, 1, 2)`` with features ``[pos_error, velocity]``,
    and *h*/*c* have shape ``(num_layers, batch, hidden_size)``.

    The network is expected to have a ``lstm`` attribute (``torch.nn.LSTM``) so that
    ``num_layers`` and ``hidden_size`` can be inferred automatically.

    Output: ``torque = network_output``, clamped to ``±max_force``.

    Stateful: maintains LSTM hidden and cell states.

    .. note::
        Requires PyTorch. Not CUDA-graph compatible
        (:meth:`is_graphable` returns ``False``).
    """

    SCALAR_PARAMS: ClassVar[set[str]] = {"network_path"}

    @dataclass
    class State:
        """LSTM hidden and cell state."""

        hidden: Any = None
        cell: Any = None

        def reset(self) -> None:
            """Reset hidden and cell state to zeros."""
            self.hidden = self.hidden.new_zeros(self.hidden.shape)
            self.cell = self.cell.new_zeros(self.cell.shape)

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
            raise ValueError("ActuatorNetLSTM requires 'network_path' argument")
        return {
            "network_path": args["network_path"],
            "max_force": args.get("max_force", math.inf),
        }

    def __init__(
        self,
        input_indices: wp.array[wp.uint32],
        output_indices: wp.array[wp.uint32],
        max_force: wp.array[float],
        network: Any = None,
        network_path: str | None = None,
        state_pos_attr: str = "joint_q",
        state_vel_attr: str = "joint_qd",
        control_target_pos_attr: str = "joint_target_pos",
        control_output_attr: str = "joint_f",
    ):
        """Initialize LSTM actuator.

        Args:
            input_indices: DOF indices for reading state and targets. Shape ``(N,)``.
            output_indices: DOF indices for writing output. Shape ``(N,)``.
            max_force: Per-actuator force limits [N or N·m]. Shape ``(N,)``.
            network: Pre-trained LSTM network (``torch.nn.Module``). If ``None``, loaded
                from *network_path*.
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

        device = input_indices.device
        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")

        self.network_path = network_path
        if network is not None:
            self.network = network.to(self._torch_device).eval()
        elif network_path is not None:
            self.network = torch.jit.load(network_path, map_location=self._torch_device).eval()
        else:
            raise ValueError("Either 'network' or 'network_path' must be provided")

        if not hasattr(self.network, "lstm"):
            raise ValueError(
                "network must expose a 'lstm' attribute (torch.nn.LSTM) for automatic num_layers/hidden_size inference"
            )
        lstm = self.network.lstm
        if not hasattr(lstm, "num_layers"):
            raise ValueError("network.lstm must be a torch.nn.LSTM (missing num_layers attribute)")
        if not lstm.batch_first:
            raise ValueError(
                "network.lstm.batch_first must be True; ActuatorNetLSTM feeds input as (batch, seq_len=1, input_size=2)"
            )
        if lstm.input_size != 2:
            raise ValueError(f"network.lstm.input_size must be 2 (pos_error, velocity); got {lstm.input_size}")
        if lstm.bidirectional:
            raise ValueError(
                "network.lstm must not be bidirectional; "
                "ActuatorNetLSTM expects num_directions=1 for hidden/cell state shapes"
            )
        if getattr(lstm, "proj_size", 0) != 0:
            raise ValueError(f"network.lstm.proj_size must be 0 (no projection); got {lstm.proj_size}")

        self._num_layers = lstm.num_layers
        self._hidden_size = lstm.hidden_size

        self._torch_indices = torch.tensor(input_indices.numpy(), dtype=torch.long, device=self._torch_device)

        self._hidden: Any = None
        self._cell: Any = None

        self.state_pos_attr = state_pos_attr
        self.state_vel_attr = state_vel_attr
        self.control_target_pos_attr = control_target_pos_attr

    def _run_controller(
        self,
        sim_state: State,
        sim_control: Control,
        controller_output: wp.array[float],
        output_indices: wp.array[wp.uint32],
        current_state: ActuatorNetLSTM.State,
        dt: float,
    ) -> None:
        """Compute LSTM network torques."""
        import torch

        current_pos = wp.to_torch(getattr(sim_state, self.state_pos_attr))
        current_vel = wp.to_torch(getattr(sim_state, self.state_vel_attr))
        target_pos = wp.to_torch(getattr(sim_control, self.control_target_pos_attr))

        pos_error = target_pos[self._torch_indices] - current_pos[self._torch_indices]
        vel = current_vel[self._torch_indices]

        # (num_actuators, 1, 2): seq_len=1, features=[pos_error, velocity]
        net_input = torch.stack([pos_error, vel], dim=1).unsqueeze(1)

        with torch.inference_mode():
            torques, (self._hidden, self._cell) = self.network(
                net_input,
                (current_state.hidden, current_state.cell),
            )

        torques = torques.reshape(self.num_actuators)
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
        current_state: ActuatorNetLSTM.State,
        next_state: ActuatorNetLSTM.State,
        dt: float,
    ) -> None:
        """Persist updated LSTM hidden/cell state."""
        if next_state is None:
            return
        next_state.hidden = self._hidden
        next_state.cell = self._cell

    def state(self) -> ActuatorNetLSTM.State:
        """Return a new state with zeroed hidden and cell tensors."""
        import torch

        return ActuatorNetLSTM.State(
            hidden=torch.zeros(self._num_layers, self.num_actuators, self._hidden_size, device=self._torch_device),
            cell=torch.zeros(self._num_layers, self.num_actuators, self._hidden_size, device=self._torch_device),
        )
