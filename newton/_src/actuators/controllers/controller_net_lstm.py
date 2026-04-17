# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from .base import Controller

if typing.TYPE_CHECKING:
    import torch


class ControllerNetLSTM(Controller):
    """LSTM-based neural network controller.

    Uses a pre-trained LSTM network to compute joint torques from position
    error and velocity. The network maintains hidden and cell state across
    timesteps to capture temporal patterns.

    The network must be callable as:
        torques, (hidden_new, cell_new) = network(input, (hidden, cell))

    where input has shape (batch, 1, 2) with features [pos_error, velocity],
    and hidden/cell have shape (num_layers, batch, hidden_size).

    The network is expected to have a ``lstm`` attribute (torch.nn.LSTM) so
    that num_layers and hidden_size can be inferred automatically.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"network_path"}

    @dataclass
    class State(Controller.State):
        """LSTM hidden and cell state."""

        hidden: torch.Tensor | None = None
        """LSTM hidden state, shape (num_layers, N, hidden_size)."""
        cell: torch.Tensor | None = None
        """LSTM cell state, shape (num_layers, N, hidden_size)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.hidden = self.hidden.new_zeros(self.hidden.shape)
                self.cell = self.cell.new_zeros(self.cell.shape)
            else:
                t = wp.to_torch(mask).bool()
                self.hidden[:, t, :] = 0.0
                self.cell[:, t, :] = 0.0

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "network_path" not in args:
            raise ValueError("ControllerNetLSTM requires 'network_path' argument")
        return {
            "network_path": args["network_path"],
        }

    def __init__(
        self,
        network: torch.nn.Module | None = None,
        network_path: str | None = None,
    ):
        """Initialize LSTM controller.

        Args:
            network: Pre-trained LSTM network.
                If None, loaded from network_path.
            network_path: Path to a TorchScript model file.
        """
        import torch

        self.network_path = network_path

        if network is not None:
            params = list(network.parameters())
            self._torch_device = params[0].device if params else torch.device("cpu")
            self.network = network.eval()
        elif network_path is not None:
            self._torch_device = torch.device("cpu")
            self.network = torch.jit.load(network_path, map_location="cpu").eval()
        else:
            raise ValueError("Either 'network' or 'network_path' must be provided")

        if not hasattr(self.network, "lstm"):
            raise ValueError("network must expose a 'lstm' attribute (torch.nn.LSTM)")
        lstm = self.network.lstm
        if not hasattr(lstm, "num_layers"):
            raise ValueError("network.lstm must be a torch.nn.LSTM (missing num_layers)")
        if not lstm.batch_first:
            raise ValueError("network.lstm.batch_first must be True")
        if lstm.input_size != 2:
            raise ValueError(f"network.lstm.input_size must be 2 (pos_error, velocity); got {lstm.input_size}")
        if lstm.bidirectional:
            raise ValueError("network.lstm must not be bidirectional")
        if getattr(lstm, "proj_size", 0) != 0:
            raise ValueError(f"network.lstm.proj_size must be 0; got {lstm.proj_size}")

        self._num_layers = lstm.num_layers
        self._hidden_size = lstm.hidden_size

        self._torch_input_indices: torch.Tensor | None = None
        self._torch_sequential_indices: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._cell: torch.Tensor | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        import torch

        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
        self.network = self.network.to(self._torch_device)
        self._torch_sequential_indices = torch.arange(num_actuators, dtype=torch.long, device=self._torch_device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNetLSTM.State:
        import torch

        return ControllerNetLSTM.State(
            hidden=torch.zeros(self._num_layers, num_actuators, self._hidden_size, device=self._torch_device),
            cell=torch.zeros(self._num_layers, num_actuators, self._hidden_size, device=self._torch_device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerNetLSTM.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        import torch

        if self._torch_input_indices is None:
            self._torch_input_indices = torch.tensor(pos_indices.numpy(), dtype=torch.long, device=self._torch_device)
            self._torch_vel_indices = torch.tensor(vel_indices.numpy(), dtype=torch.long, device=self._torch_device)

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target = wp.to_torch(target_pos)

        torch_target_pos_idx = (
            self._torch_input_indices if target_pos_indices is pos_indices else self._torch_sequential_indices
        )

        pos_error = target[torch_target_pos_idx] - current_pos[self._torch_input_indices]
        vel = current_vel[self._torch_vel_indices]

        # (N, 1, 2): seq_len=1, features=[pos_error, velocity]
        net_input = torch.stack([pos_error, vel], dim=1).unsqueeze(1)

        with torch.inference_mode():
            torques, (self._hidden, self._cell) = self.network(
                net_input,
                (state.hidden, state.cell),
            )

        torques = torques.reshape(len(forces))
        torques_wp = wp.from_torch(torques.contiguous(), dtype=wp.float32)
        wp.copy(forces, torques_wp)

    def update_state(
        self,
        current_state: ControllerNetLSTM.State,
        next_state: ControllerNetLSTM.State,
    ) -> None:
        if next_state is None:
            return
        next_state.hidden = self._hidden
        next_state.cell = self._cell
