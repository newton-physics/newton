# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import load_checkpoint
from .base import Controller

if typing.TYPE_CHECKING:
    import torch


class ControllerNeuralLSTM(Controller):
    """LSTM-based neural network controller.

    Uses a pre-trained LSTM network to compute joint effort from position
    error and velocity error. The network maintains hidden and cell state
    across timesteps to capture temporal patterns.

    The network must be callable as::

        effort, (hidden_new, cell_new) = network(input, (hidden, cell))

    where input has shape (batch, 1, 2) with features
    [pos_error, vel_error], and hidden/cell have shape
    (num_layers, batch, hidden_size).

    The network is expected to have a ``lstm`` attribute (``torch.nn.LSTM``) so
    that ``num_layers`` and ``hidden_size`` can be inferred automatically.

    Scale factors (``pos_scale``, ``vel_scale``, ``effort_scale``) are
    read from checkpoint metadata, falling back to ``1.0`` when absent.
    Supported checkpoint formats: TorchScript (``.pt`` saved with
    ``torch.jit.save``) and state-dict bundles (``{"model": state_dict,
    "metadata": {...}}`` saved with ``torch.save``).
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

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
        if "model_path" not in args:
            raise ValueError("ControllerNeuralLSTM requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("ControllerNeuralLSTM requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize LSTM controller from a checkpoint file.

        Supported checkpoint formats: TorchScript (``.pt`` saved with
        ``torch.jit.save``) and state-dict bundles (``{"model": state_dict,
        "metadata": {...}}`` saved with ``torch.save``).

        Configuration is read from checkpoint metadata:

        - ``pos_scale`` (float): position-error scaling (default ``1.0``).
        - ``vel_scale`` (float): velocity-error scaling (default ``1.0``).
        - ``effort_scale`` (float): output effort scaling (default ``1.0``).

        Args:
            model_path: Path to the checkpoint (``.pt``).
        """
        import torch

        self.model_path = model_path
        self._torch_device = torch.device("cpu")

        self.network, metadata = load_checkpoint(model_path)

        self.pos_scale = metadata.get("pos_scale", 1.0)
        self.vel_scale = metadata.get("vel_scale", 1.0)
        self.effort_scale = metadata.get("effort_scale", metadata.get("torque_scale", 1.0))

        if not hasattr(self.network, "lstm"):
            raise ValueError("network must expose a 'lstm' attribute (torch.nn.LSTM)")
        lstm = self.network.lstm
        if not hasattr(lstm, "num_layers"):
            raise ValueError("network.lstm must be a torch.nn.LSTM (missing num_layers)")
        if not lstm.batch_first:
            raise ValueError("network.lstm.batch_first must be True")
        if lstm.input_size != 2:
            raise ValueError(f"network.lstm.input_size must be 2 (pos_error, vel_error); got {lstm.input_size}")
        if lstm.bidirectional:
            raise ValueError("network.lstm must not be bidirectional")
        if getattr(lstm, "proj_size", 0) != 0:
            raise ValueError(f"network.lstm.proj_size must be 0; got {lstm.proj_size}")

        self._num_layers = lstm.num_layers
        self._hidden_size = lstm.hidden_size

        self._torch_input_indices: torch.Tensor | None = None
        self._torch_vel_indices: torch.Tensor | None = None
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

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralLSTM.State:
        import torch

        return ControllerNeuralLSTM.State(
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
        state: ControllerNeuralLSTM.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        import torch

        if self._torch_input_indices is None:
            self._torch_input_indices = torch.tensor(pos_indices.numpy(), dtype=torch.long, device=self._torch_device)
            self._torch_vel_indices = torch.tensor(vel_indices.numpy(), dtype=torch.long, device=self._torch_device)

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target_p = wp.to_torch(target_pos)
        target_v = wp.to_torch(target_vel)

        torch_target_pos_idx = (
            self._torch_input_indices if target_pos_indices is pos_indices else self._torch_sequential_indices
        )
        torch_target_vel_idx = (
            self._torch_vel_indices if target_vel_indices is vel_indices else self._torch_sequential_indices
        )

        pos_error = target_p[torch_target_pos_idx] - current_pos[self._torch_input_indices]
        vel_error = target_v[torch_target_vel_idx] - current_vel[self._torch_vel_indices]

        # (N, 1, 2): seq_len=1, features=[pos_error * scale, vel_error * scale]
        net_input = torch.stack([pos_error * self.pos_scale, vel_error * self.vel_scale], dim=1).unsqueeze(1)

        with torch.inference_mode():
            effort, (self._hidden, self._cell) = self.network(
                net_input,
                (state.hidden, state.cell),
            )

        effort = effort.reshape(len(forces)) * self.effort_scale
        effort_wp = wp.from_torch(effort.contiguous(), dtype=wp.float32)
        wp.copy(forces, effort_wp)

    def update_state(
        self,
        current_state: ControllerNeuralLSTM.State,
        next_state: ControllerNeuralLSTM.State,
    ) -> None:
        if next_state is None:
            return
        next_state.hidden = self._hidden
        next_state.cell = self._cell
