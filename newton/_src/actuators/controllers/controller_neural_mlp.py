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


class ControllerNeuralMLP(Controller):
    """MLP-based neural network controller.

    Uses a pre-trained MLP to compute joint effort from position error
    and velocity error history.

    The network receives concatenated, scaled position-error and
    velocity-error history as input.  The output is multiplied by
    ``effort_scale`` to convert from network units to physical effort
    [N or N·m].  All three scale factors default to ``1.0`` (no scaling).

    Configuration parameters (``input_order``, ``input_idx``,
    ``pos_scale``, ``vel_scale``, ``effort_scale``) are read from the
    checkpoint metadata, falling back to defaults when absent.
    Supported checkpoint formats: TorchScript (``.pt`` saved with
    ``torch.jit.save``) and state-dict bundles (``{"model": state_dict,
    "metadata": {...}}`` saved with ``torch.save``).
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(Controller.State):
        """History buffers for MLP controller."""

        pos_error_history: torch.Tensor | None = None
        """Position error history, shape (history_length, N)."""
        vel_error_history: torch.Tensor | None = None
        """Velocity error history, shape (history_length, N)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.pos_error_history.zero_()
                self.vel_error_history.zero_()
            else:
                t = wp.to_torch(mask).bool()
                self.pos_error_history[:, t] = 0.0
                self.vel_error_history[:, t] = 0.0

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "model_path" not in args:
            raise ValueError("ControllerNeuralMLP requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("ControllerNeuralMLP requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize MLP controller from a checkpoint file.

        Supported checkpoint formats: TorchScript (``.pt`` saved with
        ``torch.jit.save``) and state-dict bundles (``{"model": state_dict,
        "metadata": {...}}`` saved with ``torch.save``).

        Configuration is read from checkpoint metadata:

        - ``input_order`` (str): ``"pos_vel"`` or ``"vel_pos"`` (default ``"pos_vel"``).
        - ``input_idx`` (list[int]): history timestep indices (default ``[0]``).
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

        self.input_order = metadata.get("input_order", "pos_vel")
        if self.input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{self.input_order}'")

        self.input_idx = metadata.get("input_idx", [0])
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        self.pos_scale = metadata.get("pos_scale", 1.0)
        self.vel_scale = metadata.get("vel_scale", 1.0)
        self.effort_scale = metadata.get("effort_scale", metadata.get("torque_scale", 1.0))

        self._torch_input_indices: torch.Tensor | None = None
        self._torch_vel_indices: torch.Tensor | None = None
        self._torch_sequential_indices: torch.Tensor | None = None
        self._current_pos_error: torch.Tensor | None = None
        self._current_vel_error: torch.Tensor | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        import torch

        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
        self.network = self.network.to(self._torch_device)
        self._torch_sequential_indices = torch.arange(num_actuators, dtype=torch.long, device=self._torch_device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralMLP.State:
        import torch

        return ControllerNeuralMLP.State(
            pos_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
            vel_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
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
        state: ControllerNeuralMLP.State,
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

        self._current_pos_error = pos_error
        self._current_vel_error = vel_error

        pos_input = torch.stack(
            [pos_error if i == 0 else state.pos_error_history[i - 1] for i in self.input_idx], dim=1
        )
        vel_input = torch.stack(
            [vel_error if i == 0 else state.vel_error_history[i - 1] for i in self.input_idx], dim=1
        )

        if self.input_order == "pos_vel":
            net_input = torch.cat([pos_input * self.pos_scale, vel_input * self.vel_scale], dim=1)
        else:
            net_input = torch.cat([vel_input * self.vel_scale, pos_input * self.pos_scale], dim=1)

        with torch.inference_mode():
            effort = self.network(net_input)

        effort = effort.reshape(len(forces)) * self.effort_scale
        effort_wp = wp.from_torch(effort.contiguous(), dtype=wp.float32)
        wp.copy(forces, effort_wp)

    def update_state(
        self,
        current_state: ControllerNeuralMLP.State,
        next_state: ControllerNeuralMLP.State,
    ) -> None:
        if next_state is None:
            return
        next_state.pos_error_history = current_state.pos_error_history.roll(1, 0)
        next_state.vel_error_history = current_state.vel_error_history.roll(1, 0)
        next_state.pos_error_history[0] = self._current_pos_error
        next_state.vel_error_history[0] = self._current_vel_error
