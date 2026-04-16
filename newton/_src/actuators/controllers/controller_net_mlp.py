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


class ControllerNetMLP(Controller):
    """MLP-based neural network controller.

    Uses a pre-trained MLP to compute joint torques from position error
    and velocity history.

    The network receives concatenated position-error and velocity history
    as input and is expected to return torques in physical units. Scaling is left to the user.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"network_path", "input_order", "input_idx"}

    @dataclass
    class State(Controller.State):
        """History buffers for MLP controller."""

        pos_error_history: torch.Tensor | None = None
        """Position error history, shape (history_length, N)."""
        vel_history: torch.Tensor | None = None
        """Velocity history, shape (history_length, N)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.pos_error_history.zero_()
                self.vel_history.zero_()
            else:
                t = wp.to_torch(mask).bool()
                self.pos_error_history[:, t] = 0.0
                self.vel_history[:, t] = 0.0

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "network_path" not in args:
            raise ValueError("ControllerNetMLP requires 'network_path' argument")
        return {
            "network_path": args["network_path"],
            "input_order": args.get("input_order", "pos_vel"),
            "input_idx": args.get("input_idx", None),
        }

    def __init__(
        self,
        input_order: str = "pos_vel",
        input_idx: list[int] | None = None,
        network: torch.nn.Module | None = None,
        network_path: str | None = None,
    ):
        """Initialize MLP controller.

        Args:
            input_order: Concatenation order, ``"pos_vel"`` or ``"vel_pos"``.
                Not part of the USD schema; pass programmatically when the
                network expects a non-default layout.
            input_idx: History timestep indices to feed the network. ``0`` is
                the current step, ``1`` one step ago, etc. Defaults to ``[0]``.
                Not part of the USD schema; pass programmatically.
            network: Pre-trained network. If None, loaded from *network_path*.
            network_path: Path to a TorchScript model file.
        """
        import torch

        if input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{input_order}'")
        self.input_order = input_order
        self.input_idx = input_idx if input_idx else [0]
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

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

        self._torch_input_indices: torch.Tensor | None = None
        self._torch_sequential_indices: torch.Tensor | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        import torch

        self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
        self.network = self.network.to(self._torch_device)
        self._torch_sequential_indices = torch.arange(num_actuators, dtype=torch.long, device=self._torch_device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return False

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNetMLP.State:
        import torch

        return ControllerNetMLP.State(
            pos_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
            vel_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        feedforward: wp.array[float] | None,
        input_indices: wp.array[wp.uint32],
        target_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerNetMLP.State,
        dt: float,
        device: wp.Device | None = None,
    ) -> None:
        import torch

        if self._torch_input_indices is None:
            self._torch_input_indices = torch.tensor(input_indices.numpy(), dtype=torch.long, device=self._torch_device)

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target = wp.to_torch(target_pos)

        torch_target_idx = (
            self._torch_input_indices if target_indices is input_indices else self._torch_sequential_indices
        )

        pos_error = target[torch_target_idx] - current_pos[self._torch_input_indices]
        vel = current_vel[self._torch_input_indices]

        state.pos_error_history[0] = pos_error
        state.vel_history[0] = vel

        pos_input = torch.stack([state.pos_error_history[i] for i in self.input_idx], dim=1)
        vel_input = torch.stack([state.vel_history[i] for i in self.input_idx], dim=1)

        if self.input_order == "pos_vel":
            net_input = torch.cat([pos_input, vel_input], dim=1)
        else:
            net_input = torch.cat([vel_input, pos_input], dim=1)

        with torch.inference_mode():
            torques = self.network(net_input)

        torques = torques.reshape(len(forces))
        torques_wp = wp.from_torch(torques.contiguous(), dtype=wp.float32)
        wp.copy(forces, torques_wp)

    def update_state(
        self,
        current_state: ControllerNetMLP.State,
        next_state: ControllerNetMLP.State,
    ) -> None:
        if next_state is None:
            return
        next_state.pos_error_history = current_state.pos_error_history.roll(1, 0)
        next_state.vel_history = current_state.vel_history.roll(1, 0)
