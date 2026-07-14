# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import _looks_like_torch_checkpoint, _parse_metadata_scale, _runtime_shape, load_checkpoint, load_metadata
from .base import Controller

if typing.TYPE_CHECKING:
    import torch


@wp.kernel
def _compute_errors_kernel(
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    pos_error: wp.array[float],
    vel_error: wp.array[float],
):
    i = wp.tid()
    pi = pos_indices[i]
    vi = vel_indices[i]
    tpi = target_pos_indices[i]
    tvi = target_vel_indices[i]
    pos_error[i] = target_pos[tpi] - positions[pi]
    vel_error[i] = target_vel[tvi] - velocities[vi]


@wp.kernel
def _assemble_net_input_kernel(
    pos_error: wp.array[float],
    vel_error: wp.array[float],
    pos_history: wp.array2d[float],
    vel_history: wp.array2d[float],
    input_idx: wp.array[int],
    pos_scale: float,
    vel_scale: float,
    k_per_block: int,
    pos_first: int,
    has_history: int,
    out: wp.array2d[float],
):
    i, k = wp.tid()
    block = k // k_per_block
    j = k % k_per_block
    idx = input_idx[j]
    is_pos = block == 0 if pos_first != 0 else block == 1
    if is_pos:
        if idx == 0:
            out[i, k] = pos_error[i] * pos_scale
        else:
            if has_history != 0:
                out[i, k] = pos_history[idx - 1, i] * pos_scale
            else:
                out[i, k] = 0.0
    else:
        if idx == 0:
            out[i, k] = vel_error[i] * vel_scale
        else:
            if has_history != 0:
                out[i, k] = vel_history[idx - 1, i] * vel_scale
            else:
                out[i, k] = 0.0


@wp.kernel
def _roll_history_kernel(
    cur_pos_history: wp.array2d[float],
    cur_vel_history: wp.array2d[float],
    pos_error: wp.array[float],
    vel_error: wp.array[float],
    next_pos_history: wp.array2d[float],
    next_vel_history: wp.array2d[float],
    history_length: int,
):
    t, i = wp.tid()
    if t == 0:
        next_pos_history[0, i] = pos_error[i]
        next_vel_history[0, i] = vel_error[i]
    else:
        next_pos_history[t, i] = cur_pos_history[t - 1, i]
        next_vel_history[t, i] = cur_vel_history[t - 1, i]


@wp.kernel
def _scale_and_copy_kernel(
    src: wp.array2d[float],
    dst: wp.array[float],
    scale: float,
    cols: int,
):
    i = wp.tid()
    row = i // cols
    col = i % cols
    dst[i] = src[row, col] * scale


@wp.kernel
def _zero_masked_2d_kernel(buf: wp.array2d[float], mask: wp.array[wp.bool]):
    i, j = wp.tid()
    if mask[j]:
        buf[i, j] = 0.0


@wp.kernel(enable_backward=False)
def _assemble_scaled_input_kernel(
    q: wp.array[float],
    qd: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    pos_scale: float,
    vel_scale: float,
    pos_first: int,
    net_input: wp.array2d[float],
):
    """Scaled (pos_error, vel_error) network input from per-slot state arrays."""
    i = wp.tid()
    e_q = (target_q[i] - q[i]) * pos_scale
    e_qd = (target_qd[i] - qd[i]) * vel_scale
    if pos_first != 0:
        net_input[i, 0] = e_q
        net_input[i, 1] = e_qd
    else:
        net_input[i, 0] = e_qd
        net_input[i, 1] = e_q


@wp.kernel(enable_backward=False)
def _output_and_state_grads_kernel(
    net_output: wp.array2d[float],
    in_grad: wp.array2d[float],
    pos_col: int,
    vel_col: int,
    pos_scale: float,
    vel_scale: float,
    effort_scale: float,
    tau: wp.array[float],
    dtau_dq: wp.array[float],
    dtau_dqd: wp.array[float],
):
    """Physical effort and its state derivatives from the net output and input gradients.

    The net reads scaled errors (e_q = tq - q, e_qd = tqd - qd), so the chain
    rule gives ``d(tau)/dq = -s_t s_p d(net)/d(in_pos)`` and
    ``d(tau)/d(qd) = -s_t s_v d(net)/d(in_vel)``.
    """
    i = wp.tid()
    tau[i] = net_output[i, 0] * effort_scale
    dtau_dq[i] = -in_grad[i, pos_col] * pos_scale * effort_scale
    dtau_dqd[i] = -in_grad[i, vel_col] * vel_scale * effort_scale


class ControllerNeuralMLP(Controller):
    """MLP-based neural network controller.

    Uses a pre-trained MLP to compute joint effort from concatenated, scaled
    position-error and velocity-error history. The output is multiplied by
    ``effort_scale`` to convert from network units to physical effort
    [N or N·m].

    Configuration parameters (``input_order``, ``input_idx``,
    ``pos_scale``, ``vel_scale``, ``effort_scale``) are read from checkpoint
    metadata, falling back to defaults when absent. ``.onnx`` checkpoints run
    through Warp-NN. ``.pt`` and ``.pth`` checkpoints keep the Torch backend.

    Implicit actuation (:meth:`implicit_force_grad`) is limited to the
    Warp-NN backend with ``history_length == 1``, and depends on a workaround
    for warp-nn allocating its tensors without ``requires_grad`` — without it,
    autodiff gradients through multi-layer nets (and, under CUDA graph
    capture, even single-layer nets) are silently zero.

    TODO: drop the ``requires_grad`` workaround once fixed upstream in
    warp-nn; extend implicit support to the Torch backend and to input
    histories (``history_length > 1``).
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(Controller.State):
        """History buffers for MLP controller."""

        pos_error_history: torch.Tensor | wp.array2d[float] | None = None
        """Position error history, shape [history_length, actuator_count]."""
        vel_error_history: torch.Tensor | wp.array2d[float] | None = None
        """Velocity error history, shape [history_length, actuator_count]."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.pos_error_history.zero_()
                self.vel_error_history.zero_()
            elif type(self.pos_error_history).__module__.startswith("torch"):
                t = wp.to_torch(mask).bool()
                self.pos_error_history[:, t] = 0.0
                self.vel_error_history[:, t] = 0.0
            else:
                wp.launch(
                    _zero_masked_2d_kernel,
                    dim=self.pos_error_history.shape,
                    inputs=[self.pos_error_history, mask],
                    device=self.pos_error_history.device,
                )
                wp.launch(
                    _zero_masked_2d_kernel,
                    dim=self.vel_error_history.shape,
                    inputs=[self.vel_error_history, mask],
                    device=self.vel_error_history.device,
                )

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

        Args:
            model_path: Path to the ``.onnx``, ``.pt``, or ``.pth`` checkpoint.
        """
        self.model_path = model_path
        self._is_torch_checkpoint = _looks_like_torch_checkpoint(model_path)

        if self._is_torch_checkpoint:
            import torch

            self._torch_device = torch.device("cpu")
            self.network, metadata = load_checkpoint(model_path)
        else:
            metadata = load_metadata(model_path)
            self.network = None
            self._torch_device = None

        self.input_order = metadata.get("input_order", "pos_vel")
        if self.input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{self.input_order}'")

        self.input_idx = metadata.get("input_idx", [0])
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        if self._is_torch_checkpoint:
            self.pos_scale = metadata.get("pos_scale", 1.0)
            self.vel_scale = metadata.get("vel_scale", 1.0)
            self.effort_scale = metadata.get("effort_scale", metadata.get("torque_scale", 1.0))
        else:
            self.pos_scale = _parse_metadata_scale(metadata, "pos_scale", model_path)
            self.vel_scale = _parse_metadata_scale(metadata, "vel_scale", model_path)
            self.effort_scale = _parse_metadata_scale(metadata, "effort_scale", model_path, fallback_key="torque_scale")

        self._network = None
        self._device: wp.Device | None = None
        self._num_actuators = 0
        self._torch_input_indices: torch.Tensor | None = None
        self._torch_vel_indices: torch.Tensor | None = None
        self._torch_sequential_indices: torch.Tensor | None = None
        self._current_pos_error: torch.Tensor | None = None
        self._current_vel_error: torch.Tensor | None = None

        self._pos_error: wp.array[float] | None = None
        self._vel_error: wp.array[float] | None = None
        self._net_input: wp.array2d[float] | None = None
        self._input_idx_wp: wp.array[int] | None = None
        self._net_output_name: str | None = None
        self._net_input_name: str | None = None
        self._grad_seed: wp.array2d[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._device = device
        self._num_actuators = num_actuators

        if self._is_torch_checkpoint:
            import torch

            self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
            self.network = self.network.to(self._torch_device)
            self._torch_sequential_indices = torch.arange(num_actuators, dtype=torch.long, device=self._torch_device)
            return

        runtime, _ = load_checkpoint(
            self.model_path,
            device=device,
            batch_size=num_actuators,
            input_batch_axes=0,
        )
        self._network = runtime
        self.network = runtime
        self._net_input_name = runtime.input_names[0]
        self._net_output_name = runtime.output_names[0]

        feat = 2 * len(self.input_idx)
        self._net_input = wp.zeros((num_actuators, feat), dtype=wp.float32, device=device)
        self._pos_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._vel_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._input_idx_wp = wp.array(self.input_idx, dtype=wp.int32, device=device)

        try:
            out_shape = _runtime_shape(runtime, self._net_output_name)
        except ValueError:
            runtime({self._net_input_name: self._net_input})
            out_shape = _runtime_shape(runtime, self._net_output_name)
        if out_shape != (num_actuators, 1):
            raise ValueError(
                f"ControllerNeuralMLP: network output '{self._net_output_name}' has shape {out_shape}, "
                f"expected {(num_actuators, 1)} (one scalar effort per actuator)"
            )

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return not self._is_torch_checkpoint

    def implicit_force_grad(self):
        """Return the force-and-gradients hook for the implicit strategy, or ``None``.

        Supported for the Warp-NN (``.onnx``) backend with
        ``history_length == 1``; other configurations return ``None``. See
        :meth:`Controller.implicit_force_grad` for the hook contract.
        """
        if self._is_torch_checkpoint or self.history_length != 1 or self._network is None:
            return None
        if self._grad_seed is None:
            self._net_input.requires_grad = True
            self._grad_seed = wp.full((self._num_actuators, 1), 1.0, dtype=wp.float32, device=self._device)
            # Workaround: warp-nn allocates its tensors without requires_grad,
            # which silently zeroes tape gradients through intermediates and,
            # under CUDA graph capture, even through the output. Remove once
            # fixed upstream in warp-nn.
            for tensor in self._network._tensors.values():
                if isinstance(tensor, wp.array) and not tensor.requires_grad:
                    tensor.requires_grad = True
        return self._force_and_grad

    def _force_and_grad(
        self,
        q: wp.array[float],
        qd: wp.array[float],
        target_q: wp.array[float],
        target_qd: wp.array[float],
        tau: wp.array[float],
        dtau_dq: wp.array[float],
        dtau_dqd: wp.array[float],
    ) -> None:
        """One network forward + autodiff backward at the given per-slot state.

        Writes the physical effort and its derivatives w.r.t. position and
        velocity; the implicit strategy's Newton loop consumes them.
        """
        n = self._num_actuators
        device = self._device
        pos_first = 1 if self.input_order == "pos_vel" else 0
        pos_col = 0 if self.input_order == "pos_vel" else 1

        wp.launch(
            _assemble_scaled_input_kernel,
            dim=n,
            inputs=[q, qd, target_q, target_qd, self.pos_scale, self.vel_scale, pos_first],
            outputs=[self._net_input],
            device=device,
        )
        self._net_input.grad.zero_()
        tape = wp.Tape()
        with tape:
            out = self._network({self._net_input_name: self._net_input})[self._net_output_name]
        tape.backward(grads={out: self._grad_seed})
        wp.launch(
            _output_and_state_grads_kernel,
            dim=n,
            inputs=[
                out,
                self._net_input.grad,
                pos_col,
                1 - pos_col,
                self.pos_scale,
                self.vel_scale,
                self.effort_scale,
            ],
            outputs=[tau, dtau_dq, dtau_dqd],
            device=device,
        )

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralMLP.State:
        if self._is_torch_checkpoint:
            import torch

            return ControllerNeuralMLP.State(
                pos_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
                vel_error_history=torch.zeros(self.history_length, num_actuators, device=self._torch_device),
            )
        return ControllerNeuralMLP.State(
            pos_error_history=wp.zeros((self.history_length, num_actuators), dtype=wp.float32, device=device),
            vel_error_history=wp.zeros((self.history_length, num_actuators), dtype=wp.float32, device=device),
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
        device = device or self._device
        n = self._num_actuators

        if self._is_torch_checkpoint:
            self._compute_torch(
                positions,
                velocities,
                target_pos,
                target_vel,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                forces,
                state,
            )
            return

        wp.launch(
            _compute_errors_kernel,
            dim=n,
            inputs=[
                target_pos,
                target_vel,
                positions,
                velocities,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self._pos_error,
                self._vel_error,
            ],
            device=device,
        )

        k_per_block = len(self.input_idx)
        pos_first = 1 if self.input_order == "pos_vel" else 0
        has_history = 1 if self.history_length > 1 else 0
        wp.launch(
            _assemble_net_input_kernel,
            dim=(n, 2 * k_per_block),
            inputs=[
                self._pos_error,
                self._vel_error,
                state.pos_error_history,
                state.vel_error_history,
                self._input_idx_wp,
                self.pos_scale,
                self.vel_scale,
                k_per_block,
                pos_first,
                has_history,
                self._net_input,
            ],
            device=device,
        )

        out = self._network({self._net_input_name: self._net_input})
        effort = out[self._net_output_name]

        wp.launch(
            _scale_and_copy_kernel,
            dim=len(forces),
            inputs=[effort, forces, self.effort_scale, 1],
            device=device,
        )

    def update_state(
        self,
        current_state: ControllerNeuralMLP.State,
        next_state: ControllerNeuralMLP.State,
    ) -> None:
        if next_state is None:
            return
        if self._is_torch_checkpoint:
            next_state.pos_error_history = current_state.pos_error_history.roll(1, 0)
            next_state.vel_error_history = current_state.vel_error_history.roll(1, 0)
            next_state.pos_error_history[0] = self._current_pos_error
            next_state.vel_error_history[0] = self._current_vel_error
            return
        h, n = current_state.pos_error_history.shape
        wp.launch(
            _roll_history_kernel,
            dim=(h, n),
            inputs=[
                current_state.pos_error_history,
                current_state.vel_error_history,
                self._pos_error,
                self._vel_error,
                next_state.pos_error_history,
                next_state.vel_error_history,
                h,
            ],
            device=self._device,
        )

    def _compute_torch(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: ControllerNeuralMLP.State,
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
