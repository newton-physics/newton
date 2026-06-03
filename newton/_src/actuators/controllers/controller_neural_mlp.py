# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import _TorchModuleAdapter, load_checkpoint, load_metadata
from .base import Controller


@wp.kernel
def _compute_errors_and_zero_history_kernel(
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
    pos_history: wp.array2d[float],  # (history_length, N) past pos errors; row r = error from r+1 steps ago
    vel_history: wp.array2d[float],  # (history_length, N) past vel errors
    input_idx: wp.array[int],  # (K,) timestep offsets, 0 == current
    pos_scale: float,
    vel_scale: float,
    k_per_block: int,  # = K = len(input_idx)
    pos_first: int,  # 1 if input_order == "pos_vel"
    has_history: int,  # 0 if history_length == 1 (history arrays unused)
    out: wp.array2d[float],
):
    """Assemble scaled feature row ``out[i, :] = [pos_block | vel_block]`` (or vel|pos).

    ``out`` has shape ``(N, 2*K)``.  Each block is K-wide and gathers
    ``pos_error_history[input_idx[j]]`` (or current ``pos_error`` when
    ``input_idx[j] == 0``) for j in [0, K).
    """
    i, k = wp.tid()  # i in [0, N), k in [0, 2*K)
    block = k // k_per_block  # 0 -> first half, 1 -> second half
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
    cur_pos_history: wp.array2d[float],  # (H, N) current state
    cur_vel_history: wp.array2d[float],
    pos_error: wp.array[float],  # (N,) latest sample to insert at row 0
    vel_error: wp.array[float],
    next_pos_history: wp.array2d[float],  # (H, N) destination
    next_vel_history: wp.array2d[float],
    history_length: int,
):
    """Shift history by one timestep and write the latest error at row 0.

    ``next_history[0, i] = error[i]``;
    ``next_history[t, i] = cur_history[t - 1, i]`` for t in [1, H).
    Equivalent to ``np.roll(history, 1, axis=0)`` followed by writing
    the latest sample at index 0, but on-device.
    """
    t, i = wp.tid()
    if t == 0:
        next_pos_history[0, i] = pos_error[i]
        next_vel_history[0, i] = vel_error[i]
    else:
        next_pos_history[t, i] = cur_pos_history[t - 1, i]
        next_vel_history[t, i] = cur_vel_history[t - 1, i]


@wp.kernel
def _scale_and_copy_kernel(
    src: wp.array2d[float],  # (N, K) effort, row-major
    dst: wp.array[float],  # (count,) forces
    scale: float,
    cols: int,
):
    """Read the leading ``count`` entries of ``src`` (row-major) and scale them.

    Effectively flattens ``src`` and copies the first ``count`` values into
    ``dst`` after multiplying by ``scale``.
    """
    i = wp.tid()
    row = i // cols
    col = i % cols
    dst[i] = src[row, col] * scale


@wp.kernel
def _zero_masked_2d_kernel(buf: wp.array2d[float], mask: wp.array[wp.bool]):
    i, j = wp.tid()
    if mask[j]:
        buf[i, j] = 0.0


class ControllerNeuralMLP(Controller):
    """MLP-based neural network controller, ONNX-backed.

    Uses a pre-trained MLP (loaded from an ``.onnx`` file) to compute joint
    effort from concatenated, scaled position-error and velocity-error
    history.  The output is multiplied by ``effort_scale`` to convert from
    network units to physical effort [N or N·m].

    Configuration parameters (``input_order``, ``input_idx``,
    ``pos_scale``, ``vel_scale``, ``effort_scale``) are read from the ONNX
    model's metadata properties (a single ``metadata`` JSON property is
    preferred), falling back to defaults when absent.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(Controller.State):
        """History buffers for MLP controller."""

        pos_error_history: wp.array2d[float] | None = None
        """Position error history, shape (history_length, N)."""
        vel_error_history: wp.array2d[float] | None = None
        """Velocity error history, shape (history_length, N)."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                self.pos_error_history.zero_()
                self.vel_error_history.zero_()
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
        """Initialize MLP controller from an ONNX checkpoint file.

        Configuration is read from the model's metadata properties:

        - ``input_order`` (str): ``"pos_vel"`` or ``"vel_pos"`` (default ``"pos_vel"``).
        - ``input_idx`` (list[int]): history timestep indices (default ``[0]``).
        - ``pos_scale`` (float): position-error scaling (default ``1.0``).
        - ``vel_scale`` (float): velocity-error scaling (default ``1.0``).
        - ``effort_scale`` (float): output effort scaling (default ``1.0``).

        Args:
            model_path: Path to the ``.onnx`` checkpoint.
        """
        self.model_path = model_path

        metadata = load_metadata(model_path)

        self.input_order = metadata.get("input_order", "pos_vel")
        if self.input_order not in ("pos_vel", "vel_pos"):
            raise ValueError(f"input_order must be 'pos_vel' or 'vel_pos'; got '{self.input_order}'")

        self.input_idx = metadata.get("input_idx", [0])
        if any(i < 0 for i in self.input_idx):
            raise ValueError(f"input_idx must contain non-negative integers; got {self.input_idx}")
        self.history_length = max(self.input_idx) + 1

        self.pos_scale = float(metadata.get("pos_scale", 1.0))
        self.vel_scale = float(metadata.get("vel_scale", 1.0))
        self.effort_scale = float(metadata.get("effort_scale", metadata.get("torque_scale", 1.0)))

        self._network = None
        self._device: wp.Device | None = None
        self._num_actuators = 0

        self._pos_error: wp.array[float] | None = None
        self._vel_error: wp.array[float] | None = None
        self._net_input: wp.array2d[float] | None = None
        self._input_idx_wp: wp.array[int] | None = None
        self._net_output_name: str | None = None
        self._net_input_name: str | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._device = device
        self._num_actuators = num_actuators

        runtime, _ = load_checkpoint(
            self.model_path,
            device=str(device),
            batch_size=num_actuators,
            input_batch_axes=0,
        )
        self._network = runtime
        self._net_input_name = runtime.input_names[0]
        self._net_output_name = runtime.output_names[0]

        feat = 2 * len(self.input_idx)
        self._net_input = wp.zeros((num_actuators, feat), dtype=wp.float32, device=device)
        self._pos_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._vel_error = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._input_idx_wp = wp.array(self.input_idx, dtype=wp.int32, device=device)

        # Probe the output shape.  ``OnnxRuntime`` populates ``_shapes`` eagerly
        # during construction, but the deprecated ``_TorchModuleAdapter`` only
        # learns shapes after the first inference call.  Run a one-shot dry
        # forward with the zero-initialized ``_net_input`` if the shape is not
        # already known so legacy ``.pt``/``.pth`` checkpoints can be validated
        # the same way as ``.onnx`` ones.
        if self._net_output_name not in runtime._shapes:
            runtime({self._net_input_name: self._net_input})

        # The compute() copy assumes one effort per actuator, i.e. output shape
        # exactly ``(num_actuators, 1)``.  A multi-column output would silently
        # misalign actuator i with row 0, column i, dropping later rows; reject
        # such models up front instead of producing wrong inferences.
        out_shape = runtime._shapes[self._net_output_name]
        if out_shape != (num_actuators, 1):
            raise ValueError(
                f"ControllerNeuralMLP: network output '{self._net_output_name}' has shape {out_shape}, "
                f"expected {(num_actuators, 1)} (one scalar effort per actuator)"
            )

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        # The deprecated ``_TorchModuleAdapter`` round-trips through host
        # ``.numpy()`` and PyTorch and is not safe inside a CUDA graph capture;
        # only the Warp-backed ``OnnxRuntime`` path is graph-capturable.
        return not isinstance(self._network, _TorchModuleAdapter)

    def state(self, num_actuators: int, device: wp.Device) -> ControllerNeuralMLP.State:
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

        wp.launch(
            _compute_errors_and_zero_history_kernel,
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

        # Assemble net_input on-device: gather [pos_error_history[idx] for idx in input_idx]
        # and [vel_error_history[idx] ...] into preallocated self._net_input.
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

        # ``effort`` is guaranteed by ``finalize`` to be ``(N, 1)``; flatten
        # the leading ``len(forces)`` entries into ``forces`` in a single
        # launch.
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
        # Roll history along axis 0 on-device (oldest sample drops out, newest at row 0).
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
