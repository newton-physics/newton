# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ..utils import (
    _looks_like_torch_checkpoint,
    _parse_metadata_scale,
    _runtime_shape,
    load_checkpoint,
    load_metadata,
)
from ._linearization import (
    IMPLICIT_JACOBIAN_MARGIN,
    _assemble_linear_params_kernel,
    _gather_slot_state_kernel,
    _linear_force,
)
from .base import DriveBase

if typing.TYPE_CHECKING:
    import torch


@wp.kernel
def _compute_inputs_kernel(
    target_pos: wp.array[float],
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    pos_scale: float,
    vel_scale: float,
    out: wp.array3d[float],
):
    i = wp.tid()
    pi = pos_indices[i]
    vi = vel_indices[i]
    tpi = target_pos_indices[i]
    out[0, i, 0] = (target_pos[tpi] - positions[pi]) * pos_scale
    out[0, i, 1] = velocities[vi] * vel_scale


@wp.kernel
def _scale_effort_to_forces_kernel(
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
def _zero_masked_3d_kernel(buf: wp.array3d[float], mask: wp.array[wp.bool]):
    layer, b, h = wp.tid()
    if mask[b]:
        buf[layer, b, h] = 0.0


@wp.kernel(enable_backward=False)
def _assemble_scaled_input_3d_kernel(
    q: wp.array[float],
    qd: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    pos_scale: float,
    vel_scale: float,
    net_input: wp.array3d[float],
):
    """Scaled (pos_error, vel_error) LSTM input ``(1, N, 2)`` from per-slot state."""
    i = wp.tid()
    net_input[0, i, 0] = (target_q[i] - q[i]) * pos_scale
    net_input[0, i, 1] = qd[i] * vel_scale


@wp.kernel(enable_backward=False)
def _lstm_output_grads_kernel(
    net_output: wp.array2d[float],
    in_grad: wp.array3d[float],
    pos_scale: float,
    vel_scale: float,
    effort_scale: float,
    tau: wp.array[float],
    dtau_dq: wp.array[float],
    dtau_dqd: wp.array[float],
):
    """Effort and its state derivatives from the net output and input gradients.

    The net reads a scaled position error (e_q = tq - q) but a scaled raw
    velocity, so the chain rule gives ``d(tau)/dq = -s_t s_p d(net)/d(in_pos)``
    and ``d(tau)/d(qd) = +s_t s_v d(net)/d(in_vel)`` -- note the opposite signs.
    """
    i = wp.tid()
    tau[i] = net_output[i, 0] * effort_scale
    dtau_dq[i] = -in_grad[0, i, 0] * pos_scale * effort_scale
    dtau_dqd[i] = in_grad[0, i, 1] * vel_scale * effort_scale


class DriveNeuralLSTM(DriveBase):
    """LSTM-based neural network actuator drive.

    Uses a pre-trained LSTM network to compute joint effort from position
    error and joint velocity. Hidden and cell state are maintained across
    timesteps.

    Torch checkpoints use the Torch backend and preserve the Torch state
    interface. They accept pt2 archives (``.pt2`` saved with
    ``torch.export.save``; preferred) and the deprecated TorchScript (``.pt``
    saved with ``torch.jit.save``) and module-bundle
    (``{"model": <network module>, "metadata": {...}}`` saved with
    ``torch.save``) formats.

    ``.pt2`` and ``.onnx`` checkpoints must record ``num_layers`` and
    ``hidden_size`` in metadata; only legacy Torch checkpoints may omit them,
    since their loaded networks expose a live ``lstm`` attribute to inspect.

    ``.onnx`` checkpoints use Warp-NN. The exported ONNX model must have three
    inputs (input, initial hidden, and initial cell) and three graph outputs
    (effort, hidden output, and cell output). Metadata properties map those
    names to drive roles.

    ONNX checkpoints support the implicit effort mode through a per-step
    linearization of the network; Torch checkpoints do not and must use the
    explicit mode.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(DriveBase.State):
        """LSTM hidden and cell state."""

        hidden: torch.Tensor | wp.array3d[float] | None = None
        """LSTM hidden state, shape [num_layers, actuator_count, hidden_size]."""
        cell: torch.Tensor | wp.array3d[float] | None = None
        """LSTM cell state, shape [num_layers, actuator_count, hidden_size]."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            if mask is None:
                if type(self.hidden).__module__.startswith("torch"):
                    self.hidden = self.hidden.new_zeros(self.hidden.shape)
                    self.cell = self.cell.new_zeros(self.cell.shape)
                else:
                    self.hidden.zero_()
                    self.cell.zero_()
            elif type(self.hidden).__module__.startswith("torch"):
                # Network outputs are produced under torch.inference_mode(); in-place
                # writes to them outside that mode raise, so build new tensors instead.
                t = wp.to_torch(mask).bool().view(1, -1, 1)
                self.hidden = self.hidden.masked_fill(t, 0.0)
                self.cell = self.cell.masked_fill(t, 0.0)
            else:
                wp.launch(
                    _zero_masked_3d_kernel,
                    dim=self.hidden.shape,
                    inputs=[self.hidden, mask],
                    device=self.hidden.device,
                )
                wp.launch(
                    _zero_masked_3d_kernel,
                    dim=self.cell.shape,
                    inputs=[self.cell, mask],
                    device=self.cell.device,
                )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        if "model_path" not in args:
            raise ValueError("DriveNeuralLSTM requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("DriveNeuralLSTM requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize the LSTM drive from a checkpoint file.

        Args:
            model_path: Path to the ``.onnx``, ``.pt2``, ``.pt``, or ``.pth``
                checkpoint.
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

        if self._is_torch_checkpoint:
            self.pos_scale = metadata.get("pos_scale", 1.0)
            self.vel_scale = metadata.get("vel_scale", 1.0)
            self.effort_scale = metadata.get("effort_scale", metadata.get("torque_scale", 1.0))

            lstm = getattr(self.network, "lstm", None)
            if lstm is not None and hasattr(lstm, "num_layers"):
                if not lstm.batch_first:
                    raise ValueError("network.lstm.batch_first must be True")
                if lstm.input_size != 2:
                    raise ValueError(f"network.lstm.input_size must be 2 (pos_error, vel); got {lstm.input_size}")
                if lstm.bidirectional:
                    raise ValueError("network.lstm must not be bidirectional")
                if getattr(lstm, "proj_size", 0) != 0:
                    raise ValueError(f"network.lstm.proj_size must be 0; got {lstm.proj_size}")

                self._num_layers = lstm.num_layers
                self._hidden_size = lstm.hidden_size
                for key, expected in (("num_layers", self._num_layers), ("hidden_size", self._hidden_size)):
                    if key in metadata and int(metadata[key]) != expected:
                        raise ValueError(
                            f"Metadata '{key}' in '{model_path}' is {metadata[key]}, "
                            f"but the network's LSTM has {key}={expected}"
                        )
            elif "num_layers" in metadata and "hidden_size" in metadata:
                self._num_layers = int(metadata["num_layers"])
                self._hidden_size = int(metadata["hidden_size"])
            else:
                raise ValueError(
                    f"Cannot determine the LSTM configuration for '{model_path}': the checkpoint "
                    f"does not expose an 'lstm' module (torch.nn.LSTM) and its metadata does not "
                    f"provide 'num_layers' and 'hidden_size'. Record both in the checkpoint "
                    f"metadata when exporting pt2 archives."
                )
        else:
            self.pos_scale = _parse_metadata_scale(metadata, "pos_scale", model_path)
            self.vel_scale = _parse_metadata_scale(metadata, "vel_scale", model_path)
            self.effort_scale = _parse_metadata_scale(metadata, "effort_scale", model_path, fallback_key="torque_scale")

            for key in (
                "input_name",
                "hidden_in_name",
                "cell_in_name",
                "output_name",
                "hidden_out_name",
                "cell_out_name",
                "num_layers",
                "hidden_size",
            ):
                if key not in metadata:
                    raise ValueError(f"ONNX metadata missing required key '{key}'")

            self._input_name = metadata["input_name"]
            self._hidden_in_name = metadata["hidden_in_name"]
            self._cell_in_name = metadata["cell_in_name"]
            self._output_name = metadata["output_name"]
            self._hidden_out_name = metadata["hidden_out_name"]
            self._cell_out_name = metadata["cell_out_name"]

            self._num_layers = int(metadata["num_layers"])
            self._hidden_size = int(metadata["hidden_size"])

        self._network = None
        self._device: wp.Device | None = None
        self._num_actuators = 0
        self._torch_input_indices: torch.Tensor | None = None
        self._torch_vel_indices: torch.Tensor | None = None
        self._torch_target_pos_indices: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._cell: torch.Tensor | None = None
        self._net_input: wp.array3d[float] | None = None
        self._next_hidden: wp.array3d[float] | None = None
        self._next_cell: wp.array3d[float] | None = None
        self._grad_seed: wp.array2d[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        self._device = device
        self._num_actuators = num_actuators

        if self._is_torch_checkpoint:
            import torch

            self._torch_device = torch.device(f"cuda:{device.ordinal}" if device.is_cuda else "cpu")
            self.network = self.network.to(self._torch_device)
            return

        runtime, _ = load_checkpoint(
            self.model_path,
            device=device,
            batch_size=num_actuators,
            input_batch_axes={
                self._input_name: 1,
                self._hidden_in_name: 1,
                self._cell_in_name: 1,
            },
            requires_grad=True,
        )
        self._network = runtime
        self.network = runtime

        out_shape = _runtime_shape(runtime, self._output_name)
        if out_shape != (num_actuators, 1):
            raise ValueError(
                f"DriveNeuralLSTM: ONNX output '{self._output_name}' has shape {out_shape}, "
                f"expected {(num_actuators, 1)} (one scalar effort per actuator)"
            )

        for name in (self._hidden_out_name, self._cell_out_name):
            state_shape = _runtime_shape(runtime, name)
            expected_state_shape = (self._num_layers, num_actuators, self._hidden_size)
            if tuple(state_shape) != expected_state_shape:
                raise ValueError(
                    f"DriveNeuralLSTM: ONNX output '{name}' has shape {tuple(state_shape)}, "
                    f"expected {expected_state_shape} (num_layers, num_actuators, hidden_size)"
                )

        self._net_input = wp.zeros((1, num_actuators, 2), dtype=wp.float32, device=device)
        self._net_input.requires_grad = True
        self._grad_seed = wp.full((num_actuators, 1), 1.0, dtype=wp.float32, device=device)
        self._next_hidden = wp.zeros(
            (self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device
        )
        self._next_cell = wp.zeros(
            (self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device
        )

        # Implicit path: per-step linearization packed as [tau0, a, b, q0, qd0] and the
        # per-slot scratch it is assembled from (see prepare_implicit).
        self._lin_params = wp.zeros((num_actuators, 5), dtype=wp.float32, device=device)
        self._q0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._qd0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tq0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tqd0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tau0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._dtau_dq = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._dtau_dqd = wp.zeros(num_actuators, dtype=wp.float32, device=device)

    def is_stateful(self) -> bool:
        return True

    def is_graphable(self) -> bool:
        return not self._is_torch_checkpoint

    #: The network enters the general implicit solve as a per-step-linearized
    #: in-kernel law (see :meth:`prepare_implicit`), like any other drive.
    evaluate_force = _linear_force

    def _implicit_supported(self) -> bool:
        return not self._is_torch_checkpoint and self._network is not None

    def bind_params(self) -> wp.array2d[float] | None:
        """Linearization pack ``[tau0, a, b, q0, qd0]``; ``None`` if implicit unsupported.

        The pack is allocated in :meth:`finalize` and rewritten in place each
        step by :meth:`prepare_implicit`. ``None`` for the Torch backend.
        """
        return self._lin_params if self._implicit_supported() else None

    def prepare_implicit(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        drive_state: DriveNeuralLSTM.State | None,
        dt: float,
        inv_mass: wp.array[float] | None = None,
        device: wp.Device | None = None,
    ) -> None:
        """Refresh the linearization of the network about the current state.

        One forward + autodiff backward at the current per-slot state, with the
        incoming hidden/cell state held fixed, gives ``tau0, d(tau)/dq,
        d(tau)/dqd``, packed as ``[tau0, a, b, q0, qd0]`` into :meth:`bind_params`. The
        forward also advances hidden/cell for :meth:`update_state`.
        """
        if drive_state is None:
            raise RuntimeError("Implicit DriveNeuralLSTM requires drive state (hidden/cell)")
        device = device or self._device
        n = self._num_actuators
        wp.launch(
            _gather_slot_state_kernel,
            dim=n,
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
            ],
            outputs=[self._q0, self._qd0, self._tq0, self._tqd0],
            device=device,
        )
        self._force_and_grad(
            self._q0,
            self._qd0,
            self._tq0,
            self._tqd0,
            drive_state.hidden,
            drive_state.cell,
            self._tau0,
            self._dtau_dq,
            self._dtau_dqd,
        )
        wp.launch(
            _assemble_linear_params_kernel,
            dim=n,
            inputs=[
                self._tau0,
                self._dtau_dq,
                self._dtau_dqd,
                self._q0,
                self._qd0,
                inv_mass,
                float(dt),
                IMPLICIT_JACOBIAN_MARGIN,
            ],
            outputs=[self._lin_params],
            device=device,
        )

    def _force_and_grad(self, q, qd, target_q, target_qd, hidden, cell, tau, dtau_dq, dtau_dqd) -> None:
        """One LSTM forward + autodiff backward at the given per-slot state.

        Writes the effort and its derivatives w.r.t. position and velocity, and
        captures the new hidden/cell state for :meth:`update_state`.
        """
        n = self._num_actuators
        device = self._device
        wp.launch(
            _assemble_scaled_input_3d_kernel,
            dim=n,
            inputs=[q, qd, target_q, target_qd, self.pos_scale, self.vel_scale],
            outputs=[self._net_input],
            device=device,
        )
        self._net_input.grad.zero_()
        tape = wp.Tape()
        with tape:
            out = self._network(
                {
                    self._input_name: self._net_input,
                    self._hidden_in_name: hidden,
                    self._cell_in_name: cell,
                }
            )
            effort = out[self._output_name]
        tape.backward(grads={effort: self._grad_seed})
        wp.launch(
            _lstm_output_grads_kernel,
            dim=n,
            inputs=[effort, self._net_input.grad, self.pos_scale, self.vel_scale, self.effort_scale],
            outputs=[tau, dtau_dq, dtau_dqd],
            device=device,
        )
        wp.copy(self._next_hidden, out[self._hidden_out_name].reshape((self._num_layers, n, self._hidden_size)))
        wp.copy(self._next_cell, out[self._cell_out_name].reshape((self._num_layers, n, self._hidden_size)))

    def state(self, num_actuators: int, device: wp.Device) -> DriveNeuralLSTM.State:
        if self._is_torch_checkpoint:
            import torch

            return DriveNeuralLSTM.State(
                hidden=torch.zeros(self._num_layers, num_actuators, self._hidden_size, device=self._torch_device),
                cell=torch.zeros(self._num_layers, num_actuators, self._hidden_size, device=self._torch_device),
            )
        return DriveNeuralLSTM.State(
            hidden=wp.zeros((self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device),
            cell=wp.zeros((self._num_layers, num_actuators, self._hidden_size), dtype=wp.float32, device=device),
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
        state: DriveNeuralLSTM.State,
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
                pos_indices,
                vel_indices,
                target_pos_indices,
                forces,
                state,
            )
            return

        wp.launch(
            _compute_inputs_kernel,
            dim=n,
            inputs=[
                target_pos,
                positions,
                velocities,
                pos_indices,
                vel_indices,
                target_pos_indices,
                self.pos_scale,
                self.vel_scale,
                self._net_input,
            ],
            device=device,
        )

        out = self._network(
            {
                self._input_name: self._net_input,
                self._hidden_in_name: state.hidden,
                self._cell_in_name: state.cell,
            }
        )
        effort = out[self._output_name]
        hidden_new = out[self._hidden_out_name]
        cell_new = out[self._cell_out_name]

        wp.copy(self._next_hidden, hidden_new.reshape((self._num_layers, n, self._hidden_size)))
        wp.copy(self._next_cell, cell_new.reshape((self._num_layers, n, self._hidden_size)))

        wp.launch(
            _scale_effort_to_forces_kernel,
            dim=len(forces),
            inputs=[effort, forces, self.effort_scale, 1],
            device=device,
        )

    def update_state(
        self,
        current_state: DriveNeuralLSTM.State,
        next_state: DriveNeuralLSTM.State,
    ) -> None:
        if next_state is None:
            return
        if self._is_torch_checkpoint:
            next_state.hidden = self._hidden
            next_state.cell = self._cell
            return
        wp.copy(next_state.hidden, self._next_hidden)
        wp.copy(next_state.cell, self._next_cell)

    def _compute_torch(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        forces: wp.array[float],
        state: DriveNeuralLSTM.State,
    ) -> None:
        import torch

        if self._torch_input_indices is None:
            self._torch_input_indices = wp.to_torch(pos_indices).long()
            self._torch_vel_indices = wp.to_torch(vel_indices).long()
            self._torch_target_pos_indices = wp.to_torch(target_pos_indices).long()

        current_pos = wp.to_torch(positions)
        current_vel = wp.to_torch(velocities)
        target_p = wp.to_torch(target_pos)

        pos_error = target_p[self._torch_target_pos_indices] - current_pos[self._torch_input_indices]
        vel = current_vel[self._torch_vel_indices]

        net_input = torch.stack([pos_error * self.pos_scale, vel * self.vel_scale], dim=1).unsqueeze(1)

        with torch.inference_mode():
            effort, (self._hidden, self._cell) = self.network(
                net_input,
                (state.hidden, state.cell),
            )

        effort = effort.reshape(len(forces)) * self.effort_scale
        effort_wp = wp.from_torch(effort.contiguous(), dtype=wp.float32)
        wp.copy(forces, effort_wp)
