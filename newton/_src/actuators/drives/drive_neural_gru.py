# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import keyword
import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import warp as wp

from ...utils.onnx import require_onnx
from ..utils import load_checkpoint, load_metadata
from ._linearization import (
    IMPLICIT_JACOBIAN_MARGIN,
    _assemble_linear_params_kernel,
    _gather_slot_state_kernel,
    _linear_force,
)
from .base import DriveBase

_FEATURE_POSITION = 0
_FEATURE_TARGET_POSITION = 1
_FEATURE_POSITION_ERROR = 2
_FEATURE_VELOCITY = 3
_FEATURE_TARGET_VELOCITY = 4
_FEATURE_VELOCITY_ERROR = 5
_FEATURE_CUSTOM_INPUT = 6
_INPUT_FEATURE_CODES = {
    "position": _FEATURE_POSITION,
    "target_position": _FEATURE_TARGET_POSITION,
    "position_error": _FEATURE_POSITION_ERROR,
    "velocity": _FEATURE_VELOCITY,
    "target_velocity": _FEATURE_TARGET_VELOCITY,
    "velocity_error": _FEATURE_VELOCITY_ERROR,
}


@wp.kernel
def _compute_inputs_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    custom_input: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    feature_codes: wp.array[wp.int32],
    means: wp.array[float],
    stds: wp.array[float],
    output: wp.array3d[float],
):
    i, column = wp.tid()
    position = positions[pos_indices[i]]
    velocity = velocities[vel_indices[i]]
    feature = feature_codes[column]
    value = 0.0
    if feature == _FEATURE_POSITION:
        value = position
    elif feature == _FEATURE_TARGET_POSITION:
        value = target_pos[target_pos_indices[i]]
    elif feature == _FEATURE_POSITION_ERROR:
        value = target_pos[target_pos_indices[i]] - position
    elif feature == _FEATURE_VELOCITY:
        value = velocity
    elif feature == _FEATURE_TARGET_VELOCITY:
        value = target_vel[target_vel_indices[i]]
    elif feature == _FEATURE_VELOCITY_ERROR:
        value = target_vel[target_vel_indices[i]] - velocity
    elif feature == _FEATURE_CUSTOM_INPUT:
        value = custom_input[vel_indices[i]]
    output[0, i, column] = (value - means[column]) / stds[column]


@wp.kernel
def _write_effort_kernel(
    network_output: wp.array2d[float],
    effort_mean: float,
    effort_std: float,
    forces: wp.array[float],
):
    i = wp.tid()
    forces[i] = network_output[i, 0] * effort_std + effort_mean


@wp.kernel(enable_backward=False)
def _effort_and_slopes_kernel(
    network_output: wp.array2d[float],
    input_grad: wp.array3d[float],
    feature_codes: wp.array[wp.int32],
    stds: wp.array[float],
    effort_mean: float,
    effort_std: float,
    tau: wp.array[float],
    dtau_dq: wp.array[float],
    dtau_dqd: wp.array[float],
):
    """Effort and its state derivatives from the network output and input gradients.

    Each input column is ``(feature - mean) / std``; ``position`` and
    ``velocity`` enter with ``+1``, the ``*_error`` features with ``-1``, and
    the remaining features do not depend on the joint state.
    """
    i = wp.tid()
    tau[i] = network_output[i, 0] * effort_std + effort_mean
    dq = float(0.0)
    dqd = float(0.0)
    for column in range(input_grad.shape[2]):
        grad = input_grad[0, i, column] / stds[column]
        feature = feature_codes[column]
        if feature == _FEATURE_POSITION:
            dq += grad
        elif feature == _FEATURE_POSITION_ERROR:
            dq -= grad
        elif feature == _FEATURE_VELOCITY:
            dqd += grad
        elif feature == _FEATURE_VELOCITY_ERROR:
            dqd -= grad
    dtau_dq[i] = dq * effort_std
    dtau_dqd[i] = dqd * effort_std


@wp.kernel
def _zero_masked_hidden_kernel(hidden: wp.array3d[float], mask: wp.array[wp.bool]):
    layer, actuator, feature = wp.tid()
    if mask[actuator]:
        hidden[layer, actuator, feature] = 0.0


@dataclass(frozen=True)
class _GRUGraphInterface:
    input_name: str
    input_size: int
    hidden_in_names: tuple[str, ...]
    hidden_out_names: tuple[str, ...]
    hidden_size: int
    output_name: str


def _parse_input_feature_keys(metadata: dict[str, Any], model_path: str) -> tuple[str, ...]:
    """Return validated, ordered input feature names from checkpoint metadata."""
    input_columns = metadata.get("input_columns")
    if (
        not isinstance(input_columns, list)
        or not input_columns
        or not all(isinstance(name, str) for name in input_columns)
    ):
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' input_columns must be a non-empty list of strings; "
            f"got {input_columns!r}"
        )
    if len(set(input_columns)) != len(input_columns):
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' input_columns must not contain duplicates")
    custom = _parse_custom_input_names(metadata, model_path)
    unselected_custom = sorted(set(custom).difference(input_columns))
    if unselected_custom:
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' custom_inputs must name columns in input_columns: "
            f"{', '.join(unselected_custom)}"
        )
    unsupported = sorted(set(input_columns).difference(_INPUT_FEATURE_CODES).difference(custom))
    if unsupported:
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' has unsupported input_columns: {', '.join(unsupported)}. "
            f"Name caller-supplied columns in the checkpoint's custom_inputs metadata."
        )
    return tuple(input_columns)


def _custom_inputs(selected: tuple[str, ...]) -> tuple[str, ...]:
    """Return the selected columns that are not built-in features."""
    return tuple(name for name in selected if name not in _INPUT_FEATURE_CODES)


def _parse_custom_input_names(metadata: dict[str, Any], model_path: str) -> tuple[str, ...]:
    """Return the input columns the caller supplies, from ``custom_inputs`` metadata."""
    names = metadata.get("custom_inputs", [])
    if not isinstance(names, list) or not all(
        isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name) for name in names
    ):
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' custom_inputs must be a list of valid, non-keyword "
            f"identifiers; got {names!r}"
        )
    if len(names) > 1:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' supports at most one custom input; got {names!r}")
    clashing = sorted(set(names).intersection(_INPUT_FEATURE_CODES))
    if clashing:
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' custom_inputs may not reuse built-in feature names: "
            f"{', '.join(clashing)}"
        )
    return tuple(names)


def _read_graph_interface(model_path: str) -> _GRUGraphInterface:
    """Find the network input, per-layer hidden state, and effort output names."""
    if not os.path.isfile(model_path):
        raise ValueError(f"DriveNeuralGRU checkpoint does not exist or is not a local file: '{model_path}'")
    if os.path.splitext(model_path)[1].lower() != ".onnx":
        raise ValueError(f"DriveNeuralGRU requires an ONNX checkpoint; got '{model_path}'")

    onnx = require_onnx()
    graph = onnx.load(model_path, load_external_data=False).graph
    initializers = {item.name for item in graph.initializer}
    graph_inputs = [value for value in graph.input if value.name not in initializers]
    graph_input_names = {value.name for value in graph_inputs}
    graph_output_names = [value.name for value in graph.output]

    gru_nodes = [node for node in graph.node if node.op_type == "GRU"]
    if not gru_nodes:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' contains no GRU node")

    hidden_in_names = []
    hidden_out_names = []
    hidden_sizes = set()
    for node in gru_nodes:
        hidden_in = node.input[5] if len(node.input) > 5 else ""
        hidden_out = node.output[1] if len(node.output) > 1 else ""
        if hidden_in not in graph_input_names:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' requires an external initial_h for every GRU")
        if hidden_out not in graph_output_names:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' requires every GRU Y_h as a graph output")
        hidden_in_names.append(hidden_in)
        hidden_out_names.append(hidden_out)
        for attribute in node.attribute:
            if attribute.name == "hidden_size":
                hidden_sizes.add(int(attribute.i))
    if len(set(hidden_in_names)) != len(hidden_in_names):
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' requires a distinct initial_h for every GRU")
    if len(hidden_sizes) != 1:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' stacked GRU layers must share one hidden size")

    data_inputs = [value for value in graph_inputs if value.name not in hidden_in_names]
    effort_outputs = [name for name in graph_output_names if name not in hidden_out_names]
    if len(data_inputs) != 1 or len(effort_outputs) != 1:
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' must have one network input besides the GRU initial_h "
            f"inputs and one effort output besides the GRU Y_h outputs"
        )
    input_dims = data_inputs[0].type.tensor_type.shape.dim
    if len(input_dims) != 3 or not input_dims[2].HasField("dim_value"):
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' network input must have shape [1, batch, input_size]"
        )

    return _GRUGraphInterface(
        input_name=data_inputs[0].name,
        input_size=int(input_dims[2].dim_value),
        hidden_in_names=tuple(hidden_in_names),
        hidden_out_names=tuple(hidden_out_names),
        hidden_size=hidden_sizes.pop(),
        output_name=effort_outputs[0],
    )


class DriveNeuralGRU(DriveBase):
    """Stateful gated recurrent unit (GRU) actuator drive using Warp-NN.

    The network and its weights are read from an ONNX checkpoint and evaluated
    with Warp-NN. Hidden state is kept across timesteps through the
    double-buffered :class:`Actuator.State` lifecycle. The implicit effort mode
    is supported through a per-step linearization of the network.

    **Checkpoint metadata.** A JSON block embedded in the ONNX file must
    provide:

    .. list-table::
        :header-rows: 1
        :widths: 22 78

        * - Key
          - Meaning
        * - ``input_columns``
          - Ordered list of the network's input feature names, one per input
            column. Non-empty and duplicate-free.
        * - ``custom_inputs``
          - Optional list naming at most one column of ``input_columns`` that
            the application supplies. The name must be a valid identifier and
            must not be a built-in feature name.
        * - ``normalization``
          - ``inputs.mean`` and ``inputs.std`` per name in ``input_columns``,
            and ``targets.mean.torque`` and ``targets.std.torque``.
        * - ``sample_dt_s``
          - Timestep [s] the network was trained at. The runtime actuator
            timestep must match it.

    **Input features.** Each name in ``input_columns`` is one of the built-in
    features below, or the column named in ``custom_inputs``. Any other name is
    rejected.

    The drive computes the built-in features from the State and Control arrays
    every drive receives:

    * ``position``
    * ``target_position``
    * ``position_error`` — ``target_position - position``, without angle wrapping
    * ``velocity``
    * ``target_velocity``
    * ``velocity_error`` — ``target_velocity - velocity``

    The application supplies the custom column. It is declared through
    :attr:`DriveBase.custom_inputs` and read from ``sim_control``; see
    :ref:`custom-drive-inputs`.

    Normalization is applied per column as ``(value - mean) / std``. The
    scalar output is converted to physical torque [N or N·m] as
    ``output * torque_std + torque_mean``. Any output-head activation or scale
    is part of the exported network and is not applied again.

    **ONNX graph.** The network is evaluated by the Warp-NN ONNX runtime. It
    contains one or more ``GRU`` nodes that share one hidden size. It takes an
    input of shape ``[1, N, F]`` and, for every ``GRU`` node, an ``initial_h``
    graph input of shape ``[1, N, hidden_size]``. It returns one scalar effort
    of shape ``[N, 1]`` and, for every ``GRU`` node, its ``Y_h`` as a graph
    output.

    .. warning::

        :class:`DriveNeuralGRU` is experimental. Its checkpoint metadata
        contract, the set of supported ONNX graph shapes, and the custom
        inputs it declares may change without the normal deprecation period.
    """

    SHARED_PARAMS: ClassVar[set[str]] = {"model_path"}

    @dataclass
    class State(DriveBase.State):
        """GRU hidden state."""

        hidden: wp.array3d[float] | None = None
        """Hidden state, shape [layer_count, actuator_count, hidden_size]."""

        def reset(self, mask: wp.array[wp.bool] | None = None) -> None:
            """Reset all or selected actuator hidden-state lanes.

            Args:
                mask: Boolean mask with shape [actuator_count]. ``True``
                    entries are reset. If ``None``, all lanes are reset.
            """
            if self.hidden is None:
                raise ValueError("DriveNeuralGRU.State has no hidden array to reset")
            if mask is None:
                self.hidden.zero_()
                return
            if len(mask) < self.hidden.shape[1]:
                raise ValueError(f"mask has length {len(mask)}; expected at least {self.hidden.shape[1]}.")
            wp.launch(
                _zero_masked_hidden_kernel,
                dim=self.hidden.shape,
                inputs=[self.hidden, mask],
                device=self.hidden.device,
            )

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Validate and return the shared checkpoint argument.

        Args:
            args: User-provided drive arguments.

        Returns:
            Resolved arguments containing ``model_path``.
        """
        if "model_path" not in args:
            raise ValueError("DriveNeuralGRU requires 'model_path' argument")
        model_path = args["model_path"]
        if not model_path:
            raise ValueError("DriveNeuralGRU requires a non-empty 'model_path'")
        return {"model_path": model_path}

    def __init__(self, model_path: str):
        """Initialize the drive from an ONNX GRU network.

        Args:
            model_path: Local path to an ONNX checkpoint.
        """
        self.model_path = os.fspath(model_path)
        """Local path to the ONNX checkpoint."""
        self._graph = _read_graph_interface(self.model_path)
        metadata = load_metadata(self.model_path)

        normalization = metadata["normalization"]
        input_normalization = normalization["inputs"]
        self._input_feature_keys = _parse_input_feature_keys(metadata, self.model_path)
        self.custom_inputs = _custom_inputs(self._input_feature_keys)
        if self._graph.input_size != len(self._input_feature_keys):
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' GRU input size "
                f"{self._graph.input_size} does not match input_columns "
                f"{len(self._input_feature_keys)}"
            )
        try:
            self._input_means = tuple(float(input_normalization["mean"][name]) for name in self._input_feature_keys)
            self._input_stds = tuple(float(input_normalization["std"][name]) for name in self._input_feature_keys)
            self._effort_mean = float(normalization["targets"]["mean"]["torque"])
            self._effort_std = float(normalization["targets"]["std"]["torque"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' has invalid normalization metadata"
            ) from exc
        for name, mean, std in zip(self._input_feature_keys, self._input_means, self._input_stds, strict=True):
            if not math.isfinite(mean):
                raise ValueError(
                    f"DriveNeuralGRU checkpoint '{self.model_path}' normalization.inputs.mean.{name} "
                    f"must be finite; got {mean}"
                )
            if not math.isfinite(std) or std <= 0.0:
                raise ValueError(
                    f"DriveNeuralGRU checkpoint '{self.model_path}' normalization.inputs.std.{name} "
                    f"must be finite and positive; got {std}"
                )
        if not math.isfinite(self._effort_mean):
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' normalization.targets.mean.torque "
                f"must be finite; got {self._effort_mean}"
            )
        if not math.isfinite(self._effort_std) or self._effort_std <= 0.0:
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' normalization.targets.std.torque "
                f"must be finite and positive; got {self._effort_std}"
            )
        self._sample_dt_s = float(metadata["sample_dt_s"])

        self._device: wp.Device | None = None
        self._num_actuators = 0
        self._input_means_wp: wp.array[float] | None = None
        self._input_stds_wp: wp.array[float] | None = None
        self._input_feature_codes_wp: wp.array[wp.int32] | None = None
        self._net_input: wp.array3d[float] | None = None
        self._network: Any = None
        self._next_hidden: list[wp.array3d[float]] | None = None
        self._grad_seed: wp.array2d[float] | None = None
        self._lin_params: wp.array2d[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Load the Warp-NN ONNX runtime and inference buffers.

        Args:
            device: Device on which inference runs.
            num_actuators: Number of vectorized actuator lanes.
        """
        if self._device is not None and (self._device != device or self._num_actuators != num_actuators):
            raise RuntimeError(
                "DriveNeuralGRU is already finalized for a different actuator; construct one drive per actuator."
            )
        graph = self._graph
        try:
            network, _ = load_checkpoint(
                self.model_path,
                device=device,
                input_batch_axes={graph.input_name: 1, **dict.fromkeys(graph.hidden_in_names, 1)},
                requires_grad=True,
            )
        except ImportError as exc:
            raise ImportError(
                "DriveNeuralGRU requires Warp-NN and ONNX. Install them with `pip install newton[onnx]`."
            ) from exc

        self._device = device
        self._num_actuators = num_actuators
        self._network = network
        self._input_means_wp = wp.array(self._input_means, dtype=wp.float32, device=device)
        self._input_stds_wp = wp.array(self._input_stds, dtype=wp.float32, device=device)
        self._input_feature_codes_wp = wp.array(
            [_INPUT_FEATURE_CODES.get(name, _FEATURE_CUSTOM_INPUT) for name in self._input_feature_keys],
            dtype=wp.int32,
            device=device,
        )
        self._net_input = wp.zeros(
            (1, num_actuators, len(self._input_feature_keys)), dtype=wp.float32, device=device, requires_grad=True
        )
        hidden_shape = (len(graph.hidden_in_names), num_actuators, graph.hidden_size)
        self._grad_seed = wp.ones((num_actuators, 1), dtype=wp.float32, device=device)
        self._lin_params = wp.zeros((num_actuators, 5), dtype=wp.float32, device=device)
        self._q0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._qd0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tq0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tqd0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._tau0 = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._dtau_dq = wp.zeros(num_actuators, dtype=wp.float32, device=device)
        self._dtau_dqd = wp.zeros(num_actuators, dtype=wp.float32, device=device)

        # Populate the runtime's shape caches before optional graph capture.
        outputs = network(self._network_inputs(wp.zeros(hidden_shape, dtype=wp.float32, device=device)))
        effort_shape = tuple(outputs[graph.output_name].shape)
        if effort_shape != (num_actuators, 1):
            raise ValueError(
                f"DriveNeuralGRU: ONNX output '{graph.output_name}' has shape {effort_shape}, "
                f"expected {(num_actuators, 1)} (one scalar effort per actuator)"
            )
        for name in graph.hidden_out_names:
            state_shape = tuple(outputs[name].shape)
            if state_shape != (1, num_actuators, graph.hidden_size):
                raise ValueError(
                    f"DriveNeuralGRU: ONNX output '{name}' has shape {state_shape}, "
                    f"expected {(1, num_actuators, graph.hidden_size)}"
                )

    def is_stateful(self) -> bool:
        """Return ``True`` because the GRU maintains hidden state."""
        return True

    def is_graphable(self) -> bool:
        """Return ``True`` because inference uses graph-capturable Warp operations."""
        return True

    #: The network enters the general implicit solve as a per-step-linearized
    #: in-kernel law (see :meth:`prepare_implicit`), like any other drive.
    evaluate_force = _linear_force

    def bind_params(self) -> wp.array2d[float] | None:
        """Linearization pack ``[tau0, a, b, q0, qd0]``; ``None`` before :meth:`finalize`.

        The pack is allocated in :meth:`finalize` and rewritten in place each
        step by :meth:`prepare_implicit`.
        """
        return self._lin_params

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
        drive_state: DriveNeuralGRU.State | None,
        dt: float,
        inv_mass: wp.array[float] | None = None,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
    ) -> None:
        """Refresh the linearization of the network about the current state.

        One forward + autodiff backward at the current per-slot state, with the
        incoming hidden state held fixed, gives ``tau0, d(tau)/dq,
        d(tau)/dqd``, packed as ``[tau0, a, b, q0, qd0]`` into
        :meth:`bind_params`. The forward also advances the hidden state for
        :meth:`update_state`.
        """
        if inv_mass is None:
            raise ValueError("DriveNeuralGRU.prepare_implicit requires inv_mass (the per-slot response)")
        device = device or self._device
        n = self._num_actuators
        self._compute_net_input(
            positions,
            velocities,
            target_pos,
            target_vel,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            drive_state,
            dt,
            device,
            custom_inputs,
        )
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
        tape = wp.Tape()
        with tape:
            output = self._forward(drive_state.hidden)
        tape.backward(grads={output: self._grad_seed})
        wp.launch(
            _effort_and_slopes_kernel,
            dim=n,
            inputs=[
                output,
                self._net_input.grad,
                self._input_feature_codes_wp,
                self._input_stds_wp,
                self._effort_mean,
                self._effort_std,
            ],
            outputs=[self._tau0, self._dtau_dq, self._dtau_dqd],
            device=device,
        )
        tape.zero()
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

    def state(self, num_actuators: int, device: wp.Device) -> DriveNeuralGRU.State:
        """Create zero-initialized hidden state.

        Args:
            num_actuators: Number of vectorized actuator lanes.
            device: Device on which to allocate state.

        Returns:
            New GRU drive state.
        """
        if self._device is None:
            raise RuntimeError("DriveNeuralGRU must be finalized before creating state")
        if device != self._device:
            raise ValueError(f"GRU state device {device} must match drive device {self._device}")
        return DriveNeuralGRU.State(
            hidden=wp.zeros(
                (len(self._graph.hidden_in_names), num_actuators, self._graph.hidden_size),
                dtype=wp.float32,
                device=device,
            )
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
        state: DriveNeuralGRU.State,
        dt: float,
        device: wp.Device | None = None,
        *,
        custom_inputs: dict[str, Any] | None = None,
    ) -> None:
        """Evaluate one GRU sample and write physical effort."""
        runtime_device = device or self._device
        self._compute_net_input(
            positions,
            velocities,
            target_pos,
            target_vel,
            pos_indices,
            vel_indices,
            target_pos_indices,
            target_vel_indices,
            state,
            dt,
            runtime_device,
            custom_inputs,
        )
        output = self._forward(state.hidden)
        wp.launch(
            _write_effort_kernel,
            dim=self._num_actuators,
            inputs=[
                output,
                self._effort_mean,
                self._effort_std,
            ],
            outputs=[forces],
            device=runtime_device,
        )

    def _compute_net_input(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        target_pos: wp.array[float],
        target_vel: wp.array[float],
        pos_indices: wp.array[wp.uint32],
        vel_indices: wp.array[wp.uint32],
        target_pos_indices: wp.array[wp.uint32],
        target_vel_indices: wp.array[wp.uint32],
        state: DriveNeuralGRU.State | None,
        dt: float,
        device: wp.Device,
        custom_inputs: dict[str, Any] | None,
    ) -> None:
        """Validate one step's inputs and write the normalized network input."""
        self._next_hidden = None
        custom_input = None
        if self.custom_inputs:
            name = self.custom_inputs[0]
            custom_input = (custom_inputs or {}).get(name)
            if custom_input is None:
                raise RuntimeError(
                    f"DriveNeuralGRU input_columns includes '{name}', but no array was supplied. Pass a "
                    f"sim_control carrying '{name}' with shape (model.joint_dof_count,)."
                )
            if not isinstance(custom_input, (wp.array, wp.indexedarray, wp.fabricarray)):
                raise ValueError(
                    f"DriveNeuralGRU custom input 'sim_control.{name}' must be a Warp array; "
                    f"got {type(custom_input).__name__}."
                )
            if len(custom_input) != len(velocities):
                raise ValueError(
                    f"DriveNeuralGRU custom input 'sim_control.{name}' has length {len(custom_input)}; "
                    f"expected {len(velocities)}."
                )
        if state is None or state.hidden is None:
            raise ValueError("DriveNeuralGRU requires a current drive state with hidden data")

        try:
            runtime_dt = float(dt) if not isinstance(dt, bool) else math.nan
        except (TypeError, ValueError):
            runtime_dt = math.nan
        if (
            not math.isfinite(runtime_dt)
            or runtime_dt <= 0.0
            or not math.isclose(runtime_dt, self._sample_dt_s, rel_tol=1.0e-5, abs_tol=1.0e-9)
        ):
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' was trained for "
                f"sample_dt_s={self._sample_dt_s}; received actuator evaluation dt={dt!r}"
            )

        wp.launch(
            _compute_inputs_kernel,
            dim=(self._num_actuators, len(self._input_feature_keys)),
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                custom_input,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self._input_feature_codes_wp,
                self._input_means_wp,
                self._input_stds_wp,
            ],
            outputs=[self._net_input],
            device=device,
        )

    def _network_inputs(self, hidden: wp.array3d[float]) -> dict[str, wp.array]:
        n = self._num_actuators
        inputs = {self._graph.input_name: self._net_input}
        for layer_index, name in enumerate(self._graph.hidden_in_names):
            inputs[name] = hidden[layer_index].reshape((1, n, self._graph.hidden_size))
        return inputs

    def _forward(self, hidden: wp.array3d[float]) -> wp.array2d[float]:
        """Run the network on the current input and record the next hidden state.

        Returns the network output before effort denormalization.
        """
        outputs = self._network(self._network_inputs(hidden))
        self._next_hidden = [outputs[name] for name in self._graph.hidden_out_names]
        return outputs[self._graph.output_name]

    def update_state(
        self,
        current_state: DriveNeuralGRU.State,
        next_state: DriveNeuralGRU.State,
    ) -> None:
        """Copy the most recent GRU hidden output into the next state.

        Args:
            current_state: Current state, retained for the common drive API.
            next_state: State receiving the next hidden values.
        """
        if self._next_hidden is None:
            raise RuntimeError("DriveNeuralGRU has no next hidden state; compute must run before update_state")
        if next_state is None or next_state.hidden is None:
            raise ValueError("DriveNeuralGRU requires a next drive state")
        n = self._num_actuators
        for layer_index, layer_hidden in enumerate(self._next_hidden):
            wp.copy(next_state.hidden[layer_index], layer_hidden.reshape((n, self._graph.hidden_size)))
        self._next_hidden = None
