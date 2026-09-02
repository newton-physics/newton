# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from ..utils import _require_onnx, load_metadata
from .base import DriveBase

_FEATURE_POSITION = 0
_FEATURE_POSITION_ERROR = 1
_FEATURE_VELOCITY = 2
_FEATURE_TARGET_VELOCITY = 3
_FEATURE_VELOCITY_ERROR = 4
_FEATURE_DYNAMIC_BIAS = 5
_INPUT_FEATURE_CODES = {
    "position": _FEATURE_POSITION,
    "position_error": _FEATURE_POSITION_ERROR,
    "velocity": _FEATURE_VELOCITY,
    "target_velocity": _FEATURE_TARGET_VELOCITY,
    "velocity_error": _FEATURE_VELOCITY_ERROR,
    "dynamic_bias": _FEATURE_DYNAMIC_BIAS,
}
_DYNAMIC_BIAS_CONTROL_ATTRIBUTE = "actuator:dynamic_bias"


@wp.kernel
def _assemble_input_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    target_pos: wp.array[float],
    target_vel: wp.array[float],
    bias_force: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    target_pos_indices: wp.array[wp.uint32],
    target_vel_indices: wp.array[wp.uint32],
    feature_codes: wp.array[wp.int32],
    means: wp.array[float],
    stds: wp.array[float],
    output: wp.array2d[float],
):
    i, column = wp.tid()
    position = positions[pos_indices[i]]
    velocity = velocities[vel_indices[i]]
    feature = feature_codes[column]
    value = 0.0
    if feature == _FEATURE_POSITION:
        value = position
    elif feature == _FEATURE_POSITION_ERROR:
        value = target_pos[target_pos_indices[i]] - position
    elif feature == _FEATURE_VELOCITY:
        value = velocity
    elif feature == _FEATURE_TARGET_VELOCITY:
        value = target_vel[target_vel_indices[i]]
    elif feature == _FEATURE_VELOCITY_ERROR:
        value = target_vel[target_vel_indices[i]] - velocity
    elif feature == _FEATURE_DYNAMIC_BIAS:
        value = bias_force[vel_indices[i]]
    output[i, column] = (value - means[column]) / stds[column]


@wp.kernel
def _write_effort_kernel(
    network_output: wp.array2d[float],
    network_scale: float,
    effort_mean: float,
    effort_std: float,
    forces: wp.array[float],
):
    i = wp.tid()
    forces[i] = network_output[i, 0] * network_scale * effort_std + effort_mean


@wp.kernel
def _zero_masked_hidden_kernel(hidden: wp.array3d[float], mask: wp.array[wp.bool]):
    layer, actuator, feature = wp.tid()
    if mask[actuator]:
        hidden[layer, actuator, feature] = 0.0


@dataclass(frozen=True)
class _GRULayerDescription:
    input_size: int
    hidden_size: int
    weight_ih: np.ndarray
    weight_hh: np.ndarray
    bias_ih: np.ndarray | None
    bias_hh: np.ndarray | None


@dataclass(frozen=True)
class _GRUNetworkDescription:
    layers: tuple[_GRULayerDescription, ...]
    head_weight: np.ndarray
    head_bias: np.ndarray
    apply_tanh: bool
    output_scale: float


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
    unsupported = sorted(set(input_columns).difference(_INPUT_FEATURE_CODES))
    if unsupported:
        raise ValueError(
            f"DriveNeuralGRU checkpoint '{model_path}' has unsupported input_columns: {', '.join(unsupported)}"
        )
    return tuple(input_columns)


def _node_attributes(onnx, node) -> dict[str, Any]:
    return {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}


def _reorder_onnx_gru_gates(values: np.ndarray, hidden_size: int) -> np.ndarray:
    """Convert ONNX ``(z, r, h)`` gate order to Warp-NN ``(r, z, n)``."""
    return np.concatenate(
        (values[hidden_size : 2 * hidden_size], values[:hidden_size], values[2 * hidden_size :]), axis=0
    )


def _load_network_description(model_path: str) -> _GRUNetworkDescription:
    """Read a supported GRU, scalar output head, and weights from ONNX."""
    if not os.path.isfile(model_path):
        raise ValueError(f"DriveNeuralGRU checkpoint does not exist or is not a local file: '{model_path}'")
    if os.path.splitext(model_path)[1].lower() != ".onnx":
        raise ValueError(f"DriveNeuralGRU requires an ONNX checkpoint; got '{model_path}'")

    onnx = _require_onnx()
    model = onnx.load(model_path)
    initializers = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
    constants: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            attributes = _node_attributes(onnx, node)
            if "value" in attributes:
                constants[node.output[0]] = onnx.numpy_helper.to_array(attributes["value"])

    gru_nodes = [node for node in model.graph.node if node.op_type == "GRU"]
    if not gru_nodes:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' contains no GRU node")

    layers: list[_GRULayerDescription] = []
    hidden_size: int | None = None
    for node in gru_nodes:
        attributes = _node_attributes(onnx, node)
        direction = attributes.get("direction", b"forward")
        if direction not in ("forward", b"forward"):
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' supports only forward GRU nodes")
        if int(attributes.get("layout", 0)) != 0:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' requires GRU layout=0")
        if int(attributes.get("linear_before_reset", 0)) != 1:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' requires GRU linear_before_reset=1")
        if len(node.input) < 3 or node.input[1] not in initializers or node.input[2] not in initializers:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' must embed GRU weights")

        weight_ih_onnx = np.asarray(initializers[node.input[1]], dtype=np.float32)
        weight_hh_onnx = np.asarray(initializers[node.input[2]], dtype=np.float32)
        if weight_ih_onnx.ndim != 3 or weight_ih_onnx.shape[0] != 1:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' GRU input weights have invalid shape")
        layer_hidden_size = int(attributes.get("hidden_size", weight_ih_onnx.shape[1] // 3))
        input_size = int(weight_ih_onnx.shape[2])
        if weight_ih_onnx.shape != (1, 3 * layer_hidden_size, input_size):
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' GRU input weights have invalid shape")
        if weight_hh_onnx.shape != (1, 3 * layer_hidden_size, layer_hidden_size):
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' GRU hidden weights have invalid shape")
        if hidden_size is not None and (layer_hidden_size != hidden_size or input_size != hidden_size):
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' stacked GRU layers must share one hidden size")
        hidden_size = layer_hidden_size

        bias_ih = None
        bias_hh = None
        if len(node.input) > 3 and node.input[3]:
            if node.input[3] not in initializers:
                raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' must embed GRU biases")
            bias = np.asarray(initializers[node.input[3]], dtype=np.float32)
            if bias.shape != (1, 6 * layer_hidden_size):
                raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' GRU biases have invalid shape")
            bias = bias.reshape(6 * layer_hidden_size)
            bias_ih = _reorder_onnx_gru_gates(bias[: 3 * layer_hidden_size], layer_hidden_size).reshape(-1, 1)
            bias_hh = _reorder_onnx_gru_gates(bias[3 * layer_hidden_size :], layer_hidden_size).reshape(-1, 1)

        layers.append(
            _GRULayerDescription(
                input_size=input_size,
                hidden_size=layer_hidden_size,
                weight_ih=_reorder_onnx_gru_gates(weight_ih_onnx[0], layer_hidden_size),
                weight_hh=_reorder_onnx_gru_gates(weight_hh_onnx[0], layer_hidden_size),
                bias_ih=bias_ih,
                bias_hh=bias_hh,
            )
        )

    gemm_nodes = [node for node in model.graph.node if node.op_type == "Gemm"]
    if len(gemm_nodes) != 1:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' must contain one scalar Gemm output head")
    head = gemm_nodes[0]
    attributes = _node_attributes(onnx, head)
    if (
        int(attributes.get("transA", 0)) != 0
        or int(attributes.get("transB", 0)) != 1
        or float(attributes.get("alpha", 1.0)) != 1.0
        or float(attributes.get("beta", 1.0)) != 1.0
    ):
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' has an unsupported Gemm output head")
    if len(head.input) < 3 or head.input[1] not in initializers or head.input[2] not in initializers:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' must embed output-head weights and bias")
    head_weight = np.asarray(initializers[head.input[1]], dtype=np.float32)
    head_bias = np.asarray(initializers[head.input[2]], dtype=np.float32)
    if head_weight.shape != (1, hidden_size) or head_bias.shape != (1,):
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' output head must produce one scalar")

    consumers: dict[str, list[Any]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    value_name = head.output[0]
    apply_tanh = False
    output_scale = 1.0
    while value_name in consumers:
        next_nodes = consumers[value_name]
        if len(next_nodes) != 1:
            raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' output head must have one linear path")
        node = next_nodes[0]
        if node.op_type == "Identity":
            value_name = node.output[0]
        elif node.op_type == "Tanh" and not apply_tanh:
            apply_tanh = True
            value_name = node.output[0]
        elif node.op_type == "Mul" and output_scale == 1.0:
            scale_names = [name for name in node.input if name != value_name]
            if len(scale_names) != 1:
                raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' has an invalid output scale")
            scale = initializers.get(scale_names[0], constants.get(scale_names[0]))
            if scale is None or np.asarray(scale).size != 1:
                raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' output scale must be scalar")
            output_scale = float(np.asarray(scale).reshape(-1)[0])
            if not math.isfinite(output_scale):
                raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' output scale must be finite")
            value_name = node.output[0]
        else:
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{model_path}' has unsupported output operation '{node.op_type}'"
            )

    graph_outputs = {output.name for output in model.graph.output}
    if value_name not in graph_outputs:
        raise ValueError(f"DriveNeuralGRU checkpoint '{model_path}' output head does not reach a graph output")

    return _GRUNetworkDescription(
        layers=tuple(layers),
        head_weight=head_weight,
        head_bias=head_bias,
        apply_tanh=apply_tanh,
        output_scale=output_scale,
    )


class DriveNeuralGRU(DriveBase):
    """Stateful GRU actuator drive using Warp-NN.

    Checkpoint metadata selects an ordered set of supported joint-state,
    target, and optional caller-provided generalized-bias-force inputs. The
    network and weights are loaded from an ONNX checkpoint and evaluated with
    Warp-NN.
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
        self._description = _load_network_description(self.model_path)
        metadata = load_metadata(self.model_path)

        normalization = metadata["normalization"]
        input_normalization = normalization["inputs"]
        self._input_feature_keys = _parse_input_feature_keys(metadata, self.model_path)
        if self._description.layers[0].input_size != len(self._input_feature_keys):
            raise ValueError(
                f"DriveNeuralGRU checkpoint '{self.model_path}' GRU input size "
                f"{self._description.layers[0].input_size} does not match input_columns "
                f"{len(self._input_feature_keys)}"
            )
        self._control_input_attribute_keys = (
            (_DYNAMIC_BIAS_CONTROL_ATTRIBUTE,) if "dynamic_bias" in self._input_feature_keys else ()
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
        self._net_input: wp.array2d[float] | None = None
        self._gru_layers: list[Any] = []
        self._head: Any = None
        self._activation: Any = None
        self._next_hidden: list[wp.array2d[float]] | None = None

    def finalize(self, device: wp.Device, num_actuators: int) -> None:
        """Create the Warp-NN layers and inference buffers.

        Args:
            device: Device on which inference runs.
            num_actuators: Number of vectorized actuator lanes.
        """
        try:
            from warp_nn import nn  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "DriveNeuralGRU requires Warp-NN and ONNX. Install them with `pip install newton[onnx]`."
            ) from exc

        self._device = device
        self._num_actuators = num_actuators
        self._input_means_wp = wp.array(self._input_means, dtype=wp.float32, device=device)
        self._input_stds_wp = wp.array(self._input_stds, dtype=wp.float32, device=device)
        self._input_feature_codes_wp = wp.array(
            [_INPUT_FEATURE_CODES[name] for name in self._input_feature_keys], dtype=wp.int32, device=device
        )
        self._net_input = wp.zeros((num_actuators, len(self._input_feature_keys)), dtype=wp.float32, device=device)

        self._gru_layers = []
        with wp.ScopedDevice(device):
            for description in self._description.layers:
                layer = nn.GRUCell(
                    description.input_size,
                    description.hidden_size,
                    bias=description.bias_ih is not None,
                )
                state_dict = {
                    "weight_ih": description.weight_ih,
                    "weight_hh": description.weight_hh,
                }
                if description.bias_ih is not None:
                    state_dict["bias_ih"] = description.bias_ih
                    state_dict["bias_hh"] = description.bias_hh
                layer.load_state_dict(state_dict)
                self._gru_layers.append(layer)

            self._head = nn.Linear(self._description.layers[-1].hidden_size, 1)
            self._head.load_state_dict(
                {
                    "weight": self._description.head_weight,
                    "bias": self._description.head_bias.reshape(1, 1),
                }
            )
            self._activation = nn.Tanh() if self._description.apply_tanh else None

            # Populate Warp-NN's shape caches before optional graph capture.
            warm_hidden = wp.zeros(
                (num_actuators, self._description.layers[0].hidden_size), dtype=wp.float32, device=device
            )
            output = self._net_input
            for layer in self._gru_layers:
                output = layer(output, warm_hidden)
            output = self._head(output)
            if self._activation is not None:
                self._activation(output)

    def is_stateful(self) -> bool:
        """Return ``True`` because the GRU maintains hidden state."""
        return True

    def is_graphable(self) -> bool:
        """Return ``True`` because inference uses graph-capturable Warp operations."""
        return True

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
            raise ValueError(f"GRU state device {device} must match controller device {self._device}")
        return DriveNeuralGRU.State(
            hidden=wp.zeros(
                (len(self._description.layers), num_actuators, self._description.layers[0].hidden_size),
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
    ) -> None:
        """Evaluate one GRU sample and write physical effort."""
        control_inputs = getattr(self, "_bound_control_inputs", {})
        self._bound_control_inputs = {}
        bias_force = control_inputs.get(_DYNAMIC_BIAS_CONTROL_ATTRIBUTE)
        self._next_hidden = None
        if self._control_input_attribute_keys and bias_force is None:
            raise RuntimeError(
                "DriveNeuralGRU input_columns includes 'dynamic_bias', but no control.actuator.dynamic_bias was bound"
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

        runtime_device = device or self._device
        wp.launch(
            _assemble_input_kernel,
            dim=(self._num_actuators, len(self._input_feature_keys)),
            inputs=[
                positions,
                velocities,
                target_pos,
                target_vel,
                bias_force,
                pos_indices,
                vel_indices,
                target_pos_indices,
                target_vel_indices,
                self._input_feature_codes_wp,
                self._input_means_wp,
                self._input_stds_wp,
            ],
            outputs=[self._net_input],
            device=runtime_device,
        )

        hidden_flat = state.hidden.reshape(
            (len(self._description.layers) * self._num_actuators, self._description.layers[0].hidden_size)
        )
        output = self._net_input
        next_hidden = []
        for layer_index, layer in enumerate(self._gru_layers):
            start = layer_index * self._num_actuators
            layer_hidden = hidden_flat[start : start + self._num_actuators]
            output = layer(output, layer_hidden)
            next_hidden.append(output)
        self._next_hidden = next_hidden

        output = self._head(output)
        if self._activation is not None:
            output = self._activation(output)
        wp.launch(
            _write_effort_kernel,
            dim=self._num_actuators,
            inputs=[
                output,
                self._description.output_scale,
                self._effort_mean,
                self._effort_std,
            ],
            outputs=[forces],
            device=runtime_device,
        )

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
        next_hidden_flat = next_state.hidden.reshape(
            (len(self._description.layers) * self._num_actuators, self._description.layers[0].hidden_size)
        )
        for layer_index, layer_hidden in enumerate(self._next_hidden):
            start = layer_index * self._num_actuators
            wp.copy(next_hidden_flat[start : start + self._num_actuators], layer_hidden)
        self._next_hidden = None
