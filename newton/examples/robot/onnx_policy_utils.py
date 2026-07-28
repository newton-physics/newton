# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def _require_onnx():
    try:
        import onnx  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only on missing optional dependency
        raise ImportError(
            "Validating ONNX policy shapes requires the optional `onnx` package. "
            "Install it with `pip install newton[onnx]`."
        ) from exc
    return onnx


def _tensor_shape(value_info) -> tuple[int | None, ...]:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError(f"ONNX tensor '{value_info.name}' does not declare a shape")

    shape = []
    for dim in tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.HasField("dim_value") else None)
    return tuple(shape)


def _find_value_info(values: Iterable, name: str):
    for value_info in values:
        if value_info.name == name:
            return value_info
    raise ValueError(f"ONNX graph does not contain tensor '{name}'")


def _format_shape(shape: tuple[int | None, ...]) -> str:
    return "(" + ", ".join("?" if dim is None else str(dim) for dim in shape) + ")"


def _validate_policy_tensor_shape(
    shape: tuple[int | None, ...],
    *,
    expected_width: int,
    tensor_name: str,
    tensor_role: str,
    context: str,
) -> None:
    if len(shape) != 2:
        raise ValueError(
            f"{context}: policy {tensor_role} '{tensor_name}' has shape {_format_shape(shape)}, "
            f"expected rank 2 with width {expected_width}"
        )

    batch, width = shape
    if batch not in (None, 1):
        raise ValueError(
            f"{context}: policy {tensor_role} '{tensor_name}' has batch dimension {batch}, expected 1 or dynamic"
        )
    if width != expected_width:
        raise ValueError(
            f"{context}: policy {tensor_role} '{tensor_name}' has width {width}, expected {expected_width}"
        )


def validate_policy_io_shapes(
    policy_path: str,
    input_name: str,
    output_name: str,
    *,
    obs_width: int,
    action_width: int,
    context: str,
) -> None:
    """Validate ONNX policy input and output widths.

    Args:
        policy_path: Path to the ONNX policy file.
        input_name: ONNX graph input tensor name used for observations.
        output_name: ONNX graph output tensor name used for actions.
        obs_width: Expected observation width.
        action_width: Expected action width.
        context: Caller name included in validation errors.
    """
    onnx = _require_onnx()
    model = onnx.load(policy_path, load_external_data=False)

    graph_name = Path(policy_path).name
    error_context = f"{context} ({graph_name})"

    input_shape = _tensor_shape(_find_value_info(model.graph.input, input_name))
    output_shape = _tensor_shape(_find_value_info(model.graph.output, output_name))

    _validate_policy_tensor_shape(
        input_shape,
        expected_width=obs_width,
        tensor_name=input_name,
        tensor_role="input",
        context=error_context,
    )
    _validate_policy_tensor_shape(
        output_shape,
        expected_width=action_width,
        tensor_name=output_name,
        tensor_role="output",
        context=error_context,
    )
