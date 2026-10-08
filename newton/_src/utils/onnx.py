# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private helpers for loading Warp-NN policies with configurable batches."""

import tempfile
from pathlib import Path

import warp as wp


def load_onnx_runtime(
    path: str,
    *,
    device: wp.DeviceLike | None = None,
    input_batch_axes: int | dict[str, int] | None = None,
    requires_grad: bool = False,
):
    """Load a runtime, relaxing explicitly selected ONNX batch dimensions.

    Neural drives accept checkpoints exported with a fixed batch size. Make
    those input dimensions dynamic in a temporary copy instead of relying on
    Warp-NN's deprecated constructor overrides. Leave the source file intact.
    This preserves the existing batch override; it does not rewrite batch
    constants baked into graph operators. Prefer exports with dynamic batch
    axes. Call the runtime with the intended inputs before CUDA graph capture.
    """
    try:
        import onnx  # noqa: PLC0415
        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ONNX policy inference requires Warp-NN and ONNX. Install them with `pip install newton[onnx]`."
        ) from exc

    if input_batch_axes is None:
        return OnnxRuntime(path, device=device, requires_grad=requires_grad)

    model = onnx.load(path)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if isinstance(input_batch_axes, dict):
        unknown = set(input_batch_axes) - {value.name for value in inputs}
        if unknown:
            raise KeyError(f"Unknown ONNX batch-axis inputs: {sorted(unknown)}")
    changed = False
    for value in inputs:
        axis = input_batch_axes.get(value.name) if isinstance(input_batch_axes, dict) else input_batch_axes
        if axis is None:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if not -len(dimensions) <= axis < len(dimensions):
            raise ValueError(f"ONNX input '{value.name}' batch axis {axis} is out of range for rank {len(dimensions)}")
        if dimensions[axis].HasField("dim_value"):
            dimensions[axis].dim_param = "newton_batch"
            changed = True

    if not changed:
        return OnnxRuntime(path, device=device, requires_grad=requires_grad)

    # External tensor data has been loaded; embed it before moving the model.
    onnx.external_data_helper.convert_model_from_external_data(model)
    with tempfile.TemporaryDirectory(prefix="newton-onnx-") as directory:
        runtime_path = Path(directory) / "model.onnx"
        onnx.save(model, runtime_path)
        return OnnxRuntime(str(runtime_path), device=device, requires_grad=requires_grad)
