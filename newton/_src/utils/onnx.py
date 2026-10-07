# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import warp as wp


def load_onnx_runtime(
    path: str,
    *,
    device: wp.DeviceLike | None = None,
    batch_size: int = 1,
    input_batch_axes: int | dict[str, int] | None = None,
    requires_grad: bool = False,
):
    """Specialize input batches without changing the authored ONNX checkpoint."""
    try:
        import onnx  # noqa: PLC0415
        from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "ONNX inference requires Warp-NN and ONNX. Install them with `pip install newton[onnx]`."
        ) from exc

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    model = onnx.load(path)
    initializers = {value.name for value in model.graph.initializer}
    inputs = [value for value in model.graph.input if value.name not in initializers]
    if isinstance(input_batch_axes, dict):
        if unknown := input_batch_axes.keys() - {value.name for value in inputs}:
            raise KeyError(f"Unknown ONNX inputs in input_batch_axes: {sorted(unknown)}")
    changed = False
    for value in inputs:
        dimensions = value.type.tensor_type.shape.dim
        axis = input_batch_axes.get(value.name) if isinstance(input_batch_axes, dict) else input_batch_axes
        if axis is not None:
            if not -len(dimensions) <= axis < len(dimensions):
                raise ValueError(f"Batch axis {axis} is out of range for ONNX input '{value.name}'")
            if not dimensions[axis].HasField("dim_value") or dimensions[axis].dim_value != batch_size:
                dimensions[axis].dim_value = batch_size
                changed = True
        for dimension in dimensions:
            if not dimension.HasField("dim_value"):
                dimension.dim_value = batch_size
                changed = True

    if not changed:
        return OnnxRuntime(path, device=device, requires_grad=requires_grad)

    # Warp-NN 0.4 validates fixed dimensions instead of overriding them at preparation.
    with tempfile.TemporaryDirectory() as directory:
        batched_path = str(Path(directory) / "batched.onnx")
        onnx.save(model, batched_path)
        return OnnxRuntime(batched_path, device=device, requires_grad=requires_grad)
