# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ONNX policy inference using Warp-NN."""

from pathlib import Path

import warp as wp

from .....utils.onnx import load_onnx_runtime


class WarpOnnxPolicy:
    """Evaluate a single-input, single-output ONNX policy with Warp-NN."""

    def __init__(self, path: str | Path, device: wp.DeviceLike, batch_size: int, *, action_width: int) -> None:
        self.device = wp.get_device(device)
        self.runtime = load_onnx_runtime(str(path), device=self.device, input_batch_axes=0)
        inputs, outputs = self.runtime.inputs, self.runtime.outputs
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Policy '{path}' must have exactly one input and one output; got "
                f"inputs={[spec.name for spec in inputs]}, outputs={[spec.name for spec in outputs]}"
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        observation = wp.zeros(
            tuple(batch_size if dimension is None else dimension for dimension in inputs[0].shape),
            dtype=inputs[0].dtype,
            device=self.device,
        )
        output_shape = self.runtime({self.input_name: observation})[self.output_name].shape
        expected_output_shape = (batch_size, action_width)
        if output_shape != expected_output_shape:
            raise ValueError(f"Policy '{path}' output shape must be {expected_output_shape}, got {output_shape}")

    def __call__(self, observation: wp.array[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate a contiguous float32 Warp observation batch."""
        if observation.dtype != wp.float32:
            raise TypeError(f"Policy observations must have dtype wp.float32, got {observation.dtype}")
        if observation.device != self.device:
            raise ValueError(f"Policy observations must be on device {self.device}, got {observation.device}")
        if not observation.is_contiguous:
            raise ValueError("Policy observations must be contiguous for zero-copy Warp inference")
        return self.runtime({self.input_name: observation})[self.output_name]
