# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ONNX policy inference using Warp-NN."""

from pathlib import Path

import warp as wp


class WarpOnnxPolicy:
    """Evaluate a single-input, single-output ONNX policy with Warp-NN."""

    def __init__(self, path: str | Path, device: wp.DeviceLike, batch_size: int, *, action_width: int) -> None:
        try:
            from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Kamino ONNX policy inference requires Warp-NN. Install it with `pip install newton[onnx]`."
            ) from exc

        self.device = wp.get_device(device)
        self.runtime = OnnxRuntime(str(path), device=self.device)
        if len(self.runtime.inputs) != 1 or len(self.runtime.outputs) != 1:
            raise ValueError(
                f"Policy '{path}' must have exactly one input and one output; got "
                f"inputs={self.runtime.inputs}, outputs={self.runtime.outputs}"
            )
        self.input_name = self.runtime.inputs[0].name
        self.output_name = self.runtime.outputs[0].name
        self.runtime.prepare(batch_size=batch_size)
        input_spec = self.runtime.inputs[0]
        observation = wp.ones(
            tuple(batch_size if dimension is None else dimension for dimension in input_spec.shape),
            dtype=input_spec.dtype,
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
