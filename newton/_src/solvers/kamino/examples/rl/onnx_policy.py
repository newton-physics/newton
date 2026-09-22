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
        self.runtime = OnnxRuntime(str(path), device=self.device, batch_size=batch_size, input_batch_axes=0)
        if len(self.runtime.input_names) != 1 or len(self.runtime.output_names) != 1:
            raise ValueError(
                f"Policy '{path}' must have exactly one input and one output; got "
                f"inputs={self.runtime.input_names}, outputs={self.runtime.output_names}"
            )
        self.input_name = self.runtime.input_names[0]
        self.output_name = self.runtime.output_names[0]
        output_shape = self.runtime._shapes[self.output_name]
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
