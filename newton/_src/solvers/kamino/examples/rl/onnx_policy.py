# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""ONNX policy inference using Warp-NN."""

from pathlib import Path
from typing import TYPE_CHECKING

import warp as wp

if TYPE_CHECKING:
    import torch


class WarpOnnxPolicy:
    """Evaluate a single-input, single-output ONNX policy with Warp-NN."""

    def __init__(self, path: str | Path, device: wp.DeviceLike, batch_size: int) -> None:
        try:
            from warp_nn.runtime import OnnxRuntime  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Kamino ONNX policy inference requires Warp-NN. Install it with `pip install newton[onnx]`."
            ) from exc

        self.runtime = OnnxRuntime(str(path), device=device, batch_size=batch_size, input_batch_axes=0)
        if len(self.runtime.input_names) != 1 or len(self.runtime.output_names) != 1:
            raise ValueError(
                f"Policy '{path}' must have exactly one input and one output; got "
                f"inputs={self.runtime.input_names}, outputs={self.runtime.output_names}"
            )
        self.input_name = self.runtime.input_names[0]
        self.output_name = self.runtime.output_names[0]

    def __call__(self, observation: "torch.Tensor") -> "torch.Tensor":
        """Evaluate a contiguous float32 Torch observation batch."""
        import torch

        if observation.dtype != torch.float32:
            raise TypeError(f"Policy observations must have dtype torch.float32, got {observation.dtype}")
        if not observation.is_contiguous():
            raise ValueError("Policy observations must be contiguous for zero-copy Warp inference")
        observation_wp = wp.from_torch(observation, dtype=wp.float32)
        output_wp = self.runtime({self.input_name: observation_wp})[self.output_name]
        return wp.to_torch(output_wp)
