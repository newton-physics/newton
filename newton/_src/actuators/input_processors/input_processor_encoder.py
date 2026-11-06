# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import InputProcessorBase


@wp.kernel
def _encoder_read_kernel(
    positions: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    sim_positions: wp.array[float],
    output_pos_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
):
    i = wp.tid()
    out_pos[i] = positions[pos_indices[i]] + sim_positions[output_pos_indices[i]]


class InputProcessorEncoder(InputProcessorBase):
    """Position encoder of the actuator.

    Models the sensor that provides the position the drive reads, including
    effects of how the encoder is mounted and how it measures. Supported
    effects:

    * **Output-side mounting** (``output_joint``): the encoder measures after
      a passive joint between the actuated joint and the link, e.g. to model
      gearbox backlash, so it reads

      .. math::

          q = q_{\\text{actuated}} + q_{\\text{output}}

      This captures gearbox backlash when the output joint has a small
      motion range: the solver handles the dead zone, where the actuator
      effort does not reach the link.

    The encoder does not change the velocity: it stays that of the actuated
    joint, which is motor physics, e.g. the back-EMF of
    :class:`~newton.actuators.MotorDC`.

    The measured position is written into one entry per actuator slot, and
    :attr:`~newton.actuators.InputProcessorBase.Inputs.pos_indices` is
    replaced by the slot indices. Both buffers are allocated in
    :meth:`finalize`, so the processor can be captured in a CUDA graph.
    """

    JOINT_PARAMS: ClassVar[set[str]] = {"output_joint"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments.

        Args:
            args: User-provided arguments. ``output_joint`` is required.

        Returns:
            ``{"output_joint": ...}``.
        """
        if args.get("output_joint") is None:
            raise ValueError("InputProcessorEncoder requires 'output_joint' argument")
        return {"output_joint": args["output_joint"]}

    def __init__(
        self,
        output_joint_indices: wp.array[wp.uint32],
        output_joint_pos_indices: wp.array[wp.uint32] | None = None,
    ):
        """Initialize the encoder.

        Args:
            output_joint_indices: Per-actuator index of the output joint into
                ``joint_qd``-shaped arrays, shape ``(N,)``.
            output_joint_pos_indices: Per-actuator index of the output joint
                into ``joint_q``-shaped arrays, shape ``(N,)``. Defaults to
                *output_joint_indices*.
        """
        self.output_joint_indices = output_joint_indices
        """Output joint indices into ``joint_qd``-shaped arrays, shape (N,)."""
        self.output_joint_pos_indices = (
            output_joint_pos_indices if output_joint_pos_indices is not None else output_joint_indices
        )
        """Output joint indices into ``joint_q``-shaped arrays, shape (N,)."""
        if self.output_joint_pos_indices.shape != output_joint_indices.shape:
            raise ValueError(
                f"output_joint_pos_indices shape {self.output_joint_pos_indices.shape} must match "
                f"output_joint_indices shape {output_joint_indices.shape}"
            )
        self._num_actuators: int = 0
        self._device: wp.Device | None = None
        self._out_pos: wp.array[float] | None = None
        self._sequential_indices: wp.array[wp.uint32] | None = None

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        if len(self.output_joint_indices) != num_actuators:
            raise ValueError(
                f"output_joint_indices length {len(self.output_joint_indices)} must match "
                f"the number of actuators {num_actuators}"
            )
        self._device = device
        self._num_actuators = num_actuators
        self._out_pos = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._sequential_indices = wp.array(np.arange(num_actuators, dtype=np.uint32), device=device)

    def process(self, inputs: InputProcessorBase.Inputs, state: None, dt: float | None) -> InputProcessorBase.Inputs:
        wp.launch(
            kernel=_encoder_read_kernel,
            dim=self._num_actuators,
            inputs=[inputs.positions, inputs.pos_indices, inputs.sim_positions, self.output_joint_pos_indices],
            outputs=[self._out_pos],
            device=self._device,
        )
        return replace(inputs, positions=self._out_pos, pos_indices=self._sequential_indices)
