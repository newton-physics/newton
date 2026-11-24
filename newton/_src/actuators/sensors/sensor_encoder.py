# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import SensorBase


@wp.kernel
def _sensor_encoder_kernel(
    positions: wp.array[float],
    pos_indices: wp.array2d[wp.uint32],
    output_joint_pos_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
):
    j, i = wp.tid()
    q = positions[pos_indices[j, i]]
    if j == 0:
        q += positions[output_joint_pos_indices[i]]
    out_pos[j * pos_indices.shape[1] + i] = q


class SensorEncoder(SensorBase):
    """Position encoder behind a passive output joint.

    The encoder measures after a passive joint between the first joint of
    the actuator and the link, e.g. to model gearbox backlash, so it reads

    .. math::

        q = q_{\\text{actuated}} + q_{\\text{output}}

    for that joint; further joints of the actuator read their own position.
    This captures gearbox backlash when the output joint has a small motion
    range: the solver handles the dead zone, where the actuator effort does
    not reach the link. With a :class:`~newton.actuators.TransmissionFixed`
    of gear :math:`g`, the drive reads :math:`g \\, (q_{\\text{actuated}} +
    q_{\\text{output}})`.

    The measured positions are written into one buffer, allocated in
    :meth:`finalize`, so the encoder can be captured in a CUDA graph.
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
            raise ValueError("SensorEncoder requires 'output_joint' argument")
        return {"output_joint": args["output_joint"]}

    def __init__(
        self,
        output_joint_indices: wp.array[wp.uint32],
        *,
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

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        if len(self.output_joint_indices) != num_actuators:
            raise ValueError(
                f"output_joint_indices length {len(self.output_joint_indices)} must match "
                f"the number of actuators {num_actuators}"
            )
        self._out_pos = wp.zeros(
            num_joints * num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self._indices = wp.array(
            np.arange(num_joints * num_actuators, dtype=np.uint32).reshape(num_joints, num_actuators), device=device
        )

    def compute(
        self,
        positions: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array2d[wp.uint32]]:
        out_pos = wp.empty_like(self._out_pos) if self._out_pos.requires_grad else self._out_pos
        wp.launch(
            kernel=_sensor_encoder_kernel,
            dim=pos_indices.shape,
            inputs=[positions, pos_indices, self.output_joint_pos_indices],
            outputs=[out_pos],
            device=device,
        )
        return out_pos, self._indices
