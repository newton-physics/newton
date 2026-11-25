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
    velocities: wp.array[float],
    pos_indices: wp.array2d[wp.uint32],
    dof_indices: wp.array2d[wp.uint32],
    output_joint_pos_indices: wp.array[wp.uint32],
    output_joint_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
):
    j, i = wp.tid()
    q = positions[pos_indices[j, i]]
    qd = velocities[dof_indices[j, i]]
    if j == 0:
        q += positions[output_joint_pos_indices[i]]
        qd += velocities[output_joint_indices[i]]
    slot = j * pos_indices.shape[1] + i
    out_pos[slot] = q
    out_vel[slot] = qd


class SensorEncoder(SensorBase):
    """Encoder behind a passive output joint.

    The encoder measures after a passive joint between the first joint of
    the actuator and the link, e.g. to model gearbox backlash, so it reads

    .. math::

        q = q_{\\text{actuated}} + q_{\\text{output}}, \\qquad
        \\dot q = \\dot q_{\\text{actuated}} + \\dot q_{\\text{output}}

    for that joint, as a servo that derives its velocity from the same
    encoder; further joints of the actuator read their own position and
    velocity.

    The measured positions and velocities are written into buffers allocated
    in :meth:`finalize`, so the encoder can be captured in a CUDA graph.
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
        self._out_vel = wp.zeros(
            num_joints * num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self._indices = wp.array(
            np.arange(num_joints * num_actuators, dtype=np.uint32).reshape(num_joints, num_actuators), device=device
        )

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array2d[wp.uint32], wp.array[float], wp.array2d[wp.uint32]]:
        out_pos, out_vel = self._out_pos, self._out_vel
        if out_pos.requires_grad:
            out_pos, out_vel = wp.empty_like(out_pos), wp.empty_like(out_vel)
        wp.launch(
            kernel=_sensor_encoder_kernel,
            dim=pos_indices.shape,
            inputs=[
                positions,
                velocities,
                pos_indices,
                dof_indices,
                self.output_joint_pos_indices,
                self.output_joint_indices,
            ],
            outputs=[out_pos, out_vel],
            device=device,
        )
        return out_pos, self._indices, out_vel, self._indices
