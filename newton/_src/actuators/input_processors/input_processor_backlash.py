# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .base import InputProcessorBase


@wp.kernel
def _backlash_read_kernel(
    positions: wp.array[float],
    velocities: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    vel_indices: wp.array[wp.uint32],
    sim_positions: wp.array[float],
    sim_velocities: wp.array[float],
    backlash_pos_indices: wp.array[wp.uint32],
    backlash_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
    out_vel: wp.array[float],
):
    i = wp.tid()
    out_pos[i] = positions[pos_indices[i]] + sim_positions[backlash_pos_indices[i]]
    out_vel[i] = velocities[vel_indices[i]] + sim_velocities[backlash_indices[i]]


class InputProcessorBacklash(InputProcessorBase):
    """Measure position and velocity on the link side of a backlash joint.

    The backlash itself is modelled in the physics: a passive joint with a
    small motion range sits in series between the actuated joint and the
    link. The solver handles the dead zone, where the actuator effort does
    not reach the link. This processor models an encoder on the link side of
    that joint. The drive reads

    .. math::

        q = q_{\\text{actuated}} + q_{\\text{backlash}}, \\qquad
        \\dot q = \\dot q_{\\text{actuated}} + \\dot q_{\\text{backlash}}

    instead of the actuated joint alone.
    """

    JOINT_PARAMS: ClassVar[set[str]] = {"backlash_joint"}

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        """Resolve user-provided arguments.

        Args:
            args: User-provided arguments. ``backlash_joint`` is required.

        Returns:
            ``{"backlash_joint": ...}``.
        """
        if args.get("backlash_joint") is None:
            raise ValueError("InputProcessorBacklash requires 'backlash_joint' argument")
        return {"backlash_joint": args["backlash_joint"]}

    def __init__(
        self,
        backlash_joint_indices: wp.array[wp.uint32],
        backlash_joint_pos_indices: wp.array[wp.uint32] | None = None,
    ):
        """Initialize the backlash processor.

        Args:
            backlash_joint_indices: Per-DOF index of the backlash joint into
                ``joint_qd``-shaped arrays, shape ``(N,)``.
            backlash_joint_pos_indices: Per-DOF index of the backlash joint
                into ``joint_q``-shaped arrays, shape ``(N,)``. Defaults to
                *backlash_joint_indices*.
        """
        self.backlash_joint_indices = backlash_joint_indices
        """Backlash joint indices into ``joint_qd``-shaped arrays, shape (N,)."""
        self.backlash_joint_pos_indices = (
            backlash_joint_pos_indices if backlash_joint_pos_indices is not None else backlash_joint_indices
        )
        """Backlash joint indices into ``joint_q``-shaped arrays, shape (N,)."""
        if self.backlash_joint_pos_indices.shape != backlash_joint_indices.shape:
            raise ValueError(
                f"backlash_joint_pos_indices shape {self.backlash_joint_pos_indices.shape} must match "
                f"backlash_joint_indices shape {backlash_joint_indices.shape}"
            )
        self._num_actuators: int = 0
        self._device: wp.Device | None = None
        self._out_pos: wp.array[float] | None = None
        self._out_vel: wp.array[float] | None = None
        self._sequential_indices: wp.array[wp.uint32] | None = None

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        if len(self.backlash_joint_indices) != num_actuators:
            raise ValueError(
                f"backlash_joint_indices length {len(self.backlash_joint_indices)} must match "
                f"the number of actuators {num_actuators}"
            )
        self._device = device
        self._num_actuators = num_actuators
        self._out_pos = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._out_vel = wp.zeros(num_actuators, dtype=wp.float32, device=device, requires_grad=requires_grad)
        self._sequential_indices = wp.array(np.arange(num_actuators, dtype=np.uint32), device=device)

    def process(self, inputs: InputProcessorBase.Inputs, state: None, dt: float | None) -> InputProcessorBase.Inputs:
        wp.launch(
            kernel=_backlash_read_kernel,
            dim=self._num_actuators,
            inputs=[
                inputs.positions,
                inputs.velocities,
                inputs.pos_indices,
                inputs.vel_indices,
                inputs.sim_positions,
                inputs.sim_velocities,
                self.backlash_joint_pos_indices,
                self.backlash_joint_indices,
            ],
            outputs=[self._out_pos, self._out_vel],
            device=self._device,
        )
        return replace(
            inputs,
            positions=self._out_pos,
            velocities=self._out_vel,
            pos_indices=self._sequential_indices,
            vel_indices=self._sequential_indices,
        )
