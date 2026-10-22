# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import warp as wp

from .base import InputProcessorBase


@wp.kernel
def _backlash_read_kernel(
    positions: wp.array[float],
    pos_indices: wp.array[wp.uint32],
    sim_positions: wp.array[float],
    backlash_pos_indices: wp.array[wp.uint32],
    out_pos: wp.array[float],
):
    i = wp.tid()
    out_pos[pos_indices[i]] = positions[pos_indices[i]] + sim_positions[backlash_pos_indices[i]]


class InputProcessorBacklash(InputProcessorBase):
    """Measure position on the link side of a backlash joint.

    The backlash itself is modelled in the physics: a passive joint with a
    small motion range sits in series between the actuated joint and the
    link. The solver handles the dead zone, where the actuator effort does
    not reach the link. This processor models an encoder on the link side of
    that joint. The drive reads the position

    .. math::

        q = q_{\\text{actuated}} + q_{\\text{backlash}}

    instead of the actuated joint alone. The velocity stays that of the
    actuated joint: it is motor physics, e.g. the back-EMF of
    :class:`~newton.actuators.DriveBAM`, not an encoder reading.

    The sum is written into an array shaped like the incoming positions, at
    the actuated indices, so :attr:`~newton.actuators.InputProcessorBase.Inputs.pos_indices`
    stays valid.
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
            backlash_joint_indices: Per-actuator index of the backlash joint into
                ``joint_qd``-shaped arrays, shape ``(N,)``.
            backlash_joint_pos_indices: Per-actuator index of the backlash joint
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
        self._requires_grad: bool = False
        self._out_pos: wp.array[float] | None = None

    def finalize(self, device: wp.Device, num_actuators: int, requires_grad: bool = False) -> None:
        if len(self.backlash_joint_indices) != num_actuators:
            raise ValueError(
                f"backlash_joint_indices length {len(self.backlash_joint_indices)} must match "
                f"the number of actuators {num_actuators}"
            )
        self._device = device
        self._num_actuators = num_actuators
        self._requires_grad = requires_grad

    def process(self, inputs: InputProcessorBase.Inputs, state: None, dt: float | None) -> InputProcessorBase.Inputs:
        if self._out_pos is None or self._out_pos.shape != inputs.positions.shape:
            self._out_pos = wp.zeros(
                inputs.positions.shape, dtype=wp.float32, device=self._device, requires_grad=self._requires_grad
            )
        wp.launch(
            kernel=_backlash_read_kernel,
            dim=self._num_actuators,
            inputs=[inputs.positions, inputs.pos_indices, inputs.sim_positions, self.backlash_joint_pos_indices],
            outputs=[self._out_pos],
            device=self._device,
        )
        return replace(inputs, positions=self._out_pos)
