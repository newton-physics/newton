# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import warp as wp

from .base import TransmissionBase


@wp.kernel
def _scatter_add_kernel(
    forces: wp.array[float],
    computed_forces: wp.array[float],
    indices: wp.array[wp.uint32],
    output: wp.array[float],
    computed_output: wp.array[float],
):
    i = wp.tid()
    idx = indices[i]
    wp.atomic_add(output, idx, forces[i])
    if computed_output:
        wp.atomic_add(computed_output, idx, computed_forces[i])


class TransmissionDirect(TransmissionBase):
    """Direct drive: the actuator coordinate is its single joint.

    The actuator position and velocity are the joint's, and the effort goes
    unscaled to it. This is the default transmission of
    :class:`~newton.actuators.Actuator`; :meth:`finalize` raises
    :class:`ValueError` for an actuator with more than one joint.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        if num_joints != 1:
            raise ValueError(f"TransmissionDirect drives one joint per actuator, got {num_joints}")

    def compute(
        self,
        forces: wp.array[float],
        computed_forces: wp.array[float],
        dof_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        computed_output: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_scatter_add_kernel,
            dim=forces.shape[0],
            inputs=[forces, computed_forces, dof_indices[0]],
            outputs=[output, computed_output],
            device=device,
        )

    def compute_state(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array[wp.uint32], wp.array[float], wp.array[wp.uint32]]:
        return positions, pos_indices[0], velocities, dof_indices[0]
