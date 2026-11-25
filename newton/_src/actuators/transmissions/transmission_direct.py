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
    """Default transmission: the actuator acts on its single joint, without a ratio.

    The actuator position and velocity are the joint's, and the effort goes
    unscaled to it. :meth:`finalize` raises :class:`ValueError` for an
    actuator with more than one joint.
    """

    @classmethod
    def resolve_arguments(cls, args: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _first_row(self, rows: wp.array2d[wp.uint32]) -> wp.array[wp.uint32]:
        """Return ``rows[0]``, reusing the view built for the same array in an earlier step."""
        cache = self.__dict__.setdefault("_first_rows", {})
        cached = cache.get(id(rows))
        if cached is None or cached[0] is not rows:
            if len(cache) >= 8:
                cache.clear()
            cached = cache[id(rows)] = (rows, rows[0])
        return cached[1]

    def finalize(self, device: wp.Device, num_actuators: int, num_joints: int, requires_grad: bool = False) -> None:
        if num_joints != 1:
            raise ValueError(f"TransmissionDirect takes one joint per actuator, got {num_joints}")

    def compute(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        forces: wp.array[float],
        computed_forces: wp.array[float],
        effort_indices: wp.array2d[wp.uint32],
        output: wp.array[float],
        computed_output: wp.array[float] | None,
        device: wp.Device | None = None,
    ) -> None:
        wp.launch(
            kernel=_scatter_add_kernel,
            dim=forces.shape[0],
            inputs=[forces, computed_forces, self._first_row(effort_indices)],
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
        return positions, self._first_row(pos_indices), velocities, self._first_row(dof_indices)

    def compute_velocity(
        self,
        positions: wp.array[float],
        velocities: wp.array[float],
        pos_indices: wp.array2d[wp.uint32],
        dof_indices: wp.array2d[wp.uint32],
        device: wp.Device | None = None,
    ) -> tuple[wp.array[float], wp.array[wp.uint32]]:
        return velocities, self._first_row(dof_indices)
