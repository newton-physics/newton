# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batched single-particle anchor constraints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from ..block_csr import BlockCsrBuilder


@wp.kernel
def _accumulate_anchor_force(
    indices: wp.array[int],
    targets: wp.array[wp.vec3],
    stiffnesses: wp.array[float],
    positions: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
):
    constraint = wp.tid()
    particle = indices[constraint]
    force = -stiffnesses[constraint] * (positions[particle] - targets[constraint])
    wp.atomic_add(forces, particle, force)


class ConstraintAnchor:
    """A batch of quadratic constraints anchoring particles to targets."""

    def __init__(
        self,
        indices: Sequence[int],
        targets: Sequence[wp.vec3],
        stiffnesses: Sequence[float],
        particle_count: int,
        device: Any,
    ):
        """Create an anchor constraint batch.

        Args:
            indices: Particle index for every anchor.
            targets: World-space anchor targets [m].
            stiffnesses: Positive anchor stiffnesses [N/m].
            particle_count: Number of particles in the associated model.
            device: Warp device storing runtime arrays.
        """
        if particle_count <= 0:
            raise ValueError("particle_count must be positive")
        if len(indices) == 0 or len(indices) != len(targets) or len(indices) != len(stiffnesses):
            raise ValueError("Anchor indices, targets, and stiffnesses must have the same nonzero length")

        self.host_indices = tuple(int(index) for index in indices)
        self.host_targets = tuple(targets)
        self.host_stiffnesses = tuple(float(stiffness) for stiffness in stiffnesses)
        for index in self.host_indices:
            if index < 0 or index >= particle_count:
                raise ValueError(f"Anchor particle index {index} is outside particle_count={particle_count}")
        for target in self.host_targets:
            if not np.isfinite(np.asarray(target, dtype=np.float32)).all():
                raise ValueError("Anchor targets must be finite")
        for stiffness in self.host_stiffnesses:
            if not np.isfinite(stiffness) or stiffness <= 0.0:
                raise ValueError("Anchor stiffnesses must be finite and positive")

        self.particle_count = particle_count
        self.device = wp.get_device(device)
        self.indices = wp.array(self.host_indices, dtype=int, device=self.device)
        self.targets = wp.array(self.host_targets, dtype=wp.vec3, device=self.device)
        self.stiffnesses = wp.array(self.host_stiffnesses, dtype=float, device=self.device)

    def append_hessian(self, builder: BlockCsrBuilder) -> None:
        """Append the fixed projective-dynamics Hessian blocks."""
        if builder.row_count != self.particle_count:
            raise ValueError("Constraint and block matrix particle counts differ")
        for index, stiffness in zip(self.host_indices, self.host_stiffnesses, strict=True):
            builder.add_scaled_identity(index, index, stiffness)

    def accumulate_force(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        """Add anchor forces evaluated at ``positions`` to ``output``."""
        self._validate_runtime_arrays(positions, output)
        wp.launch(
            _accumulate_anchor_force,
            dim=len(self.indices),
            inputs=[self.indices, self.targets, self.stiffnesses, positions],
            outputs=[output],
            device=self.device,
        )

    def _validate_runtime_arrays(self, positions: wp.array[wp.vec3], output: wp.array[wp.vec3]) -> None:
        if len(positions) != self.particle_count or len(output) != self.particle_count:
            raise ValueError(f"Expected {self.particle_count} particle rows")
        if positions.device != self.device or output.device != self.device:
            raise ValueError("Constraint and runtime arrays must use the same device")
