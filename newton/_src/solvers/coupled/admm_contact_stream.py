# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Internal fixed-capacity contact streams for ADMM coupled contacts.

These streams are solver-side ADMM work buffers, not a generic coupling contact
API. They hold detected contacts and the corresponding ADMM normal
force/impulse scalars so contact detection and constraint solves share one
fixed-layout record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import warp as wp


class AdmmContactType(IntEnum):
    """Homogeneous contact endpoint type stored by an ADMM contact stream."""

    RIGID_PARTICLE = 0
    PARTICLE_PARTICLE = 1


@dataclass
class AdmmContactStream:
    """Internal fixed-capacity stream of ADMM coupling contacts.

    A stream stores contacts between side A and side B endpoints. The normal
    points from side B toward side A, so a positive ``normal_force`` applies
    ``+normal`` to side A and ``-normal`` to side B. Unused endpoint arrays are
    filled with ``-1``. The first ``count[0]`` rows are active after detection.

    The normal force and impulse arrays are intentionally part of the stream so
    detection and constraint solve can share one record layout.

    Attributes:
        contact_type: Homogeneous endpoint-pair type in this stream.
        capacity: Maximum number of contacts stored.
        count: Active contact count, shape ``[1]``.
        count_max: Maximum active contact count observed, shape ``[1]``.
        body_a: Body endpoint on side A, or ``-1``.
        body_b: Body endpoint on side B, or ``-1``.
        particle_a: Particle endpoint on side A, or ``-1``.
        particle_b: Particle endpoint on side B, or ``-1``.
        shape_a: Shape endpoint on side A, or ``-1``.
        shape_b: Shape endpoint on side B, or ``-1``.
        point_a: Contact point on side A [m].
        point_b: Contact point on side B [m].
        normal: Contact normal from side B to side A [unitless].
        distance: Minimum admissible signed gap along ``normal`` [m].
        source_id: Detector-local source id for diagnostics or warm-start keys.
        normal_force: Normal contact force applied to side A [N].
        normal_impulse: Normal contact impulse applied to side A [N s].
    """

    contact_type: int
    capacity: int
    count: wp.array[int]
    count_max: wp.array[int]
    body_a: wp.array[int]
    body_b: wp.array[int]
    particle_a: wp.array[int]
    particle_b: wp.array[int]
    shape_a: wp.array[int]
    shape_b: wp.array[int]
    point_a: wp.array[wp.vec3]
    point_b: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    distance: wp.array[float]
    source_id: wp.array[int]
    normal_force: wp.array[float]
    normal_impulse: wp.array[float]

    @classmethod
    def allocate(
        cls,
        capacity: int,
        device,
        contact_type: int = int(AdmmContactType.PARTICLE_PARTICLE),
    ) -> AdmmContactStream:
        """Allocate a stream on ``device``.

        Args:
            capacity: Maximum number of contacts stored.
            device: Warp device for stream arrays.
            contact_type: Homogeneous endpoint-pair type.

        Returns:
            Allocated contact stream.
        """
        capacity = int(capacity)
        return cls(
            contact_type=int(contact_type),
            capacity=capacity,
            count=wp.zeros(1, dtype=int, device=device),
            count_max=wp.zeros(1, dtype=int, device=device),
            body_a=wp.full(capacity, -1, dtype=int, device=device),
            body_b=wp.full(capacity, -1, dtype=int, device=device),
            particle_a=wp.full(capacity, -1, dtype=int, device=device),
            particle_b=wp.full(capacity, -1, dtype=int, device=device),
            shape_a=wp.full(capacity, -1, dtype=int, device=device),
            shape_b=wp.full(capacity, -1, dtype=int, device=device),
            point_a=wp.zeros(capacity, dtype=wp.vec3, device=device),
            point_b=wp.zeros(capacity, dtype=wp.vec3, device=device),
            normal=wp.zeros(capacity, dtype=wp.vec3, device=device),
            distance=wp.zeros(capacity, dtype=float, device=device),
            source_id=wp.full(capacity, -1, dtype=int, device=device),
            normal_force=wp.zeros(capacity, dtype=float, device=device),
            normal_impulse=wp.zeros(capacity, dtype=float, device=device),
        )


@wp.kernel(enable_backward=False)
def admm_contact_stream_reset_count_kernel(count: wp.array[int]):
    """Reset the active contact count of a stream."""
    count[0] = 0


@wp.kernel(enable_backward=False)
def admm_contact_stream_update_normal_force_kernel(
    active_count: wp.array[int],
    dt: float,
    rho: float,
    W: wp.array[float],
    normal: wp.array[wp.vec3],
    lambda_k: wp.array[wp.vec3],
    u_k: wp.array[wp.vec3],
    Jv_k: wp.array[wp.vec3],
    normal_force: wp.array[float],
    normal_impulse: wp.array[float],
):
    """Write ADMM normal force/impulse scalars into a contact stream."""
    i = wp.tid()
    if i >= active_count[0]:
        normal_force[i] = 0.0
        normal_impulse[i] = 0.0
        return

    W_i = W[i]
    force = W_i * (lambda_k[i] + rho * W_i * (u_k[i] - Jv_k[i]))
    force_mag = wp.max(0.0, wp.dot(normal[i], force))
    normal_force[i] = force_mag
    normal_impulse[i] = force_mag * dt
