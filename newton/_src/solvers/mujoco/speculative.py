# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normal constraints for separated Newton contact candidates."""

import warp as wp

from ...core.types import vec5


@wp.kernel
def restore_contact_properties(
    clearance: wp.array[float],
    saved_dim: wp.array[int],
    saved_solref: wp.array[wp.vec2],
    saved_solimp: wp.array[vec5],
    dim: wp.array[int],
    solref: wp.array[wp.vec2],
    solimp: wp.array[vec5],
):
    i = wp.tid()
    if clearance[i] > 0.0:
        dim[i] = saved_dim[i]
        solref[i] = saved_solref[i]
        solimp[i] = saved_solimp[i]


@wp.kernel
def prepare_contacts(
    count: wp.array[int],
    timestep: wp.array[float],
    world: wp.array[int],
    distance: wp.array[float],
    margin: wp.array[float],
    dim: wp.array[int],
    solref: wp.array[wp.vec2],
    solimp: wp.array[vec5],
    clearance: wp.array[float],
    saved_dim: wp.array[int],
    saved_solref: wp.array[wp.vec2],
    saved_solimp: wp.array[vec5],
):
    i = wp.tid()
    clearance[i] = 0.0
    if i < count[0]:
        gap = distance[i] - margin[i]
        if gap > 0.0:
            clearance[i] = gap
            saved_dim[i] = dim[i]
            saved_solref[i] = solref[i]
            saved_solimp[i] = solimp[i]
            # MuJoCo admits only negative-distance rows. Restore the real
            # clearance and install the predictive target after step1.
            distance[i] = margin[i] - wp.max(1.0e-8, wp.abs(margin[i]) * 1.0e-6)
            dim[i] = 1  # No friction before the surfaces meet.
            solref[i] = wp.vec2(2.0 * timestep[world[i] % timestep.shape[0]], 1.0)
            solimp[i] = vec5(0.95, 0.95, 0.001, 0.5, 2.0)


@wp.kernel
def set_contact_targets(
    count: wp.array[int],
    timestep: wp.array[float],
    world: wp.array[int],
    clearance: wp.array[float],
    address: wp.array2d[int],
    distance: wp.array[float],
    margin: wp.array[float],
    velocity: wp.array2d[float],
    position: wp.array2d[float],
    acceleration: wp.array2d[float],
):
    i = wp.tid()
    if i < count[0] and clearance[i] > 0.0:
        gap = clearance[i]
        distance[i] = gap + margin[i]
        row = address[i, 0]
        if row >= 0:
            w = world[i]
            h = timestep[w % timestep.shape[0]]
            position[w, row] = gap
            # Target v_next >= -gap / h under semi-implicit integration.
            # The unilateral response leaves separating motion unimpeded.
            acceleration[w, row] = -(velocity[w, row] + gap / h) / h
