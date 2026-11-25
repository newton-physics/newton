# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Friction components of a shape's friction mode as bits, for the free-body angular rows.

A contact gets a component when either of its shapes enables it: the bits of its two shapes are OR-ed.
"""

import numpy as np
import warp as wp

FRICTION_BIT_SLIDING = wp.constant(1)
FRICTION_BIT_TORSIONAL = wp.constant(2)
FRICTION_BIT_ROLLING = wp.constant(4)
FRICTION_BITS_ANGULAR = wp.constant(6)


# Stand-in until newton#5064 encodes ShapeFrictionMode as these bits (SLIDING=1, SLIDING_TORSIONAL=3,
# SLIDING_ROLLING=5, SLIDING_TORSIONAL_ROLLING=7): maps the ordered modes 0-3 to 0, 1, 3, 7. Then return mode.
@wp.func
def shape_friction_bits(mode: int) -> int:
    """Return the friction-component bits of one shape's friction mode."""
    bits = int(0)
    if mode > 0:
        bits = (1 << mode) - 1
    return bits


def shape_friction_bits_host(modes: np.ndarray) -> np.ndarray:
    """Return the friction-component bits of each shape's friction mode, as :func:`shape_friction_bits` does."""
    modes = np.asarray(modes, dtype=np.int64)
    return np.where(modes > 0, (1 << np.clip(modes, 0, 30)) - 1, 0)


@wp.func
def contact_friction_bits(shape_material_friction_mode: wp.array[wp.int32], shape_a: int, shape_b: int) -> int:
    """Return the friction-component bits of a rigid contact, the OR of its shapes' bits."""
    if not shape_material_friction_mode:
        return FRICTION_BIT_SLIDING
    bits = int(0)
    if shape_a >= 0:
        bits = bits | shape_friction_bits(shape_material_friction_mode[shape_a])
    if shape_b >= 0:
        bits = bits | shape_friction_bits(shape_material_friction_mode[shape_b])
    return bits
