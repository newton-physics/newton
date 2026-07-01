# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SPH setup helper namespace."""

from .sph_model import (
    SPHMaterial,
    add_sph_particle_grid,
)

__all__ = [
    "SPHMaterial",
    "add_sph_particle_grid",
]
