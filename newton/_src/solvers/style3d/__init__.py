# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Style3D solver module.

This module provides helper functions for setting up Style3D cloth assets.
Use :class:`~newton.solvers.SolverStyle3D` as the canonical public solver
import path.
"""

from .cloth import (
    add_cloth_grid,
    add_cloth_mesh,
)

__all__ = [
    "add_cloth_grid",
    "add_cloth_mesh",
]
