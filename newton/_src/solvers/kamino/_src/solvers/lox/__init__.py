# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body primal splitting for Kamino.

``solver`` orchestrates the solve. ``problem`` owns the body topology and the
constraint rows loaded from Kamino, and ``system`` assembles and factorizes the
body system, using the dense block solvers of ``llt``. ``projection`` runs the
unilateral sweeps over joint limits, joint friction and contacts as Jacobi or
colored Gauss--Seidel sweeps (colored by ``coloring``), optionally accelerated
by ``apgd`` or ``anderson``; ``contact`` holds the local contact laws and
``spatial_warmstart`` carries torsional and rolling impulses across steps.
``types`` holds the data containers, and each ``*_kernels`` module holds the
kernels of its namesake; ``world_projection_kernels`` holds the projection
kernel that runs every sweep of a world inside one thread block. ``metrics`` reconstructs a dual solution from the LOX
outputs for Kamino's solution metrics.
"""

from .solver import LOXSolver

__all__ = ["LOXSolver"]
