# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Refactored Cosserat rod simulation with unified backend architecture.

This module provides a clean separation of concerns:
- Model: Rod state data structure (CosseratRodModel)
- Solver: Backend-agnostic solver interface (CosseratSolver)
- Backends: Pluggable implementations (Reference, NumPy, WarpCPU, WarpGPU)

Example usage:
    from newton.examples.cosserat_dll.refactor import (
        CosseratRodModel,
        CosseratSolver,
        BackendType,
        create_straight_rod,
    )

    # Create a rod model
    model = create_straight_rod(
        n_particles=20,
        start_pos=(0, 0, 1),
        direction=(0, 0, -1),
        segment_length=0.05,
    )

    # Create solver with desired backend
    solver = CosseratSolver(model, backend=BackendType.WARP_GPU)

    # Step simulation
    solver.step(dt=1/240)
"""

from .model import CosseratRodModel, create_straight_rod
from .solver import BackendType, CosseratSolver

__all__ = [
    "CosseratRodModel",
    "CosseratSolver",
    "BackendType",
    "create_straight_rod",
]
