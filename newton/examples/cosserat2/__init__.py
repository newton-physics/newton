# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat rod examples with pluggable XPBD constraint solvers.

This package provides refactored Cosserat rod simulation with:
- Consolidated kernel files
- Pluggable constraint solvers (Jacobi, Thomas, Cholesky)
- Runtime-switchable solver methods via UI

Examples:
    cosserat2_aorta: Catheter navigation through aorta mesh
"""

from newton.examples.cosserat2.cosserat_rod import CosseratRod, FrictionState
from newton.examples.cosserat2.solver_cosserat_xpbd import SolverCosseratXPBD, SolverConfig
from newton.examples.cosserat2.solvers import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
    ConstraintSolverJacobi,
    ConstraintSolverThomas,
    ConstraintSolverCholesky,
    SOLVER_REGISTRY,
)

__all__ = [
    # Core classes
    "CosseratRod",
    "FrictionState",
    "SolverCosseratXPBD",
    "SolverConfig",
    # Solver types
    "ConstraintSolverBase",
    "ConstraintSolverType",
    "FrictionMethod",
    "ConstraintSolverJacobi",
    "ConstraintSolverThomas",
    "ConstraintSolverCholesky",
    "SOLVER_REGISTRY",
]
