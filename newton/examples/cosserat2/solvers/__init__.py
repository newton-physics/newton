# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pluggable constraint solvers for Cosserat rod XPBD simulation."""

from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)
from newton.examples.cosserat2.solvers.jacobi import ConstraintSolverJacobi
from newton.examples.cosserat2.solvers.thomas import ConstraintSolverThomas
from newton.examples.cosserat2.solvers.cholesky import ConstraintSolverCholesky

# Solver registry for runtime switching
SOLVER_REGISTRY = {
    ConstraintSolverType.JACOBI: ConstraintSolverJacobi,
    ConstraintSolverType.THOMAS: ConstraintSolverThomas,
    ConstraintSolverType.CHOLESKY_SINGLE: ConstraintSolverCholesky,
    ConstraintSolverType.CHOLESKY_MULTI: ConstraintSolverCholesky,  # Same class, different config
}

__all__ = [
    "ConstraintSolverBase",
    "ConstraintSolverType",
    "FrictionMethod",
    "ConstraintSolverJacobi",
    "ConstraintSolverThomas",
    "ConstraintSolverCholesky",
    "SOLVER_REGISTRY",
]
