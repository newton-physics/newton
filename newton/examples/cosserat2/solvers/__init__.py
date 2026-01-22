# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pluggable constraint solvers for Cosserat rod XPBD simulation."""

from newton.examples.cosserat2.solvers.base import (
    ConstraintSolverBase,
    ConstraintSolverType,
    FrictionMethod,
)
from newton.examples.cosserat2.solvers.cholesky import ConstraintSolverCholesky
from newton.examples.cosserat2.solvers.direct_stiff_rods import ConstraintSolverDirectStiffRods
from newton.examples.cosserat2.solvers.jacobi import ConstraintSolverJacobi
from newton.examples.cosserat2.solvers.local import ConstraintSolverLocal
from newton.examples.cosserat2.solvers.numpy_reference import ConstraintSolverNumpyReference
from newton.examples.cosserat2.solvers.thomas import ConstraintSolverThomas

# Solver registry for runtime switching
SOLVER_REGISTRY = {
    ConstraintSolverType.JACOBI: ConstraintSolverJacobi,
    ConstraintSolverType.THOMAS: ConstraintSolverThomas,
    ConstraintSolverType.CHOLESKY_SINGLE: ConstraintSolverCholesky,
    ConstraintSolverType.CHOLESKY_MULTI: ConstraintSolverCholesky,  # Same class, different config
    ConstraintSolverType.LOCAL: ConstraintSolverLocal,
    ConstraintSolverType.NUMPY_REFERENCE: ConstraintSolverNumpyReference,
    ConstraintSolverType.DIRECT_STIFF_RODS: ConstraintSolverDirectStiffRods,
}

__all__ = [
    "SOLVER_REGISTRY",
    "ConstraintSolverBase",
    "ConstraintSolverCholesky",
    "ConstraintSolverDirectStiffRods",
    "ConstraintSolverJacobi",
    "ConstraintSolverLocal",
    "ConstraintSolverNumpyReference",
    "ConstraintSolverThomas",
    "ConstraintSolverType",
    "FrictionMethod",
]
