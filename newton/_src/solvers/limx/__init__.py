# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LIMX constraint-based particle solver."""

from .constraints import ConstraintAnchor, ConstraintDihedralBending, ConstraintDistance, ConstraintTriangleElastic
from .linear_solver import PcgSolver
from .operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from .solver_newton import SolverLIMX

__all__ = [
    "CompositeLinearOperator",
    "ConstraintAnchor",
    "ConstraintDihedralBending",
    "ConstraintDistance",
    "ConstraintTriangleElastic",
    "EmptyDynamicConstraintOperator",
    "PcgSolver",
    "SolverLIMX",
]
