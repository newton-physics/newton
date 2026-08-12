# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LIMX constraint-based particle and affine solvers."""

from .affine_body import AffineBodyModel
from .constraints import (
    ConstraintAffineStaticPlaneContact,
    ConstraintAnchor,
    ConstraintDihedralBending,
    ConstraintDistance,
    ConstraintGroupAffine,
    ConstraintGroupDynamic,
    ConstraintSelfCollision,
    ConstraintStaticPlaneContact,
    ConstraintTetrahedronARAP,
    ConstraintTriangleElastic,
)
from .linear_solver import PcgSolver
from .operator import CompositeLinearOperator, EmptyDynamicConstraintOperator
from .solver_affine import SolverLIMXAffine
from .solver_newton import SolverLIMX

__all__ = [
    "AffineBodyModel",
    "CompositeLinearOperator",
    "ConstraintAffineStaticPlaneContact",
    "ConstraintAnchor",
    "ConstraintDihedralBending",
    "ConstraintDistance",
    "ConstraintGroupAffine",
    "ConstraintGroupDynamic",
    "ConstraintSelfCollision",
    "ConstraintStaticPlaneContact",
    "ConstraintTetrahedronARAP",
    "ConstraintTriangleElastic",
    "EmptyDynamicConstraintOperator",
    "PcgSolver",
    "SolverLIMX",
    "SolverLIMXAffine",
]
