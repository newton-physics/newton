# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kamino Linear Algebra Module"""

from . import utils
from .core import (
    DenseLinearOperatorData,
    DenseRectangularMultiLinearInfo,
    DenseSquareMultiLinearInfo,
)
from .factorize.hybrid_llt_solver import HybridLLTBlockedSolver

# Import the RCM and hybrid implementations from factorize rather than from
# linear because both build on the dense solver classes defined there.
from .factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from .linear import (
    ConjugateGradientSolver,
    ConjugateResidualSolver,
    ConjugateResidualSolverFused,
    DirectSolver,
    IterativeSolver,
    LinearSolver,
    LinearSolverNameToType,
    LinearSolverType,
    LinearSolverTypeToName,
    LLTBlockedSolver,
    LLTSequentialSolver,
)

# Register the reordering solver in the name<->type maps so it can be selected
# via the string "LLTBRCM" in ConstrainedDynamicsConfig.linear_solver_type.
LinearSolverNameToType["LLTBRCM"] = LLTBlockedRCMSolver
LinearSolverTypeToName[LLTBlockedRCMSolver] = "LLTBRCM"

# Widen the LinearSolverType alias to include the reordering solver. This
# matters because `delassus.py` performs a runtime
# `issubclass(solver, LinearSolverType)` check and would otherwise reject it.
LinearSolverType = (
    LLTSequentialSolver
    | LLTBlockedSolver
    | LLTBlockedRCMSolver
    | ConjugateGradientSolver
    | ConjugateResidualSolver
    | ConjugateResidualSolverFused
)

###
# Module interface
###

__all__ = [
    "ConjugateGradientSolver",
    "ConjugateResidualSolver",
    "ConjugateResidualSolverFused",
    "DenseLinearOperatorData",
    "DenseRectangularMultiLinearInfo",
    "DenseSquareMultiLinearInfo",
    "DirectSolver",
    "HybridLLTBlockedSolver",
    "IterativeSolver",
    "LLTBlockedRCMSolver",
    "LLTBlockedSolver",
    "LLTSequentialSolver",
    "LinearSolver",
    "LinearSolverNameToType",
    "LinearSolverType",
    "LinearSolverTypeToName",
    "utils",
]
