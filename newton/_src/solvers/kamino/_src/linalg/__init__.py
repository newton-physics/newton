# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kamino Linear Algebra Module"""

from . import utils
from .core import (
    DenseLinearOperatorData,
    DenseRectangularMultiLinearInfo,
    DenseSquareMultiLinearInfo,
)

# Import the RCM-reordered semi-sparse blocked LLT solver here (rather than
# from .linear) to avoid a circular import: .factorize.llt_blocked_rcm_solver
# imports DirectSolver from .linear, so .linear cannot import it back.
# At this point .linear has been fully resolved, so the downstream import is safe.
from .factorize.llt_blocked_rcm_solver import LLTBlockedRCMSolver
from .linear import (
    ConjugateGradientSolver,
    ConjugateResidualSolver,
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
    LLTSequentialSolver | LLTBlockedSolver | LLTBlockedRCMSolver | ConjugateGradientSolver | ConjugateResidualSolver
)

###
# Module interface
###

__all__ = [
    "ConjugateGradientSolver",
    "ConjugateResidualSolver",
    "DenseLinearOperatorData",
    "DenseRectangularMultiLinearInfo",
    "DenseSquareMultiLinearInfo",
    "DirectSolver",
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
