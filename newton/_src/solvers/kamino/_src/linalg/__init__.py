# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Kamino Linear Algebra Module"""

from . import utils
from .core import (
    DenseLinearOperatorData,
    DenseRectangularMultiLinearInfo,
    DenseSquareMultiLinearInfo,
)
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
    "LLTBlockedSolver",
    "LLTSequentialSolver",
    "LinearSolver",
    "LinearSolverNameToType",
    "LinearSolverType",
    "LinearSolverTypeToName",
    "utils",
]
