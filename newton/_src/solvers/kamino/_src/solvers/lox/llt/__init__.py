# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LOX-local blocked LLT solvers.

These extend Kamino's RCM-reordered blocked LLT solver with a fixed symbolic
structure and compact solve traversal, and add a hybrid solver that selects
the sequential, blocked, or RCM factorization per block size. They live with
LOX until they are upstreamed to Kamino's shared linear algebra.
"""

from .hybrid_llt_solver import HybridLLTBlockedSolver
from .info import DenseBlockInfo
from .llt_blocked_rcm_solver import LLTBlockedRCMSolver

__all__ = ["DenseBlockInfo", "HybridLLTBlockedSolver", "LLTBlockedRCMSolver"]
