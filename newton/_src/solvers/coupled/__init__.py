# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental coupled-solver framework.

.. experimental::
"""

from .interface import CouplingInterface
from .model_view import ModelView
from .solver_coupled import SolverCoupled
from .solver_coupled_admm import SolverCoupledADMM
from .solver_coupled_proxy import SolverCoupledProxy

__all__ = [
    "CouplingInterface",
    "ModelView",
    "SolverCoupled",
    "SolverCoupledADMM",
    "SolverCoupledProxy",
]
