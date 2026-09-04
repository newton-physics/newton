# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-body primal splitting for Kamino.

``solver`` orchestrates the solve; ``problem`` owns its topology and rows.
``adapter`` converts Kamino inputs and outputs, while ``system`` assembles
and solves the primal body operator. ``projection`` supplies shared local
operations to the ``jacobi`` and ``colored_gauss_seidel`` schedules.
``types`` owns splitting state and ``kernels`` updates it and its residuals.
"""

from .solver import LOXSolver

__all__ = ["LOXSolver"]
