# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Kamino: A Newton physics solver for simulating arbitrary mechanical assemblies, (i.e.
constrained rigid multi-body systems) with kinematic loops and under-/overactuation.
"""

from .solver_kamino import SolverKamino

###
# Kamino API
###

__all__ = ["SolverKamino"]
