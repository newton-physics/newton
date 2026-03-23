# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
The integrators module of Kamino provides implementations of time-integration methods.

This module provides a front-end defined by:

- :class:`IntegratorEuler`:
    A classical semi-implicit Euler time-stepping
    integrator formulated in velocity-impulse form.

- :class:`IntegratorMoreauJean`:
    A semi-implicit Moreau-Jean time-stepping
    integrator formulated in velocity-impulse
    form for non-smooth dynamical systems.
"""

from .euler import IntegratorEuler
from .moreau import IntegratorMoreauJean

##
# Module interface
##

__all__ = [
    "IntegratorEuler",
    "IntegratorMoreauJean",
]
