# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
