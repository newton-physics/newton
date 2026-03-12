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
Defines the base class for time-integrators.
"""

from __future__ import annotations

from collections.abc import Callable

import warp as wp

from ..core.control import ControlKamino
from ..core.data import DataKamino
from ..core.model import ModelKamino
from ..core.state import StateKamino
from ..geometry.contacts import ContactsKamino
from ..geometry.detector import CollisionDetector
from ..kinematics.limits import LimitsKamino

###
# Module interface
###

__all__ = ["IntegratorBase"]


###
# Module configs
###

wp.set_module_options({"enable_backward": False})


###
# Interfaces
###


class IntegratorBase:
    """
    Provides a base class that defines a common interface for time-integrators.

    A time-integrator is responsible for solving the time integration sub-problem to
    renderthe next state of the system given the current state, control inputs, and
    time-varying inequality constraints induced by joint limits and contacts.
    """

    def __init__(self, model: ModelKamino):
        """
        Initializes the time-integrator with the given :class:`ModelKamino` instance.

        Args:
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
        """
        self._model = model

    ###
    # Operations
    ###

    def integrate(
        self,
        forward: Callable,
        model: ModelKamino,
        data: DataKamino,
        state_in: StateKamino,
        state_out: StateKamino,
        control: ControlKamino,
        limits: LimitsKamino | None = None,
        contacts: ContactsKamino | None = None,
        detector: CollisionDetector | None = None,
    ):
        """
        Solves the time integration sub-problem to compute the next state of the system.

        Args:
            forward (`Callable`):
                An operator that calls the underlying solver for the forward dynamics sub-problem.
            model (`ModelKamino`):
                The model container holding the time-invariant parameters of the system being simulated.
            data (`DataKamino`):
                The data container holding the time-varying parameters of the system being simulated.
            state_in (`StateKamino`):
                The state of the system at the current time-step.
            state_out (`StateKamino`):
                The state of the system at the next time-step.
            control (`ControlKamino`):
                The control inputs applied to the system at the current time-step.
            limits (`LimitsKamino`, optional):
                The joint limits of the system at the current time-step.
                If `None`, no joint limits are considered for the current time-step.
            contacts (`ContactsKamino`, optional):
                The set of active contacts of the system at the current time-step.
                If `None`, no contacts are considered for the current time-step.
            detector (`CollisionDetector`, optional):
                The collision detector to use for generating the set of active contacts at the current time-step.\n
                If `None`, no collision detection is performed for the current time-step,
                and active contacts must be provided via the `contacts` argument.
        """
        raise NotImplementedError("Integrator is an abstract base class")
