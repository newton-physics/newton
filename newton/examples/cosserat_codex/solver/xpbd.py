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

"""XPBD-style solver for Cosserat rod simulation.

This module provides the high-level solver interface that advances
batched rod simulations using the XPBD algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from newton.examples.cosserat_codex.rod.config import RodState


class CosseratXPBDSolver:
    """XPBD-style solver wrapper that advances batched Warp rods.
    
    This solver implements the eXtended Position-Based Dynamics (XPBD)
    algorithm for simulating Cosserat elastic rods. It provides a simple
    interface compatible with Newton's solver patterns.
    
    Attributes:
        rod_batch: The RodBatch or RodState being simulated.
        linear_damping: Linear velocity damping factor.
        angular_damping: Angular velocity damping factor.
    """

    def __init__(
        self,
        rod_batch,
        linear_damping: float = 0.0,
        angular_damping: float = 0.0,
    ):
        """Initialize the XPBD solver.
        
        Args:
            rod_batch: The RodBatch or RodState to simulate.
            linear_damping: Linear velocity damping factor [0, 1].
            angular_damping: Angular velocity damping factor [0, 1].
        """
        self.rod_batch = rod_batch
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping

    def notify_model_changed(self, *args, **kwargs) -> None:
        """Notification hook for model changes (no-op for XPBD)."""
        return None

    def update_contacts(self, *args, **kwargs) -> None:
        """Update contact state (no-op for XPBD)."""
        return None

    def step(
        self,
        state_in: "RodState",
        state_out: "RodState | None",
        control: Any,
        contacts: Any,
        dt: float,
    ) -> None:
        """Advance the simulation by one time step.
        
        Args:
            state_in: Input rod state (may be modified).
            state_out: Output rod state (if None, state_in is modified in place).
            control: Control inputs (currently unused).
            contacts: Contact information (currently unused).
            dt: Time step size in seconds.
        """
        state = state_out or state_in
        for rod in state.rods:
            rod.step(dt, self.linear_damping, self.angular_damping)

    def set_damping(self, linear: float, angular: float) -> None:
        """Set damping coefficients.
        
        Args:
            linear: Linear velocity damping factor [0, 1].
            angular: Angular velocity damping factor [0, 1].
        """
        self.linear_damping = linear
        self.angular_damping = angular


__all__ = ["CosseratXPBDSolver"]
