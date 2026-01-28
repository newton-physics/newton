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

"""Example simulation class for Cosserat rod demonstration.

This module provides a simplified Example class that demonstrates the
refactored Cosserat rod simulation architecture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from newton.examples.cosserat_codex.rod.config import RodBatch, RodConfig, RodState
from newton.examples.cosserat_codex.rod.dll_rod import DefKitDirectLibrary
from newton.examples.cosserat_codex.solver.xpbd import CosseratXPBDSolver

from .constraints import apply_concentric_constraint, apply_floor_collision, apply_track_constraint
from .input_handler import InputConfig, InputState, KeyboardInputHandler

if TYPE_CHECKING:
    from newton.examples.cosserat_codex.rod.warp_rod import WarpResidentRodState


@dataclass
class SimulationConfig:
    """Configuration for the simulation.
    
    Attributes:
        dt: Time step size in seconds.
        substeps: Number of simulation substeps per frame.
        linear_damping: Linear velocity damping [0, 1].
        angular_damping: Angular velocity damping [0, 1].
        floor_z: Z coordinate of the floor plane.
        floor_restitution: Coefficient of restitution for floor collisions.
        enable_floor: Whether floor collision is enabled.
        enable_track: Whether track sliding is enabled.
        enable_concentric: Whether concentric constraint is enabled.
    """

    dt: float = 1.0 / 60.0
    substeps: int = 4
    linear_damping: float = 0.01
    angular_damping: float = 0.01
    floor_z: float = 0.0
    floor_restitution: float = 0.0
    enable_floor: bool = True
    enable_track: bool = False
    enable_concentric: bool = False


@dataclass
class TrackConfig:
    """Configuration for track sliding constraint.
    
    Attributes:
        start: 3D start point of the track.
        end: 3D end point of the track.
        insertion: Current insertion depth.
        stiffness: Sliding constraint stiffness.
    """

    start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32))
    end: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    insertion: float = 0.0
    stiffness: float = 1.0


class Example:
    """Simplified example simulation class.
    
    This class demonstrates the refactored Cosserat rod simulation with:
    - Batched GPU-resident rods
    - XPBD constraint solver
    - Optional constraints (floor, track, concentric)
    - Input handling
    
    Attributes:
        config: Simulation configuration.
        rod_batch: Batch of rod configurations.
        rod_state: GPU-resident rod states.
        solver: XPBD solver instance.
        lib: DLL library for native functions.
        device: Warp device for GPU computation.
    """

    def __init__(
        self,
        rod_configs: list[RodConfig],
        sim_config: SimulationConfig | None = None,
        dll_path: str | None = None,
        device: wp.Device | None = None,
    ):
        """Initialize the simulation.
        
        Args:
            rod_configs: List of rod configurations.
            sim_config: Simulation configuration (uses defaults if None).
            dll_path: Path to DefKitAdv DLL.
            device: Warp device for GPU computation.
        """
        self.config = sim_config or SimulationConfig()
        self.device = device or wp.get_device()

        # Initialize DLL library
        self.lib = DefKitDirectLibrary(dll_path, calling_convention="cdecl")

        # Create rod batch
        self.rod_batch = RodBatch(rod_configs)

        # Create GPU-resident rod states
        self.rod_state = self.rod_batch.create_state(
            lib=self.lib,
            device=self.device,
            use_banded=False,
            use_cuda_graph=False,
        )

        # Create solver
        self.solver = CosseratXPBDSolver(
            self.rod_state,
            linear_damping=self.config.linear_damping,
            angular_damping=self.config.angular_damping,
        )

        # Track configuration
        self.track_config = TrackConfig()

        # Input handler
        self.input_handler = KeyboardInputHandler()
        self._setup_input_callbacks()

        # Timing
        self._last_step_time = time.perf_counter()
        self._frame_count = 0

    def _setup_input_callbacks(self) -> None:
        """Set up input callbacks."""
        def on_insertion_changed(insertion: float) -> None:
            self.track_config.insertion = insertion

        def on_reset() -> None:
            self.reset()

        self.input_handler.register_callback("insertion_changed", on_insertion_changed)
        self.input_handler.register_callback("reset", on_reset)

    @property
    def rods(self) -> list["WarpResidentRodState"]:
        """Get the list of rod states."""
        return self.rod_state.rods

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.rod_state.reset()
        self.track_config.insertion = 0.0
        self.input_handler.state = InputState()

    def step(self) -> None:
        """Advance simulation by one frame (possibly multiple substeps)."""
        dt_sub = self.config.dt / self.config.substeps

        for _ in range(self.config.substeps):
            self._step_substep(dt_sub)

        self._frame_count += 1

    def _step_substep(self, dt: float) -> None:
        """Execute a single simulation substep."""
        # Apply external constraints
        self._apply_constraints()

        # Step the solver
        self.solver.step(self.rod_state, self.rod_state, None, None, dt)

        # Apply floor collision
        if self.config.enable_floor:
            for rod in self.rods:
                apply_floor_collision(rod, self.config.floor_z, self.config.floor_restitution)

    def _apply_constraints(self) -> None:
        """Apply external constraints before stepping."""
        # Track constraint
        if self.config.enable_track and len(self.rods) > 0:
            apply_track_constraint(
                self.rods[0],
                self.track_config.start,
                self.track_config.end,
                self.track_config.insertion,
                self.track_config.stiffness,
            )

        # Concentric constraint (guidewire inside catheter)
        if self.config.enable_concentric and len(self.rods) >= 2:
            inner = self.rods[0]
            outer = self.rods[1]
            insertion_diff = self.track_config.insertion  # Simplified
            apply_concentric_constraint(
                inner,
                outer,
                insertion_diff,
                stiffness=1.0,
            )

    def handle_key(self, symbol: int, modifiers: int = 0) -> bool:
        """Handle a keyboard input.
        
        Args:
            symbol: Key code.
            modifiers: Modifier keys.
        
        Returns:
            True if the key was handled.
        """
        return self.input_handler.handle_key_press(symbol, modifiers)

    def get_positions(self, rod_index: int = 0) -> np.ndarray:
        """Get positions for a rod as numpy array.
        
        Args:
            rod_index: Index of the rod.
        
        Returns:
            Positions as (N, 3) numpy array.
        """
        if rod_index >= len(self.rods):
            return np.zeros((0, 3), dtype=np.float32)
        return self.rods[rod_index].positions_numpy()

    def get_orientations(self, rod_index: int = 0) -> np.ndarray:
        """Get orientations for a rod as numpy array.
        
        Args:
            rod_index: Index of the rod.
        
        Returns:
            Orientations as (N, 4) numpy array (quaternions).
        """
        if rod_index >= len(self.rods):
            return np.zeros((0, 4), dtype=np.float32)
        return self.rods[rod_index].orientations_numpy()

    def destroy(self) -> None:
        """Clean up resources."""
        self.rod_state.destroy()


__all__ = ["Example", "SimulationConfig", "TrackConfig"]
