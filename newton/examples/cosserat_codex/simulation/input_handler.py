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

"""Keyboard input handler for Cosserat rod simulation.

This module provides keyboard input handling for interactive rod manipulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from newton.examples.cosserat_codex.rod.warp_rod import WarpResidentRodState


@dataclass
class InputState:
    """State tracking for input handling.
    
    Attributes:
        insertion: Current insertion depth.
        rotation_angle: Current rotation angle (radians).
        tip_bend_d1: Tip bend angle in d1 direction.
        tip_bend_d2: Tip bend angle in d2 direction.
        paused: Whether simulation is paused.
        selected_rod_index: Index of currently selected rod.
    """

    insertion: float = 0.0
    rotation_angle: float = 0.0
    tip_bend_d1: float = 0.0
    tip_bend_d2: float = 0.0
    paused: bool = False
    selected_rod_index: int = 0


@dataclass
class InputConfig:
    """Configuration for input handling.
    
    Attributes:
        insertion_step: Step size for insertion changes.
        rotation_step: Step size for rotation changes (degrees).
        tip_bend_step: Step size for tip bend changes (degrees).
        max_insertion: Maximum insertion depth.
        min_insertion: Minimum insertion depth.
        max_tip_bend: Maximum tip bend angle (degrees).
    """

    insertion_step: float = 0.01
    rotation_step: float = 5.0
    tip_bend_step: float = 5.0
    max_insertion: float = 1.0
    min_insertion: float = 0.0
    max_tip_bend: float = 45.0


class KeyboardInputHandler:
    """Handler for keyboard input in rod simulations.
    
    Maps keyboard keys to actions for controlling rod simulation:
    - Arrow keys: Move root position
    - W/S: Advance/retract (insertion)
    - A/D: Rotate left/right
    - Q/E: Tip bend in d1 direction
    - Z/X: Tip bend in d2 direction
    - R: Reset
    - Space: Pause/unpause
    - Tab: Select next rod
    """

    def __init__(self, config: InputConfig | None = None):
        """Initialize the input handler.
        
        Args:
            config: Input configuration. Uses defaults if None.
        """
        self.config = config or InputConfig()
        self.state = InputState()
        self._callbacks: dict[str, list[Callable]] = {}
        self._key_handlers: dict[int, Callable] = {}
        self._setup_default_handlers()

    def _setup_default_handlers(self) -> None:
        """Set up default key handlers using pyglet key codes."""
        try:
            from pyglet.window import key

            self._key_codes = {
                "up": key.UP,
                "down": key.DOWN,
                "left": key.LEFT,
                "right": key.RIGHT,
                "w": key.W,
                "s": key.S,
                "a": key.A,
                "d": key.D,
                "q": key.Q,
                "e": key.E,
                "z": key.Z,
                "x": key.X,
                "r": key.R,
                "space": key.SPACE,
                "tab": key.TAB,
                "escape": key.ESCAPE,
            }
        except ImportError:
            # Fallback to ASCII codes if pyglet not available
            self._key_codes = {
                "up": 65362,
                "down": 65364,
                "left": 65361,
                "right": 65363,
                "w": ord("w"),
                "s": ord("s"),
                "a": ord("a"),
                "d": ord("d"),
                "q": ord("q"),
                "e": ord("e"),
                "z": ord("z"),
                "x": ord("x"),
                "r": ord("r"),
                "space": ord(" "),
                "tab": ord("\t"),
                "escape": 27,
            }

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an input event.
        
        Args:
            event: Event name (e.g., "insertion_changed", "reset", "pause").
            callback: Function to call when event occurs.
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _fire_event(self, event: str, *args, **kwargs) -> None:
        """Fire an event and call all registered callbacks."""
        for callback in self._callbacks.get(event, []):
            callback(*args, **kwargs)

    def handle_key_press(self, symbol: int, modifiers: int = 0) -> bool:
        """Handle a key press event.
        
        Args:
            symbol: Key code.
            modifiers: Modifier keys (shift, ctrl, etc.).
        
        Returns:
            True if the key was handled, False otherwise.
        """
        # Check for insertion (W/S keys)
        if symbol == self._key_codes.get("w"):
            self._change_insertion(self.config.insertion_step)
            return True
        elif symbol == self._key_codes.get("s"):
            self._change_insertion(-self.config.insertion_step)
            return True

        # Check for rotation (A/D keys)
        elif symbol == self._key_codes.get("a"):
            self._change_rotation(-self.config.rotation_step)
            return True
        elif symbol == self._key_codes.get("d"):
            self._change_rotation(self.config.rotation_step)
            return True

        # Check for tip bend d1 (Q/E keys)
        elif symbol == self._key_codes.get("q"):
            self._change_tip_bend_d1(-self.config.tip_bend_step)
            return True
        elif symbol == self._key_codes.get("e"):
            self._change_tip_bend_d1(self.config.tip_bend_step)
            return True

        # Check for tip bend d2 (Z/X keys)
        elif symbol == self._key_codes.get("z"):
            self._change_tip_bend_d2(-self.config.tip_bend_step)
            return True
        elif symbol == self._key_codes.get("x"):
            self._change_tip_bend_d2(self.config.tip_bend_step)
            return True

        # Reset
        elif symbol == self._key_codes.get("r"):
            self._reset()
            return True

        # Pause
        elif symbol == self._key_codes.get("space"):
            self._toggle_pause()
            return True

        # Select next rod
        elif symbol == self._key_codes.get("tab"):
            self._fire_event("select_next_rod")
            return True

        return False

    def _change_insertion(self, delta: float) -> None:
        """Change insertion depth."""
        new_insertion = self.state.insertion + delta
        new_insertion = max(self.config.min_insertion, min(self.config.max_insertion, new_insertion))
        if new_insertion != self.state.insertion:
            self.state.insertion = new_insertion
            self._fire_event("insertion_changed", self.state.insertion)

    def _change_rotation(self, delta_degrees: float) -> None:
        """Change rotation angle."""
        self.state.rotation_angle += math.radians(delta_degrees)
        self._fire_event("rotation_changed", self.state.rotation_angle)

    def _change_tip_bend_d1(self, delta_degrees: float) -> None:
        """Change tip bend in d1 direction."""
        new_bend = self.state.tip_bend_d1 + delta_degrees
        new_bend = max(-self.config.max_tip_bend, min(self.config.max_tip_bend, new_bend))
        if new_bend != self.state.tip_bend_d1:
            self.state.tip_bend_d1 = new_bend
            self._fire_event("tip_bend_changed", self.state.tip_bend_d1, self.state.tip_bend_d2)

    def _change_tip_bend_d2(self, delta_degrees: float) -> None:
        """Change tip bend in d2 direction."""
        new_bend = self.state.tip_bend_d2 + delta_degrees
        new_bend = max(-self.config.max_tip_bend, min(self.config.max_tip_bend, new_bend))
        if new_bend != self.state.tip_bend_d2:
            self.state.tip_bend_d2 = new_bend
            self._fire_event("tip_bend_changed", self.state.tip_bend_d1, self.state.tip_bend_d2)

    def _reset(self) -> None:
        """Reset all input state."""
        self.state.insertion = 0.0
        self.state.rotation_angle = 0.0
        self.state.tip_bend_d1 = 0.0
        self.state.tip_bend_d2 = 0.0
        self._fire_event("reset")

    def _toggle_pause(self) -> None:
        """Toggle pause state."""
        self.state.paused = not self.state.paused
        self._fire_event("pause_changed", self.state.paused)


__all__ = ["InputConfig", "InputState", "KeyboardInputHandler"]
