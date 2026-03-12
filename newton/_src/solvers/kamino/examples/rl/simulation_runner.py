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

"""Sync / async simulation loop for RL examples.

In **sync** mode the viewer and physics run in lockstep on the main thread
(the current default behavior).

In **async** mode the GPU physics + policy inference run as fast as possible
on a background thread while the main thread handles viewer rendering and
joystick polling at fixed rates.  OpenGL must stay on the thread that created
the context, so rendering always happens on the main thread.
"""

from __future__ import annotations

# Python
import threading
import time


class SimulationRunner:
    """Run an RL example in sync or async mode.

    Args:
        example: An ``Example`` instance (must expose ``step``, ``sim_step``,
            ``update_input``, ``reset``, ``render``, ``joystick``, and
            ``sim_wrapper``).
        mode: ``"sync"`` (default) or ``"async"``.
        render_fps: Target rendering rate in Hz (async mode only).
        joystick_hz: Target joystick polling rate in Hz (async mode only).
    """

    def __init__(
        self,
        example,
        mode: str = "sync",
        render_fps: float = 30.0,
        joystick_hz: float = 30.0,
    ) -> None:
        if mode not in ("sync", "async"):
            raise ValueError(f"mode must be 'sync' or 'async', got {mode!r}")
        self._example = example
        self._mode = mode
        self._render_period = 1.0 / render_fps
        self._joystick_period = 1.0 / joystick_hz

        # Threading primitives (used only in async mode)
        self._lock = threading.Lock()
        self._reset_event = threading.Event()
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the loop until the viewer is closed or KeyboardInterrupt."""
        if self._mode == "sync":
            self._run_sync()
        else:
            self._run_async()

    # ------------------------------------------------------------------
    # Sync mode
    # ------------------------------------------------------------------

    def _run_sync(self) -> None:
        ex = self._example
        while ex.sim_wrapper.is_running():
            ex.step()
            ex.render()

    # ------------------------------------------------------------------
    # Async mode
    # ------------------------------------------------------------------

    def _run_async(self) -> None:
        # In async mode the joystick is polled at joystick_hz, not every
        # sim step, so its dt (used for path integration and filters)
        # must match the actual polling period.
        self._example.joystick.set_dt(self._joystick_period)

        sim_thread = threading.Thread(target=self._sim_thread_fn, daemon=True)
        sim_thread.start()
        try:
            self._main_thread_loop()
        finally:
            self._stop_event.set()
            sim_thread.join(timeout=2.0)
            # Restore dt so sync mode works if run() is called again.
            self._example.joystick.set_dt(self._example.env_dt)

    def _main_thread_loop(self) -> None:
        ex = self._example
        next_render = time.monotonic()
        next_joystick = time.monotonic()

        while ex.sim_wrapper.is_running():
            now = time.monotonic()
            acted = False

            # --- Joystick polling ---
            if now >= next_joystick:
                next_joystick += self._joystick_period
                if next_joystick < now:
                    next_joystick = now + self._joystick_period

                if ex.joystick.check_reset():
                    self._reset_event.set()

                # Single lock acquisition: snapshot root pos, run joystick
                # filter + path integration, and write commands to obs.
                with self._lock:
                    root_pos_2d = ex.sim_wrapper.q_i[:, 0, :2].clone()
                    ex.joystick.update(root_pos_2d=root_pos_2d)
                    ex.update_input()

                acted = True

            # --- Rendering ---
            if now >= next_render:
                next_render += self._render_period
                if next_render < now:
                    next_render = now + self._render_period

                with self._lock:
                    ex.render()

                acted = True

            if not acted:
                sleep_until = min(next_render, next_joystick)
                remaining = sleep_until - time.monotonic()
                if remaining > 0:
                    time.sleep(min(remaining, 0.001))

    def _sim_thread_fn(self) -> None:
        ex = self._example
        sim_period = ex.env_dt
        next_step = time.monotonic()

        while not self._stop_event.is_set():
            if self._reset_event.is_set():
                with self._lock:
                    ex.reset()
                self._reset_event.clear()
                next_step = time.monotonic()

            with self._lock:
                ex.sim_step()

            next_step += sim_period
            now = time.monotonic()
            if next_step > now:
                time.sleep(next_step - now)
            elif next_step < now - sim_period:
                # Fell too far behind, reset the clock
                next_step = now
