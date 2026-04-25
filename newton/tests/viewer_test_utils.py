# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Visualization helper for ``unittest`` tests.

This module provides :class:`ViewerScene` (a description of what to render and
how to advance one frame of simulation) and :class:`ViewerTestMixin` (a mixin
for ``unittest.TestCase`` that owns the viewer loop). Test writers describe a
scene and let the mixin handle the boilerplate: viewer construction with a
``skipTest`` fallback, optional camera placement, ``begin_frame``/``end_frame``,
default state and contact logging, frame pacing, and ``KeyboardInterrupt``.

Example
-------

.. code-block:: python

    class TestMyScene(ViewerTestMixin, unittest.TestCase):
        DEFAULT_CAMERA = (wp.vec3(8.0, 4.0, 2.5), -20.0, -90.0)

        @unittest.skip("Visual debugging - run manually to view simulation")
        def test_view(self):
            model, solver, state_0, state_1, control, contacts = build_scene()

            def step(scene, dt):
                scene.state_0, scene.state_1 = advance(
                    solver, scene.model, scene.state_0, scene.state_1,
                    scene.control, scene.contacts, dt,
                )

            self._run_viewer(
                ViewerScene(
                    model=model,
                    state_0=state_0,
                    state_1=state_1,
                    contacts=contacts,
                    control=control,
                    step_fn=step,
                ),
                label="my-scene",
            )
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import newton

__all__ = ["ViewerScene", "ViewerTestMixin"]


@dataclass
class ViewerScene:
    """Description of a scene to visualize and how to advance it one frame.

    Attributes:
        model: The :class:`newton.Model` driving the visualization.
        state_0: Current simulation state. Logged at the start of each frame.
        state_1: Scratch state used by ``step_fn`` for double-buffering.
        step_fn: Callable ``(scene, dt) -> None`` advancing the simulation by
            ``dt`` and updating ``scene.state_0``/``scene.state_1`` in place.
        contacts: Optional contact buffer; logged each frame when present.
        control: Optional :class:`newton.Control` for the solver.
        pre_step: Optional callable ``(scene, frame_index) -> None`` invoked
            before ``step_fn`` each frame (e.g. apply per-frame forces or IC).
        extra_logs: Additional per-frame log callables ``(viewer, scene) -> None``
            invoked between the default state/contacts logs and ``end_frame``.
            Use for scene-specific overlays (e.g. hydroelastic contact surfaces).
    """

    model: Any
    state_0: Any
    state_1: Any
    step_fn: Callable[["ViewerScene", float], None]
    contacts: Any | None = None
    control: Any | None = None
    pre_step: Callable[["ViewerScene", int], None] | None = None
    extra_logs: list[Callable[[Any, "ViewerScene"], None]] = field(default_factory=list)


class ViewerTestMixin:
    """unittest mixin implementing the viewer loop for visualization tests.

    Subclass this alongside ``unittest.TestCase``. Override class attributes
    (``DEFAULT_CAMERA``, ``DEFAULT_NUM_FRAMES``, etc.) to change defaults, or
    pass ``camera``/``num_frames`` per call to :meth:`_run_viewer` to override
    them on a single test.
    """

    # Camera as ``(pos, pitch, yaw)``. ``None`` leaves the viewer's default camera.
    DEFAULT_CAMERA: tuple | None = None
    DEFAULT_NUM_FRAMES: int = 300
    # Real-time (sim-time) advanced per rendered frame. ``step_fn`` receives this dt.
    DEFAULT_SIM_DT: float = 1.0 / 60.0
    # Wall-clock pause between frames (~60 fps presentation rate).
    DEFAULT_FRAME_SLEEP: float = 0.016

    def _run_viewer(
        self,
        scene: ViewerScene,
        *,
        camera: tuple | None = None,
        num_frames: int | None = None,
        label: str = "",
    ) -> None:
        """Run the viewer loop for ``scene``.

        Args:
            scene: The :class:`ViewerScene` to render and advance.
            camera: Optional ``(pos, pitch, yaw)`` overriding ``DEFAULT_CAMERA``.
            num_frames: Optional override for ``DEFAULT_NUM_FRAMES``.
            label: Short string used in the start-of-run log line.
        """
        try:
            viewer = newton.viewer.ViewerGL()
            viewer.set_model(scene.model)
            cam = camera if camera is not None else self.DEFAULT_CAMERA
            if cam is not None:
                viewer.set_camera(pos=cam[0], pitch=cam[1], yaw=cam[2])
        except Exception as e:
            self.skipTest(f"ViewerGL not available: {e}")
            return

        n_frames = num_frames if num_frames is not None else self.DEFAULT_NUM_FRAMES
        run_label = label or type(self).__name__
        print(f"\nViewer: {run_label} for {n_frames} frames. Close the viewer window or press Ctrl+C to stop.")

        sim_time = 0.0
        try:
            for frame_index in range(n_frames):
                viewer.begin_frame(sim_time)
                viewer.log_state(scene.state_0)
                if scene.contacts is not None:
                    viewer.log_contacts(scene.contacts, scene.state_0)
                for log_fn in scene.extra_logs:
                    log_fn(viewer, scene)
                viewer.end_frame()

                if scene.pre_step is not None:
                    scene.pre_step(scene, frame_index)
                scene.step_fn(scene, self.DEFAULT_SIM_DT)

                sim_time += self.DEFAULT_SIM_DT
                time.sleep(self.DEFAULT_FRAME_SLEEP)
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
