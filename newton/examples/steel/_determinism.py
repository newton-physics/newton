# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared setup of the Steel determinism examples."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel

# Bodies whose copies differ by this much [m] or more are drawn fully in DIVERGED_COLOR.
DIVERGENCE_COLOR_MAX = 1.0e-2
# Divergence below this [m] keeps BODY_COLOR; the ramp is logarithmic in between.
DIVERGENCE_COLOR_MIN = 1.0e-6
BODY_COLOR = (0.78, 0.78, 0.8)
DIVERGED_COLOR = (0.93, 0.2, 0.15)


@wp.kernel(enable_backward=False)
def compare_copies(
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    display_body_offset: int,
    display_body_q: wp.array[wp.transform],
    body_divergence: wp.array[wp.float32],
):
    """Show copy A of one mode and record how far each of its bodies is from copy B."""
    body_id = wp.tid()
    display_body_q[display_body_offset + body_id] = body_q_a[body_id]
    delta = wp.transform_get_translation(body_q_a[body_id]) - wp.transform_get_translation(body_q_b[body_id])
    body_divergence[display_body_offset + body_id] = wp.length(delta)


@wp.kernel(enable_backward=False)
def color_by_divergence(
    shape_body: wp.array[wp.int32],
    body_divergence: wp.array[wp.float32],
    default_color: wp.array[wp.vec3],
    body_color: wp.vec3,
    diverged_color: wp.vec3,
    log_min: float,
    log_max: float,
    shape_color: wp.array[wp.vec3],
):
    """Blend each body's shapes from ``body_color`` toward ``diverged_color`` with the log of its divergence."""
    shape_id = wp.tid()
    body_id = shape_body[shape_id]
    if body_id < 0:
        shape_color[shape_id] = default_color[shape_id]
        return
    weight = float(0.0)
    divergence = body_divergence[body_id]
    if divergence > 0.0:
        weight = wp.clamp((wp.log10(divergence) - log_min) / (log_max - log_min), 0.0, 1.0)
    shape_color[shape_id] = wp.lerp(body_color, diverged_color, weight)


class DeterminismExample:
    """Step one scene in two default and two deterministic Steel copies and show how the copies drift apart.

    The default copies are drawn on the left and the deterministic copies on the right; subclasses
    provide the scene and may override :meth:`apply_control`.
    """

    def __init__(
        self,
        viewer,
        scene: newton.ModelBuilder,
        solver_options: dict,
        copy_spacing: tuple[float, float, float],
        camera: dict,
    ):
        self.viewer = viewer
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        # An even substep count leaves each copy's state_0 holding the frame's result after graph replay.
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        if not wp.get_device().is_cuda:
            raise RuntimeError("The Steel determinism examples require a CUDA device.")

        self.copies = {
            mode: [_SimulationCopy(scene, solver_options, deterministic) for _ in range(2)]
            for mode, deterministic in (("default", False), ("deterministic", True))
        }
        self.scene_body_count = scene.body_count
        # Every copy reads the same control so the copies only differ by how they were simulated.
        self.control = self.copies["default"][0].model.control()

        display = newton.ModelBuilder()
        display.replicate(scene, 2)
        display.add_ground_plane()
        self.model = display.finalize()
        self.state_0 = self.model.state()
        self.body_divergence = wp.zeros(self.model.body_count, dtype=wp.float32)
        self.default_shape_color = wp.clone(self.model.shape_color)
        self.body_color = wp.vec3(*BODY_COLOR)
        self.diverged_color = wp.vec3(*DIVERGED_COLOR)
        self.max_divergence = {"default": 0.0, "deterministic": 0.0}

        self.viewer.set_model(self.model)
        self.viewer.set_world_offsets(copy_spacing)
        self.viewer.set_camera(**camera)

        self.capture()

    def apply_control(self, step_index: wp.array[wp.int32]):
        """Write the control of the next substep; the default leaves the zero control."""

    def capture(self):
        """Capture one frame of every copy in a CUDA graph."""
        self.step_index = wp.zeros(1, dtype=wp.int32)
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance every copy by one frame with the same control."""
        for _ in range(self.sim_substeps):
            self.apply_control(self.step_index)
            for copies in self.copies.values():
                for copy in copies:
                    copy.step(self.control, self.sim_dt)
            wp.launch(_advance_step_index, dim=1, outputs=[self.step_index])

    def step(self):
        """Advance the scene by one rendered frame."""
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        for mode_id, (copy_a, copy_b) in enumerate(self.copies.values()):
            wp.launch(
                compare_copies,
                dim=self.scene_body_count,
                inputs=[copy_a.state_0.body_q, copy_b.state_0.body_q, mode_id * self.scene_body_count],
                outputs=[self.state_0.body_q, self.body_divergence],
            )
        divergence = self.body_divergence.numpy().reshape(2, self.scene_body_count).max(axis=1)
        self.max_divergence = dict(zip(self.copies, (float(d) for d in divergence), strict=True))

    def render(self):
        """Draw copy A of each mode, colored by how far its bodies are from copy B."""
        wp.launch(
            color_by_divergence,
            dim=self.model.shape_count,
            inputs=[
                self.model.shape_body,
                self.body_divergence,
                self.default_shape_color,
                self.body_color,
                self.diverged_color,
                float(np.log10(DIVERGENCE_COLOR_MIN)),
                float(np.log10(DIVERGENCE_COLOR_MAX)),
            ],
            outputs=[self.model.shape_color],
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("Default copies, max divergence [mm]", 1.0e3 * self.max_divergence["default"])
        self.viewer.log_scalar(
            "Deterministic copies, max divergence [mm]", 1.0e3 * self.max_divergence["deterministic"]
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify every copy is finite, no copy dropped constraint rows and the deterministic copies are bitwise equal."""
        state_names = ("body_q", "body_qd", "joint_q", "joint_qd")
        for mode, copies in self.copies.items():
            for copy in copies:
                assert not np.any(copy.solver.constraint_overflow.numpy()), f"A {mode} copy overflowed its rows"
                for name in state_names:
                    assert np.all(np.isfinite(getattr(copy.state_0, name).numpy())), (
                        f"A {mode} copy has non-finite {name}"
                    )
        copy_a, copy_b = self.copies["deterministic"]
        for name in state_names:
            a = getattr(copy_a.state_0, name).numpy()
            b = getattr(copy_b.state_0, name).numpy()
            assert a.tobytes() == b.tobytes(), f"Deterministic copies differ in {name}"


class _SimulationCopy:
    """One model, solver and collision pipeline simulating the scene."""

    def __init__(self, scene: newton.ModelBuilder, solver_options: dict, deterministic: bool):
        builder = newton.ModelBuilder()
        builder.replicate(scene, 1)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = SolverSteel(
            self.model,
            deterministic=wp.DeterministicMode.RUN_TO_RUN if deterministic else wp.DeterministicMode.NOT_GUARANTEED,
            **solver_options,
        )
        self.collision_pipeline = newton.CollisionPipeline(self.model, deterministic=deterministic)
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

    def step(self, control: newton.Control, dt: float):
        self.state_0.clear_forces()
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, control, self.contacts, dt)
        self.state_0, self.state_1 = self.state_1, self.state_0


@wp.kernel(enable_backward=False)
def _advance_step_index(step_index: wp.array[wp.int32]):
    step_index[0] = step_index[0] + 1
