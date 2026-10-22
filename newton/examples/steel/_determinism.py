# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared setup of the Steel determinism examples."""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel

# Each mode draws its two copies at one place, magenta and green and half transparent. The GL
# viewer blends coincident copies to one gray body and shows diverging copies as two ghosts.
COPY_COLORS = ((0.9, 0.2, 0.8), (0.2, 0.8, 0.3))
COPY_OPACITY = 0.5


@wp.kernel(enable_backward=False)
def show_copies(
    body_q_a: wp.array[wp.transform],
    body_q_b: wp.array[wp.transform],
    side_offset: wp.vec3,
    display_body_offset: int,
    divergence_offset: int,
    display_body_q: wp.array[wp.transform],
    body_divergence: wp.array[wp.float32],
):
    """Place both copies of one mode at its side of the view and record how far apart each body is."""
    body_id = wp.tid()
    body_count = body_q_a.shape[0]
    q_a = body_q_a[body_id]
    q_b = body_q_b[body_id]
    p_a = wp.transform_get_translation(q_a)
    p_b = wp.transform_get_translation(q_b)
    display_body_q[display_body_offset + body_id] = wp.transform(p_a + side_offset, wp.transform_get_rotation(q_a))
    display_body_q[display_body_offset + body_count + body_id] = wp.transform(
        p_b + side_offset, wp.transform_get_rotation(q_b)
    )
    body_divergence[divergence_offset + body_id] = wp.length(p_a - p_b)


class DeterminismExample:
    """Step one scene in two default and two deterministic Steel copies and overlay the copies of each mode.

    The default pair is drawn on the left and the deterministic pair on the right; subclasses
    provide the scene and may override :meth:`apply_control`.
    """

    def __init__(
        self,
        viewer,
        scene: newton.ModelBuilder,
        solver_options: dict,
        side_offset: tuple[float, float, float],
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

        # The default pair sits at -side_offset and the deterministic pair at +side_offset.
        self.side_offsets = [wp.vec3(*side_offset) * sign for sign in (-1.0, 1.0)]
        display = newton.ModelBuilder()
        for offset in self.side_offsets:
            for color in COPY_COLORS:
                first_shape = display.shape_count
                display.add_builder(scene, xform=wp.transform(offset, wp.quat_identity()))
                for shape in range(first_shape, display.shape_count):
                    display.shape_color[shape] = color
                    display.shape_opacity[shape] = COPY_OPACITY
        display.add_ground_plane()
        self.model = display.finalize()
        self.state_0 = self.model.state()
        self.body_divergence = wp.zeros(2 * self.scene_body_count, dtype=wp.float32)
        self.max_divergence = {"default": 0.0, "deterministic": 0.0}

        self.viewer.set_model(self.model)
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
                show_copies,
                dim=self.scene_body_count,
                inputs=[
                    copy_a.state_0.body_q,
                    copy_b.state_0.body_q,
                    self.side_offsets[mode_id],
                    2 * mode_id * self.scene_body_count,
                    mode_id * self.scene_body_count,
                ],
                outputs=[self.state_0.body_q, self.body_divergence],
            )
        divergence = self.body_divergence.numpy().reshape(2, self.scene_body_count).max(axis=1)
        self.max_divergence = dict(zip(self.copies, (float(d) for d in divergence), strict=True))

    def render(self):
        """Draw both copies of each mode overlaid, the default pair left and the deterministic pair right."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("Default (left), max divergence [mm]", 1.0e3 * self.max_divergence["default"])
        self.viewer.log_scalar(
            "Deterministic (right), max divergence [mm]", 1.0e3 * self.max_divergence["deterministic"]
        )
        self.viewer.end_frame()

    def gui(self, ui):
        """Show the largest body distance between the two copies of each mode."""
        ui.text("Left: default, right: deterministic")
        ui.text("Copies in magenta and green, gray where they coincide")
        for mode, divergence in self.max_divergence.items():
            ui.text(f"{mode.capitalize()} max distance: {1.0e3 * divergence:.3f} mm")

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
        # The ground plane takes its gap from this builder.
        builder.rigid_gap = scene.rigid_gap
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
